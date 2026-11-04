"""Try INT8 recipes and compare each against the PyTorch P(AI) saved by parity_bench.py.

Recipes (all dynamic quantisation, embedding Gather left FP32 unless noted):
  rr        MatMul, per-channel, reduce_range (7-bit weights: no VPMADDUBSW saturation on AVX2-only CPUs)
  ffn       only the feed-forward MatMuls (attention projections stay FP32), reduce_range
  ffn-emb   ffn + embedding Gather quantised
  no-ffn2   every MatMul except the 24 FFN down-projections (4096->1024)
  no-ffn2-emb  no-ffn2 + embedding Gather  <- the recipe export.py ships
  ffn1-only only the FFN up-projections (1024->4096)

Results on the 34-sample set (Ryzen 7 7800X3D, 2026-11-04), |dP| vs PyTorch:
  naive MatMul        930 MB  max 0.50  mean 0.098  flips 3
  naive MatMul+Gather 537 MB  max 0.59  mean 0.100  flips 4
  rr                  930 MB  max 0.55  mean 0.095  flips 1
  ffn                1231 MB  max 0.68  mean 0.112  flips 4
  ffn-emb             838 MB  max 0.65  mean 0.104  flips 3
  no-ffn2            1232 MB  max 0.04  mean 0.008  flips 0
  no-ffn2-emb         838 MB  max 0.07  mean 0.012  flips 0
  ffn1-only          1533 MB  max 0.03  mean 0.005  flips 0
Usage:  python quant_experiments.py
"""
import json
import statistics
import sys
import time

import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from tokenizers import Tokenizer

from common import OUT_DIR
from parity_bench import encode, read_samples, run, session, windows_of


def prep():
    pre = OUT_DIR / "pre.onnx"
    if not pre.exists():
        m = onnx.load(str(OUT_DIR / "tracer-fp32.onnx"))
        del m.graph.value_info[:]
        onnx.save_model(m, str(OUT_DIR / "stripped.onnx"))
        quant_pre_process(str(OUT_DIR / "stripped.onnx"), str(pre), skip_symbolic_shape=True)
        (OUT_DIR / "stripped.onnx").unlink()
    return pre


def matmul_nodes(pre):
    m = onnx.load(str(pre), load_external_data=False)
    init = {i.name: i for i in m.graph.initializer}
    out = []
    for n in m.graph.node:
        if n.op_type != "MatMul":
            continue
        w = next((init[x] for x in n.input if x in init), None)
        if w is None:
            continue  # activation x activation (attention scores) — never quantised
        out.append((n.name, tuple(w.dims)))
    return out


def main():
    pre = prep()
    nodes = matmul_nodes(pre)
    # FFN weights are 1024x4096 / 4096x1024; attention projections are 1024x1024.
    attn = [n for n, d in nodes if 4096 not in d]
    print(f"{len(nodes)} weight MatMuls, {len(attn)} non-FFN (kept FP32 in 'ffn' recipes)")
    ffn_down = [n for n, d in nodes if d[0] == 4096]
    not_ffn_up = [n for n, d in nodes if d[1] != 4096]
    recipes = {
        "rr": dict(op_types_to_quantize=["MatMul"], nodes_to_exclude=[]),
        "ffn": dict(op_types_to_quantize=["MatMul"], nodes_to_exclude=attn),
        "ffn-emb": dict(op_types_to_quantize=["MatMul", "Gather"], nodes_to_exclude=attn),
        "no-ffn2": dict(op_types_to_quantize=["MatMul"], nodes_to_exclude=ffn_down),
        "no-ffn2-emb": dict(op_types_to_quantize=["MatMul", "Gather"], nodes_to_exclude=ffn_down),
        "ffn1-only": dict(op_types_to_quantize=["MatMul"], nodes_to_exclude=not_ffn_up),
    }
    res = json.loads((OUT_DIR / "spike-results.json").read_text())
    ref = np.array(res["probs"]["torch"])
    samples = read_samples()
    texts = [b for _, _, b in samples]
    fast = Tokenizer.from_file(str(OUT_DIR / "tokenizer.json"))
    eval_idx = [i for i, (n, m, _) in enumerate(samples) if not n.startswith("edge/") and m["expected"] != "mixed"]

    long_ai = " ".join(b for n, m, b in samples if m["expected"] == "ai")
    doc = " ".join(long_ai.split()[:500])
    w_ids, w_mask = encode(fast, windows_of(doc))
    d_ids, d_mask = encode(fast, [doc])

    for name, kw in recipes.items():
        path = OUT_DIR / f"exp-{name}.onnx"
        quantize_dynamic(str(pre), str(path), weight_type=QuantType.QInt8, per_channel=True,
                         reduce_range=True, **kw)
        s, load = session(path.name, 16)
        p = np.array([float(run(s, *encode(fast, [t]))[0]) for t in texts])
        d = np.abs(p - ref)
        flips = int(((p >= 0.5) != (ref >= 0.5)).sum())
        acc = sum((p[i] >= 0.5) == (samples[i][1]["expected"] == "ai") for i in eval_idx)
        s4, _ = session(path.name, 4)
        run(s4, w_ids[:1], w_mask[:1])
        tt = []
        for _ in range(5):
            t = time.perf_counter(); run(s4, w_ids, w_mask); run(s4, d_ids, d_mask); tt.append(time.perf_counter() - t)
        print(f"{name:8s} {path.stat().st_size / 1e6:6.0f} MB  max dP {d.max():.3f}  mean dP {d.mean():.3f}  "
              f"flips {flips}  acc {acc}/{len(eval_idx)}  500w@4thr {statistics.median(tt):.2f}s")
        worst = np.argsort(-d)[:3]
        print("          worst: " + ", ".join(f"{samples[i][0]} {ref[i]:.2f}->{p[i]:.2f}" for i in worst))
        res["probs"][f"exp-{name}"] = p.tolist()
    (OUT_DIR / "spike-results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
