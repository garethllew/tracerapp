"""Export rasbt/ai-text-detector-modernbert (ModernBERT-base, Apache-2.0) to
ONNX with the SAME graph contract as the desklib model, so Tracer's Rust
OnnxDetector runs it unchanged:

  inputs  input_ids, attention_mask  int64 [batch, seq]
  output  logits                     float [batch, 1]   P(AI) = sigmoid(logit)

The 2-class head is folded into one logit: (AI - human) / T, where T is the
model's calibration temperature (detector-config.json). sigmoid of that is
exactly the temperature-scaled softmax P(AI) the model card prescribes.

Second detector in Tracer's max-ensemble: desklib catches classic chatbot
style, this one catches plain / brand-voice AI text (see eval/hard,
compare_models.py).

Shipped artefact: rasbt-bf16w.onnx (302 MB), made by rasbt_bf16_storage.py:
weights stored as bfloat16 (the model was trained in bf16, so this is
lossless), computed in FP32. Bit-identical to FP32 on all 287 eval texts.

Quantisation does NOT work for this model (2026-11-04, 287 texts = eval
samples + paragraphs, vs FP32):
  dynamic INT8, all MatMul+Gather       151 MB  mean |dP| 0.18   56 flips
  dynamic INT8, MLP down-proj in FP32   209 MB  mean |dP| 0.040  12 flips
  dynamic INT8, attention only          558 MB  mean |dP| 0.011   4 flips
  dynamic INT8, embeddings only         483 MB  mean |dP| 0.025  11 flips
  weight-only 8-bit (MatMulNBits b32)   284 MB  mean |dP| 0.013   9 flips, 8 of 9 towards AI
  weight-only 4-bit                     228 MB  mean |dP| 0.42  121 flips
The flips were borderline paragraphs, but biased towards AI: more false
accusations, so bf16 storage it is.

Usage:  python -I export_rasbt.py && python -I rasbt_bf16_storage.py
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_models import paragraphs, read_samples  # noqa: E402

REPO = "rasbt/ai-text-detector-modernbert"
SRC = HERE / "work" / "candidates" / "rasbt"
OUT = HERE / "work" / "onnx"
FP32 = OUT / "rasbt-fp32.onnx"
INT8 = OUT / "rasbt-int8.onnx"
TOK = OUT / "rasbt-tokenizer.json"
MAX_LEN = 512


class OneLogit(torch.nn.Module):
    def __init__(self, model, temperature):
        super().__init__()
        self.model = model
        self.t = temperature

    def forward(self, input_ids, attention_mask):
        l = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return ((l[:, 1] - l[:, 0]) / self.t).unsqueeze(1)


def load_torch():
    cfg = json.loads((SRC / "detector-config.json").read_text())
    assert cfg["label_mapping"] == {"human": 0, "ai": 1}, cfg["label_mapping"]
    m = AutoModelForSequenceClassification.from_pretrained(SRC, torch_dtype=torch.float32, attn_implementation="eager")
    return OneLogit(m.eval(), float(cfg["temperature"])).eval(), cfg["temperature"]


def export():
    OUT.mkdir(parents=True, exist_ok=True)
    model, t = load_torch()
    print(f"temperature {t}")
    ids = torch.full((1, 16), 100, dtype=torch.long)
    mask = torch.ones(1, 16, dtype=torch.long)
    batch = torch.export.Dim("batch", min=1, max=64)
    seq = torch.export.Dim("seq", min=2, max=MAX_LEN)
    t0 = time.time()
    with torch.no_grad():
        prog = torch.onnx.export(model, (ids, mask), input_names=["input_ids", "attention_mask"], output_names=["logits"],
                                 dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
                                 opset_version=18, dynamo=True, optimize=True)
    raw = OUT / "rasbt-raw.onnx"
    prog.save(str(raw), external_data=False)
    m = onnx.load(str(raw), load_external_data=True)
    onnx.save_model(m, str(FP32), save_as_external_data=False)
    for p in OUT.glob("rasbt-raw.onnx*"):
        p.unlink()
    onnx.checker.check_model(str(FP32))
    print(f"exported in {time.time() - t0:.0f}s, {FP32.stat().st_size / 1e6:.0f} MB")


def quantise(name, ops, exclude_pred):
    stripped, pre = OUT / "rasbt-stripped.onnx", OUT / "rasbt-pre.onnx"
    m = onnx.load(str(FP32))
    del m.graph.value_info[:]
    onnx.save_model(m, str(stripped))
    quant_pre_process(str(stripped), str(pre), skip_symbolic_shape=True)
    stripped.unlink()
    pm = onnx.load(str(pre), load_external_data=False)
    init = {i.name: tuple(i.dims) for i in pm.graph.initializer}
    exclude = [n.name for n in pm.graph.node if n.op_type == "MatMul"
               and any(x in init and exclude_pred(init[x]) for x in n.input)]
    out = OUT / name
    quantize_dynamic(str(pre), str(out), weight_type=QuantType.QInt8, per_channel=True, reduce_range=True,
                     op_types_to_quantize=ops, nodes_to_exclude=exclude)
    pre.unlink()
    return out, len(exclude)


def session(path, threads=8):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-export", action="store_true")
    a = ap.parse_args()
    if not a.skip_export:
        export()
    import shutil
    shutil.copy(SRC / "tokenizer.json", TOK)

    fast = Tokenizer.from_file(str(TOK))
    fast.enable_truncation(MAX_LEN)
    ref = AutoTokenizer.from_pretrained(SRC)
    texts = []
    for s in read_samples():
        texts.append(s["body"])
        texts += paragraphs(s["body"])
    mism = sum(fast.encode(x).ids != ref(x, truncation=True, max_length=MAX_LEN)["input_ids"] for x in texts)
    print(f"tokenizer.json vs AutoTokenizer: {len(texts) - mism}/{len(texts)} identical")

    def onnx_probs(sess):
        out = []
        for x in texts:
            e = fast.encode(x)
            ids = np.array([e.ids], dtype=np.int64)
            out.append(float(1 / (1 + np.exp(-sess.run(["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0, 0]))))
        return np.array(out)

    model, _ = load_torch()
    with torch.no_grad():
        ref_p = []
        for x in texts:
            e = fast.encode(x)
            ids = torch.tensor([e.ids])
            ref_p.append(float(torch.sigmoid(model(ids, torch.ones_like(ids)))[0, 0]))
    ref_p = np.array(ref_p)
    del model

    variants = {"fp32": FP32}
    bf16w = OUT / "rasbt-bf16w.onnx"
    if bf16w.exists():
        variants["bf16w"] = bf16w

    print(f"\n{len(texts)} texts (whole samples + paragraphs), |dP| vs PyTorch:")
    for k, path in variants.items():
        p = onnx_probs(session(path))
        d = np.abs(p - ref_p)
        flips = int(((p >= .5) != (ref_p >= .5)).sum())
        s = session(path, 4)
        x = " ".join(texts[:6])
        e = fast.encode(x); ids = np.array([e.ids], dtype=np.int64)
        s.run(["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})
        tt = []
        for _ in range(5):
            t0 = time.perf_counter(); s.run(["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids)}); tt.append(time.perf_counter() - t0)
        print(f"  {k:12s} {path.stat().st_size / 1e6:6.0f} MB  max {d.max():.4f}  mean {d.mean():.4f}  flips {flips}  "
              f"{ids.shape[1]} tok @4thr {statistics.median(tt) * 1000:.0f} ms")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
