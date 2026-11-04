"""Export desklib/ai-text-detector-v1.01 to ONNX (FP32) and make INT8 variants.

Outputs (in work/onnx/):
  tracer-fp32.onnx          full-precision graph: (input_ids, attention_mask) -> logit [batch, 1]
  tracer-int8.onnx          dynamic INT8 (MatMul + embedding Gather), FFN down-projections kept FP32
  tokenizer.json            copied from the checkpoint (used as-is by the Rust `tokenizers` crate)

The mean pooling and classifier head are inside the graph, so the Rust side
only tokenises, runs the session and applies sigmoid.

Usage:  python export.py [--legacy]
"""
import argparse
import shutil
import time

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from common import HF_DIR, OUT_DIR, load_torch_model

OPSET = 18


def export_fp32(legacy: bool):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_torch_model()
    dummy_ids = torch.ones(2, 16, dtype=torch.long)
    dummy_mask = torch.ones(2, 16, dtype=torch.long)
    raw = OUT_DIR / "raw-fp32.onnx"
    t = time.time()
    with torch.no_grad():
        if legacy:
            torch.onnx.export(
                model, (dummy_ids, dummy_mask), str(raw),
                input_names=["input_ids", "attention_mask"], output_names=["logits"],
                dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                              "attention_mask": {0: "batch", 1: "seq"},
                              "logits": {0: "batch"}},
                opset_version=OPSET, dynamo=False,
            )
        else:
            batch = torch.export.Dim("batch", min=1, max=64)
            seq = torch.export.Dim("seq", min=2, max=512)
            prog = torch.onnx.export(
                model, (dummy_ids, dummy_mask),
                input_names=["input_ids", "attention_mask"], output_names=["logits"],
                dynamic_shapes={"input_ids": {0: batch, 1: seq},
                                "attention_mask": {0: batch, 1: seq}},
                opset_version=OPSET, dynamo=True, optimize=True,
            )
            prog.save(str(raw), external_data=False)
    print(f"exported in {time.time() - t:.1f}s")

    # Re-save as a single self-contained file (< 2 GB, so no external data).
    m = onnx.load(str(raw), load_external_data=True)
    final = OUT_DIR / "tracer-fp32.onnx"
    onnx.save_model(m, str(final), save_as_external_data=False)
    for p in OUT_DIR.glob("raw-fp32.onnx*"):
        p.unlink()
    onnx.checker.check_model(str(final))
    return final


def quantise(fp32_path):
    # The dynamo optimiser leaves value_info annotations that ONNX shape
    # inference later contradicts ("(1024) vs (1)"); drop them and let the
    # quantiser re-infer from scratch.
    stripped = OUT_DIR / "stripped.onnx"
    m = onnx.load(str(fp32_path))
    del m.graph.value_info[:]
    onnx.save_model(m, str(stripped))
    pre = OUT_DIR / "pre.onnx"
    quant_pre_process(str(stripped), str(pre), skip_symbolic_shape=True)
    stripped.unlink()
    # Naive dynamic INT8 of every MatMul drifts badly on DeBERTa-v3 (mean
    # |dP| 0.10, human texts pushed towards AI): the FFN down-projections
    # (4096 -> 1024) see post-GELU activations with large outliers that
    # per-tensor activation quantisation crushes. Keeping just those 24
    # MatMuls in FP32 restores parity (mean |dP| ~0.01, no label flips) at the
    # same speed. reduce_range = 7-bit weights, which avoids VPMADDUBSW
    # saturation on AVX2-only (non-VNNI) laptop CPUs. See quant_experiments.py.
    m = onnx.load(str(pre), load_external_data=False)
    init = {i.name: tuple(i.dims) for i in m.graph.initializer}
    ffn_down = [n.name for n in m.graph.node if n.op_type == "MatMul"
                and any(init.get(x, (0,))[0] == 4096 for x in n.input)]
    assert len(ffn_down) == 24, ffn_down
    t = time.time()
    quantize_dynamic(str(pre), str(OUT_DIR / "tracer-int8.onnx"), weight_type=QuantType.QInt8,
                     per_channel=True, reduce_range=True,
                     op_types_to_quantize=["MatMul", "Gather"], nodes_to_exclude=ffn_down)
    print(f"tracer-int8.onnx: quantised in {time.time() - t:.1f}s")
    pre.unlink()


def copy_tokenizer():
    shutil.copy(HF_DIR / "tokenizer.json", OUT_DIR / "tokenizer.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", action="store_true", help="use the TorchScript exporter")
    ap.add_argument("--skip-export", action="store_true")
    a = ap.parse_args()
    fp32 = OUT_DIR / "tracer-fp32.onnx" if a.skip_export else export_fp32(a.legacy)
    quantise(fp32)
    copy_tokenizer()
    for p in sorted(OUT_DIR.iterdir()):
        print(f"{p.name:28s} {p.stat().st_size / 1e6:9.1f} MB")
