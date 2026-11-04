# Model export

Turns `desklib/ai-text-detector-v1.01` (pinned commit `5fdea974`) into the ONNX
files Tracer downloads on first run.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python download.py        # ~1.7 GB into work/hf/
.\.venv\Scripts\python export.py          # work/onnx/tracer-fp32.onnx, tracer-int8.onnx, tokenizer.json
.\.venv\Scripts\python parity_bench.py    # tokenizer + model parity, doc-level eval, latency
```

`work/` is gitignored. `quant_experiments.py` reproduces the INT8 recipe
search (see its docstring for the results table).

## Graph contract (what the Rust side relies on)

- Inputs: `input_ids` int64 `[batch, seq]`, `attention_mask` int64 `[batch, seq]`; seq ≤ 512.
- Output: `logits` float32 `[batch, 1]`. **P(AI) = sigmoid(logit).**
- Mean pooling and the classifier head are inside the graph.
- Tokenise with `tokenizer.json` via the `tokenizers` crate (verified identical
  ids to `transformers.AutoTokenizer` on every eval sample). Pad id 0.

## Artefacts (2026-11-04 build)

| File | Size | SHA-256 |
|---|---|---|
| `tracer-int8.onnx` | 838.4 MB | `de1368a264bd8cfe9f6b72291da3db56191f645e4568f8f037cea0790d380b0b` |
| `tracer-fp32.onnx` | 1741.0 MB | `387d96146e998fba76864e2080dde34a5a0fcc42ab4dadc780c3d997b541a5a4` |
| `tokenizer.json` | 8.7 MB | `5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964` |

The INT8 hash depends on the onnxruntime quantiser version, so re-pin it in the
app whenever the model is re-exported.
