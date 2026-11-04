# Phase 0 — Model spike report

*2026-11-04 · desklib/ai-text-detector-v1.01 @ `5fdea974` · measured on a Ryzen 7 7800X3D (8C/16T), 31 GB RAM, onnxruntime 1.31.1 CPU*

## Verdict

**desklib passes on licence, export and accuracy. Ship the tuned INT8 model (838 MB).**
Speed is borderline against the 500-words-in-2-s target and needs the Phase 1
pipeline to do less redundant work. RAM is the real cost: about 1.5 GB while the
model is loaded.

## 1. Licence and architecture

| | |
|---|---|
| Model licence | **MIT** (model card metadata) |
| Base model | `microsoft/deberta-v3-large`, **MIT** |
| Training data | `liamdugan/raid`, **MIT** |
| Architecture | DeBERTa-v3-**large** (24 layers, 1024 hidden, ~435 M params), *not* base. That's why it's big. |
| Head | Attention-masked mean pooling over the last hidden state → `Linear(1024, 1)` → sigmoid. One logit, not two-class softmax. |
| Context | Card example uses `max_len=768`. We cap at 512 as the brief asks, and the eval windows are far shorter anyway. |
| Language | English only (card says so). A French sample scored 0.005, so non-English text is not reliably flagged and the brief's "tuned for English" notice is needed. |

Free to use, modify and redistribute. We must ship the MIT notices (desklib + Microsoft) in `THIRD-PARTY-NOTICES.md`, as Vixer does.

## 2. Export

- `torch.onnx.export` (dynamo) at opset 18. Pooling and the head are **inside the graph**, so Rust only tokenises, runs the session and applies sigmoid.
- **FP32 ONNX = PyTorch exactly** (max |ΔP| 0.0000 over 34 samples).
- **`tokenizer.json` gives identical token ids** to the reference `AutoTokenizer` on 34/34 samples, so the Rust `tokenizers` crate can use it as-is.

## 3. Quantisation: the naive recipe isn't safe

Standard dynamic INT8 quantisation of every MatMul drifted badly. It also drifted **in the
wrong direction**, pushing human texts towards "AI":

| Recipe | Size | max \|ΔP\| | mean \|ΔP\| | Label flips |
|---|---|---|---|---|
| Naive INT8 (all MatMuls) | 930 MB | 0.50 | 0.098 | 3 |
| Naive INT8 + embeddings | 537 MB | 0.59 | 0.100 | 4 |
| Keep attention FP32 | 838–1231 MB | 0.65–0.68 | 0.10–0.11 | 3–4 |
| **Keep the 24 FFN down-projections FP32 (+ INT8 embeddings)** | **838 MB** | **0.07** | **0.012** | **0** |
| Same, embeddings FP32 | 1232 MB | 0.04 | 0.008 | 0 |

Examples from the naive recipe: `human-linkedin-01` went 0.19 → 0.74 and `human-slack-05` 0.03 → 0.54.

Cause: the FFN down-projection inputs are post-GELU activations with large
outliers, and per-tensor activation quantisation crushes them. Excluding just those 24 MatMuls
restores parity **at the same speed**. The shipped recipe also uses
`reduce_range`, which avoids an INT8 saturation bug on older AVX2-only
laptop CPUs. I couldn't test that path on this Zen 4 machine. The worst remaining
drift (0.07) is on emoji-only and sub-15-word texts, which the length gate blocks anyway.

## 4. Size, load time, latency, RAM

Workload: 500 words of AI text → 16 overlapping 60-word windows (batched, 81
tokens) + one 512-token whole-document pass. Run-to-run noise on this machine
is ±30%, so these are typical figures:

| | FP32 | **INT8 (tuned)** |
|---|---|---|
| Download | 1741 MB | **838 MB** (+ 8.7 MB tokenizer) |
| Load (cold) | 2.0–2.1 s | **~1.0 s** |
| 500 words, 16 threads | 2.3–2.6 s | **1.1–1.8 s** |
| 500 words, 8 threads | 3.1 s | **1.9 s** |
| 500 words, 4 threads (≈ laptop) | 3.7–6.6 s | **2.2–2.3 s** |
| RAM once loaded | 2.7 GB | **1.5 GB** |

**Speed:** INT8 sits just over the 2 s target at 4 threads with my naive
windowing, which processes every word about 3×: twice in overlapping windows, once in
the doc pass. Phase 1 can cut this by building windows from sentences with a single
sentence of overlap, sorting windows by length to cut padding, and skipping the
separate doc pass when the text fits in about one window. I expect ~1.2–1.5 s on a laptop, and will measure it.

## 5. Eval (document level, whole text scored at once)

- Clear human vs clear AI, sorted at 0.5: **21/22** for PyTorch, FP32 and tuned INT8 alike. The one miss is `human-slack-01` (7 words → 0.84 AI), which is exactly what the "too short" gate is for.
- All 12 AI samples score ≥ 0.94, except an 11-word one (0.60).
- **Every mixed sample scores 0.93–1.00 at document level.** A pasted AI paragraph dominates the whole-text score, so a single doc-level number can't express "30% AI". That confirms the brief's design: the split has to come from windows.
- **Localisation works.** Scored paragraph by paragraph, every pasted AI paragraph scores 1.00, and most human paragraphs score 0.01–0.39. Short human fragments (12–35 words) sometimes land at 0.5–0.65, which matches the brief's "faint tint" band.
- Emoji-only (0.73) and URL-only (0.86) inputs read as AI if scored, so they must be gated or stripped before scoring, as the brief says.

**Caveat:** all samples are synthetic. The "human" ones were written by an AI imitating casual writing, so treat these numbers as a smoke test, not accuracy. Swapping in 10–20 of your real pre-2022 Slack messages and emails would make the eval much more meaningful.

## 6. Recommendation and decisions for you

1. **Model format:** I recommend **INT8 (tuned recipe)**. It's half the download, roughly 2× faster and uses 1.2 GB less RAM, with no label changes on the sample set.
2. **Where the ONNX lives (needs your call).** desklib publishes only PyTorch weights, so "download from Hugging Face" means hosting *our* export. Options:
   - **(Recommended)** a Hugging Face model repo under your account, e.g. `garethllew/tracer-detector-onnx`, with MIT notices and a pinned revision. Free, no bandwidth caps, resumable downloads.
   - A GitHub Release asset on `tracerapp`, under the 2 GB per-file limit. Simpler, but it ties model updates to app releases.
3. **RAM (proposal, Phase 3).** Keep the model warm while the window is open and for ~10 minutes after the last check, then unload. Reloading takes ~1 s, so the hotkey still feels instant most of the time, and the app doesn't hold 1.5 GB in the tray all day. The alternative is to always keep it warm, as the brief says. Your choice.
4. No reason to look at an alternative model. A smaller DeBERTa/RoBERTa-base detector (~150–250 MB) is the fallback if 838 MB or 1.5 GB RAM proves unacceptable.

## 7. Plan for the next phases (who builds what)

| Phase | Builder | Why |
|---|---|---|
| 1 Engine (Rust crate, scoring pipeline, eval CLI, tests) | **Opus** | Correctness-critical scoring, aggregation and confidence logic, plus ONNX/`ort` integration |
| 2 App shell + UI | **Opus** for the design system, result bar and highlight rendering; **Sonnet** for scaffolding the Tauri app from Vixer | The visual system is the product |
| 3 Hotkey, tray, settings, model download | **Sonnet** session | Mostly porting Vixer's existing tray, hotkey and download-with-SHA code |
| 4 Packaging | **Sonnet** for the workflow, signing and winget (mirroring Vixer); **Opus** for the icon options | Pipeline is a copy job; the icon is design work |

## Files

- `tools/model-export/`: `download.py`, `export.py`, `parity_bench.py`, `quant_experiments.py`, `common.py`, README with the graph contract and SHA-256s.
- `eval/samples/`: 30 samples + 4 edge cases + README.
