# Detector ensemble (desklib + rasbt): IN PROGRESS

*Status 2026-11-04: rule not chosen yet. This file is a handoff; replace it
with the full report (problem, rule, before/after tables, latency, remaining
weaknesses) once the rule is in.*

## The problem

`MaxEnsemble` (max of desklib and rasbt per window) fixes plain / brand-voice
AI text on whole texts, but at Tracer's ~50-word windows it flags human text
(see the WIP commit). The product rule is "Honest": falsely accusing a human
is worse than missing AI.

## Baselines measured so far (real pipeline, `eval` binary)

desklib only (`--model tracer-int8.onnx`):

| Set | Result |
|---|---|
| eval/hard | Gemini text 0% AI; hard AI 9/15 read >= 50%; Enron humans flagged 4/15 (01, 03, 06, 14); Wikipedia humans flagged 9/15 |
| eval/samples | clear human/AI 20/20; mixed mean error 17 pts |

**desklib alone already causes most of the window-level false positives.** So
"Enron flagged at most 1/15" needs a rule that also cuts desklib's own
window false positives, not only rasbt's. A lead: the four falsely flagged
Enron emails all have a whole-text score <= 0.37 (their windows read AI, the
whole text doesn't). So a rule where a window only counts as AI inside a
paragraph / text that also scores AI is worth testing, alongside the ideas in
the brief (rasbt counts only at a high window threshold; rasbt scored on
paragraph groups and used to lift desklib's windows only where it's
confident).

Naive max ensemble: not re-measured here (rasbt unavailable, below). The WIP
commit reports Enron 4/15 and Wikipedia 12/15 flagged.

## Tooling for choosing the rule

1. Dump every span the pipeline scores, once per model and eval set:

   ```
   cargo run --release -p tracer-engine --bin eval -- --samples eval/hard    --model tracer-int8.onnx --dump desk-hard.tsv
   cargo run --release -p tracer-engine --bin eval -- --samples eval/hard    --model rasbt-bf16w.onnx --dump rasbt-hard.tsv
   cargo run --release -p tracer-engine --bin eval -- --samples eval/samples --model tracer-int8.onnx --dump desk-samples.tsv
   cargo run --release -p tracer-engine --bin eval -- --samples eval/samples --model rasbt-bf16w.onnx --dump rasbt-samples.tsv
   ```

   Rows: `sample expected expected_share kind start end words p`, where kind is
   `win` (analyze()'s windows), `group` (paragraph groups, chunked to
   `doc_chunk_words`), `doc` (whole-text chunks) or `sent` (sentence word
   counts).

2. `python -I tools/model-export/sweep_rules.py desk-hard.tsv rasbt-hard.tsv [-v]`
   rebuilds the sentence aggregation offline and prints the summary per rule
   (Gemini AI%, hard AI hits, Enron / Wikipedia false positives, mixed error).
   On desklib alone it reproduces the eval binary's AI% exactly. Candidate
   rules go in `make_rules()`; the chosen one then goes into the engine with
   its thresholds in `EngineConfig`, and the final numbers come from the
   `eval` binary, not the script.

## Why it stopped in the cloud session

The cloud environment's network policy blocked `huggingface.co` (rasbt
download), `download.pytorch.org` (CPU torch; PyPI torch works) and
`cdn.pyke.io` (ort's prebuilt runtime; worked around by linking the
`libonnxruntime.so` from the PyPI onnxruntime 1.31.1 wheel via
`ORT_LIB_PATH` + `ORT_PREFER_DYNAMIC_LINK=1`). None of this applies on a
normal local machine.

## Still to do

- Export rasbt (`export_rasbt.py`, `rasbt_bf16_storage.py`; parity max |dP|
  0.0000 for FP32 and bf16w), dump both models, sweep rules.
- Implement the chosen rule (keep desklib-only operation possible), unit
  tests with mock detectors, `cargo test --release -p tracer-engine --lib --
  --ignored` with the models present.
- Targets: Gemini >= 50% AI; >= 12/15 hard AI; Enron flagged <= 1/15;
  Wikipedia flagged no worse than desklib alone (9/15); eval/samples 20/20
  and mixed error <= ~20 pts. If not all reachable, report the trade-off and
  pick the fewest human false positives.
- 500-word latency (`eval -- --bench --threads 4`), desklib-only vs new rule.
- Replace this file with the full report.
