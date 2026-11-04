# Detector ensemble (desklib + rasbt): context gate + style lift

*2026-11-04. Branch `detector-ensemble-wip`. All numbers below come from the
real pipeline (`eval` binary) unless marked "sweep".*

## The problem

desklib (DeBERTa-v3-large) is good at classic chatbot style. It scored a
Gemini text in a plain house brand voice as 6% AI (Pangram: 100%), and it
misses most of `eval/hard`'s plain AI texts. rasbt's ModernBERT catches that
style. The WIP fix was a per-window **max** of the two. It caught all the
plain AI, but at Tracer's ~50-word windows it flags human text: Wikipedia
leads 12/15, Enron emails 4/15.

The product rule is "Honest": falsely accusing a human is worse than missing
AI. A closer look showed that **desklib alone already causes most of the
window-level false positives** (Enron 4/15, Wikipedia 9/15). So the fix had
to cut desklib's own window false positives too, not only rasbt's.

## What the span scores showed

Both models scored every window, paragraph group and doc chunk on both eval
sets (`eval --dump`, then `tools/model-export/sweep_rules.py`):

- **desklib reads all 15 Enron emails as human as whole texts** (doc score
  0.00–0.44), while its windows call 4 of them mostly AI. Its long-passage
  read is more reliable than its 50-word read.
- **rasbt's window scores are noise on human prose.** Its windows read 7/7 AI
  on the Netflix and Twitch leads, while its whole-text score there is 0.00.
- **rasbt reads layout.** Tracer used to join a span's sentences with single
  spaces, dropping line breaks and blank lines. desklib doesn't care (every
  span scored identically either way), but rasbt does. On a pasted chatbot
  answer (`ai-claude-next-steps.txt`, the real-world miss that showed this),
  rasbt gave **0.003 with its lines joined, 0.30 with only blank lines kept,
  and 0.97 as pasted**. Spans now keep the line breaks between sentences
  (`text::span_text`).
- **With layout kept, rasbt's whole-text score splits cleanly into two
  groups.** AI reads 0.90–1.00 or ≤ 0.56; humans read 0.95–1.00 (Airbnb,
  Counter-Strike, Payroll) or ≤ 0.20. Enron reads 0.00 throughout. Every
  eval/samples text reads ≤ 0.01: rasbt only fires on the plain style.
- **rasbt's paragraph scores are useful when it's very sure**: ≥ 0.95 on all
  Gemini paragraphs and on most announcement / explainer paragraphs desklib
  misses.

## The rule

A sentence's score starts as the mean of its desklib windows, as before.
Then:

1. **Context gate.** If desklib's score for the sentence's doc chunk (the
   whole text, or the ≤ 300-word passage around it) is below **0.5**, the
   sentence's score is capped at that doc-chunk score.
2. **Style lift.** The sentence is raised to rasbt's score for its
   **paragraph** if that is ≥ **0.95**, or to rasbt's score for its **doc
   chunk** if that is ≥ **0.8**. This applies whether or not the gate capped
   it.

In words: **a sentence reads AI when desklib says so and desklib's read of
the surrounding passage agrees, or when rasbt is sure about its paragraph or
the whole passage.**

**Confidence:** a sentence counts as undecided when either its window score
or its final score is in the ambiguous band (0.15–0.85), or when the gate or
the lift flipped its verdict. Without this, a capped 0.002 looked decisive:
the pasted answer and two missed AI texts reported "High confidence human".
Now every call where the two models disagree is reported as Low or Medium
confidence: Gemini, the pasted answer, and five of the six Wikipedia false
positives are Low, and Dropbox is Medium.

Thresholds in `EngineConfig`: `context_gate` 0.5, `style_lift_threshold`
0.95, `style_doc_lift_threshold` 0.8. Setting `context_gate = 0` turns the
gate off. desklib-only operation is `analyze()` (gate, no lift). The
ensemble is `analyze_with(text, desklib, Some(rasbt), cfg)` /
`models::Ensemble::analyze`.

### Why this rule (sweep, hard set)

First sweep, rasbt input still flattened. These rows rest on paragraph
scores, which barely changed once layout was kept:

| Rule (sweep name) | Gemini | Hard AI ≥ 50% (of 15) | Enron flagged | Wikipedia flagged |
|---|---|---|---|---|
| desklib only | 0% | 9 | 4/15 | 9/15 |
| naive max (`max`) | 100% | 15 | 4/15 | 12/15 |
| rasbt counts only at window ≥ 0.9 (`rasT0.9`) | 100% | 14 | 4/15 | 11/15 |
| rasbt calibrated, logit − 2 (`rasK2.0`) | 100% | 14 | 4/15 | 11/15 |
| rasbt windows only where rasbt's paragraph ≥ 0.9 (`liftG0.9`) | 100% | 13 | 4/15 | 9/15 |
| max, gated by the paragraph (`max+cG@0.5`) | 100% | 13 | 4/15 | 7/15 |
| desklib, gated by the doc chunk (`desk+gD@0.5`) | 0% | 7 | 0/15 | 4/15 |
| gate + paragraph lift ≥ 0.95 (`either0.95/0.5`) | 100% | 12 | 0/15 | 4/15 |
| gate + paragraph lift ≥ 0.9 | 100% | 12 | 0/15 | 5/15 |
| gate + paragraph lift ≥ 0.97 | 100% | 11 | 0/15 | 4/15 |

- Gating on the **paragraph** never fixes Enron: those emails' short
  paragraphs read AI too. Only the longer **doc-chunk** read separates them.
- Every way of letting rasbt's **windows** count keeps Wikipedia at
  9–12/15. Only its paragraph and whole-text scores are safe to use.

Second sweep, layout kept, with the pasted answer added (16 hard AI texts):

| Rule | Hard AI ≥ 50% (of 16) | Enron flagged | Wikipedia flagged |
|---|---|---|---|
| gate + paragraph lift ≥ 0.95 (rule before this fix) | 12 (pasted answer: 15%) | 0/15 | 4/15 |
| + whole-text lift ≥ 0.99 | 12 | 0/15 | 6/15 |
| + whole-text lift ≥ 0.95–0.98 | 15 | 0/15 | 6/15 |
| **+ whole-text lift ≥ 0.5–0.9 (chosen: 0.8)** | **16** | **0/15** | **6/15** |

The whole-text lift costs two Wikipedia false positives (Airbnb and
Counter-Strike, already 43–46% flagged before) and catches four more AI
texts, including the pasted chatbot answer. You chose that trade
(2026-11-04). Any threshold from 0.21 to 0.89 gives the same result on these
sets; 0.8 leaves margin on both sides.

## Results (eval binary)

### eval/hard (16 AI, 30 human)

| | Gemini | Pasted chatbot answer | Hard AI ≥ 50% | Enron flagged | Wikipedia flagged |
|---|---|---|---|---|---|
| desklib only (before) | 0% | 0% | 9/16 | 4/15 (01, 03, 06, 14) | 9/15 |
| naive max (WIP) | 100% | 29% | 15/16 | 4/15 (01, 03, 06, 14) | 12/15 |
| desklib + gate (no rasbt) | 0% | 0% | 7/16 | 0/15 | 4/15 |
| **new rule** | **100%** | **100%** | **16/16** | **0/15** | **6/15** |
| target | ≥ 50% | | ≥ 12/15 | ≤ 1/15 | ≤ 9/15 |

Per sample (AI %):

| Sample | desklib | max | desklib + gate | **new** | desklib doc |
|---|---:|---:|---:|---:|---:|
| ai-claude-next-steps (pasted answer) | 0 | 29 | 0 | **100** | 0.00 |
| ai-faceit-gemini | 0 | 100 | 0 | **100** | 0.06 |
| ai-plain-announcement-01 | 81 | 100 | 0 | **100** | 0.32 |
| ai-plain-announcement-02 | 38 | 61 | 38 | **100** | 0.88 |
| ai-plain-email-01 | 78 | 78 | 78 | **78** | 0.98 |
| ai-plain-email-02 | 100 | 100 | 100 | **100** | 1.00 |
| ai-plain-explainer-01 | 78 | 100 | 0 | **100** | 0.07 |
| ai-plain-explainer-02 | 49 | 100 | 0 | **100** | 0.06 |
| ai-plain-explainer-03 | 29 | 95 | 0 | **100** | 0.13 |
| ai-plain-help-01 | 94 | 100 | 94 | **100** | 1.00 |
| ai-plain-help-02 | 83 | 83 | 83 | **83** | 0.99 |
| ai-plain-help-03 | 80 | 80 | 80 | **100** | 0.98 |
| ai-plain-internal-01 | 19 | 64 | 0 | **100** | 0.13 |
| ai-plain-internal-02 | 30 | 85 | 30 | **100** | 0.78 |
| ai-plain-release-notes-01 | 58 | 73 | 58 | **100** | 0.55 |
| ai-plain-release-notes-02 | 96 | 96 | 96 | **96** | 0.56 |
| human-enron-01 | 83 | 83 | 0 | **0** | 0.23 |
| human-enron-03 | 68 | 68 | 0 | **0** | 0.08 |
| human-enron-06 | 100 | 100 | 0 | **0** | 0.31 |
| human-enron-14 | 77 | 77 | 0 | **0** | 0.37 |
| other 11 Enron | 0–37 | 0–37 | 0 | **0** | 0.00–0.44 |
| human-wiki-airbnb | 0 | 100 | 0 | **100** ✗ | 0.09 |
| human-wiki-counter-strike-go | 67 | 85 | 0 | **100** ✗ | 0.10 |
| human-wiki-dropbox | 100 | 100 | 100 | **100** ✗ | 0.89 |
| human-wiki-elo-rating-system | 79 | 79 | 79 | **79** ✗ | 0.88 |
| human-wiki-esports | 20 | 79 | 0 | **0** | 0.02 |
| human-wiki-fitbit | 90 | 100 | 0 | **0** | 0.29 |
| human-wiki-multi-factor-auth | 82 | 82 | 0 | **16** | 0.16 |
| human-wiki-netflix | 100 | 100 | 0 | **0** | 0.16 |
| human-wiki-payroll | 100 | 100 | 100 | **100** ✗ | 0.93 |
| human-wiki-spotify | 93 | 100 | 93 | **93** ✗ | 0.71 |
| human-wiki-steam-service | 100 | 100 | 0 | **0** | 0.40 |
| human-wiki-strava | 0 | 38 | 0 | **0** | 0.02 |
| human-wiki-twitch-service | 16 | 100 | 0 | **0** | 0.01 |
| human-wiki-uber | 45 | 45 | 0 | **0** | 0.02 |
| human-wiki-valve-anti-cheat | 0 | 0 | 0 | **0** | 0.06 |

### eval/samples

| | Clear human/AI | Mixed mean error | Highest AI% on a human sample |
|---|---|---|---|
| desklib only (before) | 20/20 | 17 pts | 42 (human-linkedin-01) |
| naive max (WIP) | 20/20 | 17 pts | 42 (human-linkedin-01) |
| desklib + gate | 20/20 | 16 pts | 0 |
| **new rule** | **20/20** | **16 pts** | **0** |

Mixed samples (expected → desklib / new): cover-letter-01 60 → 100 / 100,
email-01 40 → 80 / 80, email-02 30 → 46 / **36**, email-03 30 → 31 / 31,
linkedin-01 55 → 72 / 72, slack-02 65 → 67 / 67, slack-03 30 → 32 / 32.

### Latency (`eval --bench --threads 4`, median of 7, AI sample text)

| Words | desklib only (before) | desklib + gate | **new rule** | naive max |
|---|---|---|---|---|
| 153 | 0.64 s | 0.64 s | **1.08 s** | 1.12 s |
| 307 | 1.48 s | 1.48 s | **2.24 s** | 2.31 s |
| **508** | **2.43 s** | **2.52 s** | **3.49 s** | **3.76 s** |
| 996 | 5.25 s | 5.48 s | **7.95 s** | 7.85 s |
| ~2,030 (`--samples eval/hard`) | 5.62 s (no doc pass) | 10.31 s | **15.20 s** | – |

At 500 words the new rule costs +1.1 s (+44%) over desklib alone. rasbt
scores each paragraph and each ≤ 300-word doc chunk once. Up to 1,000 words
the gate is free, because the doc pass already ran. **Above 1,000 words the
gate forces the doc pass**, which used to be skipped there, so long texts
take roughly 2.5× as long (2,030 words: 5.6 s → 15.2 s).

## Verification

- Real pipeline vs sweep: the eval binary's AI% matches the sweep on every
  sample. The `--no-gate` baselines reproduce the earlier desklib and max
  runs, and keeping layout changed no desklib span score.
- `cargo test -p tracer-engine --lib`: 60 passed. Mock-detector tests cover:
  - **gate:** caps, lowers confidence, passes when the passage agrees, stays
    local to each doc chunk;
  - **paragraph lift:** rescues a paragraph, overrides the gate, does nothing
    below the threshold, scores paragraphs and doc chunks but never windows;
  - **whole-text lift:** catches a text whose paragraphs read human;
  - **confidence:** capping doesn't make undecided windows look decisive;
  - **line breaks:** `span_text` keeps them;
  - **errors:** a wrong-length style output is an error.
- `cargo test --release -p tracer-engine --lib -- --ignored`: 3 passed,
  including a real-model test (Gemini ≥ 50%; Enron 01/03/06/14 < 50%).
- `cargo test -p tracer --lib` (app): 8 passed.
- rasbt export parity: tokenizer 287/287 identical; FP32 and bf16w max |dP|
  0.0000.

## App wiring

`src-tauri` loads `models::Ensemble` and calls `Ensemble::analyze`. The
model counts as installed only when all four files are present. The
first-run download fetches desklib from the `model-v1` release and rasbt
from the `model-v2` release (pinned sizes and SHA-256s in `download.rs`).
An install that already has model-v1 only fetches the ~305 MB of model-v2.

## Remaining weaknesses

- **6/15 Wikipedia leads read AI.** Four of them (Dropbox, Elo, Payroll,
  Spotify) desklib reads as AI as whole texts; two (Airbnb, Counter-Strike)
  rasbt reads as AI as whole texts. Five report Low confidence and Dropbox
  Medium. Encyclopedic prose is the hardest false-positive case for both
  models.
- **Small sets, tuned thresholds.** 46 hard texts and 20 samples. rasbt's
  whole-text scores split into two groups here, but nothing guarantees
  that on new text.
- **Recall that depends on rasbt.** AI that desklib only catches window by
  window is now caught only if rasbt reads it as AI. The same holds for a
  mixed text whose AI part doesn't lift desklib's passage score to 0.5, e.g.
  one plain-style AI paragraph in a human email. No eval sample covers that.
- **rasbt depends on layout.** A text pasted without its line breaks (some
  apps flatten them) loses most of rasbt's signal.
- **Latency** is +44% at 500 words and ~2.5× above 1,000 words.
- **Memory.** The app loads both models for a check (desklib ~1.5 GB, plus
  rasbt) and unloads them 60 s after the last check.

## Reproduce

```
cargo run --release -p tracer-engine --bin eval -- --samples eval/hard                       # new rule
cargo run --release -p tracer-engine --bin eval -- --samples eval/hard --model tracer-int8.onnx --no-gate   # desklib only, before
cargo run --release -p tracer-engine --bin eval -- --samples eval/hard --model max --no-gate              # naive max
cargo run --release -p tracer-engine --bin eval -- --bench --threads 4
```

Span dumps for the sweep: `--model tracer-int8.onnx --dump desk-hard.tsv` and
`--model rasbt-bf16w.onnx --dump rasbt-hard.tsv` (likewise for
`eval/samples`), then `python -I tools/model-export/sweep_rules.py
desk-hard.tsv rasbt-hard.tsv desk-samples.tsv rasbt-samples.tsv [--all] [-v]`.
