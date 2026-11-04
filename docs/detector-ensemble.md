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
- **rasbt's whole-text score is ~0.00 on every multi-paragraph text except
  Gemini** (0.85), AI or human. It reads long passages as human, so it can't
  be the context.
- **At paragraph level rasbt is useful when it's very sure**: ≥ 0.95 on 3/3
  Gemini paragraphs, and on most paragraphs of the announcement and explainer
  texts desklib misses. Only three human paragraphs in the whole set reach
  that: Airbnb (0.998), Counter-Strike (0.997) and multi-factor
  authentication (0.951).

## The rule

A sentence's score starts as the mean of its desklib windows, as before.
Then:

1. **Context gate.** If desklib's score for the sentence's doc chunk (the
   whole text, or the ≤ 300-word passage around it) is below **0.5**, the
   sentence's score is capped at that doc-chunk score.
2. **Style lift.** If rasbt's score for the sentence's paragraph (the
   paragraph group, cut into ≤ 300-word chunks) is at least **0.95**, the
   sentence's score is raised to it. This applies whether or not the gate
   capped it.

In words: **a sentence reads AI when desklib says so and desklib's read of
the surrounding passage agrees, or when rasbt is ≥ 95% sure about its
paragraph.**

Confidence: a sentence whose verdict the gate or the lift flipped counts as
undecided. Without this, a capped score like 0.07 looked decisive, and two
missed AI texts reported "High confidence human".

Why this rule, in data terms (hard set, sweep; the eval binary agrees):

| Rule (sweep name) | Gemini | Hard AI ≥ 50% | Enron flagged | Wikipedia flagged |
|---|---|---|---|---|
| desklib only | 0% | 9/15 | 4/15 | 9/15 |
| naive max (`max`) | 100% | 15/15 | 4/15 | 12/15 |
| rasbt counts only at window ≥ 0.9 (`rasT0.9`) | 100% | 14/15 | 4/15 | 11/15 |
| rasbt calibrated, logit − 2 (`rasK2.0`) | 100% | 14/15 | 4/15 | 11/15 |
| desklib + rasbt windows only where rasbt's paragraph ≥ 0.9 (`liftG0.9`) | 100% | 13/15 | 4/15 | 9/15 |
| max, gated by the paragraph (`max+cG@0.5`) | 100% | 13/15 | 4/15 | 7/15 |
| desklib, gated by the paragraph (`desk+gG@0.5`) | 0% | 9/15 | 4/15 | 6/15 |
| desklib, gated by the doc chunk (`desk+gD@0.5`) | 0% | 7/15 | **0/15** | **4/15** |
| max, gated by the doc chunk (`max+cD@0.5`) | 100% | 10/15 | 0/15 | 4/15 |
| **gate + lift ≥ 0.95 (`either0.95/0.5`, chosen)** | **100%** | **12/15** | **0/15** | **4/15** |
| gate + lift ≥ 0.9 | 100% | 12/15 | 0/15 | 5/15 |
| gate + lift ≥ 0.97 | 100% | 11/15 | 0/15 | 4/15 |
| gate + lift ≥ 0.98 | 100% | 10/15 | 0/15 | 4/15 |
| gate + lift ≥ 0.99 | 65% | 9/15 | 0/15 | 4/15 |
| gate at 0.4 + lift ≥ 0.95 | 100% | 12/15 | 0/15 | 5/15 |

- Gating on the **paragraph** never fixes Enron: those emails' short
  paragraphs read AI too. Only the longer **doc-chunk** read separates them.
- Every way of letting rasbt's **windows** count (threshold, calibration,
  lift) keeps Wikipedia at 9–12/15. Only its paragraph score is safe to use,
  and only when it's very high.
- The lift threshold is a narrow optimum. 0.9 lets Counter-Strike through
  (79% AI), and 0.97 drops a hard AI text. Both neighbours are worse, so
  0.95 is the data's choice, not a round number.

Both thresholds are in `EngineConfig` (`context_gate`, `style_lift_threshold`).
Setting `context_gate = 0` turns the gate off. Without a style detector there
is no lift, so desklib-only operation is `analyze()` and the ensemble is
`analyze_with(text, desklib, Some(rasbt), cfg)` / `models::Ensemble::analyze`.

## Results (eval binary)

### eval/hard

| | Gemini | Hard AI ≥ 50% | Enron flagged | Wikipedia flagged |
|---|---|---|---|---|
| desklib only (before) | 0% | 9/15 | 4/15 (01, 03, 06, 14) | 9/15 |
| naive max (WIP) | 100% | 15/15 | 4/15 (01, 03, 06, 14) | 12/15 |
| desklib + gate (no rasbt) | 0% | 7/15 | 0/15 | 4/15 |
| **new rule** | **100%** | **12/15** | **0/15** | **4/15** |
| target | ≥ 50% | ≥ 12/15 | ≤ 1/15 | ≤ 9/15 |

Per sample (AI %):

| Sample | desklib | max | desklib + gate | **new** | desklib doc |
|---|---:|---:|---:|---:|---:|
| ai-faceit-gemini | 0 | 100 | 0 | **100** | 0.06 |
| ai-plain-announcement-01 | 81 | 100 | 0 | **75** | 0.32 |
| ai-plain-announcement-02 | 38 | 61 | 38 | **61** | 0.88 |
| ai-plain-email-01 | 78 | 78 | 78 | **78** | 0.98 |
| ai-plain-email-02 | 100 | 100 | 100 | **100** | 1.00 |
| ai-plain-explainer-01 | 78 | 100 | 0 | **22** ✗ | 0.07 |
| ai-plain-explainer-02 | 49 | 100 | 0 | **75** | 0.06 |
| ai-plain-explainer-03 | 29 | 95 | 0 | **53** | 0.13 |
| ai-plain-help-01 | 94 | 100 | 94 | **94** | 1.00 |
| ai-plain-help-02 | 83 | 83 | 83 | **83** | 0.99 |
| ai-plain-help-03 | 80 | 80 | 80 | **80** | 0.98 |
| ai-plain-internal-01 | 19 | 64 | 0 | **20** ✗ | 0.13 |
| ai-plain-internal-02 | 30 | 85 | 30 | **30** ✗ | 0.78 |
| ai-plain-release-notes-01 | 58 | 73 | 58 | **73** | 0.55 |
| ai-plain-release-notes-02 | 96 | 96 | 96 | **96** | 0.56 |
| human-enron-01 | 83 | 83 | 0 | **0** | 0.23 |
| human-enron-03 | 68 | 68 | 0 | **0** | 0.08 |
| human-enron-06 | 100 | 100 | 0 | **0** | 0.31 |
| human-enron-14 | 77 | 77 | 0 | **0** | 0.37 |
| other 11 Enron | 0–37 | 0–37 | 0 | **0** | 0.00–0.44 |
| human-wiki-airbnb | 0 | 100 | 0 | **43** | 0.09 |
| human-wiki-counter-strike-go | 67 | 85 | 0 | **46** | 0.10 |
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
The gate didn't suppress the AI parts of mixed texts. Where the AI is
chatbot style, it dominates desklib's read of the passage (doc 0.79–1.00).

### Latency (`eval --bench --threads 4`, median of 7, AI sample text)

| Words | desklib only (before) | desklib + gate | **new rule** | naive max |
|---|---|---|---|---|
| 153 | 0.64 s | 0.64 s | **0.96 s** | 1.12 s |
| 307 | 1.48 s | 1.48 s | **1.97 s** | 2.31 s |
| **508** | **2.43 s** | **2.52 s** | **3.23 s** | **3.76 s** |
| 996 | 5.25 s | 5.48 s | **7.03 s** | 7.85 s |
| 2,030 (`--samples eval/hard`) | 5.62 s (no doc pass) | 10.31 s | **12.80 s** | – |

At 500 words the new rule costs +0.8 s (+33%) over desklib alone: rasbt
scores each paragraph once. That's cheaper than the naive max, which ran
rasbt on every window. Up to 1,000 words the gate is free, because the doc
pass already ran. **Above 1,000 words the gate forces the doc pass**, which
used to be skipped there, so long texts take about twice as long (2,030
words: 5.6 s → 10.3 s with desklib, 12.8 s with the new rule).

## Verification

- Real pipeline vs sweep: the eval binary's AI% matches the sweep's
  `either0.95/0.5` on every sample of both sets. The `--no-gate` baselines
  reproduce the earlier desklib and max runs exactly.
- `cargo test -p tracer-engine --lib`: 57 passed. New mock-detector tests:
  the gate caps (and lowers confidence), passes when the passage agrees, is
  local to each doc chunk; the lift rescues a paragraph, overrides the gate,
  does nothing below the threshold, scores paragraphs only (never windows),
  and a wrong-length style output is an error.
- `cargo test --release -p tracer-engine --lib -- --ignored`: 3 passed,
  including a new real-model test (Gemini ≥ 50%; Enron 01/03/06/14 < 50%).
  The ignored tests now find models via `models::dev_model_dir()`
  (`$TRACER_MODEL_DIR`, then the app's model folder, then
  `tools/model-export/work/onnx`).
- rasbt export parity (`export_rasbt.py --skip-export`): tokenizer 287/287
  identical; FP32 and bf16w both max |dP| 0.0000, 0 flips. The models
  already in `%APPDATA%\com.tracer.app\models` were used; the rasbt file
  there is byte-identical to the export.
- `src-tauri` still compiles against the changed engine (`cargo check -p
  tracer`, checked with a placeholder `dist/`).

## Remaining weaknesses

- **Small sets, tuned thresholds.** Both thresholds were picked on 30 hard
  texts and 20 samples. The hard-AI target (12/15) is met at its edge:
  explainer-03 reads 53%. Counter-Strike (46%) and Airbnb (43%) sit just
  under the human line. Expect some movement on new data.
- **Partial false highlights on human text.** rasbt is ≥ 95% sure about one
  human paragraph each in Airbnb, Counter-Strike and multi-factor
  authentication, so those paragraphs show as AI (43%, 46%, 16% of words).
  Counter-Strike even reports High confidence for its 46%, because desklib's
  windows agreed with the lifted paragraph. An untested idea: let rasbt lift
  only when its confident paragraphs make up ≥ 50% of the text. On these
  sets that would zero these three without losing a hard-AI hit, but it
  would also stop rasbt from flagging a single AI paragraph in a mixed text.
- **4/15 Wikipedia leads still read AI** (Dropbox, Elo, Payroll, Spotify).
  desklib reads them as AI as whole texts too (0.71–0.93), so no rule over
  these two models removes them without cutting real AI. Encyclopedic prose
  remains the hardest false-positive case.
- **Recall lost to the gate.** AI text that desklib only catches at window
  level (explainer-01 78% → 22%, internal-01) is now missed unless rasbt is
  sure. The same holds for a mixed text whose AI part doesn't lift desklib's
  passage score to 0.5, e.g. one plain-style AI paragraph in a human email.
  The eval set has no such sample.
- **rasbt can't help on long single paragraphs.** It reads long passages as
  human, and paragraph groups are only cut at 300 words. A 250-word
  single-paragraph brand-voice text depends entirely on desklib. Gemini
  works because it has three paragraphs.
- **Latency above 1,000 words** roughly doubles (see the table). If that's
  unacceptable, the trade is to skip the gate there and accept desklib's
  window false positives on long texts.
- **The app isn't wired to the ensemble yet.** `src-tauri` still loads
  desklib alone and calls `analyze()`, so on its next build it gets the gate
  (Enron 0/15, Wikipedia 4/15, but hard AI 9 → 7/15, Gemini still 0%) and
  not the lift. Shipping the ensemble needs the app to download rasbt
  (model-v2) and call `models::Ensemble::analyze`.

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
