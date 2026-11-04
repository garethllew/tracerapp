# Phase 1: Detection engine report

*2026-11-04 · `crates/tracer-engine` · INT8 model from Phase 0 · Ryzen 7 7800X3D*

## Verdict

The engine works end to end and is inside the quality bar:

- **Clear human/AI samples:** 20/20 correct.
- **Mixed samples:** typically within ~17 points of the true AI share.
- **Speed:** 500 words in **1.4 s on 4 threads**, under 1 s on 8.
- **Robustness:** no panics on weird input.
- **Parity:** Rust inference matches the Python reference to 6 decimal places.

Getting there took three fixes the brief didn't anticipate (below). **One of them
changes the brief's design (the whole-document check), and I need your OK on it.**

## What's built

| Module | What it does |
|---|---|
| `config.rs` | `EngineConfig`: every threshold in one struct (gates, windows, confidence, limits) |
| `text.rs` | Normalises and segments text into pieces that **tile the input exactly** (prose / non-prose / gap), so the UI re-renders highlights with no offset maths. Handles fenced code, bare URLs, emoji-only lines, bullets, CRLF and CJK. |
| `window.rs` | Paragraph groups, overlapping ~3-sentence / 40–70-word windows, doc chunks |
| `detector.rs` | `Detector` trait + `OnnxDetector` (ort 2.0.0-rc.15, `tokenizers` 0.23) |
| `analyze.rs` | Pipeline: gate → windows → score → per-sentence mean → word-weighted split → doc check → confidence. Returns a serde `Analysis` ready for the Tauri layer. |
| `bin/eval.rs` | `cargo run --release --bin eval`: predicted vs expected table, plus `--verbose`, `--bench`, `--raw` |

**Tests:** 47 unit tests, plus 1 real-model integration test (`cargo test --release -- --include-ignored`). They cover segmentation, windowing, grouping, aggregation, confidence, length gating, and the weird-input set: empty, whitespace, emoji-only, URL-only, 50k chars, control characters, French/Japanese/Russian/Arabic.

**Small deviation:** `Detector::score` returns `Result<Vec<f32>>`, not `Vec<f32>`, so ONNX or tokeniser failures surface as errors instead of panics.

## Three things the brief didn't anticipate

### 1. Windows must not cross paragraphs (accuracy)
With windows that cross paragraphs, a pasted AI paragraph scored ~1.00 and dragged every window touching it to "AI". Human sentences next to it lit up too ("dinner at some point?" scored 1.00), and mixed-sample error was **44 points**.

Windows now stay inside blank-line paragraphs. Paragraphs under 10 words (greetings, sign-offs, one-line LinkedIn paragraphs) merge into a neighbour first. Mixed error dropped to **17 points**.
Example, `mixed-email-03`: human sentences 0.11–0.17 around a 1.00 AI blurb.

Single line breaks don't split groups, since Slack lines are too short to judge alone.

### 2. Lowercase sentences
The Unicode sentence algorithm won't split `". "` before a lowercase letter, so casual
Slack text ("found it. it was the flag") came out as one giant sentence. I added a
lowercase-continuation split that skips abbreviations (e.g., etc., vs.), initials and ellipses.
I also stopped it splitting inside URLs (`?`/`!` in query strings).

### 3. INT8 + batching = scores depend on neighbours
The INT8 model's dynamic quantisation sets each layer's scale from the whole batch
tensor. That made a chunk's score shift depending on what else was batched with it
(0.655 → 0.674 for the same text). Chunks now run one at a time. The result is deterministic, identical to Python, and costs under 5% in speed.

## ⚠ Decision needed: the whole-document check

The brief asks for a document-level score on every text as a sanity check. On 500
words it **more than doubled the time** (3.1 s vs 1.4 s on 4 threads), because it runs long
512-token sequences. In practice it is also a weak check: any text containing an AI paragraph
scores ~1.0 at document level (Phase 0), so it rarely tells us anything.

**What I did (configurable):** the doc check runs only for texts up to **300 words**
(`doc_check_max_words`).

- **Short texts:** this is where it matters most, because there are few windows to go on. Up to ~70 words it's free: the single window doubles as the doc chunk.
- **Longer texts:** windows carry the signal. `doc_score` is `None` and there is no disagreement downgrade.
- **Disagreement rule:** "doc ≥ 0.8 while < 10% of words classed AI" (or the reverse at ≤ 0.2 / > 50%). A naive "doc vs split differ" test would fire on every mixed text, because doc ≈ 1.0 whenever any AI is present.

The options: keep this (my recommendation); run it always and accept ~3 s on laptops for long texts; or run it after the result is shown and update confidence when it lands, which is more UI complexity.

## Numbers

**Eval** (`cargo run --release --bin eval`, synthetic samples, so a smoke test only):

- Clear human/AI: **20/20**.
- Too-short gate: 5 caught (the 3 sub-15-word samples plus emoji-only and URL-only).
- Mixed: 5/7 within ±25 points, mean error 17 points. The two misses are the model scoring human paragraphs as AI-ish: a formal "My gran ran a bakery…" cover-letter opening (0.77) and a "Subject: re:" email opener (0.57). That's model behaviour, not pipeline.
- French sample flagged `non-english` with Low confidence.

**Latency** (median after warm-up; real pastes are ~10–20% slower):

| | 150 words (incl. doc) | 300 words | 500 words |
|---|---|---|---|
| 4 threads (≈ laptop) | 0.61 s | 0.82 s | **1.38 s** |
| 8 threads | 0.41 s | 0.60 s | 0.99 s |

Load once: ~1.0 s. Threads default to half the logical CPUs, capped at 8.

## Limits worth knowing

- Text beyond 5,000 prose words is shown but not scored, flagged `truncated` (`max_scored_words`). That stops a pasted novel from tying up the CPU. It isn't in the brief, so say if you'd rather drop it.
- An AI *sentence* inside a human paragraph can't be isolated well. Windows are ~3 sentences, so it gets diluted. Pasted AI *paragraphs* are caught cleanly.
- Language detection (`whatlang`) only fires when it's confident, so short non-English snippets can slip through as "English".

## Next: Phase 2 (app shell + UI)

- **Sonnet session:** scaffold the Tauri 2 app from Vixer's setup (window chrome, fonts, tokens, plugin wiring), with `src-tauri` joining this workspace.
- **Opus:** the design system, result bar, highlighted text and the tracer animation, i.e. the visual core.
