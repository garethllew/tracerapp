# Tracer eval samples

**Caveat: all samples are synthetic. The "human" ones were written by an AI imitating human style, so they are not ground truth. Replace or supplement them with real human-written text (your own old Slack messages, emails written before 2022) for a trustworthy eval.**

## Layout

```
eval/samples/
  README.md
  <expected>-<kind>-<nn>.txt   30 main samples (human / ai / mixed)
  edge/edge-*.txt              4 edge cases (emoji only, URL only, non-English, code block)
```

Length mix: 3 files under 15 words (tests the "too short" gate), 6 files of 15-50 words, most of the rest 60-200 words, and 3 files of 250-450 words (one per category).

## Front matter

Each file starts with a block between two `---` lines, then the text.

| Field | Values | Meaning |
|---|---|---|
| `expected` | `human`, `ai`, `mixed` | Ground-truth label |
| `expected_ai_share` | 0.0 - 1.0 | Approximate fraction of words that are AI-written (human = 0.0, ai = 1.0, mixed = from word counts, rounded to 0.05) |
| `kind` | `slack`, `email`, `linkedin`, `cover-letter`, `other` | Genre of the text |
| `source` | `synthetic` | Leave as `synthetic` for generated samples; use something else (e.g. `real`) for genuine text |
| `notes` | one line | What the sample tests; for `mixed`, which paragraph is the AI part |

The text body is everything after the closing `---`. Strip the front matter before feeding text to the detector.

## Adding a sample

1. Create `<expected>-<kind>-<nn>.txt` (UTF-8), using the next free number for that expected/kind pair, e.g. `human-slack-06.txt`. Edge cases go in `edge/` named `edge-<what>.txt`.
2. Add the front matter above, then paste the text.
3. For `mixed`, count words in the AI part and the whole text, set `expected_ai_share` to the ratio (round to 0.05), and say which paragraph is AI in `notes`.
4. For real human text, set `source: real` and keep `expected: human`, `expected_ai_share: 0.0`.
