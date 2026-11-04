# AI-writing tells (style hints)

*2026-11-04. Rules: `crates/tracer-engine/data/ai-tells.json` (v1.0.0, 45
rules, 44 enabled). Code: `crates/tracer-engine/src/tells.rs`. UI:
`src/components/StyleHints.tsx`.*

## What it does

Every check runs the regex rules over the whole text and lists the ones that
fire at least `minMatches` times. The list appears under the highlighted
text as **Style hints · not proof**. Each hint shows:
- its label, with the description on hover;
- how often it fired;
- a 1–5 strength from the rule's weight;
- up to three matched snippets. Invisible characters are named, e.g.
  `‹U+2011›`.

**Hybrid scoring (your decision, 2026-11-04):**
- **Near-certain artefacts (weight 5) score.** These are U+2011 hyphens,
  template placeholders like `[Your Name]`, chatbot citation leftovers
  (`oaicite`, `turn0search0`, `utm_source=chatgpt.com`) and "As an AI
  language model". Each one raises **the paragraph it sits in** to 0.95 AI.
  Those sentences count as decided for confidence, and the hint is tagged
  "copy-paste artefact · counts as AI".
- **Everything else never changes the score.** That covers em dashes, curly
  quotes, stock phrases, chatbot openers and closers, and the statistical
  signals. People write these too.

Config: `EngineConfig.artefact_min_weight` (5; 6 turns scoring off) and
`artefact_lift_score` (0.95).

### Statistical signals

These are computed at document level and shown only:
- low sentence-length variation (CV < 0.35, 5+ sentences);
- uniform paragraph lengths (CV < 0.3, 3+ paragraphs);
- high list density (> 40% of lines are bullets or headings, under 300 words).

Not implemented:
- **casual markers and contraction rate:** both only mean something in chat
  text, and Tracer can't tell what kind of text it was given;
- **lexical diversity:** the file itself calls it a weak tiebreaker;
- **perplexity:** needs another local model.

### Conversion and precision

The rules' ECMAScript patterns are converted for Rust's `regex` crate:
- flags `i`/`m` become inline `(?i)`/`(?m)`;
- the one lookahead becomes a non-capturing group.

A test checks that every enabled rule compiles and matches its own
`example`.

As the file's note suggests, `not-x-its-y` only counts a match that contains
"not" or "n't". Without that it fired on two human Wikipedia leads.

## On the eval sets

| Set | Texts with any hint | Hints per text | Texts with a scoring artefact |
|---|---|---|---|
| eval/hard AI (16) | 11 | 1.1 | 0 |
| eval/hard human (30) | 9 | 0.3 | 0 |
| eval/samples AI (12) | 9 | 2.5 | 0 |
| eval/samples human (14) | 3 | 0.2 | 0 |
| eval/samples mixed (8) | 5 | 1.2 | 0 |

No eval text contains an artefact, so every AI% is unchanged. On human
text, the hints that fire are mostly statistical signals (low sentence-length
variation, list density, uniform paragraphs) plus:
- rule-of-three on two Wikipedia leads;
- a spaced en dash on one;
- 12 zero-width characters on the Airbnb lead, which came with the Wikipedia
  text. That rule is weight 4, so it shows a hint and doesn't score.

## Known gaps

- **No in-text highlighting.** Hints list the matched text but don't
  highlight it in the pasted text. The engine has byte ranges; the UI would
  need UTF-16 offsets and spans that cross sentence boundaries.
- **Hints on human text.** A human text can show several hints (e.g. Steam:
  rule-of-three ×6). The "not proof" label and the fact that they never score
  are the guard.
