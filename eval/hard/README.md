# Hard cases

This folder holds hard cases for comparing AI-text detector models: plain, brand-voice AI text and plain human text.

AI files (`ai-plain-*.txt`) are model-written in a disciplined house style: concrete, factual, no hype words, no em-dash pairs, no bullet lists. Each file has a front-matter header (`expected`, `expected_ai_share`, `kind`, `source`, `notes`) followed by the text.

Human files will be added separately (pre-2022 Wikipedia excerpts).

## Human sources (genuine, pre-LLM)

- `human-wiki-*.txt`: lead sections of English Wikipedia articles as of May 2019 (revision ids in each file's `source`). Text is CC BY-SA 4.0, © Wikipedia contributors. Fetched by `tools/model-export/fetch_wiki_human.py`. Encyclopedic prose is a hard false-positive case: LLMs imitate it closely.
- `human-enron-*.txt`: business emails from the public Enron email corpus (2000-2001), via the `corbt/enron-emails` dataset. Fetched by `tools/model-export/fetch_enron_human.py`. This is the register closest to what Tracer users paste.

`ai-faceit-gemini.txt` is the real-world miss that prompted this set: Gemini text run through a brand-voice skill (Pangram: 100% AI; desklib: 6%).

Compare detectors with `tools/model-export/compare_models.py` and `analyse_compare.py` (see the docstrings).
