//! Normalisation and sentence segmentation.
//!
//! The input is cut into an ordered list of [`Piece`]s whose byte ranges tile
//! the original text exactly (concatenating them gives the input back), so
//! the UI can re-render the text with highlights without any offset maths.
//!
//! * `Prose`: a sentence (or a chunk of an over-long one) that gets scored.
//! * `NonProse`: fenced code blocks, URL-only lines, emoji-only lines,
//!   anything with no words. Shown, never scored.
//! * `Gap`: whitespace between pieces.
//!
//! Each prose piece also carries its `model_text`: whitespace collapsed,
//! bare URLs removed, emoji kept. That's what the model sees.

use std::ops::Range;

use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

use crate::config::EngineConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PieceKind {
    Prose,
    NonProse,
    Gap,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    pub range: Range<usize>,
    pub kind: PieceKind,
    /// Prose only: what the model sees.
    pub model_text: String,
    /// Prose only: word count of `model_text` (UAX #29 words; emoji and
    /// punctuation don't count).
    pub words: usize,
}

/// Split `text` into pieces that tile it exactly.
pub fn segment(text: &str, cfg: &EngineConfig) -> Vec<Piece> {
    let mut out = Vec::new();
    let mut pos = 0;
    for block in code_block_ranges(text) {
        prose_region(text, pos..block.start, cfg, &mut out);
        push_trimmed(text, block.clone(), PieceKind::NonProse, &mut out);
        pos = block.end;
    }
    prose_region(text, pos..text.len(), cfg, &mut out);
    out
}

/// Paragraph number for every piece. A blank line (a gap with 2+ line
/// breaks) or a code block starts a new paragraph; single line breaks don't.
pub fn paragraph_ids(text: &str, pieces: &[Piece]) -> Vec<usize> {
    let mut para = 0;
    let mut out = Vec::with_capacity(pieces.len());
    for p in pieces {
        let s = &text[p.range.clone()];
        let boundary = match p.kind {
            PieceKind::Gap => s.matches('\n').count() >= 2,
            PieceKind::NonProse => s.contains('\n'),
            PieceKind::Prose => false,
        };
        if boundary {
            para += 1;
        }
        out.push(para);
        if boundary && p.kind == PieceKind::NonProse {
            para += 1;
        }
    }
    out
}

/// Collapse whitespace and drop bare URLs. Emoji and punctuation stay.
pub fn model_text(s: &str) -> String {
    s.split_whitespace().filter(|t| !is_url(t)).collect::<Vec<_>>().join(" ")
}

/// The model input for a run of prose pieces (`idx`, ascending piece
/// indices): their `model_text`s joined with the layout between them in the
/// source, so a line break stays "\n" and a blank line "\n\n". rasbt reads
/// layout: one AI answer scored 0.97 as pasted but 0.003 with its lines
/// joined by spaces. desklib's tokenizer ignores line breaks either way.
pub fn span_text(text: &str, pieces: &[Piece], idx: &[usize]) -> String {
    let mut out = String::new();
    for (k, &i) in idx.iter().enumerate() {
        if k > 0 {
            let between = &text[pieces[idx[k - 1]].range.end..pieces[i].range.start];
            out.push_str(match between.matches('\n').count() {
                0 => " ",
                1 => "\n",
                _ => "\n\n",
            });
        }
        out.push_str(&pieces[i].model_text);
    }
    out
}

pub fn count_words(s: &str) -> usize {
    s.unicode_words().count()
}

fn is_url(token: &str) -> bool {
    let t = token.trim_start_matches(|c: char| !c.is_alphanumeric()).to_ascii_lowercase();
    t.starts_with("http://") || t.starts_with("https://") || t.starts_with("www.")
}

/// Fenced (```) code blocks, by whole lines. An unclosed fence runs to the end.
fn code_block_ranges(text: &str) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut open: Option<usize> = None;
    let mut pos = 0;
    for line in text.split_inclusive('\n') {
        let start = pos;
        pos += line.len();
        let t = line.trim();
        if !t.starts_with("```") {
            continue;
        }
        match open {
            None if t.len() > 3 && t[3..].contains("```") => out.push(start..pos), // ```one-liner```
            None => open = Some(start),
            Some(s) => {
                out.push(s..pos);
                open = None;
            }
        }
    }
    if let Some(s) = open {
        out.push(s..text.len());
    }
    out
}

fn prose_region(text: &str, region: Range<usize>, cfg: &EngineConfig, out: &mut Vec<Piece>) {
    if region.is_empty() {
        return;
    }
    // UAX #29 sentence bounds. They break after every line break too, which
    // is what we want for Slack-style text (bullets, one-line paragraphs).
    // But UAX #29 also breaks after ?/! with no space ("a.com/x?y=1"), so a
    // boundary only counts after whitespace or a non-ASCII terminator (CJK
    // "。" has no space after it).
    let s = &text[region.clone()];
    let mut starts: Vec<usize> = s
        .split_sentence_bound_indices()
        .map(|(off, _)| off)
        .filter(|&off| off == 0 || s[..off].chars().next_back().is_some_and(|c| c.is_whitespace() || !c.is_ascii()))
        .collect();
    starts.extend(lowercase_continuations(s));
    starts.sort_unstable();
    starts.dedup();
    starts.push(s.len());
    for pair in starts.windows(2) {
        let r = region.start + pair[0]..region.start + pair[1];
        let (lead, core, trail) = trim_range(text, r);
        push_gap(lead, out);
        if !core.is_empty() {
            sentence_piece(text, core, cfg, out);
        }
        push_gap(trail, out);
    }
}

/// Abbreviations that end in "." but don't end a sentence.
const ABBREVIATIONS: &[&str] = &[
    "e.g", "i.e", "etc", "vs", "cf", "approx", "mr", "mrs", "ms", "dr", "st", "no", "fig", "inc", "ltd", "jr", "sr",
    "a.m", "p.m", "eg", "ie", "esp", "incl", "dept", "est", "min", "max", "avg",
];

/// UAX #29 never breaks ". " before a lowercase letter (to protect "e.g.
/// foo"), but casual text is often all lowercase: "found it. it was the
/// flag". Break there unless the word before the period is an abbreviation,
/// a single letter (initials), or part of an ellipsis.
fn lowercase_continuations(s: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let b = s.as_bytes();
    for (i, _) in s.match_indices('.') {
        // previous char must not be '.' (ellipsis) and must be a letter/digit/closer
        if i == 0 || b[i - 1] == b'.' {
            continue;
        }
        // whitespace (no newline needed; UAX already breaks there), then a lowercase letter
        let rest = &s[i + 1..];
        let ws = rest.len() - rest.trim_start().len();
        if ws == 0 {
            continue;
        }
        let Some(next) = rest[ws..].chars().next() else { continue };
        if !next.is_lowercase() {
            continue;
        }
        let word_start = s[..i].rfind(char::is_whitespace).map_or(0, |p| p + 1);
        let word: String = s[word_start..i]
            .trim_start_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if word.chars().count() <= 1 || ABBREVIATIONS.contains(&word.as_str()) {
            continue;
        }
        out.push(i + 1 + ws);
    }
    out
}

fn sentence_piece(text: &str, r: Range<usize>, cfg: &EngineConfig, out: &mut Vec<Piece>) {
    let mt = model_text(&text[r.clone()]);
    let words = count_words(&mt);
    if words == 0 {
        out.push(Piece { range: r, kind: PieceKind::NonProse, model_text: String::new(), words: 0 });
        return;
    }
    if words <= cfg.max_sentence_words {
        out.push(Piece { range: r, kind: PieceKind::Prose, model_text: mt, words });
        return;
    }
    // Over-long sentence: cut at whitespace every N tokens.
    let tokens = whitespace_tokens(text, r);
    let n = cfg.long_sentence_chunk_words.max(1);
    let mut prev_end: Option<usize> = None;
    for group in tokens.chunks(n) {
        let cr = group[0].start..group[group.len() - 1].end;
        if let Some(pe) = prev_end {
            push_gap(pe..cr.start, out);
        }
        prev_end = Some(cr.end);
        let mt = model_text(&text[cr.clone()]);
        let words = count_words(&mt);
        let kind = if words == 0 { PieceKind::NonProse } else { PieceKind::Prose };
        let mt = if words == 0 { String::new() } else { mt };
        out.push(Piece { range: cr, kind, model_text: mt, words });
    }
}

fn whitespace_tokens(text: &str, r: Range<usize>) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text[r.clone()].char_indices() {
        let i = r.start + i;
        match (c.is_whitespace(), start) {
            (true, Some(s)) => {
                out.push(s..i);
                start = None;
            }
            (false, None) => start = Some(i),
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push(s..r.end);
    }
    out
}

/// Split a range into (leading whitespace, core, trailing whitespace).
fn trim_range(text: &str, r: Range<usize>) -> (Range<usize>, Range<usize>, Range<usize>) {
    let s = &text[r.clone()];
    let lead = s.len() - s.trim_start().len();
    let core_len = s.trim().len();
    let a = r.start + lead;
    let b = a + core_len;
    (r.start..a, a..b, b..r.end)
}

fn push_trimmed(text: &str, r: Range<usize>, kind: PieceKind, out: &mut Vec<Piece>) {
    let (lead, core, trail) = trim_range(text, r);
    push_gap(lead, out);
    if !core.is_empty() {
        out.push(Piece { range: core, kind, model_text: String::new(), words: 0 });
    }
    push_gap(trail, out);
}

fn push_gap(r: Range<usize>, out: &mut Vec<Piece>) {
    if r.is_empty() {
        return;
    }
    // Merge with a preceding gap so the UI never sees two in a row.
    if let Some(last) = out.last_mut() {
        if last.kind == PieceKind::Gap && last.range.end == r.start {
            last.range.end = r.end;
            return;
        }
    }
    out.push(Piece { range: r, kind: PieceKind::Gap, model_text: String::new(), words: 0 });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> EngineConfig {
        EngineConfig::default()
    }

    fn tiles(text: &str, pieces: &[Piece]) {
        let mut pos = 0;
        for p in pieces {
            assert_eq!(p.range.start, pos, "gap/overlap before {:?}", p);
            pos = p.range.end;
        }
        assert_eq!(pos, text.len(), "pieces don't reach the end");
        let rebuilt: String = pieces.iter().map(|p| &text[p.range.clone()]).collect();
        assert_eq!(rebuilt, text);
    }

    fn prose<'a>(text: &'a str, pieces: &[Piece]) -> Vec<&'a str> {
        pieces.iter().filter(|p| p.kind == PieceKind::Prose).map(|p| &text[p.range.clone()]).collect()
    }

    #[test]
    fn splits_sentences_and_tiles() {
        let t = "First sentence here. Second one! Third? Yes.";
        let p = segment(t, &cfg());
        tiles(t, &p);
        assert_eq!(prose(t, &p), vec!["First sentence here.", "Second one!", "Third?", "Yes."]);
    }

    #[test]
    fn span_text_keeps_line_breaks_between_sentences() {
        let t = "First one here.  Second  one.\n* bullet line\n\nNew paragraph.";
        let p = segment(t, &EngineConfig::default());
        let prose: Vec<usize> = p.iter().enumerate().filter(|(_, x)| x.kind == PieceKind::Prose).map(|(i, _)| i).collect();
        assert_eq!(span_text(t, &p, &prose), "First one here. Second one.\n* bullet line\n\nNew paragraph.");
    }

    #[test]
    fn line_breaks_and_bullets_are_boundaries() {
        let t = "status update\n- shipped the thing\n- fixed login bug\n\nthat's it";
        let p = segment(t, &cfg());
        tiles(t, &p);
        assert_eq!(prose(t, &p), vec!["status update", "- shipped the thing", "- fixed login bug", "that's it"]);
    }

    #[test]
    fn urls_dont_split_or_count() {
        let t = "see https://example.com/a.b?c=d for details. ok";
        let p = segment(t, &cfg());
        tiles(t, &p);
        assert_eq!(prose(t, &p), vec!["see https://example.com/a.b?c=d for details.", "ok"]);
        assert_eq!(p[0].model_text, "see for details.");
        assert_eq!(p[0].words, 3);
    }

    #[test]
    fn lowercase_slack_sentences_split() {
        let t = "ok found it. it was the flag. no idea why tho";
        let p = segment(t, &cfg());
        tiles(t, &p);
        assert_eq!(prose(t, &p), vec!["ok found it.", "it was the flag.", "no idea why tho"]);
    }

    #[test]
    fn abbreviations_ellipses_and_initials_dont_split() {
        let t = "bring snacks e.g. crisps etc. and dips. wait... nvm. ask j. smith vs. the others";
        let p = segment(t, &cfg());
        tiles(t, &p);
        assert_eq!(prose(t, &p), vec!["bring snacks e.g. crisps etc. and dips.", "wait... nvm.", "ask j. smith vs. the others"]);
    }

    #[test]
    fn no_mid_token_breaks() {
        let t = "Docs at https://x.com/a?b=c&d=e!f here. Next one.";
        let p = segment(t, &cfg());
        tiles(t, &p);
        assert_eq!(prose(t, &p), vec!["Docs at https://x.com/a?b=c&d=e!f here.", "Next one."]);
    }

    #[test]
    fn url_only_and_emoji_only_lines_are_non_prose() {
        let t = "https://example.com\n🎉🎉🔥\nreal words here";
        let p = segment(t, &cfg());
        tiles(t, &p);
        let kinds: Vec<_> = p.iter().filter(|p| p.kind != PieceKind::Gap).map(|p| p.kind).collect();
        assert_eq!(kinds, vec![PieceKind::NonProse, PieceKind::NonProse, PieceKind::Prose]);
    }

    #[test]
    fn emoji_kept_in_model_text_but_not_counted() {
        let t = "Shipped it 🚀 thanks all";
        let p = segment(t, &cfg());
        assert_eq!(p[0].model_text, "Shipped it 🚀 thanks all");
        assert_eq!(p[0].words, 4);
    }

    #[test]
    fn fenced_code_is_non_prose() {
        let t = "try this:\n```rust\nfn main() { println!(\"hi. there.\"); }\n```\nworked for me";
        let p = segment(t, &cfg());
        tiles(t, &p);
        assert_eq!(prose(t, &p), vec!["try this:", "worked for me"]);
        let code: Vec<_> = p.iter().filter(|p| p.kind == PieceKind::NonProse).collect();
        assert_eq!(code.len(), 1);
        assert!(t[code[0].range.clone()].starts_with("```rust"));
        assert!(t[code[0].range.clone()].ends_with("```"));
    }

    #[test]
    fn unclosed_fence_runs_to_end() {
        let t = "look:\n```\nlet x = 1;\nmore. code.";
        let p = segment(t, &cfg());
        tiles(t, &p);
        assert_eq!(prose(t, &p), vec!["look:"]);
    }

    #[test]
    fn one_line_fence_is_non_prose() {
        let t = "before\n```echo hi```\nafter";
        let p = segment(t, &cfg());
        tiles(t, &p);
        assert_eq!(prose(t, &p), vec!["before", "after"]);
    }

    #[test]
    fn long_run_on_sentence_is_chunked() {
        let words: Vec<String> = (0..200).map(|i| format!("w{i}")).collect();
        let t = words.join(" ");
        let p = segment(&t, &cfg());
        tiles(&t, &p);
        let pr: Vec<_> = p.iter().filter(|p| p.kind == PieceKind::Prose).collect();
        assert_eq!(pr.len(), 5); // 45+45+45+45+20
        assert!(pr.iter().all(|p| p.words <= cfg().max_sentence_words));
        assert_eq!(pr.iter().map(|p| p.words).sum::<usize>(), 200);
    }

    #[test]
    fn paragraphs_split_on_blank_lines_and_code_only() {
        let t = "one. two.\nthree.\n\nfour.\r\n\r\nfive.\n```\ncode\n```\nsix.";
        let p = segment(t, &cfg());
        let ids = paragraph_ids(t, &p);
        let by_sentence: Vec<(&str, usize)> =
            p.iter().zip(&ids).filter(|(p, _)| p.kind == PieceKind::Prose).map(|(p, &i)| (&t[p.range.clone()], i)).collect();
        assert_eq!(by_sentence, vec![("one.", 0), ("two.", 0), ("three.", 0), ("four.", 1), ("five.", 2), ("six.", 4)]);
    }

    #[test]
    fn whitespace_is_collapsed_for_the_model() {
        let t = "lots   of\tspace   here.";
        let p = segment(t, &cfg());
        assert_eq!(p[0].model_text, "lots of space here.");
    }

    #[test]
    fn empty_and_whitespace_only() {
        assert!(segment("", &cfg()).is_empty());
        let t = "  \n\t ";
        let p = segment(t, &cfg());
        tiles(t, &p);
        assert!(p.iter().all(|p| p.kind == PieceKind::Gap));
        assert_eq!(p.len(), 1, "consecutive gaps merge");
    }

    #[test]
    fn crlf_and_unicode_tile() {
        let t = "Héllo wörld.\r\nZweiter Satz — mit Strich.\r\n日本語の文です。次の文。";
        let p = segment(t, &cfg());
        tiles(t, &p);
        assert!(prose(t, &p).len() >= 3);
    }
}
