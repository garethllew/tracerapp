//! Overlapping windows over prose sentences, plus the document-level chunks.
//!
//! Single sentences are too short for the classifier to judge, so each
//! sentence is scored as part of windows of ~3 sentences / 40–70 words, and
//! later gets the mean of the windows that contain it.

use std::ops::Range;

use crate::config::EngineConfig;

/// Windows over sentences with the given word counts. Every sentence is in at
/// least one window; consecutive windows share `window_overlap_sentences`.
pub fn build_windows(words: &[usize], cfg: &EngineConfig) -> Vec<Range<usize>> {
    let n = words.len();
    let mut out: Vec<Range<usize>> = Vec::new();
    if n == 0 {
        return out;
    }
    let mut start = 0;
    loop {
        let mut end = start;
        let mut total = 0;
        while end < n {
            let w = words[end];
            let count = end - start;
            if count > 0 {
                if total + w > cfg.window_max_words {
                    break;
                }
                if total >= cfg.window_min_words && count >= cfg.window_target_sentences {
                    break;
                }
            }
            total += w;
            end += 1;
        }
        if end == n {
            // Last window: reach back so it isn't a scrap of one short sentence.
            while start > 0
                && total < cfg.window_min_words
                && total + words[start - 1] <= cfg.window_max_words
            {
                start -= 1;
                total += words[start];
            }
            if out.last() != Some(&(start..end)) {
                out.push(start..end);
            }
            return out;
        }
        out.push(start..end);
        let len = end - start;
        let next = if len > cfg.window_overlap_sentences { end - cfg.window_overlap_sentences } else { end };
        start = next.max(start + 1);
    }
}

/// Group sentences into paragraphs (consecutive equal `para` ids). Windows
/// never cross a group boundary: a pasted AI paragraph scores so strongly
/// that any window touching it reads as AI, which bled into the human
/// sentences around it. Paragraphs under `paragraph_min_words` (greetings,
/// sign-offs, one-line LinkedIn paragraphs) are too short to judge alone, so
/// they merge into the next paragraph, or the previous one at the end.
pub fn build_groups(words: &[usize], para: &[usize], cfg: &EngineConfig) -> Vec<Range<usize>> {
    debug_assert_eq!(words.len(), para.len());
    let mut raw: Vec<Range<usize>> = Vec::new();
    for i in 0..words.len() {
        match raw.last_mut() {
            Some(g) if para[g.start] == para[i] => g.end = i + 1,
            _ => raw.push(i..i + 1),
        }
    }
    let sum = |r: &Range<usize>| words[r.clone()].iter().sum::<usize>();
    let mut out: Vec<Range<usize>> = Vec::new();
    let mut pending: Option<Range<usize>> = None;
    for g in raw {
        let g = match pending.take() {
            Some(p) => p.start..g.end,
            None => g,
        };
        if sum(&g) < cfg.paragraph_min_words {
            pending = Some(g);
        } else {
            out.push(g);
        }
    }
    if let Some(p) = pending {
        match out.last_mut() {
            Some(last) => last.end = p.end,
            None => out.push(p),
        }
    }
    out
}

/// Windows built inside each group, as indices into the full sentence list.
pub fn build_grouped_windows(words: &[usize], groups: &[Range<usize>], cfg: &EngineConfig) -> Vec<Range<usize>> {
    groups
        .iter()
        .flat_map(|g| build_windows(&words[g.clone()], cfg).into_iter().map(move |w| g.start + w.start..g.start + w.end))
        .collect()
}

/// Each group cut into doc-sized chunks, as indices into the full sentence
/// list: the paragraph-level spans the style detector scores.
pub fn build_group_chunks(words: &[usize], groups: &[Range<usize>], cfg: &EngineConfig) -> Vec<Range<usize>> {
    groups
        .iter()
        .flat_map(|g| build_doc_chunks(&words[g.clone()], cfg).into_iter().map(move |r| g.start + r.start..g.start + r.end))
        .collect()
}

/// Consecutive, non-overlapping chunks of at most `doc_chunk_words` (a
/// single over-size sentence still gets its own chunk).
pub fn build_doc_chunks(words: &[usize], cfg: &EngineConfig) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut total = 0;
    for (i, &w) in words.iter().enumerate() {
        if i > start && total + w > cfg.doc_chunk_words {
            out.push(start..i);
            start = i;
            total = 0;
        }
        total += w;
    }
    if start < words.len() {
        out.push(start..words.len());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> EngineConfig {
        EngineConfig::default()
    }

    fn covers_all(n: usize, wins: &[Range<usize>]) {
        for i in 0..n {
            assert!(wins.iter().any(|w| w.contains(&i)), "sentence {i} not covered: {wins:?}");
        }
        for w in wins {
            assert!(!w.is_empty() && w.end <= n);
        }
    }

    #[test]
    fn empty() {
        assert!(build_windows(&[], &cfg()).is_empty());
        assert!(build_doc_chunks(&[], &cfg()).is_empty());
    }

    #[test]
    fn single_sentence_single_window() {
        assert_eq!(build_windows(&[12], &cfg()), vec![0..1]);
    }

    #[test]
    fn short_text_is_one_window() {
        // 4 sentences, 35 words: never reaches min_words, so it's all one window.
        assert_eq!(build_windows(&[10, 8, 9, 8], &cfg()), vec![0..4]);
    }

    #[test]
    fn regular_prose_three_sentence_windows_with_one_overlap() {
        let w = vec![15; 9]; // 3 sentences = 45 words >= min 40
        let wins = build_windows(&w, &cfg());
        assert_eq!(wins, vec![0..3, 2..5, 4..7, 6..9]);
        covers_all(9, &wins);
    }

    #[test]
    fn short_sentences_pack_to_min_words() {
        let w = vec![5; 20];
        let wins = build_windows(&w, &cfg());
        covers_all(20, &wins);
        // Every window but possibly the tail has >= 40 words; none over 70.
        for r in &wins {
            let t: usize = w[r.clone()].iter().sum();
            assert!(t <= 70);
        }
        for r in &wins[..wins.len() - 1] {
            assert!(w[r.clone()].iter().sum::<usize>() >= 40, "{r:?}");
        }
    }

    #[test]
    fn long_sentences_are_alone_when_needed() {
        // 60-word sentences: two would exceed max 70, so each window is one sentence.
        let w = vec![60, 60, 60];
        let wins = build_windows(&w, &cfg());
        assert_eq!(wins, vec![0..1, 1..2, 2..3]);
    }

    #[test]
    fn tail_reaches_back() {
        // Last sentence is a scrap; the final window grows backwards.
        let w = vec![20, 20, 20, 20, 3];
        let wins = build_windows(&w, &cfg());
        covers_all(5, &wins);
        let last = wins.last().unwrap();
        assert!(w[last.clone()].iter().sum::<usize>() >= 40, "{wins:?}");
    }

    #[test]
    fn zero_overlap_partitions() {
        let mut c = cfg();
        c.window_overlap_sentences = 0;
        let wins = build_windows(&[15; 9], &c);
        assert_eq!(wins, vec![0..3, 3..6, 6..9]);
    }

    #[test]
    fn many_sentences_terminates_and_covers() {
        let w: Vec<usize> = (0..5000).map(|i| 1 + (i * 7) % 30).collect();
        let wins = build_windows(&w, &cfg());
        covers_all(w.len(), &wins);
    }

    #[test]
    fn groups_follow_paragraphs_and_merge_scraps() {
        let c = cfg(); // paragraph_min_words = 10
        // greeting (2w) | body (30w) | AI para (40w) | sign-off (2w)
        let words = [2, 15, 15, 20, 20, 2];
        let para = [0, 1, 1, 2, 2, 3];
        assert_eq!(build_groups(&words, &para, &c), vec![0..3, 3..6]);
        // all one paragraph
        assert_eq!(build_groups(&[5, 5, 5], &[0, 0, 0], &c), vec![0..3]);
        // everything tiny: one group
        assert_eq!(build_groups(&[3, 3, 3], &[0, 1, 2], &c), vec![0..3]);
        assert!(build_groups(&[], &[], &c).is_empty());
    }

    #[test]
    fn grouped_windows_never_cross_groups() {
        let words = vec![15; 8];
        let groups = vec![0..3, 3..8];
        let wins = build_grouped_windows(&words, &groups, &cfg());
        covers_all(8, &wins);
        for w in &wins {
            assert!(groups.iter().any(|g| g.start <= w.start && w.end <= g.end), "{w:?} crosses a group");
        }
    }

    #[test]
    fn doc_chunks_partition_under_limit() {
        let w = vec![50; 13]; // 650 words -> 300 + 300 + 50
        let c = build_doc_chunks(&w, &cfg());
        assert_eq!(c, vec![0..6, 6..12, 12..13]);
        assert_eq!(build_doc_chunks(&[400], &cfg()), vec![0..1]);
    }

    #[test]
    fn group_chunks_stay_inside_groups() {
        // group 0..2 (100 words), group 2..10 (8 x 50 = 400 words -> 300 + 100)
        let w = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50];
        let c = build_group_chunks(&w, &[0..2, 2..10], &cfg());
        assert_eq!(c, vec![0..2, 2..8, 8..10]);
    }
}
