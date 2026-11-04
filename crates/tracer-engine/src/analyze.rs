//! The scoring pipeline: segment -> gate -> windows -> score -> aggregate.

use std::collections::HashMap;
use std::ops::Range;

use anyhow::{bail, Result};
use serde::Serialize;

use crate::config::EngineConfig;
use crate::detector::Detector;
use crate::text::{paragraph_ids, segment, PieceKind};
use crate::window::{build_doc_chunks, build_grouped_windows, build_groups};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    Low,
    Medium,
    High,
}

impl Confidence {
    fn down(self) -> Self {
        match self {
            Confidence::High => Confidence::Medium,
            _ => Confidence::Low,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Gate {
    /// Under `min_words`: not scored. "Too short to judge."
    TooShort,
    /// Under `weak_signal_words`: scored, shown greyed out as a weak signal.
    Weak,
    Normal,
}

/// One displayed run of the original text. Concatenating every segment's
/// `text` reproduces the input exactly.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    pub text: String,
    pub kind: PieceKind,
    pub words: usize,
    /// Prose that was scored: the mean P(AI) of the windows containing it.
    pub p_ai: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Analysis {
    pub gate: Gate,
    /// Prose words in the whole text (URLs, code and emoji don't count).
    pub words: usize,
    /// Prose words that were scored (< `words` only when `truncated`).
    pub scored_words: usize,
    /// Word-weighted share of scored sentences with P(AI) >= threshold.
    /// `None` when too short to score.
    pub ai_share: Option<f32>,
    /// Whole-text P(AI), the sanity check. `None` when too short to score or
    /// longer than `doc_check_max_words`.
    pub doc_score: Option<f32>,
    pub confidence: Option<Confidence>,
    /// The doc-level score and the sentence split point in opposite directions.
    pub disagreement: bool,
    /// The text looks non-English; the model is tuned for English.
    pub non_english: bool,
    /// Text beyond `max_scored_words` was left unscored.
    pub truncated: bool,
    /// Model calls' worth of chunks (windows + doc chunks, deduplicated).
    pub chunks_scored: usize,
    pub segments: Vec<Segment>,
}

pub fn analyze(text: &str, detector: &dyn Detector, cfg: &EngineConfig) -> Result<Analysis> {
    let pieces = segment(text, cfg);
    let prose: Vec<usize> = pieces.iter().enumerate().filter(|(_, p)| p.kind == PieceKind::Prose).map(|(i, _)| i).collect();
    let words: usize = prose.iter().map(|&i| pieces[i].words).sum();

    let non_english = words >= cfg.language_check_min_words && looks_non_english(prose.iter().map(|&i| pieces[i].model_text.as_str()));

    let mut a = Analysis {
        gate: Gate::TooShort,
        words,
        scored_words: 0,
        ai_share: None,
        doc_score: None,
        confidence: None,
        disagreement: false,
        non_english,
        truncated: false,
        chunks_scored: 0,
        segments: Vec::new(),
    };
    let mut p_ai: Vec<Option<f32>> = vec![None; pieces.len()];

    if words >= cfg.min_words {
        a.gate = if words < cfg.weak_signal_words { Gate::Weak } else { Gate::Normal };

        // Score a prefix of at most max_scored_words (always at least one sentence).
        let mut n_scored = 0;
        let mut acc = 0;
        for &i in &prose {
            if n_scored > 0 && acc + pieces[i].words > cfg.max_scored_words {
                break;
            }
            acc += pieces[i].words;
            n_scored += 1;
        }
        let scored = &prose[..n_scored];
        a.truncated = n_scored < prose.len();
        a.scored_words = acc;

        let sw: Vec<usize> = scored.iter().map(|&i| pieces[i].words).collect();
        let para_of = paragraph_ids(text, &pieces);
        let sp: Vec<usize> = scored.iter().map(|&i| para_of[i]).collect();
        let groups = build_groups(&sw, &sp, cfg);
        let windows = build_grouped_windows(&sw, &groups, cfg);
        let doc_chunks = if acc <= cfg.doc_check_max_words { build_doc_chunks(&sw, cfg) } else { Vec::new() };

        // Score every distinct sentence range once (a short text's single
        // window doubles as its whole-document chunk).
        let mut index: HashMap<Range<usize>, usize> = HashMap::new();
        let mut texts: Vec<String> = Vec::new();
        for r in windows.iter().chain(doc_chunks.iter()) {
            index.entry(r.clone()).or_insert_with(|| {
                texts.push(scored[r.clone()].iter().map(|&i| pieces[i].model_text.as_str()).collect::<Vec<_>>().join(" "));
                texts.len() - 1
            });
        }
        let probs = detector.score(&texts)?;
        if probs.len() != texts.len() {
            bail!("detector returned {} scores for {} chunks", probs.len(), texts.len());
        }
        let probs: Vec<f32> = probs.into_iter().map(|p| if p.is_finite() { p.clamp(0.0, 1.0) } else { 0.5 }).collect();
        a.chunks_scored = texts.len();

        // Sentence score = mean of the windows containing it.
        let mut sum = vec![0f32; n_scored];
        let mut cnt = vec![0u32; n_scored];
        for r in &windows {
            let p = probs[index[r]];
            for s in r.clone() {
                sum[s] += p;
                cnt[s] += 1;
            }
        }
        let sent_p: Vec<f32> = sum.iter().zip(&cnt).map(|(s, &c)| if c > 0 { s / c as f32 } else { 0.5 }).collect();

        let total = a.scored_words.max(1) as f32;
        let ai_words: usize = sent_p.iter().zip(&sw).filter(|(p, _)| **p >= cfg.ai_threshold).map(|(_, w)| w).sum();
        let ambiguous_words: usize = sent_p
            .iter()
            .zip(&sw)
            .filter(|(p, _)| **p >= cfg.ambiguous_low && **p <= cfg.ambiguous_high)
            .map(|(_, w)| w)
            .sum();
        let ai_share = ai_words as f32 / total;
        let doc_score = (!doc_chunks.is_empty()).then(|| {
            doc_chunks.iter().map(|r| probs[index[r]] * sw[r.clone()].iter().sum::<usize>() as f32).sum::<f32>() / total
        });

        a.disagreement = doc_score.is_some_and(|d| {
            (d >= cfg.disagreement_doc_score && ai_share < cfg.disagreement_low_share)
                || (d <= 1.0 - cfg.disagreement_doc_score && ai_share > cfg.disagreement_high_share)
        });
        a.confidence = Some(confidence(a.scored_words, ambiguous_words as f32 / total, a.disagreement, non_english, cfg));
        a.ai_share = Some(ai_share);
        a.doc_score = doc_score;
        for (k, &i) in scored.iter().enumerate() {
            p_ai[i] = Some(sent_p[k]);
        }
    }

    a.segments = pieces
        .iter()
        .zip(p_ai)
        .map(|(p, s)| Segment { text: text[p.range.clone()].to_string(), kind: p.kind, words: p.words, p_ai: s })
        .collect();
    Ok(a)
}

fn confidence(words: usize, ambiguous_share: f32, disagreement: bool, non_english: bool, cfg: &EngineConfig) -> Confidence {
    if non_english {
        return Confidence::Low;
    }
    let by_length = if words < cfg.confidence_medium_words {
        Confidence::Low
    } else if words < cfg.confidence_high_words {
        Confidence::Medium
    } else {
        Confidence::High
    };
    let by_decisiveness = if ambiguous_share >= cfg.ambiguous_share_low {
        Confidence::Low
    } else if ambiguous_share >= cfg.ambiguous_share_medium {
        Confidence::Medium
    } else {
        Confidence::High
    };
    let c = by_length.min(by_decisiveness);
    if disagreement {
        c.down()
    } else {
        c
    }
}

fn looks_non_english<'a>(texts: impl Iterator<Item = &'a str>) -> bool {
    let joined = texts.collect::<Vec<_>>().join(" ");
    match whatlang::detect(&joined) {
        Some(info) => info.lang() != whatlang::Lang::Eng && info.is_reliable(),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "leverage" reads as AI, "lol" as human; P(AI) = share of those markers.
    struct Mock;
    impl Detector for Mock {
        fn score(&self, chunks: &[String]) -> Result<Vec<f32>> {
            Ok(chunks
                .iter()
                .map(|c| {
                    let ai = c.matches("leverage").count() as f32;
                    let hu = c.matches("lol").count() as f32;
                    if ai + hu == 0.0 { 0.5 } else { ai / (ai + hu) }
                })
                .collect())
        }
    }

    struct Constant(f32);
    impl Detector for Constant {
        fn score(&self, chunks: &[String]) -> Result<Vec<f32>> {
            Ok(vec![self.0; chunks.len()])
        }
    }

    struct WrongLength;
    impl Detector for WrongLength {
        fn score(&self, _: &[String]) -> Result<Vec<f32>> {
            Ok(vec![0.5])
        }
    }

    fn cfg() -> EngineConfig {
        EngineConfig::default()
    }

    const HUMAN: &str = "lol ok so the build broke again and nobody knows why tbh";
    const AI: &str = "We will leverage synergies to drive impactful outcomes across the organisation";

    fn para(s: &str, n: usize) -> String {
        vec![format!("{s}.") ; n].join(" ")
    }

    fn rebuilt(a: &Analysis) -> String {
        a.segments.iter().map(|s| s.text.as_str()).collect()
    }

    #[test]
    fn too_short_is_not_scored() {
        let a = analyze("lol yeah ok see you at 3", &Constant(0.9), &cfg()).unwrap();
        assert_eq!(a.gate, Gate::TooShort);
        assert_eq!(a.ai_share, None);
        assert_eq!(a.confidence, None);
        assert!(a.segments.iter().all(|s| s.p_ai.is_none()));
        assert_eq!(a.chunks_scored, 0);
    }

    #[test]
    fn gate_boundaries() {
        let w = |n: usize| (0..n).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" ") + ".";
        assert_eq!(analyze(&w(14), &Constant(0.9), &cfg()).unwrap().gate, Gate::TooShort);
        assert_eq!(analyze(&w(15), &Constant(0.9), &cfg()).unwrap().gate, Gate::Weak);
        assert_eq!(analyze(&w(49), &Constant(0.9), &cfg()).unwrap().gate, Gate::Weak);
        assert_eq!(analyze(&w(50), &Constant(0.9), &cfg()).unwrap().gate, Gate::Normal);
    }

    #[test]
    fn urls_and_emoji_dont_count_towards_the_gate() {
        let t = "https://a.com https://b.com 🎉🎉🎉 ok lol";
        assert_eq!(analyze(t, &Constant(0.9), &cfg()).unwrap().gate, Gate::TooShort);
    }

    #[test]
    fn all_ai_and_all_human() {
        let ai = analyze(&para(AI, 12), &Mock, &cfg()).unwrap();
        assert_eq!(ai.ai_share, Some(1.0));
        let hu = analyze(&para(HUMAN, 14), &Mock, &cfg()).unwrap(); // 168 words: High is possible
        assert_eq!(hu.ai_share, Some(0.0));
        assert_eq!(hu.doc_score, Some(0.0));
        assert_eq!(hu.confidence, Some(Confidence::High));
        assert!(!hu.disagreement);
    }

    #[test]
    fn mixed_split_is_word_weighted_and_localised() {
        // 6 human sentences, then 6 AI sentences, then 6 human.
        let t = format!("{}\n\n{}\n\n{}", para(HUMAN, 6), para(AI, 6), para(HUMAN, 6));
        let a = analyze(&t, &Mock, &cfg()).unwrap();
        assert_eq!(rebuilt(&a), t);
        let share = a.ai_share.unwrap();
        assert!((share - 0.33).abs() < 0.12, "share {share}");
        // First and last sentences are clearly human, the middle clearly AI.
        let ps: Vec<f32> = a.segments.iter().filter_map(|s| s.p_ai).collect();
        assert_eq!(ps.len(), 18);
        assert!(ps[0] < 0.2 && ps[17] < 0.2);
        assert!(ps[8] > 0.8 && ps[9] > 0.8);
    }

    #[test]
    fn sentence_score_is_mean_of_its_windows() {
        // 9 x 15-word sentences -> windows 0..3, 2..5, 4..7, 6..9.
        let s = |tag: &str| format!("{tag} {}.", vec!["filler"; 14].join(" "));
        let t: Vec<String> = (0..9).map(|i| if i < 3 { s("leverage") } else { s("lol") }).collect();
        let a = analyze(&t.join(" "), &Mock, &cfg()).unwrap();
        let ps: Vec<f32> = a.segments.iter().filter_map(|s| s.p_ai).collect();
        // window 0..3 = 1.0, window 2..5 = 1/3; sentence 2 is in both.
        assert!((ps[0] - 1.0).abs() < 1e-6);
        assert!((ps[2] - (1.0 + 1.0 / 3.0) / 2.0).abs() < 1e-6, "{ps:?}");
        assert!((ps[3] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn short_text_scores_one_chunk_not_two() {
        let a = analyze(&para(AI, 2), &Mock, &cfg()).unwrap();
        assert_eq!(a.chunks_scored, 1, "single window should double as the doc chunk");
    }

    #[test]
    fn doc_check_only_up_to_its_word_limit() {
        let short = analyze(&para(AI, 90), &Mock, &cfg()).unwrap(); // 990 words
        assert!(short.doc_score.is_some());
        let long = analyze(&para(AI, 92), &Mock, &cfg()).unwrap(); // 1012 words
        assert!(long.doc_score.is_none());
        assert!(!long.disagreement);
        let mut off = cfg();
        off.doc_check_max_words = 0;
        assert!(analyze(&para(AI, 20), &Mock, &off).unwrap().doc_score.is_none());
    }

    #[test]
    fn confidence_caps_by_length() {
        let c = cfg();
        assert_eq!(confidence(30, 0.0, false, false, &c), Confidence::Low);
        assert_eq!(confidence(100, 0.0, false, false, &c), Confidence::Medium);
        assert_eq!(confidence(200, 0.0, false, false, &c), Confidence::High);
    }

    #[test]
    fn confidence_drops_when_undecided_or_disagreeing() {
        let c = cfg();
        assert_eq!(confidence(200, 0.3, false, false, &c), Confidence::Medium);
        assert_eq!(confidence(200, 0.6, false, false, &c), Confidence::Low);
        assert_eq!(confidence(200, 0.0, true, false, &c), Confidence::Medium);
        assert_eq!(confidence(100, 0.0, true, false, &c), Confidence::Low);
        assert_eq!(confidence(500, 0.0, false, true, &c), Confidence::Low);
    }

    #[test]
    fn undecided_scores_lower_confidence_end_to_end() {
        let a = analyze(&para(HUMAN, 20), &Constant(0.5), &cfg()).unwrap();
        assert_eq!(a.confidence, Some(Confidence::Low));
    }

    #[test]
    fn disagreement_detected() {
        // A detector that says AI for the whole doc (long chunk) but human
        // for every window can't happen with Constant; build it directly.
        struct ByLength;
        impl Detector for ByLength {
            fn score(&self, chunks: &[String]) -> Result<Vec<f32>> {
                Ok(chunks.iter().map(|c| if c.split_whitespace().count() > 100 { 0.95 } else { 0.05 }).collect())
            }
        }
        let a = analyze(&para(HUMAN, 20), &ByLength, &cfg()).unwrap();
        assert!(a.disagreement);
        assert_eq!(a.ai_share, Some(0.0));
        assert_eq!(a.confidence, Some(Confidence::Medium)); // High, down one
    }

    #[test]
    fn mismatched_detector_output_is_an_error_not_a_panic() {
        let r = analyze(&para(HUMAN, 20), &WrongLength, &cfg());
        assert!(r.is_err());
    }

    #[test]
    fn nan_scores_are_neutralised() {
        let a = analyze(&para(HUMAN, 5), &Constant(f32::NAN), &cfg()).unwrap();
        assert!(a.ai_share.unwrap().is_finite());
    }

    #[test]
    fn truncates_beyond_max_scored_words() {
        let mut c = cfg();
        c.max_scored_words = 100;
        let a = analyze(&para(HUMAN, 30), &Mock, &c).unwrap();
        assert!(a.truncated);
        assert!(a.scored_words <= 100);
        assert!(a.segments.iter().any(|s| s.kind == PieceKind::Prose && s.p_ai.is_none()));
    }

    // ---- weird input: none of these may panic, and segments must tile.

    fn survives(t: &str) -> Analysis {
        let a = analyze(t, &Mock, &cfg()).unwrap();
        assert_eq!(rebuilt(&a), t);
        a
    }

    #[test]
    fn weird_inputs() {
        assert_eq!(survives("").gate, Gate::TooShort);
        assert_eq!(survives("   \n\n  ").gate, Gate::TooShort);
        assert_eq!(survives("🎉🔥😂👍🏽🇬🇧").gate, Gate::TooShort);
        assert_eq!(survives("https://example.com/some/long/path?with=query&and=more").gate, Gate::TooShort);
        survives("```\ncode only\n```");
        survives("....!!!???,,,");
        survives("\u{200b}\u{feff}\u{0}\u{1f}");
        survives("a\r\nb\rc\n");
        survives(&"x".repeat(50_000));
        survives(&"word ".repeat(10_000));
        survives(&"Sentence here. ".repeat(4_000)); // 50k+ chars
        survives(&"🎉 ".repeat(20_000));
    }

    #[test]
    fn fifty_k_chars_mixed() {
        let chunk = format!("{} {}\n- bullet lol\n```\nx=1\n```\nhttps://x.com?a=b ", para(HUMAN, 3), para(AI, 2));
        let t = chunk.repeat(50_000 / chunk.len() + 1);
        let a = survives(&t);
        assert!(a.truncated, "50k chars is over max_scored_words");
    }

    #[test]
    fn other_languages() {
        let fr = "Salut tout le monde, je voulais juste vous dire que la réunion de demain est déplacée à quinze heures parce que la salle est prise le matin. Merci de prévenir les autres et à demain.";
        let a = survives(fr);
        assert!(a.non_english);
        assert_eq!(a.confidence, Some(Confidence::Low));
        let en = "Hi all, just a heads up that tomorrow's meeting has moved to three because the room is booked in the morning. Please let the others know and see you then.";
        assert!(!survives(en).non_english);
        survives("日本語の文です。次の文。これはテストです。もう一つの文があります。");
        survives("Привет всем! Встреча завтра в три часа. Пожалуйста, передайте остальным.");
        survives("مرحبا بالجميع، الاجتماع غدا في الساعة الثالثة.");
    }
}
