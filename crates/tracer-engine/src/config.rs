//! Every tunable threshold in one place.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineConfig {
    // ---- length gate (counted in prose words: URLs, code and emoji don't count)
    /// Below this: don't score at all ("Too short to judge").
    pub min_words: usize,
    /// Below this: score, but flag as a weak signal.
    pub weak_signal_words: usize,

    // ---- segmentation
    /// Sentences longer than this are cut into chunks (run-on Slack text,
    /// pasted walls of text) so highlights stay meaningful and windows stay
    /// well under the model's 512-token limit.
    pub max_sentence_words: usize,
    /// Chunk size used when cutting an over-long sentence.
    pub long_sentence_chunk_words: usize,

    // ---- windows
    /// Windows stay inside paragraphs (blank-line separated). Paragraphs
    /// shorter than this merge into a neighbour first.
    pub paragraph_min_words: usize,
    /// A window keeps adding sentences until it has at least this many words...
    pub window_min_words: usize,
    /// ...and this many sentences, without going over `window_max_words`.
    pub window_target_sentences: usize,
    pub window_max_words: usize,
    /// Sentences shared between consecutive windows.
    pub window_overlap_sentences: usize,

    // ---- document-level sanity check
    /// Score the whole text as well as the windows, but only when it has at
    /// most this many prose words (0 = never). The doc pass runs long
    /// sequences, so it roughly doubles the time on long texts; the user
    /// chose to accept that up to 1,000 words (2026-11-04). Up to ~70 words
    /// it's free: the single window doubles as the doc chunk.
    pub doc_check_max_words: usize,
    /// The whole text is scored in chunks of up to this many words (~1.4
    /// tokens per word keeps a chunk inside 512 tokens).
    pub doc_chunk_words: usize,
    /// Doc score at/above this while almost nothing was classed AI (or
    /// at/below `1 - this` while most was) counts as disagreement.
    pub disagreement_doc_score: f32,
    /// "Almost nothing" / "most" for the disagreement test.
    pub disagreement_low_share: f32,
    pub disagreement_high_share: f32,

    // ---- classification + confidence
    /// A sentence counts as AI when its P(AI) is at or above this.
    pub ai_threshold: f32,
    /// Scores inside this band are "undecided" for confidence. Wide on
    /// purpose: a Gemini brand-voice text scored 0.22-0.32 throughout and was
    /// reported "High confidence human" while the 0.35-0.65 band treated
    /// 0.3 as decisive. Only clear scores should earn High.
    pub ambiguous_low: f32,
    pub ambiguous_high: f32,
    /// Share of words in undecided sentences that drops confidence one /
    /// two levels.
    pub ambiguous_share_medium: f32,
    pub ambiguous_share_low: f32,
    /// Word counts that cap confidence: < medium_words => Low,
    /// < high_words => at most Medium.
    pub confidence_medium_words: usize,
    pub confidence_high_words: usize,

    // ---- limits
    /// Prose beyond this many words is shown but not scored (keeps a pasted
    /// novel from tying up the CPU for minutes).
    pub max_scored_words: usize,
    /// Minimum prose words before the language check is trusted.
    pub language_check_min_words: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            min_words: 15,
            weak_signal_words: 50,
            max_sentence_words: 60,
            long_sentence_chunk_words: 45,
            paragraph_min_words: 10,
            window_min_words: 40,
            window_target_sentences: 3,
            window_max_words: 70,
            window_overlap_sentences: 1,
            doc_check_max_words: 1000,
            doc_chunk_words: 300,
            disagreement_doc_score: 0.8,
            disagreement_low_share: 0.1,
            disagreement_high_share: 0.5,
            ai_threshold: 0.5,
            ambiguous_low: 0.15,
            ambiguous_high: 0.85,
            ambiguous_share_medium: 0.25,
            ambiguous_share_low: 0.5,
            confidence_medium_words: 50,
            confidence_high_words: 150,
            max_scored_words: 5000,
            language_check_min_words: 8,
        }
    }
}
