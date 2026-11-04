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
    /// **Ignored while `context_gate` is on**: the gate needs the doc chunks,
    /// so they're scored at any length.
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

    // ---- combining evidence (docs/detector-ensemble.md)
    /// Context gate: a sentence can only read AI when desklib's score for its
    /// doc chunk (the whole text, or the `doc_chunk_words` passage around it)
    /// is at least this; otherwise the sentence's score is capped at that
    /// chunk score. desklib's ~50-word windows misread plain human prose
    /// (Enron emails, Wikipedia) as AI while its read of the whole text stays
    /// human, so a window alone isn't enough to accuse. 0 = off.
    pub context_gate: f32,
    /// With a second, style detector (rasbt): a sentence is lifted to its
    /// paragraph's style score when that score is at least this, gate or not.
    /// rasbt catches plain / brand-voice AI that desklib reads as human, but
    /// only its confident paragraph-level reads are trustworthy: its windows
    /// fire on human prose, and 0.9 already let a Wikipedia text through.
    pub style_lift_threshold: f32,
    /// The same for the style score of a sentence's doc chunk (the whole
    /// text, or its `doc_chunk_words` passage). rasbt's whole-text read
    /// catches plain AI answers whose paragraphs each read human (a pasted
    /// chatbot answer: 0.97 whole, paragraphs 0.00-0.12). On the eval sets
    /// it is bimodal (AI 0.90-1.00 or <= 0.56; human 0.95-1.00 or <= 0.20),
    /// so 0.8 sits in the gap. Lower = more AI caught, more humans accused.
    pub style_doc_lift_threshold: f32,
    /// AI-writing tells (data/ai-tells.json) at or above this weight are
    /// copy-paste artefacts ("[Your Name]", oaicite / utm_source=chatgpt.com,
    /// U+2011 hyphens, "As an AI language model"): the paragraph they're in
    /// is raised to `artefact_lift_score`. Lighter tells are shown, never
    /// scored (user decision 2026-11-04). 6 = no tell affects the score.
    pub artefact_min_weight: u8,
    pub artefact_lift_score: f32,

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
            context_gate: 0.5,
            style_lift_threshold: 0.95,
            style_doc_lift_threshold: 0.8,
            artefact_min_weight: 5,
            artefact_lift_score: 0.95,
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
