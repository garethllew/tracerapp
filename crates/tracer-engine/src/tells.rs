//! Rule-based AI-writing tells (`data/ai-tells.json`): style hints shown
//! next to the models' verdict, with what matched.
//!
//! User decision (2026-11-04), "hybrid": every tell is shown as a reason, but
//! only near-certain copy-paste artefacts (weight >= `artefact_min_weight`:
//! placeholders like "[Your Name]", chatbot citation / utm_source leftovers,
//! U+2011 hyphens, "As an AI language model") may change the score, and only
//! for the paragraph they're in (analyze.rs). Style tells like em dashes or
//! "crucial" are typed by people too, so they never accuse on their own.
//!
//! The JSON's patterns are ECMAScript. They're converted for the `regex`
//! crate: flags `i`/`m` become inline `(?i)`/`(?m)`, `g`/`u` are dropped,
//! and the one lookahead, `(?=\s|$)`, becomes `(?:\s|$)` (the match then
//! includes the whitespace; snippets are trimmed).

use std::ops::Range;
use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

const RULES_JSON: &str = include_str!("../data/ai-tells.json");
/// Matched snippets kept per tell (the count covers all matches).
const MAX_EXAMPLES: usize = 3;
const MAX_SNIPPET_CHARS: usize = 60;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RulesFile {
    rules: Vec<RuleDef>,
    statistical_signals: Vec<SignalDef>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuleDef {
    id: String,
    category: String,
    label: String,
    description: String,
    pattern: String,
    flags: String,
    weight: u8,
    min_matches: usize,
    enabled: bool,
    example: String,
}

#[derive(Deserialize)]
struct SignalDef {
    id: String,
    label: String,
    description: String,
    weight: u8,
}

pub struct Rule {
    pub id: String,
    pub category: String,
    pub label: String,
    pub description: String,
    pub weight: u8,
    pub min_matches: usize,
    pub example: String,
    re: Regex,
}

struct Signal {
    id: String,
    label: String,
    description: String,
    weight: u8,
}

struct Catalogue {
    rules: Vec<Rule>,
    signals: Vec<Signal>,
}

/// One tell found in a text.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Tell {
    pub id: String,
    /// The JSON category, or "statistics" for a document-level signal.
    pub category: String,
    pub label: String,
    pub description: String,
    /// 1 = weak hint ... 5 = near-certain copy-paste artefact.
    pub weight: u8,
    /// Weight >= `artefact_min_weight`: may raise the score (analyze.rs).
    pub artefact: bool,
    /// Matches in the whole text (statistical signals: 1).
    pub count: usize,
    /// Up to three matched snippets, or the measured value for a signal.
    pub examples: Vec<String>,
    /// Byte ranges of every match in the input (not serialised: the UI
    /// works in UTF-16; the engine uses these to place artefacts).
    #[serde(skip)]
    pub spans: Vec<Range<usize>>,
}

/// ECMAScript pattern + flags -> `regex` crate syntax.
fn to_rust(pattern: &str, flags: &str) -> String {
    let mut inline = String::new();
    for (f, s) in [('i', "i"), ('m', "m")] {
        if flags.contains(f) {
            inline.push_str(s);
        }
    }
    let body = pattern.replace(r"(?=\s|$)", r"(?:\s|$)");
    if inline.is_empty() {
        body
    } else {
        format!("(?{inline}){body}")
    }
}

fn catalogue() -> &'static Catalogue {
    static C: OnceLock<Catalogue> = OnceLock::new();
    C.get_or_init(|| {
        // Embedded data; every pattern is compiled in the tests, so a bad
        // edit fails `cargo test` rather than the app.
        let file: RulesFile = serde_json::from_str(RULES_JSON).expect("data/ai-tells.json is invalid JSON");
        let rules = file
            .rules
            .into_iter()
            .filter(|r| r.enabled)
            .map(|r| Rule {
                re: Regex::new(&to_rust(&r.pattern, &r.flags)).unwrap_or_else(|e| panic!("tell {}: {e}", r.id)),
                id: r.id,
                category: r.category,
                label: r.label,
                description: r.description,
                weight: r.weight,
                min_matches: r.min_matches.max(1),
                example: r.example,
            })
            .collect();
        let signals = file
            .statistical_signals
            .into_iter()
            .map(|s| Signal { id: s.id, label: s.label, description: s.description, weight: s.weight })
            .collect();
        Catalogue { rules, signals }
    })
}

/// The enabled rules (for tests and the eval CLI).
pub fn rules() -> &'static [Rule] {
    &catalogue().rules
}

/// Every rule that fires at least its `min_matches` times in `text`,
/// strongest first.
pub fn find(text: &str, artefact_min_weight: u8) -> Vec<Tell> {
    let mut out: Vec<Tell> = rules()
        .iter()
        .filter_map(|r| {
            let spans: Vec<Range<usize>> = r
                .re
                .find_iter(text)
                .filter(|m| !m.is_empty() && keep(&r.id, m.as_str()))
                .map(|m| m.range())
                .collect();
            if spans.len() < r.min_matches {
                return None;
            }
            let mut examples: Vec<String> = Vec::new();
            for s in &spans {
                let snip = snippet(&text[s.clone()]);
                if !snip.is_empty() && !examples.contains(&snip) {
                    examples.push(snip);
                }
                if examples.len() == MAX_EXAMPLES {
                    break;
                }
            }
            Some(Tell {
                id: r.id.clone(),
                category: r.category.clone(),
                label: r.label.clone(),
                description: r.description.clone(),
                weight: r.weight,
                artefact: r.weight >= artefact_min_weight,
                count: spans.len(),
                examples,
                spans,
            })
        })
        .collect();
    out.sort_by(|a, b| b.weight.cmp(&a.weight).then(b.count.cmp(&a.count)));
    out
}

/// Per-rule precision filters the JSON asks for in its notes.
fn keep(id: &str, m: &str) -> bool {
    match id {
        // "Broad. For precision, require not/isn't in the match." Without it
        // it fired on two Wikipedia leads ("it is a ..., but ...").
        "not-x-its-y" => {
            let l = m.to_lowercase();
            l.split(|c: char| !c.is_alphanumeric()).any(|w| w == "not") || l.contains("n't") || l.contains("n’t")
        }
        _ => true,
    }
}

/// Readable form of a match: invisible characters named, whitespace
/// collapsed, long matches shortened.
fn snippet(m: &str) -> String {
    let named: String = m
        .chars()
        .map(|c| match c {
            '\u{2011}' => "‹U+2011›".to_string(),
            '\u{202F}' => "‹U+202F›".to_string(),
            '\u{00A0}' => "‹NBSP›".to_string(),
            '\u{2009}' | '\u{200A}' => "‹thin space›".to_string(),
            '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}' => "‹zero-width›".to_string(),
            c => c.to_string(),
        })
        .collect();
    let collapsed = named.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() > MAX_SNIPPET_CHARS {
        collapsed.chars().take(MAX_SNIPPET_CHARS - 1).collect::<String>() + "…"
    } else {
        collapsed
    }
}

/// Document-level statistical signals (display only). Implemented: low
/// sentence-length variation, uniform paragraph lengths, high list density.
/// Not implemented: casual markers and contraction rate (they depend on the
/// text being chat, which Tracer can't know), lexical diversity (a weak
/// tiebreaker by the file's own account) and perplexity (needs another model).
pub fn signals(sentence_words: &[usize], paragraph_words: &[usize], text: &str) -> Vec<Tell> {
    let mut out = Vec::new();
    let mut push = |id: &str, detail: String| {
        if let Some(s) = catalogue().signals.iter().find(|s| s.id == id) {
            out.push(Tell {
                id: s.id.clone(),
                category: "statistics".into(),
                label: s.label.clone(),
                description: s.description.clone(),
                weight: s.weight,
                artefact: false,
                count: 1,
                examples: vec![detail],
                spans: Vec::new(),
            });
        }
    };
    // Burstiness: CV of words per sentence < 0.35 over 5+ sentences.
    let sw: Vec<f32> = sentence_words.iter().filter(|&&w| w >= 3).map(|&w| w as f32).collect();
    if sw.len() >= 5 {
        let cv = cv(&sw);
        if cv < 0.35 {
            push("burstiness", format!("sentence-length variation {cv:.2} (humans usually > 0.6)"));
        }
    }
    // Paragraph uniformity: CV of words per paragraph < 0.3 with 3+ paragraphs.
    let pw: Vec<f32> = paragraph_words.iter().filter(|&&w| w > 0).map(|&w| w as f32).collect();
    if pw.len() >= 3 {
        let cv = cv(&pw);
        if cv < 0.3 {
            push("paragraph-uniformity", format!("{} paragraphs, length variation {cv:.2}", pw.len()));
        }
    }
    // List density: > 40% of lines are bullets/headings, under 300 words.
    let words: usize = sentence_words.iter().sum();
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    if words < 300 && lines.len() >= 4 {
        let listy = lines.iter().filter(|l| is_list_line(l)).count();
        if listy * 10 > lines.len() * 4 {
            push("list-density", format!("{listy} of {} lines are bullets or headings", lines.len()));
        }
    }
    out
}

fn is_list_line(l: &str) -> bool {
    let mut chars = l.chars();
    let first = chars.next().unwrap_or(' ');
    if matches!(first, '-' | '*' | '•' | '●' | '▪' | '#') {
        return true;
    }
    let digits: String = l.chars().take_while(|c| c.is_ascii_digit()).collect();
    !digits.is_empty() && matches!(l[digits.len()..].chars().next(), Some('.') | Some(')'))
}

fn cv(xs: &[f32]) -> f32 {
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    if mean == 0.0 {
        return 0.0;
    }
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    var.sqrt() / mean
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_rule_compiles_and_matches_its_own_example() {
        assert!(rules().len() >= 40, "only {} rules loaded", rules().len());
        for r in rules() {
            // min_matches can exceed one example's hits; the pattern must hit at least once.
            assert!(r.re.is_match(&r.example), "rule {} does not match its example {:?}", r.id, r.example);
        }
    }

    #[test]
    fn disabled_rules_are_skipped() {
        assert!(rules().iter().all(|r| r.id != "us-spelling"));
    }

    #[test]
    fn flags_and_lookahead_convert() {
        assert_eq!(to_rust("abc", "g"), "abc");
        assert_eq!(to_rust("^x", "gim"), "(?im)^x");
        assert_eq!(to_rust(r"Why\?(?=\s|$)", "gi"), r"(?i)Why\?(?:\s|$)");
    }

    #[test]
    fn min_matches_must_be_reached() {
        // curly-quotes needs 2: one curly apostrophe is not a tell.
        assert!(!find("it’s fine", 5).iter().any(|t| t.id == "curly-quotes"));
        assert!(find("it’s “fine”", 5).iter().any(|t| t.id == "curly-quotes"));
    }

    #[test]
    fn artefacts_are_flagged_and_sorted_first() {
        let t = find("Thanks for the update — really helpful. Best, [Your Name]", 5);
        assert_eq!(t[0].id, "placeholder");
        assert!(t[0].artefact);
        assert_eq!(t[0].examples, vec!["[Your Name]".to_string()]);
        let dash = t.iter().find(|t| t.id == "em-dash").unwrap();
        assert!(!dash.artefact);
    }

    #[test]
    fn plain_human_text_has_no_artefacts() {
        let t = "hey, can you send me the deck before 3? i'll read it on the train. thanks";
        assert!(find(t, 5).iter().all(|t| !t.artefact));
    }

    #[test]
    fn not_x_its_y_needs_a_negation() {
        assert!(find("It’s not about speed — it’s about trust.", 5).iter().any(|t| t.id == "not-x-its-y"));
        assert!(!find("It is a platform for gamers, but it is also used by studios.", 5).iter().any(|t| t.id == "not-x-its-y"));
    }

    #[test]
    fn invisible_characters_are_named_in_snippets() {
        let t = find("a 15\u{2011}year plan", 5);
        assert_eq!(t[0].id, "nb-hyphen");
        assert_eq!(t[0].examples, vec!["‹U+2011›".to_string()]);
    }

    #[test]
    fn signals_fire_on_uniform_listy_text_only() {
        let text = "- one two three four five six\n- one two three four five six\n- one two three four five six\n- one two three four five six\n- one two three four five six";
        let s = signals(&[6, 6, 6, 6, 6], &[30], text);
        let ids: Vec<&str> = s.iter().map(|t| t.id.as_str()).collect();
        assert!(ids.contains(&"burstiness") && ids.contains(&"list-density"), "{ids:?}");
        let varied = signals(&[3, 25, 6, 40, 4, 18], &[10, 60, 25], "short.\nlonger one here.");
        assert!(varied.is_empty(), "{:?}", varied.iter().map(|t| &t.id).collect::<Vec<_>>());
    }
}
