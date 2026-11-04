//! The model behind a trait so it can be swapped or ensembled later.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::{Tokenizer, TruncationParams};

/// Scores text chunks. Implementations must be cheap to call repeatedly
/// (load once, keep warm) and safe to share across threads.
pub trait Detector: Send + Sync {
    /// P(AI) in 0..=1 for each chunk, same order and length as `chunks`.
    /// (The brief sketched `-> Vec<f32>`; a `Result` lets ONNX/tokeniser
    /// failures surface as errors instead of panics.)
    fn score(&self, chunks: &[String]) -> Result<Vec<f32>>;
}

#[derive(Debug, Clone)]
pub struct DetectorOptions {
    /// ONNX Runtime intra-op threads. None = half the logical CPUs, capped at 8
    /// (more threads than physical cores was no faster in the Phase 0 bench).
    pub intra_threads: Option<usize>,
    /// Max chunks per inference call. **Keep at 1 for the INT8 model**: its
    /// dynamic quantisation picks activation scales from the whole batch
    /// tensor, so a chunk's score shifted (0.655 -> 0.674) depending on what
    /// else was batched with it. Unbatched, every chunk scores exactly as it
    /// does alone (and as in the Python reference), and on CPU batching was
    /// worth < 5% anyway. If batching is used (e.g. with the FP32 model),
    /// chunks are sorted by length and a batch stops growing once padding it
    /// would exceed `max_batch_tokens`.
    pub batch_size: usize,
    pub max_batch_tokens: usize,
    /// Model context limit; longer chunks are truncated.
    pub max_tokens: usize,
}

impl Default for DetectorOptions {
    fn default() -> Self {
        Self { intra_threads: None, batch_size: 1, max_batch_tokens: 512, max_tokens: 512 }
    }
}

pub fn default_threads() -> usize {
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    (logical / 2).clamp(1, 8)
}

/// desklib/ai-text-detector-v1.01 exported to ONNX (see tools/model-export):
/// inputs `input_ids`/`attention_mask` (int64 [batch, seq]), output `logits`
/// [batch, 1]; P(AI) = sigmoid(logit). Pooling + head are in the graph.
pub struct OnnxDetector {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    /// The tokenizer's own pad id (DeBERTa: 0, ModernBERT: 50283). Only used
    /// when batching pads shorter chunks.
    pad_id: i64,
    batch_size: usize,
    max_batch_tokens: usize,
    // Sessions keep a handle to their environment; we hold it too so its
    // lifetime is obvious.
    _env: Environment,
}

impl OnnxDetector {
    pub fn load(model: &Path, tokenizer: &Path, opts: &DetectorOptions) -> Result<Self> {
        let mut tok = Tokenizer::from_file(tokenizer)
            .map_err(|e| anyhow!("loading tokenizer {}: {e}", tokenizer.display()))?;
        tok.with_truncation(Some(TruncationParams { max_length: opts.max_tokens, ..Default::default() }))
            .map_err(|e| anyhow!("tokenizer truncation: {e}"))?;
        tok.with_padding(None);
        let pad_id = ["[PAD]", "<pad>", "<|padding|>"].iter().find_map(|t| tok.token_to_id(t)).unwrap_or(0) as i64;

        let env = environment()?;
        let threads = opts.intra_threads.unwrap_or_else(default_threads);
        let session = Session::builder(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("{e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow!("{e}"))?
            .commit_from_file(model)
            .with_context(|| format!("loading model {}", model.display()))?;
        Ok(Self {
            session: Mutex::new(session),
            tokenizer: tok,
            pad_id,
            batch_size: opts.batch_size.max(1),
            max_batch_tokens: opts.max_batch_tokens,
            _env: env,
        })
    }
}

impl Detector for OnnxDetector {
    fn score(&self, chunks: &[String]) -> Result<Vec<f32>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        let ids: Vec<Vec<i64>> = chunks
            .iter()
            .map(|c| {
                self.tokenizer
                    .encode(c.as_str(), true)
                    .map(|e| e.get_ids().iter().map(|&i| i as i64).collect())
                    .map_err(|e| anyhow!("tokenising: {e}"))
            })
            .collect::<Result<_>>()?;

        let mut order: Vec<usize> = (0..chunks.len()).collect();
        order.sort_by_key(|&i| ids[i].len());

        let lens: Vec<usize> = order.iter().map(|&i| ids[i].len()).collect();
        let mut out = vec![f32::NAN; chunks.len()];
        let mut session = self.session.lock().map_err(|_| anyhow!("model session poisoned"))?;
        for span in batch_spans(&lens, self.batch_size, self.max_batch_tokens) {
            let batch = &order[span];
            let seq = batch.iter().map(|&i| ids[i].len()).max().unwrap_or(0).max(1);
            let b = batch.len();
            let mut input_ids = vec![self.pad_id; b * seq];
            let mut mask = vec![0i64; b * seq];
            for (row, &i) in batch.iter().enumerate() {
                let toks = &ids[i];
                input_ids[row * seq..row * seq + toks.len()].copy_from_slice(toks);
                mask[row * seq..row * seq + toks.len()].fill(1);
            }
            let outputs = session.run(ort::inputs![
                "input_ids" => Tensor::from_array(([b, seq], input_ids))?,
                "attention_mask" => Tensor::from_array(([b, seq], mask))?,
            ])?;
            let (_, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
            if logits.len() != b {
                bail!("model returned {} logits for a batch of {b}", logits.len());
            }
            for (row, &i) in batch.iter().enumerate() {
                out[i] = sigmoid(logits[row]);
            }
        }
        Ok(out)
    }
}

/// Several detectors as one: a chunk's P(AI) is the MAX over members.
///
/// **Baseline only, not shipped** (eval `--model max`). Max of desklib and
/// rasbt looked right on whole texts, but at Tracer's ~50-word windows rasbt
/// fires on human prose: Wikipedia leads flagged 12/15 vs desklib's 9/15.
/// The shipped combination is `analyze_with` (context gate + style lift),
/// see docs/detector-ensemble.md.
pub struct MaxEnsemble {
    members: Vec<Box<dyn Detector>>,
}

impl MaxEnsemble {
    pub fn new(members: Vec<Box<dyn Detector>>) -> Self {
        Self { members }
    }
}

impl Detector for MaxEnsemble {
    fn score(&self, chunks: &[String]) -> Result<Vec<f32>> {
        let mut out = vec![0f32; chunks.len()];
        for m in &self.members {
            let s = m.score(chunks)?;
            if s.len() != chunks.len() {
                bail!("ensemble member returned {} scores for {} chunks", s.len(), chunks.len());
            }
            for (o, v) in out.iter_mut().zip(s) {
                // NaN never wins the max; analyze() neutralises an all-NaN result.
                if v > *o || o.is_nan() {
                    *o = v;
                }
            }
        }
        Ok(out)
    }
}

/// The process-wide ONNX Runtime environment. ort allows exactly one per
/// process and remembers it even after it's dropped, so a second `build()`
/// fails forever. The app unloads and reloads the model (engine.rs), so the
/// environment is created once and kept for the process lifetime. It's
/// small; dropping the detector still frees the model's memory.
fn environment() -> Result<Environment> {
    static ENV: Mutex<Option<Environment>> = Mutex::new(None);
    let mut slot = ENV.lock().map_err(|_| anyhow!("ONNX Runtime environment lock poisoned"))?;
    if let Some(env) = slot.as_ref() {
        return Ok(env.clone());
    }
    let env = ort::init().with_name("tracer").build().context("creating ONNX Runtime environment")?;
    *slot = Some(env.clone());
    Ok(env)
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Split ascending token lengths into batches: at most `max_items` each, and
/// `items * longest <= max_tokens` (a single over-budget chunk still runs alone).
fn batch_spans(sorted_lens: &[usize], max_items: usize, max_tokens: usize) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    for i in 0..sorted_lens.len() {
        let items = i - start + 1;
        if i > start && (items > max_items || items * sorted_lens[i] > max_tokens) {
            out.push(start..i);
            start = i;
        }
    }
    if start < sorted_lens.len() {
        out.push(start..sorted_lens.len());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The app unloads and reloads the model; ort only allows one environment
    /// per process, so a second load must reuse it rather than fail.
    #[test]
    #[ignore]
    fn load_drop_load_again() {
        let dir = crate::models::dev_model_dir();
        let (m, t) = (dir.join("tracer-int8.onnx"), dir.join("tokenizer.json"));
        let a = OnnxDetector::load(&m, &t, &DetectorOptions::default()).unwrap();
        let p1 = a.score(&["hello there, see you at three".to_string()]).unwrap()[0];
        drop(a);
        let b = OnnxDetector::load(&m, &t, &DetectorOptions::default()).expect("second load after drop");
        let p2 = b.score(&["hello there, see you at three".to_string()]).unwrap()[0];
        assert_eq!(p1, p2);
    }

    /// A chunk must score the same whatever it's scored alongside.
    /// Needs the exported model: `cargo test --release -- --ignored`.
    #[test]
    #[ignore]
    fn scores_dont_depend_on_neighbouring_chunks() {
        let dir = crate::models::dev_model_dir();
        let det = OnnxDetector::load(&dir.join("tracer-int8.onnx"), &dir.join("tokenizer.json"), &DetectorOptions::default()).unwrap();
        let chunks: Vec<String> = [
            "lol ok see you at 3",
            "I am writing to provide an overview of the upcoming workshop, which aims to foster collaboration across teams and drive alignment on our strategic objectives for the coming quarter.",
            "ok found it. it was the --no-cache flag, staging build was reusing some ancient layer from tuesday",
            "Here's the thing: public speaking isn't just about delivering a message, it's about building trust.",
        ]
        .map(String::from)
        .to_vec();
        let batched = det.score(&chunks).unwrap();
        for (c, b) in chunks.iter().zip(&batched) {
            let single = det.score(std::slice::from_ref(c)).unwrap()[0];
            assert_eq!(single, *b, "neighbours changed the score for {c:?}");
        }
    }

    struct Fixed(Vec<f32>);
    impl Detector for Fixed {
        fn score(&self, chunks: &[String]) -> Result<Vec<f32>> {
            Ok(self.0.iter().copied().cycle().take(chunks.len()).collect())
        }
    }

    #[test]
    fn ensemble_takes_the_max_per_chunk() {
        let e = MaxEnsemble::new(vec![Box::new(Fixed(vec![0.9, 0.1, 0.2])), Box::new(Fixed(vec![0.05, 0.8, 0.3]))]);
        let c: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
        assert_eq!(e.score(&c).unwrap(), vec![0.9, 0.8, 0.3]);
        assert!(MaxEnsemble::new(vec![Box::new(Fixed(vec![0.5]))]).score(&[]).unwrap().is_empty());
    }

    #[test]
    fn ensemble_rejects_a_short_member() {
        struct Short;
        impl Detector for Short {
            fn score(&self, _: &[String]) -> Result<Vec<f32>> {
                Ok(vec![0.5])
            }
        }
        let e = MaxEnsemble::new(vec![Box::new(Fixed(vec![0.1])), Box::new(Short)]);
        assert!(e.score(&["a".into(), "b".into()]).is_err());
    }

    #[test]
    fn batches_respect_count_and_token_budget() {
        assert!(batch_spans(&[], 8, 1024).is_empty());
        assert_eq!(batch_spans(&[70; 10], 8, 1024), vec![0..8, 8..10]);
        // 70-token windows then two long doc chunks: the long ones run separately.
        assert_eq!(batch_spans(&[70, 70, 80, 300, 400], 8, 1024), vec![0..3, 3..5 /* 2*400=800 */]);
        assert_eq!(batch_spans(&[100, 600, 700], 8, 1024), vec![0..1, 1..2, 2..3]);
        assert_eq!(batch_spans(&[2000], 8, 1024), vec![0..1]);
    }
}
