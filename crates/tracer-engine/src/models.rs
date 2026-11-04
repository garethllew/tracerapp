//! The shipped model set: one place for file names and how they combine,
//! shared by the app and the eval CLI.

use std::path::Path;

use anyhow::Result;

use crate::detector::{DetectorOptions, MaxEnsemble, OnnxDetector};

/// desklib/ai-text-detector-v1.01, DeBERTa-v3-large, INT8 (model-v1 release).
pub const DESKLIB_MODEL: &str = "tracer-int8.onnx";
pub const DESKLIB_TOKENIZER: &str = "tokenizer.json";
/// rasbt/ai-text-detector-modernbert, ModernBERT-base, bf16-stored weights
/// (model-v2 release). Catches plain / brand-voice AI text desklib misses.
pub const STYLE_MODEL: &str = "rasbt-bf16w.onnx";
pub const STYLE_TOKENIZER: &str = "rasbt-tokenizer.json";

pub const ALL_FILES: [&str; 4] = [DESKLIB_TOKENIZER, DESKLIB_MODEL, STYLE_TOKENIZER, STYLE_MODEL];

/// True when every file of the model set is in `dir`.
pub fn complete(dir: &Path) -> bool {
    ALL_FILES.iter().all(|f| dir.join(f).is_file())
}

/// Load both detectors and combine them (max per chunk).
pub fn load_ensemble(dir: &Path, opts: &DetectorOptions) -> Result<MaxEnsemble> {
    let desklib = OnnxDetector::load(&dir.join(DESKLIB_MODEL), &dir.join(DESKLIB_TOKENIZER), opts)?;
    let style = OnnxDetector::load(&dir.join(STYLE_MODEL), &dir.join(STYLE_TOKENIZER), opts)?;
    Ok(MaxEnsemble::new(vec![Box::new(desklib), Box::new(style)]))
}
