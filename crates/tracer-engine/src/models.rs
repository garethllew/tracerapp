//! The shipped model set: one place for file names and how they combine,
//! shared by the app and the eval CLI.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::analyze::{analyze_with, Analysis};
use crate::config::EngineConfig;
use crate::detector::{DetectorOptions, OnnxDetector};

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

/// desklib plus the rasbt style model. desklib scores the windows and the
/// context; rasbt only scores paragraphs and lifts the ones it's confident
/// about (`analyze_with`, docs/detector-ensemble.md).
pub struct Ensemble {
    pub desklib: OnnxDetector,
    pub style: OnnxDetector,
}

impl Ensemble {
    pub fn analyze(&self, text: &str, cfg: &EngineConfig) -> Result<Analysis> {
        analyze_with(text, &self.desklib, Some(&self.style), cfg)
    }
}

pub fn load_ensemble(dir: &Path, opts: &DetectorOptions) -> Result<Ensemble> {
    Ok(Ensemble {
        desklib: OnnxDetector::load(&dir.join(DESKLIB_MODEL), &dir.join(DESKLIB_TOKENIZER), opts)?,
        style: OnnxDetector::load(&dir.join(STYLE_MODEL), &dir.join(STYLE_TOKENIZER), opts)?,
    })
}

/// Where developer tools (eval CLI, model tests) look for the models:
/// $TRACER_MODEL_DIR, then the app's model folder (%APPDATA%\com.tracer.app\
/// models) when it holds the full set, then tools/model-export/work/onnx.
pub fn dev_model_dir() -> PathBuf {
    if let Ok(d) = std::env::var("TRACER_MODEL_DIR") {
        return d.into();
    }
    if let Ok(appdata) = std::env::var("APPDATA") {
        let d = Path::new(&appdata).join("com.tracer.app").join("models");
        if complete(&d) {
            return d;
        }
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/model-export/work/onnx")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(name: &str) -> String {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval/hard").join(name);
        let raw = std::fs::read_to_string(p).unwrap().replace("\r\n", "\n");
        raw.splitn(3, "---").nth(2).unwrap().trim().to_string()
    }

    /// The cases that motivated the ensemble rule, on the real models:
    /// `cargo test --release -p tracer-engine --lib -- --ignored`.
    #[test]
    #[ignore]
    fn ensemble_catches_gemini_without_accusing_enron() {
        let dir = dev_model_dir();
        assert!(complete(&dir), "model set incomplete in {}", dir.display());
        let e = load_ensemble(&dir, &DetectorOptions::default()).unwrap();
        let cfg = EngineConfig::default();
        let gemini = e.analyze(&body("ai-faceit-gemini.txt"), &cfg).unwrap().ai_share.unwrap();
        assert!(gemini >= 0.5, "gemini {gemini}");
        // desklib alone (no gate) flagged these four.
        for n in ["01", "03", "06", "14"] {
            let s = e.analyze(&body(&format!("human-enron-{n}.txt")), &cfg).unwrap().ai_share.unwrap();
            assert!(s < 0.5, "enron-{n} {s}");
        }
    }
}
