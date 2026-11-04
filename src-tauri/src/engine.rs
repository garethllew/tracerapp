//! Detector lifecycle: background load, model-dir resolution, shared state.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tracer_engine::{DetectorOptions, OnnxDetector};

pub const MODEL_FILE: &str = "tracer-int8.onnx";
pub const TOKENIZER_FILE: &str = "tokenizer.json";

pub enum EngineState {
    Loading,
    Ready { detector: Arc<OnnxDetector>, model_dir: PathBuf },
    Missing { searched: Vec<String> },
    Error(String),
}

pub type SharedEngine = Arc<RwLock<EngineState>>;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum StatusKind {
    Loading,
    Ready,
    Missing,
    Error,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineStatus {
    pub state: StatusKind,
    pub message: Option<String>,
    pub model_dir: Option<String>,
}

impl EngineState {
    pub fn status(&self) -> EngineStatus {
        match self {
            EngineState::Loading => EngineStatus { state: StatusKind::Loading, message: None, model_dir: None },
            EngineState::Ready { model_dir, .. } => EngineStatus {
                state: StatusKind::Ready,
                message: None,
                model_dir: Some(model_dir.display().to_string()),
            },
            EngineState::Missing { searched } => EngineStatus {
                state: StatusKind::Missing,
                message: Some(format!("Model file {MODEL_FILE} not found. Searched: {}", searched.join("; "))),
                model_dir: None,
            },
            EngineState::Error(e) => EngineStatus { state: StatusKind::Error, message: Some(e.clone()), model_dir: None },
        }
    }
}

/// Candidate model directories in priority order.
fn candidates<R: Runtime>(app: &AppHandle<R>) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(d) = std::env::var_os("TRACER_MODEL_DIR") {
        if !d.is_empty() {
            v.push(PathBuf::from(d));
        }
    }
    if let Ok(d) = app.path().app_data_dir() {
        v.push(d.join("models"));
    }
    if cfg!(debug_assertions) {
        v.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/model-export/work/onnx"));
    }
    v
}

/// Load the detector on a background thread, updating `state` and emitting
/// `engine-status` on every change.
pub fn spawn_load<R: Runtime>(app: AppHandle<R>, state: SharedEngine) {
    std::thread::spawn(move || {
        let dirs = candidates(&app);
        let new_state = match dirs.iter().find(|d| d.join(MODEL_FILE).is_file()) {
            None => {
                let searched: Vec<String> = dirs.iter().map(|d| d.display().to_string()).collect();
                log::warn!("model not found; searched {searched:?}");
                EngineState::Missing { searched }
            }
            Some(dir) => {
                log::info!("loading model from {}", dir.display());
                let t = Instant::now();
                match OnnxDetector::load(&dir.join(MODEL_FILE), &dir.join(TOKENIZER_FILE), &DetectorOptions::default()) {
                    Ok(det) => {
                        log::info!("engine loaded in {} ms", t.elapsed().as_millis());
                        EngineState::Ready { detector: Arc::new(det), model_dir: dir.clone() }
                    }
                    Err(e) => {
                        log::error!("engine load failed: {e:#}");
                        EngineState::Error(format!("{e:#}"))
                    }
                }
            }
        };
        let status = new_state.status();
        if let Ok(mut g) = state.write() {
            *g = new_state;
        }
        let _ = app.emit("engine-status", status);
    });
}
