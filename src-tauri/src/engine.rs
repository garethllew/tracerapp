//! Detector lifecycle.
//!
//! User decision (2026-11-04): don't load the model at startup. Load it on the
//! first check, keep it for `HOLD` after the most recent check, then unload it
//! (~2 GB freed) and load again on the next check (~2–4 s). "The model" is the
//! pair desklib + rasbt (`tracer_engine::models`, docs/detector-ensemble.md);
//! it counts as installed only when all four files are there. The engine states:
//!
//! ```text
//!   Missing ──download──► Idle ──check──► Loading ──► Ready ──HOLD idle──► Idle
//!                          ▲                 │
//!                          └──── Error ◄─────┘ (a later check retries)
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tracer_engine::models::{self, Ensemble};
use tracer_engine::DetectorOptions;

/// How long the model stays in memory after the last check.
pub const HOLD: Duration = Duration::from_secs(60);
/// How often the reaper looks for an expired hold.
const REAP_EVERY: Duration = Duration::from_secs(5);

pub enum EngineState {
    /// Model files are installed but not in memory.
    Idle { model_dir: PathBuf },
    Loading { model_dir: PathBuf },
    Ready { detector: Arc<Ensemble>, model_dir: PathBuf, last_used: Instant },
    Missing { searched: Vec<String> },
    Error(String),
}

pub type SharedEngine = Arc<RwLock<EngineState>>;

pub fn new_shared() -> SharedEngine {
    Arc::new(RwLock::new(EngineState::Missing { searched: Vec::new() }))
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum StatusKind {
    Idle,
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
        let dir = |d: &Path| Some(d.display().to_string());
        match self {
            EngineState::Idle { model_dir } => EngineStatus { state: StatusKind::Idle, message: None, model_dir: dir(model_dir) },
            EngineState::Loading { model_dir } => EngineStatus { state: StatusKind::Loading, message: None, model_dir: dir(model_dir) },
            EngineState::Ready { model_dir, .. } => EngineStatus { state: StatusKind::Ready, message: None, model_dir: dir(model_dir) },
            EngineState::Missing { searched } => EngineStatus {
                state: StatusKind::Missing,
                message: Some(format!("Model files ({}) not found. Searched: {}", models::ALL_FILES.join(", "), searched.join("; "))),
                model_dir: None,
            },
            EngineState::Error(e) => EngineStatus { state: StatusKind::Error, message: Some(e.clone()), model_dir: None },
        }
    }

    fn model_dir(&self) -> Option<&Path> {
        match self {
            EngineState::Idle { model_dir } | EngineState::Loading { model_dir } | EngineState::Ready { model_dir, .. } => Some(model_dir),
            _ => None,
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
    // Dev builds also look in the export folder; TRACER_NO_DEV_MODEL=1 skips it
    // to exercise the first-run download.
    if cfg!(debug_assertions) && std::env::var_os("TRACER_NO_DEV_MODEL").is_none() {
        v.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/model-export/work/onnx"));
    }
    v
}

fn resolve<R: Runtime>(app: &AppHandle<R>) -> EngineState {
    let dirs = candidates(app);
    match dirs.iter().find(|d| models::complete(d)) {
        Some(d) => EngineState::Idle { model_dir: d.clone() },
        None => EngineState::Missing { searched: dirs.iter().map(|d| d.display().to_string()).collect() },
    }
}

fn set<R: Runtime>(app: &AppHandle<R>, state: &SharedEngine, new: EngineState) {
    let status = new.status();
    if let Ok(mut g) = state.write() {
        *g = new;
    }
    let _ = app.emit("engine-status", status);
}

/// Startup: find the model (without loading it) and start the unload timer.
pub fn init<R: Runtime>(app: &AppHandle<R>, state: &SharedEngine) {
    let s = resolve(app);
    match &s {
        EngineState::Idle { model_dir } => log::info!("model found in {} (loads on first check)", model_dir.display()),
        _ => log::info!("model not installed; showing setup"),
    }
    set(app, state, s);
    spawn_reaper(app.clone(), state.clone());
}

/// Serialises loads: two checks arriving during a reload load once.
static LOAD_LOCK: Mutex<()> = Mutex::new(());

/// The detector for a check, loading it first if needed. Blocking (up to a
/// few seconds), so call it from `spawn_blocking`.
pub fn acquire<R: Runtime>(app: &AppHandle<R>, state: &SharedEngine) -> Result<Arc<Ensemble>, String> {
    if let Some(d) = ready_detector(state) {
        return Ok(d);
    }
    let _guard = LOAD_LOCK.lock().map_err(|_| "model load lock poisoned".to_string())?;
    // Another check may have loaded it while we waited for the lock.
    if let Some(d) = ready_detector(state) {
        return Ok(d);
    }
    let dir = {
        let g = state.read().map_err(|_| "engine state poisoned".to_string())?;
        g.model_dir().map(Path::to_path_buf)
    };
    // Missing/Error: look again (files may have been installed since).
    let dir = match dir {
        Some(d) => d,
        None => match resolve(app) {
            EngineState::Idle { model_dir } => model_dir,
            other => {
                set(app, state, other);
                return Err("The detection model is not installed.".into());
            }
        },
    };
    set(app, state, EngineState::Loading { model_dir: dir.clone() });
    let t = Instant::now();
    match models::load_ensemble(&dir, &DetectorOptions::default()) {
        Ok(det) => {
            log::info!("model loaded in {} ms", t.elapsed().as_millis());
            let det = Arc::new(det);
            set(app, state, EngineState::Ready { detector: det.clone(), model_dir: dir, last_used: Instant::now() });
            Ok(det)
        }
        Err(e) => {
            log::error!("model load failed: {e:#}");
            set(app, state, EngineState::Error(format!("{e:#}")));
            Err(format!("The model failed to load: {e:#}"))
        }
    }
}

fn ready_detector(state: &SharedEngine) -> Option<Arc<Ensemble>> {
    let mut g = state.write().ok()?;
    if let EngineState::Ready { detector, last_used, .. } = &mut *g {
        *last_used = Instant::now();
        return Some(detector.clone());
    }
    None
}

/// Restart the hold after a check finishes (the hold counts from the end of
/// the last check, not its start).
pub fn touch(state: &SharedEngine) {
    if let Ok(mut g) = state.write() {
        if let EngineState::Ready { last_used, .. } = &mut *g {
            *last_used = Instant::now();
        }
    }
}

/// Drop the model from memory now (e.g. before replacing its files).
pub fn unload<R: Runtime>(app: &AppHandle<R>, state: &SharedEngine) {
    let dir = state.read().ok().and_then(|g| g.model_dir().map(Path::to_path_buf));
    if let Some(model_dir) = dir {
        set(app, state, EngineState::Idle { model_dir });
    }
}

/// After a successful download: installed, not loaded.
pub fn model_installed<R: Runtime>(app: &AppHandle<R>, state: &SharedEngine, model_dir: PathBuf) {
    set(app, state, EngineState::Idle { model_dir });
}

pub fn installed_dir(state: &SharedEngine) -> Option<PathBuf> {
    state.read().ok().and_then(|g| g.model_dir().map(Path::to_path_buf))
}

/// Unload once the hold has passed and no check is using the detector.
fn spawn_reaper<R: Runtime>(app: AppHandle<R>, state: SharedEngine) {
    std::thread::Builder::new()
        .name("tracer-model-reaper".into())
        .spawn(move || loop {
            std::thread::sleep(REAP_EVERY);
            // Check and replace under one write lock. Checks clone the detector
            // under the same lock (ready_detector), so strong_count == 1 really
            // means no check is using it.
            let status = {
                let Ok(mut g) = state.write() else { continue };
                let expired = match &*g {
                    EngineState::Ready { detector, last_used, model_dir } if last_used.elapsed() >= HOLD && Arc::strong_count(detector) == 1 => {
                        Some(model_dir.clone())
                    }
                    _ => None,
                };
                match expired {
                    Some(model_dir) => {
                        *g = EngineState::Idle { model_dir }; // drops the detector: memory freed here
                        Some(g.status())
                    }
                    None => None,
                }
            };
            if let Some(status) = status {
                log::info!("model unloaded after {} s without a check", HOLD.as_secs());
                let _ = app.emit("engine-status", status);
            }
        })
        .expect("spawn model reaper");
}
