use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use tauri::{AppHandle, Runtime, State};
use tracer_engine::{analyze, Analysis, EngineConfig};

use crate::engine::{EngineState, EngineStatus, SharedEngine};

#[tauri::command]
pub fn engine_status(state: State<'_, SharedEngine>) -> EngineStatus {
    match state.read() {
        Ok(g) => g.status(),
        Err(_) => EngineState::Error("engine state lock poisoned".into()).status(),
    }
}

/// Never logs the text: only its size and the timing.
#[tauri::command]
pub async fn analyze_text(text: String, state: State<'_, SharedEngine>) -> Result<Analysis, String> {
    let detector = {
        let g = state.read().map_err(|_| "engine state lock poisoned".to_string())?;
        match &*g {
            EngineState::Ready { detector, .. } => detector.clone(),
            EngineState::Loading => return Err("The engine is still loading.".into()),
            EngineState::Missing { .. } => return Err("The detection model is not installed.".into()),
            EngineState::Error(e) => return Err(format!("The engine failed to load: {e}")),
        }
    };

    tauri::async_runtime::spawn_blocking(move || {
        let t = Instant::now();
        let words = text.split_whitespace().count();
        let out = catch_unwind(AssertUnwindSafe(|| analyze(&text, detector.as_ref(), &EngineConfig::default())));
        match out {
            Ok(Ok(a)) => {
                log::info!("analysed {words} words in {} ms", t.elapsed().as_millis());
                Ok(a)
            }
            // Log only that it failed, never the error detail (could echo input).
            Ok(Err(e)) => {
                log::error!("analysis failed after {} ms", t.elapsed().as_millis());
                Err(format!("Analysis failed: {e:#}"))
            }
            Err(_) => {
                log::error!("analysis panicked after {} ms", t.elapsed().as_millis());
                Err("Analysis failed unexpectedly.".to_string())
            }
        }
    })
    .await
    .map_err(|e| format!("Analysis task failed: {e}"))?
}

#[tauri::command]
pub fn show_main_window<R: Runtime>(app: AppHandle<R>) {
    crate::show_main_window(&app);
}
