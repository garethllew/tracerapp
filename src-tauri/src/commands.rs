use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use tauri::{AppHandle, Runtime, State};
use tracer_engine::{analyze, Analysis, EngineConfig};

use crate::engine::{self, EngineState, EngineStatus, SharedEngine};
use crate::hotkey::{self, HotkeyState};
use crate::settings::{self, Settings, SettingsPatch};

#[tauri::command]
pub fn engine_status(state: State<'_, SharedEngine>) -> EngineStatus {
    match state.read() {
        Ok(g) => g.status(),
        Err(_) => EngineState::Error("engine state lock poisoned".into()).status(),
    }
}

/// Loads the model on demand (engine.rs: load on check, unload after the
/// hold). Never logs the text: only its size and the timing.
#[tauri::command]
pub async fn analyze_text<R: Runtime>(app: AppHandle<R>, text: String, state: State<'_, SharedEngine>) -> Result<Analysis, String> {
    let state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let detector = engine::acquire(&app, &state)?;
        let t = Instant::now();
        let words = text.split_whitespace().count();
        let out = catch_unwind(AssertUnwindSafe(|| analyze(&text, detector.as_ref(), &EngineConfig::default())));
        drop(detector);
        engine::touch(&state); // the hold counts from the end of this check
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

// ── Settings ──────────────────────────────────────────────────────────────────

#[tauri::command]
pub fn get_settings<R: Runtime>(app: AppHandle<R>) -> Settings {
    settings::load(&app).unwrap_or_else(|e| {
        log::warn!("settings unreadable, using defaults: {e:#}");
        Settings::default()
    })
}

/// Apply a partial update. Errors are user-facing strings; on any error nothing
/// is saved and the previous hotkey stays (or is restored) in effect.
#[tauri::command]
pub fn set_settings<R: Runtime>(app: AppHandle<R>, patch: SettingsPatch) -> Result<Settings, String> {
    let cur = get_settings(app.clone());
    let mut next = cur.clone();

    if let Some(h) = patch.hotkey {
        let h = h.trim().to_string();
        hotkey::parse(&h)?;
        if let Err(e) = hotkey::apply(&app, &h) {
            // Put the saved shortcut back (it may have been suspended for capture).
            let _ = hotkey::apply(&app, &cur.hotkey);
            return Err(e);
        }
        next.hotkey = h;
    }
    if let Some(on) = patch.launch_at_startup {
        use tauri_plugin_autostart::ManagerExt;
        let al = app.autolaunch();
        let r = if on { al.enable() } else { al.disable() };
        if let Err(e) = r {
            log::warn!("autostart change failed: {e}");
            return Err("Couldn't change launch at startup".into());
        }
        next.launch_at_startup = on;
    }
    if let Some(t) = patch.theme {
        next.theme = t;
    }
    settings::save(&app, &next).map_err(|e| {
        log::error!("saving settings failed: {e:#}");
        "Couldn't save settings".to_string()
    })?;
    Ok(next)
}

/// Unbind the hotkey while the Settings UI captures a new combo.
#[tauri::command]
pub fn suspend_hotkey<R: Runtime>(app: AppHandle<R>) {
    hotkey::suspend(&app);
}

/// Re-bind the saved hotkey (capture ended without a change).
#[tauri::command]
pub fn resume_hotkey<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    hotkey::apply(&app, &get_settings(app.clone()).hotkey)
}

/// The last hotkey registration failure, if any (startup failures happen
/// before the UI is listening for `hotkey-error`).
#[tauri::command]
pub fn hotkey_error(state: State<'_, HotkeyState>) -> Option<String> {
    state.error()
}
