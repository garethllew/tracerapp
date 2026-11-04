//! Global hotkey + "check clipboard" path, shared by the hotkey and the tray.
//! The clipboard is read here in Rust (the webview needs no clipboard
//! permission). Its contents are never logged: sizes only.

use std::str::FromStr;
use std::sync::Mutex;

use tauri::{AppHandle, Emitter, Manager, Runtime};
use tauri_plugin_clipboard_manager::ClipboardExt;
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

/// Registration state: the combo currently bound (if any) and the last failure.
#[derive(Default)]
pub struct HotkeyState {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    registered: Option<String>,
    error: Option<String>,
}

impl HotkeyState {
    pub fn error(&self) -> Option<String> {
        self.inner.lock().ok().and_then(|g| g.error.clone())
    }
}

pub fn parse(combo: &str) -> Result<Shortcut, String> {
    Shortcut::from_str(combo.trim()).map_err(|_| "That isn't a valid shortcut".to_string())
}

/// Read the clipboard, bring the window up and hand the text to the UI.
pub fn check_clipboard<R: Runtime>(app: &AppHandle<R>) {
    let text = app.clipboard().read_text().unwrap_or_default();
    log::info!("clipboard check requested ({} bytes)", text.len());
    crate::show_main_window(app);
    let _ = app.emit("check-text", text);
}

/// Bind `combo` as the global shortcut, replacing whatever was bound before.
/// The old binding is only dropped once the new one is in place.
pub fn apply<R: Runtime>(app: &AppHandle<R>, combo: &str) -> Result<(), String> {
    let state = app.state::<HotkeyState>();
    let mut g = state.inner.lock().map_err(|_| "hotkey state poisoned".to_string())?;
    let result = bind(app, &mut g, combo);
    g.error = result.clone().err();
    match &result {
        Err(e) => {
            log::warn!("hotkey registration failed: {e}");
            let _ = app.emit("hotkey-error", Some(e.clone()));
        }
        Ok(()) => {
            let _ = app.emit("hotkey-error", Option::<String>::None);
        }
    }
    result
}

fn bind<R: Runtime>(app: &AppHandle<R>, g: &mut Inner, combo: &str) -> Result<(), String> {
    let shortcut = parse(combo)?;
    let gs = app.global_shortcut();
    if g.registered.as_deref().is_some_and(|c| c.eq_ignore_ascii_case(combo.trim())) && gs.is_registered(shortcut) {
        return Ok(());
    }
    gs.on_shortcut(shortcut, |app, _s, event| {
        if event.state == ShortcutState::Pressed {
            check_clipboard(app);
        }
    })
    .map_err(|_| "That shortcut is taken by another app".to_string())?;
    if let Some(old) = g.registered.take() {
        if let Ok(old_sc) = parse(&old) {
            if old_sc != shortcut {
                let _ = gs.unregister(old_sc);
            }
        }
    }
    g.registered = Some(combo.trim().to_string());
    log::info!("global shortcut registered: {}", combo.trim());
    Ok(())
}

/// Drop the binding while the Settings UI captures a new combo (a registered
/// combo is consumed by the OS and would trigger instead of reaching the field).
pub fn suspend<R: Runtime>(app: &AppHandle<R>) {
    let state = app.state::<HotkeyState>();
    let old = state.inner.lock().ok().and_then(|mut g| g.registered.take());
    if let Some(sc) = old.and_then(|o| parse(&o).ok()) {
        let _ = app.global_shortcut().unregister(sc);
    }
}
