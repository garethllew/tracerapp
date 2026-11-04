//! The only thing Tracer ever persists: three settings, in `settings.json`
//! in the app data dir (tauri-plugin-store, as Vixer's settings.rs). Pasted
//! or clipboard text is never stored.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tauri::Runtime;
use tauri_plugin_store::StoreExt;

const STORE_FILE: &str = "settings.json";
const SETTINGS_KEY: &str = "settings";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Theme {
    System,
    Dark,
    Light,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    pub hotkey: String,
    pub launch_at_startup: bool,
    pub theme: Theme,
}

pub const DEFAULT_HOTKEY: &str = "Ctrl+Shift+H";

impl Default for Settings {
    fn default() -> Self {
        Self { hotkey: DEFAULT_HOTKEY.into(), launch_at_startup: false, theme: Theme::Dark }
    }
}

/// A partial update from the UI; absent fields are left alone.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsPatch {
    pub hotkey: Option<String>,
    pub launch_at_startup: Option<bool>,
    pub theme: Option<Theme>,
}

pub fn load<R: Runtime>(app: &tauri::AppHandle<R>) -> Result<Settings> {
    let store = app.store(STORE_FILE).context("open store")?;
    match store.get(SETTINGS_KEY) {
        Some(val) => serde_json::from_value(val).context("deserialize settings"),
        None => Ok(Settings::default()),
    }
}

pub fn save<R: Runtime>(app: &tauri::AppHandle<R>, settings: &Settings) -> Result<()> {
    let store = app.store(STORE_FILE).context("open store")?;
    store.set(SETTINGS_KEY, serde_json::to_value(settings).context("serialize settings")?);
    store.save().context("save store")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_fill_missing_fields() {
        let s: Settings = serde_json::from_str(r#"{"theme":"light"}"#).unwrap();
        assert_eq!(s.hotkey, DEFAULT_HOTKEY);
        assert!(!s.launch_at_startup);
        assert_eq!(s.theme, Theme::Light);
        assert_eq!(Settings::default().theme, Theme::Dark);
    }
}
