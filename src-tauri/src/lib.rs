
use tauri::{Manager, Runtime};

mod commands;
mod download;
mod engine;
mod hotkey;
mod settings;
mod tray;

use engine::SharedEngine;

/// Bring the main window to the front (the window is created hidden).
pub(crate) fn show_main_window<R: Runtime>(app: &tauri::AppHandle<R>) {
    if let Some(w) = app.get_webview_window("main") {
        if !w.is_visible().unwrap_or(true) {
            fit_to_screen(&w);
        }
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_focus();
    }
}

/// The window is tall (2x Vixer's height) and has no native frame. On short
/// screens (1366x768, 1080p at 150%) it would run off the bottom with its
/// resize grip unreachable, so shrink it to the monitor's work area first.
fn fit_to_screen<R: Runtime>(w: &tauri::WebviewWindow<R>) {
    const MARGIN: u32 = 24; // physical px kept free above the taskbar
    let (Ok(Some(mon)), Ok(size)) = (w.current_monitor(), w.outer_size()) else { return };
    let avail = mon.work_area().size.height.saturating_sub(MARGIN * 2);
    if size.height > avail {
        let _ = w.set_size(tauri::PhysicalSize::new(size.width, avail));
        let _ = w.center();
        log::info!("window shortened to fit the screen ({} -> {} px)", size.height, avail);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Not loaded at startup: the first check loads it (engine.rs).
    let engine: SharedEngine = engine::new_shared();
    let setup_engine = engine.clone();

    tauri::Builder::default()
        .manage(engine)
        .manage(hotkey::HotkeyState::default())
        .manage(download::DownloadState::default())
        .plugin(
            tauri_plugin_log::Builder::new()
                .targets([
                    tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::Stdout),
                    tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::LogDir {
                        file_name: Some("tracer".into()),
                    }),
                ])
                // Pasted text is never logged at any level; only sizes and timings.
                .level(log::LevelFilter::Info)
                .max_file_size(2_000_000)
                .rotation_strategy(tauri_plugin_log::RotationStrategy::KeepOne)
                .level_for("tao", log::LevelFilter::Info)
                .build(),
        )
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            Some(vec!["--startup"]),
        ))
        .on_window_event(|window, event| {
            // Close hides to the tray; Quit lives in the tray menu.
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == "main" {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .setup(move |app| {
            // Shown from Rust, as Vixer does; the dark backgroundColor in
            // tauri.conf.json avoids a white flash before the page paints.
            tray::setup(app)?;
            // Launched by autostart: sit in the tray instead of popping up.
            if !std::env::args().any(|a| a == "--startup") {
                show_main_window(app.handle());
            }
            // Never fatal: a taken shortcut is surfaced via `hotkey-error` and Settings.
            let hk = settings::load(app.handle()).unwrap_or_default().hotkey;
            let _ = hotkey::apply(app.handle(), &hk);
            engine::init(app.handle(), &setup_engine);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::engine_status,
            download::model_info,
            download::download_model,
            commands::analyze_text,
            commands::show_main_window,
            commands::get_settings,
            commands::set_settings,
            commands::suspend_hotkey,
            commands::resume_hotkey,
            commands::hotkey_error,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Tracer");
}
