//! System tray (Vixer's setup_tray, extended with the Tracer actions).

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Emitter, Manager, Runtime};

pub fn setup<R: Runtime>(app: &tauri::App<R>) -> tauri::Result<()> {
    let open = MenuItem::with_id(app, "open", "Open", true, None::<&str>)?;
    let check = MenuItem::with_id(app, "check", "Check clipboard", true, None::<&str>)?;
    let settings = MenuItem::with_id(app, "settings", "Settings", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let sep = PredefinedMenuItem::separator(app)?;
    let menu = Menu::with_items(app, &[&open, &check, &settings, &sep, &quit])?;

    // Placeholder icon until the real tray artwork exists.
    let icon = tauri::image::Image::from_bytes(include_bytes!("../icons/32x32.png"))?;

    TrayIconBuilder::new()
        .icon(icon)
        .menu(&menu)
        .show_menu_on_left_click(false) // left-click toggles the window; right-click opens the menu
        .tooltip("Tracer")
        .on_menu_event(|app, event| match event.id.as_ref() {
            "open" => crate::show_main_window(app),
            "check" => crate::hotkey::check_clipboard(app),
            "settings" => {
                crate::show_main_window(app);
                let _ = app.emit("open-settings", ());
            }
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                let app = tray.app_handle();
                if let Some(w) = app.get_webview_window("main") {
                    let showing = w.is_visible().unwrap_or(false) && !w.is_minimized().unwrap_or(false);
                    if showing {
                        let _ = w.hide();
                        return;
                    }
                }
                crate::show_main_window(app);
            }
        })
        .build(app)?;
    Ok(())
}
