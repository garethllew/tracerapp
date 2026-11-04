import { invoke } from "@tauri-apps/api/core";

export type Theme = "system" | "dark" | "light";

/** The only things Tracer persists. Mirrors src-tauri/src/settings.rs. */
export interface Settings {
  hotkey: string;
  launchAtStartup: boolean;
  theme: Theme;
}

export const DEFAULT_SETTINGS: Settings = { hotkey: "Ctrl+Shift+H", launchAtStartup: false, theme: "dark" };

export const getSettings = () => invoke<Settings>("get_settings");

/** Rejects with a message the UI can show, e.g. "That shortcut is taken by another app". */
export const setSettings = (patch: Partial<Settings>) => invoke<Settings>("set_settings", { patch });

/** Unbind the hotkey while a new combo is captured, rebind if capture is abandoned. */
export const suspendHotkey = () => invoke<void>("suspend_hotkey");
export const resumeHotkey = () => invoke<void>("resume_hotkey");

/** Last registration failure (a startup failure happens before the UI listens). */
export const hotkeyError = () => invoke<string | null>("hotkey_error");

/**
 * Dark is the base (no attribute); light sets data-theme="light". "system"
 * follows prefers-color-scheme live. Returns a cleanup function.
 */
export function applyTheme(theme: Theme): () => void {
  const root = document.documentElement;
  const set = (light: boolean) => {
    if (light) root.setAttribute("data-theme", "light");
    else root.removeAttribute("data-theme");
  };
  if (theme === "system") {
    const mq = window.matchMedia("(prefers-color-scheme: light)");
    set(mq.matches);
    const h = (e: MediaQueryListEvent) => set(e.matches);
    mq.addEventListener("change", h);
    return () => mq.removeEventListener("change", h);
  }
  set(theme === "light");
  return () => {};
}
