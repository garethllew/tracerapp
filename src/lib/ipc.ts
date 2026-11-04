import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { Analysis, EngineStatus } from "./types";

export const engineStatus = () => invoke<EngineStatus>("engine_status");

export const analyzeText = (text: string) => invoke<Analysis>("analyze_text", { text });

export const showMainWindow = () => invoke<void>("show_main_window");

export const onEngineStatus = (cb: (status: EngineStatus) => void): Promise<UnlistenFn> =>
  listen<EngineStatus>("engine-status", (e) => cb(e.payload));
