import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useCallback, useEffect, useState } from "react";

export interface ModelInfo {
  name: string;
  version: string;
  sizeBytes: number;
  installed: boolean;
  dir: string | null;
}

export interface DownloadProgress {
  phase: "downloading" | "verifying" | "done";
  received: number;
  total: number;
}

export const modelInfo = () => invoke<ModelInfo>("model_info");
export const downloadModel = (force = false) => invoke<void>("download_model", { force });

export function mb(bytes: number): string {
  return `${Math.round(bytes / 1_000_000).toLocaleString()} MB`;
}

/** Download state shared by the setup screen and Settings → Model. */
export function useModelDownload() {
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const un = listen<DownloadProgress>("download-progress", (e) => setProgress(e.payload));
    return () => { un.then((f) => f()); };
  }, []);

  const start = useCallback(async (force = false) => {
    setError(null);
    setRunning(true);
    try {
      await downloadModel(force);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }, []);

  return { progress, running, error, start };
}
