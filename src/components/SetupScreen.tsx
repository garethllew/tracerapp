import { useEffect, useState } from "react";
import { mb, modelInfo, useModelDownload, type ModelInfo } from "../lib/download";

/**
 * First run: the model isn't bundled in the installer, so fetch it once.
 * After this Tracer never touches the network again.
 */
export function SetupScreen() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const { progress, running, error, start } = useModelDownload();

  useEffect(() => { modelInfo().then(setInfo).catch(() => {}); }, []);

  const total = progress?.total ?? info?.sizeBytes ?? 0;
  const received = progress?.received ?? 0;
  const pct = total ? Math.min(100, (received / total) * 100) : 0;
  const verifying = running && progress?.phase === "verifying";

  return (
    <div className="setup">
      <p className="label">One-time setup</p>
      <h1>Download the detection model</h1>
      <p className="body">
        Tracer checks text with two local AI models ({info ? mb(info.sizeBytes) : "about 1.2 GB"}). They download
        once from GitHub, are checked against fingerprints, and from then on Tracer works fully offline.
        Your text never leaves this computer.
      </p>

      <div className={`bar setup-bar${running ? "" : " idle"}`} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct)}>
        <div className="fill">
          <div className="seg ai" style={{ width: `${pct}%`, transition: "width .25s linear" }} />
        </div>
      </div>
      <div className="setup-meta mono">
        {running ? (
          <span>{verifying ? "Checking files…" : `${mb(received)} of ${mb(total)}`}</span>
        ) : error ? (
          <span className="err">{error}</span>
        ) : progress?.phase === "done" ? (
          <span>Done. Loading…</span>
        ) : (
          <span>{received > 0 ? `${mb(received)} downloaded so far` : "Not downloaded yet"}</span>
        )}
        <span>{running && !verifying ? `${Math.floor(pct)}%` : ""}</span>
      </div>

      <div className="setup-actions">
        <button className="rb sm a" onClick={() => start(false)} disabled={running}>
          {running ? "Downloading…" : error || received > 0 ? "Retry download" : "Download model"}
        </button>
      </div>

      <p className="fine mono">
        {info?.name ?? "desklib + rasbt detectors"} · {info?.version ?? "model-v1 + model-v2"} · MIT and Apache-2.0 licences.
        If the download stops, Retry picks up where it left off.
      </p>
    </div>
  );
}
