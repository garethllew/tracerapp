import { useEffect, useState } from "react";
import { mb, modelInfo, useModelDownload, type ModelInfo as Info } from "../lib/download";

/** Settings → Model: what's installed, and a re-download button. */
export function ModelInfo() {
  const [info, setInfo] = useState<Info | null>(null);
  const { progress, running, error, start } = useModelDownload();

  useEffect(() => { modelInfo().then(setInfo).catch(() => {}); }, [running]);

  const pct = progress && progress.total ? Math.floor((progress.received / progress.total) * 100) : 0;

  return (
    <div className="model-info mono">
      <dl>
        <dt>Model</dt><dd>{info?.name ?? "…"}</dd>
        <dt>Version</dt><dd>{info?.version ?? "…"}</dd>
        <dt>Size</dt><dd>{info ? mb(info.sizeBytes) : "…"}</dd>
        <dt>Status</dt><dd>{info ? (info.installed ? "Installed" : "Not installed") : "…"}</dd>
        {info?.dir && (<><dt>Folder</dt><dd className="path" title={info.dir}>{info.dir}</dd></>)}
      </dl>
      <div className="model-actions">
        <button className="rb xs" disabled={running} onClick={() => start(true)}>
          {running ? (progress?.phase === "verifying" ? "Checking…" : `Downloading ${pct}%`) : "Re-download"}
        </button>
        {error && <span className="err">{error}</span>}
        {!running && progress?.phase === "done" && !error && <span>Model re-downloaded and verified.</span>}
      </div>
    </div>
  );
}
