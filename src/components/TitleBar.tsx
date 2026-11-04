import { useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

/* Vixer's title bar, Windows layout: 34 px, wordmark left with its accent
   letter, minimise/close glyphs right. (Vixer: MixerSurface.tsx.) */
export function TitleBar({ settingsOpen, onSettings }: { settingsOpen: boolean; onSettings: () => void }) {
  const win = getCurrentWindow();
  return (
    <div
      onMouseDown={(e) => { if (e.button === 0) win.startDragging().catch(() => {}); }}
      style={{
        display: "flex", alignItems: "center", padding: "0 6px 0 13px", height: 34,
        background: "var(--tb-bg)", borderBottom: "1px solid var(--bdr)", flexShrink: 0, cursor: "grab",
      }}
    >
      <div style={{
        flex: 1, fontSize: 10.5, fontWeight: 800, letterSpacing: "0.24em", textTransform: "uppercase",
        color: "var(--t1)", fontFamily: "var(--font-display)", pointerEvents: "none",
      }}>
        Tr<span style={{ color: "var(--ai)" }}>a</span>cer
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 2 }} onMouseDown={(e) => e.stopPropagation()}>
        {/* Settings cog (Vixer's tab-cog: Feather-style gear, rotates on hover) */}
        <button
          className={`tab-cog${settingsOpen ? " on" : ""}`}
          aria-label="Settings"
          title="Settings"
          onClick={(e) => { e.stopPropagation(); onSettings(); }}
          style={{ width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", borderRadius: 4, border: "none", background: "transparent" }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </button>
        <WinCtrl glyph="—" label="Minimise" onAct={() => win.minimize()} />
        <WinCtrl glyph="×" label="Close to tray" onAct={() => win.close()} />
      </div>
    </div>
  );
}

function WinCtrl({ glyph, label, onAct }: { glyph: string; label: string; onAct: () => void }) {
  const [hov, setHov] = useState(false);
  const isX = glyph === "×";
  return (
    <button
      aria-label={label}
      onClick={(e) => { e.stopPropagation(); onAct(); }}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center",
        cursor: "pointer", borderRadius: 4, border: "none",
        color: hov ? (isX ? "#ff6060" : "var(--t2)") : "var(--t3)",
        background: hov ? (isX ? "rgba(255,60,60,.12)" : "rgba(127,127,127,.12)") : "transparent",
        fontSize: 15, lineHeight: 1, fontFamily: "var(--font-mono)",
        transition: "color .12s, background .12s",
      }}
    >
      {glyph}
    </button>
  );
}
