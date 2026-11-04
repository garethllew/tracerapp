import { useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

/* Vixer's title bar, Windows layout: 34 px, wordmark left with its accent
   letter, minimise/close glyphs right. (Vixer: MixerSurface.tsx.) */
export function TitleBar() {
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
        <WinCtrl glyph="—" label="Minimise" onAct={() => win.minimize()} />
        <WinCtrl glyph="×" label="Close" onAct={() => win.close()} />
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
