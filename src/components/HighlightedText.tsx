import { useMemo, useState } from "react";
import type { Analysis } from "../lib/types";
import { describe, sideOf, tintLevel, type Side } from "../lib/score";

/**
 * The pasted text re-rendered from the engine's segments (which tile the
 * input exactly). Each scored sentence gets a translucent tint in its side's
 * bar colour, as strong as the score is decisive.
 */
export function HighlightedText({ analysis, focus, tracing, onClick }: {
  analysis: Analysis;
  focus: Side | null;
  tracing: boolean;
  onClick: () => void;
}) {
  const [tip, setTip] = useState<{ x: number; y: number; p: number } | null>(null);

  // Delay each sentence's trace-in by its position in the text, so the
  // highlights sweep in step with the bar.
  const delays = useMemo(() => {
    const total = analysis.segments.reduce((n, s) => n + s.text.length, 0) || 1;
    let at = 0;
    return analysis.segments.map((s) => {
      const d = (at / total) * 360;
      at += s.text.length;
      return d;
    });
  }, [analysis]);

  return (
    <div
      className={`textview${tracing ? " tracing" : ""}`}
      data-focus={focus ?? undefined}
      onClick={() => {
        // Let people select text without flipping into edit mode.
        if (!window.getSelection()?.toString()) onClick();
      }}
      aria-label="Highlighted text. Click to edit."
    >
      {analysis.segments.map((seg, i) => {
        if (seg.kind === "gap") return seg.text;
        if (seg.kind === "nonProse") return <span key={i} className="np">{seg.text}</span>;
        if (seg.pAi === null) return <span key={i} className="sent unscored">{seg.text}</span>;
        const p = seg.pAi;
        return (
          <span
            key={i}
            className={`sent ${sideOf(p)}`}
            style={{ "--a": tintLevel(p).toFixed(3), "--d": `${delays[i].toFixed(0)}ms` } as React.CSSProperties}
            onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, p })}
            onMouseLeave={() => setTip(null)}
          >
            {seg.text}
          </span>
        );
      })}
      {tip && (
        <span className="tip" style={{ left: tip.x, top: tip.y }}>
          <span className="dot" style={{ background: sideOf(tip.p) === "ai" ? "var(--ai)" : "var(--hu)" }} />
          {describe(tip.p)}
        </span>
      )}
    </div>
  );
}
