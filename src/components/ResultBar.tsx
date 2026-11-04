import type { Analysis } from "../lib/types";
import { split, type Side } from "../lib/score";

const CONF_LABEL = { high: "High confidence", medium: "Medium confidence", low: "Low confidence" } as const;

export function ResultBar({ analysis, busy, stale, focus, onFocus, traceKey }: {
  analysis: Analysis | null;
  /** Analysis running for > ~300 ms: show the shimmer. */
  busy: boolean;
  /** The text was edited since this result; dim it until re-checked. */
  stale: boolean;
  focus: Side | null;
  onFocus: (s: Side | null) => void;
  /** Changes on every new result to replay the trace-in. */
  traceKey: number;
}) {
  const scored = analysis && analysis.aiShare !== null && analysis.gate !== "tooShort";
  const s = scored ? split(analysis!.aiShare!) : null;
  const weak = analysis?.gate === "weak";
  const lead: Side | null = s ? (s.ai >= s.human ? "ai" : "human") : null;

  return (
    <section className={`result${weak ? " weak" : ""}`} style={{ opacity: stale ? 0.45 : 1, transition: "opacity .15s" }} aria-live="polite">
      <div className="headline">
        {s && lead ? (
          <>
            <span className={`big ${lead === "ai" ? "ink-ai" : "ink-hu"}`}>
              {lead === "ai" ? s.ai : s.human}<small>%</small>
            </span>
            <span className={`word ${lead === "ai" ? "ink-ai" : "ink-hu"}`}>{lead === "ai" ? "AI" : "human"}</span>
            <span className="sep">·</span>
            <span className="minor">{lead === "ai" ? `${s.human}% human` : `${s.ai}% AI`}</span>
            {analysis!.confidence && (
              <span className={`badge ${analysis!.confidence}`}>{CONF_LABEL[analysis!.confidence]}</span>
            )}
          </>
        ) : (
          <span className="idle">{busy ? "Tracing…" : analysis?.gate === "tooShort" ? "Too short to judge" : "AI · human split"}</span>
        )}
      </div>

      <div
        key={traceKey}
        className={`bar${s ? " tracing" : ""}${busy ? " busy" : ""}`}
        data-focus={focus ?? undefined}
        role="img"
        aria-label={s ? `${s.ai}% AI, ${s.human}% human` : "No result yet"}
      >
        {s && (
          <>
            <div className="fill">
              {s.ai > 0 && (
                <div className="seg ai" style={{ width: `${s.ai}%` }} onMouseEnter={() => onFocus("ai")} onMouseLeave={() => onFocus(null)} />
              )}
              {s.human > 0 && (
                <div className="seg human" style={{ width: `${s.human}%` }} onMouseEnter={() => onFocus("human")} onMouseLeave={() => onFocus(null)} />
              )}
            </div>
            <div className="head" />
          </>
        )}
      </div>

      <div className="legend">
        {/* Same order as the bar: AI segment left, Human right. */}
        <span className="key label" onMouseEnter={() => onFocus("ai")} onMouseLeave={() => onFocus(null)}>
          <span className="dot ai" /> AI
        </span>
        <span className="key label" onMouseEnter={() => onFocus("human")} onMouseLeave={() => onFocus(null)}>
          <span className="dot human" /> Human
        </span>
        <Notice analysis={analysis} />
      </div>
    </section>
  );
}

function Notice({ analysis }: { analysis: Analysis | null }) {
  if (!analysis) return null;
  if (analysis.gate === "tooShort") return <span className="note warn">Paste at least a few sentences.</span>;
  const notes: string[] = [];
  if (analysis.gate === "weak") notes.push("Short text: treat this as a weak signal.");
  if (analysis.nonEnglish) notes.push("This detector is tuned for English.");
  if (analysis.truncated) notes.push(`Only the first ${analysis.scoredWords.toLocaleString()} words were checked.`);
  if (!notes.length) return null;
  return <span className="note warn">{notes.join(" ")}</span>;
}
