import type { Tell } from "../lib/types";

/**
 * AI-writing tells found in the text (engine tells.rs). Style hints, not
 * proof: only copy-paste artefacts (weight 5) count towards the score, and
 * they say so. Hover a row for what the tell means.
 */
export function StyleHints({ tells }: { tells: Tell[] }) {
  if (!tells.length) return null;
  return (
    <section className="hints" aria-label="Style hints">
      <p className="label hints-head">
        Style hints <span className="hints-sub">· not proof{tells.some((t) => t.artefact) ? "" : " · these don't change the score"}</span>
      </p>
      <ul>
        {tells.map((t) => (
          <li key={t.id} className={t.artefact ? "artefact" : undefined} title={t.description}>
            <span className="strength" aria-label={`strength ${t.weight} of 5`}>
              {[1, 2, 3, 4, 5].map((i) => <i key={i} className={i <= t.weight ? "on" : undefined} />)}
            </span>
            <span className="hint-label">{t.label}</span>
            {t.count > 1 && <span className="hint-count">×{t.count}</span>}
            {t.artefact && <span className="hint-tag">copy-paste artefact · counts as AI</span>}
            {t.examples.length > 0 && (
              <span className="hint-ex">
                {t.category === "statistics"
                  ? <span className="hint-detail">{t.examples[0]}</span>
                  : t.examples.map((e, i) => <code key={i}>{e}</code>)}
              </span>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
