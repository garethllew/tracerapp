// Shared scoring → visual mapping, so the bar and the highlights read as one system.

/** Same cut-off the engine uses to class a sentence (EngineConfig::ai_threshold). */
export const AI_THRESHOLD = 0.5;

export type Side = "ai" | "human";

export function sideOf(pAi: number): Side {
  return pAi >= AI_THRESHOLD ? "ai" : "human";
}

/**
 * How decisive a score is: 0 at P = 0.5, 1 at P = 0 or 1. A sentence at 0.95
 * gets a strong tint, 0.6 a faint one, ~0.5 almost none.
 */
export function strength(pAi: number): number {
  return Math.min(1, Math.abs(pAi - 0.5) * 2);
}

/** Tint alpha as a fraction of --tint-max. Slight ease so 0.6–0.7 stays faint. */
export function tintLevel(pAi: number): number {
  return Math.pow(strength(pAi), 1.25);
}

/** "87% likely AI" / "92% likely human" for hover. */
export function describe(pAi: number): string {
  const side = sideOf(pAi);
  // Never claim certainty: it's a rough gauge, not proof.
  const pct = Math.min(99, Math.round((side === "ai" ? pAi : 1 - pAi) * 100));
  return `${pct}% likely ${side === "ai" ? "AI" : "human"}`;
}

/** Whole-number split that always sums to 100. */
export function split(aiShare: number): { ai: number; human: number } {
  const ai = Math.round(Math.min(1, Math.max(0, aiShare)) * 100);
  return { ai, human: 100 - ai };
}

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
