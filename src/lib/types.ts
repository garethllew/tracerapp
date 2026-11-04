// Mirrors the serde output of tracer-engine (crates/tracer-engine/src/analyze.rs, text.rs).

/** Rust `Gate` (camelCase). */
export type Gate = "tooShort" | "weak" | "normal";

/** Rust `Confidence` (lowercase). */
export type Confidence = "low" | "medium" | "high";

/** Rust `PieceKind` (camelCase). */
export type PieceKind = "prose" | "nonProse" | "gap";

export interface Segment {
  text: string;
  kind: PieceKind;
  words: number;
  /** Mean P(AI) of the windows containing this prose; null when unscored. */
  pAi: number | null;
}

/** Rust `tells::Tell`: an AI-writing tell found in the text (style hint). */
export interface Tell {
  id: string;
  /** JSON category, or "statistics" for a document-level signal. */
  category: string;
  label: string;
  description: string;
  /** 1 = weak hint ... 5 = near-certain copy-paste artefact. */
  weight: number;
  /** Only artefacts raise the score (their paragraph reads AI). */
  artefact: boolean;
  count: number;
  /** Up to three matched snippets, or the measured value for a signal. */
  examples: string[];
}

export interface Analysis {
  gate: Gate;
  words: number;
  scoredWords: number;
  aiShare: number | null;
  docScore: number | null;
  confidence: Confidence | null;
  disagreement: boolean;
  nonEnglish: boolean;
  truncated: boolean;
  chunksScored: number;
  segments: Segment[];
  /** Strongest first. Shown as reasons; only artefacts affect the score. */
  tells: Tell[];
}

/** idle = installed but not in memory (loads on the next check; engine.rs). */
export type EngineState = "idle" | "loading" | "ready" | "missing" | "error";

export interface EngineStatus {
  state: EngineState;
  message: string | null;
  modelDir: string | null;
}
