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
}

export type EngineState = "loading" | "ready" | "missing" | "error";

export interface EngineStatus {
  state: EngineState;
  message: string | null;
  modelDir: string | null;
}
