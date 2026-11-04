//! Tracer's detection engine.
//!
//! ```text
//! text ─ segment ─► sentences ─ gate ─► windows ─ Detector::score ─► P(AI) per window
//!                                                     │
//!         sentence P(AI) = mean of its windows ◄──────┘
//!           capped at the doc-chunk score when that reads human (context gate)
//!           raised to the style model's paragraph score when ≥ 0.95 (style lift)
//!         split = word-weighted share of sentences with P(AI) ≥ 0.5
//!         doc score (whole text, chunked) = context gate + sanity check → confidence
//! ```

pub mod analyze;
pub mod config;
pub mod detector;
pub mod models;
pub mod text;
pub mod window;

pub use analyze::{analyze, analyze_with, Analysis, Confidence, Gate, Segment};
pub use config::EngineConfig;
pub use detector::{Detector, DetectorOptions, MaxEnsemble, OnnxDetector};
pub use text::PieceKind;
