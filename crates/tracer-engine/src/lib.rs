//! Tracer's detection engine.
//!
//! ```text
//! text ─ segment ─► sentences ─ gate ─► windows ─ Detector::score ─► P(AI) per window
//!                                                     │
//!         sentence P(AI) = mean of its windows ◄──────┘
//!         split = word-weighted share of sentences with P(AI) ≥ 0.5
//!         doc score (whole text, chunked) = sanity check → confidence
//! ```

pub mod analyze;
pub mod config;
pub mod detector;
pub mod models;
pub mod text;
pub mod window;

pub use analyze::{analyze, Analysis, Confidence, Gate, Segment};
pub use config::EngineConfig;
pub use detector::{Detector, DetectorOptions, MaxEnsemble, OnnxDetector};
pub use text::PieceKind;
