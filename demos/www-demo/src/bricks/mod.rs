//! Brick Architecture for whisper.apr Demo (PROBAR-SPEC-009)
//!
//! This module implements the canonical bricks for the whisper.apr demo:
//! - TranscriptionBrick: Displays transcription results
//! - WaveformBrick: Shows audio waveform visualization
//! - VuMeterBrick: Shows audio level indicator
//! - StatusBrick: Shows current status (loading, ready, recording)
//! - AudioBrick: Ring buffer audio capture specification
//!
//! All bricks follow the Brick Architecture where tests ARE the interface.
//!
//! # HTML Generation
//!
//! The `html_gen` module generates index.html from brick definitions,
//! ensuring zero hand-written HTML/CSS/JS (per spec requirement).
//!
//! # TUI Rendering
//!
//! The `tui` module provides ratatui-based TUI rendering of the same bricks.

pub mod audio;
pub mod html_gen;
pub mod score;
pub mod status;
pub mod transcription;
pub mod tui;
pub mod vu_meter;
pub mod waveform;

pub use audio::AudioBrick;
pub use html_gen::{generate_index_html, create_whisper_brick_house, HtmlConfig};
pub use score::{ScoreBrick, ScoreCategory};
pub use status::StatusBrick;
pub use transcription::TranscriptionBrick;
pub use tui::{TuiRenderer, render_brick_to_tui};
pub use vu_meter::VuMeterBrick;
pub use waveform::WaveformBrick;
