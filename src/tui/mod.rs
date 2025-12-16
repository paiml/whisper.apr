//! Whisper Pipeline Visualization TUI
//!
//! Interactive terminal dashboard for visualizing ASR pipeline components.
//! Like a scientist in a laboratory, observe audio → mel → encoder → decoder → text.
//!
//! ## EXTREME TDD: Tests written FIRST
//!
//! ## References
//!
//! - Radford et al. (2022): Whisper architecture
//! - Davis & Mermelstein (1980): Mel filterbank fundamentals
//! - Bahdanau et al. (2014): Attention visualization
//! - Vaswani et al. (2017): Transformer architecture
//! - Li et al. (2024): Modern ASR visualization
//!
//! ## Panels
//!
//! 1. Waveform - Raw audio amplitude
//! 2. Mel - 80-bin spectrogram heatmap
//! 3. Encoder - Layer activations
//! 4. Decoder - Token generation
//! 5. Attention - Cross-attention weights
//! 6. Transcription - Final output
//! 7. Metrics - Performance data

// TUI visualization code has some clippy style warnings that are acceptable
// for visualization code that prioritizes readability over micro-optimization
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::needless_raw_string_hashes)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::inefficient_to_string)]
#![allow(clippy::option_map_or_none)]
#![allow(clippy::format_push_string)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::map_unwrap_or)]

mod app;
mod panels;
mod visualization;

pub use app::{WhisperApp, WhisperPanel, WhisperState};
pub use panels::render_whisper_dashboard;
pub use visualization::{
    render_attention_heatmap, render_mel_spectrogram, render_waveform, MelDisplay, WaveformDisplay,
};

#[cfg(test)]
mod tests;
