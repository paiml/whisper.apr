//! Brick Architecture for whisper.apr Demo (PROBAR-SPEC-009)
//!
//! This module implements the canonical bricks for the whisper.apr demo:
//! - `TranscriptionBrick`: Displays transcription results
//! - `WaveformBrick`: Shows audio waveform visualization
//! - `VuMeterBrick`: Shows audio level indicator
//! - `AudioLevelBrick`: Audio level with dB readout (-60 to 0 dB)
//! - `StatusBrick`: Shows current status (loading, ready, recording)
//! - `AudioBrick`: Ring buffer audio capture specification
//! - `ChunkProgressBrick`: Real-time chunk processing display
//! - `TimingStatsBrick`: RTF and latency statistics
//! - `PerformanceStatsBrick`: System performance metrics
//! - `PartialResultsBrick`: Streaming partial transcription display
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
pub mod audio_level;
pub mod chunk_progress;
pub mod codegen;
pub mod compute;
pub mod file_info;
pub mod html_gen;
pub mod partial_results;
pub mod performance_stats;
pub mod progress;
pub mod score;
pub mod status;
pub mod timing_stats;
pub mod transcription;
pub mod tui;
pub mod tui_bricks;
pub mod vu_meter;
pub mod waveform;

pub use audio::AudioBrick;
pub use audio_level::{amplitude_to_db, db_to_amplitude, AudioLevelBrick, AudioLevelState};
pub use chunk_progress::{ChunkProgressBrick, ChunkProgressState, ChunkState, ChunkStats};
pub use codegen::{
    create_whisper_audio_brick, create_whisper_event_brick, create_whisper_worker_brick,
    generate_audioworklet_js_from_brick, generate_worker_js_from_brick,
};
pub use compute::{
    create_attention_score_brick, create_layer_norm_brick, create_mel_filterbank_brick,
    create_softmax_brick,
};
pub use file_info::{AudioFormat, FileInfo, FileInfoBrick};
pub use html_gen::{create_whisper_brick_house, generate_index_html, HtmlConfig};
pub use partial_results::{PartialResultsBrick, PartialResultsState, TranscriptSegment};
pub use performance_stats::{
    HealthStatus, MemoryStats, MemoryUnit, ModelLoadState, PerformanceStatsBrick,
    PerformanceStatsState,
};
pub use progress::{ProgressBrick, ProgressStage, ProgressState};
pub use score::{ScoreBrick, ScoreCategory};
pub use status::StatusBrick;
pub use timing_stats::{TimingMeasurement, TimingStatsBrick, TimingStatsState};
pub use transcription::TranscriptionBrick;
pub use tui::{render_brick_to_tui, TuiRenderer};
pub use tui_bricks::{
    AnalyzerBrick, AudioMetrics, CielabColor, CollectorBrick, CollectorError, MetricsState,
    PanelState, PanelType, Rect, RingBuffer, RtfAnalysis,
};
pub use vu_meter::VuMeterBrick;
pub use waveform::WaveformBrick;
