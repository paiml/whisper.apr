//! Web Worker support for non-blocking inference
//!
//! Provides message-based communication between main thread and worker thread
//! to prevent UI freezes during transcription.
//!
//! # Architecture
//!
//! ```text
//! Main Thread                    Worker Thread
//! ┌──────────────┐              ┌──────────────┐
//! │  JavaScript  │   postMessage │  Whisper.apr │
//! │  Application │ ────────────> │    Worker    │
//! │              │ <──────────── │              │
//! │              │   onmessage   │              │
//! └──────────────┘              └──────────────┘
//! ```
//!
//! # Message Protocol
//!
//! **Request Messages** (Main → Worker):
//! - `{ type: "init", modelType: "tiny" | "base" }`
//! - `{ type: "transcribe", audio: Float32Array, options: {...} }`
//! - `{ type: "detectLanguage", audio: Float32Array }`
//! - `{ type: "abort" }`
//!
//! **Response Messages** (Worker → Main):
//! - `{ type: "ready", modelType: "...", memoryMb: ... }`
//! - `{ type: "result", text: "...", language: "...", segments: [...] }`
//! - `{ type: "language", language: "...", confidence: ... }`
//! - `{ type: "progress", phase: "...", percent: ... }`
//! - `{ type: "error", message: "..." }`

use wasm_bindgen::prelude::*;

/// Worker message types for type-safe communication
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerMessageType {
    /// Initialize the model
    Init,
    /// Transcribe audio
    Transcribe,
    /// Detect language
    DetectLanguage,
    /// Abort current operation
    Abort,
    /// Model is ready
    Ready,
    /// Transcription result
    Result,
    /// Language detection result
    Language,
    /// Progress update
    Progress,
    /// Error occurred
    Error,
}

impl WorkerMessageType {
    /// Convert from string (for JavaScript interop)
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "init" => Some(Self::Init),
            "transcribe" => Some(Self::Transcribe),
            "detectLanguage" => Some(Self::DetectLanguage),
            "abort" => Some(Self::Abort),
            "ready" => Some(Self::Ready),
            "result" => Some(Self::Result),
            "language" => Some(Self::Language),
            "progress" => Some(Self::Progress),
            "error" => Some(Self::Error),
            _ => None,
        }
    }

    /// Convert to string (for JavaScript interop)
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Init => "init",
            Self::Transcribe => "transcribe",
            Self::DetectLanguage => "detectLanguage",
            Self::Abort => "abort",
            Self::Ready => "ready",
            Self::Result => "result",
            Self::Language => "language",
            Self::Progress => "progress",
            Self::Error => "error",
        }
    }
}

/// Worker state for tracking operations
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkerState {
    /// Not initialized
    #[default]
    Uninitialized,
    /// Model is loading
    Loading,
    /// Ready for operations
    Ready,
    /// Currently transcribing
    Transcribing,
    /// Currently detecting language
    DetectingLanguage,
    /// Operation aborted
    Aborted,
    /// Error state
    Error,
}

impl WorkerState {
    /// Check if worker is ready for operations
    #[must_use]
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Check if worker is busy
    #[must_use]
    pub fn is_busy(&self) -> bool {
        matches!(
            self,
            Self::Transcribing | Self::DetectingLanguage | Self::Loading
        )
    }

    /// Get state name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Uninitialized => "uninitialized",
            Self::Loading => "loading",
            Self::Ready => "ready",
            Self::Transcribing => "transcribing",
            Self::DetectingLanguage => "detectingLanguage",
            Self::Aborted => "aborted",
            Self::Error => "error",
        }
    }
}

/// Check if worker state is ready (JS helper)
#[wasm_bindgen(js_name = workerStateIsReady)]
pub fn worker_state_is_ready(state: WorkerState) -> bool {
    state.is_ready()
}

/// Check if worker state is busy (JS helper)
#[wasm_bindgen(js_name = workerStateIsBusy)]
pub fn worker_state_is_busy(state: WorkerState) -> bool {
    state.is_busy()
}

/// Get worker state name (JS helper)
#[wasm_bindgen(js_name = workerStateName)]
pub fn worker_state_name(state: WorkerState) -> String {
    state.name().to_string()
}

/// Progress phase for tracking transcription stages
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressPhase {
    /// Loading model weights
    LoadingModel,
    /// Resampling audio
    Resampling,
    /// Computing mel spectrogram
    MelSpectrogram,
    /// Running encoder
    Encoding,
    /// Running decoder
    Decoding,
    /// Processing results
    Postprocessing,
    /// Completed
    Complete,
}

impl ProgressPhase {
    /// Get phase name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::LoadingModel => "Loading model",
            Self::Resampling => "Resampling audio",
            Self::MelSpectrogram => "Computing mel spectrogram",
            Self::Encoding => "Running encoder",
            Self::Decoding => "Running decoder",
            Self::Postprocessing => "Processing results",
            Self::Complete => "Complete",
        }
    }

    /// Get typical duration weight (for progress estimation)
    #[must_use]
    pub const fn weight(&self) -> u32 {
        match self {
            Self::LoadingModel | Self::MelSpectrogram => 10,
            Self::Resampling | Self::Postprocessing => 5,
            Self::Encoding => 30,
            Self::Decoding => 40,
            Self::Complete => 0,
        }
    }
}

/// Worker progress tracker
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WorkerProgress {
    phase: ProgressPhase,
    phase_progress: f32,
    overall_progress: f32,
    message: String,
}

#[wasm_bindgen]
impl WorkerProgress {
    /// Create new progress tracker
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            phase: ProgressPhase::LoadingModel,
            phase_progress: 0.0,
            overall_progress: 0.0,
            message: String::new(),
        }
    }

    /// Get current phase
    #[wasm_bindgen(getter)]
    pub fn phase(&self) -> ProgressPhase {
        self.phase
    }

    /// Get phase name
    #[wasm_bindgen(js_name = phaseName)]
    pub fn phase_name(&self) -> String {
        self.phase.name().to_string()
    }

    /// Get progress within current phase (0-100)
    #[wasm_bindgen(getter, js_name = phaseProgress)]
    pub fn phase_progress(&self) -> f32 {
        self.phase_progress
    }

    /// Get overall progress (0-100)
    #[wasm_bindgen(getter, js_name = overallProgress)]
    pub fn overall_progress(&self) -> f32 {
        self.overall_progress
    }

    /// Get progress message
    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }

    /// Update progress
    pub fn update(&mut self, phase: ProgressPhase, phase_progress: f32, message: &str) {
        self.phase = phase;
        self.phase_progress = phase_progress.clamp(0.0, 100.0);
        self.message = message.to_string();
        self.calculate_overall();
    }

    /// Move to next phase
    pub fn next_phase(&mut self, phase: ProgressPhase) {
        self.phase = phase;
        self.phase_progress = 0.0;
        self.message = phase.name().to_string();
        self.calculate_overall();
    }

    /// Calculate overall progress from phase weights
    fn calculate_overall(&mut self) {
        let phases = [
            ProgressPhase::LoadingModel,
            ProgressPhase::Resampling,
            ProgressPhase::MelSpectrogram,
            ProgressPhase::Encoding,
            ProgressPhase::Decoding,
            ProgressPhase::Postprocessing,
            ProgressPhase::Complete,
        ];

        let total_weight: u32 = phases.iter().map(|p| p.weight()).sum();
        let mut completed_weight: u32 = 0;

        for p in &phases {
            if *p == self.phase {
                // Add fractional progress of current phase
                let phase_contribution = p.weight() as f32 * (self.phase_progress / 100.0);
                completed_weight += phase_contribution as u32;
                break;
            }
            completed_weight += p.weight();
        }

        self.overall_progress = (completed_weight as f32 / total_weight as f32) * 100.0;
    }

    /// Mark as complete
    pub fn complete(&mut self) {
        self.phase = ProgressPhase::Complete;
        self.phase_progress = 100.0;
        self.overall_progress = 100.0;
        self.message = "Complete".to_string();
    }
}

impl Default for WorkerProgress {
    fn default() -> Self {
        Self::new()
    }
}

/// Worker configuration options
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Model type to load
    model_type: String,
    /// Enable progress callbacks
    enable_progress: bool,
    /// Progress callback interval in ms
    progress_interval_ms: u32,
}

#[wasm_bindgen]
impl WorkerConfig {
    /// Create default configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            model_type: "tiny".to_string(),
            enable_progress: true,
            progress_interval_ms: 100,
        }
    }

    /// Set model type
    #[wasm_bindgen(setter, js_name = modelType)]
    pub fn set_model_type(&mut self, model_type: &str) {
        self.model_type = model_type.to_string();
    }

    /// Get model type
    #[wasm_bindgen(getter, js_name = modelType)]
    pub fn model_type(&self) -> String {
        self.model_type.clone()
    }

    /// Set progress enabled
    #[wasm_bindgen(setter, js_name = enableProgress)]
    pub fn set_enable_progress(&mut self, enable: bool) {
        self.enable_progress = enable;
    }

    /// Get progress enabled
    #[wasm_bindgen(getter, js_name = enableProgress)]
    pub fn enable_progress(&self) -> bool {
        self.enable_progress
    }

    /// Set progress interval
    #[wasm_bindgen(setter, js_name = progressIntervalMs)]
    pub fn set_progress_interval_ms(&mut self, interval: u32) {
        self.progress_interval_ms = interval;
    }

    /// Get progress interval
    #[wasm_bindgen(getter, js_name = progressIntervalMs)]
    pub fn progress_interval_ms(&self) -> u32 {
        self.progress_interval_ms
    }
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // WorkerMessageType Tests
    // =========================================================================

    #[test]
    fn test_worker_message_type_from_str() {
        assert_eq!(
            WorkerMessageType::from_str("init"),
            Some(WorkerMessageType::Init)
        );
        assert_eq!(
            WorkerMessageType::from_str("transcribe"),
            Some(WorkerMessageType::Transcribe)
        );
        assert_eq!(
            WorkerMessageType::from_str("detectLanguage"),
            Some(WorkerMessageType::DetectLanguage)
        );
        assert_eq!(
            WorkerMessageType::from_str("abort"),
            Some(WorkerMessageType::Abort)
        );
        assert_eq!(
            WorkerMessageType::from_str("ready"),
            Some(WorkerMessageType::Ready)
        );
        assert_eq!(
            WorkerMessageType::from_str("result"),
            Some(WorkerMessageType::Result)
        );
        assert_eq!(
            WorkerMessageType::from_str("language"),
            Some(WorkerMessageType::Language)
        );
        assert_eq!(
            WorkerMessageType::from_str("progress"),
            Some(WorkerMessageType::Progress)
        );
        assert_eq!(
            WorkerMessageType::from_str("error"),
            Some(WorkerMessageType::Error)
        );
        assert_eq!(WorkerMessageType::from_str("invalid"), None);
    }

    #[test]
    fn test_worker_message_type_as_str() {
        assert_eq!(WorkerMessageType::Init.as_str(), "init");
        assert_eq!(WorkerMessageType::Transcribe.as_str(), "transcribe");
        assert_eq!(WorkerMessageType::DetectLanguage.as_str(), "detectLanguage");
        assert_eq!(WorkerMessageType::Abort.as_str(), "abort");
        assert_eq!(WorkerMessageType::Ready.as_str(), "ready");
        assert_eq!(WorkerMessageType::Result.as_str(), "result");
        assert_eq!(WorkerMessageType::Language.as_str(), "language");
        assert_eq!(WorkerMessageType::Progress.as_str(), "progress");
        assert_eq!(WorkerMessageType::Error.as_str(), "error");
    }

    #[test]
    fn test_worker_message_type_roundtrip() {
        let types = [
            WorkerMessageType::Init,
            WorkerMessageType::Transcribe,
            WorkerMessageType::DetectLanguage,
            WorkerMessageType::Abort,
            WorkerMessageType::Ready,
            WorkerMessageType::Result,
            WorkerMessageType::Language,
            WorkerMessageType::Progress,
            WorkerMessageType::Error,
        ];

        for t in types {
            let s = t.as_str();
            let parsed = WorkerMessageType::from_str(s);
            assert_eq!(parsed, Some(t));
        }
    }

    // =========================================================================
    // WorkerState Tests
    // =========================================================================

    #[test]
    fn test_worker_state_default() {
        let state = WorkerState::default();
        assert_eq!(state, WorkerState::Uninitialized);
    }

    #[test]
    fn test_worker_state_is_ready() {
        assert!(!WorkerState::Uninitialized.is_ready());
        assert!(!WorkerState::Loading.is_ready());
        assert!(WorkerState::Ready.is_ready());
        assert!(!WorkerState::Transcribing.is_ready());
        assert!(!WorkerState::Error.is_ready());
    }

    #[test]
    fn test_worker_state_is_busy() {
        assert!(!WorkerState::Uninitialized.is_busy());
        assert!(WorkerState::Loading.is_busy());
        assert!(!WorkerState::Ready.is_busy());
        assert!(WorkerState::Transcribing.is_busy());
        assert!(WorkerState::DetectingLanguage.is_busy());
        assert!(!WorkerState::Error.is_busy());
    }

    #[test]
    fn test_worker_state_name() {
        assert_eq!(WorkerState::Uninitialized.name(), "uninitialized");
        assert_eq!(WorkerState::Loading.name(), "loading");
        assert_eq!(WorkerState::Ready.name(), "ready");
        assert_eq!(WorkerState::Transcribing.name(), "transcribing");
        assert_eq!(WorkerState::DetectingLanguage.name(), "detectingLanguage");
        assert_eq!(WorkerState::Aborted.name(), "aborted");
        assert_eq!(WorkerState::Error.name(), "error");
    }

    // =========================================================================
    // ProgressPhase Tests
    // =========================================================================

    #[test]
    fn test_progress_phase_name() {
        assert_eq!(ProgressPhase::LoadingModel.name(), "Loading model");
        assert_eq!(ProgressPhase::Resampling.name(), "Resampling audio");
        assert_eq!(
            ProgressPhase::MelSpectrogram.name(),
            "Computing mel spectrogram"
        );
        assert_eq!(ProgressPhase::Encoding.name(), "Running encoder");
        assert_eq!(ProgressPhase::Decoding.name(), "Running decoder");
        assert_eq!(ProgressPhase::Postprocessing.name(), "Processing results");
        assert_eq!(ProgressPhase::Complete.name(), "Complete");
    }

    #[test]
    fn test_progress_phase_weight() {
        // Encoding and decoding should have highest weights
        assert!(ProgressPhase::Encoding.weight() > ProgressPhase::Resampling.weight());
        assert!(ProgressPhase::Decoding.weight() > ProgressPhase::Encoding.weight());
        assert_eq!(ProgressPhase::Complete.weight(), 0);
    }

    #[test]
    fn test_progress_phase_weights_sum() {
        let total: u32 = [
            ProgressPhase::LoadingModel,
            ProgressPhase::Resampling,
            ProgressPhase::MelSpectrogram,
            ProgressPhase::Encoding,
            ProgressPhase::Decoding,
            ProgressPhase::Postprocessing,
        ]
        .iter()
        .map(|p| p.weight())
        .sum();

        assert_eq!(total, 100); // Should sum to 100 for easy percentage calculation
    }

    // =========================================================================
    // WorkerProgress Tests
    // =========================================================================

    #[test]
    fn test_worker_progress_new() {
        let progress = WorkerProgress::new();
        assert_eq!(progress.phase(), ProgressPhase::LoadingModel);
        assert!((progress.phase_progress() - 0.0).abs() < f32::EPSILON);
        assert!((progress.overall_progress() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_worker_progress_update() {
        let mut progress = WorkerProgress::new();

        progress.update(ProgressPhase::Encoding, 50.0, "Processing...");

        assert_eq!(progress.phase(), ProgressPhase::Encoding);
        assert!((progress.phase_progress() - 50.0).abs() < f32::EPSILON);
        assert_eq!(progress.message(), "Processing...");
        assert!(progress.overall_progress() > 0.0);
    }

    #[test]
    fn test_worker_progress_next_phase() {
        let mut progress = WorkerProgress::new();

        progress.next_phase(ProgressPhase::Resampling);
        assert_eq!(progress.phase(), ProgressPhase::Resampling);
        assert!((progress.phase_progress() - 0.0).abs() < f32::EPSILON);

        progress.next_phase(ProgressPhase::MelSpectrogram);
        assert_eq!(progress.phase(), ProgressPhase::MelSpectrogram);
    }

    #[test]
    fn test_worker_progress_complete() {
        let mut progress = WorkerProgress::new();

        progress.complete();

        assert_eq!(progress.phase(), ProgressPhase::Complete);
        assert!((progress.phase_progress() - 100.0).abs() < f32::EPSILON);
        assert!((progress.overall_progress() - 100.0).abs() < f32::EPSILON);
        assert_eq!(progress.message(), "Complete");
    }

    #[test]
    fn test_worker_progress_overall_increases() {
        let mut progress = WorkerProgress::new();

        let initial = progress.overall_progress();

        progress.next_phase(ProgressPhase::Encoding);
        let after_encoding = progress.overall_progress();

        assert!(after_encoding > initial);

        progress.next_phase(ProgressPhase::Decoding);
        let after_decoding = progress.overall_progress();

        assert!(after_decoding > after_encoding);
    }

    #[test]
    fn test_worker_progress_clamps_phase_progress() {
        let mut progress = WorkerProgress::new();

        progress.update(ProgressPhase::Encoding, 150.0, "Over 100");
        assert!((progress.phase_progress() - 100.0).abs() < f32::EPSILON);

        progress.update(ProgressPhase::Encoding, -50.0, "Under 0");
        assert!((progress.phase_progress() - 0.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // WorkerConfig Tests
    // =========================================================================

    #[test]
    fn test_worker_config_default() {
        let config = WorkerConfig::default();
        assert_eq!(config.model_type(), "tiny");
        assert!(config.enable_progress());
        assert_eq!(config.progress_interval_ms(), 100);
    }

    #[test]
    fn test_worker_config_setters() {
        let mut config = WorkerConfig::new();

        config.set_model_type("base");
        assert_eq!(config.model_type(), "base");

        config.set_enable_progress(false);
        assert!(!config.enable_progress());

        config.set_progress_interval_ms(200);
        assert_eq!(config.progress_interval_ms(), 200);
    }
}
