//! WAPR-DEMO-001: Real-time Microphone Transcription Demo
//!
//! Pure Rust WASM demo for real-time speech-to-text transcription.
//! Zero JavaScript - all browser APIs accessed via `web-sys`.
//!
//! # EXTREME TDD Status
//!
//! - \[x\] Red: Probar tests written (`tests/probar_tests.rs`)
//! - \[ \] Green: Implementation to pass tests
//! - \[ \] Refactor: Optimize while maintaining 95%+ coverage

use wasm_bindgen::prelude::*;

/// Demo application state machine
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DemoState {
    /// Initial state, ready to record
    #[default]
    Idle,
    /// Requesting microphone permission from browser
    RequestingPermission,
    /// Actively recording audio
    Recording,
    /// Processing recorded audio through Whisper
    Processing,
    /// Error state (permission denied, etc.)
    Error,
}

/// State transition validator for the demo state machine
pub struct StateTransition;

impl StateTransition {
    /// Check if a state transition is valid
    #[must_use]
    #[allow(clippy::unnested_or_patterns)]
    pub const fn is_valid(from: DemoState, to: DemoState) -> bool {
        matches!(
            (from, to),
            // Valid transitions
            (DemoState::Idle, DemoState::RequestingPermission)
                | (DemoState::RequestingPermission, DemoState::Recording)
                | (DemoState::RequestingPermission, DemoState::Error)
                | (DemoState::Recording, DemoState::Processing)
                | (DemoState::Processing, DemoState::Idle)
                | (DemoState::Error, DemoState::Idle)
                // Self-transitions (no-op)
                | (DemoState::Idle, DemoState::Idle)
                | (DemoState::Recording, DemoState::Recording)
        )
    }
}

/// Real-time transcription demo application
#[wasm_bindgen]
pub struct RealtimeTranscriptionDemo {
    state: DemoState,
    transcript: String,
    partial_transcript: String,
    recording_duration_ms: u32,
    error_message: Option<String>,
}

#[wasm_bindgen]
impl RealtimeTranscriptionDemo {
    /// Create a new demo instance
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: DemoState::Idle,
            transcript: String::new(),
            partial_transcript: String::new(),
            recording_duration_ms: 0,
            error_message: None,
        }
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> DemoState {
        self.state
    }

    /// Get status text for UI display
    #[must_use]
    pub fn status_text(&self) -> String {
        match self.state {
            DemoState::Idle => "Ready".to_string(),
            DemoState::RequestingPermission => "Requesting microphone...".to_string(),
            DemoState::Recording => "Recording...".to_string(),
            DemoState::Processing => "Processing...".to_string(),
            DemoState::Error => "Error".to_string(),
        }
    }

    /// Get the current transcript
    #[must_use]
    pub fn transcript(&self) -> String {
        self.transcript.clone()
    }

    /// Get partial transcript (streaming results)
    #[must_use]
    pub fn partial_transcript(&self) -> String {
        self.partial_transcript.clone()
    }

    /// Get recording duration formatted as M:SS
    #[must_use]
    pub fn recording_duration(&self) -> String {
        let seconds = self.recording_duration_ms / 1000;
        let minutes = seconds / 60;
        let secs = seconds % 60;
        format!("{minutes}:{secs:02}")
    }

    /// Get error message if in error state
    #[must_use]
    pub fn error_message(&self) -> Option<String> {
        self.error_message.clone()
    }

    /// Check if start recording button should be enabled
    #[must_use]
    pub fn can_start_recording(&self) -> bool {
        matches!(self.state, DemoState::Idle | DemoState::Error)
    }

    /// Check if stop recording button should be enabled
    #[must_use]
    pub fn can_stop_recording(&self) -> bool {
        self.state == DemoState::Recording
    }

    /// Start recording - requests microphone permission
    ///
    /// # Errors
    ///
    /// Returns error if recording cannot be started in current state.
    pub fn start_recording(&mut self) -> Result<(), JsValue> {
        if !self.can_start_recording() {
            return Err(JsValue::from_str("Cannot start recording in current state"));
        }

        self.transition_to(DemoState::RequestingPermission)?;

        // TODO: Implement actual microphone access via web-sys
        // This is a stub - will be implemented in Green phase

        Ok(())
    }

    /// Stop recording and begin processing
    ///
    /// # Errors
    ///
    /// Returns error if not currently recording.
    pub fn stop_recording(&mut self) -> Result<(), JsValue> {
        if !self.can_stop_recording() {
            return Err(JsValue::from_str("Cannot stop recording in current state"));
        }

        self.transition_to(DemoState::Processing)?;

        // TODO: Implement actual transcription
        // This is a stub - will be implemented in Green phase

        Ok(())
    }

    /// Clear the transcript
    pub fn clear_transcript(&mut self) {
        self.transcript.clear();
        self.partial_transcript.clear();
        self.recording_duration_ms = 0;
    }

    /// Handle permission granted callback
    ///
    /// # Errors
    ///
    /// Returns error if state transition is invalid.
    pub fn on_permission_granted(&mut self) -> Result<(), JsValue> {
        self.transition_to(DemoState::Recording)
    }

    /// Handle permission denied callback
    ///
    /// # Errors
    ///
    /// Returns error if state transition is invalid.
    pub fn on_permission_denied(&mut self) -> Result<(), JsValue> {
        self.error_message = Some("Microphone access denied".to_string());
        self.transition_to(DemoState::Error)
    }

    /// Handle transcription complete callback
    ///
    /// # Errors
    ///
    /// Returns error if state transition is invalid.
    pub fn on_transcription_complete(&mut self, text: &str) -> Result<(), JsValue> {
        self.transcript = text.to_string();
        self.partial_transcript.clear();
        self.transition_to(DemoState::Idle)
    }

    /// Handle partial transcription result
    pub fn on_partial_result(&mut self, text: &str) {
        self.partial_transcript = text.to_string();
    }

    /// Update recording duration (called periodically)
    pub fn update_duration(&mut self, elapsed_ms: u32) {
        self.recording_duration_ms = elapsed_ms;
    }

    /// Retry after error
    ///
    /// # Errors
    ///
    /// Returns error if not in error state.
    pub fn retry(&mut self) -> Result<(), JsValue> {
        if self.state != DemoState::Error {
            return Err(JsValue::from_str("Can only retry from error state"));
        }
        self.error_message = None;
        self.transition_to(DemoState::Idle)
    }

    /// Internal state transition with validation
    fn transition_to(&mut self, new_state: DemoState) -> Result<(), JsValue> {
        if StateTransition::is_valid(self.state, new_state) {
            self.state = new_state;
            Ok(())
        } else {
            Err(JsValue::from_str(&format!(
                "Invalid state transition: {:?} -> {:?}",
                self.state, new_state
            )))
        }
    }
}

impl Default for RealtimeTranscriptionDemo {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Browser Feature Detection
// ============================================================================

/// Check if browser supports required features
#[wasm_bindgen]
#[must_use]
pub fn check_browser_compatibility() -> BrowserCompatibility {
    BrowserCompatibility::check()
}

/// Browser compatibility status
#[wasm_bindgen]
pub struct BrowserCompatibility {
    pub audio_worklet: bool,
    pub media_devices: bool,
    pub wasm_simd: bool,
}

#[wasm_bindgen]
impl BrowserCompatibility {
    /// Check current browser's capabilities
    #[must_use]
    pub fn check() -> Self {
        // TODO: Implement actual feature detection via web-sys
        Self {
            audio_worklet: true, // Stub
            media_devices: true, // Stub
            wasm_simd: true,     // Stub
        }
    }

    /// Check if all required features are supported
    #[must_use]
    pub fn is_supported(&self) -> bool {
        self.audio_worklet && self.media_devices
    }

    /// Get warning message if not fully supported
    #[must_use]
    pub fn warning_message(&self) -> Option<String> {
        if self.is_supported() {
            None
        } else {
            Some("Your browser does not support real-time audio recording".to_string())
        }
    }
}

// ============================================================================
// Arbitrary trait for property testing
// ============================================================================

#[cfg(test)]
impl proptest::arbitrary::Arbitrary for DemoState {
    type Parameters = ();
    type Strategy = proptest::strategy::BoxedStrategy<Self>;

    fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
        use proptest::prelude::*;
        prop_oneof![
            Just(DemoState::Idle),
            Just(DemoState::RequestingPermission),
            Just(DemoState::Recording),
            Just(DemoState::Processing),
            Just(DemoState::Error),
        ]
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_demo_is_idle() {
        let demo = RealtimeTranscriptionDemo::new();
        assert_eq!(demo.state(), DemoState::Idle);
    }

    #[test]
    fn test_status_text_matches_state() {
        let demo = RealtimeTranscriptionDemo::new();
        assert_eq!(demo.status_text(), "Ready");
    }

    #[test]
    fn test_initial_duration_is_zero() {
        let demo = RealtimeTranscriptionDemo::new();
        assert_eq!(demo.recording_duration(), "0:00");
    }

    #[test]
    fn test_duration_formatting() {
        let mut demo = RealtimeTranscriptionDemo::new();
        demo.update_duration(65_000); // 1:05
        assert_eq!(demo.recording_duration(), "1:05");
    }

    #[test]
    fn test_can_start_from_idle() {
        let demo = RealtimeTranscriptionDemo::new();
        assert!(demo.can_start_recording());
    }

    #[test]
    fn test_cannot_stop_when_idle() {
        let demo = RealtimeTranscriptionDemo::new();
        assert!(!demo.can_stop_recording());
    }

    #[test]
    fn test_clear_transcript() {
        let mut demo = RealtimeTranscriptionDemo::new();
        demo.transcript = "Hello world".to_string();
        demo.clear_transcript();
        assert!(demo.transcript().is_empty());
    }

    #[test]
    fn test_valid_state_transitions() {
        assert!(StateTransition::is_valid(
            DemoState::Idle,
            DemoState::RequestingPermission
        ));
        assert!(StateTransition::is_valid(
            DemoState::RequestingPermission,
            DemoState::Recording
        ));
        assert!(StateTransition::is_valid(
            DemoState::Recording,
            DemoState::Processing
        ));
        assert!(StateTransition::is_valid(
            DemoState::Processing,
            DemoState::Idle
        ));
    }

    #[test]
    fn test_invalid_state_transitions() {
        assert!(!StateTransition::is_valid(
            DemoState::Idle,
            DemoState::Recording
        ));
        assert!(!StateTransition::is_valid(
            DemoState::Processing,
            DemoState::Recording
        ));
    }
}
