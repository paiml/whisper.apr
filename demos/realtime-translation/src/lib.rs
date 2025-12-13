//! WAPR-DEMO-003: Real-time Microphone Translation Demo
//!
//! Pure Rust WASM demo for real-time speech translation to English.
//! Supports 99 languages with automatic language detection.
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
    /// Translating recorded audio to English
    Translating,
    /// Translation complete
    Complete,
    /// Error state
    Error,
}

/// Confidence threshold for "high confidence" detection
pub const HIGH_CONFIDENCE_THRESHOLD: f32 = 70.0;

/// Check if confidence score is considered high
#[must_use]
pub fn is_high_confidence(confidence: f32) -> bool {
    confidence >= HIGH_CONFIDENCE_THRESHOLD
}

/// Detected language information
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DetectedLanguage {
    code: String,
    name: String,
    confidence: f32,
}

#[wasm_bindgen]
impl DetectedLanguage {
    /// Get language code (e.g., "es", "fr", "zh")
    #[must_use]
    pub fn code(&self) -> String {
        self.code.clone()
    }

    /// Get language name (e.g., "Spanish", "French", "Chinese")
    #[must_use]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Get detection confidence (0-100)
    #[must_use]
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Get formatted confidence string
    #[must_use]
    pub fn confidence_text(&self) -> String {
        format!("{:.0}%", self.confidence)
    }

    /// Check if this is English
    #[must_use]
    pub fn is_english(&self) -> bool {
        self.code == "en"
    }

    /// Check if confidence is high enough for reliable translation
    #[must_use]
    pub fn is_high_confidence(&self) -> bool {
        is_high_confidence(self.confidence)
    }
}

/// Real-time translation demo application
#[wasm_bindgen]
pub struct RealtimeTranslationDemo {
    state: DemoState,
    detected_language: Option<DetectedLanguage>,
    original_text: String,
    translated_text: String,
    partial_translation: String,
    recording_duration_ms: u32,
    error_message: Option<String>,
    show_side_by_side: bool,
}

#[wasm_bindgen]
impl RealtimeTranslationDemo {
    /// Create a new demo instance
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: DemoState::Idle,
            detected_language: None,
            original_text: String::new(),
            translated_text: String::new(),
            partial_translation: String::new(),
            recording_duration_ms: 0,
            error_message: None,
            show_side_by_side: false,
        }
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> DemoState {
        self.state
    }

    /// Get task type (always "translate" for this demo)
    #[must_use]
    pub fn task(&self) -> String {
        "translate".to_string()
    }

    /// Get task indicator text
    #[must_use]
    pub fn task_indicator(&self) -> String {
        "Translation → English".to_string()
    }

    /// Get status text for UI display
    #[must_use]
    pub fn status_text(&self) -> String {
        match self.state {
            DemoState::Idle => "Ready".to_string(),
            DemoState::RequestingPermission => "Requesting microphone...".to_string(),
            DemoState::Recording => "Recording...".to_string(),
            DemoState::Translating => "Translating...".to_string(),
            DemoState::Complete => "Complete".to_string(),
            DemoState::Error => "Error".to_string(),
        }
    }

    /// Get detected language if available
    #[must_use]
    pub fn detected_language(&self) -> Option<DetectedLanguage> {
        self.detected_language.clone()
    }

    /// Get detected language display text
    #[must_use]
    pub fn detected_language_text(&self) -> String {
        self.detected_language
            .as_ref()
            .map_or_else(|| "--".to_string(), |l| l.name.clone())
    }

    /// Get original transcribed text (in source language)
    #[must_use]
    pub fn original_text(&self) -> String {
        self.original_text.clone()
    }

    /// Get translated text (in English)
    #[must_use]
    pub fn translated_text(&self) -> String {
        self.translated_text.clone()
    }

    /// Get partial translation (streaming)
    #[must_use]
    pub fn partial_translation(&self) -> String {
        self.partial_translation.clone()
    }

    /// Get recording duration formatted
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

    /// Check if source is English (translation not needed)
    #[must_use]
    pub fn is_source_english(&self) -> bool {
        self.detected_language
            .as_ref()
            .is_some_and(DetectedLanguage::is_english)
    }

    /// Check if should show low confidence warning
    #[must_use]
    pub fn should_show_low_confidence_warning(&self) -> bool {
        self.detected_language
            .as_ref()
            .is_some_and(|l| !l.is_high_confidence())
    }

    /// Check if side-by-side mode is enabled
    #[must_use]
    pub fn is_side_by_side(&self) -> bool {
        self.show_side_by_side
    }

    /// Toggle side-by-side display mode
    pub fn toggle_side_by_side(&mut self) {
        self.show_side_by_side = !self.show_side_by_side;
    }

    /// Check if start recording button should be enabled
    #[must_use]
    pub fn can_start_recording(&self) -> bool {
        matches!(
            self.state,
            DemoState::Idle | DemoState::Complete | DemoState::Error
        )
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

        self.state = DemoState::RequestingPermission;
        self.error_message = None;
        Ok(())
    }

    /// Stop recording and begin translation
    ///
    /// # Errors
    ///
    /// Returns error if not currently recording.
    pub fn stop_recording(&mut self) -> Result<(), JsValue> {
        if !self.can_stop_recording() {
            return Err(JsValue::from_str("Cannot stop recording in current state"));
        }

        self.state = DemoState::Translating;
        Ok(())
    }

    /// Handle permission granted
    pub fn on_permission_granted(&mut self) {
        self.state = DemoState::Recording;
    }

    /// Handle permission denied
    pub fn on_permission_denied(&mut self) {
        self.error_message = Some("Microphone access denied".to_string());
        self.state = DemoState::Error;
    }

    /// Update detected language during recording
    pub fn update_detected_language(&mut self, code: &str, name: &str, confidence: f32) {
        self.detected_language = Some(DetectedLanguage {
            code: code.to_string(),
            name: name.to_string(),
            confidence,
        });
    }

    /// Update partial translation (streaming)
    pub fn update_partial(&mut self, text: &str) {
        self.partial_translation = text.to_string();
    }

    /// Complete translation with results
    pub fn complete_translation(&mut self, original: &str, translated: &str) {
        self.original_text = original.to_string();
        self.translated_text = translated.to_string();
        self.partial_translation.clear();
        self.state = DemoState::Complete;
    }

    /// Update recording duration
    pub fn update_duration(&mut self, elapsed_ms: u32) {
        self.recording_duration_ms = elapsed_ms;
    }

    /// Clear and reset
    pub fn clear(&mut self) {
        self.state = DemoState::Idle;
        self.detected_language = None;
        self.original_text.clear();
        self.translated_text.clear();
        self.partial_translation.clear();
        self.recording_duration_ms = 0;
        self.error_message = None;
    }
}

impl Default for RealtimeTranslationDemo {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Supported Languages
// ============================================================================

/// Get list of supported language names
#[wasm_bindgen]
#[must_use]
pub fn supported_languages() -> Vec<JsValue> {
    SUPPORTED_LANGUAGES
        .iter()
        .map(|(_, name)| JsValue::from_str(name))
        .collect()
}

/// Get count of supported languages
#[wasm_bindgen]
#[must_use]
pub fn supported_languages_count() -> usize {
    SUPPORTED_LANGUAGES.len()
}

/// Whisper's 99 supported languages
pub const SUPPORTED_LANGUAGES: &[(&str, &str)] = &[
    ("af", "Afrikaans"),
    ("am", "Amharic"),
    ("ar", "Arabic"),
    ("as", "Assamese"),
    ("az", "Azerbaijani"),
    ("ba", "Bashkir"),
    ("be", "Belarusian"),
    ("bg", "Bulgarian"),
    ("bn", "Bengali"),
    ("bo", "Tibetan"),
    ("br", "Breton"),
    ("bs", "Bosnian"),
    ("ca", "Catalan"),
    ("cs", "Czech"),
    ("cy", "Welsh"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("eu", "Basque"),
    ("fa", "Persian"),
    ("fi", "Finnish"),
    ("fo", "Faroese"),
    ("fr", "French"),
    ("gl", "Galician"),
    ("gu", "Gujarati"),
    ("ha", "Hausa"),
    ("haw", "Hawaiian"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hr", "Croatian"),
    ("ht", "Haitian Creole"),
    ("hu", "Hungarian"),
    ("hy", "Armenian"),
    ("id", "Indonesian"),
    ("is", "Icelandic"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("jw", "Javanese"),
    ("ka", "Georgian"),
    ("kk", "Kazakh"),
    ("km", "Khmer"),
    ("kn", "Kannada"),
    ("ko", "Korean"),
    ("la", "Latin"),
    ("lb", "Luxembourgish"),
    ("ln", "Lingala"),
    ("lo", "Lao"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("mg", "Malagasy"),
    ("mi", "Maori"),
    ("mk", "Macedonian"),
    ("ml", "Malayalam"),
    ("mn", "Mongolian"),
    ("mr", "Marathi"),
    ("ms", "Malay"),
    ("mt", "Maltese"),
    ("my", "Myanmar"),
    ("ne", "Nepali"),
    ("nl", "Dutch"),
    ("nn", "Norwegian Nynorsk"),
    ("no", "Norwegian"),
    ("oc", "Occitan"),
    ("pa", "Punjabi"),
    ("pl", "Polish"),
    ("ps", "Pashto"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sa", "Sanskrit"),
    ("sd", "Sindhi"),
    ("si", "Sinhala"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sn", "Shona"),
    ("so", "Somali"),
    ("sq", "Albanian"),
    ("sr", "Serbian"),
    ("su", "Sundanese"),
    ("sv", "Swedish"),
    ("sw", "Swahili"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("tg", "Tajik"),
    ("th", "Thai"),
    ("tk", "Turkmen"),
    ("tl", "Tagalog"),
    ("tr", "Turkish"),
    ("tt", "Tatar"),
    ("uk", "Ukrainian"),
    ("ur", "Urdu"),
    ("uz", "Uzbek"),
    ("vi", "Vietnamese"),
    ("yi", "Yiddish"),
    ("yo", "Yoruba"),
    ("zh", "Chinese"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let demo = RealtimeTranslationDemo::new();
        assert_eq!(demo.state(), DemoState::Idle);
        assert_eq!(demo.task(), "translate".to_string());
    }

    #[test]
    fn test_task_indicator() {
        let demo = RealtimeTranslationDemo::new();
        assert_eq!(demo.task_indicator(), "Translation → English");
    }

    #[test]
    fn test_supported_languages_count() {
        assert_eq!(supported_languages_count(), 99);
    }

    #[test]
    fn test_confidence_threshold() {
        assert!(is_high_confidence(75.0));
        assert!(is_high_confidence(70.0));
        assert!(!is_high_confidence(69.9));
        assert!(!is_high_confidence(50.0));
    }

    #[test]
    fn test_detected_language_english() {
        let lang = DetectedLanguage {
            code: "en".to_string(),
            name: "English".to_string(),
            confidence: 95.0,
        };
        assert!(lang.is_english());
        assert!(lang.is_high_confidence());
    }

    #[test]
    fn test_side_by_side_toggle() {
        let mut demo = RealtimeTranslationDemo::new();
        assert!(!demo.is_side_by_side());
        demo.toggle_side_by_side();
        assert!(demo.is_side_by_side());
        demo.toggle_side_by_side();
        assert!(!demo.is_side_by_side());
    }
}
