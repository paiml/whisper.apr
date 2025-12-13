//! WAPR-DEMO-004: Audio/Video Upload Translation Demo
//!
//! Pure Rust WASM demo for file-based speech translation to English.
//! Supports audio files (WAV, MP3, OGG) and video files (MP4, `WebM`).
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
    /// Initial state, no file selected
    #[default]
    Idle,
    /// File selected, ready to translate
    FileSelected,
    /// Detecting source language
    DetectingLanguage,
    /// Translation in progress
    Translating,
    /// Translation complete
    Complete,
    /// Error state
    Error,
}

/// Translation segment with source and target text
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct TranslationSegment {
    start_seconds: f64,
    end_seconds: f64,
    source_text: String,
    translated_text: String,
}

#[wasm_bindgen]
impl TranslationSegment {
    /// Get formatted timestamp
    #[must_use]
    pub fn timestamp(&self) -> String {
        format!(
            "[{} - {}]",
            format_timestamp(self.start_seconds),
            format_timestamp(self.end_seconds)
        )
    }

    /// Get source text
    #[must_use]
    pub fn source_text(&self) -> String {
        self.source_text.clone()
    }

    /// Get translated text
    #[must_use]
    pub fn translated_text(&self) -> String {
        self.translated_text.clone()
    }
}

/// File information
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct FileInfo {
    name: String,
    size_bytes: u64,
    duration_seconds: Option<f64>,
}

#[wasm_bindgen]
impl FileInfo {
    /// Get filename
    #[must_use]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Get file size formatted
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn size_formatted(&self) -> String {
        let mb = self.size_bytes as f64 / (1024.0 * 1024.0);
        format!("{mb:.1} MB")
    }

    /// Get duration formatted
    #[must_use]
    pub fn duration_formatted(&self) -> Option<String> {
        self.duration_seconds.map(format_duration)
    }
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
    /// Get language name
    #[must_use]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Check if English
    #[must_use]
    pub fn is_english(&self) -> bool {
        self.code == "en"
    }

    /// Get confidence percentage
    #[must_use]
    pub fn confidence_text(&self) -> String {
        format!("{:.0}%", self.confidence)
    }
}

/// Upload translation demo application
#[wasm_bindgen]
pub struct UploadTranslationDemo {
    state: DemoState,
    file_info: Option<FileInfo>,
    detected_language: Option<DetectedLanguage>,
    segments: Vec<TranslationSegment>,
    error_message: Option<String>,
    bilingual_export: bool,
    show_timestamps: bool,
}

#[wasm_bindgen]
impl UploadTranslationDemo {
    /// Create a new demo instance
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: DemoState::Idle,
            file_info: None,
            detected_language: None,
            segments: Vec::new(),
            error_message: None,
            bilingual_export: false,
            show_timestamps: false,
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

    /// Get status text
    #[must_use]
    pub fn status_text(&self) -> String {
        match self.state {
            DemoState::Idle => "Ready".to_string(),
            DemoState::FileSelected => "File selected".to_string(),
            DemoState::DetectingLanguage => "Detecting language...".to_string(),
            DemoState::Translating => "Translating...".to_string(),
            DemoState::Complete => "Complete".to_string(),
            DemoState::Error => "Error".to_string(),
        }
    }

    /// Get file info
    #[must_use]
    pub fn file_info(&self) -> Option<FileInfo> {
        self.file_info.clone()
    }

    /// Get detected language
    #[must_use]
    pub fn detected_language(&self) -> Option<DetectedLanguage> {
        self.detected_language.clone()
    }

    /// Get language pair display text
    #[must_use]
    pub fn language_pair(&self) -> String {
        self.detected_language
            .as_ref()
            .map(|l| format_language_pair(&l.name, "English"))
            .unwrap_or_default()
    }

    /// Check if source is English (translation not needed)
    #[must_use]
    pub fn is_source_english(&self) -> bool {
        self.detected_language
            .as_ref()
            .is_some_and(DetectedLanguage::is_english)
    }

    /// Get source text (all segments)
    #[must_use]
    pub fn source_text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.source_text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get translated text (all segments)
    #[must_use]
    pub fn translated_text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.translated_text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get segments count
    #[must_use]
    pub fn segments_count(&self) -> usize {
        self.segments.len()
    }

    /// Get error message
    #[must_use]
    pub fn error_message(&self) -> Option<String> {
        self.error_message.clone()
    }

    /// Check if translate button should be enabled
    #[must_use]
    pub fn can_translate(&self) -> bool {
        matches!(self.state, DemoState::FileSelected | DemoState::Complete)
    }

    /// Check if download buttons should be enabled
    #[must_use]
    pub fn can_download(&self) -> bool {
        self.state == DemoState::Complete && !self.segments.is_empty()
    }

    /// Check if bilingual export is enabled
    #[must_use]
    pub fn is_bilingual_export(&self) -> bool {
        self.bilingual_export
    }

    /// Toggle bilingual export
    pub fn toggle_bilingual_export(&mut self) {
        self.bilingual_export = !self.bilingual_export;
    }

    /// Check if timestamps are shown
    #[must_use]
    pub fn is_show_timestamps(&self) -> bool {
        self.show_timestamps
    }

    /// Toggle timestamps display
    pub fn toggle_timestamps(&mut self) {
        self.show_timestamps = !self.show_timestamps;
    }

    /// Handle file selection
    ///
    /// # Errors
    ///
    /// This function always returns `Ok(())` - errors are stored in state.
    pub fn on_file_selected(&mut self, name: &str, size_bytes: u64) -> Result<(), JsValue> {
        self.file_info = Some(FileInfo {
            name: name.to_string(),
            size_bytes,
            duration_seconds: None,
        });
        self.state = DemoState::FileSelected;
        self.error_message = None;
        Ok(())
    }

    /// Set file duration
    pub fn set_file_duration(&mut self, duration_seconds: f64) {
        if let Some(ref mut info) = self.file_info {
            info.duration_seconds = Some(duration_seconds);
        }
    }

    /// Start translation
    ///
    /// # Errors
    ///
    /// Returns error if translation cannot be started in current state.
    pub fn start_translation(&mut self) -> Result<(), JsValue> {
        if !self.can_translate() {
            return Err(JsValue::from_str("Cannot translate in current state"));
        }

        self.state = DemoState::DetectingLanguage;
        self.segments.clear();
        Ok(())
    }

    /// Set detected language
    pub fn set_detected_language(&mut self, code: &str, name: &str, confidence: f32) {
        self.detected_language = Some(DetectedLanguage {
            code: code.to_string(),
            name: name.to_string(),
            confidence,
        });

        // Move to translating state
        self.state = DemoState::Translating;
    }

    /// Add a translation segment
    pub fn add_segment(&mut self, start: f64, end: f64, source: &str, translated: &str) {
        self.segments.push(TranslationSegment {
            start_seconds: start,
            end_seconds: end,
            source_text: source.to_string(),
            translated_text: translated.to_string(),
        });
    }

    /// Complete translation
    pub fn complete(&mut self) {
        self.state = DemoState::Complete;
    }

    /// Export translation as plain text
    #[must_use]
    pub fn export_txt(&self) -> String {
        self.translated_text()
    }

    /// Export as SRT (translation only)
    #[must_use]
    pub fn export_srt(&self) -> String {
        self.segments
            .iter()
            .enumerate()
            .map(|(i, seg)| {
                format!(
                    "{}\n{} --> {}\n{}\n",
                    i + 1,
                    format_srt_timestamp(seg.start_seconds),
                    format_srt_timestamp(seg.end_seconds),
                    seg.translated_text
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Export as bilingual SRT (source + translation)
    #[must_use]
    pub fn export_bilingual_srt(&self) -> String {
        self.segments
            .iter()
            .enumerate()
            .map(|(i, seg)| {
                format!(
                    "{}\n{} --> {}\n{}\n{}\n",
                    i + 1,
                    format_srt_timestamp(seg.start_seconds),
                    format_srt_timestamp(seg.end_seconds),
                    seg.source_text,
                    seg.translated_text
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Export as VTT
    #[must_use]
    pub fn export_vtt(&self) -> String {
        let segments: Vec<_> = self
            .segments
            .iter()
            .map(|seg| {
                format!(
                    "{} --> {}\n{}",
                    format_vtt_timestamp(seg.start_seconds),
                    format_vtt_timestamp(seg.end_seconds),
                    seg.translated_text
                )
            })
            .collect();

        format!("WEBVTT\n\n{}", segments.join("\n\n"))
    }

    /// Clear and reset
    pub fn clear(&mut self) {
        self.state = DemoState::Idle;
        self.file_info = None;
        self.detected_language = None;
        self.segments.clear();
        self.error_message = None;
    }
}

impl Default for UploadTranslationDemo {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Format language pair for display
#[must_use]
pub fn format_language_pair(source: &str, target: &str) -> String {
    format!("{source} → {target}")
}

/// Format duration as M:SS or H:MM:SS
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn format_duration(seconds: f64) -> String {
    let total_seconds = seconds as u64;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let secs = total_seconds % 60;

    if hours > 0 {
        format!("{hours}:{minutes:02}:{secs:02}")
    } else {
        format!("{minutes}:{secs:02}")
    }
}

/// Format timestamp as MM:SS.mmm
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn format_timestamp(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let minutes = total_ms / 60000;
    let secs = (total_ms % 60000) / 1000;
    let ms = total_ms % 1000;
    format!("{minutes:02}:{secs:02}.{ms:03}")
}

/// Format timestamp for SRT format
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn format_srt_timestamp(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60000;
    let secs = (total_ms % 60000) / 1000;
    let ms = total_ms % 1000;
    format!("{hours:02}:{minutes:02}:{secs:02},{ms:03}")
}

/// Format timestamp for VTT format
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn format_vtt_timestamp(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60000;
    let secs = (total_ms % 60000) / 1000;
    let ms = total_ms % 1000;
    format!("{hours:02}:{minutes:02}:{secs:02}.{ms:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let demo = UploadTranslationDemo::new();
        assert_eq!(demo.state(), DemoState::Idle);
        assert_eq!(demo.task(), "translate".to_string());
    }

    #[test]
    fn test_language_pair_formatting() {
        assert_eq!(
            format_language_pair("Spanish", "English"),
            "Spanish → English"
        );
    }

    #[test]
    fn test_bilingual_srt() {
        let mut demo = UploadTranslationDemo::new();
        demo.add_segment(0.0, 2.0, "Hola", "Hello");

        let srt = demo.export_bilingual_srt();
        assert!(srt.contains("Hola"));
        assert!(srt.contains("Hello"));
        assert!(srt.contains("00:00:00,000 --> 00:00:02,000"));
    }

    #[test]
    fn test_standard_srt() {
        let mut demo = UploadTranslationDemo::new();
        demo.add_segment(0.0, 2.0, "Hola", "Hello");

        let srt = demo.export_srt();
        assert!(srt.contains("Hello"));
        assert!(!srt.contains("Hola")); // Should only have translation
    }

    #[test]
    fn test_toggle_bilingual() {
        let mut demo = UploadTranslationDemo::new();
        assert!(!demo.is_bilingual_export());
        demo.toggle_bilingual_export();
        assert!(demo.is_bilingual_export());
    }
}
