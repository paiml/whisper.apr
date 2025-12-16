//! WAPR-DEMO-004: Audio/Video Upload Translation Demo
//!
//! Pure Rust WASM demo for file-based speech translation to English.
//!
//! Note: `#![allow(dead_code)]` is used because WASM exports are called from
//! JavaScript, not Rust. Static analysis incorrectly flags them as "dead".
#![allow(dead_code)]
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

// ============================================================================
// Zero-JS Entry Point - All DOM manipulation in Rust
// ============================================================================

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::closure::Closure;

thread_local! {
    static DEMO: RefCell<Option<Rc<RefCell<UploadTranslationDemo>>>> = const { RefCell::new(None) };
}

/// Zero-JS entry point - called automatically when WASM loads
///
/// # Errors
///
/// Returns an error if the DOM is not available (no window or document).
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let window = web_sys::window().ok_or("No window")?;
    let document = window.document().ok_or("No document")?;

    let demo = Rc::new(RefCell::new(UploadTranslationDemo::new()));
    DEMO.with(|d| *d.borrow_mut() = Some(demo.clone()));

    update_ui(&document, &demo.borrow())?;

    // Translate button
    setup_button_listener(&document, "translate", {
        let demo = demo.clone();
        move |doc| {
            let _ = demo.borrow_mut().start_translation();
            let _ = update_ui(doc, &demo.borrow());
        }
    })?;

    // Clear button
    setup_button_listener(&document, "clear", {
        let demo = demo.clone();
        move |doc| {
            demo.borrow_mut().clear();
            let _ = update_ui(doc, &demo.borrow());
        }
    })?;

    // File input change
    setup_file_input(&document, "file_input", demo.clone())?;

    // Drag and drop
    setup_drag_drop(&document, "drop_zone", demo)?;

    Ok(())
}

fn update_text_displays(document: &web_sys::Document, demo: &UploadTranslationDemo) {
    if let Some(el) = document.get_element_by_id("translation_display") {
        el.set_text_content(Some(&demo.translated_text()));
    }
    if let Some(el) = document.get_element_by_id("source_display") {
        el.set_text_content(Some(&demo.source_text()));
    }
}

fn update_file_info(document: &web_sys::Document, demo: &UploadTranslationDemo) {
    if let Some(el) = document.get_element_by_id("file_info") {
        if let Some(info) = demo.file_info() {
            el.set_text_content(Some(&format!("{} ({})", info.name(), info.size_formatted())));
            let _ = el.class_list().add_1("visible");
        } else {
            el.set_text_content(None);
            let _ = el.class_list().remove_1("visible");
        }
    }
}

fn update_button_states(document: &web_sys::Document, demo: &UploadTranslationDemo) {
    if let Some(btn) = document.get_element_by_id("translate") {
        if demo.can_translate() {
            let _ = btn.remove_attribute("disabled");
        } else {
            let _ = btn.set_attribute("disabled", "true");
        }
    }
    if let Some(btn) = document.get_element_by_id("download_result") {
        if demo.can_download() {
            let _ = btn.remove_attribute("disabled");
        } else {
            let _ = btn.set_attribute("disabled", "true");
        }
    }
}

fn update_drop_zone(document: &web_sys::Document, demo: &UploadTranslationDemo) {
    if let Some(zone) = document.get_element_by_id("drop_zone") {
        if demo.file_info().is_some() {
            let _ = zone.class_list().add_1("file-selected");
        } else {
            let _ = zone.class_list().remove_1("file-selected");
        }
    }
}

#[allow(clippy::unnecessary_wraps)]
fn update_ui(document: &web_sys::Document, demo: &UploadTranslationDemo) -> Result<(), JsValue> {
    update_text_displays(document, demo);
    update_file_info(document, demo);
    update_button_states(document, demo);
    update_drop_zone(document, demo);
    Ok(())
}

fn setup_button_listener<F>(
    document: &web_sys::Document,
    id: &str,
    handler: F,
) -> Result<(), JsValue>
where
    F: Fn(&web_sys::Document) + 'static,
{
    let doc = document.clone();
    let closure = Closure::wrap(Box::new(move |_: web_sys::Event| {
        handler(&doc);
    }) as Box<dyn Fn(_)>);

    if let Some(btn) = document.get_element_by_id(id) {
        btn.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
    }
    closure.forget();
    Ok(())
}

fn setup_file_input(
    document: &web_sys::Document,
    id: &str,
    demo: Rc<RefCell<UploadTranslationDemo>>,
) -> Result<(), JsValue> {
    let doc = document.clone();
    let closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
        if let Some(target) = event.target() {
            if let Ok(input) = target.dyn_into::<web_sys::HtmlInputElement>() {
                if let Some(files) = input.files() {
                    if let Some(file) = files.get(0) {
                        let name = file.name();
                        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                        let size = file.size() as u64;
                        let _ = demo.borrow_mut().on_file_selected(&name, size);
                        let _ = update_ui(&doc, &demo.borrow());
                    }
                }
            }
        }
    }) as Box<dyn Fn(_)>);

    if let Some(input) = document.get_element_by_id(id) {
        input.add_event_listener_with_callback("change", closure.as_ref().unchecked_ref())?;
    }
    closure.forget();
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
fn setup_drag_drop(
    document: &web_sys::Document,
    id: &str,
    demo: Rc<RefCell<UploadTranslationDemo>>,
) -> Result<(), JsValue> {
    let doc = document.clone();

    let dragover = Closure::wrap(Box::new(move |event: web_sys::DragEvent| {
        event.prevent_default();
    }) as Box<dyn Fn(_)>);

    let drop_handler = {
        let demo = demo.clone();
        let doc = doc.clone();
        Closure::wrap(Box::new(move |event: web_sys::DragEvent| {
            event.prevent_default();
            if let Some(dt) = event.data_transfer() {
                if let Some(files) = dt.files() {
                    if let Some(file) = files.get(0) {
                        let name = file.name();
                        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                        let size = file.size() as u64;
                        let _ = demo.borrow_mut().on_file_selected(&name, size);
                        let _ = update_ui(&doc, &demo.borrow());
                    }
                }
            }
        }) as Box<dyn Fn(_)>)
    };

    if let Some(zone) = document.get_element_by_id(id) {
        zone.add_event_listener_with_callback("dragover", dragover.as_ref().unchecked_ref())?;
        zone.add_event_listener_with_callback("drop", drop_handler.as_ref().unchecked_ref())?;
    }
    dragover.forget();
    drop_handler.forget();
    Ok(())
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

    // =========================================================================
    // State Tests - Cover ALL states
    // =========================================================================

    #[test]
    fn test_state_idle() {
        let demo = UploadTranslationDemo::new();
        assert_eq!(demo.state(), DemoState::Idle);
    }

    #[test]
    fn test_state_file_selected() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        assert_eq!(demo.state(), DemoState::FileSelected);
    }

    #[test]
    fn test_state_detecting_language() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_translation();
        assert_eq!(demo.state(), DemoState::DetectingLanguage);
    }

    #[test]
    fn test_state_translating() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_translation();
        demo.set_detected_language("es", "Spanish", 90.0);
        assert_eq!(demo.state(), DemoState::Translating);
    }

    #[test]
    fn test_state_complete() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_translation();
        demo.set_detected_language("es", "Spanish", 90.0);
        demo.complete();
        assert_eq!(demo.state(), DemoState::Complete);
    }

    // =========================================================================
    // Status Text Tests - Cover ALL branches
    // =========================================================================

    #[test]
    fn test_status_text_idle() {
        let demo = UploadTranslationDemo::new();
        assert_eq!(demo.status_text(), "Ready");
    }

    #[test]
    fn test_status_text_file_selected() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        assert_eq!(demo.status_text(), "File selected");
    }

    #[test]
    fn test_status_text_detecting_language() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_translation();
        assert_eq!(demo.status_text(), "Detecting language...");
    }

    #[test]
    fn test_status_text_translating() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_translation();
        demo.set_detected_language("es", "Spanish", 90.0);
        assert_eq!(demo.status_text(), "Translating...");
    }

    #[test]
    fn test_status_text_complete() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_translation();
        demo.set_detected_language("es", "Spanish", 90.0);
        demo.complete();
        assert_eq!(demo.status_text(), "Complete");
    }

    #[test]
    fn test_status_text_error() {
        let mut demo = UploadTranslationDemo::new();
        demo.error_message = Some("Test error".to_string());
        demo.state = DemoState::Error;
        assert_eq!(demo.status_text(), "Error");
    }

    // =========================================================================
    // FileInfo Tests
    // =========================================================================

    #[test]
    fn test_file_info_initially_none() {
        let demo = UploadTranslationDemo::new();
        assert!(demo.file_info().is_none());
    }

    #[test]
    fn test_file_info_name() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test_audio.wav", 1024 * 1024);
        let info = demo.file_info().unwrap();
        assert_eq!(info.name(), "test_audio.wav");
    }

    #[test]
    fn test_file_info_size_formatted() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 5 * 1024 * 1024);
        let info = demo.file_info().unwrap();
        assert!(info.size_formatted().contains("5.0 MB"));
    }

    #[test]
    fn test_file_info_duration_initially_none() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let info = demo.file_info().unwrap();
        assert!(info.duration_formatted().is_none());
    }

    #[test]
    fn test_file_info_duration_set() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        demo.set_file_duration(180.0);
        let info = demo.file_info().unwrap();
        assert!(info.duration_formatted().is_some());
    }

    // =========================================================================
    // Detected Language Tests
    // =========================================================================

    #[test]
    fn test_detected_language_initially_none() {
        let demo = UploadTranslationDemo::new();
        assert!(demo.detected_language().is_none());
    }

    #[test]
    fn test_detected_language_set() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_translation();
        demo.set_detected_language("fr", "French", 85.0);
        let lang = demo.detected_language().unwrap();
        assert_eq!(lang.name(), "French");
    }

    #[test]
    fn test_detected_language_is_english() {
        let lang = DetectedLanguage {
            code: "en".to_string(),
            name: "English".to_string(),
            confidence: 95.0,
        };
        assert!(lang.is_english());
    }

    #[test]
    fn test_detected_language_not_english() {
        let lang = DetectedLanguage {
            code: "es".to_string(),
            name: "Spanish".to_string(),
            confidence: 95.0,
        };
        assert!(!lang.is_english());
    }

    #[test]
    fn test_detected_language_confidence_text() {
        let lang = DetectedLanguage {
            code: "ja".to_string(),
            name: "Japanese".to_string(),
            confidence: 87.5,
        };
        assert_eq!(lang.confidence_text(), "88%");
    }

    #[test]
    fn test_is_source_english() {
        let mut demo = UploadTranslationDemo::new();
        assert!(!demo.is_source_english());
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_translation();
        demo.set_detected_language("en", "English", 95.0);
        assert!(demo.is_source_english());
    }

    #[test]
    fn test_language_pair_empty() {
        let demo = UploadTranslationDemo::new();
        assert!(demo.language_pair().is_empty());
    }

    #[test]
    fn test_language_pair_with_detected() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_translation();
        demo.set_detected_language("de", "German", 90.0);
        let pair = demo.language_pair();
        assert!(pair.contains("German"));
        assert!(pair.contains("English"));
    }

    // =========================================================================
    // Translation Text Tests
    // =========================================================================

    #[test]
    fn test_source_text_empty() {
        let demo = UploadTranslationDemo::new();
        assert!(demo.source_text().is_empty());
    }

    #[test]
    fn test_translated_text_empty() {
        let demo = UploadTranslationDemo::new();
        assert!(demo.translated_text().is_empty());
    }

    #[test]
    fn test_add_segment() {
        let mut demo = UploadTranslationDemo::new();
        demo.add_segment(0.0, 2.0, "Hola", "Hello");
        assert_eq!(demo.segments_count(), 1);
        assert!(demo.source_text().contains("Hola"));
        assert!(demo.translated_text().contains("Hello"));
    }

    #[test]
    fn test_segments_count() {
        let mut demo = UploadTranslationDemo::new();
        assert_eq!(demo.segments_count(), 0);
        demo.add_segment(0.0, 2.0, "Hola", "Hello");
        demo.add_segment(2.0, 4.0, "Mundo", "World");
        assert_eq!(demo.segments_count(), 2);
    }

    // =========================================================================
    // TranslationSegment Tests
    // =========================================================================

    #[test]
    fn test_segment_timestamp() {
        let segment = TranslationSegment {
            start_seconds: 0.0,
            end_seconds: 2.5,
            source_text: "Hola".to_string(),
            translated_text: "Hello".to_string(),
        };
        let ts = segment.timestamp();
        assert!(ts.contains("["));
        assert!(ts.contains("]"));
        assert!(ts.contains(" - "));
    }

    #[test]
    fn test_segment_source_text() {
        let segment = TranslationSegment {
            start_seconds: 0.0,
            end_seconds: 2.0,
            source_text: "Bonjour".to_string(),
            translated_text: "Hello".to_string(),
        };
        assert_eq!(segment.source_text(), "Bonjour");
    }

    #[test]
    fn test_segment_translated_text() {
        let segment = TranslationSegment {
            start_seconds: 0.0,
            end_seconds: 2.0,
            source_text: "Guten Tag".to_string(),
            translated_text: "Good day".to_string(),
        };
        assert_eq!(segment.translated_text(), "Good day");
    }

    // =========================================================================
    // Button State Tests
    // =========================================================================

    #[test]
    fn test_can_translate_initial() {
        let demo = UploadTranslationDemo::new();
        assert!(!demo.can_translate());
    }

    #[test]
    fn test_can_translate_file_selected() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        assert!(demo.can_translate());
    }

    #[test]
    fn test_can_translate_complete() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_translation();
        demo.set_detected_language("es", "Spanish", 90.0);
        demo.complete();
        assert!(demo.can_translate());
    }

    #[test]
    fn test_can_download_initial() {
        let demo = UploadTranslationDemo::new();
        assert!(!demo.can_download());
    }

    #[test]
    fn test_can_download_complete_with_segments() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_translation();
        demo.set_detected_language("es", "Spanish", 90.0);
        demo.add_segment(0.0, 2.0, "Hola", "Hello");
        demo.complete();
        assert!(demo.can_download());
    }

    #[test]
    fn test_can_download_complete_without_segments() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_translation();
        demo.set_detected_language("es", "Spanish", 90.0);
        demo.complete();
        assert!(!demo.can_download());
    }

    // =========================================================================
    // Toggle Tests
    // =========================================================================

    #[test]
    fn test_toggle_timestamps() {
        let mut demo = UploadTranslationDemo::new();
        assert!(!demo.is_show_timestamps());
        demo.toggle_timestamps();
        assert!(demo.is_show_timestamps());
        demo.toggle_timestamps();
        assert!(!demo.is_show_timestamps());
    }

    // =========================================================================
    // Export Tests
    // =========================================================================

    #[test]
    fn test_export_txt() {
        let mut demo = UploadTranslationDemo::new();
        demo.add_segment(0.0, 2.0, "Hola", "Hello");
        demo.add_segment(2.0, 4.0, "Mundo", "World");
        let txt = demo.export_txt();
        assert!(txt.contains("Hello"));
        assert!(txt.contains("World"));
    }

    #[test]
    fn test_export_vtt() {
        let mut demo = UploadTranslationDemo::new();
        demo.add_segment(0.0, 2.0, "Hola", "Hello");
        let vtt = demo.export_vtt();
        assert!(vtt.starts_with("WEBVTT"));
        assert!(vtt.contains("Hello"));
    }

    // =========================================================================
    // Clear Tests
    // =========================================================================

    #[test]
    fn test_clear_resets_all() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_translation();
        demo.set_detected_language("es", "Spanish", 90.0);
        demo.add_segment(0.0, 2.0, "Hola", "Hello");
        demo.complete();

        demo.clear();

        assert_eq!(demo.state(), DemoState::Idle);
        assert!(demo.file_info().is_none());
        assert!(demo.detected_language().is_none());
        assert_eq!(demo.segments_count(), 0);
    }

    // =========================================================================
    // Error Tests
    // =========================================================================

    #[test]
    fn test_error_message_initially_none() {
        let demo = UploadTranslationDemo::new();
        assert!(demo.error_message().is_none());
    }

    // =========================================================================
    // Default Trait Tests
    // =========================================================================

    #[test]
    fn test_default_demo() {
        let demo = UploadTranslationDemo::default();
        assert_eq!(demo.state(), DemoState::Idle);
    }

    #[test]
    fn test_default_state() {
        let state = DemoState::default();
        assert_eq!(state, DemoState::Idle);
    }

    // =========================================================================
    // DemoState Trait Tests
    // =========================================================================

    #[test]
    fn test_state_debug() {
        let state = DemoState::Translating;
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("Translating"));
    }

    #[test]
    fn test_state_clone() {
        let state = DemoState::Complete;
        let cloned = state.clone();
        assert_eq!(state, cloned);
    }

    #[test]
    fn test_state_copy() {
        let state = DemoState::Error;
        let copied = state;
        assert_eq!(state, copied);
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_format_language_pair_various() {
        assert_eq!(format_language_pair("French", "English"), "French → English");
        assert_eq!(format_language_pair("Japanese", "English"), "Japanese → English");
    }
}
