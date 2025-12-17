//! WAPR-DEMO-002: Audio/Video Upload Transcription Demo
//!
//! Pure Rust WASM demo for file-based speech-to-text transcription.
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
    /// File selected, ready to transcribe
    FileSelected,
    /// Transcription in progress
    Transcribing,
    /// Transcription complete
    Complete,
    /// Error state
    Error,
}

/// Supported file formats
pub struct SupportedFormat;

impl SupportedFormat {
    /// Audio formats
    pub const AUDIO_EXTENSIONS: &'static [&'static str] = &["wav", "mp3", "ogg", "flac", "m4a"];

    /// Video formats (audio will be extracted)
    pub const VIDEO_EXTENSIONS: &'static [&'static str] = &["mp4", "webm", "mkv", "avi"];

    /// Check if a filename has a supported extension
    #[must_use]
    pub fn is_supported(filename: &str) -> bool {
        let ext = filename
            .rsplit('.')
            .next()
            .map(str::to_lowercase)
            .unwrap_or_default();

        Self::AUDIO_EXTENSIONS.contains(&ext.as_str())
            || Self::VIDEO_EXTENSIONS.contains(&ext.as_str())
    }

    /// Check if file is a video (needs audio extraction)
    #[must_use]
    pub fn is_video(filename: &str) -> bool {
        let ext = filename
            .rsplit('.')
            .next()
            .map(str::to_lowercase)
            .unwrap_or_default();

        Self::VIDEO_EXTENSIONS.contains(&ext.as_str())
    }

    /// Get supported formats string for display
    #[must_use]
    pub fn supported_formats_string() -> String {
        let audio: Vec<_> = Self::AUDIO_EXTENSIONS
            .iter()
            .map(|e| e.to_uppercase())
            .collect();
        let video: Vec<_> = Self::VIDEO_EXTENSIONS
            .iter()
            .map(|e| e.to_uppercase())
            .collect();

        format!("Supported: {}, {}", audio.join(", "), video.join(", "))
    }
}

/// File information
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct FileInfo {
    name: String,
    size_bytes: u64,
    duration_seconds: Option<f64>,
    is_video: bool,
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

    /// Check if audio was extracted from video
    #[must_use]
    pub fn is_video(&self) -> bool {
        self.is_video
    }
}

/// Transcription progress information
#[wasm_bindgen]
#[derive(Debug, Clone, Default)]
pub struct TranscriptionProgress {
    current_chunk: usize,
    total_chunks: usize,
    percent_complete: f32,
    estimated_remaining_seconds: Option<f64>,
}

#[wasm_bindgen]
impl TranscriptionProgress {
    /// Get chunk progress text
    #[must_use]
    pub fn chunk_text(&self) -> String {
        format!("Chunk {}/{}", self.current_chunk, self.total_chunks)
    }

    /// Get percentage
    #[must_use]
    pub fn percent(&self) -> f32 {
        self.percent_complete
    }

    /// Get estimated time remaining
    #[must_use]
    pub fn estimated_time(&self) -> Option<String> {
        self.estimated_remaining_seconds.map(|s| {
            if s < 60.0 {
                format!("{s:.0}s remaining")
            } else {
                format!("{:.0}m remaining", s / 60.0)
            }
        })
    }
}

/// Transcript segment with timestamp
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    start_seconds: f64,
    end_seconds: f64,
    text: String,
}

#[wasm_bindgen]
impl TranscriptSegment {
    /// Get formatted timestamp
    #[must_use]
    pub fn timestamp(&self) -> String {
        format!(
            "[{} - {}]",
            format_timestamp(self.start_seconds),
            format_timestamp(self.end_seconds)
        )
    }

    /// Get text content
    #[must_use]
    pub fn text(&self) -> String {
        self.text.clone()
    }
}

/// Upload transcription demo application
#[wasm_bindgen]
pub struct UploadTranscriptionDemo {
    state: DemoState,
    file_info: Option<FileInfo>,
    progress: TranscriptionProgress,
    segments: Vec<TranscriptSegment>,
    error_message: Option<String>,
    cancelled: bool,
}

#[wasm_bindgen]
impl UploadTranscriptionDemo {
    /// Create a new demo instance
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: DemoState::Idle,
            file_info: None,
            progress: TranscriptionProgress::default(),
            segments: Vec::new(),
            error_message: None,
            cancelled: false,
        }
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> DemoState {
        self.state
    }

    /// Get file info if available
    #[must_use]
    pub fn file_info(&self) -> Option<FileInfo> {
        self.file_info.clone()
    }

    /// Get current progress
    #[must_use]
    pub fn progress(&self) -> TranscriptionProgress {
        self.progress.clone()
    }

    /// Get full transcript text
    #[must_use]
    pub fn transcript_text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get segments count
    #[must_use]
    pub fn segments_count(&self) -> usize {
        self.segments.len()
    }

    /// Get error message if in error state
    #[must_use]
    pub fn error_message(&self) -> Option<String> {
        self.error_message.clone()
    }

    /// Check if transcribe button should be enabled
    #[must_use]
    pub fn can_transcribe(&self) -> bool {
        matches!(self.state, DemoState::FileSelected | DemoState::Complete)
    }

    /// Check if download button should be enabled
    #[must_use]
    pub fn can_download(&self) -> bool {
        self.state == DemoState::Complete && !self.segments.is_empty()
    }

    /// Handle file selection
    ///
    /// # Errors
    ///
    /// This function always returns `Ok(())` - errors are stored in state.
    pub fn on_file_selected(&mut self, name: &str, size_bytes: u64) -> Result<(), JsValue> {
        // Check format
        if !SupportedFormat::is_supported(name) {
            self.error_message = Some(format!(
                "Unsupported format. {}",
                SupportedFormat::supported_formats_string()
            ));
            self.state = DemoState::Error;
            return Ok(());
        }

        // Check file size (100MB limit)
        #[allow(clippy::items_after_statements)]
        const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;
        if size_bytes > MAX_FILE_SIZE {
            self.error_message = Some("File too large. Maximum: 100MB".to_string());
            self.state = DemoState::Error;
            return Ok(());
        }

        self.file_info = Some(FileInfo {
            name: name.to_string(),
            size_bytes,
            duration_seconds: None,
            is_video: SupportedFormat::is_video(name),
        });
        self.state = DemoState::FileSelected;
        self.error_message = None;

        Ok(())
    }

    /// Update file duration after audio analysis
    pub fn set_file_duration(&mut self, duration_seconds: f64) {
        if let Some(ref mut info) = self.file_info {
            info.duration_seconds = Some(duration_seconds);
        }
    }

    /// Start transcription
    ///
    /// # Errors
    ///
    /// Returns error if transcription cannot be started in current state.
    pub fn start_transcription(&mut self) -> Result<(), JsValue> {
        if !self.can_transcribe() {
            return Err(JsValue::from_str("Cannot transcribe in current state"));
        }

        self.state = DemoState::Transcribing;
        self.cancelled = false;
        self.segments.clear();

        // Calculate chunks based on duration
        let duration = self
            .file_info
            .as_ref()
            .and_then(|f| f.duration_seconds)
            .unwrap_or(30.0);
        let total_chunks = calculate_chunks(duration);

        self.progress = TranscriptionProgress {
            current_chunk: 0,
            total_chunks,
            percent_complete: 0.0,
            estimated_remaining_seconds: None,
        };

        Ok(())
    }

    /// Cancel transcription
    pub fn cancel(&mut self) {
        if self.state == DemoState::Transcribing {
            self.cancelled = true;
            self.state = DemoState::FileSelected;
        }
    }

    /// Check if cancelled
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Update progress during transcription
    pub fn update_progress(&mut self, chunk: usize, percent: f32, estimated_seconds: Option<f64>) {
        self.progress.current_chunk = chunk;
        self.progress.percent_complete = percent;
        self.progress.estimated_remaining_seconds = estimated_seconds;
    }

    /// Add a transcription segment
    pub fn add_segment(&mut self, start: f64, end: f64, text: &str) {
        self.segments.push(TranscriptSegment {
            start_seconds: start,
            end_seconds: end,
            text: text.to_string(),
        });
    }

    /// Complete transcription
    pub fn complete(&mut self) {
        self.state = DemoState::Complete;
        self.progress.percent_complete = 100.0;
    }

    /// Export transcript as plain text
    #[must_use]
    pub fn export_txt(&self) -> String {
        self.transcript_text()
    }

    /// Export transcript as SRT subtitle format
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
                    seg.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Export transcript as `WebVTT` format
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
                    seg.text
                )
            })
            .collect();

        format!("WEBVTT\n\n{}", segments.join("\n\n"))
    }

    /// Clear and reset
    pub fn clear(&mut self) {
        self.state = DemoState::Idle;
        self.file_info = None;
        self.progress = TranscriptionProgress::default();
        self.segments.clear();
        self.error_message = None;
        self.cancelled = false;
    }
}

impl Default for UploadTranscriptionDemo {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate number of 30-second chunks for a given duration
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn calculate_chunks(duration_seconds: f64) -> usize {
    const CHUNK_DURATION: f64 = 30.0;
    ((duration_seconds / CHUNK_DURATION).ceil() as usize).max(1)
}

/// Format duration as M:SS or H:MM:SS
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn format_duration(seconds: f64) -> String {
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

/// Format timestamp for SRT format: HH:MM:SS,mmm
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn format_srt_timestamp(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60000;
    let secs = (total_ms % 60000) / 1000;
    let ms = total_ms % 1000;
    format!("{hours:02}:{minutes:02}:{secs:02},{ms:03}")
}

/// Format timestamp for VTT format: HH:MM:SS.mmm
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
    static DEMO: RefCell<Option<Rc<RefCell<UploadTranscriptionDemo>>>> = const { RefCell::new(None) };
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

    let demo = Rc::new(RefCell::new(UploadTranscriptionDemo::new()));
    DEMO.with(|d| *d.borrow_mut() = Some(demo.clone()));

    update_ui(&document, &demo.borrow())?;

    // Transcribe button
    setup_button_listener(&document, "transcribe", {
        let demo = demo.clone();
        move |doc| {
            let _ = demo.borrow_mut().start_transcription();
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

#[allow(clippy::unnecessary_wraps)]
fn update_ui(document: &web_sys::Document, demo: &UploadTranscriptionDemo) -> Result<(), JsValue> {
    if let Some(el) = document.get_element_by_id("transcript_display") {
        el.set_text_content(Some(&demo.transcript_text()));
    }

    // Update file info display
    if let Some(el) = document.get_element_by_id("file_info") {
        if let Some(info) = demo.file_info() {
            el.set_text_content(Some(&format!(
                "{} ({})",
                info.name(),
                info.size_formatted()
            )));
            let _ = el.class_list().add_1("visible");
        } else {
            el.set_text_content(None);
            let _ = el.class_list().remove_1("visible");
        }
    }

    // Update transcribe button
    if let Some(btn) = document.get_element_by_id("transcribe") {
        if demo.can_transcribe() {
            let _ = btn.remove_attribute("disabled");
        } else {
            let _ = btn.set_attribute("disabled", "true");
        }
    }

    // Update download button
    if let Some(btn) = document.get_element_by_id("download_result") {
        if demo.can_download() {
            let _ = btn.remove_attribute("disabled");
        } else {
            let _ = btn.set_attribute("disabled", "true");
        }
    }

    // Update drop zone class
    if let Some(zone) = document.get_element_by_id("drop_zone") {
        if demo.file_info().is_some() {
            let _ = zone.class_list().add_1("file-selected");
        } else {
            let _ = zone.class_list().remove_1("file-selected");
        }
    }

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
    demo: Rc<RefCell<UploadTranscriptionDemo>>,
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
    demo: Rc<RefCell<UploadTranscriptionDemo>>,
) -> Result<(), JsValue> {
    let doc = document.clone();

    // Dragover - prevent default to allow drop
    let dragover = Closure::wrap(Box::new(move |event: web_sys::DragEvent| {
        event.prevent_default();
    }) as Box<dyn Fn(_)>);

    // Drop handler
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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_formats() {
        assert!(SupportedFormat::is_supported("audio.wav"));
        assert!(SupportedFormat::is_supported("audio.MP3"));
        assert!(SupportedFormat::is_supported("video.mp4"));
        assert!(!SupportedFormat::is_supported("doc.pdf"));
    }

    #[test]
    fn test_is_video() {
        assert!(SupportedFormat::is_video("video.mp4"));
        assert!(SupportedFormat::is_video("video.webm"));
        assert!(!SupportedFormat::is_video("audio.wav"));
    }

    #[test]
    fn test_chunk_calculation() {
        assert_eq!(calculate_chunks(30.0), 1);
        assert_eq!(calculate_chunks(60.0), 2);
        assert_eq!(calculate_chunks(90.0), 3);
        assert_eq!(calculate_chunks(91.0), 4);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(65.0), "1:05");
        assert_eq!(format_duration(3661.0), "1:01:01");
        assert_eq!(format_duration(0.0), "0:00");
    }

    #[test]
    fn test_srt_export() {
        let mut demo = UploadTranscriptionDemo::new();
        demo.add_segment(0.0, 2.5, "Hello world");
        demo.add_segment(2.5, 5.0, "How are you");

        let srt = demo.export_srt();
        assert!(srt.contains("00:00:00,000 --> 00:00:02,500"));
        assert!(srt.contains("Hello world"));
    }

    #[test]
    fn test_file_size_limit() {
        let mut demo = UploadTranscriptionDemo::new();
        let result = demo.on_file_selected("huge.wav", 200 * 1024 * 1024);
        assert!(result.is_ok());
        assert_eq!(demo.state(), DemoState::Error);
        assert!(demo.error_message().unwrap().contains("too large"));
    }

    // =========================================================================
    // State Tests
    // =========================================================================

    #[test]
    fn test_initial_state_idle() {
        let demo = UploadTranscriptionDemo::new();
        assert_eq!(demo.state(), DemoState::Idle);
    }

    #[test]
    fn test_state_file_selected() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        assert_eq!(demo.state(), DemoState::FileSelected);
    }

    #[test]
    fn test_state_transcribing() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_transcription();
        assert_eq!(demo.state(), DemoState::Transcribing);
    }

    #[test]
    fn test_state_complete() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_transcription();
        demo.complete();
        assert_eq!(demo.state(), DemoState::Complete);
    }

    // =========================================================================
    // SupportedFormat Tests
    // =========================================================================

    #[test]
    fn test_supported_audio_formats() {
        assert!(SupportedFormat::is_supported("audio.wav"));
        assert!(SupportedFormat::is_supported("audio.mp3"));
        assert!(SupportedFormat::is_supported("audio.ogg"));
        assert!(SupportedFormat::is_supported("audio.flac"));
        assert!(SupportedFormat::is_supported("audio.m4a"));
    }

    #[test]
    fn test_supported_video_formats() {
        assert!(SupportedFormat::is_supported("video.mp4"));
        assert!(SupportedFormat::is_supported("video.webm"));
        assert!(SupportedFormat::is_supported("video.mkv"));
        assert!(SupportedFormat::is_supported("video.avi"));
    }

    #[test]
    fn test_unsupported_formats() {
        assert!(!SupportedFormat::is_supported("doc.pdf"));
        assert!(!SupportedFormat::is_supported("image.png"));
        assert!(!SupportedFormat::is_supported("text.txt"));
    }

    #[test]
    fn test_case_insensitive_formats() {
        assert!(SupportedFormat::is_supported("audio.WAV"));
        assert!(SupportedFormat::is_supported("video.MP4"));
        assert!(SupportedFormat::is_supported("audio.WaV"));
    }

    #[test]
    fn test_supported_formats_string() {
        let formats = SupportedFormat::supported_formats_string();
        assert!(formats.contains("WAV"));
        assert!(formats.contains("MP4"));
    }

    // =========================================================================
    // FileInfo Tests
    // =========================================================================

    #[test]
    fn test_file_info_name() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test_audio.wav", 1024 * 1024);
        let info = demo.file_info().unwrap();
        assert_eq!(info.name(), "test_audio.wav");
    }

    #[test]
    fn test_file_info_size_formatted() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 10 * 1024 * 1024);
        let info = demo.file_info().unwrap();
        assert!(info.size_formatted().contains("10.0 MB"));
    }

    #[test]
    fn test_file_info_duration_initially_none() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let info = demo.file_info().unwrap();
        assert!(info.duration_formatted().is_none());
    }

    #[test]
    fn test_file_info_duration_set() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        demo.set_file_duration(125.5);
        let info = demo.file_info().unwrap();
        assert!(info.duration_formatted().is_some());
    }

    #[test]
    fn test_file_info_is_video() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("video.mp4", 1000);
        let info = demo.file_info().unwrap();
        assert!(info.is_video());
    }

    #[test]
    fn test_file_info_is_not_video() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("audio.wav", 1000);
        let info = demo.file_info().unwrap();
        assert!(!info.is_video());
    }

    // =========================================================================
    // TranscriptionProgress Tests
    // =========================================================================

    #[test]
    fn test_progress_initial() {
        let demo = UploadTranscriptionDemo::new();
        let progress = demo.progress();
        assert_eq!(progress.percent(), 0.0);
    }

    #[test]
    fn test_progress_chunk_text() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_transcription();
        demo.update_progress(1, 50.0, Some(30.0));
        let progress = demo.progress();
        assert!(progress.chunk_text().contains("/"));
    }

    #[test]
    fn test_progress_estimated_time_seconds() {
        let progress = TranscriptionProgress {
            current_chunk: 1,
            total_chunks: 5,
            percent_complete: 20.0,
            estimated_remaining_seconds: Some(45.0),
        };
        let estimated = progress.estimated_time().unwrap();
        assert!(estimated.contains("s remaining"));
    }

    #[test]
    fn test_progress_estimated_time_minutes() {
        let progress = TranscriptionProgress {
            current_chunk: 1,
            total_chunks: 5,
            percent_complete: 20.0,
            estimated_remaining_seconds: Some(120.0),
        };
        let estimated = progress.estimated_time().unwrap();
        assert!(estimated.contains("m remaining"));
    }

    #[test]
    fn test_progress_estimated_time_none() {
        let progress = TranscriptionProgress::default();
        assert!(progress.estimated_time().is_none());
    }

    // =========================================================================
    // Transcription Tests
    // =========================================================================

    #[test]
    fn test_transcript_text_empty_initially() {
        let demo = UploadTranscriptionDemo::new();
        assert!(demo.transcript_text().is_empty());
    }

    #[test]
    fn test_transcript_text_with_segments() {
        let mut demo = UploadTranscriptionDemo::new();
        demo.add_segment(0.0, 2.0, "Hello");
        demo.add_segment(2.0, 4.0, "World");
        let text = demo.transcript_text();
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }

    #[test]
    fn test_segments_count() {
        let mut demo = UploadTranscriptionDemo::new();
        assert_eq!(demo.segments_count(), 0);
        demo.add_segment(0.0, 2.0, "Hello");
        assert_eq!(demo.segments_count(), 1);
        demo.add_segment(2.0, 4.0, "World");
        assert_eq!(demo.segments_count(), 2);
    }

    // =========================================================================
    // Button State Tests
    // =========================================================================

    #[test]
    fn test_can_transcribe_initial() {
        let demo = UploadTranscriptionDemo::new();
        assert!(!demo.can_transcribe());
    }

    #[test]
    fn test_can_transcribe_file_selected() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        assert!(demo.can_transcribe());
    }

    #[test]
    fn test_can_transcribe_complete() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_transcription();
        demo.complete();
        assert!(demo.can_transcribe());
    }

    #[test]
    fn test_can_download_initial() {
        let demo = UploadTranscriptionDemo::new();
        assert!(!demo.can_download());
    }

    #[test]
    fn test_can_download_complete_with_segments() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_transcription();
        demo.add_segment(0.0, 2.0, "Hello");
        demo.complete();
        assert!(demo.can_download());
    }

    #[test]
    fn test_can_download_complete_without_segments() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_transcription();
        demo.complete();
        assert!(!demo.can_download());
    }

    // =========================================================================
    // Cancel Tests
    // =========================================================================

    #[test]
    fn test_is_cancelled_initial() {
        let demo = UploadTranscriptionDemo::new();
        assert!(!demo.is_cancelled());
    }

    #[test]
    fn test_cancel_during_transcription() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_transcription();
        demo.cancel();
        assert!(demo.is_cancelled());
        assert_eq!(demo.state(), DemoState::FileSelected);
    }

    #[test]
    fn test_cancel_not_during_transcription() {
        let mut demo = UploadTranscriptionDemo::new();
        demo.cancel();
        assert!(!demo.is_cancelled());
    }

    // =========================================================================
    // Export Tests
    // =========================================================================

    #[test]
    fn test_export_txt() {
        let mut demo = UploadTranscriptionDemo::new();
        demo.add_segment(0.0, 2.0, "Hello");
        demo.add_segment(2.0, 4.0, "World");
        let txt = demo.export_txt();
        assert!(txt.contains("Hello"));
        assert!(txt.contains("World"));
    }

    #[test]
    fn test_export_vtt() {
        let mut demo = UploadTranscriptionDemo::new();
        demo.add_segment(0.0, 2.0, "Hello world");
        let vtt = demo.export_vtt();
        assert!(vtt.starts_with("WEBVTT"));
        assert!(vtt.contains("Hello world"));
    }

    // =========================================================================
    // Clear Tests
    // =========================================================================

    #[test]
    fn test_clear_resets_all() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_transcription();
        demo.add_segment(0.0, 2.0, "Hello");
        demo.complete();

        demo.clear();

        assert_eq!(demo.state(), DemoState::Idle);
        assert!(demo.file_info().is_none());
        assert_eq!(demo.segments_count(), 0);
        assert!(!demo.is_cancelled());
    }

    // =========================================================================
    // Error Tests
    // =========================================================================

    #[test]
    fn test_unsupported_format_error() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("document.pdf", 1000);
        assert_eq!(demo.state(), DemoState::Error);
        assert!(demo.error_message().is_some());
    }

    #[test]
    fn test_error_message_initially_none() {
        let demo = UploadTranscriptionDemo::new();
        assert!(demo.error_message().is_none());
    }

    // =========================================================================
    // TranscriptSegment Tests
    // =========================================================================

    #[test]
    fn test_segment_timestamp() {
        let mut demo = UploadTranscriptionDemo::new();
        demo.add_segment(0.0, 2.5, "Hello");
        // Access the segment through export
        let srt = demo.export_srt();
        assert!(srt.contains("00:00:00,000"));
        assert!(srt.contains("00:00:02,500"));
    }

    #[test]
    fn test_segment_text_getter() {
        let segment = TranscriptSegment {
            start_seconds: 0.0,
            end_seconds: 5.0,
            text: "Test text content".to_string(),
        };
        assert_eq!(segment.text(), "Test text content");
    }

    #[test]
    fn test_segment_timestamp_format() {
        let segment = TranscriptSegment {
            start_seconds: 61.5,
            end_seconds: 125.123,
            text: "Test".to_string(),
        };
        let ts = segment.timestamp();
        assert!(ts.contains("01:01.500"));
        assert!(ts.contains("02:05.123"));
    }

    #[test]
    fn test_segment_clone() {
        let segment = TranscriptSegment {
            start_seconds: 1.0,
            end_seconds: 2.0,
            text: "Cloned".to_string(),
        };
        let cloned = segment.clone();
        assert_eq!(segment.text(), cloned.text());
    }

    #[test]
    fn test_segment_debug() {
        let segment = TranscriptSegment {
            start_seconds: 0.0,
            end_seconds: 1.0,
            text: "Debug".to_string(),
        };
        let debug_str = format!("{:?}", segment);
        assert!(debug_str.contains("TranscriptSegment"));
    }

    // =========================================================================
    // Default Trait Tests
    // =========================================================================

    #[test]
    fn test_default_demo() {
        let demo = UploadTranscriptionDemo::default();
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
        let state = DemoState::Transcribing;
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("Transcribing"));
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
    // Progress Default Tests
    // =========================================================================

    #[test]
    fn test_progress_default() {
        let progress = TranscriptionProgress::default();
        assert_eq!(progress.percent(), 0.0);
    }

    // =========================================================================
    // FileInfo Additional Tests
    // =========================================================================

    #[test]
    fn test_file_info_clone() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let info = demo.file_info().unwrap();
        let cloned = info.clone();
        assert_eq!(info.name(), cloned.name());
    }

    #[test]
    fn test_file_info_debug() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let info = demo.file_info().unwrap();
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("FileInfo"));
    }

    #[test]
    fn test_file_info_duration_formatted_with_hours() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        demo.set_file_duration(3661.0); // 1 hour, 1 minute, 1 second
        let info = demo.file_info().unwrap();
        let formatted = info.duration_formatted().unwrap();
        assert!(formatted.contains("1:01:01"));
    }

    // =========================================================================
    // TranscriptionProgress Additional Tests
    // =========================================================================

    #[test]
    fn test_progress_clone() {
        let progress = TranscriptionProgress {
            current_chunk: 2,
            total_chunks: 5,
            percent_complete: 40.0,
            estimated_remaining_seconds: Some(60.0),
        };
        let cloned = progress.clone();
        assert_eq!(progress.percent(), cloned.percent());
    }

    #[test]
    fn test_progress_debug() {
        let progress = TranscriptionProgress::default();
        let debug_str = format!("{:?}", progress);
        assert!(debug_str.contains("TranscriptionProgress"));
    }

    // =========================================================================
    // Helper Functions Tests
    // =========================================================================

    #[test]
    fn test_format_duration_with_hours() {
        assert_eq!(format_duration(7325.0), "2:02:05");
    }

    #[test]
    fn test_format_duration_exact_hour() {
        assert_eq!(format_duration(3600.0), "1:00:00");
    }

    #[test]
    fn test_calculate_chunks_boundary() {
        assert_eq!(calculate_chunks(30.0), 1);
        assert_eq!(calculate_chunks(30.1), 2);
        assert_eq!(calculate_chunks(0.0), 1);
        assert_eq!(calculate_chunks(1.0), 1);
    }

    // =========================================================================
    // Start Transcription Edge Cases
    // =========================================================================

    // Note: test_start_transcription_no_file omitted because JsValue::from_str
    // panics on non-WASM targets. This path is tested via browser tests.

    #[test]
    fn test_start_transcription_clears_previous() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        let _ = demo.start_transcription();
        demo.add_segment(0.0, 1.0, "First");
        demo.complete();

        // Start again
        let _ = demo.start_transcription();
        assert_eq!(demo.segments_count(), 0);
        assert!(!demo.is_cancelled());
    }

    #[test]
    fn test_start_transcription_uses_file_duration() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1000);
        demo.set_file_duration(90.0); // 3 chunks
        let _ = demo.start_transcription();
        let progress = demo.progress();
        assert!(progress.chunk_text().contains("/3"));
    }

    // =========================================================================
    // Export Additional Tests
    // =========================================================================

    #[test]
    fn test_export_srt_multiple_segments() {
        let mut demo = UploadTranscriptionDemo::new();
        demo.add_segment(0.0, 2.0, "First");
        demo.add_segment(2.0, 4.0, "Second");
        demo.add_segment(4.0, 6.0, "Third");
        let srt = demo.export_srt();
        assert!(srt.contains("1\n"));
        assert!(srt.contains("2\n"));
        assert!(srt.contains("3\n"));
        assert!(srt.contains("First"));
        assert!(srt.contains("Second"));
        assert!(srt.contains("Third"));
    }

    #[test]
    fn test_export_vtt_multiple_segments() {
        let mut demo = UploadTranscriptionDemo::new();
        demo.add_segment(0.0, 2.0, "First");
        demo.add_segment(2.0, 4.0, "Second");
        let vtt = demo.export_vtt();
        assert!(vtt.starts_with("WEBVTT"));
        assert!(vtt.contains("First"));
        assert!(vtt.contains("Second"));
        assert!(vtt.contains("-->"));
    }

    #[test]
    fn test_export_vtt_timestamp_format() {
        let mut demo = UploadTranscriptionDemo::new();
        demo.add_segment(3661.123, 3665.456, "Test");
        let vtt = demo.export_vtt();
        // VTT format: HH:MM:SS.mmm
        assert!(vtt.contains("01:01:01.123"));
        assert!(vtt.contains("01:01:05.456"));
    }

    #[test]
    fn test_export_srt_timestamp_format() {
        let mut demo = UploadTranscriptionDemo::new();
        demo.add_segment(3661.123, 3665.456, "Test");
        let srt = demo.export_srt();
        // SRT format: HH:MM:SS,mmm
        assert!(srt.contains("01:01:01,123"));
        assert!(srt.contains("01:01:05,456"));
    }

    // =========================================================================
    // File Validation Additional Tests
    // =========================================================================

    #[test]
    fn test_file_selection_resets_error() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("bad.pdf", 1000);
        assert_eq!(demo.state(), DemoState::Error);

        let _ = demo.on_file_selected("good.wav", 1000);
        assert_eq!(demo.state(), DemoState::FileSelected);
        assert!(demo.error_message().is_none());
    }

    #[test]
    fn test_case_insensitive_extension() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.WAV", 1000);
        assert_eq!(demo.state(), DemoState::FileSelected);

        demo.clear();
        let _ = demo.on_file_selected("test.Mp3", 1000);
        assert_eq!(demo.state(), DemoState::FileSelected);
    }

    #[test]
    fn test_file_no_extension() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("noextension", 1000);
        assert_eq!(demo.state(), DemoState::Error);
    }

    // =========================================================================
    // UploadTranscriptionDemo Clone/Debug Tests
    // =========================================================================

    #[test]
    fn test_demo_file_info_none_initially() {
        let demo = UploadTranscriptionDemo::new();
        assert!(demo.file_info().is_none());
    }
}
