//! WAPR-DEMO-002: Audio/Video Upload Transcription Demo
//!
//! Pure Rust WASM demo for file-based speech-to-text transcription.
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
}
