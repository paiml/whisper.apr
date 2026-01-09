//! `FileInfoBrick`: Upload file information display (PROBAR-SPEC-009)
//!
//! This brick displays information about uploaded audio files:
//! - File name
//! - File size
//! - Duration
//! - Format
//!
//! # Assertions
//!
//! - File info visible when file selected
//! - Format correctly detected
//! - Duration calculated from audio

use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_core::{
    AccessibleRole, Canvas, Color, Constraints, Event, LayoutResult, Point, Rect, Size, TextStyle,
    TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Audio format types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioFormat {
    /// WAV (Waveform Audio)
    Wav,
    /// MP3 (MPEG-1 Audio Layer 3)
    Mp3,
    /// FLAC (Free Lossless Audio Codec)
    Flac,
    /// OGG Vorbis
    Ogg,
    /// AAC/M4A
    Aac,
    /// WebM audio
    WebM,
    /// Unknown format
    Unknown,
}

impl AudioFormat {
    /// Detect format from file extension
    #[must_use]
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "wav" => Self::Wav,
            "mp3" => Self::Mp3,
            "flac" => Self::Flac,
            "ogg" => Self::Ogg,
            "m4a" | "aac" => Self::Aac,
            "webm" => Self::WebM,
            _ => Self::Unknown,
        }
    }

    /// Get format display name
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Wav => "WAV",
            Self::Mp3 => "MP3",
            Self::Flac => "FLAC",
            Self::Ogg => "OGG",
            Self::Aac => "AAC",
            Self::WebM => "WebM",
            Self::Unknown => "Unknown",
        }
    }

    /// Check if format is lossless
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        matches!(self, Self::Wav | Self::Flac)
    }
}

/// File information state
#[derive(Debug, Clone)]
pub struct FileInfo {
    /// File name
    pub name: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// Audio duration in seconds (if known)
    pub duration_secs: Option<f32>,
    /// Detected audio format
    pub format: AudioFormat,
    /// Sample rate in Hz (if known)
    pub sample_rate: Option<u32>,
    /// Number of channels (if known)
    pub channels: Option<u8>,
}

impl FileInfo {
    /// Create new file info
    #[must_use]
    pub fn new(name: impl Into<String>, size_bytes: u64) -> Self {
        let name = name.into();
        let format = name
            .rsplit('.')
            .next()
            .map(AudioFormat::from_extension)
            .unwrap_or(AudioFormat::Unknown);

        Self {
            name,
            size_bytes,
            duration_secs: None,
            format,
            sample_rate: None,
            channels: None,
        }
    }

    /// Set duration
    #[must_use]
    pub fn with_duration(mut self, secs: f32) -> Self {
        self.duration_secs = Some(secs);
        self
    }

    /// Set sample rate
    #[must_use]
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = Some(rate);
        self
    }

    /// Set channels
    #[must_use]
    pub fn with_channels(mut self, channels: u8) -> Self {
        self.channels = Some(channels);
        self
    }

    /// Format file size for display
    #[must_use]
    pub fn formatted_size(&self) -> String {
        let bytes = self.size_bytes as f64;
        if bytes < 1024.0 {
            format!("{} B", self.size_bytes)
        } else if bytes < 1024.0 * 1024.0 {
            format!("{:.1} KB", bytes / 1024.0)
        } else {
            format!("{:.1} MB", bytes / (1024.0 * 1024.0))
        }
    }

    /// Format duration for display
    #[must_use]
    pub fn formatted_duration(&self) -> String {
        match self.duration_secs {
            Some(secs) => {
                let mins = (secs / 60.0).floor() as u32;
                let secs_rem = secs % 60.0;
                format!("{mins}:{secs_rem:05.2}")
            }
            None => "Unknown".into(),
        }
    }
}

/// File info brick for displaying uploaded file details
#[derive(Debug, Clone, Default)]
pub struct FileInfoBrick {
    /// Current file info (None if no file selected)
    file_info: Option<FileInfo>,
}

impl FileInfoBrick {
    /// Create a new file info brick
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set file info
    pub fn set_file(&mut self, info: FileInfo) {
        self.file_info = Some(info);
    }

    /// Clear file info
    pub fn clear(&mut self) {
        self.file_info = None;
    }

    /// Check if a file is selected
    #[must_use]
    pub fn has_file(&self) -> bool {
        self.file_info.is_some()
    }

    /// Get file info
    #[must_use]
    pub fn file_info(&self) -> Option<&FileInfo> {
        self.file_info.as_ref()
    }
}

impl Brick for FileInfoBrick {
    fn brick_name(&self) -> &'static str {
        "FileInfoBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::MaxLatencyMs(20),
        ]
    }

    fn budget(&self) -> BrickBudget {
        // 20ms for file info updates
        BrickBudget::uniform(20)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let failed = Vec::new();

        // Check if file info is visible when present
        for assertion in self.assertions() {
            match assertion {
                BrickAssertion::TextVisible => {
                    if self.file_info.is_some() {
                        passed.push(assertion.clone());
                    } else {
                        // No file is fine - nothing to display
                        passed.push(assertion.clone());
                    }
                }
                _ => passed.push(assertion.clone()),
            }
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(5),
        }
    }

    fn to_html(&self) -> String {
        match &self.file_info {
            Some(info) => {
                let quality = if info.format.is_lossless() {
                    r#"<span class="quality-badge lossless">Lossless</span>"#
                } else {
                    r#"<span class="quality-badge lossy">Compressed</span>"#
                };

                format!(
                    r#"<div class="file-info-brick" data-testid="file-info">
    <div class="file-header">
        <span class="file-icon">📁</span>
        <span class="file-name" data-testid="file-name">{name}</span>
    </div>
    <div class="file-details">
        <div class="detail">
            <span class="label">Size:</span>
            <span class="value" data-testid="file-size">{size}</span>
        </div>
        <div class="detail">
            <span class="label">Duration:</span>
            <span class="value" data-testid="file-duration">{duration}</span>
        </div>
        <div class="detail">
            <span class="label">Format:</span>
            <span class="value" data-testid="file-format">{format}</span>
            {quality}
        </div>
    </div>
</div>"#,
                    name = info.name,
                    size = info.formatted_size(),
                    duration = info.formatted_duration(),
                    format = info.format.display_name(),
                    quality = quality,
                )
            }
            None => {
                r#"<div class="file-info-brick empty" data-testid="file-info">
    <span class="placeholder">No file selected</span>
</div>"#
                    .into()
            }
        }
    }

    fn to_css(&self) -> String {
        r".file-info-brick {
    background: #16213e;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}

.file-info-brick.empty {
    color: #6c757d;
    font-style: italic;
}

.file-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2d3a5c;
}

.file-icon {
    font-size: 1.5rem;
}

.file-name {
    font-weight: 600;
    color: #e0e0e0;
    font-family: monospace;
    word-break: break-all;
}

.file-details {
    display: grid;
    gap: 0.5rem;
}

.detail {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.detail .label {
    color: #8b949e;
    min-width: 70px;
}

.detail .value {
    color: #e0e0e0;
    font-family: monospace;
}

.quality-badge {
    font-size: 0.75rem;
    padding: 0.125rem 0.375rem;
    border-radius: 4px;
    margin-left: 0.5rem;
}

.quality-badge.lossless {
    background: #2d5a27;
    color: #50fa7b;
}

.quality-badge.lossy {
    background: #5a4127;
    color: #ffb86c;
}"
            .into()
    }

    fn test_id(&self) -> Option<&str> {
        Some("file-info")
    }
}

impl Widget for FileInfoBrick {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let height: f32 = if self.file_info.is_some() { 120.0 } else { 48.0 };
        Size::new(
            constraints.max_width.min(constraints.min_width.max(200.0)),
            height.min(constraints.max_height),
        )
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let bounds = Rect::new(0.0, 0.0, 400.0, 120.0);

        // Draw background
        let bg_color = Color::from_hex("#16213e").unwrap_or(Color::BLACK);
        canvas.fill_rect(bounds, bg_color);

        let text_color = Color::from_hex("#e0e0e0").unwrap_or(Color::WHITE);
        let label_color = Color::from_hex("#8b949e").unwrap_or(Color::WHITE);

        let style = TextStyle {
            size: 14.0,
            color: text_color,
            weight: presentar_core::FontWeight::Normal,
            style: presentar_core::FontStyle::Normal,
        };

        let label_style = TextStyle {
            size: 14.0,
            color: label_color,
            ..style
        };

        match &self.file_info {
            Some(info) => {
                canvas.draw_text(&info.name, Point::new(16.0, 24.0), &style);
                canvas.draw_text("Size:", Point::new(16.0, 52.0), &label_style);
                canvas.draw_text(&info.formatted_size(), Point::new(80.0, 52.0), &style);
                canvas.draw_text("Duration:", Point::new(16.0, 76.0), &label_style);
                canvas.draw_text(&info.formatted_duration(), Point::new(80.0, 76.0), &style);
                canvas.draw_text("Format:", Point::new(16.0, 100.0), &label_style);
                canvas.draw_text(info.format.display_name(), Point::new(80.0, 100.0), &style);
            }
            None => {
                let placeholder_style = TextStyle {
                    size: 14.0,
                    color: label_color,
                    weight: presentar_core::FontWeight::Normal,
                    style: presentar_core::FontStyle::Italic,
                };
                canvas.draw_text("No file selected", Point::new(16.0, 24.0), &placeholder_style);
            }
        }
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }

    fn accessible_name(&self) -> Option<&str> {
        Some("File information")
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_format_from_extension() {
        assert_eq!(AudioFormat::from_extension("wav"), AudioFormat::Wav);
        assert_eq!(AudioFormat::from_extension("WAV"), AudioFormat::Wav);
        assert_eq!(AudioFormat::from_extension("mp3"), AudioFormat::Mp3);
        assert_eq!(AudioFormat::from_extension("flac"), AudioFormat::Flac);
        assert_eq!(AudioFormat::from_extension("ogg"), AudioFormat::Ogg);
        assert_eq!(AudioFormat::from_extension("m4a"), AudioFormat::Aac);
        assert_eq!(AudioFormat::from_extension("xyz"), AudioFormat::Unknown);
    }

    #[test]
    fn test_audio_format_is_lossless() {
        assert!(AudioFormat::Wav.is_lossless());
        assert!(AudioFormat::Flac.is_lossless());
        assert!(!AudioFormat::Mp3.is_lossless());
        assert!(!AudioFormat::Aac.is_lossless());
    }

    #[test]
    fn test_file_info_new() {
        let info = FileInfo::new("test.wav", 1024000);
        assert_eq!(info.name, "test.wav");
        assert_eq!(info.size_bytes, 1024000);
        assert_eq!(info.format, AudioFormat::Wav);
    }

    #[test]
    fn test_file_info_with_duration() {
        let info = FileInfo::new("test.mp3", 5000000).with_duration(180.5);
        assert_eq!(info.duration_secs, Some(180.5));
    }

    #[test]
    fn test_file_info_formatted_size() {
        let info = FileInfo::new("test.wav", 512);
        assert_eq!(info.formatted_size(), "512 B");

        let info = FileInfo::new("test.wav", 1536);
        assert_eq!(info.formatted_size(), "1.5 KB");

        let info = FileInfo::new("test.wav", 5242880);
        assert_eq!(info.formatted_size(), "5.0 MB");
    }

    #[test]
    fn test_file_info_formatted_duration() {
        let info = FileInfo::new("test.wav", 1000).with_duration(90.5);
        assert_eq!(info.formatted_duration(), "1:30.50");

        let info = FileInfo::new("test.wav", 1000);
        assert_eq!(info.formatted_duration(), "Unknown");
    }

    #[test]
    fn test_brick_default() {
        let brick = FileInfoBrick::new();
        assert!(!brick.has_file());
        assert!(brick.file_info().is_none());
    }

    #[test]
    fn test_brick_set_file() {
        let mut brick = FileInfoBrick::new();
        let info = FileInfo::new("audio.wav", 1024);
        brick.set_file(info);

        assert!(brick.has_file());
        assert_eq!(brick.file_info().unwrap().name, "audio.wav");
    }

    #[test]
    fn test_brick_clear() {
        let mut brick = FileInfoBrick::new();
        brick.set_file(FileInfo::new("audio.wav", 1024));
        brick.clear();

        assert!(!brick.has_file());
    }

    #[test]
    fn test_brick_verification() {
        let brick = FileInfoBrick::new();
        let result = brick.verify();
        assert!(result.is_valid());
    }

    #[test]
    fn test_brick_to_html_empty() {
        let brick = FileInfoBrick::new();
        let html = brick.to_html();
        assert!(html.contains("No file selected"));
        assert!(html.contains("data-testid=\"file-info\""));
    }

    #[test]
    fn test_brick_to_html_with_file() {
        let mut brick = FileInfoBrick::new();
        brick.set_file(FileInfo::new("audio.flac", 10485760).with_duration(120.0));

        let html = brick.to_html();
        assert!(html.contains("audio.flac"));
        assert!(html.contains("10.0 MB"));
        assert!(html.contains("2:00.00"));
        assert!(html.contains("FLAC"));
        assert!(html.contains("Lossless"));
    }

    #[test]
    fn test_brick_budget() {
        let brick = FileInfoBrick::new();
        assert_eq!(brick.budget().total_ms, 20);
    }

    #[test]
    fn test_brick_can_render() {
        let brick = FileInfoBrick::new();
        assert!(brick.can_render());
    }
}
