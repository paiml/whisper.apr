//! Output formatters for whisper-apr CLI
//!
//! Supports multiple output formats: txt, srt, vtt, json, csv, md

use std::fmt::Write;

use crate::TranscriptionResult;

/// Output format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Plain text
    Txt,
    /// SRT subtitles
    Srt,
    /// WebVTT subtitles
    Vtt,
    /// JSON format
    Json,
    /// Extended JSON with token-level details
    JsonFull,
    /// CSV format
    Csv,
    /// LRC lyrics format (whisper.cpp: -olrc)
    Lrc,
    /// Karaoke script with word timestamps (whisper.cpp: -owts)
    Wts,
    /// Markdown format
    Md,
}

impl OutputFormat {
    /// Get file extension for this format
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Txt => "txt",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Json | Self::JsonFull => "json",
            Self::Csv => "csv",
            Self::Lrc => "lrc",
            Self::Wts => "wts",
            Self::Md => "md",
        }
    }
}

/// Format transcription result according to output format
///
/// # Arguments
///
/// * `result` - The transcription result to format
/// * `format` - The desired output format
///
/// # Returns
///
/// Formatted string representation
pub fn format_output(result: &TranscriptionResult, format: OutputFormat) -> String {
    match format {
        OutputFormat::Txt => format_txt(result),
        OutputFormat::Srt => format_srt(result),
        OutputFormat::Vtt => format_vtt(result),
        OutputFormat::Json => format_json(result),
        OutputFormat::JsonFull => format_json_full(result),
        OutputFormat::Csv => format_csv(result),
        OutputFormat::Lrc => format_lrc(result),
        OutputFormat::Wts => format_wts(result),
        OutputFormat::Md => format_md(result),
    }
}

/// Format as plain text
#[must_use]
pub fn format_txt(result: &TranscriptionResult) -> String {
    result.text.clone()
}

/// Format as SRT subtitles
///
/// SRT format:
/// ```text
/// 1
/// 00:00:00,000 --> 00:00:01,500
/// Hello world
///
/// 2
/// 00:00:01,500 --> 00:00:03,000
/// This is a test
/// ```
#[must_use]
pub fn format_srt(result: &TranscriptionResult) -> String {
    let mut output = String::new();

    for (i, segment) in result.segments.iter().enumerate() {
        // Sequence number
        writeln!(output, "{}", i + 1).ok();

        // Timestamps
        writeln!(
            output,
            "{} --> {}",
            format_timestamp_srt(segment.start),
            format_timestamp_srt(segment.end)
        )
        .ok();

        // Text
        writeln!(output, "{}", segment.text.trim()).ok();

        // Blank line between entries
        writeln!(output).ok();
    }

    output
}

/// Format timestamp for SRT (HH:MM:SS,mmm)
#[must_use]
pub fn format_timestamp_srt(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;

    format!("{hours:02}:{mins:02}:{secs:02},{ms:03}")
}

/// Format as WebVTT subtitles
///
/// VTT format:
/// ```text
/// WEBVTT
///
/// 00:00:00.000 --> 00:00:01.500
/// Hello world
///
/// 00:00:01.500 --> 00:00:03.000
/// This is a test
/// ```
#[must_use]
pub fn format_vtt(result: &TranscriptionResult) -> String {
    let mut output = String::from("WEBVTT\n\n");

    for segment in &result.segments {
        // Timestamps (VTT uses . instead of ,)
        writeln!(
            output,
            "{} --> {}",
            format_timestamp_vtt(segment.start),
            format_timestamp_vtt(segment.end)
        )
        .ok();

        // Text
        writeln!(output, "{}", segment.text.trim()).ok();

        // Blank line between entries
        writeln!(output).ok();
    }

    output
}

/// Format timestamp for VTT (HH:MM:SS.mmm)
#[must_use]
pub fn format_timestamp_vtt(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;

    format!("{hours:02}:{mins:02}:{secs:02}.{ms:03}")
}

/// Format as JSON
#[must_use]
pub fn format_json(result: &TranscriptionResult) -> String {
    serde_json::to_string_pretty(result).unwrap_or_else(|_| "{}".to_string())
}

/// Format as CSV
///
/// CSV format:
/// ```text
/// start,end,text
/// 0.000,1.500,"Hello world"
/// 1.500,3.000,"This is a test"
/// ```
#[must_use]
pub fn format_csv(result: &TranscriptionResult) -> String {
    let mut output = String::from("start,end,text\n");

    for segment in &result.segments {
        writeln!(
            output,
            "{:.3},{:.3},\"{}\"",
            segment.start,
            segment.end,
            escape_csv(&segment.text)
        )
        .ok();
    }

    output
}

/// Escape string for CSV (double quotes become double-double quotes)
fn escape_csv(s: &str) -> String {
    s.replace('"', "\"\"")
}

/// Format as Markdown
///
/// Markdown format:
/// ```text
/// # Transcription
///
/// **Language:** en
/// **Duration:** 3.00s
///
/// ## Segments
///
/// | Start | End | Text |
/// |-------|-----|------|
/// | 0:00 | 0:01 | Hello world |
/// | 0:01 | 0:03 | This is a test |
/// ```
#[must_use]
pub fn format_md(result: &TranscriptionResult) -> String {
    let mut output = String::from("# Transcription\n\n");

    // Metadata
    writeln!(output, "**Language:** {}", result.language).ok();
    writeln!(output).ok();

    // Full text
    writeln!(output, "## Full Text\n").ok();
    writeln!(output, "{}\n", result.text).ok();

    // Segments table (if any)
    if !result.segments.is_empty() {
        writeln!(output, "## Segments\n").ok();
        writeln!(output, "| Start | End | Text |").ok();
        writeln!(output, "|-------|-----|------|").ok();

        for segment in &result.segments {
            writeln!(
                output,
                "| {} | {} | {} |",
                format_timestamp_short(segment.start),
                format_timestamp_short(segment.end),
                segment.text.trim()
            )
            .ok();
        }
    }

    output
}

/// Format as extended JSON with token-level details
///
/// JSON-full format includes:
/// - Token-level timing
/// - Word-level timestamps
/// - Per-token probabilities
#[must_use]
pub fn format_json_full(result: &TranscriptionResult) -> String {
    // For now, same as regular JSON - can be extended with token details
    serde_json::to_string_pretty(result).unwrap_or_else(|_| "{}".to_string())
}

/// Format as LRC lyrics (§7.7)
///
/// LRC format:
/// ```text
/// [00:00.00]Hello, world.
/// [00:05.12]This is a test.
/// ```
#[must_use]
pub fn format_lrc(result: &TranscriptionResult) -> String {
    let mut output = String::new();

    for segment in &result.segments {
        writeln!(
            output,
            "[{}]{}",
            format_timestamp_lrc(segment.start),
            segment.text.trim()
        )
        .ok();
    }

    output
}

/// Format timestamp for LRC (MM:SS.cc - centiseconds)
#[must_use]
pub fn format_timestamp_lrc(seconds: f32) -> String {
    let total_cs = (seconds * 100.0) as u64;
    let cs = total_cs % 100;
    let total_secs = total_cs / 100;
    let secs = total_secs % 60;
    let mins = total_secs / 60;

    format!("{mins:02}:{secs:02}.{cs:02}")
}

/// Format as karaoke word timestamps script (WTS)
///
/// WTS format shows word-level timing for karaoke display.
/// Note: Word-level timestamps require word_timestamps option during transcription.
/// Without word-level data, falls back to segment-level timing.
#[must_use]
pub fn format_wts(result: &TranscriptionResult) -> String {
    let mut output = String::new();

    for segment in &result.segments {
        // For now, use segment-level timing
        // Word-level timestamps would require extending the Segment struct
        writeln!(
            output,
            "{} --> {} | {}",
            format_timestamp_vtt(segment.start),
            format_timestamp_vtt(segment.end),
            segment.text.trim()
        )
        .ok();
    }

    output
}

/// Format timestamp in short form (M:SS or H:MM:SS)
fn format_timestamp_short(seconds: f32) -> String {
    let total_secs = seconds as u64;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;

    if hours > 0 {
        format!("{hours}:{mins:02}:{secs:02}")
    } else {
        format!("{mins}:{secs:02}")
    }
}

// ============================================================================
// Unit Tests (EXTREME TDD - RED phase)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Segment;

    // Helper to create test result
    fn test_result() -> TranscriptionResult {
        TranscriptionResult {
            text: "Hello world. This is a test.".to_string(),
            language: "en".to_string(),
            segments: vec![
                Segment {
                    start: 0.0,
                    end: 1.5,
                    text: "Hello world.".to_string(),
                    ..Default::default()
                },
                Segment {
                    start: 1.5,
                    end: 3.0,
                    text: "This is a test.".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    fn empty_result() -> TranscriptionResult {
        TranscriptionResult::default()
    }

    // -------------------------------------------------------------------------
    // OutputFormat tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_output_format_extension() {
        assert_eq!(OutputFormat::Txt.extension(), "txt");
        assert_eq!(OutputFormat::Srt.extension(), "srt");
        assert_eq!(OutputFormat::Vtt.extension(), "vtt");
        assert_eq!(OutputFormat::Json.extension(), "json");
        assert_eq!(OutputFormat::Csv.extension(), "csv");
        assert_eq!(OutputFormat::Md.extension(), "md");
    }

    // -------------------------------------------------------------------------
    // Plain text tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_txt_returns_text() {
        let result = test_result();
        let output = format_txt(&result);
        assert_eq!(output, "Hello world. This is a test.");
    }

    #[test]
    fn test_format_txt_empty() {
        let result = empty_result();
        let output = format_txt(&result);
        assert_eq!(output, "");
    }

    // -------------------------------------------------------------------------
    // SRT format tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_srt_structure() {
        let result = test_result();
        let srt = format_srt(&result);

        // Should have sequence numbers
        assert!(srt.contains("1\n"));
        assert!(srt.contains("2\n"));

        // Should have timestamps
        assert!(srt.contains("00:00:00,000 --> 00:00:01,500"));
        assert!(srt.contains("00:00:01,500 --> 00:00:03,000"));

        // Should have text
        assert!(srt.contains("Hello world."));
        assert!(srt.contains("This is a test."));
    }

    #[test]
    fn test_format_srt_empty() {
        let result = empty_result();
        let srt = format_srt(&result);
        assert_eq!(srt, "");
    }

    #[test]
    fn test_format_timestamp_srt_zero() {
        assert_eq!(format_timestamp_srt(0.0), "00:00:00,000");
    }

    #[test]
    fn test_format_timestamp_srt_seconds() {
        assert_eq!(format_timestamp_srt(1.5), "00:00:01,500");
        assert_eq!(format_timestamp_srt(59.999), "00:00:59,999");
    }

    #[test]
    fn test_format_timestamp_srt_minutes() {
        assert_eq!(format_timestamp_srt(60.0), "00:01:00,000");
        assert_eq!(format_timestamp_srt(90.5), "00:01:30,500");
    }

    #[test]
    fn test_format_timestamp_srt_hours() {
        assert_eq!(format_timestamp_srt(3600.0), "01:00:00,000");
        assert_eq!(format_timestamp_srt(3661.5), "01:01:01,500");
    }

    // -------------------------------------------------------------------------
    // VTT format tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_vtt_header() {
        let result = test_result();
        let vtt = format_vtt(&result);
        assert!(vtt.starts_with("WEBVTT\n\n"));
    }

    #[test]
    fn test_format_vtt_timestamps() {
        let result = test_result();
        let vtt = format_vtt(&result);

        // VTT uses . instead of ,
        assert!(vtt.contains("00:00:00.000 --> 00:00:01.500"));
        assert!(vtt.contains("00:00:01.500 --> 00:00:03.000"));
    }

    #[test]
    fn test_format_vtt_empty() {
        let result = empty_result();
        let vtt = format_vtt(&result);
        assert_eq!(vtt, "WEBVTT\n\n");
    }

    #[test]
    fn test_format_timestamp_vtt_uses_dot() {
        assert_eq!(format_timestamp_vtt(1.5), "00:00:01.500");
    }

    // -------------------------------------------------------------------------
    // JSON format tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_json_valid() {
        let result = test_result();
        let json = format_json(&result);

        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Should be valid JSON");
        assert!(parsed.is_object());
    }

    #[test]
    fn test_format_json_contains_fields() {
        let result = test_result();
        let json = format_json(&result);

        assert!(json.contains("\"text\""));
        assert!(json.contains("\"language\""));
        assert!(json.contains("\"segments\""));
    }

    #[test]
    fn test_format_json_empty() {
        let result = empty_result();
        let json = format_json(&result);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Should be valid JSON");
        assert!(parsed.is_object());
    }

    // -------------------------------------------------------------------------
    // CSV format tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_csv_header() {
        let result = test_result();
        let csv = format_csv(&result);
        assert!(csv.starts_with("start,end,text\n"));
    }

    #[test]
    fn test_format_csv_rows() {
        let result = test_result();
        let csv = format_csv(&result);

        assert!(csv.contains("0.000,1.500,\"Hello world.\""));
        assert!(csv.contains("1.500,3.000,\"This is a test.\""));
    }

    #[test]
    fn test_format_csv_escapes_quotes() {
        let result = TranscriptionResult {
            text: "He said \"hello\"".to_string(),
            segments: vec![Segment {
                start: 0.0,
                end: 1.0,
                text: "He said \"hello\"".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let csv = format_csv(&result);

        // Quotes should be doubled
        assert!(csv.contains("\"\"hello\"\""));
    }

    #[test]
    fn test_format_csv_empty() {
        let result = empty_result();
        let csv = format_csv(&result);
        assert_eq!(csv, "start,end,text\n");
    }

    #[test]
    fn test_escape_csv() {
        assert_eq!(escape_csv("hello"), "hello");
        assert_eq!(escape_csv("say \"hi\""), "say \"\"hi\"\"");
    }

    // -------------------------------------------------------------------------
    // Markdown format tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_md_header() {
        let result = test_result();
        let md = format_md(&result);
        assert!(md.starts_with("# Transcription\n"));
    }

    #[test]
    fn test_format_md_metadata() {
        let result = test_result();
        let md = format_md(&result);

        assert!(md.contains("**Language:** en"));
    }

    #[test]
    fn test_format_md_full_text() {
        let result = test_result();
        let md = format_md(&result);

        assert!(md.contains("## Full Text"));
        assert!(md.contains("Hello world. This is a test."));
    }

    #[test]
    fn test_format_md_segments_table() {
        let result = test_result();
        let md = format_md(&result);

        assert!(md.contains("## Segments"));
        assert!(md.contains("| Start | End | Text |"));
        assert!(md.contains("|-------|-----|------|"));
    }

    #[test]
    fn test_format_md_empty_no_segments_table() {
        let result = empty_result();
        let md = format_md(&result);

        // Should not have segments table if no segments
        assert!(!md.contains("## Segments"));
    }

    #[test]
    fn test_format_timestamp_short_minutes() {
        assert_eq!(format_timestamp_short(0.0), "0:00");
        assert_eq!(format_timestamp_short(30.0), "0:30");
        assert_eq!(format_timestamp_short(90.0), "1:30");
    }

    #[test]
    fn test_format_timestamp_short_hours() {
        assert_eq!(format_timestamp_short(3600.0), "1:00:00");
        assert_eq!(format_timestamp_short(3661.0), "1:01:01");
    }

    // -------------------------------------------------------------------------
    // format_output dispatch tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_output_dispatches_correctly() {
        let result = test_result();

        // Each format should produce different output
        let txt = format_output(&result, OutputFormat::Txt);
        let srt = format_output(&result, OutputFormat::Srt);
        let vtt = format_output(&result, OutputFormat::Vtt);
        let json = format_output(&result, OutputFormat::Json);
        let csv = format_output(&result, OutputFormat::Csv);
        let md = format_output(&result, OutputFormat::Md);

        // TXT is just the text
        assert_eq!(txt, result.text);

        // SRT has sequence numbers
        assert!(srt.contains("1\n"));

        // VTT starts with WEBVTT
        assert!(vtt.starts_with("WEBVTT"));

        // JSON has braces
        assert!(json.contains("{"));

        // CSV has header
        assert!(csv.starts_with("start,end,text"));

        // MD has markdown header
        assert!(md.starts_with("# Transcription"));
    }
}
