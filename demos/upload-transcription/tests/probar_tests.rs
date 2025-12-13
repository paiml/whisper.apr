//! WAPR-DEMO-002: Audio/Video Upload Transcription - Tests
//!
//! EXTREME TDD: Tests for file handling, transcription, and export.
//!
//! Quality Gates:
//! - Button coverage: 100%
//! - State coverage: 100%
//! - Error path coverage: ≥95%

#![allow(clippy::unwrap_used, clippy::expect_used)]

use whisper_apr_demo_upload_transcription::*;

/// State machine tests
mod state_machine_tests {
    use super::*;

    #[test]
    fn test_initial_state_idle() {
        let demo = UploadTranscriptionDemo::new();
        assert_eq!(demo.state(), DemoState::Idle);
    }

    #[test]
    fn test_file_selection_changes_state() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("audio.wav", 1000);
        assert_eq!(demo.state(), DemoState::FileSelected);
    }

    #[test]
    fn test_clear_returns_to_idle() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("audio.wav", 1000);
        demo.clear();
        assert_eq!(demo.state(), DemoState::Idle);
    }
}

/// File format tests
mod format_tests {
    use super::*;

    #[test]
    fn test_supported_wav() {
        assert!(SupportedFormat::is_supported("audio.wav"));
    }

    #[test]
    fn test_supported_mp3() {
        assert!(SupportedFormat::is_supported("audio.mp3"));
    }

    #[test]
    fn test_supported_ogg() {
        assert!(SupportedFormat::is_supported("audio.ogg"));
    }

    #[test]
    fn test_supported_mp4() {
        assert!(SupportedFormat::is_supported("video.mp4"));
    }

    #[test]
    fn test_supported_webm() {
        assert!(SupportedFormat::is_supported("video.webm"));
    }

    #[test]
    fn test_unsupported_pdf() {
        assert!(!SupportedFormat::is_supported("document.pdf"));
    }

    #[test]
    fn test_unsupported_png() {
        assert!(!SupportedFormat::is_supported("image.png"));
    }

    #[test]
    fn test_is_video_mp4() {
        assert!(SupportedFormat::is_video("video.mp4"));
    }

    #[test]
    fn test_is_not_video_wav() {
        assert!(!SupportedFormat::is_video("audio.wav"));
    }
}

/// File info tests
mod file_info_tests {
    use super::*;

    #[test]
    fn test_no_file_info_initially() {
        let demo = UploadTranscriptionDemo::new();
        assert!(demo.file_info().is_none());
    }

    #[test]
    fn test_file_info_after_selection() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 1_500_000);
        let info = demo.file_info().expect("Should have file info");
        assert_eq!(info.name(), "test.wav");
    }

    #[test]
    fn test_file_size_formatting() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("test.wav", 5_000_000);
        let info = demo.file_info().expect("Should have file info");
        assert!(info.size_formatted().contains("MB"));
    }
}

/// Button state tests
mod button_tests {
    use super::*;

    #[test]
    fn test_transcribe_disabled_initially() {
        let demo = UploadTranscriptionDemo::new();
        assert!(!demo.can_transcribe());
    }

    #[test]
    fn test_download_disabled_initially() {
        let demo = UploadTranscriptionDemo::new();
        assert!(!demo.can_download());
    }

    #[test]
    fn test_transcribe_enabled_after_valid_file() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("audio.wav", 1000);
        assert!(demo.can_transcribe());
    }
}

/// Progress tests
mod progress_tests {
    use super::*;

    #[test]
    fn test_progress_valid_range() {
        let demo = UploadTranscriptionDemo::new();
        let progress = demo.progress();
        let percent = progress.percent();
        assert!((0.0..=100.0).contains(&percent));
    }

    #[test]
    fn test_progress_chunk_text_exists() {
        let demo = UploadTranscriptionDemo::new();
        let progress = demo.progress();
        let _ = progress.chunk_text();
    }
}

/// Export format tests
mod export_tests {
    use super::*;

    #[test]
    fn test_txt_export_empty_initially() {
        let demo = UploadTranscriptionDemo::new();
        assert!(demo.export_txt().is_empty());
    }

    #[test]
    fn test_srt_export_format() {
        let mut demo = UploadTranscriptionDemo::new();
        demo.add_segment(0.0, 2.0, "Hello world");
        let srt = demo.export_srt();
        assert!(srt.contains("Hello world"));
        assert!(srt.contains("-->"));
    }

    #[test]
    fn test_vtt_export_format() {
        let mut demo = UploadTranscriptionDemo::new();
        demo.add_segment(0.0, 2.0, "Hello world");
        let vtt = demo.export_vtt();
        assert!(vtt.contains("WEBVTT"));
        assert!(vtt.contains("Hello world"));
    }
}

/// Transcript display tests
mod transcript_tests {
    use super::*;

    #[test]
    fn test_transcript_empty_initially() {
        let demo = UploadTranscriptionDemo::new();
        assert!(demo.transcript_text().is_empty());
    }

    #[test]
    fn test_segments_count_initially_zero() {
        let demo = UploadTranscriptionDemo::new();
        assert_eq!(demo.segments_count(), 0);
    }
}

/// Helper function tests
mod helper_tests {
    use super::*;

    #[test]
    fn test_chunk_calculation_short() {
        let chunks = calculate_chunks(15.0);
        assert_eq!(chunks, 1);
    }

    #[test]
    fn test_chunk_calculation_medium() {
        let chunks = calculate_chunks(90.0);
        assert_eq!(chunks, 3);
    }

    #[test]
    fn test_format_duration_minutes() {
        let formatted = format_duration(65.0);
        assert_eq!(formatted, "1:05");
    }

    #[test]
    fn test_format_duration_hours() {
        let formatted = format_duration(3661.0);
        assert_eq!(formatted, "1:01:01");
    }
}

/// Error handling tests
mod error_tests {
    use super::*;

    #[test]
    fn test_unsupported_format_error() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("document.pdf", 1000);
        assert_eq!(demo.state(), DemoState::Error);
        assert!(demo.error_message().is_some());
    }

    #[test]
    fn test_file_too_large_error() {
        let mut demo = UploadTranscriptionDemo::new();
        let _ = demo.on_file_selected("huge.wav", 200 * 1024 * 1024);
        assert_eq!(demo.state(), DemoState::Error);
        assert!(demo.error_message().unwrap().contains("too large"));
    }
}
