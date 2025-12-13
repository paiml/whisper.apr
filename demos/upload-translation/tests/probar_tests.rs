//! WAPR-DEMO-004: Audio/Video Upload Translation - Tests
//!
//! EXTREME TDD: Tests for file upload, translation, and bilingual export.
//!
//! Quality Gates:
//! - Button coverage: 100%
//! - State coverage: 100%
//! - Error path coverage: ≥95%

#![allow(clippy::unwrap_used, clippy::expect_used)]

use whisper_apr_demo_upload_translation::*;

/// State machine tests
mod state_machine_tests {
    use super::*;

    #[test]
    fn test_initial_state_idle() {
        let demo = UploadTranslationDemo::new();
        assert_eq!(demo.state(), DemoState::Idle);
    }

    #[test]
    fn test_file_selection_changes_state() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("audio.wav", 1000);
        assert_eq!(demo.state(), DemoState::FileSelected);
    }

    #[test]
    fn test_clear_returns_to_idle() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("audio.wav", 1000);
        demo.clear();
        assert_eq!(demo.state(), DemoState::Idle);
    }
}

/// Task tests
mod task_tests {
    use super::*;

    #[test]
    fn test_task_is_translate() {
        let demo = UploadTranslationDemo::new();
        assert_eq!(demo.task(), "translate");
    }

    #[test]
    fn test_task_indicator() {
        let demo = UploadTranslationDemo::new();
        assert_eq!(demo.task_indicator(), "Translation → English");
    }
}

/// File format tests
mod format_tests {
    use super::*;

    #[test]
    fn test_accepts_wav() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("audio.wav", 1000);
        assert!(demo.can_translate());
    }

    #[test]
    fn test_accepts_mp3() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("audio.mp3", 1000);
        assert!(demo.can_translate());
    }

    #[test]
    fn test_accepts_mp4() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("video.mp4", 1000);
        assert!(demo.can_translate());
    }

    #[test]
    fn test_accepts_webm() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("video.webm", 1000);
        assert!(demo.can_translate());
    }
}

/// Language tests
mod language_tests {
    use super::*;

    #[test]
    fn test_detected_language_initially_none() {
        let demo = UploadTranslationDemo::new();
        assert!(demo.detected_language().is_none());
    }

    #[test]
    fn test_language_pair_formatting() {
        let pair = format_language_pair("Spanish", "English");
        assert_eq!(pair, "Spanish → English");
    }

    #[test]
    fn test_language_pair_empty_initially() {
        let demo = UploadTranslationDemo::new();
        assert!(demo.language_pair().is_empty());
    }

    #[test]
    fn test_is_source_english_false_initially() {
        let demo = UploadTranslationDemo::new();
        assert!(!demo.is_source_english());
    }
}

/// File info tests
mod file_info_tests {
    use super::*;

    #[test]
    fn test_no_file_info_initially() {
        let demo = UploadTranslationDemo::new();
        assert!(demo.file_info().is_none());
    }

    #[test]
    fn test_file_info_after_selection() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("speech.mp3", 2_000_000);
        let info = demo.file_info().expect("Should have file info");
        assert_eq!(info.name(), "speech.mp3");
    }
}

/// Export tests
mod export_tests {
    use super::*;

    #[test]
    fn test_txt_export_empty() {
        let demo = UploadTranslationDemo::new();
        let txt = demo.export_txt();
        assert!(txt.is_empty());
    }

    #[test]
    fn test_srt_export_with_segment() {
        let mut demo = UploadTranslationDemo::new();
        demo.add_segment(0.0, 2.0, "Hola", "Hello");
        let srt = demo.export_srt();
        assert!(srt.contains("Hello"));
        assert!(srt.contains("-->"));
    }

    #[test]
    fn test_vtt_export_with_segment() {
        let mut demo = UploadTranslationDemo::new();
        demo.add_segment(0.0, 2.0, "Hola", "Hello");
        let vtt = demo.export_vtt();
        assert!(vtt.contains("WEBVTT"));
        assert!(vtt.contains("Hello"));
    }

    #[test]
    fn test_bilingual_srt_export() {
        let mut demo = UploadTranslationDemo::new();
        demo.add_segment(0.0, 2.0, "Hola mundo", "Hello world");
        let bilingual = demo.export_bilingual_srt();
        assert!(bilingual.contains("Hola mundo"));
        assert!(bilingual.contains("Hello world"));
    }
}

/// Button state tests
mod button_tests {
    use super::*;

    #[test]
    fn test_translate_disabled_initially() {
        let demo = UploadTranslationDemo::new();
        assert!(!demo.can_translate());
    }

    #[test]
    fn test_download_disabled_initially() {
        let demo = UploadTranslationDemo::new();
        assert!(!demo.can_download());
    }

    #[test]
    fn test_translate_enabled_after_file_selection() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("audio.wav", 1000);
        assert!(demo.can_translate());
    }
}

/// Translation output tests
mod output_tests {
    use super::*;

    #[test]
    fn test_source_text_initially_empty() {
        let demo = UploadTranslationDemo::new();
        assert!(demo.source_text().is_empty());
    }

    #[test]
    fn test_translated_text_initially_empty() {
        let demo = UploadTranslationDemo::new();
        assert!(demo.translated_text().is_empty());
    }

    #[test]
    fn test_segments_count_initially_zero() {
        let demo = UploadTranslationDemo::new();
        assert_eq!(demo.segments_count(), 0);
    }
}

/// Status tests
mod status_tests {
    use super::*;

    #[test]
    fn test_status_ready_when_idle() {
        let demo = UploadTranslationDemo::new();
        assert_eq!(demo.status_text(), "Ready");
    }

    #[test]
    fn test_status_file_selected() {
        let mut demo = UploadTranslationDemo::new();
        let _ = demo.on_file_selected("audio.wav", 1000);
        assert_eq!(demo.status_text(), "File selected");
    }
}

/// Toggle tests
mod toggle_tests {
    use super::*;

    #[test]
    fn test_bilingual_export_default_off() {
        let demo = UploadTranslationDemo::new();
        assert!(!demo.is_bilingual_export());
    }

    #[test]
    fn test_toggle_bilingual_export() {
        let mut demo = UploadTranslationDemo::new();
        demo.toggle_bilingual_export();
        assert!(demo.is_bilingual_export());
    }

    #[test]
    fn test_timestamps_default_off() {
        let demo = UploadTranslationDemo::new();
        assert!(!demo.is_show_timestamps());
    }

    #[test]
    fn test_toggle_timestamps() {
        let mut demo = UploadTranslationDemo::new();
        demo.toggle_timestamps();
        assert!(demo.is_show_timestamps());
    }
}
