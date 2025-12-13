//! WAPR-DEMO-003: Real-time Microphone Translation - Tests
//!
//! EXTREME TDD: Tests for translation, language detection, and display modes.
//!
//! Quality Gates:
//! - Button coverage: 100%
//! - State coverage: 100%
//! - Error path coverage: ≥95%

use whisper_apr_demo_realtime_translation::*;

/// State machine tests
mod state_machine_tests {
    use super::*;

    #[test]
    fn test_initial_state_idle() {
        let demo = RealtimeTranslationDemo::new();
        assert_eq!(demo.state(), DemoState::Idle);
    }

    #[test]
    fn test_start_recording_changes_state() {
        let mut demo = RealtimeTranslationDemo::new();
        let _ = demo.start_recording();
        assert_eq!(demo.state(), DemoState::RequestingPermission);
    }

    #[test]
    fn test_clear_resets_transcripts() {
        let mut demo = RealtimeTranslationDemo::new();
        let _ = demo.start_recording();
        demo.clear();
        assert_eq!(demo.translated_text(), "");
        assert_eq!(demo.original_text(), "");
    }
}

/// Translation task tests
mod translation_tests {
    use super::*;

    #[test]
    fn test_task_is_translate() {
        let demo = RealtimeTranslationDemo::new();
        assert_eq!(demo.task(), "translate");
    }

    #[test]
    fn test_translation_initially_empty() {
        let demo = RealtimeTranslationDemo::new();
        assert_eq!(demo.translated_text(), "");
    }

    #[test]
    fn test_original_text_initially_empty() {
        let demo = RealtimeTranslationDemo::new();
        assert_eq!(demo.original_text(), "");
    }

    #[test]
    fn test_partial_translation_initially_empty() {
        let demo = RealtimeTranslationDemo::new();
        assert_eq!(demo.partial_translation(), "");
    }

    #[test]
    fn test_task_indicator() {
        let demo = RealtimeTranslationDemo::new();
        assert_eq!(demo.task_indicator(), "Translation → English");
    }
}

/// Language detection tests
mod language_tests {
    use super::*;

    #[test]
    fn test_no_language_detected_initially() {
        let demo = RealtimeTranslationDemo::new();
        assert!(demo.detected_language().is_none());
    }

    #[test]
    fn test_detected_language_text_default() {
        let demo = RealtimeTranslationDemo::new();
        assert_eq!(demo.detected_language_text(), "--");
    }

    #[test]
    fn test_confidence_threshold_high() {
        assert!(is_high_confidence(75.0));
    }

    #[test]
    fn test_confidence_threshold_low() {
        assert!(!is_high_confidence(65.0));
    }

    #[test]
    fn test_confidence_at_threshold() {
        assert!(is_high_confidence(70.0));
    }

    #[test]
    fn test_supported_languages_count() {
        assert_eq!(supported_languages_count(), 99);
    }
}

/// Display mode tests
mod display_mode_tests {
    use super::*;

    #[test]
    fn test_default_not_side_by_side() {
        let demo = RealtimeTranslationDemo::new();
        assert!(!demo.is_side_by_side());
    }

    #[test]
    fn test_toggle_side_by_side() {
        let mut demo = RealtimeTranslationDemo::new();
        demo.toggle_side_by_side();
        assert!(demo.is_side_by_side());
    }

    #[test]
    fn test_toggle_side_by_side_back() {
        let mut demo = RealtimeTranslationDemo::new();
        demo.toggle_side_by_side();
        demo.toggle_side_by_side();
        assert!(!demo.is_side_by_side());
    }
}

/// Button state tests
mod button_tests {
    use super::*;

    #[test]
    fn test_start_enabled_when_idle() {
        let demo = RealtimeTranslationDemo::new();
        assert!(demo.can_start_recording());
    }

    #[test]
    fn test_stop_disabled_when_idle() {
        let demo = RealtimeTranslationDemo::new();
        assert!(!demo.can_stop_recording());
    }

    #[test]
    fn test_start_disabled_after_starting() {
        let mut demo = RealtimeTranslationDemo::new();
        let _ = demo.start_recording();
        assert!(!demo.can_start_recording());
    }
}

/// Status display tests
mod status_tests {
    use super::*;

    #[test]
    fn test_status_text_not_empty() {
        let demo = RealtimeTranslationDemo::new();
        assert!(!demo.status_text().is_empty());
    }

    #[test]
    fn test_recording_duration_format() {
        let demo = RealtimeTranslationDemo::new();
        let duration = demo.recording_duration();
        assert!(duration.contains(':'));
    }

    #[test]
    fn test_status_ready_when_idle() {
        let demo = RealtimeTranslationDemo::new();
        assert_eq!(demo.status_text(), "Ready");
    }
}

/// Error handling tests
mod error_tests {
    use super::*;

    #[test]
    fn test_no_error_initially() {
        let demo = RealtimeTranslationDemo::new();
        assert!(demo.error_message().is_none());
    }

    #[test]
    fn test_permission_denied_sets_error() {
        let mut demo = RealtimeTranslationDemo::new();
        let _ = demo.start_recording();
        demo.on_permission_denied();
        assert_eq!(demo.state(), DemoState::Error);
        assert!(demo.error_message().is_some());
    }
}

/// Permission flow tests
mod permission_tests {
    use super::*;

    #[test]
    fn test_permission_granted_starts_recording() {
        let mut demo = RealtimeTranslationDemo::new();
        let _ = demo.start_recording();
        demo.on_permission_granted();
        assert_eq!(demo.state(), DemoState::Recording);
    }
}
