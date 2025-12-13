//! WAPR-DEMO-001: Real-time Microphone Transcription - Tests
//!
//! EXTREME TDD: Tests for state machine, button states, and core functionality.
//!
//! Quality Gates:
//! - Button coverage: 100%
//! - State coverage: 100%
//! - Error path coverage: ≥95%

use whisper_apr_demo_realtime_transcription::*;

/// State machine tests
mod state_machine_tests {
    use super::*;

    #[test]
    fn test_initial_state_is_idle() {
        let demo = RealtimeTranscriptionDemo::new();
        assert_eq!(demo.state(), DemoState::Idle);
    }

    #[test]
    fn test_valid_transitions_idle_to_requesting() {
        assert!(StateTransition::is_valid(
            DemoState::Idle,
            DemoState::RequestingPermission
        ));
    }

    #[test]
    fn test_valid_transitions_requesting_to_recording() {
        assert!(StateTransition::is_valid(
            DemoState::RequestingPermission,
            DemoState::Recording
        ));
    }

    #[test]
    fn test_valid_transitions_requesting_to_error() {
        assert!(StateTransition::is_valid(
            DemoState::RequestingPermission,
            DemoState::Error
        ));
    }

    #[test]
    fn test_valid_transitions_recording_to_processing() {
        assert!(StateTransition::is_valid(
            DemoState::Recording,
            DemoState::Processing
        ));
    }

    #[test]
    fn test_valid_transitions_processing_to_idle() {
        assert!(StateTransition::is_valid(
            DemoState::Processing,
            DemoState::Idle
        ));
    }

    #[test]
    fn test_valid_transitions_error_to_idle() {
        assert!(StateTransition::is_valid(DemoState::Error, DemoState::Idle));
    }

    #[test]
    fn test_invalid_transition_idle_to_recording() {
        assert!(!StateTransition::is_valid(
            DemoState::Idle,
            DemoState::Recording
        ));
    }

    #[test]
    fn test_invalid_transition_processing_to_recording() {
        assert!(!StateTransition::is_valid(
            DemoState::Processing,
            DemoState::Recording
        ));
    }
}

/// Button state tests
mod button_tests {
    use super::*;

    #[test]
    fn test_start_enabled_when_idle() {
        let demo = RealtimeTranscriptionDemo::new();
        assert!(demo.can_start_recording());
    }

    #[test]
    fn test_stop_disabled_when_idle() {
        let demo = RealtimeTranscriptionDemo::new();
        assert!(!demo.can_stop_recording());
    }

    #[test]
    fn test_start_disabled_after_starting() {
        let mut demo = RealtimeTranscriptionDemo::new();
        let _ = demo.start_recording();
        assert!(!demo.can_start_recording());
    }
}

/// Transcription display tests
mod transcription_tests {
    use super::*;

    #[test]
    fn test_transcript_initially_empty() {
        let demo = RealtimeTranscriptionDemo::new();
        assert_eq!(demo.transcript(), "");
    }

    #[test]
    fn test_partial_transcript_initially_empty() {
        let demo = RealtimeTranscriptionDemo::new();
        assert_eq!(demo.partial_transcript(), "");
    }

    #[test]
    fn test_clear_resets_transcript() {
        let mut demo = RealtimeTranscriptionDemo::new();
        let _ = demo.start_recording();
        demo.clear_transcript();
        assert_eq!(demo.transcript(), "");
    }
}

/// Status display tests
mod status_tests {
    use super::*;

    #[test]
    fn test_status_text_not_empty() {
        let demo = RealtimeTranscriptionDemo::new();
        assert!(!demo.status_text().is_empty());
    }

    #[test]
    fn test_status_text_user_friendly() {
        let demo = RealtimeTranscriptionDemo::new();
        let status = demo.status_text();
        assert!(!status.contains('_'), "Status should be user-friendly");
    }

    #[test]
    fn test_recording_duration_format() {
        let demo = RealtimeTranscriptionDemo::new();
        let duration = demo.recording_duration();
        assert!(duration.contains(':'), "Duration should be M:SS format");
    }
}

/// Browser compatibility tests
mod compatibility_tests {
    use super::*;

    #[test]
    fn test_browser_compatibility_check_exists() {
        let compat = check_browser_compatibility();
        let _ = compat.is_supported();
    }

    #[test]
    fn test_compatibility_warning_message() {
        let compat = check_browser_compatibility();
        let _ = compat.warning_message();
    }
}

/// Stability tests (non-wasm compatible)
mod stability_tests {
    use super::*;

    #[test]
    fn test_multiple_clears() {
        let mut demo = RealtimeTranscriptionDemo::new();
        for _ in 0..10 {
            demo.clear_transcript();
        }
        assert_eq!(demo.state(), DemoState::Idle);
    }
}
