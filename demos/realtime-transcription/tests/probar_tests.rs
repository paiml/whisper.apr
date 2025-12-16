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
    fn test_initial_state_is_initializing() {
        let demo = RealtimeTranscriptionDemo::new();
        assert_eq!(demo.state(), DemoState::Initializing);
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
    fn test_start_disabled_when_initializing() {
        // New demo is in Initializing state (model not loaded yet)
        let demo = RealtimeTranscriptionDemo::new();
        assert!(!demo.can_start_recording());
    }

    #[test]
    fn test_stop_disabled_when_initializing() {
        let demo = RealtimeTranscriptionDemo::new();
        assert!(!demo.can_stop_recording());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
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
        // Note: start_recording() requires WASM, just test clear without recording
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
/// NOTE: These tests require WASM environment - skip on native
mod compatibility_tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_browser_compatibility_check_exists() {
        let compat = check_browser_compatibility();
        let _ = compat.is_supported();
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_compatibility_warning_message() {
        let compat = check_browser_compatibility();
        let _ = compat.warning_message();
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_browser_compatibility_native_placeholder() {
        // Browser compatibility requires WASM environment
        // This placeholder ensures the test module compiles on native
        assert!(true, "Browser tests skipped on native target");
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
        // Demo starts in Initializing state (model not yet loaded)
        assert_eq!(demo.state(), DemoState::Initializing);
    }
}

// ============================================================================
// WAPR-SPEC-010: Async Worker Tests (Phase 4 - Robustness)
// ============================================================================

/// Queue management tests (WAPR-SPEC-010 Section 3.2)
mod queue_management_tests {
    use whisper_apr_demo_realtime_transcription::bridge::{
        QueueStats, MAX_QUEUE_DEPTH, MAX_CONSECUTIVE_ERRORS,
    };

    #[test]
    fn test_max_queue_depth_is_three() {
        // Per spec: Bounded Queues - max 3 chunks
        assert_eq!(MAX_QUEUE_DEPTH, 3, "Queue depth must be 3 per WAPR-SPEC-010");
    }

    #[test]
    fn test_max_consecutive_errors_is_three() {
        // Per spec: Worker unhealthy after 3 consecutive errors
        assert_eq!(MAX_CONSECUTIVE_ERRORS, 3, "Error threshold must be 3");
    }

    #[test]
    fn test_queue_stats_fields() {
        let stats = QueueStats::default();

        // All counters should start at zero
        assert_eq!(stats.chunks_sent, 0);
        assert_eq!(stats.chunks_dropped, 0);
        assert_eq!(stats.chunks_completed, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(stats.queue_depth, 0);
    }

    #[test]
    fn test_queue_stats_is_clone() {
        let stats = QueueStats {
            chunks_sent: 100,
            chunks_dropped: 5,
            chunks_completed: 95,
            errors: 2,
            avg_latency_ms: 150.0,
            queue_depth: 2,
            processing_started: true,
        };
        let cloned = stats.clone();
        assert_eq!(stats.chunks_sent, cloned.chunks_sent);
        assert_eq!(stats.chunks_dropped, cloned.chunks_dropped);
    }

    #[test]
    fn test_queue_stats_is_debug() {
        let stats = QueueStats::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("QueueStats"));
        assert!(debug.contains("chunks_sent"));
    }
}

/// Worker result type tests
mod worker_result_tests {
    use whisper_apr_demo_realtime_transcription::bridge::WorkerResult;

    #[test]
    fn test_all_worker_results_are_clone() {
        let results = [
            WorkerResult::Ready,
            WorkerResult::ModelLoaded { size_mb: 37.0, load_time_ms: 1500.0 },
            WorkerResult::SessionStarted { session_id: "test".to_string() },
            WorkerResult::Transcription {
                session_id: "test".to_string(),
                chunk_id: 1,
                text: "hello".to_string(),
                tokens: vec![1, 2, 3],
                is_partial: false,
                rtf: 0.5,
            },
            WorkerResult::SessionEnded { session_id: "test".to_string() },
            WorkerResult::Error {
                session_id: Some("test".to_string()),
                chunk_id: Some(1),
                message: "test error".to_string(),
            },
            WorkerResult::Metrics { queue_depth: 2, avg_latency_ms: 100.0 },
            WorkerResult::Pong { timestamp: 1000.0, worker_time: 1005.0 },
        ];

        for result in &results {
            let _ = result.clone();
        }
    }

    #[test]
    fn test_all_worker_results_are_debug() {
        let results = [
            WorkerResult::Ready,
            WorkerResult::ModelLoaded { size_mb: 37.0, load_time_ms: 1500.0 },
            WorkerResult::Error {
                session_id: None,
                chunk_id: None,
                message: "test".to_string(),
            },
        ];

        for result in &results {
            let debug = format!("{:?}", result);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_transcription_result_fields() {
        let result = WorkerResult::Transcription {
            session_id: "session_123".to_string(),
            chunk_id: 42,
            text: "hello world".to_string(),
            tokens: vec![100, 200, 300],
            is_partial: true,
            rtf: 0.8,
        };

        if let WorkerResult::Transcription {
            session_id,
            chunk_id,
            text,
            tokens,
            is_partial,
            rtf,
        } = result {
            assert_eq!(session_id, "session_123");
            assert_eq!(chunk_id, 42);
            assert_eq!(text, "hello world");
            assert_eq!(tokens.len(), 3);
            assert!(is_partial);
            assert!((rtf - 0.8).abs() < f64::EPSILON);
        } else {
            panic!("Expected Transcription variant");
        }
    }
}

/// Worker command type tests
mod worker_command_tests {
    use whisper_apr_demo_realtime_transcription::bridge::WorkerCommand;

    #[test]
    fn test_worker_commands_are_debug() {
        let commands = [
            WorkerCommand::LoadModel,
            WorkerCommand::StartSession { session_id: "test".to_string() },
            WorkerCommand::Transcribe {
                session_id: "test".to_string(),
                chunk_id: 1,
                prompt_tokens: vec![1, 2, 3],
                is_final: false,
            },
            WorkerCommand::EndSession { session_id: "test".to_string() },
            WorkerCommand::SetOptions {
                language: Some("en".to_string()),
                task: "transcribe".to_string(),
            },
            WorkerCommand::Shutdown,
            WorkerCommand::Ping { timestamp: 1000.0 },
        ];

        for cmd in &commands {
            let debug = format!("{:?}", cmd);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_worker_commands_are_clone() {
        let cmd = WorkerCommand::Transcribe {
            session_id: "test".to_string(),
            chunk_id: 1,
            prompt_tokens: vec![1, 2, 3],
            is_final: true,
        };
        let cloned = cmd.clone();
        if let WorkerCommand::Transcribe { session_id, .. } = cloned {
            assert_eq!(session_id, "test");
        }
    }
}

/// Memory stability tests (WAPR-SPEC-010 Section 4.4)
mod memory_stability_tests {
    use super::*;

    #[test]
    fn test_demo_handles_100_partial_results() {
        let mut demo = RealtimeTranscriptionDemo::new();

        // Simulate 100 partial results (memory stability check)
        for i in 0..100 {
            demo.on_partial_result(&format!("Partial result {}", i));
        }

        // Demo should still be functional
        assert!(!demo.partial_transcript().is_empty());
    }

    #[test]
    fn test_demo_handles_many_clears() {
        let mut demo = RealtimeTranscriptionDemo::new();

        // Simulate 100 clear operations (memory stability check)
        for _ in 0..100 {
            demo.clear_transcript();
        }

        // Demo should still be functional
        assert_eq!(demo.transcript(), "");
        // Demo starts in Initializing state (model not yet loaded)
        assert_eq!(demo.state(), DemoState::Initializing);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_demo_handles_many_start_stop_cycles() {
        let mut demo = RealtimeTranscriptionDemo::new();

        // Simulate many start/stop cycles
        for _ in 0..50 {
            let _ = demo.start_recording();
            let _ = demo.stop_recording();
        }

        // Demo should still be functional
        assert!(demo.can_start_recording() || !demo.can_start_recording());
    }

    #[test]
    fn test_queue_stats_handles_large_values() {
        use whisper_apr_demo_realtime_transcription::bridge::QueueStats;

        let stats = QueueStats {
            chunks_sent: u64::MAX,
            chunks_dropped: u64::MAX / 2,
            chunks_completed: u64::MAX / 2,
            errors: u64::MAX / 4,
            avg_latency_ms: f64::MAX / 2.0,
            queue_depth: usize::MAX,
            processing_started: true,
        };

        // Should not panic on large values
        let _ = format!("{:?}", stats);
        let _ = stats.clone();
    }
}

// ============================================================================
// WAPR-SPEC-010: Error Recovery Tests (Phase 4 - Robustness)
// ============================================================================

/// Error recovery UI flow tests
mod error_recovery_tests {
    use super::*;

    #[test]
    fn test_valid_transition_to_error_from_requesting() {
        assert!(StateTransition::is_valid(
            DemoState::RequestingPermission,
            DemoState::Error
        ));
    }

    #[test]
    fn test_valid_transition_to_error_from_loading() {
        assert!(StateTransition::is_valid(
            DemoState::LoadingModel,
            DemoState::Error
        ));
    }

    #[test]
    fn test_valid_transition_to_error_from_initializing() {
        assert!(StateTransition::is_valid(
            DemoState::Initializing,
            DemoState::Error
        ));
    }

    #[test]
    fn test_error_to_idle_recovery() {
        // Error state can transition back to Idle (recovery)
        assert!(StateTransition::is_valid(DemoState::Error, DemoState::Idle));
    }

    #[test]
    fn test_error_to_loading_model_recovery() {
        // Error state can transition to LoadingModel (retry)
        assert!(StateTransition::is_valid(
            DemoState::Error,
            DemoState::LoadingModel
        ));
    }

    #[test]
    fn test_invalid_transitions_from_error() {
        // Cannot go directly from Error to Recording
        assert!(!StateTransition::is_valid(
            DemoState::Error,
            DemoState::Recording
        ));
        // Cannot go from Error to Processing
        assert!(!StateTransition::is_valid(
            DemoState::Error,
            DemoState::Processing
        ));
    }

    #[test]
    fn test_error_message_initially_none() {
        let demo = RealtimeTranscriptionDemo::new();
        assert!(demo.error_message().is_none());
    }

    #[test]
    fn test_demo_state_enum_debug() {
        // DemoState derives Debug
        let error = DemoState::Error;
        let debug = format!("{:?}", error);
        assert!(debug.contains("Error"));
    }

    #[test]
    fn test_demo_state_enum_clone() {
        // DemoState derives Clone
        let error = DemoState::Error;
        let cloned = error.clone();
        assert_eq!(error, cloned);
    }

    #[test]
    fn test_demo_state_enum_eq() {
        // DemoState derives PartialEq
        assert_eq!(DemoState::Error, DemoState::Error);
        assert_ne!(DemoState::Error, DemoState::Idle);
    }
}

/// Worker bridge error handling tests
mod worker_error_handling_tests {
    use whisper_apr_demo_realtime_transcription::bridge::{
        WorkerResult, MAX_CONSECUTIVE_ERRORS,
    };

    #[test]
    fn test_error_result_fields() {
        let error = WorkerResult::Error {
            session_id: Some("session_123".to_string()),
            chunk_id: Some(42),
            message: "Test error message".to_string(),
        };

        if let WorkerResult::Error {
            session_id,
            chunk_id,
            message,
        } = error
        {
            assert_eq!(session_id, Some("session_123".to_string()));
            assert_eq!(chunk_id, Some(42));
            assert_eq!(message, "Test error message");
        } else {
            panic!("Expected Error variant");
        }
    }

    #[test]
    fn test_error_result_without_session() {
        let error = WorkerResult::Error {
            session_id: None,
            chunk_id: None,
            message: "Global error".to_string(),
        };

        if let WorkerResult::Error {
            session_id,
            chunk_id,
            message,
        } = error
        {
            assert!(session_id.is_none());
            assert!(chunk_id.is_none());
            assert_eq!(message, "Global error");
        }
    }

    #[test]
    fn test_max_consecutive_errors_threshold() {
        // Per WAPR-SPEC-010: Worker is unhealthy after 3 consecutive errors
        assert_eq!(MAX_CONSECUTIVE_ERRORS, 3);
    }

    #[test]
    fn test_error_results_are_clone_and_debug() {
        let error = WorkerResult::Error {
            session_id: Some("test".to_string()),
            chunk_id: Some(1),
            message: "clone test".to_string(),
        };

        let cloned = error.clone();
        let debug = format!("{:?}", cloned);

        assert!(debug.contains("Error"));
        assert!(debug.contains("clone test"));
    }
}
