//! Probar Full Stack Integration Tests for whisper.apr
#![allow(clippy::unwrap_used, clippy::field_reassign_with_default)]
//!
//! Tests the complete probar testing infrastructure on a real WASM project:
//! - Zero-JS validation (PROBAR-SPEC-012)
//! - Worker harness testing (PROBAR-SPEC-013)
//! - Docker cross-browser testing (PROBAR-SPEC-014)
//! - COOP/COEP header validation
//! - WASM thread capabilities
//! - Streaming UX validation
//!
//! NOTE: This test file requires experimental probar APIs not yet in jugar-probar 1.0.
//! Enable with: cargo test --features probar-experimental

// Skip entire file until experimental probar APIs are available
#![cfg(feature = "probar-experimental")]

use jugar_probar::prelude::*;
use std::path::PathBuf;
use std::time::Duration;

// =============================================================================
// Zero-JS Validation Tests (PROBAR-SPEC-012)
// =============================================================================

mod zero_js_tests {
    use super::*;

    #[test]
    fn test_whisper_apr_zero_js_compliance() {
        // whisper.apr is a WASM-first project - it should pass zero-JS validation
        let validator = ZeroJsValidator::new();

        // Validate src directory (should be pure Rust)
        let src_path = std::path::Path::new("src");
        if src_path.exists() {
            let result = validator.validate_directory(src_path);
            assert!(result.is_ok(), "Failed to validate src directory");

            let validation = result.unwrap();
            // src should have no JS files
            assert!(
                validation.unauthorized_js_files.is_empty(),
                "Found unauthorized JS files in src: {:?}",
                validation.unauthorized_js_files
            );
            assert!(
                validation.forbidden_directories.is_empty(),
                "Found forbidden directories: {:?}",
                validation.forbidden_directories
            );
        }
    }

    #[test]
    fn test_zero_js_strict_mode() {
        let config = ZeroJsConfig::strict();
        assert!(config.require_manifest);
        assert!(config.check_dangerous_patterns);
        assert!(!config.allow_wasm_inline_scripts);

        let validator = ZeroJsValidator::with_config(config);

        // Create temp directory with clean WASM structure
        let temp = tempfile::TempDir::new().unwrap();
        std::fs::write(temp.path().join("app.wasm"), b"fake wasm").unwrap();
        std::fs::write(
            temp.path().join("index.html"),
            "<!DOCTYPE html><html><head></head><body></body></html>",
        )
        .unwrap();

        let result = validator.validate_directory(temp.path()).unwrap();
        assert!(
            result.is_valid(),
            "Clean WASM structure should pass: {}",
            result
        );
    }

    #[test]
    fn test_zero_js_dangerous_pattern_detection() {
        let validator = ZeroJsValidator::new();

        // Test various dangerous patterns
        let dangerous_code = "const x = eval('1 + 1');";
        let violations =
            validator.validate_js_content(dangerous_code, std::path::Path::new("test.js"));
        assert!(!violations.is_empty(), "Should detect eval()");

        let inner_html = "element.innerHTML = userInput;";
        let violations = validator.validate_js_content(inner_html, std::path::Path::new("test.js"));
        assert!(!violations.is_empty(), "Should detect innerHTML");
    }

    #[test]
    fn test_zero_js_html_inline_script_detection() {
        let validator = ZeroJsValidator::with_config(ZeroJsConfig {
            allow_wasm_inline_scripts: false,
            ..Default::default()
        });

        let html = r#"
<!DOCTYPE html>
<html>
<head>
    <script>alert('hello');</script>
</head>
<body></body>
</html>
"#;

        let violations = validator.validate_html_content(html, std::path::Path::new("index.html"));
        assert!(!violations.is_empty(), "Should detect inline script");
    }

    #[test]
    fn test_zero_js_wasm_inline_allowed() {
        let validator = ZeroJsValidator::new(); // Default allows WASM inline

        let html = r#"
<!DOCTYPE html>
<html>
<head>
    <script>
        // __PROBAR_WASM_GENERATED__
        WebAssembly.instantiate(module);
    </script>
</head>
<body></body>
</html>
"#;

        let violations = validator.validate_html_content(html, std::path::Path::new("index.html"));
        assert!(
            violations.is_empty(),
            "WASM-generated scripts should be allowed"
        );
    }

    #[test]
    fn test_zero_js_validation_result_display() {
        let result = ZeroJsValidationResult {
            valid: true,
            verified_js_files: vec![PathBuf::from("worker.js")],
            ..Default::default()
        };

        let display = format!("{}", result);
        assert!(display.contains("PASSED"));
        assert!(display.contains("Verified JS files: 1"));
    }

    #[test]
    fn test_zero_js_violation_count() {
        let result = ZeroJsValidationResult {
            unauthorized_js_files: vec![PathBuf::from("a.js"), PathBuf::from("b.js")],
            forbidden_directories: vec![PathBuf::from("node_modules")],
            ..Default::default()
        };

        assert_eq!(result.violation_count(), 3);
    }
}

// =============================================================================
// Worker Harness Tests (PROBAR-SPEC-013)
// =============================================================================

mod worker_harness_tests {
    use super::*;

    #[test]
    fn test_worker_test_config_default() {
        let config = WorkerTestConfig::default();
        assert_eq!(config.init_timeout, Duration::from_secs(10));
        assert_eq!(config.command_timeout, Duration::from_secs(30));
        assert_eq!(config.max_messages, 1000);
        assert!(config.test_error_recovery);
        assert!(config.test_memory_leaks);
        assert_eq!(config.stress_iterations, 100);
        assert!(config.verify_lamport_ordering);
        assert!(config.test_shared_memory);
        assert_eq!(config.ring_buffer_size, 16384);
    }

    #[test]
    fn test_worker_test_config_minimal() {
        let config = WorkerTestConfig::minimal();
        assert_eq!(config.init_timeout, Duration::from_secs(5));
        assert_eq!(config.stress_iterations, 10);
        assert!(!config.test_error_recovery);
        assert!(!config.test_memory_leaks);
    }

    #[test]
    fn test_worker_test_config_comprehensive() {
        let config = WorkerTestConfig::comprehensive();
        assert_eq!(config.init_timeout, Duration::from_secs(30));
        assert_eq!(config.stress_iterations, 1000);
        assert!(config.test_error_recovery);
        assert!(config.test_memory_leaks);
    }

    #[test]
    fn test_worker_lifecycle_states() {
        assert_eq!(
            WorkerLifecycleState::default(),
            WorkerLifecycleState::NotCreated
        );
        assert_eq!(format!("{}", WorkerLifecycleState::Ready), "Ready");
        assert_eq!(
            format!("{}", WorkerLifecycleState::Processing),
            "Processing"
        );
        assert_eq!(
            format!("{}", WorkerLifecycleState::Terminated),
            "Terminated"
        );
    }

    #[test]
    fn test_worker_harness_creation() {
        let harness = WorkerTestHarness::new();
        assert!(harness.config().verify_lamport_ordering);
        assert!(harness.config().test_shared_memory);
    }

    #[test]
    fn test_worker_harness_with_config() {
        let config = WorkerTestConfig::minimal();
        let harness = WorkerTestHarness::with_config(config);
        assert_eq!(harness.config().stress_iterations, 10);
    }

    #[test]
    fn test_worker_harness_default() {
        let harness = WorkerTestHarness::default();
        assert!(harness.config().test_shared_memory);
    }

    #[test]
    fn test_worker_lifecycle_transitions() {
        let harness = WorkerTestHarness::new();
        let failures = harness.test_lifecycle_transitions();
        assert!(
            failures.is_empty(),
            "Valid lifecycle transitions should pass"
        );
    }

    #[test]
    fn test_lamport_ordering_valid() {
        let harness = WorkerTestHarness::new();
        let timestamps = vec![1, 2, 3, 4, 5, 10, 20, 30];
        let failures = harness.verify_message_ordering(&timestamps);
        assert!(failures.is_empty(), "Valid Lamport ordering should pass");
    }

    #[test]
    fn test_lamport_ordering_invalid() {
        let harness = WorkerTestHarness::new();
        let timestamps = vec![1, 2, 5, 3, 6]; // 3 < 5 violates ordering
        let failures = harness.verify_message_ordering(&timestamps);
        assert!(!failures.is_empty(), "Invalid ordering should be detected");
        assert_eq!(failures[0].category, WorkerTestCategory::Ordering);
    }

    #[test]
    fn test_lamport_ordering_disabled() {
        let config = WorkerTestConfig {
            verify_lamport_ordering: false,
            ..Default::default()
        };
        let harness = WorkerTestHarness::with_config(config);
        let timestamps = vec![5, 3, 1]; // Invalid but should pass when disabled
        let failures = harness.verify_message_ordering(&timestamps);
        assert!(failures.is_empty(), "Ordering check should be skipped");
    }

    #[test]
    fn test_ring_buffer_basic() {
        let harness = WorkerTestHarness::new();
        let config = RingBufferTestConfig::default();
        let result = harness.test_ring_buffer(&config);
        assert!(result.writes_succeeded > 0);
        assert!(result.reads_succeeded > 0);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let harness = WorkerTestHarness::new();
        let config = RingBufferTestConfig {
            buffer_size: 1024,
            sample_size: 512,
            num_samples: 100, // Much more than capacity (2)
            test_overflow: true,
            test_underrun: true,
            test_concurrent: true,
        };
        let result = harness.test_ring_buffer(&config);
        assert!(result.overflows_detected > 0, "Should detect overflow");
    }

    #[test]
    fn test_ring_buffer_config_default() {
        let config = RingBufferTestConfig::default();
        assert_eq!(config.buffer_size, 16384);
        assert_eq!(config.sample_size, 512);
        assert_eq!(config.num_samples, 1000);
        assert!(config.test_overflow);
        assert!(config.test_underrun);
        assert!(config.test_concurrent);
    }

    #[test]
    fn test_shared_memory_basic() {
        let harness = WorkerTestHarness::new();
        let config = SharedMemoryTestConfig::default();
        let result = harness.test_shared_memory(&config);
        assert!(
            result.atomics_correct,
            "Atomic operations should be correct"
        );
    }

    #[test]
    fn test_shared_memory_disabled() {
        let worker_config = WorkerTestConfig {
            test_shared_memory: false,
            ..Default::default()
        };
        let harness = WorkerTestHarness::with_config(worker_config);
        let sm_config = SharedMemoryTestConfig::default();
        let result = harness.test_shared_memory(&sm_config);
        assert!(result.is_passed(), "Should pass when disabled");
    }

    #[test]
    fn test_shared_memory_config_default() {
        let config = SharedMemoryTestConfig::default();
        assert_eq!(config.buffer_size, 4096);
        assert_eq!(config.num_atomic_ops, 1000);
        assert!(config.test_wait_notify);
        assert!(config.test_concurrent_writes);
        assert_eq!(config.wait_timeout, Duration::from_millis(100));
    }

    #[test]
    fn test_worker_metrics_memory_leak_detection() {
        let mut metrics = WorkerMetrics::default();
        metrics.memory_start = 1000;
        metrics.memory_end = 1200; // 20% growth
        assert!(metrics.has_memory_leak(), "Should detect 20% memory growth");
    }

    #[test]
    fn test_worker_metrics_no_memory_leak() {
        let mut metrics = WorkerMetrics::default();
        metrics.memory_start = 1000;
        metrics.memory_end = 1050; // 5% growth
        assert!(!metrics.has_memory_leak(), "5% growth should not be a leak");
    }

    #[test]
    fn test_worker_metrics_memory_growth() {
        let mut metrics = WorkerMetrics::default();
        metrics.memory_start = 1000;
        metrics.memory_end = 1500;
        assert_eq!(metrics.memory_growth(), 500);
    }

    #[test]
    fn test_worker_test_result_display() {
        let result = WorkerTestResult {
            passed: true,
            lifecycle_passed: true,
            ordering_passed: true,
            metrics: WorkerMetrics {
                initialization_time: Duration::from_millis(100),
                average_message_latency: Duration::from_micros(500),
                messages_processed: 1000,
                ..Default::default()
            },
            ..Default::default()
        };
        let display = format!("{}", result);
        assert!(display.contains("PASSED"));
        assert!(display.contains("Messages processed: 1000"));
    }

    #[test]
    fn test_worker_test_failure_display() {
        let failure = WorkerTestFailure {
            category: WorkerTestCategory::Ordering,
            description: "Out of order".to_string(),
            expected: "5".to_string(),
            actual: "3".to_string(),
        };
        let display = format!("{}", failure);
        assert!(display.contains("Ordering"));
        assert!(display.contains("Out of order"));
    }

    #[test]
    fn test_worker_js_generation() {
        let lifecycle_js = WorkerTestHarness::lifecycle_test_js();
        assert!(lifecycle_js.contains("__PROBAR_WORKER_STATES__"));
        assert!(lifecycle_js.contains("recordState"));
        assert!(lifecycle_js.contains("recordTransition"));

        let ring_buffer_js = WorkerTestHarness::ring_buffer_test_js(16384);
        assert!(ring_buffer_js.contains("__PROBAR_RING_BUFFER__"));
        assert!(ring_buffer_js.contains("SharedArrayBuffer"));
        assert!(ring_buffer_js.contains("16384"));

        let shared_memory_js = WorkerTestHarness::shared_memory_test_js(4096);
        assert!(shared_memory_js.contains("__PROBAR_SHARED_MEMORY__"));
        assert!(shared_memory_js.contains("testAtomicAdd"));
        assert!(shared_memory_js.contains("4096"));
    }

    #[test]
    fn test_worker_error_display() {
        let err = WorkerTestError::InitializationFailed("worker failed".to_string());
        assert!(err.to_string().contains("Initialization"));

        let err = WorkerTestError::Timeout("10s".to_string());
        assert!(err.to_string().contains("Timeout"));

        let err = WorkerTestError::ProtocolError("bad message".to_string());
        assert!(err.to_string().contains("Protocol"));

        let err = WorkerTestError::CdpError("connection lost".to_string());
        assert!(err.to_string().contains("CDP"));
    }

    #[test]
    fn test_harness_validate_results() {
        let harness = WorkerTestHarness::new();
        let result = WorkerTestResult {
            passed: true,
            lifecycle_passed: true,
            ordering_passed: true,
            shared_memory_passed: true,
            ring_buffer_passed: true,
            error_recovery_passed: true,
            memory_leak_passed: true,
            failures: vec![],
            ..Default::default()
        };
        assert!(harness.validate_results(&result));
    }
}

// =============================================================================
// Docker Cross-Browser Tests (PROBAR-SPEC-014)
// =============================================================================

#[cfg(feature = "docker")]
mod docker_tests {
    use super::*;

    #[test]
    fn test_docker_browser_enum() {
        assert_eq!(DockerBrowser::Chrome.default_cdp_port(), 9222);
        assert_eq!(DockerBrowser::Firefox.default_cdp_port(), 9223);
        assert_eq!(DockerBrowser::WebKit.default_cdp_port(), 9224);

        assert_eq!(DockerBrowser::Chrome.image_name(), "probar-chrome:latest");
        assert_eq!(DockerBrowser::Firefox.image_name(), "probar-firefox:latest");
        assert_eq!(DockerBrowser::WebKit.image_name(), "probar-webkit:latest");

        let browsers = DockerBrowser::all();
        assert_eq!(browsers.len(), 3);
    }

    #[test]
    fn test_docker_browser_from_str() {
        use std::str::FromStr;

        assert_eq!(
            DockerBrowser::from_str("chrome").unwrap(),
            DockerBrowser::Chrome
        );
        assert_eq!(
            DockerBrowser::from_str("firefox").unwrap(),
            DockerBrowser::Firefox
        );
        assert_eq!(
            DockerBrowser::from_str("webkit").unwrap(),
            DockerBrowser::WebKit
        );
        assert!(DockerBrowser::from_str("invalid").is_err());
    }

    #[test]
    fn test_coop_coep_config() {
        let config = CoopCoepConfig::default();
        assert!(config.shared_array_buffer_available());

        let disabled = CoopCoepConfig::disabled();
        assert!(!disabled.shared_array_buffer_available());
    }

    #[test]
    fn test_coop_coep_header_validation() {
        let mut headers = HashMap::new();
        headers.insert(
            "Cross-Origin-Opener-Policy".to_string(),
            "same-origin".to_string(),
        );
        headers.insert(
            "Cross-Origin-Embedder-Policy".to_string(),
            "require-corp".to_string(),
        );

        let result = validate_coop_coep_headers(&headers);
        assert!(result.is_ok());
        assert!(
            result.unwrap(),
            "Valid headers should enable SharedArrayBuffer"
        );
    }

    #[test]
    fn test_coop_coep_missing_headers() {
        let headers: HashMap<String, String> = HashMap::new();
        let result = validate_coop_coep_headers(&headers);
        assert!(result.is_ok());
        assert!(
            !result.unwrap(),
            "Missing headers should disable SharedArrayBuffer"
        );
    }

    #[test]
    fn test_container_state_enum() {
        let state = ContainerState::NotCreated;
        assert_eq!(state, ContainerState::NotCreated);

        let running = ContainerState::Running;
        assert_eq!(format!("{:?}", running), "Running");
    }

    #[test]
    fn test_container_config_for_browser() {
        let chrome_config = ContainerConfig::for_browser(DockerBrowser::Chrome);
        assert!(chrome_config.ports.contains(&(9222, 9222)));

        let firefox_config = ContainerConfig::for_browser(DockerBrowser::Firefox);
        assert!(firefox_config.ports.contains(&(9223, 9223)));
    }

    #[test]
    fn test_docker_test_result() {
        let result =
            DockerTestResult::passed("test_wasm_init".to_string(), Duration::from_millis(150));
        assert!(result.passed);
        assert_eq!(result.name, "test_wasm_init");
        assert_eq!(result.duration, Duration::from_millis(150));

        let failed = DockerTestResult::failed(
            "test_memory".to_string(),
            Duration::from_millis(200),
            "assertion failed".to_string(),
        );
        assert!(!failed.passed);
        assert!(failed.error.is_some());
    }

    #[test]
    fn test_docker_test_results() {
        let mut results = DockerTestResults::new(DockerBrowser::Chrome);

        results.add_result(DockerTestResult::passed(
            "test1".to_string(),
            Duration::from_millis(100),
        ));
        results.add_result(DockerTestResult::failed(
            "test2".to_string(),
            Duration::from_millis(200),
            "error".to_string(),
        ));

        assert!(!results.all_passed());
        assert_eq!(results.total(), 2);
        assert_eq!(results.pass_rate(), 50.0);

        let display = format!("{}", results);
        assert!(display.contains("chrome"));
        assert!(display.contains("1 passed"));
        assert!(display.contains("1 failed"));
    }

    #[test]
    fn test_docker_error_display() {
        let err = DockerError::DaemonUnavailable("not running".to_string());
        assert!(err.to_string().contains("unavailable") || err.to_string().contains("Daemon"));

        let err = DockerError::ConfigError("bad config".to_string());
        assert!(err.to_string().contains("Config") || err.to_string().contains("config"));
    }

    #[test]
    fn test_docker_test_runner_builder() {
        let builder = DockerTestRunner::builder()
            .browser(DockerBrowser::Chrome)
            .with_coop_coep(true)
            .timeout(Duration::from_secs(60))
            .cleanup(true)
            .capture_logs(true);

        // Builder pattern should work without errors
        // Actual building would require Docker daemon
        let _ = builder;
    }

    #[test]
    fn test_parallel_runner_builder() {
        let builder = ParallelRunner::builder()
            .browsers(&DockerBrowser::all())
            .tests(&["tests/e2e.rs"]);

        // Builder pattern should work without errors
        let _ = builder;
    }
}

// =============================================================================
// WASM Capabilities Tests
// =============================================================================

mod capabilities_tests {
    use super::*;

    #[test]
    fn test_wasm_thread_capabilities_full_support() {
        let caps = WasmThreadCapabilities::full_support();
        assert!(caps.shared_array_buffer);
        assert!(caps.atomics);
        assert!(caps.cross_origin_isolated);
        assert!(caps.is_secure_context);
        assert!(caps.assert_threading_ready().is_ok());
    }

    #[test]
    fn test_wasm_thread_capabilities_no_support() {
        let caps = WasmThreadCapabilities::no_support();
        assert!(!caps.shared_array_buffer);
        assert!(!caps.atomics);
        assert!(caps.assert_threading_ready().is_err());
    }

    #[test]
    fn test_capability_status() {
        let available = CapabilityStatus::Available;
        let unavailable = CapabilityStatus::Unavailable("reason".to_string());
        let unknown = CapabilityStatus::Unknown;

        assert_eq!(format!("{:?}", available), "Available");
        assert!(matches!(unavailable, CapabilityStatus::Unavailable(_)));
        assert!(matches!(unknown, CapabilityStatus::Unknown));
    }

    #[test]
    fn test_required_headers() {
        // RequiredHeaders has const values
        assert_eq!(RequiredHeaders::COOP, "same-origin");
        assert_eq!(RequiredHeaders::COEP, "require-corp");
    }

    #[test]
    fn test_worker_emulator() {
        let emulator = WorkerEmulator::new();
        // Default state is Uninitialized
        assert_eq!(emulator.state(), WorkerState::Uninitialized);
    }

    #[test]
    fn test_worker_state() {
        let uninitialized = WorkerState::Uninitialized;
        let processing = WorkerState::Processing;
        let terminated = WorkerState::Terminated;

        assert_eq!(format!("{:?}", uninitialized), "Uninitialized");
        assert_eq!(format!("{:?}", processing), "Processing");
        assert_eq!(format!("{:?}", terminated), "Terminated");
    }
}

// =============================================================================
// Strict Mode Tests
// =============================================================================

mod strict_mode_tests {
    use super::*;

    #[test]
    fn test_wasm_strict_mode_production() {
        let mode = WasmStrictMode::production();
        // Production mode should be constructable
        let _ = mode;
    }

    #[test]
    fn test_wasm_strict_mode_development() {
        let mode = WasmStrictMode::development();
        // Development mode should be constructable
        let _ = mode;
    }

    #[test]
    fn test_wasm_strict_mode_minimal() {
        let mode = WasmStrictMode::minimal();
        // Minimal mode should be constructable
        let _ = mode;
    }

    #[test]
    fn test_e2e_test_checklist() {
        let checklist = E2ETestChecklist::new();
        // Checklist should be constructable
        let _ = checklist;
    }

    #[test]
    fn test_console_capture() {
        let capture = ConsoleCapture::new();
        assert!(capture.messages().is_empty());
    }

    #[test]
    fn test_console_severity() {
        let log = ConsoleSeverity::Log;
        let warn = ConsoleSeverity::Warn;
        let error = ConsoleSeverity::Error;

        assert_eq!(format!("{:?}", log), "Log");
        assert_eq!(format!("{:?}", warn), "Warn");
        assert_eq!(format!("{:?}", error), "Error");
    }
}

// =============================================================================
// Streaming UX Validation Tests
// =============================================================================

mod streaming_validation_tests {
    use super::*;

    #[test]
    fn test_streaming_ux_validator() {
        let validator = StreamingUxValidator::new();
        // Validator should be constructable
        let _ = validator;
    }

    #[test]
    fn test_streaming_state() {
        let idle = StreamingState::Idle;
        let buffering = StreamingState::Buffering;
        let streaming = StreamingState::Streaming;
        let stalled = StreamingState::Stalled;

        assert_eq!(format!("{:?}", idle), "Idle");
        assert_eq!(format!("{:?}", buffering), "Buffering");
        assert_eq!(format!("{:?}", streaming), "Streaming");
        assert_eq!(format!("{:?}", stalled), "Stalled");
    }

    #[test]
    fn test_streaming_metric() {
        let latency = StreamingMetric::Latency(Duration::from_millis(100));
        let buffer = StreamingMetric::BufferLevel(0.75);
        let frame_dropped = StreamingMetric::FrameDropped;

        assert!(format!("{:?}", latency).contains("Latency"));
        assert!(format!("{:?}", buffer).contains("BufferLevel"));
        assert!(format!("{:?}", frame_dropped).contains("FrameDropped"));
    }

    #[test]
    fn test_vu_meter_config() {
        let config = VuMeterConfig::default();
        // VuMeterConfig has min_level, max_level, update_rate_hz, etc.
        assert!(config.min_level < config.max_level);
        assert!(config.update_rate_hz > 0.0);
    }
}

// =============================================================================
// Integration Tests - Full Stack Validation
// =============================================================================

mod integration_tests {
    use super::*;

    /// Test that whisper.apr's test infrastructure is properly set up
    #[test]
    fn test_probar_dependency_available() {
        // If this test compiles and runs, probar is properly linked
        let _harness = WorkerTestHarness::new();
        let _validator = ZeroJsValidator::new();
        let _config = WorkerTestConfig::default();
    }

    /// Comprehensive test simulating real whisper.apr WASM testing scenario
    #[test]
    fn test_whisper_wasm_testing_scenario() {
        // 1. Zero-JS validation
        let validator = ZeroJsValidator::with_config(ZeroJsConfig::default());
        let temp = tempfile::TempDir::new().unwrap();

        // Create clean WASM structure
        std::fs::write(temp.path().join("whisper.wasm"), b"wasm binary").unwrap();
        std::fs::write(
            temp.path().join("index.html"),
            r#"<!DOCTYPE html>
<html>
<head>
    <script>
        // __PROBAR_WASM_GENERATED__
        WebAssembly.instantiate(wasmModule);
    </script>
</head>
<body><div id="app"></div></body>
</html>"#,
        )
        .unwrap();

        let zero_js_result = validator.validate_directory(temp.path()).unwrap();
        assert!(
            zero_js_result.is_valid(),
            "WASM structure should pass zero-JS validation: {}",
            zero_js_result
        );

        // 2. Worker harness validation
        let harness = WorkerTestHarness::with_config(WorkerTestConfig::comprehensive());

        // Verify lifecycle transitions are valid
        let lifecycle_failures = harness.test_lifecycle_transitions();
        assert!(
            lifecycle_failures.is_empty(),
            "Lifecycle transitions should be valid"
        );

        // Verify message ordering
        let audio_timestamps: Vec<u64> = (0..100).map(|i| i * 10).collect();
        let ordering_failures = harness.verify_message_ordering(&audio_timestamps);
        assert!(
            ordering_failures.is_empty(),
            "Audio frame timestamps should be ordered"
        );

        // 3. Ring buffer test (simulating audio worklet)
        let ring_config = RingBufferTestConfig {
            buffer_size: 65536, // 64KB for audio
            sample_size: 512,   // 512 samples per chunk
            num_samples: 500,
            test_overflow: true,
            test_underrun: true,
            test_concurrent: true,
        };
        let ring_result = harness.test_ring_buffer(&ring_config);
        assert!(
            ring_result.writes_succeeded > 0,
            "Ring buffer should accept writes"
        );

        // 4. Shared memory test (for WASM threading)
        let shared_config = SharedMemoryTestConfig {
            buffer_size: 8192,
            num_atomic_ops: 500,
            test_wait_notify: true,
            test_concurrent_writes: true,
            wait_timeout: Duration::from_millis(50),
        };
        let shared_result = harness.test_shared_memory(&shared_config);
        assert!(
            shared_result.atomics_correct,
            "Atomic operations should work"
        );
    }

    /// Test metrics and result validation
    #[test]
    fn test_comprehensive_metrics() {
        let harness = WorkerTestHarness::new();

        // Create a full test result
        let result = WorkerTestResult {
            passed: true,
            lifecycle_passed: true,
            ordering_passed: true,
            shared_memory_passed: true,
            ring_buffer_passed: true,
            error_recovery_passed: true,
            memory_leak_passed: true,
            failures: vec![],
            metrics: WorkerMetrics {
                initialization_time: Duration::from_millis(50),
                average_message_latency: Duration::from_micros(100),
                max_message_latency: Duration::from_millis(5),
                messages_processed: 10000,
                messages_dropped: 0,
                memory_start: 1024 * 1024,      // 1MB
                memory_end: 1024 * 1024 + 1024, // 1MB + 1KB (acceptable growth)
                error_recoveries: 0,
            },
        };

        assert!(result.is_passed(), "Full result should pass");
        assert!(harness.validate_results(&result), "Validation should pass");
        assert!(
            !result.metrics.has_memory_leak(),
            "Should not detect memory leak"
        );
    }
}
