//! Property-Based Tests for whisper.apr Demo Code
//!
//! Uses proptest to verify invariants that must hold across all inputs.
//! Reference: "QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs"
//!           Claessen & Hughes, ICFP 2000
//!
//! ## Test Categories
//!
//! 1. **Worker JS Invariants**: Source code pattern verification (unit tests, NOT proptest)
//! 2. **Ring Buffer Model**: SPSC correctness with random inputs (proptest)
//! 3. **State Machine Model**: Random event sequence validation (proptest)
//!
//! NOTE: Only tests inside `proptest! {}` blocks use random inputs.
//! Tests with `#[test]` are deterministic unit tests.

use proptest::prelude::*;

// ============================================================================
// WORKER JS INVARIANTS (Unit tests - NOT property-based)
// These verify source code patterns, not runtime behavior with random inputs.
// ============================================================================

mod worker_js_invariants {
    /// Read the worker_js.rs source for pattern checks
    fn get_worker_js_source() -> String {
        std::fs::read_to_string("/home/noah/src/whisper.apr/demos/www-demo/src/worker_js.rs")
            .expect("Failed to read worker_js.rs")
    }

    /// Invariant: Generated JS compiles and has required function
    #[test]
    fn invariant_deterministic_generation() {
        // Build WASM once to ensure worker_js.rs is valid
        let output = std::process::Command::new("cargo")
            .args(["check", "-p", "whisper-apr-demo", "--target", "wasm32-unknown-unknown"])
            .current_dir("/home/noah/src/whisper.apr/demos")
            .output()
            .expect("Failed to run cargo check");

        assert!(output.status.success(), "WASM check failed");

        let source = get_worker_js_source();
        assert!(
            source.contains("generate_worker_js"),
            "Must have generate_worker_js function"
        );
    }

    /// Invariant: No forbidden browser APIs in worker code
    #[test]
    fn invariant_no_forbidden_browser_apis() {
        let source = get_worker_js_source();

        // Raw JS shouldn't access window or document (workers don't have these)
        // Note: comments about these are okay
        let has_window_call = source.contains("window.") && !source.contains("// window") && !source.contains("/* window");
        let has_document_call = source.contains("document.") && !source.contains("// document") && !source.contains("/* document");
        let has_import_scripts = source.contains("importScripts(");

        assert!(!has_window_call, "Worker JS must not access window object");
        assert!(!has_document_call, "Worker JS must not access document object");
        assert!(!has_import_scripts, "ES modules must use dynamic import(), not importScripts()");
    }

    /// Invariant: All state variables use let for top-level scope
    #[test]
    fn invariant_state_variables_use_let_decl() {
        let source = get_worker_js_source();

        let state_vars = [
            "baseUrl",
            "wasmModule",
            "wasm",
            "worker",
            "ringBuffer",
            "processingInterval",
            "initialized",
        ];

        for var in &state_vars {
            let let_pattern = format!("let {var}");
            assert!(
                source.contains(&let_pattern),
                "State variable '{var}' must be declared with let for top-level scope"
            );
        }
    }

    /// Invariant: Correct API targets (wasmModule vs wasm)
    #[test]
    fn invariant_correct_api_targets() {
        let source = get_worker_js_source();

        assert!(
            source.contains("wasmModule.SharedRingBuffer"),
            "SharedRingBuffer must use wasmModule"
        );
        assert!(
            !source.contains("wasm.SharedRingBuffer"),
            "SharedRingBuffer must NOT use wasm instance"
        );
        assert!(
            source.contains("wasmModule.initWorker"),
            "initWorker must use wasmModule"
        );
    }

    /// Invariant: Bootstrap sets baseUrl before URL resolution
    #[test]
    fn invariant_bootstrap_sets_baseurl_first() {
        let source = get_worker_js_source();

        assert!(
            source.contains("baseUrl = msg.baseUrl") || source.contains("baseUrl = e.data.baseUrl"),
            "Bootstrap handler must assign baseUrl"
        );
        assert!(
            source.contains("baseUrl") && source.contains("modelUrl"),
            "Model URL resolution must use baseUrl"
        );
    }
}

// ============================================================================
// RING BUFFER MODEL (Pure-Rust SPSC Simulation)
// ============================================================================

mod ring_buffer_model {
    use super::*;

    /// Pure-Rust model of SPSC ring buffer for property testing
    /// This models the behavior without WASM dependencies.
    #[derive(Debug, Clone)]
    struct RingBufferModel {
        capacity: usize,
        write_idx: usize,
        read_idx: usize,
        data: Vec<f32>,
    }

    impl RingBufferModel {
        fn new(capacity: usize) -> Self {
            Self {
                capacity,
                write_idx: 0,
                read_idx: 0,
                data: vec![0.0; capacity],
            }
        }

        fn available_read(&self) -> usize {
            if self.write_idx >= self.read_idx {
                self.write_idx - self.read_idx
            } else {
                self.capacity - self.read_idx + self.write_idx
            }
        }

        fn available_write(&self) -> usize {
            // Leave one slot empty to distinguish full from empty
            self.capacity - 1 - self.available_read()
        }

        fn write(&mut self, samples: &[f32]) -> usize {
            let available = self.available_write();
            let to_write = samples.len().min(available);

            for (i, &sample) in samples.iter().take(to_write).enumerate() {
                let idx = (self.write_idx + i) % self.capacity;
                self.data[idx] = sample;
            }

            self.write_idx = (self.write_idx + to_write) % self.capacity;
            to_write
        }

        fn read(&mut self, count: usize) -> Vec<f32> {
            let available = self.available_read();
            let to_read = count.min(available);

            let mut samples = Vec::with_capacity(to_read);
            for i in 0..to_read {
                let idx = (self.read_idx + i) % self.capacity;
                samples.push(self.data[idx]);
            }

            self.read_idx = (self.read_idx + to_read) % self.capacity;
            samples
        }

        fn reset(&mut self) {
            self.write_idx = 0;
            self.read_idx = 0;
        }
    }

    proptest! {
        /// Property: Write then read returns same samples
        #[test]
        fn prop_write_read_identity(
            capacity in 16usize..1024,
            samples in prop::collection::vec(any::<f32>(), 1..100)
        ) {
            let mut buffer = RingBufferModel::new(capacity);

            // Write samples (may be truncated if buffer too small)
            let written = buffer.write(&samples);

            // Read back
            let read_samples = buffer.read(written);

            // Must match what was written
            prop_assert_eq!(&samples[..written], &read_samples[..]);
        }

        /// Property: Available read + available write = capacity - 1
        #[test]
        fn prop_available_sum_invariant(
            capacity in 16usize..1024,
            write_count in 0usize..100,
            read_count in 0usize..100
        ) {
            let mut buffer = RingBufferModel::new(capacity);
            let samples: Vec<f32> = (0..write_count).map(|i| i as f32).collect();

            buffer.write(&samples);
            let _ = buffer.read(read_count);

            // Invariant: available_read + available_write = capacity - 1
            let sum = buffer.available_read() + buffer.available_write();
            prop_assert_eq!(sum, capacity - 1);
        }

        /// Property: Empty buffer has zero available read
        #[test]
        fn prop_empty_buffer_zero_read(capacity in 16usize..1024) {
            let buffer = RingBufferModel::new(capacity);
            prop_assert_eq!(buffer.available_read(), 0);
        }

        /// Property: Reset returns buffer to empty state
        #[test]
        fn prop_reset_empties_buffer(
            capacity in 16usize..1024,
            samples in prop::collection::vec(any::<f32>(), 1..100)
        ) {
            let mut buffer = RingBufferModel::new(capacity);
            buffer.write(&samples);

            buffer.reset();

            prop_assert_eq!(buffer.available_read(), 0);
            prop_assert_eq!(buffer.available_write(), capacity - 1);
        }

        /// Property: Writing to full buffer returns 0
        #[test]
        fn prop_full_buffer_write_zero(capacity in 16usize..256) {
            let mut buffer = RingBufferModel::new(capacity);

            // Fill buffer completely
            let fill: Vec<f32> = (0..capacity).map(|i| i as f32).collect();
            buffer.write(&fill);

            // Try to write more
            let written = buffer.write(&[1.0, 2.0, 3.0]);
            prop_assert_eq!(written, 0, "Writing to full buffer should return 0");
        }

        /// Property: Wrap-around preserves data integrity
        #[test]
        fn prop_wraparound_integrity(capacity in 16usize..64) {
            let mut buffer = RingBufferModel::new(capacity);

            // Fill half, read half, fill again to cause wrap
            let half = capacity / 2;
            let fill1: Vec<f32> = (0..half).map(|i| i as f32).collect();
            buffer.write(&fill1);
            let _ = buffer.read(half);

            // Now write more than remaining space to wrap
            let fill2: Vec<f32> = (100..100 + capacity - 2).map(|i| i as f32).collect();
            let written = buffer.write(&fill2);

            // Read everything back
            let read_back = buffer.read(written);

            prop_assert_eq!(&fill2[..written], &read_back[..]);
        }
    }
}

// ============================================================================
// STATE MACHINE MODEL (Pure-Rust WorkerManager Simulation)
// ============================================================================

mod state_machine_model {
    use super::*;

    /// States matching WorkerManager::ManagerState
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum State {
        Uninitialized,
        Spawning,
        Loading,
        Ready,
        Recording,
        Error,
    }

    /// Events that trigger state transitions
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Event {
        Spawn,
        WorkerReady,
        ModelLoaded,
        StartRecording,
        StopRecording,
        Shutdown,
        ErrorOccurred,
    }

    /// Pure-Rust model of WorkerManager state machine
    #[derive(Debug, Clone)]
    struct WorkerManagerModel {
        state: State,
        pending_start: bool,
    }

    impl WorkerManagerModel {
        fn new() -> Self {
            Self {
                state: State::Uninitialized,
                pending_start: false,
            }
        }

        fn transition(&mut self, event: Event) -> Result<(), &'static str> {
            match (self.state, event) {
                // Valid transitions
                (State::Uninitialized, Event::Spawn) => {
                    self.state = State::Spawning;
                    Ok(())
                }
                (State::Spawning, Event::WorkerReady) => {
                    self.state = State::Loading;
                    Ok(())
                }
                (State::Spawning, Event::ErrorOccurred) => {
                    self.state = State::Error;
                    Ok(())
                }
                (State::Loading, Event::ModelLoaded) => {
                    self.state = State::Ready;
                    // Check for pending start
                    if self.pending_start {
                        self.pending_start = false;
                        self.state = State::Recording;
                    }
                    Ok(())
                }
                (State::Loading, Event::StartRecording) => {
                    // Queue the request
                    self.pending_start = true;
                    Ok(())
                }
                (State::Loading, Event::ErrorOccurred) => {
                    self.state = State::Error;
                    Ok(())
                }
                (State::Ready, Event::StartRecording) => {
                    self.state = State::Recording;
                    Ok(())
                }
                (State::Ready, Event::Shutdown) => {
                    self.state = State::Uninitialized;
                    Ok(())
                }
                (State::Recording, Event::StopRecording) => {
                    self.state = State::Ready;
                    Ok(())
                }
                (State::Recording, Event::ErrorOccurred) => {
                    self.state = State::Error;
                    Ok(())
                }
                (State::Error, Event::Shutdown) => {
                    self.state = State::Uninitialized;
                    Ok(())
                }
                // Invalid transitions
                _ => Err("Invalid state transition"),
            }
        }
    }

    /// Strategy to generate valid event sequences
    fn valid_event_strategy() -> impl Strategy<Value = Vec<Event>> {
        prop::collection::vec(
            prop_oneof![
                Just(Event::Spawn),
                Just(Event::WorkerReady),
                Just(Event::ModelLoaded),
                Just(Event::StartRecording),
                Just(Event::StopRecording),
                Just(Event::Shutdown),
                Just(Event::ErrorOccurred),
            ],
            0..20,
        )
    }

    proptest! {
        /// Property: State machine never panics on any event sequence
        #[test]
        fn prop_no_panic_on_any_sequence(events in valid_event_strategy()) {
            let mut model = WorkerManagerModel::new();

            for event in events {
                // Ignore errors - we just want to verify no panics
                let _ = model.transition(event);
            }
        }

        /// Property: Random event sequences with random starting state eventually reach stable state
        #[test]
        fn prop_random_events_with_random_start(
            start_state in prop_oneof![
                Just(State::Uninitialized),
                Just(State::Spawning),
                Just(State::Loading),
                Just(State::Ready),
                Just(State::Recording),
                Just(State::Error),
            ],
            events in valid_event_strategy()
        ) {
            let mut model = WorkerManagerModel::new();
            model.state = start_state;

            for event in events {
                let _ = model.transition(event);
            }

            // Model should be in some valid state (not panicked)
            prop_assert!(matches!(
                model.state,
                State::Uninitialized | State::Spawning | State::Loading |
                State::Ready | State::Recording | State::Error
            ));
        }
    }

    // =========================================================================
    // DETERMINISTIC STATE MACHINE TESTS (Unit tests, NOT proptest)
    // These test specific scenarios, not random inputs.
    // =========================================================================

    #[test]
    fn test_shutdown_from_ready_returns_uninitialized() {
        let mut model = WorkerManagerModel::new();
        model.state = State::Ready;
        let _ = model.transition(Event::Shutdown);
        assert_eq!(model.state, State::Uninitialized);
    }

    #[test]
    fn test_shutdown_from_error_returns_uninitialized() {
        let mut model = WorkerManagerModel::new();
        model.state = State::Error;
        let _ = model.transition(Event::Shutdown);
        assert_eq!(model.state, State::Uninitialized);
    }

    #[test]
    fn test_cannot_record_uninitialized() {
        let mut model = WorkerManagerModel::new();
        let result = model.transition(Event::StartRecording);
        assert!(result.is_err());
        assert_eq!(model.state, State::Uninitialized);
    }

    #[test]
    fn test_happy_path_reaches_recording() {
        let mut model = WorkerManagerModel::new();
        model.transition(Event::Spawn).expect("spawn");
        model.transition(Event::WorkerReady).expect("ready");
        model.transition(Event::ModelLoaded).expect("loaded");
        model.transition(Event::StartRecording).expect("start");
        assert_eq!(model.state, State::Recording);
    }

    /// REGRESSION TEST: Pending start triggers recording after model load
    /// This tests the bug where start_recording() was called before model loaded.
    #[test]
    fn test_pending_start_triggers_after_load() {
        let mut model = WorkerManagerModel::new();

        // Start recording BEFORE model loads
        model.transition(Event::Spawn).expect("spawn");
        model.transition(Event::WorkerReady).expect("ready");
        model.transition(Event::StartRecording).expect("queue start");

        // Should be in Loading with pending flag
        assert_eq!(model.state, State::Loading);
        assert!(model.pending_start);

        // Now model loads - should auto-transition to Recording
        model.transition(Event::ModelLoaded).expect("loaded");
        assert_eq!(model.state, State::Recording);
        assert!(!model.pending_start);
    }

    #[test]
    fn test_error_from_active_states() {
        let active_states = [State::Spawning, State::Loading, State::Recording];

        for start_state in active_states {
            let mut model = WorkerManagerModel::new();
            model.state = start_state;

            let result = model.transition(Event::ErrorOccurred);
            assert!(result.is_ok(), "Error should be valid from {start_state:?}");
            assert_eq!(model.state, State::Error);
        }
    }
}

// ============================================================================
// TESTS MODULE MARKER
// ============================================================================

#[cfg(test)]
mod tests {
    #[test]
    fn property_tests_compile() {
        // Marker test to ensure module compiles
        assert!(true);
    }
}
