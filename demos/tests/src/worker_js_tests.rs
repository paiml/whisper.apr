//! Worker JS Generation Regression Tests
//!
//! These tests verify the generated Web Worker JavaScript follows correct patterns.
//! Instead of duplicating the generator, we test the actual output via WASM build.
//!
//! Also includes regression tests for WorkerManager state synchronization bugs.

#[cfg(test)]
mod tests {
    use std::process::Command;

    /// Get the generated worker JS by running a simple extraction
    fn get_generated_worker_js() -> String {
        // Build the WASM to ensure worker_js.rs is compiled
        let output = Command::new("cargo")
            .args(["build", "-p", "whisper-apr-demo", "--target", "wasm32-unknown-unknown"])
            .current_dir("/home/noah/src/whisper.apr/demos")
            .output()
            .expect("Failed to build WASM");

        if !output.status.success() {
            panic!("WASM build failed: {}", String::from_utf8_lossy(&output.stderr));
        }

        // The worker JS is generated at runtime, so we can't easily extract it.
        // Instead, we'll read the source and verify the patterns there.
        std::fs::read_to_string("/home/noah/src/whisper.apr/demos/www-demo/src/worker_js.rs")
            .expect("Failed to read worker_js.rs")
    }

    /// WAPR-QA-REGRESSION-001: SharedRingBuffer uses wasmModule, not wasm
    ///
    /// The bug was: `dot(id("wasm"), "SharedRingBuffer")`
    /// The fix is: `dot(id("wasmModule"), "SharedRingBuffer")`
    #[test]
    fn regression_sharedbuffer_uses_wasmmodule() {
        let source = get_generated_worker_js();

        // Should NOT have the buggy pattern
        assert!(
            !source.contains(r#"dot(id("wasm"), "SharedRingBuffer")"#),
            "REGRESSION: Found buggy 'wasm.SharedRingBuffer' pattern in source"
        );

        // Should have the correct pattern
        assert!(
            source.contains(r#"dot(id("wasmModule"), "SharedRingBuffer")"#),
            "Missing correct 'wasmModule.SharedRingBuffer' pattern"
        );
    }

    /// WAPR-QA-REGRESSION-002: wasmModule is a top-level variable
    ///
    /// The bug was: `const wasmModule` inside try block (block-scoped)
    /// The fix is: `let wasmModule` at top level + `assign` inside block
    #[test]
    fn regression_wasmmodule_is_top_level() {
        let source = get_generated_worker_js();

        // Should have let_decl for wasmModule at top level
        assert!(
            source.contains(r#".let_decl("wasmModule""#),
            "wasmModule should be declared as top-level let"
        );

        // Should use assign, not const_decl for wasmModule inside bootstrap
        // Pattern is multi-line: Stmt::assign(\n    "wasmModule",
        assert!(
            source.contains("Stmt::assign") &&
            source.lines().any(|line| line.contains(r#""wasmModule","#)),
            "wasmModule should use Stmt::assign in bootstrap handler"
        );

        // Should NOT use const_decl for wasmModule (that was the bug)
        assert!(
            !source.contains(r#"const_decl("wasmModule""#) &&
            !source.contains(r#"const_decl(\n                        "wasmModule""#),
            "wasmModule should NOT use const_decl (block-scoped)"
        );
    }

    /// WAPR-QA-REGRESSION-003: baseUrl is a top-level variable
    ///
    /// The bug was: `const baseUrl` inside bootstrap block (block-scoped)
    /// The fix is: `let baseUrl` at top level + `assign` inside block
    #[test]
    fn regression_baseurl_is_top_level() {
        let source = get_generated_worker_js();

        // Should have let_decl for baseUrl at top level
        assert!(
            source.contains(r#".let_decl("baseUrl""#),
            "baseUrl should be declared as top-level let"
        );

        // Should use assign, not const_decl for baseUrl inside bootstrap
        assert!(
            source.contains(r#"Stmt::assign("baseUrl""#),
            "baseUrl should use Stmt::assign in bootstrap handler"
        );
    }

    /// WAPR-QA-REGRESSION-004: Model URL resolution uses baseUrl
    ///
    /// Relative model URLs (starting with /) need baseUrl prefix in workers.
    #[test]
    fn regression_model_url_uses_baseurl() {
        let source = get_generated_worker_js();

        // Should check if URL starts with /
        assert!(
            source.contains(r#"startsWith"#) && source.contains(r#""/""#),
            "Model URL should check if it starts with /"
        );

        // Should combine baseUrl with relative URL
        assert!(
            source.contains("baseUrl") && source.contains("modelUrl"),
            "Model URL resolution should use baseUrl"
        );
    }

    /// Verify all required state variables are top-level
    #[test]
    fn all_state_variables_are_top_level() {
        let source = get_generated_worker_js();

        let required_vars = [
            "baseUrl",
            "wasmModule",
            "wasm",
            "worker",
            "ringBuffer",
            "processingInterval",
            "initialized",
        ];

        for var in &required_vars {
            let pattern = format!(r#".let_decl("{var}""#);
            assert!(
                source.contains(&pattern),
                "Missing top-level let declaration for: {var}"
            );
        }
    }

    /// Verify initWorker uses wasmModule (not wasm)
    #[test]
    fn regression_initworker_uses_wasmmodule() {
        let source = get_generated_worker_js();

        assert!(
            source.contains(r#"dot(id("wasmModule"), "initWorker")"#),
            "initWorker should be called on wasmModule"
        );
    }

    // =========================================================================
    // WORKER MANAGER STATE SYNC REGRESSION TESTS
    // =========================================================================

    fn get_worker_manager_source() -> String {
        std::fs::read_to_string("/home/noah/src/whisper.apr/demos/www-demo/src/worker_manager.rs")
            .expect("Failed to read worker_manager.rs")
    }

    /// WAPR-QA-REGRESSION-005: WorkerManager must use shared state_ptr, not separate state field
    ///
    /// The bug was: WorkerManager had both `state: ManagerState` and a local `state_ptr`
    /// created in `spawn()`. The closure updated `state_ptr`, but methods checked `self.state`.
    /// This caused `start_recording()` to fail because `self.state` was never updated.
    ///
    /// The fix: Use `state_ptr: Rc<RefCell<ManagerState>>` as a field, and all methods
    /// must read/write through `self.state_ptr.borrow()` / `self.state_ptr.borrow_mut()`.
    #[test]
    fn regression_worker_manager_uses_shared_state_ptr() {
        let source = get_worker_manager_source();

        // Must have state_ptr as a field, NOT a separate state field
        assert!(
            source.contains("state_ptr: Rc<RefCell<ManagerState>>"),
            "WorkerManager must use state_ptr: Rc<RefCell<ManagerState>> as field"
        );

        // Should NOT have a separate `state: ManagerState` field (that was the bug)
        // Check that the struct definition doesn't have `state: ManagerState`
        let struct_pattern = "pub struct WorkerManager";
        if let Some(struct_start) = source.find(struct_pattern) {
            // Find the closing brace of the struct
            let struct_section = &source[struct_start..];
            if let Some(brace_end) = struct_section.find('}') {
                let struct_def = &struct_section[..brace_end];
                // The struct should NOT contain `state: ManagerState` (only state_ptr)
                assert!(
                    !struct_def.contains("state: ManagerState"),
                    "WorkerManager must NOT have separate 'state: ManagerState' field - use state_ptr"
                );
            }
        }
    }

    /// WAPR-QA-REGRESSION-006: State checks must use state_ptr.borrow()
    ///
    /// All state checks must go through the shared Rc<RefCell<>>, not a direct field access.
    #[test]
    fn regression_state_checks_use_borrow() {
        let source = get_worker_manager_source();

        // is_ready() must use state_ptr.borrow()
        assert!(
            source.contains("*self.state_ptr.borrow() == ManagerState::Ready"),
            "is_ready() must check *self.state_ptr.borrow()"
        );

        // is_recording() must use state_ptr.borrow()
        assert!(
            source.contains("*self.state_ptr.borrow() == ManagerState::Recording"),
            "is_recording() must check *self.state_ptr.borrow()"
        );
    }

    /// WAPR-QA-REGRESSION-007: spawn() must use self.state_ptr, not create local state_ptr
    ///
    /// The bug was: spawn() created `let state_ptr = Rc::new(...)` locally, which was
    /// disconnected from self. The closure captured this local, so self.state was never updated.
    #[test]
    fn regression_spawn_uses_self_state_ptr() {
        let source = get_worker_manager_source();

        // spawn() should use self.state_ptr.clone() for the closure
        assert!(
            source.contains("let state_ptr_clone = self.state_ptr.clone()"),
            "spawn() must clone self.state_ptr for closure, not create new Rc"
        );

        // spawn() should NOT create a new local state_ptr (that was the bug)
        // This pattern was the bug: `let state_ptr = Rc::new(RefCell::new(ManagerState::Spawning))`
        assert!(
            !source.contains("let state_ptr = Rc::new(RefCell::new(ManagerState"),
            "spawn() must NOT create new local state_ptr - must use self.state_ptr"
        );
    }

    /// WAPR-QA-REGRESSION-008: All WorkerResult variants must be handled explicitly
    ///
    /// The bug was: `_ => {}` catch-all silently ignored WorkerResult::Result,
    /// so final transcription was never sent to the UI.
    ///
    /// This test ensures no catch-all `_ => {}` exists in the message handler.
    #[test]
    fn regression_all_worker_results_handled() {
        let source = get_worker_manager_source();

        // Must NOT have catch-all `_ => {}` in the match
        // The pattern `_ => {}` or `_ => { }` silently drops unhandled cases
        assert!(
            !source.contains("_ => {}"),
            "WorkerManager must NOT have `_ => {{}}` catch-all - all WorkerResult variants must be handled explicitly"
        );

        // Must handle WorkerResult::Result
        assert!(
            source.contains("WorkerResult::Result {"),
            "WorkerManager must handle WorkerResult::Result for final transcription"
        );

        // Must handle WorkerResult::Progress
        assert!(
            source.contains("WorkerResult::Progress {"),
            "WorkerManager must handle WorkerResult::Progress"
        );

        // Must handle WorkerResult::Metrics
        assert!(
            source.contains("WorkerResult::Metrics {"),
            "WorkerManager must handle WorkerResult::Metrics"
        );
    }
}
