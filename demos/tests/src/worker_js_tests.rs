//! Worker JS Generation Regression Tests
//!
//! These tests verify the generated Web Worker JavaScript follows correct patterns.
//! Tests check both the raw JS output patterns and the Rust source structure.

#[cfg(test)]
mod tests {
    /// Get the worker_js.rs source
    fn get_worker_js_source() -> String {
        std::fs::read_to_string("/home/noah/src/whisper.apr/demos/www-demo/src/worker_js.rs")
            .expect("Failed to read worker_js.rs")
    }

    /// Get the worker_manager.rs source
    fn get_worker_manager_source() -> String {
        std::fs::read_to_string("/home/noah/src/whisper.apr/demos/www-demo/src/worker_manager.rs")
            .unwrap_or_default()
    }

    /// WAPR-QA-REGRESSION-001: SharedRingBuffer uses wasmModule, not wasm
    #[test]
    fn regression_sharedbuffer_uses_wasmmodule() {
        let source = get_worker_js_source();

        // Should NOT have the buggy pattern (accessing SharedRingBuffer directly on wasm)
        assert!(
            !source.contains("wasm.SharedRingBuffer"),
            "REGRESSION: Found buggy 'wasm.SharedRingBuffer' pattern in source"
        );

        // Should have the correct pattern
        assert!(
            source.contains("wasmModule.SharedRingBuffer"),
            "Missing correct 'wasmModule.SharedRingBuffer' pattern"
        );
    }

    /// WAPR-QA-REGRESSION-002: wasmModule is a top-level variable
    #[test]
    fn regression_wasmmodule_is_top_level() {
        let source = get_worker_js_source();

        // Should have let declaration for wasmModule at top level
        assert!(
            source.contains("let wasmModule"),
            "wasmModule should be declared as top-level let"
        );

        // Should use assignment inside bootstrap handler
        assert!(
            source.contains("wasmModule = await import"),
            "wasmModule should be assigned via import in bootstrap handler"
        );
    }

    /// WAPR-QA-REGRESSION-003: baseUrl is a top-level variable
    #[test]
    fn regression_baseurl_is_top_level() {
        let source = get_worker_js_source();

        // Should have let declaration for baseUrl at top level
        assert!(
            source.contains("let baseUrl"),
            "baseUrl should be declared as top-level let"
        );

        // Should use assignment inside bootstrap handler
        assert!(
            source.contains("baseUrl = msg.baseUrl") || source.contains("baseUrl = e.data.baseUrl"),
            "baseUrl should be assigned in bootstrap handler"
        );
    }

    /// WAPR-QA-REGRESSION-004: Model URL resolution uses baseUrl
    #[test]
    fn regression_model_url_uses_baseurl() {
        let source = get_worker_js_source();

        // Should check if URL starts with /
        assert!(
            source.contains("startsWith('/')") || source.contains("startsWith(\"/\")"),
            "Model URL should check if it starts with /"
        );

        // Should combine baseUrl with relative URL
        assert!(
            source.contains("baseUrl + msg.modelUrl") || source.contains("baseUrl + modelUrl"),
            "Model URL resolution should use baseUrl"
        );
    }

    /// Verify all required state variables are top-level let
    #[test]
    fn all_state_variables_are_top_level() {
        let source = get_worker_js_source();

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
            let pattern = format!("let {var}");
            assert!(
                source.contains(&pattern),
                "Missing top-level let declaration for: {var}"
            );
        }
    }

    /// Verify initWorker uses wasmModule (not wasm)
    #[test]
    fn regression_initworker_uses_wasmmodule() {
        let source = get_worker_js_source();

        // Should use wasmModule.initWorker()
        assert!(
            source.contains("wasmModule.initWorker"),
            "initWorker should be called on wasmModule"
        );
    }

    /// WAPR-QA-REGRESSION-005: isDone check in processAudioTick
    #[test]
    fn regression_isdone_in_process_tick() {
        let source = get_worker_js_source();

        // Must check isDone in processAudioTick
        assert!(
            source.contains("isDone()") && source.contains("processAudioTick"),
            "processAudioTick must check isDone() for stop detection"
        );
    }

    /// WAPR-QA-REGRESSION-006: Worker sends ready message
    #[test]
    fn regression_worker_sends_ready() {
        let source = get_worker_js_source();

        // Must send ready message after init (PascalCase for serde tag compatibility)
        assert!(
            source.contains("type: 'Ready'") || source.contains("\"type\": \"Ready\"") ||
            source.contains("{ type: \"Ready\" }") || source.contains("{type:\"Ready\"}"),
            "Worker must send Ready message (PascalCase for serde)"
        );
    }

    /// Manager marks buffer done on stop
    #[test]
    fn manager_marks_done_on_stop() {
        let source = get_worker_manager_source();
        if source.is_empty() {
            return; // File doesn't exist yet
        }

        assert!(
            source.contains("mark_done"),
            "Manager must call mark_done() on ring buffer when stopping"
        );
    }

    /// Manager handles all WorkerResult variants (no silent catch-all)
    #[test]
    fn manager_handles_all_worker_results() {
        let source = get_worker_manager_source();
        if source.is_empty() {
            return; // File doesn't exist yet
        }

        // Must NOT have catch-all that silently drops results
        assert!(
            !source.contains("_ => {}") && !source.contains("_ => { }"),
            "Manager must NOT have catch-all '_ => {{}}' - all WorkerResult variants must be handled"
        );
    }

    /// Manager dispatches transcription events
    #[test]
    fn manager_dispatches_transcription_events() {
        let source = get_worker_manager_source();
        if source.is_empty() {
            return;
        }

        assert!(
            source.contains("dispatch_transcription") || source.contains("whisper-transcription"),
            "Manager must dispatch transcription events to main thread"
        );
    }
}
