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

    // ========================================================================
    // PROBAR-SPEC-009: Cross-Language Contract Tests
    // ========================================================================

    /// PROBAR-SPEC-009 Improvement #3: Cross-Language Contract Tests
    ///
    /// Validates that JS postMessage types match WorkerResult variants.
    /// This prevents serde tag mismatches (BH-005) where JS uses 'ready' but
    /// Rust expects 'Ready'.
    #[test]
    fn contract_js_message_types_match_worker_result() {
        use whisper_apr_demo::worker::WorkerResult;

        // Get the generated JS
        let js = whisper_apr_demo::worker_js::generate_worker_js();

        // Extract all postMessage({ type: '...' }) patterns from JS
        // Uses regex-like manual parsing to find all type strings
        let mut js_types: Vec<String> = Vec::new();

        // Pattern 1: self.postMessage({ type: 'XXX' })
        for line in js.lines() {
            if line.contains("postMessage") && line.contains("type:") {
                // Extract the type value after type:
                if let Some(type_start) = line.find("type:") {
                    let after_type = &line[type_start + 5..];
                    // Find quoted string
                    if let Some(quote_start) = after_type.find('\'') {
                        let after_quote = &after_type[quote_start + 1..];
                        if let Some(quote_end) = after_quote.find('\'') {
                            let type_value = &after_quote[..quote_end];
                            if !js_types.contains(&type_value.to_string()) {
                                js_types.push(type_value.to_string());
                            }
                        }
                    }
                }
            }
        }

        println!("Found JS postMessage types: {:?}", js_types);

        // Get WorkerResult variant names using serde
        // WorkerResult has #[serde(tag = "type")] so variant names become type values
        let worker_result_variants = ["Ready", "ModelLoaded", "Partial", "Result", "Error", "Progress", "Metrics"];

        // Verify each JS type exists in WorkerResult variants
        for js_type in &js_types {
            assert!(
                worker_result_variants.contains(&js_type.as_str()),
                "JS postMessage type '{}' does not match any WorkerResult variant.\n\
                 WorkerResult variants: {:?}\n\
                 This causes serde deserialization to fail silently (BH-005).",
                js_type,
                worker_result_variants
            );
        }

        // Verify the critical types are sent by JS
        assert!(
            js_types.contains(&"Ready".to_string()),
            "JS must send 'Ready' message for serde to deserialize as WorkerResult::Ready"
        );
        assert!(
            js_types.contains(&"Error".to_string()),
            "JS must send 'Error' message for error handling"
        );
    }

    /// Contract test: Verify serde roundtrip for WorkerResult
    ///
    /// Tests that WorkerResult serializes to expected JS-compatible format
    /// and can be deserialized back.
    #[test]
    fn contract_worker_result_serde_roundtrip() {
        use whisper_apr_demo::worker::{WorkerResult, Segment};
        use serde_json;

        // Test all variants
        let test_cases: Vec<WorkerResult> = vec![
            WorkerResult::Ready,
            WorkerResult::ModelLoaded { size_mb: 39.0, load_time_ms: 1500.0 },
            WorkerResult::Partial { text: "Hello".to_string(), is_final: false },
            WorkerResult::Result {
                text: "Hello world".to_string(),
                segments: vec![Segment { start_ms: 0, end_ms: 1000, text: "Hello world".to_string() }]
            },
            WorkerResult::Error { message: "Test error".to_string() },
            WorkerResult::Progress { phase: "Loading".to_string(), percent: 50.0 },
            WorkerResult::Metrics { rtf: 1.5, chunks_processed: 10, samples_read: 16000 },
        ];

        for original in test_cases {
            // Serialize to JSON (as serde_wasm_bindgen would)
            let json = serde_json::to_string(&original)
                .expect("Serialization should succeed");

            println!("Serialized {:?} to: {}", std::mem::discriminant(&original), json);

            // Verify the type tag is present and matches variant name
            assert!(
                json.contains("\"type\":"),
                "Serialized JSON must contain type tag: {}",
                json
            );

            // Deserialize back
            let deserialized: WorkerResult = serde_json::from_str(&json)
                .expect("Deserialization should succeed");

            // Verify roundtrip (using discriminant since we can't derive PartialEq easily)
            assert_eq!(
                std::mem::discriminant(&original),
                std::mem::discriminant(&deserialized),
                "Roundtrip should preserve variant type"
            );
        }
    }
}
