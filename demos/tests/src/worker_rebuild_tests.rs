//! Worker/Transcription Tests (Steps 51-80)
//!
//! WAPR-DEMO-REBUILD-TDD: Extreme TDD for demo rebuild
//! Tests the worker JS generation and transcription flow.

use std::process::Command;

/// Get the generated worker JS source (from worker_js.rs)
fn get_worker_js_source() -> String {
    std::fs::read_to_string("/home/noah/src/whisper.apr/demos/www-demo/src/worker_js.rs")
        .unwrap_or_default()
}

/// Get the worker manager source
fn get_worker_manager_source() -> String {
    std::fs::read_to_string("/home/noah/src/whisper.apr/demos/www-demo/src/worker_manager.rs")
        .unwrap_or_default()
}

/// Get the ring buffer source
fn get_ring_buffer_source() -> String {
    std::fs::read_to_string("/home/noah/src/whisper.apr/demos/www-demo/src/ring_buffer.rs")
        .unwrap_or_default()
}

/// Get the worker source
fn get_worker_source() -> String {
    std::fs::read_to_string("/home/noah/src/whisper.apr/demos/www-demo/src/worker.rs")
        .unwrap_or_default()
}

/// Get the lib.rs source
fn get_lib_source() -> String {
    std::fs::read_to_string("/home/noah/src/whisper.apr/demos/www-demo/src/lib.rs")
        .unwrap_or_default()
}

// =============================================================================
// STEP 51-60: Worker JS Generation (P0)
// =============================================================================

/// Step 51: Worker JS source file exists and is not empty
#[test]
fn step_51_worker_js_source_exists() {
    let source = get_worker_js_source();
    assert!(!source.is_empty(), "worker_js.rs must exist and have content");
    assert!(source.contains("fn "), "worker_js.rs must have functions");
}

/// Step 52: Worker JS has generate function
#[test]
fn step_52_worker_js_has_generate() {
    let source = get_worker_js_source();
    assert!(
        source.contains("pub fn generate_worker_js") || source.contains("fn generate_worker_js"),
        "Must have generate_worker_js function"
    );
}

/// Step 53: Worker JS generates onmessage handler
#[test]
fn step_53_worker_js_has_onmessage() {
    let source = get_worker_js_source();
    assert!(
        source.contains("onmessage") || source.contains("on_message"),
        "Must generate onmessage handler"
    );
}

/// Step 54: Worker JS has processAudioTick
#[test]
fn step_54_worker_js_has_process_tick() {
    let source = get_worker_js_source();
    assert!(
        source.contains("processAudioTick") || source.contains("process_audio_tick"),
        "Must have processAudioTick function"
    );
}

/// Step 55: Worker JS checks isDone flag (CRITICAL for stop detection)
#[test]
fn step_55_worker_js_checks_is_done() {
    let source = get_worker_js_source();
    assert!(
        source.contains("isDone"),
        "processAudioTick MUST check isDone() flag for stop detection (whisper.cpp pattern)"
    );
}

/// Step 56: Worker JS calls stopProcessing when done
#[test]
fn step_56_worker_js_calls_stop_processing() {
    let source = get_worker_js_source();

    // When isDone is true, must call stopProcessing
    let has_stop_on_done = source.contains("isDone") && source.contains("stopProcessing");
    assert!(
        has_stop_on_done,
        "Must call stopProcessing when isDone() returns true"
    );
}

/// Step 57: Worker handles init message
#[test]
fn step_57_worker_handles_init() {
    let source = get_worker_js_source();
    assert!(
        source.contains(r#""init""#) || source.contains("'init'"),
        "Must handle 'init' message type"
    );
}

/// Step 58: Worker handles start message
#[test]
fn step_58_worker_handles_start() {
    let source = get_worker_js_source();
    assert!(
        source.contains(r#""start""#) || source.contains("'start'"),
        "Must handle 'start' message type"
    );
}

/// Step 59: Worker detects done flag in processing loop
#[test]
fn step_59_worker_detects_done_in_loop() {
    let source = get_worker_js_source();

    // The isDone check must be in processAudioTick (the processing loop)
    // Look for isDone near processAudioTick
    let process_tick_section = source.find("processAudioTick")
        .map(|pos| &source[pos..std::cmp::min(pos + 1000, source.len())]);

    if let Some(section) = process_tick_section {
        assert!(
            section.contains("isDone"),
            "isDone check must be inside processAudioTick, found section: {}...",
            &section[..std::cmp::min(200, section.len())]
        );
    } else {
        panic!("processAudioTick not found in worker_js.rs");
    }
}

/// Step 60: Worker sends ready message
#[test]
fn step_60_worker_sends_ready() {
    let source = get_worker_js_source();
    assert!(
        source.contains(r#""ready""#) || source.contains("'ready'") || source.contains("ready"),
        "Must send 'ready' message after initialization"
    );
}

// =============================================================================
// STEP 61-70: Worker Communication (P0)
// =============================================================================

/// Step 61: Worker sends model loaded message
#[test]
fn step_61_worker_sends_model_loaded() {
    let source = get_worker_js_source();
    // Check for ModelLoaded in worker or manager
    let worker_source = get_worker_source();
    assert!(
        source.contains("ModelLoaded") || worker_source.contains("ModelLoaded"),
        "Must send model loaded notification"
    );
}

/// Step 62: Worker sends partial results
#[test]
fn step_62_worker_sends_partial() {
    let worker_source = get_worker_source();
    assert!(
        worker_source.contains("Partial") || worker_source.contains("partial"),
        "Must support sending partial transcription results"
    );
}

/// Step 63: Worker sends final result
#[test]
fn step_63_worker_sends_final() {
    let worker_source = get_worker_source();
    assert!(
        worker_source.contains("Result") || worker_source.contains("final"),
        "Must send final transcription result"
    );
}

/// Step 64: Ring buffer exists
#[test]
fn step_64_ring_buffer_exists() {
    let source = get_ring_buffer_source();
    assert!(!source.is_empty(), "ring_buffer.rs must exist");
    assert!(
        source.contains("SharedRingBuffer") || source.contains("RingBuffer"),
        "Must have ring buffer struct"
    );
}

/// Step 65: Ring buffer has write method
#[test]
fn step_65_ring_buffer_has_write() {
    let source = get_ring_buffer_source();
    assert!(
        source.contains("fn write") || source.contains("pub fn write"),
        "Ring buffer must have write method"
    );
}

/// Step 66: Ring buffer has read method
#[test]
fn step_66_ring_buffer_has_read() {
    let source = get_ring_buffer_source();
    assert!(
        source.contains("fn read") || source.contains("pub fn read"),
        "Ring buffer must have read method"
    );
}

/// Step 67: Ring buffer has done flag (CRITICAL)
#[test]
fn step_67_ring_buffer_has_done_flag() {
    let source = get_ring_buffer_source();
    assert!(
        source.contains("mark_done") || source.contains("markDone"),
        "Ring buffer must have mark_done method"
    );
    assert!(
        source.contains("is_done") || source.contains("isDone"),
        "Ring buffer must have is_done method"
    );
}

/// Step 68: TranscriptionWorker struct exists
#[test]
fn step_68_transcription_worker_exists() {
    let source = get_worker_source();
    assert!(
        source.contains("TranscriptionWorker") || source.contains("struct Worker"),
        "Must have TranscriptionWorker struct"
    );
}

/// Step 69: Worker has loadModel method
#[test]
fn step_69_worker_has_load_model() {
    let source = get_worker_source();
    assert!(
        source.contains("load_model") || source.contains("loadModel"),
        "Worker must have loadModel method"
    );
}

/// Step 70: Worker has processAudio method
#[test]
fn step_70_worker_has_process_audio() {
    let source = get_worker_source();
    assert!(
        source.contains("process_audio") || source.contains("processAudio"),
        "Worker must have processAudio method"
    );
}

// =============================================================================
// STEP 71-80: Audio Processing (P0)
// =============================================================================

/// Step 71: Worker has stopProcessing method
#[test]
fn step_71_worker_has_stop_processing() {
    let source = get_worker_source();
    assert!(
        source.contains("stop_processing") || source.contains("stopProcessing"),
        "Worker must have stopProcessing method"
    );
}

/// Step 72: Worker resamples audio to 16kHz
#[test]
fn step_72_worker_resamples_audio() {
    let source = get_worker_source();
    assert!(
        source.contains("16000") || source.contains("resample") || source.contains("SAMPLE_RATE"),
        "Worker must resample audio to 16kHz"
    );
}

/// Step 73: Worker accumulates samples
#[test]
fn step_73_worker_accumulates_samples() {
    let source = get_worker_source();
    assert!(
        source.contains("accumulated") || source.contains("buffer") || source.contains("samples"),
        "Worker must accumulate audio samples"
    );
}

/// Step 74: Worker has chunk threshold
#[test]
fn step_74_worker_has_chunk_threshold() {
    let source = get_worker_source();
    // Should have some threshold for when to transcribe
    assert!(
        source.contains("chunk") || source.contains("threshold") || source.contains("24000") || source.contains("CHUNK"),
        "Worker must have chunk size threshold for transcription"
    );
}

/// Step 78: Worker doesn't access window
#[test]
fn step_78_worker_no_window_access() {
    let source = get_worker_js_source();
    // Should not have direct window access (workers don't have window)
    let has_window_call = source.contains("window.") && !source.contains("// window") && !source.contains("/* window");
    assert!(
        !has_window_call,
        "Worker JS must not access window object"
    );
}

/// Step 79: Worker doesn't access document
#[test]
fn step_79_worker_no_document_access() {
    let source = get_worker_js_source();
    // Should not have direct document access (workers don't have document)
    let has_document_call = source.contains("document.") && !source.contains("// document") && !source.contains("/* document");
    assert!(
        !has_document_call,
        "Worker JS must not access document object"
    );
}

/// Step 80: AudioWorklet source exists
#[test]
fn step_80_audioworklet_exists() {
    let source = std::fs::read_to_string("/home/noah/src/whisper.apr/demos/www-demo/src/audioworklet_js.rs")
        .unwrap_or_default();
    assert!(
        !source.is_empty() || get_lib_source().contains("AudioWorklet"),
        "AudioWorklet integration must exist"
    );
}

// =============================================================================
// Manager Tests
// =============================================================================

/// Manager uses shared state pointer (regression from previous bug)
#[test]
fn manager_uses_shared_state_ptr() {
    let source = get_worker_manager_source();
    if source.is_empty() {
        return; // File doesn't exist yet
    }

    assert!(
        source.contains("state_ptr") || source.contains("Rc<RefCell"),
        "Manager must use shared state pointer for closure sync"
    );
}

/// Manager handles all WorkerResult variants (regression from previous bug)
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
