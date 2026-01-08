//! Whisper.apr Demo - WASM Speech Recognition
//!
//! WAPR-DEMO-REBUILD-TDD: Test-first implementation
//! Reference: whisper.cpp stream.wasm (simple polling architecture)
//!
//! # Brick Architecture (PROBAR-SPEC-009)
//!
//! This demo uses the Brick Architecture where tests ARE the interface.
//! All UI components are defined by their assertions and budgets.

#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use wasm_bindgen::prelude::*;

pub mod audioworklet_js;
pub mod bricks;
pub mod ring_buffer;
pub mod worker;
pub mod worker_js;
pub mod worker_manager;

// Re-export types for JS access
pub use ring_buffer::SharedRingBuffer;
pub use worker::TranscriptionWorker;
pub use worker_manager::WorkerManager;

/// Initialize a worker instance (called from worker JS)
#[wasm_bindgen(js_name = initWorker)]
pub fn init_worker() -> TranscriptionWorker {
    TranscriptionWorker::init_worker()
}

/// Initialize the demo when WASM loads
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    // Initialize console error panic hook for better error messages
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    web_sys::console::log_1(&"[WASM] start() called".into());

    // TODO: Initialize UI and state machine
    // This will be implemented to pass the UX flow tests

    Ok(())
}
