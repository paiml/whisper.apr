//! Whisper.apr Demo - WASM Speech Recognition
//!
//! WAPR-DEMO-REBUILD-TDD: Test-first implementation
//! Reference: whisper.cpp stream.wasm (simple polling architecture)
//!
//! # Brick Architecture (PROBAR-SPEC-009)
//!
//! This demo uses the Brick Architecture where tests ARE the interface.
//! All UI components are defined by their assertions and budgets.
//!
//! PROBAR-SPEC-009: Uses tracing for structured logging instead of console.log.

#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use tracing::info;
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

// Re-export brick-generated JS functions (PROBAR-SPEC-009-P7)
// These replace the hand-written worker_js and audioworklet_js modules
pub use bricks::{
    generate_audioworklet_js_from_brick, generate_worker_js_from_brick,
    create_whisper_audio_brick, create_whisper_event_brick, create_whisper_worker_brick,
};

/// Initialize a worker instance (called from worker JS)
#[wasm_bindgen(js_name = initWorker)]
#[must_use] 
pub fn init_worker() -> TranscriptionWorker {
    TranscriptionWorker::init_worker()
}

/// Initialize the demo when WASM loads
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    // Initialize console error panic hook for better error messages
    // Note: This feature is optional and may not be compiled in
    #[allow(unexpected_cfgs)]
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    // Initialize tracing for browser devtools
    tracing_wasm::set_as_global_default();

    info!("WASM start() called");

    // UI and state machine initialization tracked in WAPR-UI-001
    // Implementation will pass UX flow tests

    Ok(())
}
