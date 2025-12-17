//! Transcription Web Worker
//!
//! Runs transcription in a dedicated background thread to prevent UI blocking.
//! The worker loads its own copy of the model and processes audio asynchronously.
//!
//! # Architecture
//!
//! ```text
//! Main Thread                    Worker Thread
//! ┌──────────────┐              ┌──────────────┐
//! │ Audio Capture│──postMessage─>│ Transcribe   │
//! │ UI Updates   │<─postMessage──│ Model        │
//! └──────────────┘              └──────────────┘
//! ```
//!
//! # Message Protocol
//!
//! ## Commands (Main -> Worker)
//! - `load_model`: Load model from Uint8Array
//! - `transcribe`: Transcribe Float32Array audio with session context
//! - `ping`: Latency test
//! - `shutdown`: Graceful shutdown
//!
//! ## Results (Worker -> Main)
//! - `ready`: Worker initialized
//! - `model_loaded`: Model ready
//! - `transcription`: Transcription result with context
//! - `pong`: Latency response
//! - `error`: Error with context

use std::cell::RefCell;
use std::collections::HashMap;

use tracing::{debug, error, info, warn};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

use whisper_apr::{TranscribeOptions, WhisperApr};

/// Session state for context management
struct TranscriptionSession {
    /// Last N tokens for context seeding
    last_tokens: Vec<u32>,
    /// Number of chunks processed
    chunks_processed: u32,
}

impl TranscriptionSession {
    fn new() -> Self {
        Self {
            last_tokens: Vec::new(),
            chunks_processed: 0,
        }
    }
}

thread_local! {
    static WORKER_MODEL: RefCell<Option<WhisperApr>> = const { RefCell::new(None) };
    static SESSIONS: RefCell<HashMap<String, TranscriptionSession>> = RefCell::new(HashMap::new());
}

/// Message types for worker communication
#[wasm_bindgen]
pub struct WorkerMessage;

/// Entry point called when running as a worker
#[wasm_bindgen]
pub fn worker_entry() {
    console_error_panic_hook::set_once();

    // Initialize tracing for worker context
    // Note: tracing_wasm may not work in worker context, use console fallback
    let _ = tracing_wasm::try_set_as_global_default();

    info!("Worker entry point called");

    let global = js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>();

    let onmessage: Closure<dyn Fn(MessageEvent)> = Closure::new(move |event: MessageEvent| {
        let data = event.data();

        // Handle different message types
        if let Some(obj) = data.dyn_ref::<js_sys::Object>() {
            let msg_type = js_sys::Reflect::get(obj, &"type".into())
                .ok()
                .and_then(|v| v.as_string());

            match msg_type.as_deref() {
                Some("load_model") => handle_load_model(obj),
                Some("transcribe") => handle_transcribe(obj),
                Some("ping") => handle_ping(obj),
                Some("start_session") => handle_start_session(obj),
                Some("end_session") => handle_end_session(obj),
                Some("shutdown") => handle_shutdown(),
                Some(other) => {
                    warn!(msg_type = other, "Unknown message type");
                }
                None => {
                    warn!("Message missing type field");
                }
            }
        }
    });

    global.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
    onmessage.forget();

    // Signal worker is ready
    post_result("ready", "Worker initialized");
    info!("Worker ready");
}

/// Handle model loading
fn handle_load_model(obj: &js_sys::Object) {
    info!("Loading model in worker");
    let start_time = js_sys::Date::now();

    if let Ok(buffer) = js_sys::Reflect::get(obj, &"data".into()) {
        if let Some(array) = buffer.dyn_ref::<js_sys::Uint8Array>() {
            let bytes = array.to_vec();
            let size_mb = bytes.len() as f64 / 1_000_000.0;

            match WhisperApr::load_from_apr(&bytes) {
                Ok(model) => {
                    WORKER_MODEL.with(|m| *m.borrow_mut() = Some(model));
                    let load_time_ms = js_sys::Date::now() - start_time;
                    info!(size_mb, load_time_ms, "Model loaded in worker");
                    post_model_loaded(size_mb, load_time_ms);
                }
                Err(e) => {
                    error!(error = ?e, "Model load failed");
                    post_error(&format!("Model load failed: {e:?}"), None);
                }
            }
        } else {
            post_error("Invalid model data format", None);
        }
    } else {
        post_error("Missing model data", None);
    }
}

/// Handle transcription request
fn handle_transcribe(obj: &js_sys::Object) {
    let chunk_id = js_sys::Reflect::get(obj, &"chunk_id".into())
        .ok()
        .and_then(|v| v.as_f64())
        .map(|v| v as u32)
        .unwrap_or(0);

    let session_id = js_sys::Reflect::get(obj, &"session_id".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();

    let is_final = js_sys::Reflect::get(obj, &"is_final".into())
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    debug!(chunk_id, session_id = %session_id, is_final, "Processing transcription");

    // Get audio data
    let audio = match js_sys::Reflect::get(obj, &"audio".into()) {
        Ok(buffer) => {
            if let Some(array) = buffer.dyn_ref::<js_sys::Float32Array>() {
                array.to_vec()
            } else {
                post_error("Invalid audio format", Some(chunk_id));
                return;
            }
        }
        Err(_) => {
            post_error("Missing audio data", Some(chunk_id));
            return;
        }
    };

    // Get prompt tokens for context
    let prompt_tokens: Vec<u32> = js_sys::Reflect::get(obj, &"prompt_tokens".into())
        .ok()
        .and_then(|v| v.dyn_ref::<js_sys::Uint32Array>().map(|a| a.to_vec()))
        .unwrap_or_default();

    let audio_duration = audio.len() as f64 / 16000.0;
    let start_time = js_sys::Date::now();

    // Post acknowledgment immediately to confirm message received
    post_processing_started(chunk_id, audio.len());

    WORKER_MODEL.with(|model_cell| {
        let model = model_cell.borrow();
        if let Some(model) = model.as_ref() {
            info!(chunk_id, audio_len = audio.len(), "Starting transcription");

            // Build transcription options with context
            let options = TranscribeOptions::default();

            // Seed with prompt tokens if available
            if !prompt_tokens.is_empty() {
                debug!(
                    prompt_token_count = prompt_tokens.len(),
                    "Using prompt tokens for context"
                );
                // Note: TranscribeOptions may need to support prompt_tokens
                // For now we use default options
            }

            info!(chunk_id, "Calling model.transcribe()...");

            // Call transcribe directly (no diagnostic mel/encode steps to avoid double work)
            match model.transcribe(&audio, options) {
                Ok(result) => {
                    let elapsed_ms = js_sys::Date::now() - start_time;
                    let rtf = (elapsed_ms / 1000.0) / audio_duration;

                    info!(
                        chunk_id,
                        text = %result.text,
                        elapsed_ms,
                        rtf,
                        "Transcription complete"
                    );

                    // Update session state
                    SESSIONS.with(|sessions| {
                        let mut sessions = sessions.borrow_mut();
                        let session = sessions
                            .entry(session_id.clone())
                            .or_insert_with(TranscriptionSession::new);
                        session.chunks_processed += 1;
                        // Store last tokens for next chunk context (max 224 per Whisper spec)
                        // Note: result.tokens would need to be exposed by WhisperApr
                        session.last_tokens.clear();
                    });

                    post_transcription(chunk_id, &session_id, &result.text, rtf, is_final);
                }
                Err(e) => {
                    error!(chunk_id, error = %e, "Transcription failed");
                    post_error(&format!("Transcription failed: {e:?}"), Some(chunk_id));
                }
            }
        } else {
            post_error("Model not loaded", Some(chunk_id));
        }
    });
}

/// Handle ping for latency measurement
fn handle_ping(obj: &js_sys::Object) {
    let timestamp = js_sys::Reflect::get(obj, &"timestamp".into())
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let worker_time = js_sys::Date::now();

    debug!(timestamp, worker_time, "Ping received");

    post_pong(timestamp, worker_time);
}

/// Handle session start
fn handle_start_session(obj: &js_sys::Object) {
    let session_id = js_sys::Reflect::get(obj, &"session_id".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();

    info!(session_id = %session_id, "Starting session");

    SESSIONS.with(|sessions| {
        sessions
            .borrow_mut()
            .insert(session_id.clone(), TranscriptionSession::new());
    });

    post_session_started(&session_id);
}

/// Handle session end
fn handle_end_session(obj: &js_sys::Object) {
    let session_id = js_sys::Reflect::get(obj, &"session_id".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();

    info!(session_id = %session_id, "Ending session");

    SESSIONS.with(|sessions| {
        sessions.borrow_mut().remove(&session_id);
    });

    post_session_ended(&session_id);
}

/// Handle shutdown
fn handle_shutdown() {
    info!("Worker shutting down");

    // Clear model to free memory
    WORKER_MODEL.with(|m| *m.borrow_mut() = None);

    // Clear sessions
    SESSIONS.with(|s| s.borrow_mut().clear());

    // Close the worker
    let global = js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>();
    global.close();
}

// ============================================================================
// Message Posting Helpers
// ============================================================================

fn post_result(msg_type: &str, data: &str) {
    let global = js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>();
    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&obj, &"type".into(), &msg_type.into());
    let _ = js_sys::Reflect::set(&obj, &"data".into(), &data.into());
    let _ = global.post_message(&obj);
}

fn post_model_loaded(size_mb: f64, load_time_ms: f64) {
    let global = js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>();
    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&obj, &"type".into(), &"model_loaded".into());
    let _ = js_sys::Reflect::set(&obj, &"size_mb".into(), &JsValue::from(size_mb));
    let _ = js_sys::Reflect::set(&obj, &"load_time_ms".into(), &JsValue::from(load_time_ms));
    let _ = js_sys::Reflect::set(
        &obj,
        &"data".into(),
        &format!("Loaded {size_mb:.2}MB in {load_time_ms:.0}ms").into(),
    );
    let _ = global.post_message(&obj);
}

/// Post acknowledgment that processing has started (for diagnostic purposes)
fn post_processing_started(chunk_id: u32, audio_samples: usize) {
    let global = js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>();
    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&obj, &"type".into(), &"processing_started".into());
    let _ = js_sys::Reflect::set(&obj, &"chunk_id".into(), &JsValue::from(chunk_id));
    let _ = js_sys::Reflect::set(&obj, &"audio_samples".into(), &JsValue::from(audio_samples as f64));
    let _ = global.post_message(&obj);
}

fn post_transcription(chunk_id: u32, session_id: &str, text: &str, rtf: f64, is_partial: bool) {
    let global = js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>();
    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&obj, &"type".into(), &"transcription".into());
    let _ = js_sys::Reflect::set(&obj, &"chunk_id".into(), &JsValue::from(chunk_id));
    let _ = js_sys::Reflect::set(&obj, &"session_id".into(), &session_id.into());
    let _ = js_sys::Reflect::set(&obj, &"data".into(), &text.into());
    let _ = js_sys::Reflect::set(&obj, &"rtf".into(), &JsValue::from(rtf));
    let _ = js_sys::Reflect::set(&obj, &"is_partial".into(), &JsValue::from(is_partial));
    let _ = global.post_message(&obj);
}

fn post_pong(timestamp: f64, worker_time: f64) {
    let global = js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>();
    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&obj, &"type".into(), &"pong".into());
    let _ = js_sys::Reflect::set(&obj, &"timestamp".into(), &JsValue::from(timestamp));
    let _ = js_sys::Reflect::set(&obj, &"worker_time".into(), &JsValue::from(worker_time));
    let _ = global.post_message(&obj);
}

fn post_session_started(session_id: &str) {
    let global = js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>();
    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&obj, &"type".into(), &"session_started".into());
    let _ = js_sys::Reflect::set(&obj, &"session_id".into(), &session_id.into());
    let _ = global.post_message(&obj);
}

fn post_session_ended(session_id: &str) {
    let global = js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>();
    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&obj, &"type".into(), &"session_ended".into());
    let _ = js_sys::Reflect::set(&obj, &"session_id".into(), &session_id.into());
    let _ = global.post_message(&obj);
}

fn post_error(error: &str, chunk_id: Option<u32>) {
    let global = js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>();
    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&obj, &"type".into(), &"error".into());
    let _ = js_sys::Reflect::set(&obj, &"error".into(), &error.into());
    if let Some(id) = chunk_id {
        let _ = js_sys::Reflect::set(&obj, &"chunk_id".into(), &JsValue::from(id));
    }
    let _ = global.post_message(&obj);
}

// ============================================================================
// Legacy API (for backwards compatibility during transition)
// ============================================================================

/// Create a transcription worker (legacy API)
///
/// Returns a Worker that can process transcription requests without blocking.
#[wasm_bindgen]
pub fn create_transcription_worker(
    wasm_url: &str,
    on_message: &js_sys::Function,
) -> Result<web_sys::Worker, JsValue> {
    // Create worker script that loads our WASM module
    let worker_script = format!(
        r#"
        import init, {{ worker_entry }} from '{}';

        (async () => {{
            await init();
            worker_entry();
        }})();
        "#,
        wasm_url
    );

    // Create blob URL for worker
    let blob_parts = js_sys::Array::new();
    blob_parts.push(&worker_script.into());

    let options = web_sys::BlobPropertyBag::new();
    options.set_type("application/javascript");

    let blob = web_sys::Blob::new_with_str_sequence_and_options(&blob_parts, &options)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)?;

    // Create worker with module type
    let worker_opts = web_sys::WorkerOptions::new();
    worker_opts.set_type(web_sys::WorkerType::Module);

    let worker = web_sys::Worker::new_with_options(&url, &worker_opts)?;

    // Set up message handler
    let callback = on_message.clone();
    let onmessage: Closure<dyn Fn(MessageEvent)> = Closure::new(move |event: MessageEvent| {
        let _ = callback.call1(&JsValue::NULL, &event.data());
    });

    worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
    onmessage.forget();

    // Clean up blob URL
    let _ = web_sys::Url::revoke_object_url(&url);

    Ok(worker)
}

/// Send model data to worker for loading (legacy API)
#[wasm_bindgen]
pub fn worker_load_model(worker: &web_sys::Worker, model_bytes: &[u8]) -> Result<(), JsValue> {
    let array = js_sys::Uint8Array::from(model_bytes);
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(&obj, &"type".into(), &"load_model".into())?;
    js_sys::Reflect::set(&obj, &"data".into(), &array)?;
    worker.post_message(&obj)
}

/// Send audio to worker for transcription (legacy API)
#[wasm_bindgen]
pub fn worker_transcribe(worker: &web_sys::Worker, audio: &[f32]) -> Result<(), JsValue> {
    let array = js_sys::Float32Array::from(audio);
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(&obj, &"type".into(), &"transcribe".into())?;
    js_sys::Reflect::set(&obj, &"audio".into(), &array)?;
    worker.post_message(&obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcription_session_new() {
        let session = TranscriptionSession::new();
        assert!(session.last_tokens.is_empty());
        assert_eq!(session.chunks_processed, 0);
    }
}
