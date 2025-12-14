//! Worker Bridge - Main thread communication with Transcription Worker
//!
//! Provides non-blocking communication between the main thread and the
//! dedicated transcription Web Worker.
//!
//! # Architecture
//!
//! ```text
//! Main Thread                          Worker Thread
//! ┌────────────────┐                   ┌────────────────┐
//! │  WorkerBridge  │───postMessage────>│  worker_entry  │
//! │                │<──postMessage─────│                │
//! └────────────────┘                   └────────────────┘
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use tracing::{debug, error, info, warn};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{MessageEvent, Worker};

/// Message types sent to the worker
#[derive(Debug, Clone)]
pub enum WorkerCommand {
    /// Load model from bytes
    LoadModel,
    /// Start a new transcription session
    StartSession { session_id: String },
    /// Transcribe audio chunk
    Transcribe {
        session_id: String,
        chunk_id: u32,
        prompt_tokens: Vec<u32>,
        is_final: bool,
    },
    /// End transcription session
    EndSession { session_id: String },
    /// Configure transcription options
    SetOptions {
        language: Option<String>,
        task: String,
    },
    /// Shutdown worker
    Shutdown,
    /// Ping for latency testing
    Ping { timestamp: f64 },
}

/// Message types received from the worker
#[derive(Debug, Clone)]
pub enum WorkerResult {
    /// Worker is ready
    Ready,
    /// Model loaded successfully
    ModelLoaded { size_mb: f64, load_time_ms: f64 },
    /// Session started
    SessionStarted { session_id: String },
    /// Transcription result
    Transcription {
        session_id: String,
        chunk_id: u32,
        text: String,
        tokens: Vec<u32>,
        is_partial: bool,
        rtf: f64,
    },
    /// Session ended
    SessionEnded { session_id: String },
    /// Error occurred
    Error {
        session_id: Option<String>,
        chunk_id: Option<u32>,
        message: String,
    },
    /// Metrics update
    Metrics {
        queue_depth: usize,
        avg_latency_ms: f64,
    },
    /// Pong response
    Pong { timestamp: f64, worker_time: f64 },
}

/// Callback type for worker results
pub type ResultCallback = Rc<RefCell<dyn FnMut(WorkerResult)>>;

/// Pending request tracking
struct PendingRequest {
    chunk_id: u32,
    sent_at: f64,
}

/// Bridge for communicating with the transcription worker
pub struct WorkerBridge {
    worker: Worker,
    pending: HashMap<u32, PendingRequest>,
    next_chunk_id: u32,
    result_callback: Option<ResultCallback>,
    ready: bool,
    _on_message: Closure<dyn Fn(MessageEvent)>,
    _on_error: Closure<dyn Fn(web_sys::ErrorEvent)>,
}

impl WorkerBridge {
    /// Create a new worker bridge
    ///
    /// # Arguments
    ///
    /// * `wasm_url` - URL to the WASM module (e.g., "/pkg/whisper_apr_demo_realtime_transcription.js")
    ///
    /// # Errors
    ///
    /// Returns error if worker creation fails.
    pub fn new(wasm_url: &str) -> Result<Rc<RefCell<Self>>, JsValue> {
        info!(wasm_url, "Creating WorkerBridge");
        let start_time = js_sys::Date::now();

        // Create worker bootstrap script (minimal JS - only imports)
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

        let blob_options = web_sys::BlobPropertyBag::new();
        blob_options.set_type("application/javascript");

        let blob = web_sys::Blob::new_with_str_sequence_and_options(&blob_parts, &blob_options)?;
        let blob_url = web_sys::Url::create_object_url_with_blob(&blob)?;

        // Create worker with module type
        let worker_options = web_sys::WorkerOptions::new();
        worker_options.set_type(web_sys::WorkerType::Module);

        let worker = Worker::new_with_options(&blob_url, &worker_options)?;

        // Clean up blob URL (worker has already loaded it)
        let _ = web_sys::Url::revoke_object_url(&blob_url);

        // Create bridge instance wrapped in Rc<RefCell> for shared ownership
        let bridge = Rc::new(RefCell::new(Self {
            worker: worker.clone(),
            pending: HashMap::new(),
            next_chunk_id: 0,
            result_callback: None,
            ready: false,
            // Placeholders - will be replaced below
            _on_message: Closure::wrap(Box::new(|_: MessageEvent| {}) as Box<dyn Fn(MessageEvent)>),
            _on_error: Closure::wrap(
                Box::new(|_: web_sys::ErrorEvent| {}) as Box<dyn Fn(web_sys::ErrorEvent)>
            ),
        }));

        // Set up message handler
        let bridge_for_message = bridge.clone();
        let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
            let data = event.data();
            if let Err(e) = Self::handle_message(&bridge_for_message, data) {
                error!(error = ?e, "Failed to handle worker message");
            }
        }) as Box<dyn Fn(MessageEvent)>);

        worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));

        // Set up error handler
        let bridge_for_error = bridge.clone();
        let on_error = Closure::wrap(Box::new(move |event: web_sys::ErrorEvent| {
            let message = event.message();
            error!(message = %message, "Worker error");
            let mut bridge = bridge_for_error.borrow_mut();
            if let Some(ref callback) = bridge.result_callback {
                let mut cb = callback.borrow_mut();
                cb(WorkerResult::Error {
                    session_id: None,
                    chunk_id: None,
                    message,
                });
            }
        }) as Box<dyn Fn(web_sys::ErrorEvent)>);

        worker.set_onerror(Some(on_error.as_ref().unchecked_ref()));

        // Store closures to prevent them from being dropped
        {
            let mut bridge_mut = bridge.borrow_mut();
            bridge_mut._on_message = on_message;
            bridge_mut._on_error = on_error;
        }

        let elapsed = js_sys::Date::now() - start_time;
        info!(elapsed_ms = elapsed, "WorkerBridge created");

        Ok(bridge)
    }

    /// Handle incoming message from worker
    fn handle_message(bridge: &Rc<RefCell<Self>>, data: JsValue) -> Result<(), JsValue> {
        let obj = data
            .dyn_ref::<js_sys::Object>()
            .ok_or_else(|| JsValue::from_str("Expected object from worker"))?;

        let msg_type = js_sys::Reflect::get(obj, &"type".into())?
            .as_string()
            .ok_or_else(|| JsValue::from_str("Missing message type"))?;

        debug!(msg_type = %msg_type, "Received worker message");

        let result = match msg_type.as_str() {
            "ready" => {
                bridge.borrow_mut().ready = true;
                info!("Worker ready");
                WorkerResult::Ready
            }
            "model_loaded" => {
                let data_str = js_sys::Reflect::get(obj, &"data".into())?
                    .as_string()
                    .unwrap_or_default();
                info!(data = %data_str, "Model loaded in worker");
                WorkerResult::ModelLoaded {
                    size_mb: 0.0,      // TODO: Parse from data
                    load_time_ms: 0.0, // TODO: Parse from data
                }
            }
            "transcription" => {
                let text = js_sys::Reflect::get(obj, &"data".into())?
                    .as_string()
                    .unwrap_or_default();
                let chunk_id = js_sys::Reflect::get(obj, &"chunk_id".into())
                    .ok()
                    .and_then(|v| v.as_f64())
                    .map(|v| v as u32)
                    .unwrap_or(0);
                let rtf = js_sys::Reflect::get(obj, &"rtf".into())
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);

                // Remove from pending
                bridge.borrow_mut().pending.remove(&chunk_id);

                WorkerResult::Transcription {
                    session_id: String::new(),
                    chunk_id,
                    text,
                    tokens: Vec::new(),
                    is_partial: false,
                    rtf,
                }
            }
            "pong" => {
                let timestamp = js_sys::Reflect::get(obj, &"timestamp".into())
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let worker_time = js_sys::Reflect::get(obj, &"worker_time".into())
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                WorkerResult::Pong {
                    timestamp,
                    worker_time,
                }
            }
            "error" => {
                let message = js_sys::Reflect::get(obj, &"error".into())?
                    .as_string()
                    .unwrap_or_else(|| "Unknown error".to_string());
                let chunk_id = js_sys::Reflect::get(obj, &"chunk_id".into())
                    .ok()
                    .and_then(|v| v.as_f64())
                    .map(|v| v as u32);

                warn!(message = %message, chunk_id = ?chunk_id, "Worker error");

                // Remove from pending if chunk_id present
                if let Some(id) = chunk_id {
                    bridge.borrow_mut().pending.remove(&id);
                }

                WorkerResult::Error {
                    session_id: None,
                    chunk_id,
                    message,
                }
            }
            _ => {
                warn!(msg_type = %msg_type, "Unknown message type from worker");
                return Ok(());
            }
        };

        // Invoke callback
        let bridge_ref = bridge.borrow();
        if let Some(ref callback) = bridge_ref.result_callback {
            let mut cb = callback.borrow_mut();
            cb(result);
        }

        Ok(())
    }

    /// Set the callback for worker results
    pub fn set_result_callback<F>(&mut self, callback: F)
    where
        F: FnMut(WorkerResult) + 'static,
    {
        self.result_callback = Some(Rc::new(RefCell::new(callback)));
    }

    /// Check if worker is ready
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Send model bytes to worker for loading
    ///
    /// # Errors
    ///
    /// Returns error if message sending fails.
    pub fn load_model(&self, model_bytes: &[u8]) -> Result<(), JsValue> {
        info!(size_bytes = model_bytes.len(), "Sending model to worker");

        let array = js_sys::Uint8Array::from(model_bytes);
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"type".into(), &"load_model".into())?;
        js_sys::Reflect::set(&obj, &"data".into(), &array)?;

        self.worker.post_message(&obj)
    }

    /// Send audio chunk for transcription
    ///
    /// Returns the chunk_id for tracking.
    ///
    /// # Errors
    ///
    /// Returns error if message sending fails.
    pub fn transcribe(
        &mut self,
        audio: &[f32],
        session_id: &str,
        prompt_tokens: &[u32],
        is_final: bool,
    ) -> Result<u32, JsValue> {
        let chunk_id = self.next_chunk_id;
        self.next_chunk_id = self.next_chunk_id.wrapping_add(1);

        debug!(
            chunk_id,
            audio_samples = audio.len(),
            session_id,
            "Sending audio to worker"
        );

        let audio_array = js_sys::Float32Array::from(audio);
        let tokens_array = js_sys::Uint32Array::from(prompt_tokens);

        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"type".into(), &"transcribe".into())?;
        js_sys::Reflect::set(&obj, &"audio".into(), &audio_array)?;
        js_sys::Reflect::set(&obj, &"chunk_id".into(), &JsValue::from(chunk_id))?;
        js_sys::Reflect::set(&obj, &"session_id".into(), &session_id.into())?;
        js_sys::Reflect::set(&obj, &"prompt_tokens".into(), &tokens_array)?;
        js_sys::Reflect::set(&obj, &"is_final".into(), &JsValue::from(is_final))?;

        // Use Transferable for zero-copy transfer
        let transfer = js_sys::Array::new();
        transfer.push(&audio_array.buffer());

        self.worker.post_message_with_transfer(&obj, &transfer)?;

        // Track pending request
        self.pending.insert(
            chunk_id,
            PendingRequest {
                chunk_id,
                sent_at: js_sys::Date::now(),
            },
        );

        Ok(chunk_id)
    }

    /// Send ping to measure round-trip latency
    ///
    /// # Errors
    ///
    /// Returns error if message sending fails.
    pub fn ping(&self) -> Result<f64, JsValue> {
        let timestamp = js_sys::Date::now();

        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"type".into(), &"ping".into())?;
        js_sys::Reflect::set(&obj, &"timestamp".into(), &JsValue::from(timestamp))?;

        self.worker.post_message(&obj)?;

        Ok(timestamp)
    }

    /// Shutdown the worker
    ///
    /// # Errors
    ///
    /// Returns error if message sending fails.
    pub fn shutdown(&self) -> Result<(), JsValue> {
        info!("Shutting down worker");

        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"type".into(), &"shutdown".into())?;

        self.worker.post_message(&obj)
    }

    /// Get number of pending requests
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Terminate the worker immediately
    pub fn terminate(&self) {
        info!("Terminating worker");
        self.worker.terminate();
    }
}

impl Drop for WorkerBridge {
    fn drop(&mut self) {
        info!("WorkerBridge dropped, terminating worker");
        self.worker.terminate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_command_debug() {
        let cmd = WorkerCommand::Ping { timestamp: 0.0 };
        let debug_str = format!("{:?}", cmd);
        assert!(debug_str.contains("Ping"));
    }

    #[test]
    fn test_worker_result_debug() {
        let result = WorkerResult::Ready;
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("Ready"));
    }

    #[test]
    fn test_worker_result_error_has_fields() {
        let result = WorkerResult::Error {
            session_id: Some("test".to_string()),
            chunk_id: Some(42),
            message: "Test error".to_string(),
        };
        if let WorkerResult::Error {
            session_id,
            chunk_id,
            message,
        } = result
        {
            assert_eq!(session_id, Some("test".to_string()));
            assert_eq!(chunk_id, Some(42));
            assert_eq!(message, "Test error");
        } else {
            panic!("Expected Error variant");
        }
    }

    #[test]
    fn test_worker_result_transcription_has_fields() {
        let result = WorkerResult::Transcription {
            session_id: "sess1".to_string(),
            chunk_id: 1,
            text: "hello".to_string(),
            tokens: vec![1, 2, 3],
            is_partial: false,
            rtf: 0.5,
        };
        if let WorkerResult::Transcription {
            session_id,
            chunk_id,
            text,
            tokens,
            is_partial,
            rtf,
        } = result
        {
            assert_eq!(session_id, "sess1");
            assert_eq!(chunk_id, 1);
            assert_eq!(text, "hello");
            assert_eq!(tokens, vec![1, 2, 3]);
            assert!(!is_partial);
            assert!((rtf - 0.5).abs() < f64::EPSILON);
        } else {
            panic!("Expected Transcription variant");
        }
    }
}
