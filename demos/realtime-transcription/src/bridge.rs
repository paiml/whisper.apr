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

/// Maximum number of pending chunks before dropping (per spec Section 3.2)
pub const MAX_QUEUE_DEPTH: usize = 3;

/// Pending request tracking
struct PendingRequest {
    chunk_id: u32,
    sent_at: f64,
}

/// Queue statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    /// Total chunks sent to worker
    pub chunks_sent: u64,
    /// Total chunks dropped due to queue overflow
    pub chunks_dropped: u64,
    /// Total chunks completed successfully
    pub chunks_completed: u64,
    /// Total errors received
    pub errors: u64,
    /// Average round-trip latency in ms
    pub avg_latency_ms: f64,
    /// Current queue depth
    pub queue_depth: usize,
    /// Whether processing has started (diagnostic)
    pub processing_started: bool,
}

/// Maximum consecutive errors before worker is considered unhealthy
pub const MAX_CONSECUTIVE_ERRORS: u32 = 3;

/// Bridge for communicating with the transcription worker
pub struct WorkerBridge {
    worker: Worker,
    pending: HashMap<u32, PendingRequest>,
    next_chunk_id: u32,
    result_callback: Option<ResultCallback>,
    ready: bool,
    stats: QueueStats,
    latency_samples: Vec<f64>,
    /// Consecutive error count for health checking
    consecutive_errors: u32,
    /// Whether the worker has encountered a fatal error
    fatal_error: bool,
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

        // Get the full origin URL for proper module resolution in worker context
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window"))?;
        let location = window.location();
        let origin = location.origin()?;
        let full_wasm_url = format!("{}{}", origin, wasm_url);

        // Create worker bootstrap script with error handling
        let worker_script = format!(
            r#"
console.log('[Worker] Bootstrap starting, loading from: {}');

(async () => {{
    try {{
        const module = await import('{}');
        console.log('[Worker] Module imported successfully');
        await module.default();
        console.log('[Worker] WASM initialized');
        module.worker_entry();
        console.log('[Worker] Entry point called');
    }} catch (error) {{
        console.error('[Worker] Bootstrap failed:', error);
        self.postMessage({{ type: 'error', error: 'Bootstrap failed: ' + error.message }});
    }}
}})();
"#,
            full_wasm_url, full_wasm_url
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
            stats: QueueStats::default(),
            latency_samples: Vec::with_capacity(100),
            consecutive_errors: 0,
            fatal_error: false,
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
                let size_mb = js_sys::Reflect::get(obj, &"size_mb".into())
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let load_time_ms = js_sys::Reflect::get(obj, &"load_time_ms".into())
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                info!(size_mb, load_time_ms, "Model loaded in worker");
                WorkerResult::ModelLoaded {
                    size_mb,
                    load_time_ms,
                }
            }
            "processing_started" => {
                let chunk_id = js_sys::Reflect::get(obj, &"chunk_id".into())
                    .ok()
                    .and_then(|v| v.as_f64())
                    .map(|v| v as u32)
                    .unwrap_or(0);
                let audio_samples = js_sys::Reflect::get(obj, &"audio_samples".into())
                    .ok()
                    .and_then(|v| v.as_f64())
                    .map(|v| v as usize)
                    .unwrap_or(0);
                info!(chunk_id, audio_samples, "Worker started processing chunk");
                // Mark this in stats for diagnostics
                bridge.borrow_mut().stats.processing_started = true;
                return Ok(()); // Don't invoke callback for internal diagnostic messages
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

                // Calculate latency and update stats
                {
                    let mut bridge_mut = bridge.borrow_mut();
                    if let Some(pending) = bridge_mut.pending.remove(&chunk_id) {
                        let latency = js_sys::Date::now() - pending.sent_at;
                        bridge_mut.latency_samples.push(latency);
                        // Keep only last 100 samples
                        if bridge_mut.latency_samples.len() > 100 {
                            bridge_mut.latency_samples.remove(0);
                        }
                        // Update average latency
                        let sum: f64 = bridge_mut.latency_samples.iter().sum();
                        bridge_mut.stats.avg_latency_ms = sum / bridge_mut.latency_samples.len() as f64;
                    }
                    bridge_mut.stats.chunks_completed += 1;
                    bridge_mut.stats.queue_depth = bridge_mut.pending.len();
                    // Reset consecutive errors on success (error recovery)
                    bridge_mut.consecutive_errors = 0;
                }

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

                // Remove from pending and update error stats
                {
                    let mut bridge_mut = bridge.borrow_mut();
                    if let Some(id) = chunk_id {
                        bridge_mut.pending.remove(&id);
                    }
                    bridge_mut.stats.errors += 1;
                    bridge_mut.stats.queue_depth = bridge_mut.pending.len();
                    // Track consecutive errors for health checking
                    bridge_mut.consecutive_errors += 1;
                    if bridge_mut.consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        error!(
                            consecutive = bridge_mut.consecutive_errors,
                            "Worker unhealthy - too many consecutive errors"
                        );
                        bridge_mut.fatal_error = true;
                    }
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
    /// Returns Ok(Some(chunk_id)) if sent, Ok(None) if dropped due to queue overflow.
    ///
    /// # Queue Management (WAPR-SPEC-010 Section 3.2)
    ///
    /// If the pending queue exceeds MAX_QUEUE_DEPTH (3), the oldest chunk is dropped
    /// and the new chunk is queued. This implements backpressure to prevent unbounded
    /// memory growth during slow transcription.
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
    ) -> Result<Option<u32>, JsValue> {
        // Check queue depth and drop oldest if needed
        if self.pending.len() >= MAX_QUEUE_DEPTH {
            // Find and remove oldest pending request
            if let Some(oldest_id) = self.pending.keys().min().copied() {
                warn!(
                    dropped_chunk_id = oldest_id,
                    queue_depth = self.pending.len(),
                    "Queue overflow - dropping oldest chunk"
                );
                self.pending.remove(&oldest_id);
                self.stats.chunks_dropped += 1;
            }
        }

        let chunk_id = self.next_chunk_id;
        self.next_chunk_id = self.next_chunk_id.wrapping_add(1);

        debug!(
            chunk_id,
            audio_samples = audio.len(),
            session_id,
            queue_depth = self.pending.len(),
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

        // Update stats
        self.stats.chunks_sent += 1;
        self.stats.queue_depth = self.pending.len();

        Ok(Some(chunk_id))
    }

    /// Check if queue would overflow with another chunk
    #[must_use]
    pub fn would_overflow(&self) -> bool {
        self.pending.len() >= MAX_QUEUE_DEPTH
    }

    /// Get current queue statistics
    #[must_use]
    pub fn stats(&self) -> &QueueStats {
        &self.stats
    }

    /// Check if the worker is healthy
    ///
    /// Returns false if the worker has encountered too many consecutive errors
    /// or has a fatal error condition.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        !self.fatal_error && self.consecutive_errors < MAX_CONSECUTIVE_ERRORS
    }

    /// Check if the worker needs to be restarted
    #[must_use]
    pub fn needs_restart(&self) -> bool {
        self.fatal_error
    }

    /// Get consecutive error count
    #[must_use]
    pub fn consecutive_errors(&self) -> u32 {
        self.consecutive_errors
    }

    /// Reset error state (call after successful restart)
    pub fn reset_error_state(&mut self) {
        self.consecutive_errors = 0;
        self.fatal_error = false;
        info!("Worker error state reset");
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

    // =========================================================================
    // Queue Management Tests (WAPR-SPEC-010 Section 3.2)
    // =========================================================================

    #[test]
    fn test_max_queue_depth_is_three() {
        // Per spec: "Bounded Queues: Drop oldest chunks if queue exceeds N (backpressure)"
        // N = 3 per Section 3.2
        assert_eq!(MAX_QUEUE_DEPTH, 3);
    }

    #[test]
    fn test_queue_stats_default() {
        let stats = QueueStats::default();
        assert_eq!(stats.chunks_sent, 0);
        assert_eq!(stats.chunks_dropped, 0);
        assert_eq!(stats.chunks_completed, 0);
        assert_eq!(stats.errors, 0);
        assert!((stats.avg_latency_ms - 0.0).abs() < f64::EPSILON);
        assert_eq!(stats.queue_depth, 0);
    }

    #[test]
    fn test_queue_stats_debug() {
        let stats = QueueStats {
            chunks_sent: 10,
            chunks_dropped: 2,
            chunks_completed: 8,
            errors: 1,
            avg_latency_ms: 150.5,
            queue_depth: 2,
            processing_started: true,
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("chunks_sent: 10"));
        assert!(debug_str.contains("chunks_dropped: 2"));
    }

    #[test]
    fn test_queue_stats_clone() {
        let stats = QueueStats {
            chunks_sent: 5,
            chunks_dropped: 1,
            chunks_completed: 4,
            errors: 0,
            avg_latency_ms: 100.0,
            queue_depth: 1,
            processing_started: true,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.chunks_sent, stats.chunks_sent);
        assert_eq!(cloned.chunks_dropped, stats.chunks_dropped);
    }

    #[test]
    fn test_worker_result_metrics_variant() {
        let result = WorkerResult::Metrics {
            queue_depth: 2,
            avg_latency_ms: 125.5,
        };
        if let WorkerResult::Metrics { queue_depth, avg_latency_ms } = result {
            assert_eq!(queue_depth, 2);
            assert!((avg_latency_ms - 125.5).abs() < f64::EPSILON);
        } else {
            panic!("Expected Metrics variant");
        }
    }

    #[test]
    fn test_worker_result_pong_variant() {
        let result = WorkerResult::Pong {
            timestamp: 1000.0,
            worker_time: 1005.0,
        };
        if let WorkerResult::Pong { timestamp, worker_time } = result {
            assert!((timestamp - 1000.0).abs() < f64::EPSILON);
            assert!((worker_time - 1005.0).abs() < f64::EPSILON);
        } else {
            panic!("Expected Pong variant");
        }
    }

    // =========================================================================
    // Error Recovery Tests (WAPR-SPEC-010 Section 4.4)
    // =========================================================================

    #[test]
    fn test_max_consecutive_errors_is_three() {
        // Per spec: Worker should be considered unhealthy after 3 consecutive errors
        assert_eq!(MAX_CONSECUTIVE_ERRORS, 3);
    }

    #[test]
    fn test_worker_result_model_loaded() {
        let result = WorkerResult::ModelLoaded {
            size_mb: 37.5,
            load_time_ms: 1500.0,
        };
        if let WorkerResult::ModelLoaded { size_mb, load_time_ms } = result {
            assert!((size_mb - 37.5).abs() < f64::EPSILON);
            assert!((load_time_ms - 1500.0).abs() < f64::EPSILON);
        } else {
            panic!("Expected ModelLoaded variant");
        }
    }

    #[test]
    fn test_worker_result_session_started() {
        let result = WorkerResult::SessionStarted {
            session_id: "test_session".to_string(),
        };
        if let WorkerResult::SessionStarted { session_id } = result {
            assert_eq!(session_id, "test_session");
        } else {
            panic!("Expected SessionStarted variant");
        }
    }

    #[test]
    fn test_worker_result_session_ended() {
        let result = WorkerResult::SessionEnded {
            session_id: "test_session".to_string(),
        };
        if let WorkerResult::SessionEnded { session_id } = result {
            assert_eq!(session_id, "test_session");
        } else {
            panic!("Expected SessionEnded variant");
        }
    }
}
