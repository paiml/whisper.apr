//! Worker Manager - Coordinates main thread and worker
//!
//! Uses shared state via Rc<`RefCell`<>> for closure synchronization.
//! Marks buffer done on stop (worker polls isDone flag).
//!
//! PROBAR-SPEC-009: Uses tracing for structured logging instead of console.log.
//! PROBAR-WEBSYS-001: Uses probar web_sys_gen abstractions where applicable.

use std::cell::RefCell;
use std::rc::Rc;
use tracing::{debug, error, info, warn};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;

// Use probar's generated web_sys abstractions (PROBAR-WEBSYS-001)
use jugar_probar::brick::{get_base_url, BlobUrl, CustomEventDispatcher, EventDetail};

use crate::ring_buffer::SharedRingBuffer;
use crate::worker::{create_worker_blob_url, WorkerResult};

/// Manager state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagerState {
    Uninitialized,
    Spawning,
    Loading,
    Ready,
    Recording,
    Error,
}

/// Manages the transcription worker
#[wasm_bindgen]
pub struct WorkerManager {
    worker: Option<web_sys::Worker>,
    state_ptr: Rc<RefCell<ManagerState>>,
    ring_buffer: Option<SharedRingBuffer>,
    model_url: String,
    on_message_closure: Option<Closure<dyn Fn(web_sys::MessageEvent)>>,
}

#[wasm_bindgen]
impl WorkerManager {
    /// Create a new worker manager
    #[wasm_bindgen(constructor)]
    #[must_use] 
    pub fn new() -> WorkerManager {
        WorkerManager {
            worker: None,
            state_ptr: Rc::new(RefCell::new(ManagerState::Uninitialized)),
            ring_buffer: None,
            model_url: String::new(),
            on_message_closure: None,
        }
    }

    /// Spawn the worker
    pub fn spawn(&mut self, model_url: &str) -> Result<(), JsValue> {
        // Guard against double-spawning
        if self.worker.is_some() {
            warn!("Worker already spawned, ignoring");
            return Ok(());
        }

        self.model_url = model_url.to_string();
        *self.state_ptr.borrow_mut() = ManagerState::Spawning;

        // Create worker from blob URL
        let worker_url = create_worker_blob_url()?;

        let worker_options = web_sys::WorkerOptions::new();
        worker_options.set_type(web_sys::WorkerType::Module);

        let worker = web_sys::Worker::new_with_options(&worker_url, &worker_options)?;

        // Clean up blob URL using probar abstraction (PROBAR-WEBSYS-001)
        BlobUrl::revoke(&worker_url).map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Set up message handler using shared state
        let state_ptr_clone = self.state_ptr.clone();
        let _model_url_clone = self.model_url.clone();

        let on_message = Closure::wrap(Box::new(move |event: web_sys::MessageEvent| {
            let data = event.data();

            if let Ok(result) = serde_wasm_bindgen::from_value::<WorkerResult>(data.clone()) {
                match result {
                    WorkerResult::Ready => {
                        info!("Worker ready");
                        *state_ptr_clone.borrow_mut() = ManagerState::Loading;

                        // Dispatch event for main thread to know worker is ready
                        if let Some(window) = web_sys::window() {
                            if let Ok(event) = web_sys::CustomEvent::new("whisper-worker-ready") {
                                let _ = window.dispatch_event(&event);
                            }
                        }
                    }
                    WorkerResult::ModelLoaded { size_mb, load_time_ms } => {
                        info!(size_mb = size_mb, load_time_ms = load_time_ms, "Model loaded");
                        *state_ptr_clone.borrow_mut() = ManagerState::Ready;

                        // Dispatch event for main thread
                        if let Some(window) = web_sys::window() {
                            let detail = js_sys::Object::new();
                            let _ = js_sys::Reflect::set(&detail, &"sizeMb".into(), &size_mb.into());
                            let _ = js_sys::Reflect::set(&detail, &"loadTimeMs".into(), &load_time_ms.into());

                            let init = web_sys::CustomEventInit::new();
                            init.set_detail(&detail);

                            if let Ok(event) = web_sys::CustomEvent::new_with_event_init_dict(
                                "whisper-model-loaded",
                                &init,
                            ) {
                                let _ = window.dispatch_event(&event);
                            }
                        }
                    }
                    WorkerResult::Partial { text, is_final } => {
                        if is_final {
                            info!(text = %text, is_final = is_final, "Final transcription");
                        } else {
                            debug!(text = %text, "Partial transcription");
                        }
                        dispatch_transcription(&text, is_final);
                    }
                    WorkerResult::Result { text, .. } => {
                        info!(text = %text, "Final result");
                        *state_ptr_clone.borrow_mut() = ManagerState::Ready;
                        dispatch_transcription(&text, true);
                    }
                    WorkerResult::Error { message } => {
                        error!(message = %message, "Worker error");
                        *state_ptr_clone.borrow_mut() = ManagerState::Error;
                    }
                    WorkerResult::Progress { phase, percent } => {
                        debug!(phase = %phase, percent = percent, "Progress update");
                    }
                    WorkerResult::Metrics { rtf, chunks_processed, samples_read } => {
                        debug!(rtf = rtf, chunks_processed = chunks_processed, samples_read = samples_read, "Metrics");
                    }
                }
            }
        }) as Box<dyn Fn(web_sys::MessageEvent)>);

        worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));

        // Send bootstrap message using probar abstraction (PROBAR-WEBSYS-001)
        let base_url = get_base_url().unwrap_or_default();

        debug!(base_url = %base_url, "Sending bootstrap message");

        let bootstrap = js_sys::Object::new();
        js_sys::Reflect::set(&bootstrap, &"type".into(), &"bootstrap".into())?;
        js_sys::Reflect::set(&bootstrap, &"baseUrl".into(), &base_url.into())?;

        worker.post_message(&bootstrap)?;

        self.worker = Some(worker);
        self.on_message_closure = Some(on_message);

        info!("Worker spawned");
        Ok(())
    }

    /// Send init message with ring buffer and model URL
    pub fn send_init(&mut self) -> Result<(), JsValue> {
        let worker = self.worker.as_ref().ok_or("Worker not spawned")?;

        // Create ring buffer if needed
        if self.ring_buffer.is_none() {
            self.ring_buffer = Some(SharedRingBuffer::new(144_000)?); // 3 seconds at 48kHz
        }

        let buffer = self.ring_buffer.as_ref().ok_or("No ring buffer")?;

        let msg = js_sys::Object::new();
        js_sys::Reflect::set(&msg, &"type".into(), &"init".into())?;
        js_sys::Reflect::set(&msg, &"buffer".into(), &buffer.buffer())?;
        js_sys::Reflect::set(&msg, &"modelUrl".into(), &self.model_url.clone().into())?;

        worker.post_message(&msg)?;
        Ok(())
    }

    /// Start recording
    #[wasm_bindgen(js_name = startRecording)]
    pub fn start_recording(&mut self, sample_rate: u32) -> Result<(), JsValue> {
        let worker = self.worker.as_ref().ok_or("Worker not spawned")?;

        if *self.state_ptr.borrow() != ManagerState::Ready {
            return Err(JsValue::from_str("Worker not ready"));
        }

        // Reset ring buffer
        if let Some(ref buffer) = self.ring_buffer {
            buffer.reset()?;
        }

        // Send start message
        let msg = js_sys::Object::new();
        js_sys::Reflect::set(&msg, &"type".into(), &"start".into())?;
        js_sys::Reflect::set(&msg, &"sampleRate".into(), &f64::from(sample_rate).into())?;

        worker.post_message(&msg)?;
        *self.state_ptr.borrow_mut() = ManagerState::Recording;

        info!("Recording started");
        Ok(())
    }

    /// Stop recording
    #[wasm_bindgen(js_name = stopRecording)]
    pub fn stop_recording(&mut self) -> Result<(), JsValue> {
        if *self.state_ptr.borrow() != ManagerState::Recording {
            return Ok(());
        }

        // Mark ring buffer as done - worker will detect via isDone() polling
        if let Some(ref buffer) = self.ring_buffer {
            buffer.mark_done()?;
        }

        *self.state_ptr.borrow_mut() = ManagerState::Ready;
        info!("Recording stopped");
        Ok(())
    }

    /// Get ring buffer for `AudioWorklet`
    #[wasm_bindgen(js_name = getRingBuffer)]
    #[must_use]
    pub fn get_ring_buffer(&self) -> Option<SharedRingBuffer> {
        self.ring_buffer.clone()
    }

    /// Check if ready
    #[wasm_bindgen(js_name = isReady)]
    #[must_use]
    pub fn is_ready(&self) -> bool {
        *self.state_ptr.borrow() == ManagerState::Ready
    }

    /// Check if recording
    #[wasm_bindgen(js_name = isRecording)]
    #[must_use]
    pub fn is_recording(&self) -> bool {
        *self.state_ptr.borrow() == ManagerState::Recording
    }
}

impl Default for WorkerManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for WorkerManager {
    fn drop(&mut self) {
        if let Some(ref worker) = self.worker {
            worker.terminate();
        }
    }
}

/// Dispatch transcription event to main thread
/// Uses probar's CustomEventDispatcher abstraction (PROBAR-WEBSYS-001)
fn dispatch_transcription(text: &str, is_final: bool) {
    #[derive(serde::Serialize)]
    struct TranscriptionDetail<'a> {
        text: &'a str,
        #[serde(rename = "isFinal")]
        is_final: bool,
    }

    let detail = TranscriptionDetail { text, is_final };
    let dispatcher = CustomEventDispatcher::new("whisper-transcription");
    let _ = dispatcher.dispatch_with_detail(EventDetail::json(&detail));
}
