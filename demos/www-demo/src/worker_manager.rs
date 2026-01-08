//! Worker Manager for Main Thread
//!
//! Manages Web Worker lifecycle and message passing from main thread.
//! Provides async API for non-blocking transcription.

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::ring_buffer::SharedRingBuffer;
use crate::worker::{create_worker_blob_url, WorkerResult};

/// Callback for transcription results
pub type ResultCallback = Rc<RefCell<dyn FnMut(WorkerResult)>>;

/// Worker manager state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagerState {
    Uninitialized,
    Spawning,
    Loading,
    Ready,
    Recording,
    Error,
}

/// Manages communication with transcription worker
#[wasm_bindgen]
pub struct WorkerManager {
    worker: Option<web_sys::Worker>,
    /// CRITICAL: This is the authoritative state, shared with the onmessage closure.
    /// Do NOT use a separate `state` field - that causes desync bugs.
    /// The closure updates this via `state_ptr_clone`, and all state checks read from it.
    state_ptr: Rc<RefCell<ManagerState>>,
    ring_buffer: Option<SharedRingBuffer>,
    on_message_closure: Option<Closure<dyn Fn(web_sys::MessageEvent)>>,
    on_error_closure: Option<Closure<dyn Fn(web_sys::ErrorEvent)>>,
    /// If start_recording is called before model loads, store sample_rate here
    pending_start_sample_rate: Option<u32>,
}

#[wasm_bindgen]
impl WorkerManager {
    /// Create a new worker manager
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            worker: None,
            state_ptr: Rc::new(RefCell::new(ManagerState::Uninitialized)),
            ring_buffer: None,
            on_message_closure: None,
            on_error_closure: None,
            pending_start_sample_rate: None,
        }
    }

    /// Get current state
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn state(&self) -> String {
        match *self.state_ptr.borrow() {
            ManagerState::Uninitialized => "uninitialized".to_string(),
            ManagerState::Spawning => "spawning".to_string(),
            ManagerState::Loading => "loading".to_string(),
            ManagerState::Ready => "ready".to_string(),
            ManagerState::Recording => "recording".to_string(),
            ManagerState::Error => "error".to_string(),
        }
    }

    /// Check if worker is ready
    #[wasm_bindgen(js_name = isReady)]
    #[must_use]
    pub fn is_ready(&self) -> bool {
        *self.state_ptr.borrow() == ManagerState::Ready
    }

    /// Check if currently recording
    #[wasm_bindgen(js_name = isRecording)]
    #[must_use]
    pub fn is_recording(&self) -> bool {
        *self.state_ptr.borrow() == ManagerState::Recording
    }

    /// Check if there's a pending start request (called before model was ready)
    #[wasm_bindgen(js_name = pendingStartSampleRate)]
    #[must_use]
    pub fn pending_start_sample_rate(&self) -> Option<u32> {
        self.pending_start_sample_rate
    }

    /// Spawn worker and initialize with model
    #[wasm_bindgen]
    pub fn spawn(&mut self, model_url: &str) -> Result<(), JsValue> {
        if self.worker.is_some() {
            return Err(JsValue::from_str("Worker already spawned"));
        }

        *self.state_ptr.borrow_mut() = ManagerState::Spawning;

        // Create ring buffer (3 seconds at 48kHz)
        let ring_buffer = SharedRingBuffer::new(48000 * 3)?;
        self.ring_buffer = Some(ring_buffer);

        // Create worker from blob URL with ES module type
        // CRITICAL: wasm-bindgen generates ES modules, so type: 'module' is REQUIRED
        let worker_url = create_worker_blob_url()?;

        let worker_options = web_sys::WorkerOptions::new();
        worker_options.set_type(web_sys::WorkerType::Module);

        let worker = web_sys::Worker::new_with_options(&worker_url, &worker_options)?;

        // Clean up blob URL
        web_sys::Url::revoke_object_url(&worker_url)?;

        // Store model URL for later
        let model_url = model_url.to_string();
        let ring_buffer_ref = self.ring_buffer.as_ref().unwrap().buffer();

        // Set up message handler - use the shared state_ptr from self
        let state_ptr_clone = self.state_ptr.clone();

        let on_message = Closure::wrap(Box::new(move |event: web_sys::MessageEvent| {
            let data = event.data();

            // Try to parse as WorkerResult
            if let Ok(result) = serde_wasm_bindgen::from_value::<WorkerResult>(data.clone()) {
                match &result {
                    WorkerResult::Ready => {
                        web_sys::console::log_1(&"[Manager] Worker ready, sending init".into());
                        *state_ptr_clone.borrow_mut() = ManagerState::Loading;
                    }
                    WorkerResult::ModelLoaded {
                        size_mb,
                        load_time_ms,
                    } => {
                        web_sys::console::log_1(
                            &format!(
                                "[Manager] Model loaded: {size_mb:.1}MB in {load_time_ms:.0}ms"
                            )
                            .into(),
                        );
                        *state_ptr_clone.borrow_mut() = ManagerState::Ready;

                        // Dispatch event so main thread can start recording if pending
                        if let Some(window) = web_sys::window() {
                            let detail = js_sys::Object::new();
                            let _ =
                                js_sys::Reflect::set(&detail, &"sizeMb".into(), &(*size_mb).into());
                            let _ = js_sys::Reflect::set(
                                &detail,
                                &"loadTimeMs".into(),
                                &(*load_time_ms).into(),
                            );

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
                        if *is_final {
                            web_sys::console::log_1(&format!("[Manager] Final: {text}").into());
                        } else {
                            web_sys::console::log_1(&format!("[Manager] Partial: {text}").into());
                        }
                        // Dispatch custom event for UI to handle
                        if let Some(window) = web_sys::window() {
                            let detail = js_sys::Object::new();
                            let _ = js_sys::Reflect::set(&detail, &"text".into(), &text.into());
                            let _ = js_sys::Reflect::set(
                                &detail,
                                &"isFinal".into(),
                                &(*is_final).into(),
                            );

                            let init = web_sys::CustomEventInit::new();
                            init.set_detail(&detail);

                            if let Ok(event) = web_sys::CustomEvent::new_with_event_init_dict(
                                "whisper-transcription",
                                &init,
                            ) {
                                let _ = window.dispatch_event(&event);
                            }
                        }
                    }
                    WorkerResult::Error { message } => {
                        web_sys::console::error_1(&format!("[Manager] Error: {message}").into());
                        *state_ptr_clone.borrow_mut() = ManagerState::Error;
                    }
                    WorkerResult::Result { text, segments: _ } => {
                        // Final transcription result from stopProcessing()
                        web_sys::console::log_1(&format!("[Manager] Final result: {text}").into());
                        *state_ptr_clone.borrow_mut() = ManagerState::Ready;

                        // Dispatch as final transcription event
                        if let Some(window) = web_sys::window() {
                            let detail = js_sys::Object::new();
                            let _ = js_sys::Reflect::set(&detail, &"text".into(), &text.into());
                            let _ = js_sys::Reflect::set(&detail, &"isFinal".into(), &true.into());

                            let init = web_sys::CustomEventInit::new();
                            init.set_detail(&detail);

                            if let Ok(event) = web_sys::CustomEvent::new_with_event_init_dict(
                                "whisper-transcription",
                                &init,
                            ) {
                                let _ = window.dispatch_event(&event);
                            }
                        }
                    }
                    WorkerResult::Progress { phase, percent } => {
                        web_sys::console::log_1(
                            &format!("[Manager] Progress: {phase} {percent:.0}%").into(),
                        );
                    }
                    WorkerResult::Metrics { rtf, chunks_processed, samples_read } => {
                        web_sys::console::log_1(
                            &format!("[Manager] Metrics: RTF={rtf:.2}, chunks={chunks_processed}, samples={samples_read}").into(),
                        );
                    }
                }
            }
        }) as Box<dyn Fn(web_sys::MessageEvent)>);

        worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        self.on_message_closure = Some(on_message);

        // Set up error handler
        let on_error = Closure::wrap(Box::new(move |event: web_sys::ErrorEvent| {
            web_sys::console::error_1(
                &format!("[Manager] Worker error: {}", event.message()).into(),
            );
        }) as Box<dyn Fn(web_sys::ErrorEvent)>);

        worker.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        self.on_error_closure = Some(on_error);

        self.worker = Some(worker);

        // Get base URL for worker to load WASM
        let window = web_sys::window().ok_or("No window")?;
        let location = window.location();
        let base_url = format!(
            "{}//{}",
            location.protocol().unwrap_or_default(),
            location.host().unwrap_or_default()
        );

        // First send bootstrap message with base URL
        let worker_ref = self.worker.as_ref().unwrap();
        let bootstrap_msg = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&bootstrap_msg, &"type".into(), &"bootstrap".into());
        let _ = js_sys::Reflect::set(
            &bootstrap_msg,
            &"baseUrl".into(),
            &JsValue::from_str(&base_url),
        );
        worker_ref.post_message(&bootstrap_msg)?;

        web_sys::console::log_1(
            &format!("[Manager] Bootstrap sent with baseUrl: {base_url}").into(),
        );

        // Send init message after bootstrap (worker will handle sequencing)
        // Use a short delay to allow the bootstrap to complete
        let worker_clone = self.worker.as_ref().unwrap().clone();
        let model_url_clone = model_url.clone();

        let init_closure = Closure::once(Box::new(move || {
            let init_msg = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&init_msg, &"type".into(), &"init".into());
            let _ = js_sys::Reflect::set(&init_msg, &"modelUrl".into(), &model_url_clone.into());
            let _ = js_sys::Reflect::set(&init_msg, &"buffer".into(), &ring_buffer_ref);

            if let Err(e) = worker_clone.post_message(&init_msg) {
                web_sys::console::error_1(&format!("Failed to send init: {e:?}").into());
            }
        }) as Box<dyn FnOnce()>);

        // Send init after bootstrap delay
        window.set_timeout_with_callback_and_timeout_and_arguments_0(
            init_closure.as_ref().unchecked_ref(),
            500, // Longer delay to wait for WASM load
        )?;
        init_closure.forget();

        web_sys::console::log_1(&"[Manager] Worker spawned".into());
        Ok(())
    }

    /// Start recording and transcription
    ///
    /// If called before model is ready (state == Loading), the request is queued
    /// and will be executed automatically when the model finishes loading.
    #[wasm_bindgen(js_name = startRecording)]
    pub fn start_recording(&mut self, sample_rate: u32) -> Result<(), JsValue> {
        if self.worker.is_none() {
            return Err(JsValue::from_str("Worker not spawned"));
        }

        let current_state = *self.state_ptr.borrow();

        // If model is still loading, queue the start request
        if current_state == ManagerState::Loading {
            web_sys::console::log_1(
                &"[Manager] Model still loading, queueing start request...".into(),
            );
            self.pending_start_sample_rate = Some(sample_rate);
            return Ok(());
        }

        if current_state != ManagerState::Ready {
            return Err(JsValue::from_str("Worker not ready"));
        }

        // Clear pending request
        self.pending_start_sample_rate = None;

        // Reset ring buffer
        if let Some(ref buffer) = self.ring_buffer {
            buffer.reset()?;
        }

        // Send start message
        let msg = js_sys::Object::new();
        js_sys::Reflect::set(&msg, &"type".into(), &"start".into())?;
        js_sys::Reflect::set(&msg, &"sampleRate".into(), &sample_rate.into())?;

        self.worker.as_ref().unwrap().post_message(&msg)?;
        *self.state_ptr.borrow_mut() = ManagerState::Recording;

        web_sys::console::log_1(&"[Manager] Recording started".into());
        Ok(())
    }

    /// Stop recording
    #[wasm_bindgen(js_name = stopRecording)]
    pub fn stop_recording(&mut self) -> Result<(), JsValue> {
        let worker = self.worker.as_ref().ok_or("Worker not spawned")?;

        if *self.state_ptr.borrow() != ManagerState::Recording {
            return Ok(()); // Already stopped
        }

        // Mark ring buffer as done
        if let Some(ref buffer) = self.ring_buffer {
            buffer.mark_done()?;
        }

        // Send stop message
        let msg = js_sys::Object::new();
        js_sys::Reflect::set(&msg, &"type".into(), &"stop".into())?;

        worker.post_message(&msg)?;
        *self.state_ptr.borrow_mut() = ManagerState::Ready;

        web_sys::console::log_1(&"[Manager] Recording stopped".into());
        Ok(())
    }

    /// Get ring buffer for `AudioWorklet`
    #[wasm_bindgen(js_name = getRingBuffer)]
    #[must_use]
    pub fn get_ring_buffer(&self) -> Option<SharedRingBuffer> {
        self.ring_buffer.clone()
    }

    /// Shutdown worker
    #[wasm_bindgen]
    pub fn shutdown(&mut self) -> Result<(), JsValue> {
        if let Some(ref worker) = self.worker {
            let msg = js_sys::Object::new();
            js_sys::Reflect::set(&msg, &"type".into(), &"shutdown".into())?;
            worker.post_message(&msg)?;
            worker.terminate();
        }

        self.worker = None;
        self.ring_buffer = None;
        *self.state_ptr.borrow_mut() = ManagerState::Uninitialized;
        self.on_message_closure = None;
        self.on_error_closure = None;

        web_sys::console::log_1(&"[Manager] Worker shutdown".into());
        Ok(())
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
