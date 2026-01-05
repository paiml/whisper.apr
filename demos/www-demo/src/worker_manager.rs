//! Worker Manager for Main Thread
//!
//! Manages Web Worker lifecycle and message passing from main thread.
//! Provides async API for non-blocking transcription.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use crate::ring_buffer::SharedRingBuffer;
use crate::worker::{WorkerResult, create_worker_blob_url};

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
    state: ManagerState,
    ring_buffer: Option<SharedRingBuffer>,
    on_message_closure: Option<Closure<dyn Fn(web_sys::MessageEvent)>>,
    on_error_closure: Option<Closure<dyn Fn(web_sys::ErrorEvent)>>,
}

#[wasm_bindgen]
impl WorkerManager {
    /// Create a new worker manager
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            worker: None,
            state: ManagerState::Uninitialized,
            ring_buffer: None,
            on_message_closure: None,
            on_error_closure: None,
        }
    }

    /// Get current state
    #[wasm_bindgen(getter)]
    pub fn state(&self) -> String {
        match self.state {
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
    pub fn is_ready(&self) -> bool {
        self.state == ManagerState::Ready
    }

    /// Check if currently recording
    #[wasm_bindgen(js_name = isRecording)]
    pub fn is_recording(&self) -> bool {
        self.state == ManagerState::Recording
    }

    /// Spawn worker and initialize with model
    #[wasm_bindgen]
    pub fn spawn(&mut self, model_url: &str) -> Result<(), JsValue> {
        if self.worker.is_some() {
            return Err(JsValue::from_str("Worker already spawned"));
        }

        self.state = ManagerState::Spawning;

        // Create ring buffer (3 seconds at 48kHz)
        let ring_buffer = SharedRingBuffer::new(48000 * 3)?;
        self.ring_buffer = Some(ring_buffer);

        // Create worker from blob URL
        let worker_url = create_worker_blob_url()?;
        let worker = web_sys::Worker::new(&worker_url)?;

        // Clean up blob URL
        web_sys::Url::revoke_object_url(&worker_url)?;

        // Store model URL for later
        let model_url = model_url.to_string();
        let ring_buffer_ref = self.ring_buffer.as_ref().unwrap().buffer();

        // Set up message handler
        let state_ptr = Rc::new(RefCell::new(ManagerState::Spawning));
        let state_ptr_clone = state_ptr.clone();

        let on_message = Closure::wrap(Box::new(move |event: web_sys::MessageEvent| {
            let data = event.data();

            // Try to parse as WorkerResult
            if let Ok(result) = serde_wasm_bindgen::from_value::<WorkerResult>(data.clone()) {
                match &result {
                    WorkerResult::Ready => {
                        web_sys::console::log_1(&"[Manager] Worker ready, sending init".into());
                        *state_ptr_clone.borrow_mut() = ManagerState::Loading;
                    }
                    WorkerResult::ModelLoaded { size_mb, load_time_ms } => {
                        web_sys::console::log_1(
                            &format!("[Manager] Model loaded: {:.1}MB in {:.0}ms",
                                     size_mb, load_time_ms).into()
                        );
                        *state_ptr_clone.borrow_mut() = ManagerState::Ready;
                    }
                    WorkerResult::Partial { text, is_final } => {
                        if *is_final {
                            web_sys::console::log_1(
                                &format!("[Manager] Final: {}", text).into()
                            );
                        } else {
                            web_sys::console::log_1(
                                &format!("[Manager] Partial: {}", text).into()
                            );
                        }
                        // Dispatch custom event for UI to handle
                        if let Some(window) = web_sys::window() {
                            let detail = js_sys::Object::new();
                            let _ = js_sys::Reflect::set(&detail, &"text".into(), &text.into());
                            let _ = js_sys::Reflect::set(&detail, &"isFinal".into(), &(*is_final).into());

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
                        web_sys::console::error_1(
                            &format!("[Manager] Error: {}", message).into()
                        );
                        *state_ptr_clone.borrow_mut() = ManagerState::Error;
                    }
                    _ => {}
                }
            }
        }) as Box<dyn Fn(web_sys::MessageEvent)>);

        worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        self.on_message_closure = Some(on_message);

        // Set up error handler
        let on_error = Closure::wrap(Box::new(move |event: web_sys::ErrorEvent| {
            web_sys::console::error_1(
                &format!("[Manager] Worker error: {}", event.message()).into()
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
        let _ = js_sys::Reflect::set(&bootstrap_msg, &"baseUrl".into(), &JsValue::from_str(&base_url));
        worker_ref.post_message(&bootstrap_msg)?;

        web_sys::console::log_1(&format!("[Manager] Bootstrap sent with baseUrl: {}", base_url).into());

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
                web_sys::console::error_1(&format!("Failed to send init: {:?}", e).into());
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
    #[wasm_bindgen(js_name = startRecording)]
    pub fn start_recording(&mut self, sample_rate: u32) -> Result<(), JsValue> {
        let worker = self.worker.as_ref().ok_or("Worker not spawned")?;

        if self.state != ManagerState::Ready {
            return Err(JsValue::from_str("Worker not ready"));
        }

        // Reset ring buffer
        if let Some(ref buffer) = self.ring_buffer {
            buffer.reset()?;
        }

        // Send start message
        let msg = js_sys::Object::new();
        js_sys::Reflect::set(&msg, &"type".into(), &"start".into())?;
        js_sys::Reflect::set(&msg, &"sampleRate".into(), &sample_rate.into())?;

        worker.post_message(&msg)?;
        self.state = ManagerState::Recording;

        web_sys::console::log_1(&"[Manager] Recording started".into());
        Ok(())
    }

    /// Stop recording
    #[wasm_bindgen(js_name = stopRecording)]
    pub fn stop_recording(&mut self) -> Result<(), JsValue> {
        let worker = self.worker.as_ref().ok_or("Worker not spawned")?;

        if self.state != ManagerState::Recording {
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
        self.state = ManagerState::Ready;

        web_sys::console::log_1(&"[Manager] Recording stopped".into());
        Ok(())
    }

    /// Get ring buffer for AudioWorklet
    #[wasm_bindgen(js_name = getRingBuffer)]
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
        self.state = ManagerState::Uninitialized;
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
