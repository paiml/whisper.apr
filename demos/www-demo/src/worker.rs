//! Web Worker for Non-Blocking Whisper Inference
//!
//! Runs Whisper inference on dedicated worker thread to prevent UI freezes.
//! Communicates with main thread via postMessage and SharedArrayBuffer.
//!
//! # References
//! - Kocher et al. (2019), "Spectre Attacks" - SharedArrayBuffer security
//! - WebAssembly Threads Proposal (2023) - Shared memory semantics

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::rc::Rc;

use crate::ring_buffer::SharedRingBuffer;

/// Commands sent from the main thread to the worker
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum WorkerCommand {
    /// Initialize worker with model
    #[serde(rename = "init")]
    Init {
        model_url: String,
    },
    /// Start processing audio from ring buffer
    #[serde(rename = "start")]
    Start {
        sample_rate: u32,
    },
    /// Stop processing
    #[serde(rename = "stop")]
    Stop,
    /// Update transcription options
    #[serde(rename = "options")]
    SetOptions {
        language: Option<String>,
        task: Option<String>,
    },
    /// Shutdown worker
    #[serde(rename = "shutdown")]
    Shutdown,
}

/// Results sent from the worker to the main thread
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum WorkerResult {
    /// Worker is ready
    #[serde(rename = "ready")]
    Ready,
    /// Model loaded successfully
    #[serde(rename = "modelLoaded")]
    ModelLoaded {
        size_mb: f64,
        load_time_ms: f64,
    },
    /// Partial transcription result
    #[serde(rename = "partial")]
    Partial {
        text: String,
        is_final: bool,
    },
    /// Final transcription result
    #[serde(rename = "result")]
    Result {
        text: String,
        segments: Vec<TranscriptSegment>,
    },
    /// Progress update
    #[serde(rename = "progress")]
    Progress {
        phase: String,
        percent: f32,
    },
    /// Error occurred
    #[serde(rename = "error")]
    Error {
        message: String,
    },
    /// Performance metrics
    #[serde(rename = "metrics")]
    Metrics {
        rtf: f64,
        chunks_processed: u64,
        samples_read: u64,
    },
}

/// Transcript segment with timing
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TranscriptSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// Worker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    Uninitialized,
    Loading,
    Ready,
    Processing,
    Error,
}

/// Transcription worker running in Web Worker context
#[wasm_bindgen]
pub struct TranscriptionWorker {
    state: WorkerState,
    ring_buffer: Option<SharedRingBuffer>,
    model: Option<Rc<whisper_apr::wasm::WhisperAprWasm>>,
    sample_rate: u32,
    chunks_processed: u64,
    samples_read: u64,
    accumulated_audio: Vec<f32>,
    last_partial: String,
}

#[wasm_bindgen]
impl TranscriptionWorker {
    /// Create new worker instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            state: WorkerState::Uninitialized,
            ring_buffer: None,
            model: None,
            sample_rate: 16000,
            chunks_processed: 0,
            samples_read: 0,
            accumulated_audio: Vec::with_capacity(16000 * 30), // 30 seconds
            last_partial: String::new(),
        }
    }

    /// Set the ring buffer (called after receiving SharedArrayBuffer)
    #[wasm_bindgen(js_name = setRingBuffer)]
    pub fn set_ring_buffer(&mut self, buffer: SharedRingBuffer) {
        self.ring_buffer = Some(buffer);
        web_sys::console::log_1(&"[Worker] Ring buffer attached".into());
    }

    /// Get current state
    #[wasm_bindgen(getter)]
    pub fn state(&self) -> String {
        match self.state {
            WorkerState::Uninitialized => "uninitialized".to_string(),
            WorkerState::Loading => "loading".to_string(),
            WorkerState::Ready => "ready".to_string(),
            WorkerState::Processing => "processing".to_string(),
            WorkerState::Error => "error".to_string(),
        }
    }

    /// Load model from URL
    #[wasm_bindgen(js_name = loadModel)]
    pub async fn load_model(&mut self, url: &str) -> Result<JsValue, JsValue> {
        use wasm_bindgen_futures::JsFuture;

        self.state = WorkerState::Loading;
        let start = js_sys::Date::now();

        web_sys::console::log_1(&format!("[Worker] Loading model from: {}", url).into());

        // Fetch model (works in both window and worker contexts)
        let global = js_sys::global();
        let fetch_fn = js_sys::Reflect::get(&global, &"fetch".into())
            .map_err(|_| JsValue::from_str("fetch not available"))?;
        let fetch_fn = fetch_fn.dyn_into::<js_sys::Function>()
            .map_err(|_| JsValue::from_str("fetch is not a function"))?;

        let promise = fetch_fn.call1(&global, &url.into())?;
        let resp = JsFuture::from(js_sys::Promise::from(promise)).await?;
        let resp: web_sys::Response = resp.dyn_into()?;

        if !resp.ok() {
            self.state = WorkerState::Error;
            return Err(JsValue::from_str(&format!("Failed to fetch model: {}", resp.status())));
        }

        let array_buffer = JsFuture::from(resp.array_buffer()?).await?;
        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        let model_bytes = uint8_array.to_vec();
        let size_mb = model_bytes.len() as f64 / 1_000_000.0;

        web_sys::console::log_1(&format!("[Worker] Model downloaded: {:.1} MB", size_mb).into());

        // Initialize Whisper model
        match whisper_apr::wasm::WhisperAprWasm::from_apr_bytes(&model_bytes) {
            Ok(model) => {
                self.model = Some(Rc::new(model));
                self.state = WorkerState::Ready;
                let load_time_ms = js_sys::Date::now() - start;

                web_sys::console::log_1(
                    &format!("[Worker] Model initialized in {:.0}ms", load_time_ms).into()
                );

                let result = WorkerResult::ModelLoaded { size_mb, load_time_ms };
                Ok(serde_wasm_bindgen::to_value(&result)?)
            }
            Err(e) => {
                self.state = WorkerState::Error;
                Err(JsValue::from_str(&format!("Model init failed: {:?}", e)))
            }
        }
    }

    /// Start processing audio from ring buffer
    #[wasm_bindgen(js_name = startProcessing)]
    pub fn start_processing(&mut self, sample_rate: u32) -> Result<(), JsValue> {
        if self.state != WorkerState::Ready {
            return Err(JsValue::from_str("Worker not ready"));
        }
        if self.ring_buffer.is_none() {
            return Err(JsValue::from_str("No ring buffer attached"));
        }

        self.sample_rate = sample_rate;
        self.state = WorkerState::Processing;
        self.accumulated_audio.clear();
        self.chunks_processed = 0;
        self.samples_read = 0;

        web_sys::console::log_1(&"[Worker] Started processing".into());
        Ok(())
    }

    /// Process available audio from ring buffer
    ///
    /// Returns partial transcription if available
    #[wasm_bindgen(js_name = processAudio)]
    pub fn process_audio(&mut self) -> Result<JsValue, JsValue> {
        if self.state != WorkerState::Processing {
            return Ok(JsValue::NULL);
        }

        let Some(ref buffer) = self.ring_buffer else {
            return Ok(JsValue::NULL);
        };

        // Read available samples
        let available = buffer.available_read()?;
        if available == 0 {
            return Ok(JsValue::NULL);
        }

        let samples = buffer.read(available)?;
        self.samples_read += samples.len() as u64;

        // Resample if needed (AudioWorklet runs at device rate, Whisper needs 16kHz)
        let resampled = if self.sample_rate != 16000 {
            self.resample(&samples, self.sample_rate, 16000)
        } else {
            samples
        };

        self.accumulated_audio.extend_from_slice(&resampled);

        // Check if we have enough for transcription (1.5s chunks)
        let chunk_samples = 16000 * 3 / 2; // 1.5 seconds at 16kHz
        if self.accumulated_audio.len() >= chunk_samples {
            return self.transcribe_chunk();
        }

        // Return partial if we have at least 0.5s
        let partial_samples = 16000 / 2;
        if self.accumulated_audio.len() >= partial_samples {
            return self.transcribe_partial();
        }

        Ok(JsValue::NULL)
    }

    /// Transcribe a full chunk
    fn transcribe_chunk(&mut self) -> Result<JsValue, JsValue> {
        let Some(ref model) = self.model else {
            return Ok(JsValue::NULL);
        };

        let chunk_samples = 16000 * 3 / 2;
        let audio: Vec<f32> = self.accumulated_audio.drain(..chunk_samples).collect();

        let start = js_sys::Date::now();

        // Create transcription options
        let options = whisper_apr::wasm::TranscribeOptionsWasm::new();

        // Transcribe
        match model.transcribe(&audio, options) {
            Ok(result) => {
                let elapsed = js_sys::Date::now() - start;
                let audio_duration = audio.len() as f64 / 16000.0 * 1000.0;
                let rtf = elapsed / audio_duration;

                self.chunks_processed += 1;

                web_sys::console::log_1(
                    &format!("[Worker] Chunk {} transcribed in {:.0}ms (RTF: {:.2})",
                             self.chunks_processed, elapsed, rtf).into()
                );

                let worker_result = WorkerResult::Partial {
                    text: result.text(),
                    is_final: true,
                };
                Ok(serde_wasm_bindgen::to_value(&worker_result)?)
            }
            Err(e) => {
                web_sys::console::error_1(&format!("[Worker] Transcription error: {:?}", e).into());
                Ok(JsValue::NULL)
            }
        }
    }

    /// Transcribe partial audio for responsive feedback
    fn transcribe_partial(&mut self) -> Result<JsValue, JsValue> {
        let Some(ref model) = self.model else {
            return Ok(JsValue::NULL);
        };

        // Don't process if too little audio
        if self.accumulated_audio.len() < 8000 {
            return Ok(JsValue::NULL);
        }

        let options = whisper_apr::wasm::TranscribeOptionsWasm::new();

        // Transcribe accumulated audio without consuming it
        match model.transcribe(&self.accumulated_audio, options) {
            Ok(result) => {
                let text = result.text();

                // Only return if different from last partial
                if text != self.last_partial && !text.is_empty() {
                    self.last_partial = text.clone();
                    let worker_result = WorkerResult::Partial {
                        text,
                        is_final: false,
                    };
                    return Ok(serde_wasm_bindgen::to_value(&worker_result)?);
                }
            }
            Err(_) => {}
        }

        Ok(JsValue::NULL)
    }

    /// Simple linear resampling
    fn resample(&self, samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        if from_rate == to_rate {
            return samples.to_vec();
        }

        let ratio = from_rate as f64 / to_rate as f64;
        let new_len = (samples.len() as f64 / ratio) as usize;
        let mut result = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = i as f64 * ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(samples.len() - 1);
            let frac = src_idx - idx0 as f64;

            let sample = samples[idx0] * (1.0 - frac as f32) + samples[idx1] * frac as f32;
            result.push(sample);
        }

        result
    }

    /// Stop processing
    #[wasm_bindgen(js_name = stopProcessing)]
    pub fn stop_processing(&mut self) -> Result<JsValue, JsValue> {
        if self.state != WorkerState::Processing {
            return Ok(JsValue::NULL);
        }

        // Transcribe remaining audio
        let final_result = if !self.accumulated_audio.is_empty() {
            let Some(ref model) = self.model else {
                return Ok(JsValue::NULL);
            };

            let options = whisper_apr::wasm::TranscribeOptionsWasm::new();
            match model.transcribe(&self.accumulated_audio, options) {
                Ok(result) => {
                    self.accumulated_audio.clear();
                    Some(WorkerResult::Result {
                        text: result.text(),
                        segments: vec![], // TODO: Add segment extraction
                    })
                }
                Err(_) => None
            }
        } else {
            None
        };

        self.state = WorkerState::Ready;

        match final_result {
            Some(result) => Ok(serde_wasm_bindgen::to_value(&result)?),
            None => Ok(JsValue::NULL),
        }
    }

    /// Get metrics
    #[wasm_bindgen(js_name = getMetrics)]
    pub fn get_metrics(&self) -> Result<JsValue, JsValue> {
        let result = WorkerResult::Metrics {
            rtf: 0.0, // TODO: Calculate running average
            chunks_processed: self.chunks_processed,
            samples_read: self.samples_read,
        };
        Ok(serde_wasm_bindgen::to_value(&result)?)
    }
}

impl Default for TranscriptionWorker {
    fn default() -> Self {
        Self::new()
    }
}

/// Worker entry point - called when worker script loads
#[wasm_bindgen(js_name = initWorker)]
pub fn init_worker() -> TranscriptionWorker {
    console_error_panic_hook::set_once();
    tracing_wasm::set_as_global_default();

    web_sys::console::log_1(&"[Worker] Initialized".into());

    TranscriptionWorker::new()
}

/// JavaScript code for worker bootstrap
pub const WORKER_JS: &str = r#"
// Whisper.apr Transcription Worker
// Loads WASM and processes audio from SharedArrayBuffer

let wasm;
let worker;
let ringBuffer;
let processingInterval;
let initialized = false;

self.onmessage = async function(e) {
    const msg = e.data;

    // First message must be 'bootstrap' to load WASM
    if (msg.type === 'bootstrap') {
        const baseUrl = msg.baseUrl || '';
        try {
            // Import WASM module using absolute URL
            importScripts(baseUrl + '/pkg/whisper_apr_demo.js');

            // Initialize WASM
            wasm = await wasm_bindgen(baseUrl + '/pkg/whisper_apr_demo_bg.wasm');

            // Create worker instance
            worker = wasm.initWorker();
            initialized = true;

            // Notify main thread we're ready
            self.postMessage({ type: 'ready' });
        } catch (e) {
            self.postMessage({ type: 'error', message: 'Worker init failed: ' + e.toString() });
        }
        return;
    }

    // Ignore other messages until initialized
    if (!initialized) {
        console.warn('[Worker] Not initialized, ignoring message:', msg.type);
        return;
    }

    switch (msg.type) {
        case 'init':
            // Attach ring buffer from SharedArrayBuffer
            if (msg.buffer) {
                ringBuffer = wasm.SharedRingBuffer.fromBuffer(msg.buffer);
                worker.setRingBuffer(ringBuffer);
            }

            // Load model
            try {
                const result = await worker.loadModel(msg.modelUrl);
                self.postMessage(result);
            } catch (e) {
                self.postMessage({ type: 'error', message: e.toString() });
            }
            break;

        case 'start':
            worker.startProcessing(msg.sampleRate);

            // Start processing loop
            processingInterval = setInterval(() => {
                const result = worker.processAudio();
                if (result) {
                    self.postMessage(result);
                }
            }, 50); // Process every 50ms
            break;

        case 'stop':
            if (processingInterval) {
                clearInterval(processingInterval);
                processingInterval = null;
            }

            const result = worker.stopProcessing();
            if (result) {
                self.postMessage(result);
            }
            break;

        case 'metrics':
            self.postMessage(worker.getMetrics());
            break;

        case 'shutdown':
            if (processingInterval) {
                clearInterval(processingInterval);
            }
            self.close();
            break;
    }
};
"#;

/// Create blob URL for worker script
#[wasm_bindgen(js_name = createWorkerBlobUrl)]
pub fn create_worker_blob_url() -> Result<String, JsValue> {
    let blob_parts = js_sys::Array::new();
    blob_parts.push(&JsValue::from_str(WORKER_JS));

    let options = web_sys::BlobPropertyBag::new();
    options.set_type("application/javascript");

    let blob = web_sys::Blob::new_with_blob_sequence_and_options(&blob_parts, &options)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)?;

    Ok(url)
}
