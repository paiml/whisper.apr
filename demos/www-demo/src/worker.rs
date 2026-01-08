//! Transcription Worker - Runs Whisper inference
//!
//! Handles model loading, audio processing, and transcription.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::ring_buffer::SharedRingBuffer;

// Re-export WASM types from whisper-apr
use whisper_apr::wasm::{TranscribeOptionsWasm, WhisperAprWasm};

/// Worker state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    Uninitialized,
    Ready,
    Processing,
    Error,
}

/// Messages sent from worker to main thread
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerResult {
    Ready,
    ModelLoaded { size_mb: f64, load_time_ms: f64 },
    Partial { text: String, is_final: bool },
    Result { text: String, segments: Vec<Segment> },
    Error { message: String },
    Progress { phase: String, percent: f64 },
    Metrics { rtf: f64, chunks_processed: u32, samples_read: u64 },
}

/// Transcription segment with timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start_ms: u32,
    pub end_ms: u32,
    pub text: String,
}

/// Transcription worker
#[wasm_bindgen]
pub struct TranscriptionWorker {
    state: WorkerState,
    model: Option<WhisperAprWasm>,
    ring_buffer: Option<SharedRingBuffer>,
    accumulated_audio: Vec<f32>,
    sample_rate: u32,
    samples_read: u64,
    chunks_processed: u32,
}

// Constants
const WHISPER_SAMPLE_RATE: u32 = 16000;
const CHUNK_SAMPLES: usize = 24000; // 1.5 seconds at 16kHz

#[wasm_bindgen]
impl TranscriptionWorker {
    /// Create a new worker instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> TranscriptionWorker {
        TranscriptionWorker {
            state: WorkerState::Uninitialized,
            model: None,
            ring_buffer: None,
            accumulated_audio: Vec::new(),
            sample_rate: WHISPER_SAMPLE_RATE,
            samples_read: 0,
            chunks_processed: 0,
        }
    }

    /// Initialize worker as ready
    #[wasm_bindgen(js_name = initWorker)]
    pub fn init_worker() -> TranscriptionWorker {
        web_sys::console::log_1(&"[Worker] Initialized".into());
        let mut worker = TranscriptionWorker::new();
        worker.state = WorkerState::Ready;
        worker
    }

    /// Set the ring buffer for audio data
    #[wasm_bindgen(js_name = setRingBuffer)]
    pub fn set_ring_buffer(&mut self, buffer: SharedRingBuffer) {
        web_sys::console::log_1(&"[Worker] Ring buffer attached".into());
        self.ring_buffer = Some(buffer);
    }

    /// Load the whisper model
    #[wasm_bindgen(js_name = loadModel)]
    pub async fn load_model(&mut self, url: String) -> Result<JsValue, JsValue> {
        let start = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);

        web_sys::console::log_1(&format!("[Worker] Loading model from: {url}").into());

        // Fetch model
        let response = web_sys::window()
            .ok_or("No window")?
            .fetch_with_str(&url);
        let response = wasm_bindgen_futures::JsFuture::from(response).await?;
        let response: web_sys::Response = response.dyn_into()?;

        if !response.ok() {
            return Err(JsValue::from_str(&format!("Failed to fetch model: {}", response.status())));
        }

        let buffer = wasm_bindgen_futures::JsFuture::from(response.array_buffer()?).await?;
        let array = js_sys::Uint8Array::new(&buffer);
        let bytes = array.to_vec();
        let size_mb = bytes.len() as f64 / 1_000_000.0;

        web_sys::console::log_1(&format!("[Worker] Model downloaded: {size_mb:.1} MB").into());

        // Initialize model from APR bytes
        self.model = Some(WhisperAprWasm::from_apr_bytes(&bytes)?);

        let end = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);
        let load_time_ms = end - start;

        web_sys::console::log_1(&format!("[Worker] Model initialized in {load_time_ms:.0}ms").into());

        self.state = WorkerState::Ready;

        let result = WorkerResult::ModelLoaded { size_mb, load_time_ms };
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Start processing audio
    #[wasm_bindgen(js_name = startProcessing)]
    pub fn start_processing(&mut self, sample_rate: u32) -> Result<(), JsValue> {
        web_sys::console::log_1(&format!(
            "[Worker] startProcessing called: state={:?}, has_buffer={}, sample_rate={}",
            self.state,
            self.ring_buffer.is_some(),
            sample_rate
        ).into());

        if self.state != WorkerState::Ready {
            return Err(JsValue::from_str("Worker not ready"));
        }

        self.sample_rate = sample_rate;
        self.accumulated_audio.clear();
        self.samples_read = 0;
        self.chunks_processed = 0;
        self.state = WorkerState::Processing;

        web_sys::console::log_1(&"[Worker] ✓ Started processing, state now Processing".into());
        Ok(())
    }

    /// Process available audio from ring buffer
    #[wasm_bindgen(js_name = processAudio)]
    pub fn process_audio(&mut self) -> Result<JsValue, JsValue> {
        if self.state != WorkerState::Processing {
            return Ok(JsValue::NULL);
        }

        let buffer = self.ring_buffer.as_ref().ok_or("No ring buffer")?;
        let available = buffer.available()?;

        if available == 0 {
            return Ok(JsValue::NULL);
        }

        web_sys::console::log_1(
            &format!("[Worker] processAudio: Reading {available} samples from buffer").into(),
        );

        let samples = buffer.read(available)?;
        self.samples_read += samples.len() as u64;

        // Resample if needed (44.1kHz -> 16kHz)
        let resampled = if self.sample_rate != WHISPER_SAMPLE_RATE {
            resample(&samples, self.sample_rate, WHISPER_SAMPLE_RATE)
        } else {
            samples
        };

        self.accumulated_audio.extend(resampled);

        let accumulated_seconds = self.accumulated_audio.len() as f64 / WHISPER_SAMPLE_RATE as f64;
        web_sys::console::log_1(&format!(
            "[Worker] Accumulated: {} samples ({:.1}s), need {} for chunk",
            self.accumulated_audio.len(),
            accumulated_seconds,
            CHUNK_SAMPLES
        ).into());

        // Check if we have enough for a chunk
        if self.accumulated_audio.len() >= CHUNK_SAMPLES {
            return self.transcribe_chunk();
        }

        Ok(JsValue::NULL)
    }

    /// Transcribe accumulated audio chunk
    fn transcribe_chunk(&mut self) -> Result<JsValue, JsValue> {
        let model = self.model.as_ref().ok_or("Model not loaded")?;

        // Take chunk from accumulated audio
        let chunk: Vec<f32> = self.accumulated_audio.drain(..CHUNK_SAMPLES).collect();
        self.chunks_processed += 1;

        let options = TranscribeOptionsWasm::new();
        let result = model.transcribe(&chunk, options)?;

        let text = result.text();
        if !text.trim().is_empty() {
            web_sys::console::log_1(&format!("[Worker] Partial: {text}").into());

            let partial = WorkerResult::Partial {
                text,
                is_final: false,
            };
            return serde_wasm_bindgen::to_value(&partial)
                .map_err(|e| JsValue::from_str(&e.to_string()));
        }

        Ok(JsValue::NULL)
    }

    /// Stop processing and get final result
    #[wasm_bindgen(js_name = stopProcessing)]
    pub fn stop_processing(&mut self) -> Result<JsValue, JsValue> {
        if self.state != WorkerState::Processing {
            return Ok(JsValue::NULL);
        }

        // Transcribe remaining audio
        let final_result = if self.accumulated_audio.is_empty() {
            None
        } else {
            let model = self.model.as_ref();
            if let Some(model) = model {
                let options = TranscribeOptionsWasm::new();
                match model.transcribe(&self.accumulated_audio, options) {
                    Ok(result) => {
                        self.accumulated_audio.clear();
                        Some(WorkerResult::Result {
                            text: result.text(),
                            segments: vec![],
                        })
                    }
                    Err(_) => None,
                }
            } else {
                None
            }
        };

        self.state = WorkerState::Ready;

        match final_result {
            Some(result) => serde_wasm_bindgen::to_value(&result)
                .map_err(|e| JsValue::from_str(&e.to_string())),
            None => Ok(JsValue::NULL),
        }
    }

    /// Get current metrics
    #[wasm_bindgen(js_name = getMetrics)]
    pub fn get_metrics(&self) -> Result<JsValue, JsValue> {
        let rtf = 0.0; // TODO: Calculate actual RTF
        let result = WorkerResult::Metrics {
            rtf,
            chunks_processed: self.chunks_processed,
            samples_read: self.samples_read,
        };
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for TranscriptionWorker {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple linear resampling
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let new_len = (samples.len() as f64 / ratio) as usize;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 * ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;

        let sample = if idx + 1 < samples.len() {
            samples[idx] * (1.0 - frac as f32) + samples[idx + 1] * frac as f32
        } else if idx < samples.len() {
            samples[idx]
        } else {
            0.0
        };

        resampled.push(sample);
    }

    resampled
}

/// Create blob URL for worker script
#[wasm_bindgen(js_name = createWorkerBlobUrl)]
pub fn create_worker_blob_url() -> Result<String, JsValue> {
    let worker_js = crate::worker_js::generate_worker_js();

    let blob_parts = js_sys::Array::new();
    blob_parts.push(&JsValue::from_str(&worker_js));

    let options = web_sys::BlobPropertyBag::new();
    options.set_type("application/javascript");

    let blob = web_sys::Blob::new_with_blob_sequence_and_options(&blob_parts, &options)?;
    web_sys::Url::create_object_url_with_blob(&blob)
}
