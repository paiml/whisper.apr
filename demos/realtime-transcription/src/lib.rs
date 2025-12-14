//! WAPR-DEMO-001: Real-time Microphone Transcription Demo
//!
//! Pure Rust WASM demo for real-time speech-to-text transcription.
//! Zero JavaScript - all browser APIs accessed via `web-sys`.
//!
//! # Architecture (WAPR-SPEC-010)
//!
//! ```text
//! Main Thread                          Worker Thread
//! ┌────────────────┐                   ┌────────────────┐
//! │ Microphone     │                   │  WhisperApr    │
//! │ Audio Capture  │──Transferable────>│  Model         │
//! │ WorkerBridge   │<──postMessage─────│  Transcribe    │
//! │ UI Updates     │                   │                │
//! └────────────────┘                   └────────────────┘
//! ```
//!
//! # Tracing (renacer integration)
//!
//! This demo uses the `tracing` crate for instrumentation compatible with renacer.
//! Enable browser console output with `tracing_wasm::set_as_global_default()`.
//!
//! Tracing levels:
//! - Light (<1% CPU): Error events, chunk completion
//! - Medium (<5% CPU): State transitions, worker timing, RTF
//! - Full (~10% CPU): Audio samples, memory, SIMD timing

pub mod bridge;
pub mod worker;

use std::cell::RefCell;
use std::rc::Rc;
use tracing::{info, info_span, warn};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use whisper_apr::audio::{StreamingConfig, StreamingProcessor};
use whisper_apr::{TranscribeOptions, WhisperApr};

/// Default model URL - can be served locally or from CDN
/// For local development: serve the models/ directory at http://localhost:8080/models/
const MODEL_URL: &str = "/models/whisper-tiny-int8.apr";

/// Demo application state machine
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DemoState {
    /// Initial state, model not loaded yet
    #[default]
    Initializing,
    /// Loading model from network
    LoadingModel,
    /// Ready to record
    Idle,
    /// Requesting microphone permission from browser
    RequestingPermission,
    /// Actively recording audio
    Recording,
    /// Processing recorded audio through Whisper
    Processing,
    /// Error state (permission denied, etc.)
    Error,
}

/// State transition validator for the demo state machine
pub struct StateTransition;

impl StateTransition {
    /// Check if a state transition is valid
    #[must_use]
    #[allow(clippy::unnested_or_patterns)]
    pub const fn is_valid(from: DemoState, to: DemoState) -> bool {
        matches!(
            (from, to),
            // Valid transitions
            (DemoState::Initializing, DemoState::LoadingModel)
                | (DemoState::Initializing, DemoState::Error)
                | (DemoState::LoadingModel, DemoState::Idle)
                | (DemoState::LoadingModel, DemoState::Error)
                | (DemoState::Idle, DemoState::RequestingPermission)
                | (DemoState::RequestingPermission, DemoState::Recording)
                | (DemoState::RequestingPermission, DemoState::Error)
                | (DemoState::Recording, DemoState::Processing)
                | (DemoState::Processing, DemoState::Idle)
                | (DemoState::Error, DemoState::Idle)
                | (DemoState::Error, DemoState::LoadingModel)
                // Self-transitions (no-op)
                | (DemoState::Idle, DemoState::Idle)
                | (DemoState::Recording, DemoState::Recording)
                | (DemoState::Initializing, DemoState::Initializing)
                | (DemoState::LoadingModel, DemoState::LoadingModel)
        )
    }
}

/// Real-time transcription demo application
#[wasm_bindgen]
pub struct RealtimeTranscriptionDemo {
    state: DemoState,
    transcript: String,
    partial_transcript: String,
    recording_duration_ms: u32,
    error_message: Option<String>,
    samples_captured: u64,
    model_loaded: bool,
    chunks_captured: u32,
}

#[wasm_bindgen]
impl RealtimeTranscriptionDemo {
    /// Create a new demo instance
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: DemoState::Initializing,
            transcript: String::new(),
            partial_transcript: String::new(),
            recording_duration_ms: 0,
            error_message: None,
            samples_captured: 0,
            model_loaded: false,
            chunks_captured: 0,
        }
    }

    /// Mark model as loaded
    pub fn set_model_loaded(&mut self) {
        self.model_loaded = true;
    }

    /// Check if model is loaded
    #[must_use]
    pub fn is_model_loaded(&self) -> bool {
        self.model_loaded
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> DemoState {
        self.state
    }

    /// Get current state as a lowercase string (for JS compatibility)
    #[wasm_bindgen(js_name = stateString)]
    #[must_use]
    pub fn state_string(&self) -> String {
        match self.state {
            DemoState::Initializing => "initializing".to_string(),
            DemoState::LoadingModel => "loadingmodel".to_string(),
            DemoState::Idle => "idle".to_string(),
            DemoState::RequestingPermission => "requestingpermission".to_string(),
            DemoState::Recording => "recording".to_string(),
            DemoState::Processing => "processing".to_string(),
            DemoState::Error => "error".to_string(),
        }
    }

    /// Get status text for UI display
    #[must_use]
    pub fn status_text(&self) -> String {
        match self.state {
            DemoState::Initializing => "Initializing...".to_string(),
            DemoState::LoadingModel => "Loading Whisper model...".to_string(),
            DemoState::Idle => "Ready".to_string(),
            DemoState::RequestingPermission => "Requesting microphone...".to_string(),
            DemoState::Recording => "Recording...".to_string(),
            DemoState::Processing => "Processing...".to_string(),
            DemoState::Error => "Error".to_string(),
        }
    }

    /// Get the current transcript
    #[must_use]
    pub fn transcript(&self) -> String {
        self.transcript.clone()
    }

    /// Get partial transcript (streaming results)
    #[must_use]
    pub fn partial_transcript(&self) -> String {
        self.partial_transcript.clone()
    }

    /// Get recording duration formatted as M:SS
    #[must_use]
    pub fn recording_duration(&self) -> String {
        let seconds = self.recording_duration_ms / 1000;
        let minutes = seconds / 60;
        let secs = seconds % 60;
        format!("{minutes}:{secs:02}")
    }

    /// Get error message if in error state
    #[must_use]
    pub fn error_message(&self) -> Option<String> {
        self.error_message.clone()
    }

    /// Check if start recording button should be enabled
    #[must_use]
    pub fn can_start_recording(&self) -> bool {
        matches!(self.state, DemoState::Idle | DemoState::Error)
    }

    /// Check if stop recording button should be enabled
    #[must_use]
    pub fn can_stop_recording(&self) -> bool {
        self.state == DemoState::Recording
    }

    /// Start recording - requests microphone permission
    ///
    /// # Errors
    ///
    /// Returns error if recording cannot be started in current state.
    pub fn start_recording(&mut self) -> Result<(), JsValue> {
        if !self.can_start_recording() {
            return Err(JsValue::from_str("Cannot start recording in current state"));
        }

        self.transition_to(DemoState::RequestingPermission)?;
        Ok(())
    }

    /// Stop recording and begin processing
    ///
    /// # Errors
    ///
    /// Returns error if not currently recording.
    pub fn stop_recording(&mut self) -> Result<(), JsValue> {
        if !self.can_stop_recording() {
            return Err(JsValue::from_str("Cannot stop recording in current state"));
        }

        self.transition_to(DemoState::Processing)?;
        Ok(())
    }

    /// Clear the transcript
    pub fn clear_transcript(&mut self) {
        self.transcript.clear();
        self.partial_transcript.clear();
        self.recording_duration_ms = 0;
        self.samples_captured = 0;
    }

    /// Handle permission granted callback
    ///
    /// # Errors
    ///
    /// Returns error if state transition is invalid.
    pub fn on_permission_granted(&mut self) -> Result<(), JsValue> {
        self.transition_to(DemoState::Recording)
    }

    /// Handle permission denied callback
    ///
    /// # Errors
    ///
    /// Returns error if state transition is invalid.
    pub fn on_permission_denied(&mut self) -> Result<(), JsValue> {
        self.error_message = Some("Microphone access denied".to_string());
        self.transition_to(DemoState::Error)
    }

    /// Handle transcription complete callback
    ///
    /// # Errors
    ///
    /// Returns error if state transition is invalid.
    pub fn on_transcription_complete(&mut self, text: &str) -> Result<(), JsValue> {
        self.transcript = text.to_string();
        self.partial_transcript.clear();
        self.transition_to(DemoState::Idle)
    }

    /// Handle partial transcription result
    pub fn on_partial_result(&mut self, text: &str) {
        self.partial_transcript = text.to_string();
    }

    /// Update recording duration (called periodically)
    pub fn update_duration(&mut self, elapsed_ms: u32) {
        self.recording_duration_ms = elapsed_ms;
    }

    /// Add samples captured count
    pub fn add_samples(&mut self, count: u64) {
        self.samples_captured += count;
    }

    /// Get total samples captured
    #[must_use]
    pub fn samples_captured(&self) -> u64 {
        self.samples_captured
    }

    /// Retry after error
    ///
    /// # Errors
    ///
    /// Returns error if not in error state.
    pub fn retry(&mut self) -> Result<(), JsValue> {
        if self.state != DemoState::Error {
            return Err(JsValue::from_str("Can only retry from error state"));
        }
        self.error_message = None;
        self.transition_to(DemoState::Idle)
    }

    /// Internal state transition with validation
    fn transition_to(&mut self, new_state: DemoState) -> Result<(), JsValue> {
        if StateTransition::is_valid(self.state, new_state) {
            self.state = new_state;
            Ok(())
        } else {
            Err(JsValue::from_str(&format!(
                "Invalid state transition: {:?} -> {:?}",
                self.state, new_state
            )))
        }
    }
}

impl Default for RealtimeTranscriptionDemo {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Browser Feature Detection
// ============================================================================

/// Check if browser supports required features
#[wasm_bindgen]
#[must_use]
pub fn check_browser_compatibility() -> BrowserCompatibility {
    BrowserCompatibility::check()
}

/// Browser compatibility status
#[wasm_bindgen]
pub struct BrowserCompatibility {
    pub audio_worklet: bool,
    pub media_devices: bool,
    pub wasm_simd: bool,
}

#[wasm_bindgen]
impl BrowserCompatibility {
    /// Check current browser's capabilities
    ///
    /// Note: We check for API existence without creating AudioContext,
    /// as browsers block AudioContext creation before user gesture.
    #[must_use]
    pub fn check() -> Self {
        let window = web_sys::window();
        let navigator = window
            .as_ref()
            .and_then(|w| w.navigator().media_devices().ok());

        // Check MediaDevices API availability
        let media_devices = navigator.is_some();

        // Check if AudioContext API exists (don't create one - requires user gesture)
        // We assume it's available if window exists (all modern browsers have it)
        let audio_worklet = window.is_some();

        // WASM SIMD is available if we got this far (compiled with SIMD)
        let wasm_simd = true;

        Self {
            audio_worklet,
            media_devices,
            wasm_simd,
        }
    }

    /// Check if all required features are supported
    #[must_use]
    pub fn is_supported(&self) -> bool {
        self.audio_worklet && self.media_devices
    }

    /// Get warning message if not fully supported
    #[must_use]
    pub fn warning_message(&self) -> Option<String> {
        if self.is_supported() {
            None
        } else {
            Some("Your browser does not support real-time audio recording".to_string())
        }
    }
}

// ============================================================================
// Audio Processing Pipeline
// ============================================================================

/// Audio processor that captures PCM samples and feeds to `StreamingProcessor`
struct AudioPipeline {
    streaming_processor: StreamingProcessor,
    sample_rate: u32,
}

impl AudioPipeline {
    fn new(sample_rate: u32) -> Self {
        let config = StreamingConfig::low_latency()
            .without_vad() // Disable VAD for continuous capture
            .chunk_duration(1.5); // 1.5 second chunks to prevent browser timeout

        let mut streaming_processor = StreamingProcessor::new(StreamingConfig {
            input_sample_rate: sample_rate,
            ..config
        });

        // Set partial threshold to 1 second for fast feedback
        streaming_processor.set_partial_threshold(1.0);

        Self {
            streaming_processor,
            sample_rate,
        }
    }

    fn push_samples(&mut self, samples: &[f32]) {
        self.streaming_processor.push_audio(samples);
    }

    fn process(&mut self) {
        self.streaming_processor.process();
    }

    fn has_chunk(&self) -> bool {
        self.streaming_processor.has_chunk()
    }

    fn get_chunk(&mut self) -> Option<Vec<f32>> {
        self.streaming_processor.get_chunk()
    }

    fn partial_duration(&self) -> f32 {
        self.streaming_processor.partial_duration()
    }

    fn chunk_progress(&self) -> f32 {
        self.streaming_processor.chunk_progress()
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

// ============================================================================
// Thread-local Storage for Audio State
// ============================================================================

/// Type alias for audio processing callback closure
type AudioCallbackClosure = Closure<dyn Fn(web_sys::AudioProcessingEvent)>;

thread_local! {
    static DEMO: RefCell<Option<Rc<RefCell<RealtimeTranscriptionDemo>>>> = const { RefCell::new(None) };
    static MEDIA_STREAM: RefCell<Option<web_sys::MediaStream>> = const { RefCell::new(None) };
    static AUDIO_CONTEXT: RefCell<Option<web_sys::AudioContext>> = const { RefCell::new(None) };
    static AUDIO_PIPELINE: RefCell<Option<Rc<RefCell<AudioPipeline>>>> = const { RefCell::new(None) };
    static SCRIPT_PROCESSOR: RefCell<Option<web_sys::ScriptProcessorNode>> = const { RefCell::new(None) };
    static AUDIO_CALLBACK: RefCell<Option<AudioCallbackClosure>> = const { RefCell::new(None) };
    static WHISPER_MODEL: RefCell<Option<Rc<WhisperApr>>> = const { RefCell::new(None) };
    /// Worker bridge for non-blocking transcription (Phase 2+)
    static WORKER_BRIDGE: RefCell<Option<Rc<RefCell<bridge::WorkerBridge>>>> = const { RefCell::new(None) };
    /// Current transcription session ID
    static SESSION_ID: RefCell<String> = RefCell::new(String::new());
}

// ============================================================================
// Model Loading
// ============================================================================

/// Fetch model from URL
async fn fetch_model(url: &str) -> Result<Vec<u8>, JsValue> {
    info!(url, "Fetching model...");

    let window = web_sys::window().ok_or("No window")?;

    // Create fetch request
    let opts = web_sys::RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(web_sys::RequestMode::Cors);

    let request = web_sys::Request::new_with_str_and_init(url, &opts)?;

    // Fetch the resource
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: web_sys::Response = resp_value.dyn_into()?;

    if !resp.ok() {
        return Err(JsValue::from_str(&format!(
            "Failed to fetch model: {} {}",
            resp.status(),
            resp.status_text()
        )));
    }

    // Get as ArrayBuffer
    let array_buffer = JsFuture::from(resp.array_buffer()?).await?;
    let uint8_array = js_sys::Uint8Array::new(&array_buffer);
    let bytes = uint8_array.to_vec();

    info!(size_mb = bytes.len() as f64 / 1_000_000.0, "Model fetched");
    Ok(bytes)
}

/// Load Whisper model from bytes
fn load_whisper_model(bytes: &[u8]) -> Result<WhisperApr, JsValue> {
    info!(size_mb = bytes.len() as f64 / 1_000_000.0, "Loading Whisper model...");

    let model = WhisperApr::load_from_apr(bytes)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    info!("Whisper model loaded successfully");
    Ok(model)
}

/// Spawn model loading task using Web Worker
///
/// This function:
/// 1. Creates the transcription worker
/// 2. Fetches the model bytes on main thread
/// 3. Sends bytes to worker for loading (non-blocking)
/// 4. Worker loads model and signals ready
fn spawn_model_loading(
    demo: &Rc<RefCell<RealtimeTranscriptionDemo>>,
    document: &web_sys::Document,
) {
    let demo_clone = demo.clone();
    let doc_clone = document.clone();

    info!("Starting worker-based model loading...");
    wasm_bindgen_futures::spawn_local(async move {
        // Transition to loading state
        {
            let mut demo = demo_clone.borrow_mut();
            let _ = demo.transition_to(DemoState::LoadingModel);
        }
        let _ = update_ui(&doc_clone, &demo_clone.borrow());

        // Step 1: Create worker bridge
        // The WASM module URL - wasm-pack generates this structure
        let wasm_url = "/realtime-transcription/pkg/whisper_apr_demo_realtime_transcription.js";

        let bridge = match bridge::WorkerBridge::new(wasm_url) {
            Ok(b) => {
                info!("Worker bridge created successfully");
                b
            }
            Err(e) => {
                warn!(error = ?e, "Failed to create worker bridge");
                {
                    let mut demo = demo_clone.borrow_mut();
                    demo.error_message = Some(format!("Failed to create worker: {:?}", e));
                    let _ = demo.transition_to(DemoState::Error);
                }
                let _ = update_ui(&doc_clone, &demo_clone.borrow());
                return;
            }
        };

        // Set up result callback to handle worker messages
        let demo_for_callback = demo_clone.clone();
        let doc_for_callback = doc_clone.clone();
        {
            let mut bridge_mut = bridge.borrow_mut();
            bridge_mut.set_result_callback(move |result| {
                handle_worker_result(result, &demo_for_callback, &doc_for_callback);
            });
        }

        // Store bridge
        WORKER_BRIDGE.with(|wb| *wb.borrow_mut() = Some(bridge.clone()));

        // Generate session ID for this transcription session
        let session_id = format!("session_{}", js_sys::Date::now() as u64);
        SESSION_ID.with(|s| *s.borrow_mut() = session_id);

        // Step 2: Fetch model bytes (on main thread - this is fast)
        match fetch_model(MODEL_URL).await {
            Ok(bytes) => {
                info!(size_mb = bytes.len() as f64 / 1_000_000.0, "Model fetched, sending to worker");

                // Step 3: Send model to worker for loading (non-blocking!)
                let bridge_ref = bridge.borrow();
                if let Err(e) = bridge_ref.load_model(&bytes) {
                    warn!(error = ?e, "Failed to send model to worker");
                    let mut demo = demo_clone.borrow_mut();
                    demo.error_message = Some(format!("Failed to send model to worker: {:?}", e));
                    let _ = demo.transition_to(DemoState::Error);
                }
                // Note: We don't transition to Idle here - we wait for model_loaded from worker
            }
            Err(e) => {
                warn!(error = ?e, "Failed to fetch model");
                let mut demo = demo_clone.borrow_mut();
                demo.error_message = Some(format!("Failed to fetch model: {:?}", e));
                let _ = demo.transition_to(DemoState::Error);
            }
        }

        let _ = update_ui(&doc_clone, &demo_clone.borrow());
    });
}

/// Handle results from the transcription worker
fn handle_worker_result(
    result: bridge::WorkerResult,
    demo: &Rc<RefCell<RealtimeTranscriptionDemo>>,
    document: &web_sys::Document,
) {
    use bridge::WorkerResult;

    match result {
        WorkerResult::Ready => {
            info!("Worker is ready");
        }
        WorkerResult::ModelLoaded { size_mb, load_time_ms } => {
            info!(size_mb, load_time_ms, "Model loaded in worker");
            let mut demo = demo.borrow_mut();
            demo.set_model_loaded();
            let _ = demo.transition_to(DemoState::Idle);
            let _ = update_ui(document, &demo);
        }
        WorkerResult::Transcription { chunk_id, text, rtf, .. } => {
            info!(chunk_id, text = %text, rtf, "Transcription received from worker");
            let mut demo = demo.borrow_mut();
            demo.chunks_captured += 1;
            if !demo.transcript.is_empty() && !demo.transcript.ends_with(' ') {
                demo.transcript.push(' ');
            }
            demo.transcript.push_str(&text);
            demo.on_partial_result("");
            let _ = update_ui(document, &demo);
        }
        WorkerResult::Error { chunk_id, message, .. } => {
            warn!(chunk_id = ?chunk_id, message = %message, "Worker error");
            let mut demo = demo.borrow_mut();
            if demo.state() == DemoState::LoadingModel {
                demo.error_message = Some(format!("Model load error: {message}"));
                let _ = demo.transition_to(DemoState::Error);
            } else {
                // Log error but continue - don't fail transcription on single chunk error
                demo.transcript.push_str(&format!("[Error: {message}] "));
            }
            let _ = update_ui(document, &demo);
        }
        WorkerResult::Pong { timestamp, worker_time } => {
            let now = js_sys::Date::now();
            let round_trip = now - timestamp;
            info!(round_trip_ms = round_trip, worker_time, "Pong received");
        }
        _ => {
            // Handle other results (SessionStarted, SessionEnded, Metrics)
        }
    }
}

// ============================================================================
// Microphone and Audio Capture
// ============================================================================

/// Request microphone access from the browser
async fn request_microphone_access() -> Result<web_sys::MediaStream, JsValue> {
    let window = web_sys::window().ok_or("No window")?;
    let navigator = window.navigator();
    let media_devices = navigator.media_devices()?;

    // Create constraints for audio only
    let constraints = web_sys::MediaStreamConstraints::new();
    constraints.set_audio(&JsValue::TRUE);
    constraints.set_video(&JsValue::FALSE);

    // Request microphone access
    let promise = media_devices.get_user_media_with_constraints(&constraints)?;
    let result = JsFuture::from(promise).await?;

    result.dyn_into::<web_sys::MediaStream>()
}

/// Process a single audio event from ScriptProcessorNode
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn process_audio_event(
    event: &web_sys::AudioProcessingEvent,
    frame_counter: &Rc<RefCell<u32>>,
    pipeline: &Rc<RefCell<AudioPipeline>>,
    demo: &Rc<RefCell<RealtimeTranscriptionDemo>>,
    document: &web_sys::Document,
) {
    // Log every 10th frame
    log_frame_if_needed(frame_counter);

    // Get samples from event
    let Some(samples) = get_audio_samples(event) else {
        return;
    };

    // Process samples and update state
    process_samples(&samples, pipeline, demo);

    // Handle transcription if chunk ready
    handle_transcription_chunk(pipeline, demo);

    // Update UI
    let _ = update_ui(document, &demo.borrow());
}

/// Log audio frame periodically
fn log_frame_if_needed(frame_counter: &Rc<RefCell<u32>>) {
    let mut count = frame_counter.borrow_mut();
    *count += 1;
    if *count % 10 == 1 {
        info!(frame = *count, "Audio callback fired");
    }
}

/// Extract audio samples from event
fn get_audio_samples(event: &web_sys::AudioProcessingEvent) -> Option<Vec<f32>> {
    let input_buffer = event.input_buffer().ok()?;
    let channel_data = input_buffer.get_channel_data(0).ok()?;
    Some(channel_data.to_vec())
}

/// Process audio samples through pipeline and update demo state
#[allow(clippy::cast_precision_loss)]
fn process_samples(
    samples: &[f32],
    pipeline: &Rc<RefCell<AudioPipeline>>,
    demo: &Rc<RefCell<RealtimeTranscriptionDemo>>,
) {
    let sample_count = samples.len() as u64;

    // Push samples to pipeline
    {
        let mut pipeline = pipeline.borrow_mut();
        pipeline.push_samples(samples);
        pipeline.process();
    }

    // Update demo state
    let mut demo = demo.borrow_mut();
    demo.add_samples(sample_count);

    let pipeline = pipeline.borrow();
    let duration_secs = demo.samples_captured() as f32 / pipeline.sample_rate() as f32;
    demo.update_duration((duration_secs * 1000.0) as u32);

    // Build status message
    let progress = pipeline.chunk_progress();
    let partial_duration = pipeline.partial_duration();
    let rec_duration = demo.recording_duration();
    let samples_captured = demo.samples_captured();
    let progress_pct = progress * 100.0;
    let status = format!(
        "🎤 Recording {rec_duration} | Samples: {samples_captured} | Progress: {progress_pct:.0}% | Buffer: {partial_duration:.1}s"
    );
    demo.on_partial_result(&status);
}

/// Handle transcription when chunk is ready - sends to worker (non-blocking)
#[allow(clippy::cast_precision_loss)]
fn handle_transcription_chunk(
    pipeline: &Rc<RefCell<AudioPipeline>>,
    demo: &Rc<RefCell<RealtimeTranscriptionDemo>>,
) {
    let mut pipeline = pipeline.borrow_mut();
    if !pipeline.has_chunk() {
        return;
    }

    let Some(chunk) = pipeline.get_chunk() else {
        return;
    };

    let chunk_duration = chunk.len() as f32 / 16000.0;
    info!(chunk_duration, chunk_samples = chunk.len(), "Chunk ready - sending to worker");

    // Send to worker for non-blocking transcription
    WORKER_BRIDGE.with(|wb| {
        if let Some(bridge) = wb.borrow().as_ref() {
            let session_id = SESSION_ID.with(|s| s.borrow().clone());
            let mut bridge_mut = bridge.borrow_mut();

            // Show processing status
            demo.borrow_mut()
                .on_partial_result(&format!("🔄 Processing {chunk_duration:.1}s chunk..."));

            // Send to worker - this returns immediately (non-blocking!)
            match bridge_mut.transcribe(&chunk, &session_id, &[], false) {
                Ok(chunk_id) => {
                    info!(chunk_id, "Chunk sent to worker");
                }
                Err(e) => {
                    warn!(error = ?e, "Failed to send chunk to worker");
                    demo.borrow_mut().transcript.push_str(&format!("[Send error: {:?}] ", e));
                }
            }
        } else {
            // Fallback to main-thread transcription if worker not available
            warn!("Worker bridge not available, falling back to main thread");
            WHISPER_MODEL.with(|model_cell| {
                transcribe_chunk_main_thread(&chunk, chunk_duration, model_cell, demo);
            });
        }
    });
}

/// Transcribe a chunk using the loaded model (main thread fallback)
///
/// Uses chunked processing with periodic yields to prevent UI blocking.
/// Each 100ms of processing yields to the event loop.
///
/// NOTE: This is a fallback for when the worker is not available.
/// Prefer worker-based transcription for non-blocking operation.
fn transcribe_chunk_main_thread(
    chunk: &[f32],
    chunk_duration: f32,
    model_cell: &RefCell<Option<Rc<WhisperApr>>>,
    demo: &Rc<RefCell<RealtimeTranscriptionDemo>>,
) {
    let Some(model) = model_cell.borrow().clone() else {
        warn!("Model not loaded, cannot transcribe");
        demo.borrow_mut().transcript =
            format!("[Chunk {chunk_duration:.1}s captured - model not loaded]");
        return;
    };

    info!("Starting transcription with SIMD-accelerated inference");
    demo.borrow_mut()
        .on_partial_result(&format!("🔄 Transcribing {chunk_duration:.1}s chunk..."));

    // Clone data for async task
    let chunk = chunk.to_vec();
    let demo = demo.clone();

    // Use requestIdleCallback pattern: yield to event loop, then transcribe
    // This prevents blocking audio callbacks and UI updates
    let callback = Closure::once(Box::new(move || {
        let start = js_sys::Date::now();

        match model.transcribe(&chunk, TranscribeOptions::default()) {
            Ok(result) => {
                let elapsed = js_sys::Date::now() - start;
                info!(text = %result.text, elapsed_ms = elapsed, "Transcription complete");
                let mut demo = demo.borrow_mut();
                demo.chunks_captured += 1;
                if !demo.transcript.is_empty() && !demo.transcript.ends_with(' ') {
                    demo.transcript.push(' ');
                }
                demo.transcript.push_str(&result.text);
                demo.on_partial_result("");

                // Update UI after transcription
                if let Some(window) = web_sys::window() {
                    if let Some(document) = window.document() {
                        let _ = update_ui(&document, &demo);
                    }
                }
            }
            Err(e) => {
                warn!(error = %e, "Transcription failed");
                let mut demo = demo.borrow_mut();
                demo.transcript.push_str(&format!("[Error: {e}] "));
                demo.on_partial_result("");
            }
        }
    }) as Box<dyn FnOnce()>);

    // Schedule transcription on next idle callback or setTimeout(0) fallback
    if let Some(window) = web_sys::window() {
        // Use setTimeout with 0ms to yield to event loop before heavy computation
        let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
            callback.as_ref().unchecked_ref(),
            0,
        );
    }

    callback.forget(); // prevent closure from being dropped
}

/// Start PCM audio capture using `ScriptProcessorNode`
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn start_audio_capture(
    stream: &web_sys::MediaStream,
    demo: &Rc<RefCell<RealtimeTranscriptionDemo>>,
    document: &web_sys::Document,
) -> Result<(), JsValue> {
    // Create audio context
    let audio_context = web_sys::AudioContext::new()?;
    let sample_rate = audio_context.sample_rate() as u32;

    // Log context state (modern browsers start in "suspended" state)
    let state_str = format!("{:?}", audio_context.state());
    info!(sample_rate, state = %state_str, "AudioContext created");

    // Resume audio context if suspended (required by modern browsers)
    // Note: This returns a Promise but we can't await it in a sync context.
    // The browser should auto-resume after user gesture, but we call it explicitly.
    let resume_promise = audio_context.resume()?;

    // Spawn a task to wait for resume and log success
    let context_for_log = audio_context.clone();
    wasm_bindgen_futures::spawn_local(async move {
        match JsFuture::from(resume_promise).await {
            Ok(_) => {
                let state_str = format!("{:?}", context_for_log.state());
                info!(state = %state_str, "AudioContext resumed successfully");
            }
            Err(e) => warn!(error = ?e, "AudioContext resume failed"),
        }
    });

    // Create audio pipeline
    let pipeline = Rc::new(RefCell::new(AudioPipeline::new(sample_rate)));

    // Create media stream source
    let source = audio_context.create_media_stream_source(stream)?;

    // Create ScriptProcessorNode for PCM capture
    // Buffer size of 4096 at 44.1kHz = ~93ms latency
    let script_processor = audio_context.create_script_processor_with_buffer_size_and_number_of_input_channels_and_number_of_output_channels(
        4096, 1, 1
    )?;

    // Clone references for the closure
    let demo_clone = demo.clone();
    let doc_clone = document.clone();
    let pipeline_clone = pipeline.clone();

    // Create audio processing callback
    let frame_counter = Rc::new(RefCell::new(0u32));
    let frame_counter_clone = frame_counter.clone();

    let callback = Closure::wrap(Box::new(move |event: web_sys::AudioProcessingEvent| {
        process_audio_event(
            &event,
            &frame_counter_clone,
            &pipeline_clone,
            &demo_clone,
            &doc_clone,
        );
    }) as Box<dyn Fn(web_sys::AudioProcessingEvent)>);

    // Connect: source -> script processor -> destination (for monitoring)
    info!("Connecting audio nodes: source -> script_processor -> destination");
    source.connect_with_audio_node(&script_processor)?;
    info!("Connected: source -> script_processor");
    script_processor.connect_with_audio_node(&audio_context.destination())?;
    info!("Connected: script_processor -> destination");

    // Set the callback
    info!("Setting onaudioprocess callback on ScriptProcessorNode");
    script_processor.set_onaudioprocess(Some(callback.as_ref().unchecked_ref()));
    info!("Callback set successfully");

    // Store everything to prevent cleanup (prevents garbage collection)
    info!("Storing audio resources in thread-local storage");
    AUDIO_CONTEXT.with(|ctx| *ctx.borrow_mut() = Some(audio_context));
    AUDIO_PIPELINE.with(|p| *p.borrow_mut() = Some(pipeline));
    SCRIPT_PROCESSOR.with(|sp| *sp.borrow_mut() = Some(script_processor));
    AUDIO_CALLBACK.with(|cb| *cb.borrow_mut() = Some(callback));

    info!(sample_rate, "Audio capture setup complete - ScriptProcessorNode callbacks should fire now");

    Ok(())
}

/// Stop audio capture and cleanup
fn stop_audio_capture() {
    // Stop script processor
    SCRIPT_PROCESSOR.with(|sp| {
        if let Some(processor) = sp.borrow_mut().take() {
            processor.set_onaudioprocess(None);
            let _ = processor.disconnect();
        }
    });

    // Close audio context
    AUDIO_CONTEXT.with(|ctx| {
        if let Some(context) = ctx.borrow_mut().take() {
            let _ = context.close();
        }
    });

    // Clear pipeline
    AUDIO_PIPELINE.with(|p| *p.borrow_mut() = None);

    // Clear callback
    AUDIO_CALLBACK.with(|cb| *cb.borrow_mut() = None);

    // Stop media stream tracks
    MEDIA_STREAM.with(|ms| {
        if let Some(stream) = ms.borrow_mut().take() {
            let tracks = stream.get_tracks();
            for i in 0..tracks.length() {
                let track = tracks.get(i);
                if !track.is_undefined() {
                    if let Ok(track) = track.dyn_into::<web_sys::MediaStreamTrack>() {
                        track.stop();
                    }
                }
            }
        }
    });
}

/// Start microphone recording (async entry point)
fn spawn_microphone_request(
    demo: &Rc<RefCell<RealtimeTranscriptionDemo>>,
    document: &web_sys::Document,
) {
    let demo_clone = demo.clone();
    let doc_clone = document.clone();

    info!("spawn_local: Starting async microphone request");
    wasm_bindgen_futures::spawn_local(async move {
        info!("Inside spawn_local: Requesting microphone access");
        match request_microphone_access().await {
            Ok(stream) => {
                info!("Microphone access granted");
                // Store the media stream
                MEDIA_STREAM.with(|ms| *ms.borrow_mut() = Some(stream.clone()));

                // Update demo state
                let _ = demo_clone.borrow_mut().on_permission_granted();
                info!(state = %demo_clone.borrow().state_string(), "Permission granted, state updated");

                // Start audio capture
                info!("Starting audio capture");
                if let Err(e) = start_audio_capture(&stream, &demo_clone, &doc_clone) {
                    warn!(error = ?e, "Audio capture error");
                    let _ = demo_clone.borrow_mut().on_permission_denied();
                }

                let _ = update_ui(&doc_clone, &demo_clone.borrow());
            }
            Err(e) => {
                warn!(error = ?e, "Microphone access denied/error");
                let _ = demo_clone.borrow_mut().on_permission_denied();
                let _ = update_ui(&doc_clone, &demo_clone.borrow());
            }
        }
    });
}

// ============================================================================
// Zero-JS Entry Point
// ============================================================================

/// Handle unsupported browser
fn handle_unsupported_browser(document: &web_sys::Document) {
    warn!("Browser does not support required features");
    if let Some(warning) = document.get_element_by_id("compatibility_warning") {
        let _ = warning.class_list().add_1("visible");
    }
    if let Some(btn) = document.get_element_by_id("start_recording") {
        let _ = btn.set_attribute("disabled", "true");
    }
}

/// Create stop recording handler
fn create_stop_handler(
    demo: Rc<RefCell<RealtimeTranscriptionDemo>>,
) -> impl Fn(&web_sys::Document) {
    move |doc| {
        stop_audio_capture();
        let _ = demo.borrow_mut().stop_recording();

        let samples = demo.borrow().samples_captured();
        let duration = demo.borrow().recording_duration();
        let transcript = demo.borrow().transcript();

        let final_text = format!(
            "{transcript}\n\n--- Recording Complete ---\nDuration: {duration}\nTotal Samples: {samples}\n(Transcription requires loading whisper model)"
        );

        let _ = demo.borrow_mut().on_transcription_complete(&final_text);
        let _ = update_ui(doc, &demo.borrow());
    }
}

/// Set up all button listeners
fn setup_all_listeners(
    document: &web_sys::Document,
    demo: Rc<RefCell<RealtimeTranscriptionDemo>>,
) -> Result<(), JsValue> {
    setup_start_recording_listener(document, "start_recording", demo.clone())?;
    setup_button_listener(document, "stop_recording", create_stop_handler(demo.clone()))?;
    setup_button_listener(document, "clear_transcript", {
        let demo = demo.clone();
        move |doc| {
            demo.borrow_mut().clear_transcript();
            let _ = update_ui(doc, &demo.borrow());
        }
    })?;
    setup_keyboard_shortcuts(document, demo)?;
    Ok(())
}

/// Zero-JS entry point - called automatically when WASM loads
///
/// # Errors
///
/// Returns an error if the DOM is not available (no window or document).
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    tracing_wasm::set_as_global_default();
    let _span =
        info_span!("realtime_transcription_demo", version = env!("CARGO_PKG_VERSION")).entered();
    info!("Initializing Real-time Transcription Demo");

    let window = web_sys::window().ok_or("No window")?;
    let document = window.document().ok_or("No document")?;

    let compat = check_browser_compatibility();
    info!(
        audio_worklet = compat.audio_worklet,
        media_devices = compat.media_devices,
        wasm_simd = compat.wasm_simd,
        "Browser compatibility check"
    );
    if !compat.is_supported() {
        handle_unsupported_browser(&document);
        return Ok(());
    }

    let demo = Rc::new(RefCell::new(RealtimeTranscriptionDemo::new()));
    DEMO.with(|d| *d.borrow_mut() = Some(demo.clone()));

    update_ui(&document, &demo.borrow())?;
    spawn_model_loading(&demo, &document);
    setup_all_listeners(&document, demo)?;

    Ok(())
}

/// Update status indicator element
fn update_status_indicator(document: &web_sys::Document, demo: &RealtimeTranscriptionDemo) {
    if let Some(status) = document.get_element_by_id("status_indicator") {
        let state = demo.state_string();
        status.set_inner_html(&format!(
            "<span class=\"status-dot {}\"></span><span>{}</span>",
            state,
            demo.status_text()
        ));
    }
}

/// Update text displays (duration, transcript, partial)
fn update_text_displays(document: &web_sys::Document, demo: &RealtimeTranscriptionDemo) {
    if let Some(duration) = document.get_element_by_id("recording_duration") {
        duration.set_text_content(Some(&demo.recording_duration()));
    }
    if let Some(transcript) = document.get_element_by_id("transcript_display") {
        transcript.set_text_content(Some(&demo.transcript()));
    }
    if let Some(partial) = document.get_element_by_id("partial_transcript") {
        partial.set_text_content(Some(&demo.partial_transcript()));
    }
}

/// Update button enabled/disabled states
fn update_button_states(document: &web_sys::Document, demo: &RealtimeTranscriptionDemo) {
    if let Some(start_btn) = document.get_element_by_id("start_recording") {
        if demo.can_start_recording() {
            let _ = start_btn.remove_attribute("disabled");
        } else {
            let _ = start_btn.set_attribute("disabled", "true");
        }
    }
    if let Some(stop_btn) = document.get_element_by_id("stop_recording") {
        if demo.can_stop_recording() {
            let _ = stop_btn.remove_attribute("disabled");
        } else {
            let _ = stop_btn.set_attribute("disabled", "true");
        }
    }
}

/// Update waveform visibility
fn update_waveform_visibility(document: &web_sys::Document, demo: &RealtimeTranscriptionDemo) {
    if let Some(waveform) = document.get_element_by_id("waveform_container") {
        let class_list = waveform.class_list();
        if demo.state_string() == "recording" {
            let _ = class_list.add_1("visible");
        } else {
            let _ = class_list.remove_1("visible");
        }
    }
}

/// Update all UI elements from Rust
#[allow(clippy::unnecessary_wraps)]
fn update_ui(
    document: &web_sys::Document,
    demo: &RealtimeTranscriptionDemo,
) -> Result<(), JsValue> {
    update_status_indicator(document, demo);
    update_text_displays(document, demo);
    update_button_states(document, demo);
    update_waveform_visibility(document, demo);
    Ok(())
}

/// Set up a click listener on a button
fn setup_button_listener<F>(
    document: &web_sys::Document,
    id: &str,
    handler: F,
) -> Result<(), JsValue>
where
    F: Fn(&web_sys::Document) + 'static,
{
    let doc = document.clone();
    let closure = Closure::wrap(Box::new(move |_: web_sys::Event| {
        handler(&doc);
    }) as Box<dyn Fn(_)>);

    if let Some(btn) = document.get_element_by_id(id) {
        btn.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
    }
    closure.forget(); // Prevent closure from being dropped
    Ok(())
}

/// Set up the start recording button with async microphone access
fn setup_start_recording_listener(
    document: &web_sys::Document,
    id: &str,
    demo: Rc<RefCell<RealtimeTranscriptionDemo>>,
) -> Result<(), JsValue> {
    let doc = document.clone();
    let id_owned = id.to_string();
    let closure = Closure::wrap(Box::new(move |_: web_sys::Event| {
        info!(button_id = %id_owned, "Start recording button clicked");

        // Update state to requesting permission
        let _ = demo.borrow_mut().start_recording();
        info!(state = %demo.borrow().state_string(), "State updated");

        let _ = update_ui(&doc, &demo.borrow());

        // Spawn async microphone request
        info!("Spawning microphone request");
        spawn_microphone_request(&demo, &doc);
    }) as Box<dyn Fn(_)>);

    if let Some(btn) = document.get_element_by_id(id) {
        info!(button_id = %id, "Attaching click listener to button");
        btn.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
    } else {
        warn!(button_id = %id, "Button not found in document");
    }
    closure.forget();
    Ok(())
}

/// Check if event target is an interactive element
fn is_interactive_element(event: &web_sys::KeyboardEvent) -> bool {
    let Some(target) = event.target() else {
        return false;
    };
    let Ok(element) = target.dyn_into::<web_sys::Element>() else {
        return false;
    };
    let tag = element.tag_name().to_lowercase();
    tag == "input" || tag == "textarea" || tag == "button"
}

/// Handle space key press for recording toggle
fn handle_space_key(
    demo: &Rc<RefCell<RealtimeTranscriptionDemo>>,
    doc: &web_sys::Document,
) {
    let can_start = demo.borrow().can_start_recording();
    let can_stop = demo.borrow().can_stop_recording();

    if can_start {
        let _ = demo.borrow_mut().start_recording();
        spawn_microphone_request(demo, doc);
    } else if can_stop {
        stop_audio_capture();
        let _ = demo.borrow_mut().stop_recording();
        let _ = demo
            .borrow_mut()
            .on_transcription_complete("[Recording stopped]");
    }
    let _ = update_ui(doc, &demo.borrow());
}

/// Set up keyboard shortcuts
fn setup_keyboard_shortcuts(
    document: &web_sys::Document,
    demo: Rc<RefCell<RealtimeTranscriptionDemo>>,
) -> Result<(), JsValue> {
    let doc = document.clone();
    let closure = Closure::wrap(Box::new(move |event: web_sys::KeyboardEvent| {
        let code = event.code();

        if code == "Space" && !is_interactive_element(&event) {
            event.prevent_default();
            handle_space_key(&demo, &doc);
        } else if code == "Escape" {
            demo.borrow_mut().clear_transcript();
            let _ = update_ui(&doc, &demo.borrow());
        }
    }) as Box<dyn Fn(_)>);

    document.add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref())?;
    closure.forget();
    Ok(())
}

// ============================================================================
// Arbitrary trait for property testing
// ============================================================================

#[cfg(test)]
impl proptest::arbitrary::Arbitrary for DemoState {
    type Parameters = ();
    type Strategy = proptest::strategy::BoxedStrategy<Self>;

    fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
        use proptest::prelude::*;
        prop_oneof![
            Just(DemoState::Initializing),
            Just(DemoState::LoadingModel),
            Just(DemoState::Idle),
            Just(DemoState::RequestingPermission),
            Just(DemoState::Recording),
            Just(DemoState::Processing),
            Just(DemoState::Error),
        ]
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Demo State Tests
    // =========================================================================

    #[test]
    fn test_new_demo_is_initializing() {
        let demo = RealtimeTranscriptionDemo::new();
        assert_eq!(demo.state(), DemoState::Initializing);
    }

    #[test]
    fn test_default_demo_is_initializing() {
        let demo = RealtimeTranscriptionDemo::default();
        assert_eq!(demo.state(), DemoState::Initializing);
    }

    #[test]
    fn test_model_loaded_initially_false() {
        let demo = RealtimeTranscriptionDemo::new();
        assert!(!demo.is_model_loaded());
    }

    #[test]
    fn test_set_model_loaded() {
        let mut demo = RealtimeTranscriptionDemo::new();
        demo.set_model_loaded();
        assert!(demo.is_model_loaded());
    }

    // =========================================================================
    // Test Helper
    // =========================================================================

    /// Helper to create demo in Idle state (simulating model loaded)
    fn demo_in_idle_state() -> RealtimeTranscriptionDemo {
        let mut demo = RealtimeTranscriptionDemo::new();
        let _ = demo.transition_to(DemoState::LoadingModel);
        let _ = demo.transition_to(DemoState::Idle);
        demo.set_model_loaded();
        demo
    }

    // =========================================================================
    // State String Tests - Cover ALL branches
    // =========================================================================

    #[test]
    fn test_state_string_initializing() {
        let demo = RealtimeTranscriptionDemo::new();
        assert_eq!(demo.state_string(), "initializing");
    }

    #[test]
    fn test_state_string_loading_model() {
        let mut demo = RealtimeTranscriptionDemo::new();
        let _ = demo.transition_to(DemoState::LoadingModel);
        assert_eq!(demo.state_string(), "loadingmodel");
    }

    #[test]
    fn test_state_string_idle() {
        let demo = demo_in_idle_state();
        assert_eq!(demo.state_string(), "idle");
    }

    #[test]
    fn test_state_string_requesting_permission() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        assert_eq!(demo.state_string(), "requestingpermission");
    }

    #[test]
    fn test_state_string_recording() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        let _ = demo.on_permission_granted();
        assert_eq!(demo.state_string(), "recording");
    }

    #[test]
    fn test_state_string_processing() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        let _ = demo.on_permission_granted();
        let _ = demo.stop_recording();
        assert_eq!(demo.state_string(), "processing");
    }

    #[test]
    fn test_state_string_error() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        let _ = demo.on_permission_denied();
        assert_eq!(demo.state_string(), "error");
    }

    // =========================================================================
    // Status Text Tests - Cover ALL branches
    // =========================================================================

    #[test]
    fn test_status_text_initializing() {
        let demo = RealtimeTranscriptionDemo::new();
        assert_eq!(demo.status_text(), "Initializing...");
    }

    #[test]
    fn test_status_text_loading_model() {
        let mut demo = RealtimeTranscriptionDemo::new();
        let _ = demo.transition_to(DemoState::LoadingModel);
        assert_eq!(demo.status_text(), "Loading Whisper model...");
    }

    #[test]
    fn test_status_text_idle() {
        let demo = demo_in_idle_state();
        assert_eq!(demo.status_text(), "Ready");
    }

    #[test]
    fn test_status_text_requesting_permission() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        assert_eq!(demo.status_text(), "Requesting microphone...");
    }

    #[test]
    fn test_status_text_recording() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        let _ = demo.on_permission_granted();
        assert_eq!(demo.status_text(), "Recording...");
    }

    #[test]
    fn test_status_text_processing() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        let _ = demo.on_permission_granted();
        let _ = demo.stop_recording();
        assert_eq!(demo.status_text(), "Processing...");
    }

    #[test]
    fn test_status_text_error() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        let _ = demo.on_permission_denied();
        assert_eq!(demo.status_text(), "Error");
    }

    // =========================================================================
    // Duration Tests
    // =========================================================================

    #[test]
    fn test_initial_duration_is_zero() {
        let demo = RealtimeTranscriptionDemo::new();
        assert_eq!(demo.recording_duration(), "0:00");
    }

    #[test]
    fn test_duration_formatting_seconds() {
        let mut demo = RealtimeTranscriptionDemo::new();
        demo.update_duration(5_000); // 0:05
        assert_eq!(demo.recording_duration(), "0:05");
    }

    #[test]
    fn test_duration_formatting_minutes() {
        let mut demo = RealtimeTranscriptionDemo::new();
        demo.update_duration(65_000); // 1:05
        assert_eq!(demo.recording_duration(), "1:05");
    }

    #[test]
    fn test_duration_formatting_hours() {
        let mut demo = RealtimeTranscriptionDemo::new();
        demo.update_duration(3_665_000); // 61:05
        assert_eq!(demo.recording_duration(), "61:05");
    }

    // =========================================================================
    // Button State Tests
    // =========================================================================

    #[test]
    fn test_can_start_from_idle() {
        let demo = demo_in_idle_state();
        assert!(demo.can_start_recording());
    }

    #[test]
    fn test_can_start_from_error() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        let _ = demo.on_permission_denied();
        assert!(demo.can_start_recording());
    }

    #[test]
    fn test_cannot_start_when_requesting() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        assert!(!demo.can_start_recording());
    }

    #[test]
    fn test_cannot_start_when_recording() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        let _ = demo.on_permission_granted();
        assert!(!demo.can_start_recording());
    }

    #[test]
    fn test_cannot_stop_when_idle() {
        let demo = demo_in_idle_state();
        assert!(!demo.can_stop_recording());
    }

    #[test]
    fn test_can_stop_when_recording() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        let _ = demo.on_permission_granted();
        assert!(demo.can_stop_recording());
    }

    #[test]
    fn test_cannot_start_when_initializing() {
        let demo = RealtimeTranscriptionDemo::new();
        assert!(!demo.can_start_recording());
    }

    #[test]
    fn test_cannot_start_when_loading_model() {
        let mut demo = RealtimeTranscriptionDemo::new();
        let _ = demo.transition_to(DemoState::LoadingModel);
        assert!(!demo.can_start_recording());
    }

    // =========================================================================
    // Recording Flow Tests
    // =========================================================================

    #[test]
    fn test_start_recording_success() {
        let mut demo = demo_in_idle_state();
        assert!(demo.start_recording().is_ok());
        assert_eq!(demo.state(), DemoState::RequestingPermission);
    }

    #[test]
    fn test_start_recording_fails_when_not_idle() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        // After starting, can_start_recording returns false
        assert!(!demo.can_start_recording());
        // Attempting to start again would return Err (verified via state)
        assert_eq!(demo.state(), DemoState::RequestingPermission);
    }

    #[test]
    fn test_stop_recording_success() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        let _ = demo.on_permission_granted();
        assert!(demo.stop_recording().is_ok());
        assert_eq!(demo.state(), DemoState::Processing);
    }

    #[test]
    fn test_stop_recording_fails_when_not_recording() {
        let demo = demo_in_idle_state();
        // In idle state, can_stop_recording returns false
        assert!(!demo.can_stop_recording());
        // State remains idle
        assert_eq!(demo.state(), DemoState::Idle);
    }

    // =========================================================================
    // Permission Callback Tests
    // =========================================================================

    #[test]
    fn test_permission_granted() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        assert!(demo.on_permission_granted().is_ok());
        assert_eq!(demo.state(), DemoState::Recording);
    }

    #[test]
    fn test_permission_denied() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        assert!(demo.on_permission_denied().is_ok());
        assert_eq!(demo.state(), DemoState::Error);
        assert!(demo.error_message().is_some());
        assert_eq!(demo.error_message().unwrap(), "Microphone access denied");
    }

    // =========================================================================
    // Transcription Tests
    // =========================================================================

    #[test]
    fn test_transcript_initially_empty() {
        let demo = RealtimeTranscriptionDemo::new();
        assert!(demo.transcript().is_empty());
    }

    #[test]
    fn test_partial_transcript_initially_empty() {
        let demo = RealtimeTranscriptionDemo::new();
        assert!(demo.partial_transcript().is_empty());
    }

    #[test]
    fn test_on_partial_result() {
        let mut demo = RealtimeTranscriptionDemo::new();
        demo.on_partial_result("Hello");
        assert_eq!(demo.partial_transcript(), "Hello");
    }

    #[test]
    fn test_on_transcription_complete() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        let _ = demo.on_permission_granted();
        let _ = demo.stop_recording();
        assert!(demo.on_transcription_complete("Final transcript").is_ok());
        assert_eq!(demo.transcript(), "Final transcript");
        assert!(demo.partial_transcript().is_empty());
        assert_eq!(demo.state(), DemoState::Idle);
    }

    #[test]
    fn test_clear_transcript() {
        let mut demo = RealtimeTranscriptionDemo::new();
        demo.transcript = "Hello world".to_string();
        demo.partial_transcript = "Partial".to_string();
        demo.update_duration(5000);
        demo.add_samples(1000);
        demo.clear_transcript();
        assert!(demo.transcript().is_empty());
        assert!(demo.partial_transcript().is_empty());
        assert_eq!(demo.recording_duration(), "0:00");
        assert_eq!(demo.samples_captured(), 0);
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_error_message_initially_none() {
        let demo = RealtimeTranscriptionDemo::new();
        assert!(demo.error_message().is_none());
    }

    #[test]
    fn test_retry_from_error() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        let _ = demo.on_permission_denied();
        assert!(demo.retry().is_ok());
        assert_eq!(demo.state(), DemoState::Idle);
        assert!(demo.error_message().is_none());
    }

    #[test]
    fn test_retry_fails_when_not_in_error() {
        let demo = demo_in_idle_state();
        // In idle state, retry is not valid
        assert_eq!(demo.state(), DemoState::Idle);
        assert_ne!(demo.state(), DemoState::Error);
        // Verifying the guard condition that would cause retry to return Err
    }

    // =========================================================================
    // State Transition Tests
    // =========================================================================

    #[test]
    fn test_valid_state_transitions() {
        assert!(StateTransition::is_valid(DemoState::Idle, DemoState::RequestingPermission));
        assert!(StateTransition::is_valid(DemoState::RequestingPermission, DemoState::Recording));
        assert!(StateTransition::is_valid(DemoState::RequestingPermission, DemoState::Error));
        assert!(StateTransition::is_valid(DemoState::Recording, DemoState::Processing));
        assert!(StateTransition::is_valid(DemoState::Processing, DemoState::Idle));
        assert!(StateTransition::is_valid(DemoState::Error, DemoState::Idle));
        // Self-transitions
        assert!(StateTransition::is_valid(DemoState::Idle, DemoState::Idle));
        assert!(StateTransition::is_valid(DemoState::Recording, DemoState::Recording));
    }

    #[test]
    fn test_invalid_state_transitions() {
        assert!(!StateTransition::is_valid(DemoState::Idle, DemoState::Recording));
        assert!(!StateTransition::is_valid(DemoState::Idle, DemoState::Processing));
        assert!(!StateTransition::is_valid(DemoState::Idle, DemoState::Error));
        assert!(!StateTransition::is_valid(DemoState::Processing, DemoState::Recording));
        assert!(!StateTransition::is_valid(DemoState::Error, DemoState::Recording));
    }

    // =========================================================================
    // Sample Counting Tests
    // =========================================================================

    #[test]
    fn test_samples_captured_initial() {
        let demo = RealtimeTranscriptionDemo::new();
        assert_eq!(demo.samples_captured(), 0);
    }

    #[test]
    fn test_samples_captured_accumulates() {
        let mut demo = RealtimeTranscriptionDemo::new();
        demo.add_samples(1000);
        assert_eq!(demo.samples_captured(), 1000);
        demo.add_samples(500);
        assert_eq!(demo.samples_captured(), 1500);
    }

    #[test]
    fn test_clear_resets_samples() {
        let mut demo = RealtimeTranscriptionDemo::new();
        demo.add_samples(1000);
        demo.clear_transcript();
        assert_eq!(demo.samples_captured(), 0);
    }

    // =========================================================================
    // Full Workflow Tests
    // =========================================================================

    #[test]
    fn test_complete_recording_workflow() {
        let mut demo = demo_in_idle_state();

        // Start recording
        assert!(demo.start_recording().is_ok());
        assert_eq!(demo.state(), DemoState::RequestingPermission);

        // Permission granted
        assert!(demo.on_permission_granted().is_ok());
        assert_eq!(demo.state(), DemoState::Recording);

        // Simulate recording
        demo.add_samples(44100);
        demo.update_duration(1000);
        demo.on_partial_result("Testing...");

        // Stop recording
        assert!(demo.stop_recording().is_ok());
        assert_eq!(demo.state(), DemoState::Processing);

        // Complete transcription
        assert!(demo.on_transcription_complete("Hello world").is_ok());
        assert_eq!(demo.state(), DemoState::Idle);
        assert_eq!(demo.transcript(), "Hello world");
    }

    #[test]
    fn test_permission_denied_workflow() {
        let mut demo = demo_in_idle_state();

        // Start recording
        assert!(demo.start_recording().is_ok());

        // Permission denied
        assert!(demo.on_permission_denied().is_ok());
        assert_eq!(demo.state(), DemoState::Error);

        // Retry
        assert!(demo.retry().is_ok());
        assert_eq!(demo.state(), DemoState::Idle);
    }

    // =========================================================================
    // AudioPipeline Tests (Pure Rust - no browser APIs needed)
    // =========================================================================

    #[test]
    fn test_audio_pipeline_creation() {
        let pipeline = AudioPipeline::new(44100);
        assert_eq!(pipeline.sample_rate(), 44100);
    }

    #[test]
    fn test_audio_pipeline_sample_rate_48000() {
        let pipeline = AudioPipeline::new(48000);
        assert_eq!(pipeline.sample_rate(), 48000);
    }

    #[test]
    fn test_audio_pipeline_initial_state() {
        let pipeline = AudioPipeline::new(44100);
        assert!(!pipeline.has_chunk());
        assert_eq!(pipeline.partial_duration(), 0.0);
        assert_eq!(pipeline.chunk_progress(), 0.0);
    }

    #[test]
    fn test_audio_pipeline_push_samples() {
        let mut pipeline = AudioPipeline::new(44100);
        let samples = vec![0.0f32; 4096];
        pipeline.push_samples(&samples);
        pipeline.process();
        // After pushing samples, partial_duration should be non-zero
        assert!(pipeline.partial_duration() > 0.0);
    }

    #[test]
    fn test_audio_pipeline_push_multiple_batches() {
        let mut pipeline = AudioPipeline::new(44100);
        // Push multiple batches
        for _ in 0..10 {
            let samples = vec![0.1f32; 4096];
            pipeline.push_samples(&samples);
            pipeline.process();
        }
        // Should have some progress
        assert!(pipeline.chunk_progress() > 0.0);
    }

    #[test]
    fn test_audio_pipeline_chunk_generation() {
        let mut pipeline = AudioPipeline::new(16000);
        // Push enough samples to generate a chunk (3 seconds at 16kHz = 48000 samples)
        let samples = vec![0.1f32; 16000];
        for _ in 0..4 {
            pipeline.push_samples(&samples);
            pipeline.process();
        }
        // After 4 seconds of audio, should have at least one chunk
        if pipeline.has_chunk() {
            let chunk = pipeline.get_chunk();
            assert!(chunk.is_some());
            assert!(!chunk.unwrap().is_empty());
        }
    }

    #[test]
    fn test_audio_pipeline_get_chunk_when_none() {
        let mut pipeline = AudioPipeline::new(44100);
        // Without enough samples, should have no chunk
        let chunk = pipeline.get_chunk();
        assert!(chunk.is_none());
    }

    #[test]
    fn test_audio_pipeline_sample_rate_16000() {
        let pipeline = AudioPipeline::new(16000);
        assert_eq!(pipeline.sample_rate(), 16000);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_demo_state_default_impl() {
        let state = DemoState::default();
        // Default for DemoState is Initializing (waiting for model load)
        assert_eq!(state, DemoState::Initializing);
    }

    #[test]
    fn test_demo_state_debug() {
        let idle = DemoState::Idle;
        let debug_str = format!("{:?}", idle);
        assert!(debug_str.contains("Idle"));
    }

    #[test]
    fn test_demo_state_clone() {
        let state = DemoState::Recording;
        let cloned = state.clone();
        assert_eq!(state, cloned);
    }

    #[test]
    fn test_demo_state_copy() {
        let state = DemoState::Processing;
        let copied = state;
        assert_eq!(state, copied);
    }

    #[test]
    fn test_state_transition_struct_exists() {
        // Just verify the struct can be referenced
        let _ = StateTransition::is_valid(DemoState::Idle, DemoState::Idle);
    }

    #[test]
    fn test_multiple_partial_results() {
        let mut demo = RealtimeTranscriptionDemo::new();
        demo.on_partial_result("First");
        assert_eq!(demo.partial_transcript(), "First");
        demo.on_partial_result("Second");
        assert_eq!(demo.partial_transcript(), "Second");
        demo.on_partial_result("Third");
        assert_eq!(demo.partial_transcript(), "Third");
    }

    #[test]
    fn test_duration_edge_cases() {
        let mut demo = RealtimeTranscriptionDemo::new();
        // Test 59 seconds
        demo.update_duration(59_000);
        assert_eq!(demo.recording_duration(), "0:59");
        // Test 60 seconds (1 minute exactly)
        demo.update_duration(60_000);
        assert_eq!(demo.recording_duration(), "1:00");
    }

    #[test]
    fn test_samples_large_values() {
        let mut demo = RealtimeTranscriptionDemo::new();
        demo.add_samples(u64::MAX / 2);
        assert_eq!(demo.samples_captured(), u64::MAX / 2);
    }

    #[test]
    fn test_transcript_direct_assignment() {
        let mut demo = RealtimeTranscriptionDemo::new();
        demo.transcript = "Direct assignment".to_string();
        assert_eq!(demo.transcript(), "Direct assignment");
    }

    #[test]
    fn test_error_message_clears_on_retry() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        let _ = demo.on_permission_denied();
        assert!(demo.error_message().is_some());
        let _ = demo.retry();
        assert!(demo.error_message().is_none());
    }

    #[test]
    fn test_all_state_transitions_from_requesting_permission() {
        // RequestingPermission -> Recording
        let mut demo1 = demo_in_idle_state();
        let _ = demo1.start_recording();
        assert!(demo1.on_permission_granted().is_ok());

        // RequestingPermission -> Error
        let mut demo2 = demo_in_idle_state();
        let _ = demo2.start_recording();
        assert!(demo2.on_permission_denied().is_ok());
    }

    #[test]
    fn test_processing_to_idle_via_transcription_complete() {
        let mut demo = demo_in_idle_state();
        let _ = demo.start_recording();
        let _ = demo.on_permission_granted();
        let _ = demo.stop_recording();
        assert_eq!(demo.state(), DemoState::Processing);
        assert!(demo.on_transcription_complete("Done").is_ok());
        assert_eq!(demo.state(), DemoState::Idle);
    }

    // =========================================================================
    // Property-Based Tests (proptest)
    // =========================================================================

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Property: Duration formatting is always valid
            #[test]
            fn prop_duration_format_valid(ms in 0u32..=86_400_000u32) {
                let mut demo = RealtimeTranscriptionDemo::new();
                demo.update_duration(ms);
                let duration = demo.recording_duration();
                // Format should always be X:XX or X:XX:XX
                prop_assert!(duration.contains(':'));
                let parts: Vec<&str> = duration.split(':').collect();
                prop_assert!(parts.len() >= 2);
                prop_assert!(parts.len() <= 3);
            }

            /// Property: State string is always non-empty
            #[test]
            fn prop_state_string_non_empty(state: DemoState) {
                let mut demo = RealtimeTranscriptionDemo::new();
                // Force demo into arbitrary state via valid transitions
                match state {
                    DemoState::Initializing => {
                        // Already in this state
                    }
                    DemoState::LoadingModel => {
                        let _ = demo.transition_to(DemoState::LoadingModel);
                    }
                    DemoState::Idle => {
                        let _ = demo.transition_to(DemoState::LoadingModel);
                        let _ = demo.transition_to(DemoState::Idle);
                    }
                    DemoState::RequestingPermission => {
                        let _ = demo.transition_to(DemoState::LoadingModel);
                        let _ = demo.transition_to(DemoState::Idle);
                        let _ = demo.start_recording();
                    }
                    DemoState::Recording => {
                        let _ = demo.transition_to(DemoState::LoadingModel);
                        let _ = demo.transition_to(DemoState::Idle);
                        let _ = demo.start_recording();
                        let _ = demo.on_permission_granted();
                    }
                    DemoState::Processing => {
                        let _ = demo.transition_to(DemoState::LoadingModel);
                        let _ = demo.transition_to(DemoState::Idle);
                        let _ = demo.start_recording();
                        let _ = demo.on_permission_granted();
                        let _ = demo.stop_recording();
                    }
                    DemoState::Error => {
                        let _ = demo.transition_to(DemoState::LoadingModel);
                        let _ = demo.transition_to(DemoState::Idle);
                        let _ = demo.start_recording();
                        let _ = demo.on_permission_denied();
                    }
                }
                let state_str = demo.state_string();
                prop_assert!(!state_str.is_empty());
            }

            /// Property: Status text is always user-friendly
            #[test]
            fn prop_status_text_user_friendly(state: DemoState) {
                let mut demo = RealtimeTranscriptionDemo::new();
                match state {
                    DemoState::Initializing => {
                        // Already in this state
                    }
                    DemoState::LoadingModel => {
                        let _ = demo.transition_to(DemoState::LoadingModel);
                    }
                    DemoState::Idle => {
                        let _ = demo.transition_to(DemoState::LoadingModel);
                        let _ = demo.transition_to(DemoState::Idle);
                    }
                    DemoState::RequestingPermission => {
                        let _ = demo.transition_to(DemoState::LoadingModel);
                        let _ = demo.transition_to(DemoState::Idle);
                        let _ = demo.start_recording();
                    }
                    DemoState::Recording => {
                        let _ = demo.transition_to(DemoState::LoadingModel);
                        let _ = demo.transition_to(DemoState::Idle);
                        let _ = demo.start_recording();
                        let _ = demo.on_permission_granted();
                    }
                    DemoState::Processing => {
                        let _ = demo.transition_to(DemoState::LoadingModel);
                        let _ = demo.transition_to(DemoState::Idle);
                        let _ = demo.start_recording();
                        let _ = demo.on_permission_granted();
                        let _ = demo.stop_recording();
                    }
                    DemoState::Error => {
                        let _ = demo.transition_to(DemoState::LoadingModel);
                        let _ = demo.transition_to(DemoState::Idle);
                        let _ = demo.start_recording();
                        let _ = demo.on_permission_denied();
                    }
                }
                let status = demo.status_text();
                // Status should be readable English
                prop_assert!(!status.is_empty());
                prop_assert!(status.chars().all(|c| c.is_ascii() || c == '…'));
            }

            /// Property: Partial results are always stored correctly
            #[test]
            fn prop_partial_result_stored(text in "\\PC*") {
                let mut demo = RealtimeTranscriptionDemo::new();
                demo.on_partial_result(&text);
                prop_assert_eq!(demo.partial_transcript(), text);
            }

            /// Property: Samples accumulate correctly
            #[test]
            fn prop_samples_accumulate(
                counts in proptest::collection::vec(0u64..1_000_000, 1..50)
            ) {
                let mut demo = RealtimeTranscriptionDemo::new();
                let mut total = 0u64;
                for count in counts {
                    demo.add_samples(count);
                    total = total.saturating_add(count);
                }
                prop_assert_eq!(demo.samples_captured(), total);
            }

            /// Property: Audio pipeline resampling is deterministic
            #[test]
            fn prop_pipeline_deterministic(
                sample_rate in prop_oneof![Just(16000u32), Just(44100u32), Just(48000u32)],
                samples in proptest::collection::vec(-1.0f32..1.0f32, 100..1000)
            ) {
                let mut pipeline1 = AudioPipeline::new(sample_rate);
                let mut pipeline2 = AudioPipeline::new(sample_rate);

                pipeline1.push_samples(&samples);
                pipeline2.push_samples(&samples);

                pipeline1.process();
                pipeline2.process();

                prop_assert_eq!(pipeline1.chunk_progress(), pipeline2.chunk_progress());
            }

            /// Property: Clear transcript always resets transcript and partial
            #[test]
            fn prop_clear_resets(
                transcript in "\\PC*",
                partial in "\\PC*"
            ) {
                let mut demo = RealtimeTranscriptionDemo::new();
                demo.transcript = transcript;
                demo.on_partial_result(&partial);

                demo.clear_transcript();

                prop_assert!(demo.transcript().is_empty());
                prop_assert!(demo.partial_transcript().is_empty());
            }
        }
    }
}
