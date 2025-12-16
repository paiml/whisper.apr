//! Web Worker-based Transcription Demo (Zero JS)
//!
//! Uses wasm-bindgen-rayon for automatic Web Worker thread pool management.
//! All logic is pure Rust - no JavaScript required.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    WASM Module                               │
//! │  ┌─────────────────────────────────────────────────────────┐│
//! │  │              wasm-bindgen-rayon                         ││
//! │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              ││
//! │  │  │ Worker 1 │  │ Worker 2 │  │ Worker N │  (auto-managed)│
//! │  │  └────┬─────┘  └────┬─────┘  └────┬─────┘              ││
//! │  │       └─────────────┼─────────────┘                     ││
//! │  │                     ▼                                   ││
//! │  │            rayon::par_iter()                            ││
//! │  └─────────────────────────────────────────────────────────┘│
//! │                          │                                   │
//! │  ┌─────────────┐         ▼          ┌───────────────────┐  │
//! │  │   Audio     │───> Transcribe ───>│    UI Update      │  │
//! │  │  Capture    │    (parallel ops)  │  (non-blocking)   │  │
//! │  └─────────────┘                    └───────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Build Instructions
//!
//! ```bash
//! # Build with parallel feature (requires atomics)
//! RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
//!   wasm-pack build --target web --features parallel \
//!   -- -Z build-std=std,panic_abort
//! ```

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use tracing::info;
use whisper_apr::{TranscribeOptions, WhisperApr};

// ============================================================================
// Thread Pool Initialization (re-export from whisper-apr)
// ============================================================================

/// Initialize the rayon thread pool with specified worker count
///
/// Must be called before any parallel operations.
/// Requires COOP/COEP headers for SharedArrayBuffer.
#[cfg(feature = "parallel")]
#[wasm_bindgen(js_name = initThreadPool)]
pub async fn init_thread_pool(num_threads: usize) -> Result<(), JsValue> {
    whisper_apr::wasm::init_thread_pool(num_threads)
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to init thread pool: {e:?}")))
}

/// Check if threading is available (COOP/COEP headers present)
#[wasm_bindgen(js_name = isThreadingAvailable)]
pub fn is_threading_available() -> bool {
    whisper_apr::wasm::is_threaded_available()
}

/// Get optimal thread count for this environment
#[wasm_bindgen(js_name = getOptimalThreadCount)]
pub fn get_optimal_thread_count() -> usize {
    whisper_apr::wasm::optimal_thread_count()
}

// ============================================================================
// Demo State Machine
// ============================================================================

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DemoState {
    #[default]
    Uninitialized,
    InitializingThreads,
    LoadingModel,
    Ready,
    Recording,
    Transcribing,
    Error,
}

impl DemoState {
    fn name(self) -> &'static str {
        match self {
            Self::Uninitialized => "uninitialized",
            Self::InitializingThreads => "initializing_threads",
            Self::LoadingModel => "loading_model",
            Self::Ready => "ready",
            Self::Recording => "recording",
            Self::Transcribing => "transcribing",
            Self::Error => "error",
        }
    }
}

// ============================================================================
// Demo Application
// ============================================================================

thread_local! {
    static MODEL: RefCell<Option<Rc<WhisperApr>>> = const { RefCell::new(None) };
    static DEMO: RefCell<Option<Rc<RefCell<WorkerDemo>>>> = const { RefCell::new(None) };
}

/// Worker-based transcription demo
#[wasm_bindgen]
pub struct WorkerDemo {
    state: DemoState,
    transcript: String,
    error_message: Option<String>,
    thread_count: usize,
    model_size_mb: f64,
    last_rtf: Option<f64>,
}

#[wasm_bindgen]
impl WorkerDemo {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            state: DemoState::Uninitialized,
            transcript: String::new(),
            error_message: None,
            thread_count: 1,
            model_size_mb: 0.0,
            last_rtf: None,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn state(&self) -> DemoState {
        self.state
    }

    #[wasm_bindgen(js_name = stateName)]
    pub fn state_name(&self) -> String {
        self.state.name().to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn transcript(&self) -> String {
        self.transcript.clone()
    }

    #[wasm_bindgen(getter, js_name = errorMessage)]
    pub fn error_message(&self) -> Option<String> {
        self.error_message.clone()
    }

    #[wasm_bindgen(getter, js_name = threadCount)]
    pub fn thread_count(&self) -> usize {
        self.thread_count
    }

    #[wasm_bindgen(getter, js_name = modelSizeMb)]
    pub fn model_size_mb(&self) -> f64 {
        self.model_size_mb
    }

    #[wasm_bindgen(getter, js_name = lastRtf)]
    pub fn last_rtf(&self) -> Option<f64> {
        self.last_rtf
    }

    /// Clear transcript
    #[wasm_bindgen(js_name = clearTranscript)]
    pub fn clear_transcript(&mut self) {
        self.transcript.clear();
    }

    /// Append to transcript
    pub fn append_transcript(&mut self, text: &str) {
        if !self.transcript.is_empty() && !self.transcript.ends_with(' ') {
            self.transcript.push(' ');
        }
        self.transcript.push_str(text);
    }

    /// Set error state
    pub fn set_error(&mut self, message: &str) {
        self.state = DemoState::Error;
        self.error_message = Some(message.to_string());
    }

    /// Update RTF metric
    pub fn set_rtf(&mut self, rtf: f64) {
        self.last_rtf = Some(rtf);
    }
}

impl Default for WorkerDemo {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the demo (call once on page load)
#[wasm_bindgen(js_name = initDemo)]
pub async fn init_demo() -> Result<WorkerDemo, JsValue> {
    console_error_panic_hook::set_once();
    tracing_wasm::set_as_global_default();

    info!("Initializing Worker Transcription Demo");

    let mut demo = WorkerDemo::new();

    // Check threading availability
    if is_threading_available() {
        demo.state = DemoState::InitializingThreads;
        let thread_count = get_optimal_thread_count();
        demo.thread_count = thread_count;

        info!(threads = thread_count, "Threading available, initializing pool");

        #[cfg(feature = "parallel")]
        {
            init_thread_pool(thread_count).await?;
            info!(threads = thread_count, "Thread pool initialized");
        }
    } else {
        info!("Threading not available (missing COOP/COEP headers), running sequential");
        demo.thread_count = 1;
    }

    demo.state = DemoState::LoadingModel;
    Ok(demo)
}

/// Load model from bytes
#[wasm_bindgen(js_name = loadModel)]
pub fn load_model(demo: &mut WorkerDemo, model_bytes: &[u8]) -> Result<(), JsValue> {
    let size_mb = model_bytes.len() as f64 / 1_048_576.0;
    info!(size_mb, "Loading Whisper model");

    demo.model_size_mb = size_mb;

    match WhisperApr::load_from_apr(model_bytes) {
        Ok(model) => {
            MODEL.with(|m| *m.borrow_mut() = Some(Rc::new(model)));
            demo.state = DemoState::Ready;
            info!("Model loaded successfully");
            Ok(())
        }
        Err(e) => {
            demo.set_error(&format!("Failed to load model: {e:?}"));
            Err(JsValue::from_str(&format!("Failed to load model: {e:?}")))
        }
    }
}

/// Transcribe audio (uses rayon parallel ops internally if available)
#[wasm_bindgen]
pub fn transcribe(demo: &mut WorkerDemo, audio: &[f32]) -> Result<String, JsValue> {
    demo.state = DemoState::Transcribing;

    let start = js_sys::Date::now();

    MODEL.with(|model_cell| {
        let model = model_cell.borrow();
        let Some(model) = model.as_ref() else {
            demo.set_error("Model not loaded");
            return Err(JsValue::from_str("Model not loaded"));
        };

        match model.transcribe(audio, TranscribeOptions::default()) {
            Ok(result) => {
                let end = js_sys::Date::now();
                let duration_ms = end - start;
                let audio_duration = audio.len() as f64 / 16000.0;
                let rtf = (duration_ms / 1000.0) / audio_duration;

                demo.set_rtf(rtf);
                demo.append_transcript(&result.text);
                demo.state = DemoState::Ready;

                info!(
                    text = %result.text,
                    duration_ms,
                    audio_duration,
                    rtf,
                    "Transcription complete"
                );

                Ok(result.text)
            }
            Err(e) => {
                demo.set_error(&format!("Transcription failed: {e:?}"));
                Err(JsValue::from_str(&format!("Transcription failed: {e:?}")))
            }
        }
    })
}

// ============================================================================
// Zero-JS Entry Point
// ============================================================================

/// Auto-start entry point
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    tracing_wasm::set_as_global_default();
    info!("Worker Transcription Demo WASM module loaded");
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_state_name() {
        assert_eq!(DemoState::Ready.name(), "ready");
        assert_eq!(DemoState::Transcribing.name(), "transcribing");
    }

    #[test]
    fn test_demo_new() {
        let demo = WorkerDemo::new();
        assert_eq!(demo.state(), DemoState::Uninitialized);
        assert!(demo.transcript().is_empty());
    }

    #[test]
    fn test_demo_append_transcript() {
        let mut demo = WorkerDemo::new();
        demo.append_transcript("Hello");
        demo.append_transcript("World");
        assert_eq!(demo.transcript(), "Hello World");
    }
}
