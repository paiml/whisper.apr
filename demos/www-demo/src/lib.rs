//! WAPR-ZERO-JS-001: Zero-JavaScript Whisper.apr Demo
//!
//! Real-time streaming transcription - text appears as you speak.
//! ALL DOM creation and logic happens in Rust via `#[wasm_bindgen(start)]`.
//!
//! # Testing
//!
//! Worker context detection is tested via:
//! 1. `wasm_bindgen_test` - runs in actual browser headless
//! 2. Feature flag `test-mocks` - enables mock window for unit tests
//! 3. Property tests via proptest
//! 4. Mutation tests via cargo-mutants

// WASM demo code uses many casts that are safe for browser contexts
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cognitive_complexity)]
#![allow(clippy::similar_names)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::single_match_else)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::unused_self)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::type_complexity)]
#![allow(clippy::unnecessary_wraps)]

use std::cell::RefCell;
use std::rc::Rc;
use tracing::{info, warn};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use whisper_apr::wasm::{GpuDetectionWasm, TranscribeOptionsWasm, WhisperAprWasm};

// =============================================================================
// CONTEXT DETECTION MODULE (Testable)
// =============================================================================

/// Execution context for WASM module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmContext {
    /// Running in main browser window (has window, document, DOM)
    MainThread,
    /// Running in Web Worker (has self, postMessage, but NO window)
    Worker,
    /// Running in native Rust tests (no web APIs)
    #[cfg(feature = "test-mocks")]
    NativeTest,
}

impl WasmContext {
    /// Detect current execution context
    ///
    /// # Returns
    /// - `MainThread` if `window` exists (browser main thread)
    /// - `Worker` if no `window` (Web Worker context)
    ///
    /// # Platform Behavior
    /// - WASM: Uses `web_sys::window()` for real detection
    /// - Native: Always returns `Worker` (no web APIs available)
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(feature = "test-mocks")]
        {
            // In test mode with mocks, check mock state
            if MOCK_CONTEXT.with(|ctx| ctx.borrow().is_some()) {
                return MOCK_CONTEXT.with(|ctx| ctx.borrow().unwrap());
            }
        }

        // Real detection via web_sys (only on WASM target)
        #[cfg(target_arch = "wasm32")]
        {
            if web_sys::window().is_some() {
                return Self::MainThread;
            }
            return Self::Worker;
        }

        // Native targets: no web APIs, treat as Worker context
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::Worker
        }
    }

    /// Check if we're in worker context (no window)
    #[must_use]
    pub fn is_worker(&self) -> bool {
        matches!(self, Self::Worker)
    }

    /// Check if we're in main thread context (has window)
    #[must_use]
    pub fn is_main_thread(&self) -> bool {
        matches!(self, Self::MainThread)
    }
}

#[cfg(feature = "test-mocks")]
thread_local! {
    static MOCK_CONTEXT: RefCell<Option<WasmContext>> = const { RefCell::new(None) };
}

/// Set mock context for testing (only available with test-mocks feature)
#[cfg(feature = "test-mocks")]
pub fn set_mock_context(ctx: WasmContext) {
    MOCK_CONTEXT.with(|c| *c.borrow_mut() = Some(ctx));
}

/// Clear mock context after testing
#[cfg(feature = "test-mocks")]
pub fn clear_mock_context() {
    MOCK_CONTEXT.with(|c| *c.borrow_mut() = None);
}

pub mod audio_worklet;
pub mod audioworklet_js;
pub mod bridge;
pub mod ring_buffer;
pub mod worker;
pub mod worker_js;
pub mod worker_manager;

use audio_worklet::setup_audio_worklet;
use worker_manager::WorkerManager;

// Pareto optimal: INT8 with filterbank (37MB) - smallest working model
// INT4 quantization breaks accuracy, FP32 (146MB) offers no quality improvement
const MODEL_URL: &str = "/models/whisper-tiny-int8-fb.apr";

/// Application state
struct App {
    status: String,
    transcript: String,
    partial_text: String,
    is_recording: bool,
    is_starting: bool, // Jidoka: prevents accidental stop during startup
    model: Option<Rc<WhisperAprWasm>>, // For file upload (main thread)
    model_loaded: bool,                // For button state
    sample_rate: u32,
    // World-class UX state
    audio_level: f32,        // RMS level 0.0-1.0
    streaming_state: String, // "Listening", "Recording", "Transcribing"
}

impl Default for App {
    fn default() -> Self {
        Self {
            status: "Loading model...".to_string(),
            transcript: String::new(),
            partial_text: String::new(),
            is_recording: false,
            is_starting: false,
            model: None,
            model_loaded: false,
            sample_rate: 48000,
            // World-class UX defaults
            audio_level: 0.0,
            streaming_state: "Listening".to_string(),
        }
    }
}

thread_local! {
    static APP: RefCell<App> = RefCell::new(App::default());
    static AUDIO_CONTEXT: RefCell<Option<web_sys::AudioContext>> = const { RefCell::new(None) };
    static MEDIA_STREAM: RefCell<Option<web_sys::MediaStream>> = const { RefCell::new(None) };
    static WORKLET_NODE: RefCell<Option<web_sys::AudioWorkletNode>> = const { RefCell::new(None) };
    static WORKER_MANAGER: RefCell<Option<WorkerManager>> = const { RefCell::new(None) };
    static TRANSCRIPTION_LISTENER: RefCell<Option<Closure<dyn Fn(web_sys::CustomEvent)>>> = const { RefCell::new(None) };
}

/// Zero-JS entry point
///
/// # Context Detection
///
/// This function detects whether it's running in:
/// - Main thread (browser window) → Initialize UI
/// - Web Worker → Skip UI, just set up tracing
///
/// The detection uses `WasmContext::detect()` which is testable via:
/// - `test-mocks` feature flag for unit tests
/// - `wasm_bindgen_test` for browser tests
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    // Debug: log directly to console before tracing setup
    web_sys::console::log_1(&"[WASM] start() called".into());

    tracing_wasm::set_as_global_default();

    web_sys::console::log_1(&"[WASM] tracing initialized".into());

    // Detect execution context using testable abstraction
    let context = WasmContext::detect();

    // Workers load the same WASM but don't need UI initialization
    if context.is_worker() {
        web_sys::console::log_1(&"[WASM] Running in worker context, skipping UI init".into());
        return Ok(());
    }

    // Main thread: get window and initialize UI
    let window = web_sys::window().ok_or("No window in main thread context")?;

    info!("WAPR-REALTIME: Initializing streaming demo");
    let document = window.document().ok_or("No document")?;
    let body = document.body().ok_or("No body")?;

    body.set_inner_html("");

    // Set dark theme on body
    body.style()
        .set_css_text("background: #0d1117; color: #c9d1d9; margin: 0; padding: 0;");

    // Main container
    let main = create_element(&document, "main")?;
    set_styles(
        &main,
        "max-width: 800px; margin: 0 auto; padding: 2rem; font-family: system-ui, sans-serif;",
    )?;

    // Header
    let header = create_element(&document, "header")?;
    set_styles(&header, "text-align: center; margin-bottom: 2rem;")?;

    let h1 = create_element(&document, "h1")?;
    h1.set_text_content(Some("Whisper.apr"));
    set_styles(
        &h1,
        "color: #58a6ff; font-size: 2.5rem; margin-bottom: 0.5rem;",
    )?;
    header.append_child(&h1)?;

    let subtitle = create_element(&document, "p")?;
    subtitle.set_text_content(Some("Real-Time Speech Recognition"));
    set_styles(&subtitle, "color: #8b949e; font-size: 1.1rem;")?;
    header.append_child(&subtitle)?;
    main.append_child(&header)?;

    // Status
    let status = create_element(&document, "div")?;
    status.set_id("status");
    status.set_text_content(Some("Loading model..."));
    set_styles(&status, "background: #161b22; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 1.5rem; color: #8b949e; font-size: 0.9rem;")?;
    main.append_child(&status)?;

    // Recording indicator (with .recording-indicator class for probar)
    let indicator = create_element(&document, "div")?;
    indicator.set_id("indicator");
    indicator.set_class_name("recording-indicator");
    set_styles(&indicator, "display: none; flex-direction: column; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;")?;

    // Top row: recording dot + state label
    let indicator_top = create_element(&document, "div")?;
    set_styles(
        &indicator_top,
        "display: flex; align-items: center; gap: 0.5rem;",
    )?;

    let dot = create_element(&document, "span")?;
    dot.set_id("recording_dot");
    set_styles(&dot, "width: 12px; height: 12px; background: #f85149; border-radius: 50%; animation: pulse 1s infinite;")?;
    indicator_top.append_child(&dot)?;

    let label = create_element(&document, "span")?;
    label.set_id("state_label");
    label.set_text_content(Some("Listening..."));
    set_styles(&label, "color: #f85149; font-weight: 600;")?;
    indicator_top.append_child(&label)?;
    indicator.append_child(&indicator_top)?;

    // Audio level VU meter
    let vu_container = create_element(&document, "div")?;
    set_styles(
        &vu_container,
        "display: flex; align-items: center; gap: 0.5rem; width: 100%; max-width: 300px;",
    )?;

    let vu_label = create_element(&document, "span")?;
    vu_label.set_text_content(Some("🎙️"));
    set_styles(&vu_label, "font-size: 1.2rem;")?;
    vu_container.append_child(&vu_label)?;

    let vu_outer = create_element(&document, "div")?;
    vu_outer.set_id("vu_meter_outer");
    set_styles(
        &vu_outer,
        "flex: 1; height: 8px; background: #21262d; border-radius: 4px; overflow: hidden;",
    )?;

    let vu_inner = create_element(&document, "div")?;
    vu_inner.set_id("vu_meter");
    set_styles(&vu_inner, "height: 100%; width: 0%; background: linear-gradient(90deg, #238636, #2ea043, #f0883e); transition: width 50ms ease-out; border-radius: 4px;")?;
    vu_outer.append_child(&vu_inner)?;
    vu_container.append_child(&vu_outer)?;
    indicator.append_child(&vu_container)?;

    // Chunk progress bar
    let progress_container = create_element(&document, "div")?;
    set_styles(
        &progress_container,
        "display: flex; align-items: center; gap: 0.5rem; width: 100%; max-width: 300px;",
    )?;

    let progress_label = create_element(&document, "span")?;
    progress_label.set_id("progress_label");
    progress_label.set_text_content(Some("Chunk: 0%"));
    set_styles(
        &progress_label,
        "font-size: 0.75rem; color: #8b949e; min-width: 70px;",
    )?;
    progress_container.append_child(&progress_label)?;

    let progress_outer = create_element(&document, "div")?;
    progress_outer.set_id("chunk_progress_outer");
    set_styles(
        &progress_outer,
        "flex: 1; height: 4px; background: #21262d; border-radius: 2px; overflow: hidden;",
    )?;

    let progress_inner = create_element(&document, "div")?;
    progress_inner.set_id("chunk_progress");
    set_styles(&progress_inner, "height: 100%; width: 0%; background: #58a6ff; transition: width 100ms ease-out; border-radius: 2px;")?;
    progress_outer.append_child(&progress_inner)?;
    progress_container.append_child(&progress_outer)?;
    indicator.append_child(&progress_container)?;

    main.append_child(&indicator)?;

    // Button container for record and upload
    let button_container = create_element(&document, "div")?;
    set_styles(
        &button_container,
        "display: flex; gap: 1rem; margin-bottom: 1.5rem;",
    )?;

    // Record button
    let record_btn = document
        .create_element("button")?
        .dyn_into::<web_sys::HtmlButtonElement>()?;
    record_btn.set_id("record");
    record_btn.set_inner_text("🎤 Record");
    record_btn.set_disabled(true);
    set_styles(
        &record_btn.clone().dyn_into::<web_sys::HtmlElement>()?,
        "flex: 1; padding: 1.25rem 2rem; font-size: 1.25rem; font-weight: 600; border: none; border-radius: 12px; cursor: pointer; background: linear-gradient(135deg, #238636, #2ea043); color: white; transition: all 0.2s; box-shadow: 0 4px 12px rgba(35, 134, 54, 0.3);",
    )?;
    button_container.append_child(&record_btn)?;

    // Upload button
    let upload_btn = document
        .create_element("button")?
        .dyn_into::<web_sys::HtmlButtonElement>()?;
    upload_btn.set_id("upload");
    upload_btn.set_inner_text("📁 Upload Audio");
    upload_btn.set_disabled(true);
    set_styles(
        &upload_btn.clone().dyn_into::<web_sys::HtmlElement>()?,
        "flex: 1; padding: 1.25rem 2rem; font-size: 1.25rem; font-weight: 600; border: none; border-radius: 12px; cursor: pointer; background: linear-gradient(135deg, #1f6feb, #388bfd); color: white; transition: all 0.2s; box-shadow: 0 4px 12px rgba(31, 111, 235, 0.3);",
    )?;
    button_container.append_child(&upload_btn)?;

    // Hidden file input
    let file_input = document
        .create_element("input")?
        .dyn_into::<web_sys::HtmlInputElement>()?;
    file_input.set_id("file_input");
    file_input.set_type("file");
    file_input.set_accept("audio/*,video/*,.wav,.mp3,.mp4,.m4a,.ogg,.webm,.flac");
    set_styles(
        &file_input.clone().dyn_into::<web_sys::HtmlElement>()?,
        "display: none;",
    )?;
    button_container.append_child(&file_input)?;

    main.append_child(&button_container)?;

    // Transcript container
    let transcript_container = create_element(&document, "div")?;
    set_styles(
        &transcript_container,
        "background: #161b22; border-radius: 12px; overflow: hidden;",
    )?;

    let transcript_header = create_element(&document, "div")?;
    transcript_header.set_text_content(Some("Transcript"));
    set_styles(&transcript_header, "background: #21262d; padding: 0.75rem 1rem; color: #8b949e; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;")?;
    transcript_container.append_child(&transcript_header)?;

    let transcript = create_element(&document, "div")?;
    transcript.set_id("transcript");
    transcript.set_attribute("aria-live", "polite")?;
    set_styles(
        &transcript,
        "padding: 1.5rem; min-height: 200px; color: #c9d1d9; line-height: 1.8; font-size: 1.2rem;",
    )?;
    transcript.set_text_content(Some("Transcript will appear here as you speak..."));
    transcript_container.append_child(&transcript)?;
    main.append_child(&transcript_container)?;

    // Partial text (live updates)
    let partial = create_element(&document, "div")?;
    partial.set_id("partial");
    set_styles(
        &partial,
        "color: #8b949e; font-style: italic; margin-top: 0.5rem; min-height: 1.5rem;",
    )?;
    transcript_container.append_child(&partial)?;

    body.append_child(&main)?;

    // Add CSS animation for pulse
    let style = document.create_element("style")?;
    style.set_text_content(Some(
        r"
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.1); }
        }
        #record:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(35, 134, 54, 0.4);
        }
        #record:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
    ",
    ));
    document.head().ok_or("No head")?.append_child(&style)?;

    // Record button click handler
    let document_clone = document.clone();
    let onclick = Closure::wrap(Box::new(move |_: web_sys::MouseEvent| {
        handle_record_click(&document_clone);
    }) as Box<dyn Fn(_)>);
    record_btn.set_onclick(Some(onclick.as_ref().unchecked_ref()));
    onclick.forget();

    // Upload button click handler - triggers file input
    let file_input_clone = file_input.clone();
    let upload_onclick = Closure::wrap(Box::new(move |_: web_sys::MouseEvent| {
        file_input_clone.click();
    }) as Box<dyn Fn(_)>);
    upload_btn.set_onclick(Some(upload_onclick.as_ref().unchecked_ref()));
    upload_onclick.forget();

    // File input change handler
    let document_clone2 = document.clone();
    let file_onchange = Closure::wrap(Box::new(move |_: web_sys::Event| {
        web_sys::console::log_1(&"[WASM] File input change event fired!".into());
        handle_file_upload(&document_clone2);
    }) as Box<dyn Fn(_)>);
    file_input.set_onchange(Some(file_onchange.as_ref().unchecked_ref()));
    file_onchange.forget();

    // Load model
    web_sys::console::log_1(&"[WASM] UI created, starting model load".into());
    spawn_model_load(document);

    web_sys::console::log_1(&"[WASM] start() complete".into());
    info!("WAPR-REALTIME: Initialization complete");
    Ok(())
}

fn create_element(
    document: &web_sys::Document,
    tag: &str,
) -> Result<web_sys::HtmlElement, JsValue> {
    document
        .create_element(tag)?
        .dyn_into::<web_sys::HtmlElement>()
        .map_err(|e| JsValue::from_str(&format!("Cast failed: {e:?}")))
}

fn set_styles(element: &web_sys::HtmlElement, styles: &str) -> Result<(), JsValue> {
    element.style().set_css_text(styles);
    Ok(())
}

fn handle_record_click(document: &web_sys::Document) {
    let t0 = web_sys::window().unwrap().performance().unwrap().now();
    web_sys::console::log_1(&format!("[PERF] handle_record_click START t={t0:.2}ms").into());

    let model_loaded = APP.with(|app| app.borrow().model_loaded);
    if !model_loaded {
        web_sys::console::log_1(&"[PERF] No model, returning".into());
        return;
    }

    // Jidoka: Prevent accidental stop during startup sequence
    let is_starting = APP.with(|app| app.borrow().is_starting);
    if is_starting {
        web_sys::console::log_1(&"[PERF] is_starting=true, ignoring click (Jidoka)".into());
        return;
    }

    let is_recording = APP.with(|app| app.borrow().is_recording);
    web_sys::console::log_1(
        &format!(
            "[PERF] is_recording={} t={:.2}ms",
            is_recording,
            web_sys::window().unwrap().performance().unwrap().now() - t0
        )
        .into(),
    );

    if is_recording {
        stop_recording(document);
    } else {
        // Start recording with worker architecture
        web_sys::console::log_1(&"[PERF] Starting worker-based recording...".into());

        APP.with(|app| {
            let mut app = app.borrow_mut();
            app.is_recording = true;
            app.is_starting = true; // Jidoka: lock until recording established
            app.status = "Starting...".to_string();
            app.transcript.clear();
            app.partial_text.clear();
            app.streaming_state = "Starting".to_string();
        });

        update_ui(document);

        let doc = document.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Err(e) = start_recording(&doc).await {
                warn!(error = ?e, "Failed to start recording");
                APP.with(|app| {
                    let mut app = app.borrow_mut();
                    app.is_recording = false;
                    app.is_starting = false;
                    app.status = format!("Error: {e:?}");
                });
                update_ui(&doc);
            }
        });
    }
}

fn stop_recording(document: &web_sys::Document) {
    info!("Stopping recording...");

    // Stop worker recording
    WORKER_MANAGER.with(|wm| {
        if let Some(ref mut manager) = *wm.borrow_mut() {
            let _ = manager.stop_recording();
        }
    });

    // Stop AudioWorklet
    WORKLET_NODE.with(|node| {
        if let Some(worklet) = node.borrow_mut().take() {
            let _ = worklet.disconnect();
        }
    });

    // Close AudioContext
    AUDIO_CONTEXT.with(|ctx| {
        if let Some(context) = ctx.borrow_mut().take() {
            let _ = context.close();
        }
    });

    // Stop MediaStream tracks
    MEDIA_STREAM.with(|ms| {
        if let Some(stream) = ms.borrow_mut().take() {
            let tracks = stream.get_tracks();
            for i in 0..tracks.length() {
                if let Some(track) = tracks.get(i).dyn_ref::<web_sys::MediaStreamTrack>() {
                    track.stop();
                }
            }
        }
    });

    // Update app state
    APP.with(|app| {
        let mut app = app.borrow_mut();
        app.is_recording = false;
        app.is_starting = false;
        app.partial_text.clear();
        app.status = "Ready".to_string();
        app.audio_level = 0.0;
        app.streaming_state = "Listening".to_string();
    });

    update_ui(document);
}

/// Handle audio file upload and transcription
fn handle_file_upload(document: &web_sys::Document) {
    web_sys::console::log_1(&"[WASM] handle_file_upload called".into());
    let file_input = match document.get_element_by_id("file_input") {
        Some(el) => match el.dyn_into::<web_sys::HtmlInputElement>() {
            Ok(input) => input,
            Err(_) => return,
        },
        None => return,
    };

    let files = match file_input.files() {
        Some(f) => f,
        None => return,
    };

    let file = match files.get(0) {
        Some(f) => f,
        None => return,
    };

    let file_name = file.name();
    info!("Processing uploaded file: {}", file_name);

    // Update UI to show processing
    APP.with(|app| {
        app.borrow_mut().status = format!("Processing {file_name}...");
        app.borrow_mut().transcript.clear();
        app.borrow_mut().partial_text.clear();
    });
    update_ui(document);

    // Disable buttons while processing
    if let Some(btn) = document.get_element_by_id("upload") {
        if let Ok(btn) = btn.dyn_into::<web_sys::HtmlButtonElement>() {
            btn.set_disabled(true);
        }
    }
    if let Some(btn) = document.get_element_by_id("record") {
        if let Ok(btn) = btn.dyn_into::<web_sys::HtmlButtonElement>() {
            btn.set_disabled(true);
        }
    }

    // Spawn async task to process the file
    let doc = document.clone();
    wasm_bindgen_futures::spawn_local(async move {
        match process_audio_file(file, &doc).await {
            Ok(transcript) => {
                APP.with(|app| {
                    let mut app = app.borrow_mut();
                    app.transcript = transcript;
                    app.status = "Ready".to_string();
                });
            }
            Err(e) => {
                warn!(error = ?e, "Failed to process audio file");
                APP.with(|app| {
                    app.borrow_mut().status = format!("Error: {e:?}");
                });
            }
        }
        update_ui(&doc);
    });
}

/// Process an uploaded audio file and return transcript
async fn process_audio_file(
    file: web_sys::File,
    document: &web_sys::Document,
) -> Result<String, JsValue> {
    // Read file as ArrayBuffer
    let array_buffer = JsFuture::from(file.array_buffer()).await?;

    APP.with(|app| {
        app.borrow_mut().status = "Decoding audio...".to_string();
    });
    update_ui(document);

    // Create AudioContext to decode audio
    let audio_context = web_sys::AudioContext::new()?;

    // Convert JsValue to ArrayBuffer
    let array_buffer: js_sys::ArrayBuffer = array_buffer.dyn_into()?;

    // Decode the audio data
    let audio_buffer: web_sys::AudioBuffer =
        JsFuture::from(audio_context.decode_audio_data(&array_buffer)?)
            .await?
            .dyn_into()?;

    let sample_rate = audio_buffer.sample_rate();
    let num_channels = audio_buffer.number_of_channels();
    let length = audio_buffer.length() as usize;
    let duration = audio_buffer.duration();

    info!(
        sample_rate = sample_rate,
        channels = num_channels,
        length = length,
        duration = duration,
        "Audio decoded"
    );

    APP.with(|app| {
        app.borrow_mut().status = format!("Transcribing {duration:.1}s of audio...");
    });
    update_ui(document);

    // Get audio data (mix to mono if stereo)
    let samples: Vec<f32> = if num_channels == 1 {
        audio_buffer.get_channel_data(0)?.clone()
    } else {
        // Mix stereo to mono
        let left = audio_buffer.get_channel_data(0)?;
        let right = audio_buffer.get_channel_data(1)?;
        left.iter()
            .zip(right.iter())
            .map(|(l, r)| (l + r) / 2.0)
            .collect()
    };

    // Resample to 16kHz if needed (Whisper expects 16kHz)
    let samples_16k = if (sample_rate - 16000.0).abs() < 1.0 {
        samples
    } else {
        resample_audio(&samples, sample_rate as u32, 16000)
    };

    // Debug: log audio stats
    let audio_min = samples_16k.iter().copied().fold(f32::INFINITY, f32::min);
    let audio_max = samples_16k
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let audio_rms: f32 =
        (samples_16k.iter().map(|x| x * x).sum::<f32>() / samples_16k.len() as f32).sqrt();
    info!(
        samples = samples_16k.len(),
        duration_s = samples_16k.len() as f32 / 16000.0,
        min = audio_min,
        max = audio_max,
        rms = audio_rms,
        "Audio ready for transcription"
    );

    // Get model and transcribe
    let result = APP.with(|app| {
        let app = app.borrow();
        if let Some(model) = &app.model {
            // Use the model's transcribe function directly (not streaming)
            let options = TranscribeOptionsWasm::new();
            info!("Starting transcription...");
            match model.transcribe(&samples_16k, options) {
                Ok(result) => {
                    let text = result.text().clone();
                    info!(text_len = text.len(), "Transcription complete");
                    Ok(text)
                }
                Err(e) => Err(JsValue::from_str(&format!("Transcription failed: {e:?}"))),
            }
        } else {
            Err(JsValue::from_str("Model not loaded"))
        }
    })?;

    // Close audio context
    let _ = audio_context.close();

    Ok(result)
}

/// Simple linear interpolation resampling
fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = f64::from(from_rate) / f64::from(to_rate);
    let new_len = (samples.len() as f64 / ratio) as usize;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 * ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;

        let sample = if idx + 1 < samples.len() {
            samples[idx] * (1.0 - frac as f32) + samples[idx + 1] * frac as f32
        } else {
            samples[idx.min(samples.len() - 1)]
        };

        resampled.push(sample);
    }

    resampled
}

async fn start_recording(document: &web_sys::Document) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or("No window")?;
    let navigator = window.navigator();
    let media_devices = navigator.media_devices()?;

    // Get microphone access
    let constraints = web_sys::MediaStreamConstraints::new();
    constraints.set_audio(&JsValue::TRUE);
    constraints.set_video(&JsValue::FALSE);

    let promise = media_devices.get_user_media_with_constraints(&constraints)?;
    let stream: web_sys::MediaStream = JsFuture::from(promise).await?.dyn_into()?;

    let context = web_sys::AudioContext::new()?;
    let sample_rate = context.sample_rate() as u32;

    APP.with(|app| {
        app.borrow_mut().sample_rate = sample_rate;
    });

    info!(sample_rate, "Setting up worker-based recording");

    // Create WorkerManager if not exists
    let manager_exists = WORKER_MANAGER.with(|wm| wm.borrow().is_some());
    if !manager_exists {
        let mut manager = WorkerManager::new();
        manager.spawn(MODEL_URL)?;
        WORKER_MANAGER.with(|wm| {
            *wm.borrow_mut() = Some(manager);
        });

        // Wait for worker to be ready (model loading)
        // TODO: This should be event-driven, not polling
        web_sys::console::log_1(&"[Worker] Spawned, waiting for model load...".into());
    }

    // Get ring buffer from worker manager
    let ring_buffer = WORKER_MANAGER
        .with(|wm| {
            wm.borrow()
                .as_ref()
                .and_then(worker_manager::WorkerManager::get_ring_buffer)
        })
        .ok_or("No ring buffer")?;

    // Create audio source
    let source = context.create_media_stream_source(&stream)?;

    // Set up AudioWorklet (non-blocking audio capture)
    let worklet_node = setup_audio_worklet(&context, &ring_buffer, &source).await?;

    // Start worker processing (may be queued if model still loading)
    // Note: start_recording may fail if worker not ready yet - that's OK,
    // we'll retry when model-loaded event fires
    WORKER_MANAGER.with(|wm| {
        if let Some(ref mut manager) = *wm.borrow_mut() {
            let _ = manager.start_recording(sample_rate);
        }
    });

    // Set up event listener for model-loaded (to start recording and unlock Jidoka)
    // This is the ONLY place where is_starting gets cleared - ensures button
    // stays disabled until recording is truly established
    let sample_rate_for_closure = sample_rate;
    let doc_for_model_loaded = document.clone();
    let on_model_loaded = Closure::wrap(Box::new(move |_event: web_sys::CustomEvent| {
        web_sys::console::log_1(&"[Main] Model loaded event received".into());

        // Try to start recording (will succeed now that model is ready)
        let started = WORKER_MANAGER.with(|wm| {
            if let Some(ref mut manager) = *wm.borrow_mut() {
                manager.start_recording(sample_rate_for_closure).is_ok()
            } else {
                false
            }
        });

        // Clear is_starting (Jidoka unlock) - this is the critical transition
        // that re-enables the stop button
        APP.with(|app| {
            let mut app = app.borrow_mut();
            app.is_starting = false;
            app.status = "Listening...".to_string();
            app.streaming_state = "Recording".to_string();
        });
        update_ui(&doc_for_model_loaded);

        web_sys::console::log_1(
            &format!("[Main] Jidoka unlocked, recording_started={started}").into(),
        );
    }) as Box<dyn Fn(web_sys::CustomEvent)>);

    window.add_event_listener_with_callback(
        "whisper-model-loaded",
        on_model_loaded.as_ref().unchecked_ref(),
    )?;
    on_model_loaded.forget(); // Keep closure alive

    // Set up event listener for transcription results
    let doc = document.clone();
    let on_transcription = Closure::wrap(Box::new(move |event: web_sys::CustomEvent| {
        let detail = event.detail();

        let text = js_sys::Reflect::get(&detail, &"text".into())
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();

        let is_final = js_sys::Reflect::get(&detail, &"isFinal".into())
            .ok()
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        APP.with(|app| {
            let mut app = app.borrow_mut();
            if is_final {
                if !text.is_empty() {
                    if !app.transcript.is_empty() {
                        app.transcript.push(' ');
                    }
                    app.transcript.push_str(&text);
                    app.partial_text.clear();
                }
                app.streaming_state = "Recording".to_string();
            } else {
                app.partial_text = text;
                app.streaming_state = "Transcribing".to_string();
            }
        });

        update_ui(&doc);
    }) as Box<dyn Fn(web_sys::CustomEvent)>);

    window.add_event_listener_with_callback(
        "whisper-transcription",
        on_transcription.as_ref().unchecked_ref(),
    )?;

    // Store references
    AUDIO_CONTEXT.with(|ctx| {
        *ctx.borrow_mut() = Some(context);
    });
    MEDIA_STREAM.with(|ms| {
        *ms.borrow_mut() = Some(stream);
    });
    WORKLET_NODE.with(|node| {
        *node.borrow_mut() = Some(worklet_node);
    });
    TRANSCRIPTION_LISTENER.with(|cb| {
        *cb.borrow_mut() = Some(on_transcription);
    });

    info!(sample_rate, "Started worker-based real-time recording");
    Ok(())
}

fn update_ui(document: &web_sys::Document) {
    APP.with(|app| {
        let app = app.borrow();

        if let Some(el) = document.get_element_by_id("status") {
            el.set_text_content(Some(&app.status));
        }

        if let Some(el) = document.get_element_by_id("transcript") {
            if app.transcript.is_empty() && app.partial_text.is_empty() {
                el.set_text_content(Some("Transcript will appear here as you speak..."));
                let _ = el
                    .dyn_ref::<web_sys::HtmlElement>()
                    .map(|e| e.style().set_property("color", "#8b949e"));
            } else {
                el.set_text_content(Some(&app.transcript));
                let _ = el
                    .dyn_ref::<web_sys::HtmlElement>()
                    .map(|e| e.style().set_property("color", "#c9d1d9"));
            }
        }

        if let Some(el) = document.get_element_by_id("partial") {
            if app.partial_text.is_empty() {
                el.set_text_content(Some(""));
            } else {
                el.set_text_content(Some(&format!("{}▌", app.partial_text)));
            }
        }

        if let Some(indicator) = document.get_element_by_id("indicator") {
            if let Ok(el) = indicator.dyn_into::<web_sys::HtmlElement>() {
                if app.is_recording {
                    let _ = el.style().set_property("display", "flex");
                } else {
                    let _ = el.style().set_property("display", "none");
                }
            }
        }

        // World-class UX: VU meter
        if let Some(el) = document.get_element_by_id("vu_meter") {
            if let Ok(el) = el.dyn_into::<web_sys::HtmlElement>() {
                let width_pct = (app.audio_level * 100.0).min(100.0);
                let _ = el.style().set_property("width", &format!("{width_pct}%"));
            }
        }

        // World-class UX: State label
        if let Some(el) = document.get_element_by_id("state_label") {
            let (label, color) = match app.streaming_state.as_str() {
                "Starting" => ("Starting...", "#6e7681"),
                "Listening" => ("Listening...", "#8b949e"),
                "Transcribing" => ("Transcribing...", "#58a6ff"),
                // Default to Recording state (includes "Recording" and any unknown states)
                _ => ("Recording...", "#f85149"),
            };
            el.set_text_content(Some(label));
            if let Ok(html_el) = el.dyn_into::<web_sys::HtmlElement>() {
                let _ = html_el.style().set_property("color", color);
            }
        }

        // Hide chunk progress for now (worker doesn't report progress yet)
        if let Some(el) = document.get_element_by_id("chunk_progress_outer") {
            if let Ok(el) = el.dyn_into::<web_sys::HtmlElement>() {
                let _ = el.style().set_property("display", "none");
            }
        }

        if let Some(btn) = document.get_element_by_id("record") {
            if let Ok(btn) = btn.dyn_into::<web_sys::HtmlButtonElement>() {
                // Jidoka: Disable during startup to prevent accidental stop
                btn.set_disabled(!app.model_loaded || app.is_starting);

                if app.is_starting {
                    // Starting state: show spinner/starting indicator
                    btn.set_inner_text("⏳ Starting...");
                    let _ = btn
                        .style()
                        .set_property("background", "linear-gradient(135deg, #6e7681, #8b949e)");
                    let _ = btn
                        .style()
                        .set_property("box-shadow", "0 4px 12px rgba(110, 118, 129, 0.3)");
                } else if app.is_recording {
                    btn.set_inner_text("⏹ Stop");
                    let _ = btn
                        .style()
                        .set_property("background", "linear-gradient(135deg, #da3633, #f85149)");
                    let _ = btn
                        .style()
                        .set_property("box-shadow", "0 4px 12px rgba(248, 81, 73, 0.3)");
                } else {
                    btn.set_inner_text("🎤 Record");
                    let _ = btn
                        .style()
                        .set_property("background", "linear-gradient(135deg, #238636, #2ea043)");
                    let _ = btn
                        .style()
                        .set_property("box-shadow", "0 4px 12px rgba(35, 134, 54, 0.3)");
                }
            }
        }

        // Upload button - enabled when model is loaded and not recording/starting
        if let Some(btn) = document.get_element_by_id("upload") {
            if let Ok(btn) = btn.dyn_into::<web_sys::HtmlButtonElement>() {
                btn.set_disabled(!app.model_loaded || app.is_recording || app.is_starting);
            }
        }
    });
}

fn spawn_model_load(document: web_sys::Document) {
    wasm_bindgen_futures::spawn_local(async move {
        // Detect GPU capabilities
        let gpu_detection = GpuDetectionWasm::for_inference();
        let gpu_info = if gpu_detection.available() {
            format!(
                "GPU: {} ({})",
                gpu_detection.device_name(),
                gpu_detection.backend_name()
            )
        } else {
            "GPU: Not available (using SIMD)".to_string()
        };
        info!("GPU detection: {}", gpu_info);

        info!("Fetching model from {}...", MODEL_URL);

        APP.with(|app| {
            app.borrow_mut().status = "Downloading model (9MB)...".to_string();
        });
        update_ui(&document);

        match fetch_model(MODEL_URL).await {
            Ok(bytes) => {
                info!(size = bytes.len(), "Model downloaded, initializing...");

                APP.with(|app| {
                    app.borrow_mut().status = "Initializing model...".to_string();
                });
                update_ui(&document);

                match WhisperAprWasm::from_apr_bytes(&bytes) {
                    Ok(model) => {
                        info!("Model initialized successfully");
                        let ready_status = if gpu_detection.available() {
                            format!(
                                "Ready ({}) - Click to start speaking",
                                gpu_detection.backend_name()
                            )
                        } else {
                            "Ready (WASM SIMD) - Click to start speaking".to_string()
                        };
                        APP.with(|app| {
                            let mut app = app.borrow_mut();
                            app.model = Some(Rc::new(model));
                            app.model_loaded = true;
                            app.status = ready_status;
                        });
                    }
                    Err(e) => {
                        warn!(error = ?e, "Failed to initialize model");
                        APP.with(|app| {
                            app.borrow_mut().status = format!("Model error: {e:?}");
                        });
                    }
                }
            }
            Err(e) => {
                warn!(error = ?e, "Failed to fetch model");
                APP.with(|app| {
                    app.borrow_mut().status = format!("Download failed: {e:?}");
                });
            }
        }

        update_ui(&document);
    });
}

async fn fetch_model(url: &str) -> Result<Vec<u8>, JsValue> {
    let window = web_sys::window().ok_or("No window")?;
    let response: web_sys::Response = JsFuture::from(window.fetch_with_str(url))
        .await?
        .dyn_into()?;

    if !response.ok() {
        return Err(JsValue::from_str(&format!("HTTP {}", response.status())));
    }

    let array_buffer = JsFuture::from(response.array_buffer()?).await?;
    let uint8_array = js_sys::Uint8Array::new(&array_buffer);
    Ok(uint8_array.to_vec())
}

// =============================================================================
// TESTS: Worker Context Detection
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Unit Tests (with test-mocks feature)
    // =========================================================================

    #[test]
    #[cfg(feature = "test-mocks")]
    fn test_context_detect_worker_mock() {
        set_mock_context(WasmContext::Worker);
        let ctx = WasmContext::detect();
        assert!(ctx.is_worker());
        assert!(!ctx.is_main_thread());
        clear_mock_context();
    }

    #[test]
    #[cfg(feature = "test-mocks")]
    fn test_context_detect_main_thread_mock() {
        set_mock_context(WasmContext::MainThread);
        let ctx = WasmContext::detect();
        assert!(ctx.is_main_thread());
        assert!(!ctx.is_worker());
        clear_mock_context();
    }

    #[test]
    #[cfg(feature = "test-mocks")]
    fn test_context_detect_native_test_mock() {
        set_mock_context(WasmContext::NativeTest);
        let ctx = WasmContext::detect();
        assert!(!ctx.is_worker());
        assert!(!ctx.is_main_thread());
        clear_mock_context();
    }

    #[test]
    fn test_wasm_context_is_worker() {
        assert!(WasmContext::Worker.is_worker());
        assert!(!WasmContext::MainThread.is_worker());
    }

    #[test]
    fn test_wasm_context_is_main_thread() {
        assert!(WasmContext::MainThread.is_main_thread());
        assert!(!WasmContext::Worker.is_main_thread());
    }

    #[test]
    fn test_wasm_context_equality() {
        assert_eq!(WasmContext::Worker, WasmContext::Worker);
        assert_eq!(WasmContext::MainThread, WasmContext::MainThread);
        assert_ne!(WasmContext::Worker, WasmContext::MainThread);
    }

    #[test]
    fn test_wasm_context_debug() {
        let worker = format!("{:?}", WasmContext::Worker);
        let main = format!("{:?}", WasmContext::MainThread);
        assert!(worker.contains("Worker"));
        assert!(main.contains("MainThread"));
    }

    #[test]
    fn test_wasm_context_clone() {
        let ctx = WasmContext::Worker;
        let cloned = ctx;
        assert_eq!(ctx, cloned);
    }

    // =========================================================================
    // Native context detection test (no mocks)
    // In native Rust tests, web_sys::window() returns None
    // =========================================================================

    #[test]
    #[cfg(not(feature = "test-mocks"))]
    fn test_context_detect_native_is_worker() {
        // In native Rust tests, window() returns None, so it's detected as Worker
        let ctx = WasmContext::detect();
        assert!(
            ctx.is_worker(),
            "Native tests should detect as Worker (no window)"
        );
    }
}

// =============================================================================
// WASM BROWSER TESTS (run via wasm-pack test --headless --chrome)
// =============================================================================

#[cfg(target_arch = "wasm32")]
#[cfg(test)]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    /// Test that main thread correctly detects window
    #[wasm_bindgen_test]
    fn test_main_thread_has_window() {
        // This test runs in the main browser thread
        let ctx = WasmContext::detect();

        // In main browser thread, window exists
        assert!(
            ctx.is_main_thread(),
            "Browser main thread should detect as MainThread"
        );
        assert!(
            !ctx.is_worker(),
            "Browser main thread should NOT detect as Worker"
        );
    }

    /// Test window actually exists in main thread
    #[wasm_bindgen_test]
    fn test_web_sys_window_exists() {
        let window = web_sys::window();
        assert!(
            window.is_some(),
            "web_sys::window() should return Some in main thread"
        );
    }

    /// Test context detection is consistent
    #[wasm_bindgen_test]
    fn test_context_detection_consistent() {
        let ctx1 = WasmContext::detect();
        let ctx2 = WasmContext::detect();
        assert_eq!(ctx1, ctx2, "Context detection should be consistent");
    }

    /// Test WasmContext enum variants work correctly
    #[wasm_bindgen_test]
    fn test_context_enum_methods() {
        let worker = WasmContext::Worker;
        let main = WasmContext::MainThread;

        assert!(worker.is_worker());
        assert!(!worker.is_main_thread());
        assert!(main.is_main_thread());
        assert!(!main.is_worker());
    }
}

// =============================================================================
// PROPERTY TESTS (run via cargo test --features test-mocks)
// =============================================================================

#[cfg(test)]
#[cfg(feature = "test-mocks")]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    /// Generate arbitrary WasmContext values
    fn arb_context() -> impl Strategy<Value = WasmContext> {
        prop_oneof![
            Just(WasmContext::MainThread),
            Just(WasmContext::Worker),
            Just(WasmContext::NativeTest),
        ]
    }

    proptest! {
        /// Property: is_worker and is_main_thread are mutually exclusive (except NativeTest)
        #[test]
        fn prop_worker_main_thread_exclusive(ctx in arb_context()) {
            match ctx {
                WasmContext::MainThread => {
                    prop_assert!(ctx.is_main_thread());
                    prop_assert!(!ctx.is_worker());
                }
                WasmContext::Worker => {
                    prop_assert!(ctx.is_worker());
                    prop_assert!(!ctx.is_main_thread());
                }
                WasmContext::NativeTest => {
                    prop_assert!(!ctx.is_worker());
                    prop_assert!(!ctx.is_main_thread());
                }
            }
        }

        /// Property: context equality is reflexive
        #[test]
        fn prop_context_equality_reflexive(ctx in arb_context()) {
            prop_assert_eq!(ctx, ctx);
        }

        /// Property: mock context overrides detection
        #[test]
        fn prop_mock_context_overrides(ctx in arb_context()) {
            set_mock_context(ctx);
            let detected = WasmContext::detect();
            prop_assert_eq!(detected, ctx, "Mock should override detection");
            clear_mock_context();
        }

        /// Property: clearing mock returns to native detection
        #[test]
        fn prop_clear_mock_returns_native(ctx in arb_context()) {
            set_mock_context(ctx);
            clear_mock_context();
            // After clearing, detection should return Worker (no window in native tests)
            let detected = WasmContext::detect();
            prop_assert!(detected.is_worker(), "Native detection should return Worker");
        }
    }
}
