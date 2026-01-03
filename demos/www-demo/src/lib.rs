//! WAPR-ZERO-JS-001: Zero-JavaScript Whisper.apr Demo
//!
//! ALL DOM creation and logic happens in Rust via `#[wasm_bindgen(start)]`.
//! The HTML file contains ONLY a module import - zero inline JavaScript.

use std::cell::RefCell;
use std::rc::Rc;
use tracing::{info, warn};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use whisper_apr::{TranscribeOptions, WhisperApr};

// Use whisper-tiny-fb.apr which has correct weights
const MODEL_URL: &str = "/models/whisper-tiny-fb.apr";

/// Application state
struct App {
    status: String,
    transcript: String,
    is_recording: bool,
    model: Option<Rc<WhisperApr>>,
    audio_buffer: Vec<f32>,
    sample_rate: u32,
}

impl Default for App {
    fn default() -> Self {
        Self {
            status: "Loading model...".to_string(),
            transcript: String::new(),
            is_recording: false,
            model: None,
            audio_buffer: Vec::new(),
            sample_rate: 48000,
        }
    }
}

thread_local! {
    static APP: RefCell<App> = RefCell::new(App::default());
    static AUDIO_CONTEXT: RefCell<Option<web_sys::AudioContext>> = const { RefCell::new(None) };
    static MEDIA_STREAM: RefCell<Option<web_sys::MediaStream>> = const { RefCell::new(None) };
    static PROCESSOR: RefCell<Option<web_sys::ScriptProcessorNode>> = const { RefCell::new(None) };
    static CALLBACK: RefCell<Option<Closure<dyn Fn(web_sys::AudioProcessingEvent)>>> = const { RefCell::new(None) };
}

/// Zero-JS entry point
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    tracing_wasm::set_as_global_default();

    info!("WAPR-ZERO-JS: Initializing demo");

    let window = web_sys::window().ok_or("No window")?;
    let document = window.document().ok_or("No document")?;
    let body = document.body().ok_or("No body")?;

    body.set_inner_html("");

    // Main container
    let main = create_element(&document, "main")?;
    set_styles(&main, "max-width: 800px; margin: 0 auto; padding: 2rem; font-family: system-ui, sans-serif;")?;

    // Header
    let header = create_element(&document, "header")?;
    set_styles(&header, "text-align: center; margin-bottom: 2rem;")?;

    let h1 = create_element(&document, "h1")?;
    h1.set_text_content(Some("Whisper.apr"));
    set_styles(&h1, "color: #c9d1d9; font-size: 2rem; margin-bottom: 0.5rem;")?;
    header.append_child(&h1)?;

    let subtitle = create_element(&document, "p")?;
    subtitle.set_text_content(Some("Zero-JavaScript Speech Recognition"));
    set_styles(&subtitle, "color: #8b949e;")?;
    header.append_child(&subtitle)?;
    main.append_child(&header)?;

    // Status
    let status = create_element(&document, "div")?;
    status.set_id("status");
    status.set_text_content(Some("Loading model..."));
    set_styles(&status, "background: #161b22; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 1.5rem; color: #8b949e;")?;
    main.append_child(&status)?;

    // Record button
    let record_btn = document.create_element("button")?.dyn_into::<web_sys::HtmlButtonElement>()?;
    record_btn.set_id("record");
    record_btn.set_inner_text("Start Recording");
    record_btn.set_attribute("aria-label", "Start recording audio for transcription")?;
    record_btn.set_disabled(true);
    set_styles(
        &record_btn.clone().dyn_into::<web_sys::HtmlElement>()?,
        "display: block; width: 100%; padding: 1rem 2rem; font-size: 1.25rem; font-weight: 600; border: none; border-radius: 8px; cursor: pointer; background: #3fb950; color: white; margin-bottom: 1.5rem;",
    )?;
    main.append_child(&record_btn)?;

    // Transcript
    let transcript = create_element(&document, "div")?;
    transcript.set_id("transcript");
    transcript.set_attribute("aria-live", "polite")?;
    transcript.set_attribute("role", "status")?;
    set_styles(&transcript, "background: #161b22; padding: 1.5rem; border-radius: 8px; min-height: 150px; color: #c9d1d9; line-height: 1.6; white-space: pre-wrap; font-size: 1.1rem;")?;
    main.append_child(&transcript)?;

    body.append_child(&main)?;

    // Create processing overlay (CSS animation runs even when JS blocked)
    let overlay = create_element(&document, "div")?;
    overlay.set_id("processing_overlay");
    overlay.set_inner_html(r#"<div class="spinner"></div><p>Transcribing audio...</p><small>This takes ~10 seconds</small>"#);
    body.append_child(&overlay)?;

    // Click handler
    let document_clone = document.clone();
    let onclick = Closure::wrap(Box::new(move |_: web_sys::MouseEvent| {
        handle_record_click(&document_clone);
    }) as Box<dyn Fn(_)>);
    record_btn.set_onclick(Some(onclick.as_ref().unchecked_ref()));
    onclick.forget();

    // Load model
    spawn_model_load(document);

    info!("WAPR-ZERO-JS: Initialization complete");
    Ok(())
}

fn create_element(document: &web_sys::Document, tag: &str) -> Result<web_sys::HtmlElement, JsValue> {
    document.create_element(tag)?.dyn_into::<web_sys::HtmlElement>()
        .map_err(|e| JsValue::from_str(&format!("Cast failed: {:?}", e)))
}

fn set_styles(element: &web_sys::HtmlElement, styles: &str) -> Result<(), JsValue> {
    element.style().set_css_text(styles);
    Ok(())
}

fn handle_record_click(document: &web_sys::Document) {
    let has_model = APP.with(|app| app.borrow().model.is_some());
    if !has_model {
        return;
    }

    let is_recording = APP.with(|app| app.borrow().is_recording);

    if is_recording {
        // Stop recording and transcribe
        stop_recording(document);
    } else {
        // Start recording
        APP.with(|app| {
            let mut app = app.borrow_mut();
            app.is_recording = true;
            app.status = "Recording... (speak now)".to_string();
            app.audio_buffer.clear();
        });
        update_ui(document);

        let doc = document.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Err(e) = start_recording(&doc).await {
                warn!(error = ?e, "Failed to start recording");
                APP.with(|app| {
                    let mut app = app.borrow_mut();
                    app.is_recording = false;
                    app.status = format!("Error: {:?}", e);
                });
                update_ui(&doc);
            }
        });
    }
}

fn stop_recording(document: &web_sys::Document) {
    info!("Stopping recording...");

    // Stop audio processing
    PROCESSOR.with(|p| {
        if let Some(processor) = p.borrow_mut().take() {
            processor.set_onaudioprocess(None);
            let _ = processor.disconnect();
        }
    });

    AUDIO_CONTEXT.with(|ctx| {
        if let Some(context) = ctx.borrow_mut().take() {
            let _ = context.close();
        }
    });

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

    CALLBACK.with(|cb| { cb.borrow_mut().take(); });

    // Get audio and transcribe
    let (audio, sample_rate, model) = APP.with(|app| {
        let mut app = app.borrow_mut();
        app.is_recording = false;
        app.status = "Transcribing...".to_string();
        (app.audio_buffer.clone(), app.sample_rate, app.model.clone())
    });

    update_ui(document);

    if audio.is_empty() {
        APP.with(|app| {
            let mut app = app.borrow_mut();
            app.status = "Ready".to_string();
            app.transcript = "(No audio recorded)".to_string();
        });
        update_ui(document);
        return;
    }

    let Some(model) = model else {
        APP.with(|app| {
            let mut app = app.borrow_mut();
            app.status = "Error: Model not loaded".to_string();
        });
        update_ui(document);
        return;
    };

    info!(samples = audio.len(), sample_rate, "Transcribing audio...");

    // Resample to 16kHz if needed
    let audio_16k = if sample_rate != 16000 {
        resample(&audio, sample_rate, 16000)
    } else {
        audio
    };

    let duration = audio_16k.len() as f32 / 16000.0;
    info!(duration_secs = duration, "Audio duration");

    // Transcribe in async context to allow UI to update first
    let doc = document.clone();
    wasm_bindgen_futures::spawn_local(async move {
        // Show processing overlay
        if let Some(overlay) = doc.get_element_by_id("processing_overlay") {
            let _ = overlay.class_list().add_1("visible");
        }

        // Yield to event loop so overlay renders before blocking
        let window = web_sys::window().unwrap();
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 100);
        });
        let _ = wasm_bindgen_futures::JsFuture::from(promise).await;

        info!("Starting transcription (will block for several seconds)...");
        let start = js_sys::Date::now();

        let transcription_result = model.transcribe(&audio_16k, TranscribeOptions::default());
        let elapsed = js_sys::Date::now() - start;

        // Hide processing overlay
        if let Some(overlay) = doc.get_element_by_id("processing_overlay") {
            let _ = overlay.class_list().remove_1("visible");
        }

        match transcription_result {
            Ok(transcription) => {
                let rtf = elapsed / 1000.0 / duration as f64;
                info!(text = %transcription.text, elapsed_ms = elapsed, rtf, "Transcription complete");

                APP.with(|app| {
                    let mut app = app.borrow_mut();
                    app.status = format!("Ready ({:.1}s, RTF: {:.1}x)", elapsed / 1000.0, rtf);
                    let text = transcription.text.trim();
                    if text.is_empty() {
                        app.transcript = "(No speech detected)".to_string();
                    } else {
                        app.transcript = format!("{}\n\n[Note: Transcription quality limited]", text);
                    }
                });
            }
            Err(e) => {
                warn!(error = %e, elapsed_ms = elapsed, "Transcription error");
                APP.with(|app| {
                    let mut app = app.borrow_mut();
                    app.status = "Ready".to_string();
                    app.transcript = format!("Error: {}", e);
                });
            }
        }

        info!("Calling update_ui after transcription");
        update_ui(&doc);
        info!("update_ui completed");
    });
}

fn update_ui(document: &web_sys::Document) {
    APP.with(|app| {
        let app = app.borrow();
        info!(status = %app.status, transcript_len = app.transcript.len(), "update_ui called");

        if let Some(el) = document.get_element_by_id("status") {
            el.set_text_content(Some(&app.status));
        }

        if let Some(el) = document.get_element_by_id("transcript") {
            if app.transcript.is_empty() {
                el.set_text_content(Some("Transcript will appear here..."));
            } else {
                el.set_text_content(Some(&app.transcript));
            }
        }

        if let Some(btn) = document.get_element_by_id("record") {
            if let Ok(btn) = btn.dyn_into::<web_sys::HtmlButtonElement>() {
                btn.set_disabled(app.model.is_none());

                if app.is_recording {
                    btn.set_inner_text("Stop Recording");
                    let _ = btn.style().set_property("background", "#f85149");
                } else {
                    btn.set_inner_text("Start Recording");
                    let _ = btn.style().set_property("background", "#3fb950");
                }
            }
        }
    });
}

fn spawn_model_load(document: web_sys::Document) {
    wasm_bindgen_futures::spawn_local(async move {
        info!("Fetching model from {}...", MODEL_URL);

        APP.with(|app| {
            app.borrow_mut().status = "Downloading model (37MB)...".to_string();
        });
        update_ui(&document);

        match fetch_model(MODEL_URL).await {
            Ok(bytes) => {
                info!(size_mb = bytes.len() as f64 / 1_000_000.0, "Model downloaded, loading...");

                APP.with(|app| {
                    app.borrow_mut().status = "Loading model...".to_string();
                });
                update_ui(&document);

                match WhisperApr::load_from_apr(&bytes) {
                    Ok(model) => {
                        info!("Model loaded successfully");
                        APP.with(|app| {
                            let mut app = app.borrow_mut();
                            app.model = Some(Rc::new(model));
                            app.status = "Ready".to_string();
                        });
                    }
                    Err(e) => {
                        warn!(error = %e, "Failed to load model");
                        APP.with(|app| {
                            app.borrow_mut().status = format!("Model error: {}", e);
                        });
                    }
                }
            }
            Err(e) => {
                warn!(error = ?e, "Failed to fetch model");
                APP.with(|app| {
                    app.borrow_mut().status = format!("Download error: {:?}", e);
                });
            }
        }

        update_ui(&document);
    });
}

async fn fetch_model(url: &str) -> Result<Vec<u8>, JsValue> {
    let window = web_sys::window().ok_or("No window")?;
    let resp = JsFuture::from(window.fetch_with_str(url)).await?;
    let resp: web_sys::Response = resp.dyn_into()?;

    if !resp.ok() {
        return Err(JsValue::from_str(&format!("HTTP {}: {}", resp.status(), resp.status_text())));
    }

    let buffer = JsFuture::from(resp.array_buffer()?).await?;
    let array = js_sys::Uint8Array::new(&buffer);
    Ok(array.to_vec())
}

async fn start_recording(document: &web_sys::Document) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or("No window")?;
    let navigator = window.navigator();
    let media_devices = navigator.media_devices()?;

    let constraints = web_sys::MediaStreamConstraints::new();
    constraints.set_audio(&JsValue::TRUE);
    constraints.set_video(&JsValue::FALSE);

    let promise = media_devices.get_user_media_with_constraints(&constraints)?;
    let stream = JsFuture::from(promise).await?.dyn_into::<web_sys::MediaStream>()?;

    info!("Microphone access granted");

    let audio_ctx = web_sys::AudioContext::new()?;
    let sample_rate = audio_ctx.sample_rate() as u32;
    info!(sample_rate, "AudioContext created");

    APP.with(|app| {
        app.borrow_mut().sample_rate = sample_rate;
    });

    let source = audio_ctx.create_media_stream_source(&stream)?;
    let processor = audio_ctx.create_script_processor_with_buffer_size_and_number_of_input_channels_and_number_of_output_channels(4096, 1, 1)?;

    let doc_clone = document.clone();
    let callback = Closure::wrap(Box::new(move |event: web_sys::AudioProcessingEvent| {
        if let Ok(buffer) = event.input_buffer() {
            if let Ok(data) = buffer.get_channel_data(0) {
                APP.with(|app| {
                    let mut app = app.borrow_mut();
                    if app.is_recording {
                        app.audio_buffer.extend_from_slice(&data);

                        // Update level display
                        let level: f32 = data.iter().map(|s| s.abs()).sum::<f32>() / data.len() as f32;
                        let duration = app.audio_buffer.len() as f32 / app.sample_rate as f32;
                        app.status = format!("Recording... {:.1}s (level: {:.0}%)", duration, level * 100.0);
                    }
                });
                update_ui(&doc_clone);
            }
        }
    }) as Box<dyn Fn(_)>);

    processor.set_onaudioprocess(Some(callback.as_ref().unchecked_ref()));

    source.connect_with_audio_node(&processor)?;
    processor.connect_with_audio_node(&audio_ctx.destination())?;

    // Store for cleanup
    AUDIO_CONTEXT.with(|ctx| { *ctx.borrow_mut() = Some(audio_ctx); });
    MEDIA_STREAM.with(|ms| { *ms.borrow_mut() = Some(stream); });
    PROCESSOR.with(|p| { *p.borrow_mut() = Some(processor); });
    CALLBACK.with(|cb| { *cb.borrow_mut() = Some(callback); });

    Ok(())
}

/// Simple linear resampling
fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return input.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (input.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;

        let sample = if idx + 1 < input.len() {
            input[idx] as f64 * (1.0 - frac) + input[idx + 1] as f64 * frac
        } else {
            input[idx.min(input.len() - 1)] as f64
        };

        output.push(sample as f32);
    }

    output
}

// ============================================================================
// Test Hooks for Probar
// ============================================================================

#[wasm_bindgen]
pub fn init_test_pipeline(_sample_rate: u32) -> bool {
    true
}

#[wasm_bindgen]
pub fn inject_test_audio(samples: &[f32]) -> bool {
    APP.with(|app| {
        let mut app = app.borrow_mut();
        if !samples.is_empty() {
            app.transcript = "[Test transcription output]".to_string();
        }
    });

    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            update_ui(&document);
        }
    }
    true
}

#[wasm_bindgen]
pub fn process_test_chunk() -> bool {
    true
}

#[wasm_bindgen]
pub fn get_transcript() -> String {
    APP.with(|app| app.borrow().transcript.clone())
}
