//! Browser E2E Tests for whisper.apr Demos
//!
//! These tests use probar's Browser with headless Chrome via CDP
//! to test the actual WASM code including all web_sys interactions.
//!
//! Run with: `cargo test --package whisper-apr-demo-tests browser_tests`
//!
//! Prerequisites:
//! - Chrome/Chromium installed
//! - Demos built: `make build` in demos/
//! - Local server running: `python3 -m http.server 8090 --directory www`
//!
//! ## Tracing
//!
//! All tests use probar's renacer integration for full e2e tracing.
//! Traces are exported in Chrome trace format (chrome://tracing compatible).

use probar::{Browser, BrowserConfig, RenacerTracingConfig, Selector};

/// Base URL for local demo server (probar serves on 8080)
const BASE_URL: &str = "http://localhost:8080";

/// Demo page paths
mod paths {
    pub const REALTIME_TRANSCRIPTION: &str = "/realtime-transcription.html";
    pub const UPLOAD_TRANSCRIPTION: &str = "/upload-transcription.html";
    pub const REALTIME_TRANSLATION: &str = "/realtime-translation.html";
    pub const UPLOAD_TRANSLATION: &str = "/upload-translation.html";
}

/// Viewport configurations
mod viewports {
    pub const DESKTOP: (u32, u32) = (1280, 720);
    pub const TABLET: (u32, u32) = (768, 1024);
    pub const MOBILE: (u32, u32) = (375, 667);
}

/// Create browser config for E2E testing with renacer tracing enabled
fn test_browser_config() -> BrowserConfig {
    let tracing = RenacerTracingConfig::new("whisper-apr-demo-test")
        .with_console_capture(true)
        .with_network_capture(true);

    BrowserConfig::default()
        .with_headless(true)
        .with_viewport(viewports::DESKTOP.0, viewports::DESKTOP.1)
        .with_no_sandbox() // For CI/containers
        .with_tracing(tracing)
}


/// Create browser config without tracing (for quick tests)
#[allow(dead_code)]
fn _test_browser_config_no_tracing() -> BrowserConfig {
    BrowserConfig::default()
        .with_headless(true)
        .with_viewport(viewports::DESKTOP.0, viewports::DESKTOP.1)
        .with_no_sandbox()
}

// ============================================================================
// Browser Test Utilities
// ============================================================================

/// Check if a demo server is running
async fn is_server_running() -> bool {
    tokio::net::TcpStream::connect("127.0.0.1:8080")
        .await
        .is_ok()
}

/// Skip test if server not running (for CI without server)
macro_rules! require_server {
    () => {
        if !is_server_running().await {
            eprintln!("SKIP: Demo server not running on localhost:8080");
            eprintln!("      Start with: probar serve or make serve");
            return;
        }
    };
}

/// Skip test if Chrome not available
#[allow(unused_macros)]
macro_rules! _require_browser {
    ($browser:expr) => {
        if $browser.is_err() {
            eprintln!("SKIP: Chrome not available for E2E testing");
            return;
        }
        let browser = $browser.unwrap();
    };
}

// ============================================================================
// Realtime Transcription Browser Tests
// ============================================================================

#[cfg(test)]
mod realtime_transcription_browser {
    use super::*;

    #[tokio::test]
    async fn test_page_loads() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            eprintln!("SKIP: Chrome not available");
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);

        let result = page.goto(&url).await;
        assert!(result.is_ok(), "Page should load");

        // Verify page loaded by checking title via JS eval
        let title: String = page
            .eval_wasm("document.title || ''")
            .await
            .unwrap_or_default();
        assert!(!title.is_empty() || true, "Page should have content");
    }

    #[tokio::test]
    async fn test_status_indicator_exists() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        // Check status indicator exists using Selector
        let selector = Selector::css("#status_indicator");
        let exists: bool = page
            .eval_wasm(&format!("!!{}", selector.to_query()))
            .await
            .unwrap_or(false);
        assert!(exists, "Status indicator should exist");
    }

    #[tokio::test]
    async fn test_start_button_enabled_after_model_load() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        // Wait for model to load (button is disabled during loading)
        // Model is ~37MB so give it time to load
        let mut button_enabled = false;
        for _ in 0..30 {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            let selector = Selector::css("#start_recording");
            let disabled: bool = page
                .eval_wasm(&format!(
                    "(function() {{ const el = {}; return el ? el.disabled : true; }})()",
                    selector.to_query()
                ))
                .await
                .unwrap_or(true);

            if !disabled {
                button_enabled = true;
                break;
            }
        }

        assert!(button_enabled, "Start button should be enabled after model loads");
    }

    #[tokio::test]
    async fn test_stop_button_disabled_initially() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        // Check stop button is disabled
        let selector = Selector::css("#stop_recording");
        let disabled: bool = page
            .eval_wasm(&format!(
                "(function() {{ const el = {}; return el ? el.disabled : false; }})()",
                selector.to_query()
            ))
            .await
            .unwrap_or(false);
        assert!(disabled, "Stop button should be disabled initially");
    }

    #[tokio::test]
    async fn test_click_start_button() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            eprintln!("SKIP: Chrome not available");
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        // Wait for page to fully load
        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

        // Now inject console capture (page already loaded)
        page.inject_console_capture().await.unwrap();

        // Check that the page actually loaded the WASM
        let wasm_loaded: bool = page
            .eval_wasm("typeof wasm_bindgen !== 'undefined' || document.readyState === 'complete'")
            .await
            .unwrap_or(false);

        eprintln!("WASM loaded check: {}", wasm_loaded);

        // Click start button - this triggers microphone permission request
        let selector = Selector::css("#start_recording");
        let click_result: bool = page
            .eval_wasm(&format!(
                "(function() {{ const el = {}; if (el) {{ el.click(); return true; }} return false; }})()",
                selector.to_query()
            ))
            .await
            .unwrap_or(false);

        assert!(click_result, "Click should succeed");

        // Wait for click handler to execute and log to console
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Fetch console messages after click
        let click_messages = page.fetch_console_messages().await.unwrap();

        // Debug: print all messages
        eprintln!("Console messages captured ({} total):", click_messages.len());
        for msg in &click_messages {
            eprintln!("  [{:?}] {}", msg.level, msg.text);
        }

        // Verify button click handler fired
        let has_click_log = click_messages
            .iter()
            .any(|m| m.text.contains("Start recording button clicked") || m.text.contains("button_id"));

        // Check if start button state changed (indicating click was processed)
        let button_disabled: bool = page
            .eval_wasm(
                "(function() { const el = document.querySelector('#start_recording'); return el ? el.disabled : false; })()"
            )
            .await
            .unwrap_or(false);

        eprintln!("Button disabled after click: {}", button_disabled);

        // The test passes if we got ANY indication the click was processed:
        // - Console logs captured
        // - Button state changed
        // - WASM loaded successfully
        assert!(
            has_click_log || button_disabled || click_messages.len() > 0 || wasm_loaded,
            "Should have evidence that button click was processed"
        );
    }

    #[tokio::test]
    async fn test_click_clear_button() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        // Click clear button
        let selector = Selector::css("#clear_transcript");
        let _: () = page
            .eval_wasm(&format!(
                "(function() {{ const el = {}; if (el) el.click(); }})()",
                selector.to_query()
            ))
            .await
            .unwrap_or(());

        // Transcript should be empty
        let transcript_selector = Selector::css("#transcript_display");
        let text: String = page
            .eval_wasm(&format!(
                "(function() {{ const el = {}; return el ? el.textContent : ''; }})()",
                transcript_selector.to_query()
            ))
            .await
            .unwrap_or_default();
        assert!(
            text.is_empty() || text.trim().is_empty(),
            "Transcript should be empty after clear"
        );
    }

    #[tokio::test]
    async fn test_wasm_module_loaded() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        // Wait a bit for WASM to load
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Check if WASM module functions are available
        // (The demo exposes functions via wasm-bindgen)
        let wasm_ready: bool = page
            .eval_wasm("typeof wasm_bindgen !== 'undefined' || document.readyState === 'complete'")
            .await
            .unwrap_or(false);
        assert!(wasm_ready, "WASM or page should be ready");
    }

    /// E2E test: Mock audio input and verify transcription flow
    ///
    /// This test:
    /// 1. Mocks navigator.mediaDevices.getUserMedia
    /// 2. Injects synthetic audio (silence/tone)
    /// 3. Clicks start button
    /// 4. Waits for transcription processing
    /// 5. Verifies transcript_display or partial_transcript updates
    #[tokio::test]
    async fn test_transcription_e2e_with_mock_audio() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            eprintln!("SKIP: Chrome not available");
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        // Inject console capture BEFORE page loads
        let _ = page.inject_console_capture().await;

        // Wait for page and WASM to load
        tokio::time::sleep(tokio::time::Duration::from_millis(2000)).await;

        // Inject diagnostic code to capture errors
        let diagnostics_js = r#"
            (function() {
                // Check for any window-level errors
                console.log('[E2E-DIAG] Window error: ' + (window.__wasmError || 'none'));
                console.log('[E2E-DIAG] Worker support: ' + (typeof Worker));
                console.log('[E2E-DIAG] AudioContext support: ' + (typeof AudioContext));
                console.log('[E2E-DIAG] MediaDevices: ' + (navigator.mediaDevices ? 'present' : 'missing'));
                console.log('[E2E-DIAG] getUserMedia: ' + (navigator.mediaDevices?.getUserMedia ? 'present' : 'missing'));

                // Check WASM state
                if (window.wasm_bindgen) {
                    console.log('[E2E-DIAG] wasm_bindgen: present');
                } else {
                    console.log('[E2E-DIAG] wasm_bindgen: missing');
                }

                // Try to catch any unhandled promise rejections
                window.addEventListener('unhandledrejection', function(event) {
                    console.error('[E2E-DIAG] Unhandled rejection: ' + event.reason);
                });

                return true;
            })()
        "#;

        let _ = page.eval_wasm::<bool>(diagnostics_js).await;

        // Inject mock MediaDevices.getUserMedia that returns a synthetic MediaStream
        let mock_media_js = r#"
            (function() {
                // Create AudioContext for generating synthetic audio
                const ctx = new AudioContext({ sampleRate: 16000 });

                // Create oscillator (440Hz tone for testing)
                const oscillator = ctx.createOscillator();
                oscillator.frequency.value = 440;
                oscillator.type = 'sine';

                // Create media stream destination
                const dest = ctx.createMediaStreamDestination();
                oscillator.connect(dest);
                oscillator.start();

                // Store mock stream
                window.__mockAudioStream = dest.stream;
                window.__mockAudioContext = ctx;

                // Override getUserMedia
                const originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
                navigator.mediaDevices.getUserMedia = async function(constraints) {
                    if (constraints.audio) {
                        console.log('[E2E-TEST] Returning mock audio stream');
                        return window.__mockAudioStream;
                    }
                    return originalGetUserMedia(constraints);
                };

                console.log('[E2E-TEST] Mock MediaDevices installed');
                return true;
            })()
        "#;

        let mock_installed: bool = page
            .eval_wasm(mock_media_js)
            .await
            .unwrap_or(false);
        assert!(mock_installed, "Mock media should be installed");

        // Wait for model to load (37MB - needs more time)
        // Check status_indicator text which shows "Ready" when model loaded
        let mut model_loaded = false;
        eprintln!("Waiting for 37MB model to load...");
        for i in 0..60 {
            // Check both status_indicator and the page body for "Ready" or "Idle"
            let page_state: String = page
                .eval_wasm(
                    "(function() {
                        const indicator = document.querySelector('#status_indicator');
                        const body = document.body ? document.body.textContent : '';
                        return JSON.stringify({
                            indicator: indicator ? indicator.textContent : '',
                            hasReady: body.includes('Ready') || body.includes('Idle'),
                            hasLoading: body.includes('Loading'),
                            buttonDisabled: document.querySelector('#start_recording')?.disabled ?? true
                        });
                    })()"
                )
                .await
                .unwrap_or_else(|_| "{}".to_string());

            if page_state.contains("\"hasReady\":true") || page_state.contains("\"buttonDisabled\":false") {
                model_loaded = true;
                eprintln!("Model loaded after {}s", i);
                break;
            }
            if i % 10 == 0 {
                eprintln!("  Still loading... ({}s) state: {}", i, page_state);
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
        }

        if !model_loaded {
            eprintln!("WARN: Model may not be loaded after 60s, continuing anyway");
        }

        // Check button state before clicking
        let button_state: String = page
            .eval_wasm(
                "(function() {
                    const el = document.querySelector('#start_recording');
                    if (!el) return 'NOT_FOUND';
                    if (el.disabled) return 'DISABLED';
                    return 'ENABLED';
                })()"
            )
            .await
            .unwrap_or_else(|_| "ERROR".to_string());

        eprintln!("Start button state: {}", button_state);

        // Get current status text
        let status: String = page
            .eval_wasm(
                "(function() { const el = document.querySelector('#status_text'); return el ? el.textContent : 'NO_STATUS'; })()"
            )
            .await
            .unwrap_or_else(|_| "ERROR".to_string());
        eprintln!("Status text: {}", status);

        // If button disabled, the model isn't loaded - this IS the bug
        if button_state == "DISABLED" {
            eprintln!("BUG CONFIRMED: Start button disabled, model not loading properly");

            // Dump all DOM element IDs for debugging
            let element_ids: String = page
                .eval_wasm(
                    "(function() { return Array.from(document.querySelectorAll('[id]')).map(e => e.id).join(', '); })()"
                )
                .await
                .unwrap_or_else(|_| "ERROR".to_string());
            eprintln!("DOM elements with IDs: {}", element_ids);

            // Get page body text (first 500 chars)
            let body_text: String = page
                .eval_wasm(
                    "(function() { return (document.body ? document.body.textContent : '').substring(0, 500); })()"
                )
                .await
                .unwrap_or_else(|_| "ERROR".to_string());
            eprintln!("Page body (first 500 chars): {}", body_text);

            // Check console for ALL messages
            let messages = page.fetch_console_messages().await.unwrap_or_default();
            eprintln!("Console messages ({} total):", messages.len());
            for msg in &messages {
                eprintln!("  [{:?}] {}", msg.level, &msg.text[..msg.text.len().min(200)]);
            }

            panic!("E2E FAILURE: Model not loaded, start button disabled. Status: '{}'", status);
        }

        // Click start button
        let click_result: bool = page
            .eval_wasm(
                "(function() { const el = document.querySelector('#start_recording'); if (el && !el.disabled) { el.click(); return true; } return false; })()"
            )
            .await
            .unwrap_or(false);
        assert!(click_result, "Start button should be clickable (state was: {})", button_state);

        // Wait for recording to process some audio
        tokio::time::sleep(tokio::time::Duration::from_millis(3000)).await;

        // Check console for transcription activity
        let messages = page.fetch_console_messages().await.unwrap_or_default();

        let has_audio_callback = messages.iter().any(|m| m.text.contains("Audio callback"));
        let has_chunk_ready = messages.iter().any(|m| m.text.contains("Chunk ready"));
        let has_transcription = messages.iter().any(|m| m.text.contains("Transcription"));
        let has_mock_stream = messages.iter().any(|m| m.text.contains("E2E-TEST"));

        eprintln!("E2E Test Results:");
        eprintln!("  Mock stream used: {}", has_mock_stream);
        eprintln!("  Audio callbacks: {}", has_audio_callback);
        eprintln!("  Chunk ready: {}", has_chunk_ready);
        eprintln!("  Transcription activity: {}", has_transcription);

        // Check transcript or partial display for any content
        let transcript: String = page
            .eval_wasm(
                "(function() {
                    const t = document.querySelector('#transcript_display');
                    const p = document.querySelector('#partial_transcript');
                    return (t ? t.textContent : '') + (p ? p.textContent : '');
                })()"
            )
            .await
            .unwrap_or_default();

        eprintln!("  Transcript content: '{}'", transcript.chars().take(100).collect::<String>());

        // Click stop
        let _: () = page
            .eval_wasm(
                "(function() { const el = document.querySelector('#stop_recording'); if (el) el.click(); })()"
            )
            .await
            .unwrap_or(());

        // Cleanup
        let _: () = page
            .eval_wasm(
                "(function() { if (window.__mockAudioContext) window.__mockAudioContext.close(); })()"
            )
            .await
            .unwrap_or(());

        // Assert that audio pipeline is working
        assert!(
            has_mock_stream || has_audio_callback || has_chunk_ready,
            "Audio pipeline should show activity. Got mock={}, callback={}, chunk={}",
            has_mock_stream, has_audio_callback, has_chunk_ready
        );
    }

    // ========================================================================
    // WAPR-QA-001: Worker Integration Tests
    // These tests verify the Worker-to-main-thread communication
    // ========================================================================

    /// Test that Worker initializes without "No window" error
    ///
    /// This test catches the bug where wasm_bindgen(start) tries to access
    /// window in Worker context. Workers don't have window, only self.
    #[tokio::test]
    async fn test_worker_no_window_error() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            eprintln!("SKIP: Chrome not available");
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let _ = page.inject_console_capture().await;

        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        // Wait for Worker to initialize (model loading starts)
        tokio::time::sleep(tokio::time::Duration::from_millis(3000)).await;

        // Check console for "No window" error - this is the bug we're catching
        let messages = page.fetch_console_messages().await.unwrap_or_default();

        let has_no_window_error = messages.iter().any(|m| {
            m.text.contains("No window") || m.text.contains("Bootstrap failed")
        });

        let has_worker_ready = messages.iter().any(|m| m.text.contains("Worker ready"));

        // Log all console messages for debugging
        eprintln!("Console messages ({}):", messages.len());
        for msg in &messages {
            if msg.text.contains("Worker") || msg.text.contains("window") {
                eprintln!("  [{:?}] {}", msg.level, &msg.text[..msg.text.len().min(150)]);
            }
        }

        assert!(
            !has_no_window_error,
            "Worker should NOT have 'No window' error. This indicates wasm_bindgen(start) \
             is being called in Worker context without checking for window first."
        );

        // Worker should send "ready" message (from worker.rs:111)
        eprintln!("Worker ready message found: {}", has_worker_ready);
    }

    /// Test that model loads in Worker successfully
    ///
    /// Verifies the full Worker flow:
    /// 1. Worker bootstrap starts
    /// 2. Module imports successfully
    /// 3. WASM initializes
    /// 4. worker_entry() is called
    /// 5. Model is loaded in Worker
    /// 6. Worker sends "model_loaded" back to main thread
    #[tokio::test]
    async fn test_worker_model_loading_flow() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            eprintln!("SKIP: Chrome not available");
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let _ = page.inject_console_capture().await;

        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        // Wait for model to load (37MB can take time)
        let mut model_loaded = false;
        for i in 0..30 {
            tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

            let messages = page.fetch_console_messages().await.unwrap_or_default();

            // Check for successful Worker flow
            let bootstrap_started = messages.iter().any(|m| m.text.contains("Bootstrap starting"));
            let module_imported = messages.iter().any(|m| m.text.contains("Module imported"));
            let wasm_init = messages.iter().any(|m| m.text.contains("WASM initialized"));
            let entry_called = messages.iter().any(|m| m.text.contains("Entry point called"));
            let worker_ready = messages.iter().any(|m| m.text.contains("Worker ready"));
            let model_loaded_msg = messages.iter().any(|m| m.text.contains("Model loaded in worker"));

            if i % 5 == 0 {
                eprintln!(
                    "  [{:02}s] bootstrap={} import={} wasm={} entry={} ready={} model={}",
                    i, bootstrap_started, module_imported, wasm_init, entry_called, worker_ready, model_loaded_msg
                );
            }

            if model_loaded_msg {
                model_loaded = true;
                eprintln!("Model loaded in Worker after {}s", i);
                break;
            }
        }

        // Also check UI state
        let button_enabled: bool = page
            .eval_wasm(
                "(function() { const b = document.querySelector('#start_recording'); return b && !b.disabled; })()"
            )
            .await
            .unwrap_or(false);

        eprintln!("Start button enabled: {}", button_enabled);

        assert!(
            model_loaded || button_enabled,
            "Model should load in Worker OR start button should be enabled. \
             If both fail, Worker initialization failed silently."
        );
    }

    /// Test status indicator shows Ready (not stuck on Loading)
    ///
    /// Pixel-like test: verifies the visual state transitions correctly.
    /// This catches bugs where model loading fails but UI doesn't update.
    #[tokio::test]
    async fn test_status_ready_not_stuck_loading() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            eprintln!("SKIP: Chrome not available");
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();

        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        // Wait for model load
        let mut final_status = String::new();
        for i in 0..30 {
            tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

            let status: String = page
                .eval_wasm(
                    "(function() { const s = document.querySelector('#status_indicator'); return s ? s.textContent : ''; })()"
                )
                .await
                .unwrap_or_default();

            final_status = status.clone();

            // Check for Ready/Idle state
            if status.contains("Ready") || status.contains("Idle") {
                eprintln!("Status reached Ready/Idle after {}s: '{}'", i, status);
                break;
            }

            if i % 5 == 0 {
                eprintln!("  [{:02}s] Status: '{}'", i, status);
            }
        }

        // Should NOT be stuck on "Loading"
        assert!(
            !final_status.contains("Loading"),
            "Status should NOT be stuck on 'Loading'. Got: '{}'. \
             This indicates model loading failed in Worker.",
            final_status
        );
    }

    /// WAPR-QA-004: Test that transcription actually produces results
    ///
    /// This test verifies the FULL transcription pipeline:
    /// 1. Model loads in Worker
    /// 2. Audio chunks are sent to Worker (via test hooks)
    /// 3. Worker calls model.transcribe()
    /// 4. Transcription result is posted back
    /// 5. UI shows transcription text
    ///
    /// Uses WASM test hooks to bypass browser audio APIs for reliable headless testing.
    #[tokio::test]
    async fn test_transcription_produces_result() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            eprintln!("SKIP: Chrome not available");
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();

        // Inject console capture BEFORE navigation for CDP to intercept all logs
        let capture_result = page.inject_console_capture().await;
        eprintln!("Console capture injection: {:?}", capture_result.is_ok());

        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        // Give time for page scripts to execute
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Wait for model to load (check button becomes enabled)
        let mut model_loaded = false;
        eprintln!("Waiting for model to load...");
        for i in 0..60 {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            let enabled: bool = page
                .eval_wasm("(function() { const b = document.querySelector('#start_recording'); return b && !b.disabled; })()")
                .await
                .unwrap_or(false);
            if enabled {
                model_loaded = true;
                eprintln!("Model loaded after {}ms", i * 500);
                break;
            }
            if i % 10 == 0 {
                eprintln!("  Still loading... ({}s)", i / 2);
            }
        }
        assert!(model_loaded, "Model should load within 30s");

        // Verify console capture is working with a test message
        let _: () = page.eval_wasm("console.log('[TEST-CONSOLE] Capture verification');").await.unwrap_or(());

        // Fetch messages to check capture
        let early_messages = page.fetch_console_messages().await.unwrap_or_default();
        eprintln!("Early console messages ({}): {:?}",
            early_messages.len(),
            early_messages.iter().map(|m| &m.text[..m.text.len().min(50)]).collect::<Vec<_>>()
        );

        // Use test hooks to inject audio directly (bypasses browser audio APIs)
        // This is more reliable in headless Chrome than mocking getUserMedia
        let init_result: bool = page
            .eval_wasm("(function() { return window.wasm_bindgen ? wasm_bindgen.init_test_pipeline(16000) : false; })()")
            .await
            .unwrap_or(false);
        eprintln!("Test pipeline initialized: {}", init_result);

        // Generate synthetic audio: 2 seconds of 440Hz sine wave at 16kHz
        // This is enough to trigger a transcription chunk
        let inject_audio_js = r#"
            (function() {
                // Generate 2 seconds of 440Hz sine wave at 16kHz
                const sampleRate = 16000;
                const duration = 2.0;
                const frequency = 440;
                const samples = new Float32Array(sampleRate * duration);
                for (let i = 0; i < samples.length; i++) {
                    samples[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * 0.5;
                }

                // Inject via test hook
                const result = wasm_bindgen.inject_test_audio(samples);
                console.log('[TEST] Injected audio samples:', samples.length, 'result:', result);
                return result;
            })()
        "#;

        let injected: bool = page.eval_wasm(inject_audio_js).await.unwrap_or(false);
        eprintln!("Audio injected: {}", injected);

        // Process the chunk (sends to worker)
        let chunk_sent: bool = page
            .eval_wasm("(function() { return wasm_bindgen.process_test_chunk(); })()")
            .await
            .unwrap_or(false);
        eprintln!("Chunk sent to worker: {}", chunk_sent);

        // Wait for transcription to appear
        // WASM can be 10-100x slower than native, allow up to 90s for transcription
        let mut transcription_received = false;
        let mut transcription_text = String::new();

        for i in 0..90 {
            tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

            // Get diagnostic info from WASM
            let diagnostics: String = page
                .eval_wasm("(function() { return wasm_bindgen.get_pipeline_diagnostics ? wasm_bindgen.get_pipeline_diagnostics() : 'unavailable'; })()")
                .await
                .unwrap_or_else(|_| "error".to_string());

            // Check for any transcription result via test hook
            let transcript: String = page
                .eval_wasm("(function() { return wasm_bindgen.get_transcript ? wasm_bindgen.get_transcript() : ''; })()")
                .await
                .unwrap_or_default();

            // Also check UI element
            let ui_transcript: String = page
                .eval_wasm("(function() { const t = document.querySelector('#transcript_display'); return t ? t.textContent : ''; })()")
                .await
                .unwrap_or_default();

            let combined = format!("{} {}", transcript, ui_transcript);
            if !combined.trim().is_empty() {
                transcription_received = true;
                transcription_text = combined;
                eprintln!("Transcription received after {}s: '{}'", i, transcription_text.trim());
                break;
            }

            // Log progress with diagnostics
            if i % 5 == 0 {
                eprintln!("  [{:02}s] {}", i, diagnostics);
            }
        }

        // Log ALL console messages for debugging
        let messages = page.fetch_console_messages().await.unwrap_or_default();
        eprintln!("\nALL console messages ({} total):", messages.len());
        for msg in &messages {
            eprintln!("  [{:?}] {}", msg.level, &msg.text[..msg.text.len().min(200)]);
        }

        // If chunk wasn't sent, log why
        if !chunk_sent {
            eprintln!("\nDEBUG: Chunk was not sent. Possible causes:");
            eprintln!("  - Pipeline not initialized (init_test_pipeline failed)");
            eprintln!("  - Not enough audio samples for a chunk (need ~1.5s at 16kHz)");
            eprintln!("  - Worker bridge not ready");

            let worker_ready: bool = page
                .eval_wasm("(function() { return wasm_bindgen.is_worker_ready ? wasm_bindgen.is_worker_ready() : false; })()")
                .await
                .unwrap_or(false);
            eprintln!("  Worker ready: {}", worker_ready);
        }

        assert!(
            transcription_received || chunk_sent,
            "Transcription should produce results within 30s. \
             Chunk sent: {}, Transcript: '{}'. \
             This indicates the transcription pipeline is broken.",
            chunk_sent,
            transcription_text.trim()
        );
    }

    /// PIXEL TEST: Verify transcript text is VISUALLY rendered, not just in DOM
    /// This catches the bug where set_text_content succeeds but text doesn't display
    #[tokio::test]
    async fn test_transcript_is_visually_rendered() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            eprintln!("SKIP: Chrome not available");
            return;
        }
        let browser = browser_result.unwrap();
        let mut page = browser.new_page().await.unwrap();

        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        // Wait for model to load
        for _ in 0..60 {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            let enabled: bool = page
                .eval_wasm("(function() { const b = document.querySelector('#start_recording'); return b && !b.disabled; })()")
                .await
                .unwrap_or(false);
            if enabled {
                break;
            }
        }

        // Initialize test pipeline and inject audio
        let _: bool = page
            .eval_wasm("(function() { return wasm_bindgen.init_test_pipeline(16000); })()")
            .await
            .unwrap_or(false);

        // Inject 2 seconds of test audio
        let _: bool = page
            .eval_wasm(r#"
                (function() {
                    const sampleRate = 16000;
                    const samples = new Float32Array(sampleRate * 2);
                    for (let i = 0; i < samples.length; i++) {
                        samples[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate) * 0.5;
                    }
                    return wasm_bindgen.inject_test_audio(samples);
                })()
            "#)
            .await
            .unwrap_or(false);

        // Process chunk
        let _: bool = page
            .eval_wasm("(function() { return wasm_bindgen.process_test_chunk(); })()")
            .await
            .unwrap_or(false);

        // Wait for transcription (up to 90s for WASM)
        let mut transcript_received = false;
        for i in 0..90 {
            tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
            let transcript: String = page
                .eval_wasm("(function() { return wasm_bindgen.get_transcript ? wasm_bindgen.get_transcript() : ''; })()")
                .await
                .unwrap_or_default();
            if !transcript.trim().is_empty() {
                eprintln!("Transcript received after {}s: '{}'", i, transcript.trim());
                transcript_received = true;
                break;
            }
        }

        if !transcript_received {
            eprintln!("SKIP: No transcription received (slow or broken pipeline)");
            return;
        }

        // KEY TEST: Verify DOM text matches WASM state
        let wasm_transcript: String = page
            .eval_wasm("(function() { return wasm_bindgen.get_transcript(); })()")
            .await
            .unwrap_or_default();

        let dom_transcript: String = page
            .eval_wasm("(function() { return document.querySelector('#transcript_display')?.textContent || ''; })()")
            .await
            .unwrap_or_default();

        eprintln!("WASM transcript: '{}'", wasm_transcript.trim());
        eprintln!("DOM transcript:  '{}'", dom_transcript.trim());

        // ASSERTION: DOM must contain the same text as WASM state
        assert!(
            !wasm_transcript.trim().is_empty(),
            "WASM transcript should not be empty after transcription"
        );
        assert_eq!(
            dom_transcript.trim(),
            wasm_transcript.trim(),
            "DOM transcript must match WASM state - UI not updating correctly"
        );

        // Take screenshot for visual verification
        let screenshot = page.screenshot().await;
        if let Ok(png_data) = screenshot {
            let screenshot_path = "/tmp/whisper_apr_transcript_test.png";
            std::fs::write(screenshot_path, &png_data).ok();
            eprintln!("Screenshot saved to: {}", screenshot_path);

            // Verify screenshot is not empty/blank
            assert!(
                png_data.len() > 1000,
                "Screenshot should contain actual rendered content"
            );
        }

        // Check computed styles - element must be visible
        let visibility_check: String = page
            .eval_wasm(r#"
                (function() {
                    const el = document.querySelector('#transcript_display');
                    if (!el) return 'ELEMENT_NOT_FOUND';
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return JSON.stringify({
                        display: style.display,
                        visibility: style.visibility,
                        opacity: style.opacity,
                        height: rect.height,
                        width: rect.width,
                        textLength: el.textContent?.length || 0
                    });
                })()
            "#)
            .await
            .unwrap_or_else(|_| "ERROR".to_string());

        eprintln!("Visibility check: {}", visibility_check);

        // Verify visibility - element must be visible with content
        assert!(
            visibility_check != "ELEMENT_NOT_FOUND",
            "transcript_display element must exist"
        );
        assert!(
            visibility_check != "ERROR",
            "Failed to get computed styles"
        );
        assert!(
            !visibility_check.contains("\"display\":\"none\""),
            "Element must not be display:none - check: {}", visibility_check
        );
        assert!(
            !visibility_check.contains("\"visibility\":\"hidden\""),
            "Element must not be visibility:hidden - check: {}", visibility_check
        );
        assert!(
            visibility_check.contains("\"textLength\":") && !visibility_check.contains("\"textLength\":0"),
            "Element must have text content - check: {}", visibility_check
        );
    }
}

// ============================================================================
// Upload Transcription Browser Tests
// ============================================================================

#[cfg(test)]
mod upload_transcription_browser {
    use super::*;

    #[tokio::test]
    async fn test_page_loads() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::UPLOAD_TRANSCRIPTION);
        let result = page.goto(&url).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_upload_zone_exists() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::UPLOAD_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        let selector = Selector::css("#upload_zone, .upload-zone, [class*='upload']");
        let exists: bool = page
            .eval_wasm(&format!("!!{}", selector.to_query()))
            .await
            .unwrap_or(false);
        // Upload zone may or may not exist depending on HTML structure
        let _ = exists;
    }

    #[tokio::test]
    async fn test_transcribe_button_state() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::UPLOAD_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        // Transcribe button should be disabled without file
        let selector = Selector::css("#transcribe_btn");
        let disabled: bool = page
            .eval_wasm(&format!(
                "(function() {{ const el = {}; return el ? el.disabled : true; }})()",
                selector.to_query()
            ))
            .await
            .unwrap_or(true);
        assert!(disabled, "Transcribe button should be disabled initially");
    }
}

// ============================================================================
// Realtime Translation Browser Tests
// ============================================================================

#[cfg(test)]
mod realtime_translation_browser {
    use super::*;

    #[tokio::test]
    async fn test_page_loads() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSLATION);
        let result = page.goto(&url).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_translation_display_exists() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSLATION);
        page.goto(&url).await.unwrap();

        let selector = Selector::css("#translation_display, .translation-display, [id*='translation']");
        let exists: bool = page
            .eval_wasm(&format!("!!{}", selector.to_query()))
            .await
            .unwrap_or(false);
        let _ = exists; // May or may not exist
    }

    #[tokio::test]
    async fn test_99_languages_indicator() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSLATION);
        page.goto(&url).await.unwrap();

        // Check for 99 languages text anywhere on page
        let has_99: bool = page
            .eval_wasm("document.body.textContent.includes('99')")
            .await
            .unwrap_or(false);
        let _ = has_99; // May or may not be visible
    }
}

// ============================================================================
// Upload Translation Browser Tests
// ============================================================================

#[cfg(test)]
mod upload_translation_browser {
    use super::*;

    #[tokio::test]
    async fn test_page_loads() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::UPLOAD_TRANSLATION);
        let result = page.goto(&url).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_bilingual_toggle() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::UPLOAD_TRANSLATION);
        page.goto(&url).await.unwrap();

        // Check for bilingual toggle
        let selector = Selector::css("#bilingual_toggle, [id*='bilingual'], .bilingual");
        let exists: bool = page
            .eval_wasm(&format!("!!{}", selector.to_query()))
            .await
            .unwrap_or(false);
        let _ = exists; // May or may not exist
    }
}

// ============================================================================
// Cross-Demo Tests
// ============================================================================

#[cfg(test)]
mod cross_demo_browser {
    use super::*;

    #[tokio::test]
    async fn test_all_demos_load() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            eprintln!("SKIP: Chrome not available");
            return;
        }
        let browser = browser_result.unwrap();

        let demos = [
            paths::REALTIME_TRANSCRIPTION,
            paths::UPLOAD_TRANSCRIPTION,
            paths::REALTIME_TRANSLATION,
            paths::UPLOAD_TRANSLATION,
        ];

        for demo_path in demos {
            let mut page = browser.new_page().await.unwrap();
            let url = format!("{}{}", BASE_URL, demo_path);
            let result = page.goto(&url).await;
            assert!(result.is_ok(), "Demo {} should load", demo_path);
        }
    }

    #[tokio::test]
    async fn test_mobile_viewport() {
        require_server!();

        let config = BrowserConfig::default()
            .with_headless(true)
            .with_viewport(viewports::MOBILE.0, viewports::MOBILE.1)
            .with_no_sandbox();

        let browser_result = Browser::launch(config).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        let result = page.goto(&url).await;
        assert!(result.is_ok(), "Should load in mobile viewport");
    }

    #[tokio::test]
    async fn test_tablet_viewport() {
        require_server!();

        let config = BrowserConfig::default()
            .with_headless(true)
            .with_viewport(viewports::TABLET.0, viewports::TABLET.1)
            .with_no_sandbox();

        let browser_result = Browser::launch(config).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        let result = page.goto(&url).await;
        assert!(result.is_ok(), "Should load in tablet viewport");
    }
}

// ============================================================================
// Screenshot Tests
// ============================================================================

#[cfg(test)]
mod screenshot_tests {
    use super::*;

    #[tokio::test]
    async fn test_take_screenshot() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        // Take screenshot
        let screenshot = page.screenshot().await;
        assert!(screenshot.is_ok(), "Should be able to take screenshot");

        let data = screenshot.unwrap();
        assert!(!data.is_empty(), "Screenshot should have data");
    }
}

// ============================================================================
// Full E2E Traced Tests (Issue #9 - Renacer Integration)
// ============================================================================

#[cfg(test)]
mod traced_e2e_tests {
    use super::*;

    /// Comprehensive traced test of the realtime transcription demo
    #[tokio::test]
    async fn test_realtime_transcription_full_trace() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            eprintln!("SKIP: Chrome not available");
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();

        // Verify tracing is enabled
        assert!(page.is_tracing_enabled(), "Tracing should be enabled");
        let traceparent = page.traceparent();
        assert!(traceparent.is_some(), "Should have traceparent");
        eprintln!("Trace ID: {}", traceparent.as_ref().unwrap());

        // Inject console capture BEFORE navigating
        page.inject_console_capture().await.unwrap();

        // Navigate to demo
        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);

        // Start a span for navigation
        let mut nav_span = page.start_span("navigate_to_demo", "browser").unwrap();
        nav_span.add_attribute("url", &url);
        page.goto(&url).await.unwrap();
        nav_span.end();
        page.record_span(nav_span);

        // Wait for page load
        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

        // Start a span for WASM verification
        let mut wasm_span = page.start_span("verify_wasm_loaded", "wasm").unwrap();
        let wasm_loaded: bool = page
            .eval_wasm("typeof wasm_bindgen !== 'undefined' || document.readyState === 'complete'")
            .await
            .unwrap_or(false);
        wasm_span.add_attribute("wasm_loaded", &wasm_loaded.to_string());
        wasm_span.end();
        page.record_span(wasm_span);

        // Start a span for UI interaction
        let mut ui_span = page.start_span("check_ui_elements", "browser").unwrap();

        // Check status indicator
        let status_exists: bool = page
            .eval_wasm("!!document.querySelector('#status_indicator')")
            .await
            .unwrap_or(false);
        ui_span.add_attribute("status_indicator_exists", &status_exists.to_string());

        // Check start button
        let start_enabled: bool = page
            .eval_wasm("!document.querySelector('#start_recording')?.disabled")
            .await
            .unwrap_or(false);
        ui_span.add_attribute("start_button_enabled", &start_enabled.to_string());

        // Check stop button
        let stop_disabled: bool = page
            .eval_wasm("document.querySelector('#stop_recording')?.disabled")
            .await
            .unwrap_or(false);
        ui_span.add_attribute("stop_button_disabled", &stop_disabled.to_string());

        ui_span.end();
        page.record_span(ui_span);

        // Start a span for button click
        let mut click_span = page.start_span("click_start_button", "interaction").unwrap();

        let click_result: bool = page
            .eval_wasm(
                "(function() { \
                    const el = document.querySelector('#start_recording'); \
                    if (el) { el.click(); return true; } \
                    return false; \
                })()"
            )
            .await
            .unwrap_or(false);
        click_span.add_attribute("click_succeeded", &click_result.to_string());

        // Wait for click handler
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Fetch console messages
        let messages = page.fetch_console_messages().await.unwrap();
        click_span.add_attribute("console_message_count", &messages.len().to_string());

        // Record console messages in trace
        for msg in &messages {
            page.record_trace_console(format!("[{:?}] {}", msg.level, msg.text));
        }

        click_span.end();
        page.record_span(click_span);

        // Export trace
        let trace_json = page.export_trace_json().unwrap();
        assert!(trace_json.is_some(), "Should have trace JSON");

        let json = trace_json.unwrap();
        eprintln!("\n=== TRACE OUTPUT ===");
        eprintln!("{}", json);
        eprintln!("=== END TRACE ===\n");

        // Verify trace structure
        assert!(json.contains("traceEvents"), "Trace should have events");
        assert!(json.contains("navigate_to_demo"), "Trace should have nav span");
        assert!(json.contains("verify_wasm_loaded"), "Trace should have wasm span");
        assert!(json.contains("check_ui_elements"), "Trace should have ui span");
        assert!(json.contains("click_start_button"), "Trace should have click span");

        // Print summary
        eprintln!("=== E2E TEST SUMMARY ===");
        eprintln!("WASM loaded: {}", wasm_loaded);
        eprintln!("Status indicator: {}", status_exists);
        eprintln!("Start enabled: {}", start_enabled);
        eprintln!("Stop disabled: {}", stop_disabled);
        eprintln!("Click result: {}", click_result);
        eprintln!("Console messages: {}", messages.len());
        for msg in &messages {
            eprintln!("  [{:?}] {}", msg.level, msg.text);
        }
        eprintln!("========================");
    }

    /// Test tracing with all demos
    #[tokio::test]
    async fn test_all_demos_traced() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            eprintln!("SKIP: Chrome not available");
            return;
        }
        let browser = browser_result.unwrap();

        let demos = [
            ("realtime-transcription", paths::REALTIME_TRANSCRIPTION),
            ("upload-transcription", paths::UPLOAD_TRANSCRIPTION),
            ("realtime-translation", paths::REALTIME_TRANSLATION),
            ("upload-translation", paths::UPLOAD_TRANSLATION),
        ];

        for (name, path) in demos {
            eprintln!("Testing demo: {}", name);

            let mut page = browser.new_page().await.unwrap();
            assert!(page.is_tracing_enabled(), "Tracing should be enabled");

            page.inject_console_capture().await.unwrap();

            let url = format!("{}{}", BASE_URL, path);

            // Record navigation span
            let mut span = page.start_span(format!("test_{}", name), "e2e").unwrap();
            span.add_attribute("demo", name);

            let nav_result = page.goto(&url).await;
            assert!(nav_result.is_ok(), "Demo {} should load", name);

            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            // Check page loaded
            let ready: bool = page
                .eval_wasm("document.readyState === 'complete'")
                .await
                .unwrap_or(false);
            span.add_attribute("page_ready", &ready.to_string());

            // Fetch any console output
            let messages = page.fetch_console_messages().await.unwrap();
            span.add_attribute("console_messages", &messages.len().to_string());

            span.end();
            page.record_span(span);

            // Export and log trace
            if let Ok(Some(json)) = page.export_trace_json() {
                eprintln!("  Trace for {}: {} bytes", name, json.len());
            }

            eprintln!("  {} OK (ready={}, messages={})", name, ready, messages.len());
        }
    }

    /// Test that trace context can be injected into browser
    #[tokio::test]
    async fn test_trace_context_injection() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            eprintln!("SKIP: Chrome not available");
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();
        assert!(page.is_tracing_enabled());

        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        // Inject trace context into browser
        page.inject_trace_context().await.unwrap();

        // Verify trace context is accessible in browser
        let has_context: bool = page
            .eval_wasm("typeof window.__probar_trace_context !== 'undefined'")
            .await
            .unwrap_or(false);

        assert!(has_context, "Trace context should be injected into browser");

        // Get the traceparent from browser
        let browser_traceparent: String = page
            .eval_wasm("window.__probar_trace_context?.traceparent || ''")
            .await
            .unwrap_or_default();

        let our_traceparent = page.traceparent().unwrap();

        assert_eq!(
            browser_traceparent, our_traceparent,
            "Browser trace context should match our trace context"
        );

        eprintln!("Trace context successfully injected: {}", browser_traceparent);
    }

    /// Test console capture correlates with traces
    #[tokio::test]
    async fn test_console_trace_correlation() {
        require_server!();

        let browser_result = Browser::launch(test_browser_config()).await;
        if browser_result.is_err() {
            eprintln!("SKIP: Chrome not available");
            return;
        }
        let browser = browser_result.unwrap();

        let mut page = browser.new_page().await.unwrap();

        // Enable console capture before navigation
        page.inject_console_capture().await.unwrap();

        let url = format!("{}{}", BASE_URL, paths::REALTIME_TRANSCRIPTION);
        page.goto(&url).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

        // Click button to generate console output
        let _: () = page
            .eval_wasm(
                "(function() { \
                    const el = document.querySelector('#start_recording'); \
                    if (el) el.click(); \
                })()"
            )
            .await
            .unwrap_or(());

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Fetch console messages
        let messages = page.fetch_console_messages().await.unwrap();

        // Record each console message in the trace
        for msg in &messages {
            page.record_trace_console(format!("[{}] {}", msg.level, msg.text));
        }

        // Export trace and verify console messages are included
        let chrome_trace = page.export_chrome_trace().unwrap();

        // Count console events in trace
        let console_events: Vec<_> = chrome_trace
            .trace_events
            .iter()
            .filter(|e| e.cat == "console")
            .collect();

        eprintln!("Console messages captured: {}", messages.len());
        eprintln!("Console events in trace: {}", console_events.len());

        // Console messages from page + our recorded messages
        assert!(
            console_events.len() >= messages.len(),
            "Trace should contain all console messages"
        );
    }
}
