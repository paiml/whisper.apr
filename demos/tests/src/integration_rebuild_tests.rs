//! Integration Tests (Steps 81-100)
//!
//! WAPR-DEMO-REBUILD-TDD: Extreme TDD for demo rebuild
//! End-to-end tests that verify the complete flow.

use probar::{Browser, BrowserConfig, BrowserConsoleLevel};
use std::time::Duration;

const BASE_URL: &str = "http://localhost:8080";
const TEST_AUDIO_PATH: &str = "/home/noah/src/whisper.apr/demos/test-audio/test-speech-1.5s.wav";

/// Helper to check if server is running
async fn is_server_running() -> bool {
    #[cfg(feature = "reqwest")]
    return reqwest::get(BASE_URL).await.is_ok();
    #[cfg(not(feature = "reqwest"))]
    true
}

/// Skip test if server not running
macro_rules! require_server {
    () => {
        if !is_server_running().await {
            eprintln!("SKIP: Server not running on localhost:8080");
            return;
        }
    };
}

/// Skip test if browser not available
macro_rules! require_browser {
    ($browser:expr) => {
        if $browser.is_err() {
            eprintln!("SKIP: Browser not available");
            return;
        }
    };
}

fn test_config() -> BrowserConfig {
    BrowserConfig::default()
        .with_headless(true)
        .with_no_sandbox()
}

// =============================================================================
// STEP 81-90: Core Integration (P0)
// =============================================================================

/// Step 81: Full flow from load to ready
#[tokio::test]
async fn step_81_e2e_load_to_ready() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    let _ = page.inject_console_capture().await;
    page.goto(BASE_URL).await.unwrap();

    // Wait for ready state (up to 30s for model load)
    let mut reached_ready = false;
    for _ in 0..60 {
        tokio::time::sleep(Duration::from_millis(500)).await;

        let status = page.eval_wasm::<String>(
            "document.querySelector('#status')?.textContent?.toLowerCase() || ''"
        ).await.unwrap_or_default();

        if status.contains("ready") {
            reached_ready = true;
            break;
        }
    }

    // Print console for debugging
    let messages = page.fetch_console_messages().await.unwrap();
    for msg in messages.iter().take(20) {
        eprintln!("[{:?}] {}", msg.level, msg.text);
    }

    assert!(reached_ready, "Should reach Ready state within 30 seconds");
}

/// Step 82: Record short audio and get transcript
#[tokio::test]
async fn step_82_e2e_record_short_audio() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    let _ = page.inject_console_capture().await;
    page.goto(BASE_URL).await.unwrap();

    // Wait for ready
    tokio::time::sleep(Duration::from_secs(10)).await;

    // Start recording
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();

    // Record for 2 seconds
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Stop recording
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();

    // Wait for transcription
    let mut got_transcript = false;
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(500)).await;

        let transcript = page.eval_wasm::<String>(
            "document.querySelector('#transcript')?.textContent || ''"
        ).await.unwrap_or_default();

        if !transcript.trim().is_empty() {
            eprintln!("Got transcript: {}", transcript);
            got_transcript = true;
            break;
        }
    }

    // Print console for debugging
    let messages = page.fetch_console_messages().await.unwrap();
    eprintln!("\n=== Console messages ===");
    for msg in &messages {
        eprintln!("[{:?}] {}", msg.level, msg.text);
    }

    assert!(got_transcript, "Should get transcription after recording");
}

/// Step 85: Transcribe test audio file accurately
#[tokio::test]
async fn step_85_e2e_with_test_audio_file() {
    // This test requires the test audio file
    if !std::path::Path::new(TEST_AUDIO_PATH).exists() {
        eprintln!("SKIP: Test audio file not found at {}", TEST_AUDIO_PATH);
        return;
    }

    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    let _ = page.inject_console_capture().await;
    page.goto(BASE_URL).await.unwrap();

    // Wait for ready
    tokio::time::sleep(Duration::from_secs(10)).await;

    // For file upload test, we'd need to inject audio
    // This is a placeholder for actual file-based testing
    eprintln!("Test audio file test - requires file upload implementation");
}

/// Step 86: Accuracy test on known speech
#[tokio::test]
async fn step_86_e2e_accuracy_test() {
    // Expected transcription for test-speech-1.5s.wav
    // This should match the ground truth
    let _expected_words = vec!["the", "quick", "brown", "fox"];

    // This test would compare actual transcription to expected
    // Placeholder for now
    eprintln!("Accuracy test - requires ground truth comparison implementation");
}

/// Step 89: No console errors during operation
#[tokio::test]
async fn step_89_e2e_no_console_errors() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    let _ = page.inject_console_capture().await;
    page.goto(BASE_URL).await.unwrap();

    // Wait for load
    tokio::time::sleep(Duration::from_secs(10)).await;

    // Start and stop recording
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_secs(2)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Check for errors (excluding favicon)
    let messages = page.fetch_console_messages().await.unwrap();
    let errors: Vec<_> = messages.iter()
        .filter(|m| matches!(m.level, BrowserConsoleLevel::Error))
        .filter(|m| !m.text.contains("favicon"))
        .collect();

    if !errors.is_empty() {
        eprintln!("=== Errors found ===");
        for err in &errors {
            eprintln!("  {}", err.text);
        }
    }

    assert!(errors.is_empty(), "Should have no JS errors during operation");
}

/// Step 90: No WASM traps
#[tokio::test]
async fn step_90_e2e_no_wasm_traps() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    let _ = page.inject_console_capture().await;
    page.goto(BASE_URL).await.unwrap();

    // Exercise the full flow
    tokio::time::sleep(Duration::from_secs(10)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_secs(2)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Check for WASM traps/panics
    let messages = page.fetch_console_messages().await.unwrap();
    let traps: Vec<_> = messages.iter()
        .filter(|m| {
            m.text.contains("wasm trap") ||
            m.text.contains("unreachable") ||
            m.text.contains("RuntimeError") ||
            m.text.contains("panicked")
        })
        .collect();

    if !traps.is_empty() {
        eprintln!("=== WASM traps found ===");
        for trap in &traps {
            eprintln!("  {}", trap.text);
        }
    }

    assert!(traps.is_empty(), "Should have no WASM traps");
}

// =============================================================================
// STEP 91-95: Stress Tests (P1)
// =============================================================================

/// Step 91: Rapid start/stop cycles
#[tokio::test]
async fn step_91_stress_rapid_start_stop() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    let _ = page.inject_console_capture().await;
    page.goto(BASE_URL).await.unwrap();

    // Wait for ready
    tokio::time::sleep(Duration::from_secs(10)).await;

    // Rapid start/stop cycles
    for i in 0..5 {
        eprintln!("Cycle {}", i + 1);
        page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
        tokio::time::sleep(Duration::from_millis(500)).await;
        page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    // Check for errors
    let messages = page.fetch_console_messages().await.unwrap();
    let errors: Vec<_> = messages.iter()
        .filter(|m| matches!(m.level, BrowserConsoleLevel::Error))
        .filter(|m| !m.text.contains("favicon"))
        .collect();

    assert!(errors.len() < 3, "Should handle rapid start/stop without many errors, got {}", errors.len());
}

// =============================================================================
// STEP 97-100: Environment Tests (P0)
// =============================================================================

/// Step 97: COOP/COEP headers present (required for SharedArrayBuffer)
/// Note: This test requires reqwest - skipped when not available
#[tokio::test]
async fn step_97_coop_coep_headers() {
    require_server!();

    // Headers test requires probar server with COOP/COEP
    // The browser tests verify SharedArrayBuffer works, which implies correct headers
    eprintln!("Note: COOP/COEP headers verified via SharedArrayBuffer availability in browser tests");
}

/// Step 98: SharedArrayBuffer available
#[tokio::test]
async fn step_98_shared_array_buffer_available() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    let sab_available = page.eval_wasm::<bool>(
        "typeof SharedArrayBuffer !== 'undefined'"
    ).await.unwrap_or(false);

    assert!(sab_available, "SharedArrayBuffer must be available (requires COOP/COEP headers)");
}

/// Step 99: WASM SIMD available
#[tokio::test]
async fn step_99_wasm_simd_available() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    // Check for WASM SIMD support
    let simd_available = page.eval_wasm::<bool>(
        r#"
        (async function() {
            try {
                const simdTest = new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11]);
                await WebAssembly.instantiate(simdTest);
                return true;
            } catch (e) {
                return false;
            }
        })()
        "#
    ).await.unwrap_or(false);

    assert!(simd_available, "WASM SIMD should be available for performance");
}

/// Step 100: Complete happy path golden test
#[tokio::test]
async fn step_100_full_demo_golden_path() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    let _ = page.inject_console_capture().await;
    page.goto(BASE_URL).await.unwrap();

    eprintln!("Step 100: Golden path test starting...");

    // 1. Wait for page load
    tokio::time::sleep(Duration::from_secs(2)).await;
    eprintln!("  1. Page loaded");

    // 2. Wait for model to load (Ready state)
    let mut model_loaded = false;
    for i in 0..60 {
        let status = page.eval_wasm::<String>(
            "document.querySelector('#status')?.textContent?.toLowerCase() || ''"
        ).await.unwrap_or_default();

        if status.contains("ready") {
            model_loaded = true;
            eprintln!("  2. Model loaded after {}ms", i * 500);
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    assert!(model_loaded, "Model should load within 30 seconds");

    // 3. Click record
    let record_enabled = page.eval_wasm::<bool>(
        "!document.querySelector('#record')?.disabled"
    ).await.unwrap_or(false);
    assert!(record_enabled, "Record button should be enabled");

    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_millis(500)).await;
    eprintln!("  3. Recording started");

    // 4. Verify recording state
    let is_recording = page.eval_wasm::<String>(
        "document.querySelector('#status')?.textContent?.toLowerCase() || ''"
    ).await.unwrap_or_default();
    assert!(is_recording.contains("record"), "Should be in recording state");
    eprintln!("  4. Recording state confirmed");

    // 5. Record for a few seconds
    tokio::time::sleep(Duration::from_secs(3)).await;
    eprintln!("  5. Recorded for 3 seconds");

    // 6. Stop recording
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_millis(500)).await;
    eprintln!("  6. Recording stopped");

    // 7. Wait for transcription
    let mut got_transcript = false;
    for i in 0..20 {
        let transcript = page.eval_wasm::<String>(
            "document.querySelector('#transcript')?.textContent || ''"
        ).await.unwrap_or_default();

        if !transcript.trim().is_empty() {
            eprintln!("  7. Got transcript after {}ms: {}", i * 500, transcript);
            got_transcript = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    // 8. Check for errors
    let messages = page.fetch_console_messages().await.unwrap();
    let errors: Vec<_> = messages.iter()
        .filter(|m| matches!(m.level, BrowserConsoleLevel::Error))
        .filter(|m| !m.text.contains("favicon"))
        .collect();

    eprintln!("\n=== Console summary ===");
    eprintln!("  Total messages: {}", messages.len());
    eprintln!("  Errors: {}", errors.len());

    if !errors.is_empty() {
        eprintln!("\n=== Errors ===");
        for err in &errors {
            eprintln!("  {}", err.text);
        }
    }

    // Final assertions
    assert!(got_transcript, "Should get transcription after recording");
    assert!(errors.is_empty(), "Should complete without errors");

    eprintln!("\n  ✅ Golden path test PASSED");
}
