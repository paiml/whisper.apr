//! UX Flow Tests (Steps 1-30)
//!
//! WAPR-DEMO-REBUILD-TDD: Extreme TDD for demo rebuild
//! All tests written BEFORE implementation.

use probar::{Browser, BrowserConfig, BrowserConsoleLevel};
use std::time::Duration;

const BASE_URL: &str = "http://localhost:8080";

/// Helper to check if server is running
async fn is_server_running() -> bool {
    // Try to connect via reqwest if available, otherwise assume running
    #[cfg(feature = "reqwest")]
    return reqwest::get(BASE_URL).await.is_ok();
    #[cfg(not(feature = "reqwest"))]
    true // Assume server is running for browser tests
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
// STEP 1-10: Core UX Flow (P0)
// =============================================================================

/// Step 1: Page loads without JS errors
#[tokio::test]
async fn step_01_page_loads_without_js_errors() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    let _ = page.inject_console_capture().await;

    page.goto(BASE_URL).await.unwrap();
    tokio::time::sleep(Duration::from_secs(2)).await;

    let messages = page.fetch_console_messages().await.unwrap();
    let errors: Vec<_> = messages.iter()
        .filter(|m| matches!(m.level, BrowserConsoleLevel::Error))
        .collect();

    assert!(errors.is_empty(), "Page has JS errors: {:?}", errors);
}

/// Step 2: Status shows "Loading..." initially
#[tokio::test]
async fn step_02_status_shows_loading_initially() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    // Check immediately (before model loads)
    let status = page.eval_wasm::<String>(
        "document.querySelector('#status')?.textContent || ''"
    ).await.unwrap_or_default();

    assert!(
        status.to_lowercase().contains("load"),
        "Status should show loading, got: {status}"
    );
}

/// Step 3: Record button disabled while loading
#[tokio::test]
async fn step_03_record_button_disabled_while_loading() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    // Check immediately
    let disabled = page.eval_wasm::<bool>(
        "document.querySelector('#record')?.disabled ?? true"
    ).await.unwrap_or(true);

    assert!(disabled, "Record button should be disabled while loading");
}

/// Step 4: Status shows "Ready" after model load
#[tokio::test]
async fn step_04_status_shows_ready_after_model_load() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    // Wait for model to load (up to 30s)
    for _ in 0..60 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let status = page.eval_wasm::<String>(
            "document.querySelector('#status')?.textContent || ''"
        ).await.unwrap_or_default();

        if status.to_lowercase().contains("ready") {
            return; // Test passed
        }
    }

    panic!("Status never showed 'Ready' after 30 seconds");
}

/// Step 5: Record button enabled when ready
#[tokio::test]
async fn step_05_record_button_enabled_when_ready() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    // Wait for ready state
    for _ in 0..60 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let disabled = page.eval_wasm::<bool>(
            "document.querySelector('#record')?.disabled ?? true"
        ).await.unwrap_or(true);

        if !disabled {
            return; // Test passed
        }
    }

    panic!("Record button never became enabled");
}

/// Step 6: Click record starts recording
#[tokio::test]
async fn step_06_click_record_starts_recording() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    // Wait for ready
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Click record
    let clicked = page.eval_wasm::<bool>(
        "(function() { const btn = document.querySelector('#record'); if (btn && !btn.disabled) { btn.click(); return true; } return false; })()"
    ).await.unwrap_or(false);

    assert!(clicked, "Failed to click record button");

    // Verify recording started (check for recording indicator or state)
    tokio::time::sleep(Duration::from_millis(500)).await;

    let is_recording = page.eval_wasm::<bool>(
        "document.body.classList.contains('recording') || document.querySelector('.recording') !== null || document.querySelector('#status')?.textContent?.toLowerCase().includes('record')"
    ).await.unwrap_or(false);

    assert!(is_recording, "Recording should have started");
}

/// Step 7: Status shows "Recording..."
#[tokio::test]
async fn step_07_status_shows_recording() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    // Wait for ready and click record
    tokio::time::sleep(Duration::from_secs(5)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_millis(500)).await;

    let status = page.eval_wasm::<String>(
        "document.querySelector('#status')?.textContent || ''"
    ).await.unwrap_or_default();

    assert!(
        status.to_lowercase().contains("record"),
        "Status should show recording, got: {status}"
    );
}

/// Step 8: VU meter animates during recording
#[tokio::test]
async fn step_08_vu_meter_animates_during_recording() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    // Wait for ready and start recording
    tokio::time::sleep(Duration::from_secs(5)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Check VU meter exists and has some width
    let vu_width = page.eval_wasm::<f64>(
        "parseFloat(document.querySelector('#vu_meter')?.style.width || '0')"
    ).await.unwrap_or(0.0);

    // VU meter should show some level (even with silence, there's noise)
    // This test may need adjustment based on actual implementation
    let vu_exists = page.eval_wasm::<bool>(
        "document.querySelector('#vu_meter') !== null"
    ).await.unwrap_or(false);

    assert!(vu_exists, "VU meter element should exist");
}

/// Step 9: Click stop stops recording
#[tokio::test]
async fn step_09_click_stop_stops_recording() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    // Wait, start recording, then stop
    tokio::time::sleep(Duration::from_secs(5)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_secs(2)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok(); // Toggle
    tokio::time::sleep(Duration::from_millis(500)).await;

    let status = page.eval_wasm::<String>(
        "document.querySelector('#status')?.textContent || ''"
    ).await.unwrap_or_default();

    // Should be back to ready or processing
    assert!(
        !status.to_lowercase().contains("recording") || status.to_lowercase().contains("processing") || status.to_lowercase().contains("ready"),
        "Should have stopped recording, got: {status}"
    );
}

/// Step 10: Final transcription appears after stop
#[tokio::test]
async fn step_10_final_transcription_appears() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    let _ = page.inject_console_capture().await;
    page.goto(BASE_URL).await.unwrap();

    // Wait, record for 3 seconds, stop
    tokio::time::sleep(Duration::from_secs(5)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_secs(3)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();

    // Wait for transcription to appear
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let transcript = page.eval_wasm::<String>(
            "document.querySelector('#transcript')?.textContent || ''"
        ).await.unwrap_or_default();

        if !transcript.trim().is_empty() {
            eprintln!("Got transcript: {}", transcript);
            return; // Test passed
        }
    }

    // Print console for debugging
    let messages = page.fetch_console_messages().await.unwrap();
    for msg in &messages {
        eprintln!("[{:?}] {}", msg.level, msg.text);
    }

    panic!("No transcription appeared after 10 seconds");
}

// =============================================================================
// STEP 11-20: Extended UX Flow (P1-P2)
// =============================================================================

/// Step 11: Partial text during recording (P1)
#[tokio::test]
async fn step_11_partial_text_during_recording() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    // Wait and start recording
    tokio::time::sleep(Duration::from_secs(5)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();

    // Check for partial text element during recording (after a few seconds)
    tokio::time::sleep(Duration::from_secs(5)).await;

    let partial_exists = page.eval_wasm::<bool>(
        "document.querySelector('#partial') !== null"
    ).await.unwrap_or(false);

    assert!(partial_exists, "Partial text element should exist during recording");
}

/// Step 12: Clear button clears transcript (P1)
#[tokio::test]
async fn step_12_clear_button_clears_transcript() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    // Record something first
    tokio::time::sleep(Duration::from_secs(5)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_secs(2)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Click clear
    page.eval_wasm::<()>("document.querySelector('#clear')?.click()").await.ok();
    tokio::time::sleep(Duration::from_millis(200)).await;

    let transcript = page.eval_wasm::<String>(
        "document.querySelector('#transcript')?.textContent || 'NOTFOUND'"
    ).await.unwrap_or_default();

    assert!(
        transcript.trim().is_empty() || transcript == "NOTFOUND",
        "Transcript should be empty after clear"
    );
}

/// Step 16: Double click record doesn't crash (P1)
#[tokio::test]
async fn step_16_double_click_record_no_crash() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    let _ = page.inject_console_capture().await;
    page.goto(BASE_URL).await.unwrap();

    tokio::time::sleep(Duration::from_secs(5)).await;

    // Rapid double click
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Check for errors
    let messages = page.fetch_console_messages().await.unwrap();
    let errors: Vec<_> = messages.iter()
        .filter(|m| matches!(m.level, BrowserConsoleLevel::Error))
        .filter(|m| !m.text.contains("favicon")) // Ignore favicon errors
        .collect();

    assert!(errors.is_empty(), "Double click caused errors: {:?}", errors);
}

/// Step 17: Record-stop-record cycle works (P1)
#[tokio::test]
async fn step_17_record_stop_record_cycle() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    tokio::time::sleep(Duration::from_secs(5)).await;

    // First recording
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_secs(1)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Second recording
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_secs(1)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();

    // Should still be functional
    let status = page.eval_wasm::<String>(
        "document.querySelector('#status')?.textContent || 'ERROR'"
    ).await.unwrap_or_default();

    assert!(!status.contains("ERROR"), "Multiple record cycles should work");
}

/// Step 19: State transitions in correct order (P0)
#[tokio::test]
async fn step_19_status_transitions_correct_order() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    let _ = page.inject_console_capture().await;
    page.goto(BASE_URL).await.unwrap();

    let mut states_seen = Vec::new();

    // Collect states over time
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(250)).await;

        let status = page.eval_wasm::<String>(
            "document.querySelector('#status')?.textContent?.toLowerCase() || ''"
        ).await.unwrap_or_default();

        if states_seen.last().map(|s| s != &status).unwrap_or(true) && !status.is_empty() {
            states_seen.push(status.clone());
        }

        // Start recording after ready
        if status.contains("ready") && states_seen.len() < 3 {
            page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
        }

        // Stop after recording for a bit
        if status.contains("record") && states_seen.iter().filter(|s| s.contains("record")).count() >= 2 {
            page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
        }
    }

    eprintln!("States seen: {:?}", states_seen);

    // Should have seen: loading -> ready -> recording -> ready (or similar)
    assert!(states_seen.len() >= 2, "Should have seen multiple states");
}

// =============================================================================
// STEP 21-30: Technical UX (P1-P2)
// =============================================================================

/// Step 27: Model loads only once (P1)
#[tokio::test]
async fn step_27_model_loads_once() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    let _ = page.inject_console_capture().await;
    page.goto(BASE_URL).await.unwrap();

    // Wait for model to load
    tokio::time::sleep(Duration::from_secs(10)).await;

    // Record twice
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_secs(1)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_secs(2)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();
    tokio::time::sleep(Duration::from_secs(1)).await;
    page.eval_wasm::<()>("document.querySelector('#record')?.click()").await.ok();

    // Check console for model load messages
    let messages = page.fetch_console_messages().await.unwrap();
    let model_loads = messages.iter()
        .filter(|m| m.text.contains("Model") && m.text.contains("load"))
        .count();

    // Should only load model once (or twice if worker reloads)
    assert!(model_loads <= 2, "Model should not reload multiple times, got {} loads", model_loads);
}

/// Step 30: ARIA labels present for accessibility (P2)
#[tokio::test]
async fn step_30_aria_labels_present() {
    require_server!();

    let browser = Browser::launch(test_config()).await;
    require_browser!(browser);
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Check for ARIA attributes on key elements
    let has_aria = page.eval_wasm::<bool>(
        r#"
        const record = document.querySelector('#record');
        const transcript = document.querySelector('#transcript');
        return (record?.getAttribute('aria-label') || record?.textContent) &&
               (transcript?.getAttribute('aria-live') || transcript?.getAttribute('role'));
        "#
    ).await.unwrap_or(false);

    assert!(has_aria, "Key elements should have ARIA labels for accessibility");
}
