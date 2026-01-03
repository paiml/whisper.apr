//! WAPR-ZERO-JS-001: Zero-JavaScript Demo Tests
//!
//! TDD: These tests are written FIRST and must FAIL until implementation exists.
//!
//! The whisper.apr demo must work with ZERO inline JavaScript.
//! All logic lives in Rust via `#[wasm_bindgen(start)]`.
//!
//! Run with: `cargo test --package whisper-apr-demo-tests zero_js`

use probar::{Browser, BrowserConfig};

const BASE_URL: &str = "http://localhost:8080";

/// Browser config for pixel-perfect testing
fn pixel_test_config() -> BrowserConfig {
    BrowserConfig::default()
        .with_headless(true)
        .with_viewport(1280, 720)
        .with_no_sandbox()
}

/// Check if server is running
async fn server_available() -> bool {
    tokio::net::TcpStream::connect("127.0.0.1:8080")
        .await
        .is_ok()
}

macro_rules! require_server {
    () => {
        if !server_available().await {
            eprintln!("SKIP: Server not running on localhost:8080");
            eprintln!("      Start with: probar serve");
            return;
        }
    };
}

// ============================================================================
// PHASE 1: Page Loads (No JS Required)
// ============================================================================

#[tokio::test]
async fn test_index_page_exists() {
    require_server!();

    let browser = Browser::launch(pixel_test_config()).await;
    if browser.is_err() {
        eprintln!("SKIP: Chrome not available");
        return;
    }
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    let result = page.goto(BASE_URL).await;

    assert!(result.is_ok(), "Index page must exist at /");
}

#[tokio::test]
async fn test_page_has_no_inline_javascript() {
    require_server!();

    let browser = Browser::launch(pixel_test_config()).await;
    if browser.is_err() {
        return;
    }
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    // Check for inline script tags (should be ZERO or only module import)
    let inline_scripts: i32 = page
        .eval_wasm(
            r#"(function() {
                const scripts = document.querySelectorAll('script:not([src])');
                let inlineCount = 0;
                scripts.forEach(s => {
                    // Only allow: import init from '...'; await init();
                    const content = s.textContent.trim();
                    const isMinimalImport = /^import\s+init.*from.*;\s*await\s+init\(\);?\s*$/.test(content);
                    if (!isMinimalImport && content.length > 0) {
                        inlineCount++;
                        console.log('[ZERO-JS-VIOLATION] Inline JS:', content.substring(0, 100));
                    }
                });
                return inlineCount;
            })()"#,
        )
        .await
        .unwrap_or(999);

    assert_eq!(
        inline_scripts, 0,
        "Page must have ZERO inline JavaScript (found {})",
        inline_scripts
    );
}

#[tokio::test]
async fn test_wasm_auto_initializes() {
    require_server!();

    let browser = Browser::launch(pixel_test_config()).await;
    if browser.is_err() {
        return;
    }
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    // Wait for WASM to auto-initialize via #[wasm_bindgen(start)]
    let mut initialized = false;
    for _ in 0..20 {
        tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;

        // Check if status indicator shows "Ready" or similar
        let status: String = page
            .eval_wasm(
                "(function() {
                    const el = document.querySelector('#status');
                    return el ? el.textContent : '';
                })()",
            )
            .await
            .unwrap_or_default();

        if status.contains("Ready") || status.contains("Idle") {
            initialized = true;
            break;
        }
    }

    assert!(
        initialized,
        "WASM must auto-initialize via #[wasm_bindgen(start)]"
    );
}

// ============================================================================
// PHASE 2: UI Elements Exist (Rendered by Rust)
// ============================================================================

#[tokio::test]
async fn test_status_element_exists() {
    require_server!();

    let browser = Browser::launch(pixel_test_config()).await;
    if browser.is_err() {
        return;
    }
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let exists: bool = page
        .eval_wasm("!!document.querySelector('#status')")
        .await
        .unwrap_or(false);

    assert!(exists, "Status element #status must exist");
}

#[tokio::test]
async fn test_record_button_exists() {
    require_server!();

    let browser = Browser::launch(pixel_test_config()).await;
    if browser.is_err() {
        return;
    }
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let exists: bool = page
        .eval_wasm("!!document.querySelector('#record')")
        .await
        .unwrap_or(false);

    assert!(exists, "Record button #record must exist");
}

#[tokio::test]
async fn test_transcript_element_exists() {
    require_server!();

    let browser = Browser::launch(pixel_test_config()).await;
    if browser.is_err() {
        return;
    }
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let exists: bool = page
        .eval_wasm("!!document.querySelector('#transcript')")
        .await
        .unwrap_or(false);

    assert!(exists, "Transcript element #transcript must exist");
}

// ============================================================================
// PHASE 3: Pixel-Perfect Visual Tests
// ============================================================================

#[tokio::test]
async fn test_pixel_initial_state() {
    require_server!();

    let browser = Browser::launch(pixel_test_config()).await;
    if browser.is_err() {
        return;
    }
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    // Wait for render
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Take screenshot
    let screenshot = page.screenshot().await;
    assert!(screenshot.is_ok(), "Screenshot must succeed");

    let png_data = screenshot.unwrap();
    assert!(png_data.len() > 5000, "Screenshot must have content");

    // Save for visual inspection
    let path = "/tmp/whisper_apr_zero_js_initial.png";
    std::fs::write(path, &png_data).ok();
    eprintln!("Screenshot saved: {}", path);

    // Decode and verify reasonable dimensions
    let img = image::load_from_memory(&png_data).unwrap();
    assert!(img.width() >= 600, "Width must be at least 600px, got {}", img.width());
    assert!(img.height() >= 400, "Height must be at least 400px, got {}", img.height());

    // Verify NOT a blank white page (has actual content)
    let rgba = img.to_rgba8();
    let mut non_white_pixels = 0;
    for pixel in rgba.pixels() {
        if pixel[0] < 250 || pixel[1] < 250 || pixel[2] < 250 {
            non_white_pixels += 1;
        }
    }

    assert!(
        non_white_pixels > 10000,
        "Page must not be blank white (found {} non-white pixels)",
        non_white_pixels
    );
}

#[tokio::test]
async fn test_pixel_dark_theme() {
    require_server!();

    let browser = Browser::launch(pixel_test_config()).await;
    if browser.is_err() {
        return;
    }
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Get background color
    let bg_color: String = page
        .eval_wasm(
            "(function() {
                const style = window.getComputedStyle(document.body);
                return style.backgroundColor;
            })()",
        )
        .await
        .unwrap_or_default();

    eprintln!("Background color: {}", bg_color);

    // Should be dark theme (rgb values < 50)
    // Expected: rgb(13, 17, 23) or similar dark color
    assert!(
        bg_color.contains("rgb") && !bg_color.contains("255"),
        "Page must use dark theme, got: {}",
        bg_color
    );
}

// ============================================================================
// PHASE 4: Transcription Flow (E2E)
// ============================================================================

#[tokio::test]
async fn test_transcription_produces_output() {
    require_server!();

    let browser = Browser::launch(pixel_test_config()).await;
    if browser.is_err() {
        return;
    }
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    // Wait for model to load
    let mut ready = false;
    for _ in 0..60 {
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        let status: String = page
            .eval_wasm("document.querySelector('#status')?.textContent || ''")
            .await
            .unwrap_or_default();

        if status.contains("Ready") {
            ready = true;
            break;
        }
    }

    assert!(ready, "Model must load within 30s");

    // Inject test audio via WASM test hook
    let injected: bool = page
        .eval_wasm(
            r#"(function() {
                if (!window.whisper_apr) return false;

                // Generate 2s of 440Hz sine wave at 16kHz
                const samples = new Float32Array(32000);
                for (let i = 0; i < samples.length; i++) {
                    samples[i] = Math.sin(2 * Math.PI * 440 * i / 16000) * 0.5;
                }

                return window.whisper_apr.inject_test_audio(samples);
            })()"#,
        )
        .await
        .unwrap_or(false);

    assert!(injected, "Test audio must be injectable");

    // Process and wait for transcription
    let _: bool = page
        .eval_wasm("window.whisper_apr?.process_test_chunk() || false")
        .await
        .unwrap_or(false);

    // Wait for transcription result
    let mut has_transcript = false;
    for _ in 0..60 {
        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

        let transcript: String = page
            .eval_wasm("document.querySelector('#transcript')?.textContent || ''")
            .await
            .unwrap_or_default();

        if !transcript.trim().is_empty() {
            has_transcript = true;
            eprintln!("Transcript: '{}'", transcript.trim());
            break;
        }
    }

    assert!(has_transcript, "Transcription must produce output within 60s");
}

// ============================================================================
// PHASE 5: Accessibility
// ============================================================================

#[tokio::test]
async fn test_has_proper_aria_labels() {
    require_server!();

    let browser = Browser::launch(pixel_test_config()).await;
    if browser.is_err() {
        return;
    }
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Check record button has aria-label
    let has_aria: bool = page
        .eval_wasm(
            "(function() {
                const btn = document.querySelector('#record');
                return btn && btn.hasAttribute('aria-label');
            })()",
        )
        .await
        .unwrap_or(false);

    assert!(has_aria, "Record button must have aria-label");
}

#[tokio::test]
async fn test_transcript_has_live_region() {
    require_server!();

    let browser = Browser::launch(pixel_test_config()).await;
    if browser.is_err() {
        return;
    }
    let browser = browser.unwrap();

    let mut page = browser.new_page().await.unwrap();
    page.goto(BASE_URL).await.unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Check transcript has aria-live for screen readers
    let has_live: bool = page
        .eval_wasm(
            "(function() {
                const el = document.querySelector('#transcript');
                return el && el.getAttribute('aria-live') === 'polite';
            })()",
        )
        .await
        .unwrap_or(false);

    assert!(has_live, "Transcript must have aria-live='polite'");
}
