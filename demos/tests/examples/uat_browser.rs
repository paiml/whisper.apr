//! User Acceptance Testing - Foreground Browser Mode
//!
//! Run with: cargo run --example uat_browser -p whisper-apr-demo-tests
//!
//! This opens a visible browser window for manual UAT testing.

use probar::{Browser, BrowserConfig, Page, Selector};
use std::time::Duration;

const BASE_URL: &str = "http://localhost:8090";

/// Check if demo server is running
async fn check_server() -> bool {
    if tokio::net::TcpStream::connect("127.0.0.1:8090")
        .await
        .is_err()
    {
        eprintln!("ERROR: Demo server not running!");
        eprintln!("Start with: python3 -m http.server 8090 --directory www");
        return false;
    }
    println!("Server running on localhost:8090");
    true
}

/// Launch browser in foreground mode
async fn launch_browser() -> Option<Browser> {
    let config = BrowserConfig::default()
        .with_headless(false)
        .with_viewport(1280, 800)
        .with_no_sandbox();

    println!("Launching Chrome in foreground mode...\n");

    match Browser::launch(config).await {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("Failed to launch browser: {}", e);
            eprintln!("Make sure Chrome/Chromium is installed");
            None
        }
    }
}

/// Test a single demo page
async fn test_demo(
    browser: &Browser,
    name: &str,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing: {}", name);
    println!("  URL: {}{}", BASE_URL, path);

    let mut page = browser.new_page().await?;
    let url = format!("{}{}", BASE_URL, path);
    page.goto(&url).await?;

    tokio::time::sleep(Duration::from_secs(2)).await;

    let title: String = page
        .eval_wasm("document.title || 'Untitled'")
        .await
        .unwrap_or_else(|_| "Error".to_string());
    println!("  Title: {}", title);

    let has_errors: bool = page
        .eval_wasm("window.__errors && window.__errors.length > 0")
        .await
        .unwrap_or(false);
    if has_errors {
        println!("  WARNING: Console errors detected");
    } else {
        println!("  Status: OK");
    }

    println!("\n  [Pausing 3s for visual inspection...]");
    tokio::time::sleep(Duration::from_secs(3)).await;
    Ok(())
}

/// Run interactive test on realtime transcription page
async fn run_interactive_test(page: &mut Page) -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Checking initial state...");

    let start_btn = Selector::css("#start_recording");
    let start_enabled: bool = page
        .eval_wasm(&format!(
            "(function() {{ const el = {}; return el && !el.disabled; }})()",
            start_btn.to_query()
        ))
        .await
        .unwrap_or(false);
    println!("   Start button enabled: {}", start_enabled);

    let stop_btn = Selector::css("#stop_recording");
    let stop_disabled: bool = page
        .eval_wasm(&format!(
            "(function() {{ const el = {}; return el && el.disabled; }})()",
            stop_btn.to_query()
        ))
        .await
        .unwrap_or(false);
    println!("   Stop button disabled: {}", stop_disabled);

    println!("\n2. Clicking START button...");
    let _: () = page
        .eval_wasm(&format!(
            "(function() {{ const el = {}; if (el) el.click(); }})()",
            start_btn.to_query()
        ))
        .await
        .unwrap_or(());

    tokio::time::sleep(Duration::from_secs(2)).await;

    let status: String = page
        .eval_wasm(
            "(function() { \
                const el = document.querySelector('#status_indicator, #status, .status'); \
                return el ? el.textContent : 'unknown'; \
            })()",
        )
        .await
        .unwrap_or_else(|_| "unknown".to_string());
    println!("   Status after click: {}", status.trim());

    println!("\n3. Taking screenshot...");
    let screenshot = page.screenshot().await?;
    println!("   Screenshot size: {} bytes", screenshot.len());

    let screenshot_path = "/tmp/whisper-apr-uat-screenshot.png";
    std::fs::write(screenshot_path, &screenshot)?;
    println!("   Saved to: {}", screenshot_path);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("===========================================");
    println!("  whisper.apr Demo UAT - Foreground Mode");
    println!("===========================================\n");

    if !check_server().await {
        return Ok(());
    }

    let Some(browser) = launch_browser().await else {
        return Ok(());
    };

    // Test each demo
    let demos = [
        ("Realtime Transcription", "/realtime-transcription.html"),
        ("Upload Transcription", "/upload-transcription.html"),
        ("Realtime Translation", "/realtime-translation.html"),
        ("Upload Translation", "/upload-translation.html"),
    ];

    for (name, path) in demos {
        test_demo(&browser, name, path).await?;
    }

    // Interactive demo
    println!("===========================================");
    println!("  Interactive Test: Realtime Transcription");
    println!("===========================================\n");

    let mut page = browser.new_page().await?;
    page.goto(&format!("{}/realtime-transcription.html", BASE_URL))
        .await?;
    tokio::time::sleep(Duration::from_secs(1)).await;

    run_interactive_test(&mut page).await?;

    println!("\n===========================================");
    println!("  UAT Complete - Browser will stay open");
    println!("  Press Ctrl+C to exit");
    println!("===========================================\n");

    println!("Browser is open for manual testing...");
    println!("You can interact with the demos manually now.\n");

    loop {
        tokio::time::sleep(Duration::from_secs(60)).await;
    }
}
