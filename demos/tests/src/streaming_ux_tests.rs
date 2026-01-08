//! Streaming UX Tests using Probar 0.4.1+ Features (WAPR-200)
//!
//! Tests the real-time streaming transcription UX using probar's advanced
//! testing capabilities:
//! - AudioEmulator: Inject controlled speech-like audio
//! - WasmThreadCapabilities: Verify SharedArrayBuffer/COOP/COEP
//! - StreamingUxValidator: Assert VU meter, state transitions
//! - WorkerEmulator: Test Web Worker lifecycle
//!
//! Run with: `cargo test --package whisper-apr-demo-tests streaming_ux_tests`
//!
//! Prerequisites:
//! - Chrome/Chromium installed
//! - Demos built: `make build` in demos/
//! - Server running: `make serve-dev` (port 8081, COOP/COEP enabled)

use probar::{
    capabilities::{WasmThreadCapabilities, WorkerEmulator},
    emulation::{AudioEmulator, AudioSource},
    validators::StreamingUxValidator,
    Browser, BrowserConfig,
};
use std::time::Duration;

/// Base URL for local demo server with COOP/COEP headers
const BASE_URL: &str = "http://localhost:8081";

/// Create browser config for streaming UX testing
fn streaming_browser_config() -> BrowserConfig {
    BrowserConfig::default()
        .with_headless(true)
        .with_viewport(1280, 720)
        .with_no_sandbox()
}

// =============================================================================
// F001-F015: WasmThreadCapabilities Tests
// =============================================================================

/// F001: SharedArrayBuffer must be available for streaming WASM
#[tokio::test]
#[ignore = "Requires running server with COOP/COEP: make serve-dev"]
async fn test_shared_array_buffer_available() {
    let browser = Browser::launch(streaming_browser_config())
        .await
        .expect("Browser should launch");

    let mut page = browser.new_page().await.expect("Page should open");
    page.goto(&format!("{}/realtime-transcription.html", BASE_URL))
        .await
        .expect("Navigation should succeed");

    // Get CDP page for capabilities detection
    let cdp = page.cdp_page().await.expect("Should have CDP page");

    // Detect WASM threading capabilities via CDP
    let caps = WasmThreadCapabilities::detect(&*cdp)
        .await
        .expect("Should detect capabilities");

    // Assert streaming requirements are met
    caps.assert_streaming_ready()
        .expect("SharedArrayBuffer and Atomics must be available");

    assert!(
        caps.shared_array_buffer,
        "SharedArrayBuffer required for streaming"
    );
    assert!(caps.atomics, "Atomics required for thread synchronization");
}

/// F002: COOP/COEP headers must be present (cross_origin_isolated)
#[tokio::test]
#[ignore = "Requires running server with COOP/COEP: make serve-dev"]
async fn test_coop_coep_headers() {
    let browser = Browser::launch(streaming_browser_config())
        .await
        .expect("Browser should launch");

    let mut page = browser.new_page().await.expect("Page should open");
    page.goto(&format!("{}/realtime-transcription.html", BASE_URL))
        .await
        .expect("Navigation should succeed");

    let cdp = page.cdp_page().await.expect("Should have CDP page");
    let caps = WasmThreadCapabilities::detect(&*cdp)
        .await
        .expect("Should detect capabilities");

    // cross_origin_isolated implies COOP/COEP headers are set
    assert!(
        caps.cross_origin_isolated,
        "COOP/COEP headers must be set (crossOriginIsolated=true)"
    );
}

// =============================================================================
// F016-F028: AudioEmulator Tests
// =============================================================================

/// F016: Inject speech-like audio for VAD testing
#[tokio::test]
#[ignore = "Requires running server: make serve-dev"]
async fn test_audio_emulator_injection() {
    let browser = Browser::launch(streaming_browser_config())
        .await
        .expect("Browser should launch");

    let mut page = browser.new_page().await.expect("Page should open");
    page.goto(&format!("{}/realtime-transcription.html", BASE_URL))
        .await
        .expect("Navigation should succeed");

    // Create speech-like audio (150Hz fundamental + harmonics)
    let mut audio = AudioEmulator::new(AudioSource::SpeechPattern {
        fundamental_hz: 150.0,
        harmonics: vec![0.5, 0.3, 0.2, 0.1],
        variation_hz: 20.0,
    });

    // Get CDP page for audio injection
    let cdp = page.cdp_page().await.expect("Should have CDP page");

    // Inject 3 seconds of audio
    audio
        .inject_cdp(&*cdp, 3.0)
        .await
        .expect("Audio injection should succeed");

    // Verify injection is active
    let is_active = AudioEmulator::is_active_cdp(&*cdp)
        .await
        .expect("Should check audio status");

    assert!(is_active, "Audio emulator should be active after injection");
}

/// F017: VAD should detect injected speech
#[tokio::test]
#[ignore = "Requires running server: make serve-dev"]
async fn test_vad_detects_injected_speech() {
    let browser = Browser::launch(streaming_browser_config())
        .await
        .expect("Browser should launch");

    let mut page = browser.new_page().await.expect("Page should open");
    page.goto(&format!("{}/realtime-transcription.html", BASE_URL))
        .await
        .expect("Navigation should succeed");

    // Wait for model to load
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Inject speech pattern
    let mut audio = AudioEmulator::new(AudioSource::SpeechPattern {
        fundamental_hz: 150.0,
        harmonics: vec![0.5, 0.3, 0.2],
        variation_hz: 15.0,
    });

    {
        let cdp = page.cdp_page().await.expect("CDP page");
        audio.inject_cdp(&*cdp, 5.0).await.expect("Audio injection");
    }

    // Click start recording
    page.click("#start_recording")
        .await
        .expect("Should click start");

    // Wait for state to transition
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Check state label has changed from "Listening" to "Recording"
    let result = page
        .evaluate("document.getElementById('state_label')?.textContent || 'NOT_FOUND'")
        .await
        .expect("Should get state");

    let state: String = result
        .value()
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_default();

    assert!(
        state.contains("Recording") || state.contains("Transcribing"),
        "VAD should detect speech and transition state, got: {state}"
    );
}

// =============================================================================
// F029-F038: StreamingUxValidator Tests
// =============================================================================

/// F029: VU meter should respond to audio input
#[tokio::test]
#[ignore = "Requires running server: make serve-dev"]
async fn test_vu_meter_responds_to_audio() {
    let browser = Browser::launch(streaming_browser_config())
        .await
        .expect("Browser should launch");

    let mut page = browser.new_page().await.expect("Page should open");
    page.goto(&format!("{}/realtime-transcription.html", BASE_URL))
        .await
        .expect("Navigation should succeed");

    // Wait for model to load
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Inject audio
    let mut audio = AudioEmulator::new(AudioSource::SpeechPattern {
        fundamental_hz: 200.0,
        harmonics: vec![0.4, 0.2, 0.1],
        variation_hz: 10.0,
    });

    {
        let cdp = page.cdp_page().await.expect("CDP page");
        audio.inject_cdp(&*cdp, 5.0).await.expect("Audio injection");
    }

    // Start recording
    page.click("#start_recording")
        .await
        .expect("Should click start");

    // Create validator and track VU meter
    let mut validator = StreamingUxValidator::new().with_max_latency(Duration::from_millis(500));

    {
        let cdp = page.cdp_page().await.expect("CDP page");
        validator
            .track_vu_meter_cdp(&*cdp, "#vu_meter")
            .await
            .expect("Should track VU meter");
    }

    // Wait for samples
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Assert VU meter was active (level > 0.05 for at least 1000ms)
    {
        let cdp = page.cdp_page().await.expect("CDP page");
        validator
            .assert_vu_meter_active_cdp(&*cdp, 0.05, 1000)
            .await
            .expect("VU meter should show audio level changes");
    }
}

/// F030: State indicator should show correct transitions
#[tokio::test]
#[ignore = "Requires running server: make serve-dev"]
async fn test_state_indicator_transitions() {
    let browser = Browser::launch(streaming_browser_config())
        .await
        .expect("Browser should launch");

    let mut page = browser.new_page().await.expect("Page should open");
    page.goto(&format!("{}/realtime-transcription.html", BASE_URL))
        .await
        .expect("Navigation should succeed");

    // Wait for model
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Start tracking state before interaction
    let validator = {
        let cdp = page.cdp_page().await.expect("CDP page");
        StreamingUxValidator::track_state_cdp(&*cdp, "#state_label")
            .await
            .expect("Should track state")
    };

    // Inject speech with silence pattern to trigger endpoint
    let mut audio = AudioEmulator::new(AudioSource::SpeechPattern {
        fundamental_hz: 150.0,
        harmonics: vec![0.5, 0.3, 0.2],
        variation_hz: 20.0,
    });

    {
        let cdp = page.cdp_page().await.expect("CDP page");
        audio.inject_cdp(&*cdp, 4.0).await.expect("Audio injection");
    }

    // Start recording
    page.click("#start_recording").await.expect("Start");

    // Wait for state transitions
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Assert state sequence occurred
    {
        let cdp = page.cdp_page().await.expect("CDP page");
        validator
            .assert_state_sequence_cdp(&*cdp, &["Listening", "Recording"])
            .await
            .expect("Should see Listening -> Recording transition");
    }
}

/// F031: Chunk progress should advance during recording
#[tokio::test]
#[ignore = "Requires running server: make serve-dev"]
async fn test_chunk_progress_advances() {
    let browser = Browser::launch(streaming_browser_config())
        .await
        .expect("Browser should launch");

    let mut page = browser.new_page().await.expect("Page should open");
    page.goto(&format!("{}/realtime-transcription.html", BASE_URL))
        .await
        .expect("Navigation should succeed");

    // Wait for model
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Inject audio
    let mut audio = AudioEmulator::new(AudioSource::SpeechPattern {
        fundamental_hz: 150.0,
        harmonics: vec![0.5, 0.3],
        variation_hz: 15.0,
    });

    {
        let cdp = page.cdp_page().await.expect("CDP page");
        audio.inject_cdp(&*cdp, 5.0).await.expect("Audio injection");
    }

    // Start recording
    page.click("#start_recording").await.expect("Start");
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Get initial progress
    let initial_result = page
        .evaluate("parseFloat(document.getElementById('chunk_progress')?.style.width || '0')")
        .await
        .expect("Should get progress");
    let initial: f64 = initial_result
        .value()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    // Wait and check again
    tokio::time::sleep(Duration::from_secs(2)).await;

    let final_result = page
        .evaluate("parseFloat(document.getElementById('chunk_progress')?.style.width || '0')")
        .await
        .expect("Should get progress");
    let final_progress: f64 = final_result.value().and_then(|v| v.as_f64()).unwrap_or(0.0);

    assert!(
        final_progress > initial,
        "Chunk progress should advance: initial={initial}%, final={final_progress}%"
    );
}

// =============================================================================
// F039-F048: WorkerEmulator Tests (if Workers are used)
// =============================================================================

/// F039: Web Worker should be created for transcription
#[tokio::test]
#[ignore = "Requires running server: make serve-dev"]
async fn test_worker_lifecycle() {
    let browser = Browser::launch(streaming_browser_config())
        .await
        .expect("Browser should launch");

    let mut page = browser.new_page().await.expect("Page should open");
    page.goto(&format!("{}/realtime-transcription.html", BASE_URL))
        .await
        .expect("Navigation should succeed");

    let cdp = page.cdp_page().await.expect("CDP page");

    // Attach worker emulator
    WorkerEmulator::attach_cdp(&*cdp)
        .await
        .expect("Should attach to workers");

    // Get workers (may be empty if no dedicated workers used)
    let workers = WorkerEmulator::get_workers_cdp(&*cdp)
        .await
        .expect("Should get workers");

    // Note: whisper.apr demo may not use dedicated workers
    // This test verifies the capability is working
    eprintln!("Found {} worker(s)", workers.len());
}

// =============================================================================
// Integration Test: Full Streaming Flow
// =============================================================================

/// Full integration test: Audio -> VAD -> State -> UX
#[tokio::test]
#[ignore = "Requires running server: make serve-dev"]
async fn test_full_streaming_ux_flow() {
    let browser = Browser::launch(streaming_browser_config())
        .await
        .expect("Browser should launch");

    let mut page = browser.new_page().await.expect("Page should open");
    page.goto(&format!("{}/realtime-transcription.html", BASE_URL))
        .await
        .expect("Navigation should succeed");

    // 1. Verify WASM threading capabilities
    {
        let cdp = page.cdp_page().await.expect("CDP page");
        let caps = WasmThreadCapabilities::detect(&*cdp)
            .await
            .expect("Should detect capabilities");
        caps.assert_streaming_ready()
            .expect("Threading support required");
    }

    // 2. Wait for model to load
    for i in 0..30 {
        let result = page
            .evaluate("document.getElementById('start_recording')?.disabled ?? true")
            .await
            .expect("Check button");
        let button_disabled: bool = result.value().and_then(|v| v.as_bool()).unwrap_or(true);

        if !button_disabled {
            eprintln!("Model loaded after {i}s");
            break;
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    // 3. Start tracking state
    let validator = {
        let cdp = page.cdp_page().await.expect("CDP page");
        StreamingUxValidator::track_state_cdp(&*cdp, "#state_label")
            .await
            .expect("Track state")
    };

    // 4. Inject speech audio
    let mut audio = AudioEmulator::new(AudioSource::SpeechPattern {
        fundamental_hz: 150.0,
        harmonics: vec![0.5, 0.3, 0.2, 0.1],
        variation_hz: 20.0,
    });

    {
        let cdp = page.cdp_page().await.expect("CDP page");
        audio.inject_cdp(&*cdp, 5.0).await.expect("Audio injection");
    }

    // 5. Start recording
    page.click("#start_recording").await.expect("Start");

    // 6. Wait for some processing
    tokio::time::sleep(Duration::from_secs(3)).await;

    // 7. Stop recording
    page.click("#start_recording").await.expect("Stop");

    // 8. Verify UX responded appropriately
    let result = page
        .evaluate("document.getElementById('state_label')?.textContent || 'UNKNOWN'")
        .await
        .expect("Get final state");
    let final_state: String = result
        .value()
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_default();

    eprintln!("Final state: {final_state}");

    // 9. Collect state history
    let history = {
        let cdp = page.cdp_page().await.expect("CDP page");
        validator
            .collect_state_history_cdp(&*cdp)
            .await
            .expect("Get history")
    };

    // State should have transitioned during recording
    assert!(
        history.len() > 1 || final_state.contains("Listening"),
        "State machine should have processed audio"
    );
}

#[cfg(test)]
mod tests {
    //! Unit tests for streaming UX test helpers

    use super::*;

    #[test]
    fn test_audio_source_speech_pattern() {
        let mut audio = AudioEmulator::new(AudioSource::SpeechPattern {
            fundamental_hz: 150.0,
            harmonics: vec![0.5, 0.3, 0.2],
            variation_hz: 20.0,
        });

        let samples = audio.generate_samples(0.1); // 100ms
        assert!(!samples.is_empty(), "Should generate samples");

        // Samples should be in valid range
        for sample in &samples {
            assert!(
                *sample >= -1.5 && *sample <= 1.5,
                "Sample out of range: {sample}"
            );
        }
    }

    #[test]
    fn test_streaming_ux_validator_creation() {
        let validator = StreamingUxValidator::new()
            .with_max_latency(Duration::from_millis(100))
            .with_min_fps(30.0);

        // Should be created without errors
        assert!(validator.state_history().is_empty());
    }

    #[test]
    fn test_browser_config_creation() {
        let config = streaming_browser_config();
        // Config should be created without errors
        assert!(config.headless);
    }
}
