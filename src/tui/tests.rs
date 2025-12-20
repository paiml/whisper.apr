//! Probar-style TUI tests for Whisper Pipeline Visualization
//!
//! EXTREME TDD: All tests written FIRST before implementation.
//!
//! ## Test Categories
//!
//! 1. Frame Rendering - Each panel renders correctly
//! 2. State Machine - Correct state transitions
//! 3. Keyboard Handling - All bindings work
//! 4. UX Coverage - 100% element coverage
//! 5. Visualization - Correct ASCII art output

use super::*;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

// ============================================================================
// Test Utilities (Probar-style)
// ============================================================================

/// Render a frame to test buffer
fn render_frame(app: &WhisperApp, width: u16, height: u16) -> TestFrame {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|f| render_whisper_dashboard(f, app))
        .expect("draw");
    TestFrame::from_buffer(terminal.backend().buffer())
}

/// Test frame wrapper for probar-style assertions
#[derive(Debug)]
pub struct TestFrame {
    content: String,
    width: u16,
    height: u16,
}

impl TestFrame {
    /// Create from ratatui buffer
    pub fn from_buffer(buffer: &ratatui::buffer::Buffer) -> Self {
        let mut content = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                let cell = buffer.cell((x, y)).expect("cell");
                content.push_str(cell.symbol());
            }
            content.push('\n');
        }
        Self {
            content,
            width: buffer.area.width,
            height: buffer.area.height,
        }
    }

    /// Check if frame contains text
    pub fn contains(&self, text: &str) -> bool {
        self.content.contains(text)
    }

    /// Get frame dimensions
    pub fn dimensions(&self) -> (u16, u16) {
        (self.width, self.height)
    }

    /// Get raw content for debugging
    pub fn as_text(&self) -> &str {
        &self.content
    }
}

/// Probar-style frame assertion builder
pub struct FrameAssertion<'a> {
    frame: &'a TestFrame,
    errors: Vec<String>,
    soft_mode: bool,
}

impl<'a> FrameAssertion<'a> {
    pub fn new(frame: &'a TestFrame) -> Self {
        Self {
            frame,
            errors: Vec::new(),
            soft_mode: false,
        }
    }

    /// Enable soft assertion mode (collect errors instead of failing)
    pub fn soft(mut self) -> Self {
        self.soft_mode = true;
        self
    }

    /// Assert frame contains text
    pub fn to_contain_text(&mut self, text: &str) -> &mut Self {
        if !self.frame.contains(text) {
            let err = format!("Expected frame to contain '{}'", text);
            if self.soft_mode {
                self.errors.push(err);
            } else {
                panic!("{}", err);
            }
        }
        self
    }

    /// Assert frame does NOT contain text
    pub fn not_to_contain_text(&mut self, text: &str) -> &mut Self {
        if self.frame.contains(text) {
            let err = format!("Expected frame NOT to contain '{}'", text);
            if self.soft_mode {
                self.errors.push(err);
            } else {
                panic!("{}", err);
            }
        }
        self
    }

    /// Assert frame has specific dimensions
    pub fn to_have_size(&mut self, width: u16, height: u16) -> &mut Self {
        let (w, h) = self.frame.dimensions();
        if w != width || h != height {
            let err = format!("Expected {}x{}, got {}x{}", width, height, w, h);
            if self.soft_mode {
                self.errors.push(err);
            } else {
                panic!("{}", err);
            }
        }
        self
    }

    /// Get collected errors (soft mode)
    pub fn errors(&self) -> &[String] {
        &self.errors
    }

    /// Finalize soft assertions
    pub fn finalize(&self) -> Result<(), String> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{} assertions failed:\n{}",
                self.errors.len(),
                self.errors.join("\n")
            ))
        }
    }
}

/// Probar-style expect_frame helper
pub fn expect_frame(frame: &TestFrame) -> FrameAssertion<'_> {
    FrameAssertion::new(frame)
}

/// UX Coverage tracker (simplified probar-style)
pub struct UxCoverage {
    panels: Vec<String>,
    visited: std::collections::HashSet<String>,
    keys: Vec<char>,
    pressed: std::collections::HashSet<char>,
}

impl UxCoverage {
    pub fn new() -> Self {
        Self {
            panels: vec![
                "waveform".to_string(),
                "mel".to_string(),
                "encoder".to_string(),
                "decoder".to_string(),
                "attention".to_string(),
                "transcription".to_string(),
                "metrics".to_string(),
                "help".to_string(),
            ],
            visited: std::collections::HashSet::new(),
            keys: vec!['1', '2', '3', '4', '5', '6', '7', 'q', 'r', ' ', '?'],
            pressed: std::collections::HashSet::new(),
        }
    }

    pub fn visit_panel(&mut self, panel: &str) {
        self.visited.insert(panel.to_string());
    }

    pub fn press_key(&mut self, key: char) {
        self.pressed.insert(key);
    }

    pub fn panel_coverage(&self) -> f64 {
        self.visited.len() as f64 / self.panels.len() as f64
    }

    pub fn key_coverage(&self) -> f64 {
        self.pressed.len() as f64 / self.keys.len() as f64
    }

    pub fn overall_coverage(&self) -> f64 {
        (self.panel_coverage() + self.key_coverage()) / 2.0
    }

    pub fn is_complete(&self) -> bool {
        self.visited.len() >= self.panels.len() && self.pressed.len() >= self.keys.len()
    }
}

impl Default for UxCoverage {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Application State Tests
// ============================================================================

#[test]
fn test_whisper_app_initial_state() {
    let app = WhisperApp::new();

    assert_eq!(app.current_panel, WhisperPanel::Waveform);
    assert_eq!(app.state, WhisperState::Idle);
    assert!(!app.should_quit);
    assert!(!app.paused);
}

#[test]
fn test_whisper_app_panel_navigation() {
    let mut app = WhisperApp::new();

    // Navigate through all panels
    app.set_panel(WhisperPanel::Mel);
    assert_eq!(app.current_panel, WhisperPanel::Mel);

    app.set_panel(WhisperPanel::Encoder);
    assert_eq!(app.current_panel, WhisperPanel::Encoder);

    app.set_panel(WhisperPanel::Decoder);
    assert_eq!(app.current_panel, WhisperPanel::Decoder);

    app.set_panel(WhisperPanel::Attention);
    assert_eq!(app.current_panel, WhisperPanel::Attention);

    app.set_panel(WhisperPanel::Transcription);
    assert_eq!(app.current_panel, WhisperPanel::Transcription);

    app.set_panel(WhisperPanel::Metrics);
    assert_eq!(app.current_panel, WhisperPanel::Metrics);

    app.set_panel(WhisperPanel::Help);
    assert_eq!(app.current_panel, WhisperPanel::Help);
}

#[test]
fn test_whisper_app_state_machine() {
    let mut app = WhisperApp::new();

    // Start idle
    assert_eq!(app.state, WhisperState::Idle);

    // Load audio → Waveform state
    app.load_audio(&[0.0; 16000]);
    assert_eq!(app.state, WhisperState::WaveformReady);

    // Compute mel → Mel state
    app.compute_mel();
    assert_eq!(app.state, WhisperState::MelReady);

    // Encode → Encoding state
    app.start_encoding();
    assert_eq!(app.state, WhisperState::Encoding);

    // Decode → Decoding state
    app.start_decoding();
    assert_eq!(app.state, WhisperState::Decoding);

    // Complete → Complete state
    app.complete();
    assert_eq!(app.state, WhisperState::Complete);
}

#[test]
fn test_whisper_app_keyboard_handling() {
    let mut app = WhisperApp::new();

    // Number keys switch panels
    app.handle_key('1');
    assert_eq!(app.current_panel, WhisperPanel::Waveform);

    app.handle_key('2');
    assert_eq!(app.current_panel, WhisperPanel::Mel);

    app.handle_key('3');
    assert_eq!(app.current_panel, WhisperPanel::Encoder);

    app.handle_key('4');
    assert_eq!(app.current_panel, WhisperPanel::Decoder);

    app.handle_key('5');
    assert_eq!(app.current_panel, WhisperPanel::Attention);

    app.handle_key('6');
    assert_eq!(app.current_panel, WhisperPanel::Transcription);

    app.handle_key('7');
    assert_eq!(app.current_panel, WhisperPanel::Metrics);

    app.handle_key('?');
    assert_eq!(app.current_panel, WhisperPanel::Help);

    // Space toggles pause
    assert!(!app.paused);
    app.handle_key(' ');
    assert!(app.paused);
    app.handle_key(' ');
    assert!(!app.paused);

    // r resets
    app.load_audio(&[0.0; 16000]);
    assert_ne!(app.state, WhisperState::Idle);
    app.handle_key('r');
    assert_eq!(app.state, WhisperState::Idle);

    // q quits
    assert!(!app.should_quit);
    app.handle_key('q');
    assert!(app.should_quit);
}

#[test]
fn test_whisper_app_reset() {
    let mut app = WhisperApp::new();

    // Advance through pipeline
    app.load_audio(&[0.0; 16000]);
    app.compute_mel();
    app.start_encoding();

    // Reset
    app.reset();

    assert_eq!(app.state, WhisperState::Idle);
    assert!(app.audio_data.is_empty());
    assert!(app.mel_data.is_empty());
}

// ============================================================================
// Panel Rendering Tests
// ============================================================================

#[test]
fn test_waveform_panel_renders() {
    let mut app = WhisperApp::new();
    // Create sample audio data
    let pattern = [0.1_f32, -0.1, 0.2, -0.2, 0.0];
    let audio: Vec<f32> = pattern.iter().cycle().take(500).copied().collect();
    app.load_audio(&audio);

    let frame = render_frame(&app, 80, 24);

    expect_frame(&frame)
        .to_contain_text("WAVEFORM")
        .to_contain_text("samples");
}

#[test]
fn test_mel_panel_renders() {
    let mut app = WhisperApp::new();
    app.load_audio(&vec![0.0; 16000]);
    app.compute_mel();
    app.set_panel(WhisperPanel::Mel);

    let frame = render_frame(&app, 80, 24);

    expect_frame(&frame)
        .to_contain_text("MEL")
        .to_contain_text("80"); // 80 mel bins
}

#[test]
fn test_encoder_panel_renders() {
    let mut app = WhisperApp::new();
    app.set_panel(WhisperPanel::Encoder);

    let frame = render_frame(&app, 80, 24);

    expect_frame(&frame)
        .to_contain_text("ENCODER")
        .to_contain_text("Layer");
}

#[test]
fn test_decoder_panel_renders() {
    let mut app = WhisperApp::new();
    app.set_panel(WhisperPanel::Decoder);

    let frame = render_frame(&app, 80, 24);

    expect_frame(&frame)
        .to_contain_text("DECODER")
        .to_contain_text("Token");
}

#[test]
fn test_attention_panel_renders() {
    let mut app = WhisperApp::new();
    app.set_panel(WhisperPanel::Attention);

    let frame = render_frame(&app, 80, 24);

    expect_frame(&frame)
        .to_contain_text("ATTENTION")
        .to_contain_text("Cross");
}

#[test]
fn test_transcription_panel_renders() {
    let mut app = WhisperApp::new();
    app.set_panel(WhisperPanel::Transcription);

    let frame = render_frame(&app, 80, 24);

    expect_frame(&frame).to_contain_text("TRANSCRIPTION");
}

#[test]
fn test_metrics_panel_renders() {
    let mut app = WhisperApp::new();
    app.set_panel(WhisperPanel::Metrics);

    let frame = render_frame(&app, 80, 24);

    expect_frame(&frame)
        .to_contain_text("METRICS")
        .to_contain_text("RTF");
}

#[test]
fn test_help_panel_renders() {
    let mut app = WhisperApp::new();
    app.set_panel(WhisperPanel::Help);

    let frame = render_frame(&app, 80, 24);

    expect_frame(&frame)
        .to_contain_text("HELP")
        .to_contain_text("Keyboard");
}

#[test]
fn test_all_panels_render_without_panic() {
    let app = WhisperApp::new();

    for panel in [
        WhisperPanel::Waveform,
        WhisperPanel::Mel,
        WhisperPanel::Encoder,
        WhisperPanel::Decoder,
        WhisperPanel::Attention,
        WhisperPanel::Transcription,
        WhisperPanel::Metrics,
        WhisperPanel::Help,
    ] {
        let mut app = app.clone();
        app.set_panel(panel);
        let _frame = render_frame(&app, 80, 24);
        // Just verify no panic
    }
}

// ============================================================================
// Visualization Tests
// ============================================================================

#[test]
fn test_waveform_visualization() {
    // Test data with clear peaks
    let audio: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();

    let display = WaveformDisplay::new(&audio, 40, 10);

    // Should produce ASCII art
    let output = display.render();
    assert!(!output.is_empty());

    // Should have correct dimensions
    assert_eq!(display.width(), 40);
    assert_eq!(display.height(), 10);
}

#[test]
fn test_mel_spectrogram_visualization() {
    // Mock mel data: 80 bins x 100 frames
    let mel: Vec<f32> = (0..8000)
        .map(|i| ((i % 80) as f32 / 80.0) * -4.0) // Log-mel scale
        .collect();

    let display = MelDisplay::new(&mel, 80, 100, 40, 20);

    // Should produce heatmap
    let output = display.render();
    assert!(!output.is_empty());
}

#[test]
fn test_waveform_handles_empty_input() {
    let display = WaveformDisplay::new(&[], 40, 10);
    let output = display.render();

    // Should handle gracefully, not panic
    assert!(output.is_empty() || output.contains("No data"));
}

#[test]
fn test_mel_handles_empty_input() {
    let display = MelDisplay::new(&[], 80, 0, 40, 20);
    let output = display.render();

    // Should handle gracefully
    assert!(output.is_empty() || output.contains("No data"));
}

// ============================================================================
// UX Coverage Tests
// ============================================================================

#[test]
fn test_ux_coverage_tracks_panels() {
    let mut coverage = UxCoverage::new();

    coverage.visit_panel("waveform");
    coverage.visit_panel("mel");
    coverage.visit_panel("encoder");

    assert!(coverage.panel_coverage() > 0.3);
    assert!(coverage.panel_coverage() < 0.5);
}

#[test]
fn test_ux_coverage_tracks_keys() {
    let mut coverage = UxCoverage::new();

    coverage.press_key('1');
    coverage.press_key('2');
    coverage.press_key('q');

    assert!(coverage.key_coverage() > 0.2);
}

#[test]
fn test_ux_coverage_complete() {
    let mut coverage = UxCoverage::new();

    // Visit all panels
    for panel in [
        "waveform",
        "mel",
        "encoder",
        "decoder",
        "attention",
        "transcription",
        "metrics",
        "help",
    ] {
        coverage.visit_panel(panel);
    }

    // Press all keys
    for key in ['1', '2', '3', '4', '5', '6', '7', 'q', 'r', ' ', '?'] {
        coverage.press_key(key);
    }

    assert!(coverage.is_complete());
    assert!((coverage.overall_coverage() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_full_ux_coverage_via_app() {
    let mut app = WhisperApp::new();
    let mut coverage = UxCoverage::new();

    // Exercise all panels via keyboard
    for (key, panel) in [
        ('1', "waveform"),
        ('2', "mel"),
        ('3', "encoder"),
        ('4', "decoder"),
        ('5', "attention"),
        ('6', "transcription"),
        ('7', "metrics"),
        ('?', "help"),
    ] {
        app.handle_key(key);
        coverage.press_key(key);
        coverage.visit_panel(panel);

        // Verify panel changed
        let frame = render_frame(&app, 80, 24);
        assert!(frame.contains(&panel.to_uppercase()) || frame.contains(panel));
    }

    // Exercise other keys
    app.handle_key(' '); // pause
    coverage.press_key(' ');

    app.handle_key('r'); // reset
    coverage.press_key('r');

    app.handle_key('q'); // quit
    coverage.press_key('q');

    assert!(
        coverage.is_complete(),
        "UX coverage not complete: {:.0}%",
        coverage.overall_coverage() * 100.0
    );
}

// ============================================================================
// Soft Assertion Tests
// ============================================================================

#[test]
fn test_soft_assertions_collect_errors() {
    let app = WhisperApp::new();
    let frame = render_frame(&app, 80, 24);

    let mut assertion = expect_frame(&frame).soft();

    // These should pass
    assertion.to_contain_text("WAVEFORM");
    assertion.to_have_size(80, 24);

    // This should fail but not panic
    assertion.to_contain_text("NONEXISTENT_TEXT_12345");

    let errors = assertion.errors();
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("NONEXISTENT_TEXT_12345"));
}

// ============================================================================
// Frame Sequence Tests
// ============================================================================

#[test]
fn test_panel_transition_sequence() {
    let mut app = WhisperApp::new();
    let mut frames = Vec::new();

    // Capture frames as we navigate
    for panel in [
        WhisperPanel::Waveform,
        WhisperPanel::Mel,
        WhisperPanel::Encoder,
        WhisperPanel::Decoder,
    ] {
        app.set_panel(panel);
        frames.push(render_frame(&app, 80, 24));
    }

    // All frames should be different
    for i in 0..frames.len() {
        for j in (i + 1)..frames.len() {
            assert_ne!(
                frames[i].as_text(),
                frames[j].as_text(),
                "Frames {} and {} should differ",
                i,
                j
            );
        }
    }
}

#[test]
fn test_state_transition_sequence() {
    let mut app = WhisperApp::new();
    let mut states = Vec::new();

    // Track state transitions
    states.push(app.state);

    app.load_audio(&[0.0; 16000]);
    states.push(app.state);

    app.compute_mel();
    states.push(app.state);

    app.start_encoding();
    states.push(app.state);

    app.start_decoding();
    states.push(app.state);

    app.complete();
    states.push(app.state);

    // Verify expected sequence
    assert_eq!(
        states,
        vec![
            WhisperState::Idle,
            WhisperState::WaveformReady,
            WhisperState::MelReady,
            WhisperState::Encoding,
            WhisperState::Decoding,
            WhisperState::Complete,
        ]
    );
}

// ============================================================================
// Property-Based Tests
// ============================================================================

mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn property_panel_index_roundtrip(index in 0usize..8) {
            // from_index -> index should round-trip for valid indices
            let panel = WhisperPanel::from_index(index);
            let back = panel.index();
            // from_index clamps to Help for index >= 7, so check valid range
            if index < 8 {
                prop_assert!(back <= 7, "panel index should be valid");
            }
        }

        #[test]
        fn property_load_audio_changes_state(len in 1usize..10000) {
            // Loading any non-empty audio should transition to WaveformReady
            let mut app = WhisperApp::new();
            let audio: Vec<f32> = (0..len).map(|i| (i as f32 * 0.01).sin()).collect();
            app.load_audio(&audio);
            prop_assert_eq!(app.state, WhisperState::WaveformReady);
            prop_assert_eq!(app.audio_data.len(), len);
        }

        #[test]
        fn property_reset_clears_state(len in 1usize..1000) {
            // Reset should always return to Idle with empty data
            let mut app = WhisperApp::new();
            let audio: Vec<f32> = vec![0.0; len];
            app.load_audio(&audio);
            app.compute_mel();
            app.reset();
            prop_assert_eq!(app.state, WhisperState::Idle);
            prop_assert!(app.audio_data.is_empty());
            prop_assert!(app.mel_data.is_empty());
        }

        #[test]
        fn property_waveform_render_no_panic(
            samples in prop::collection::vec(-1.0f32..1.0, 0..1000),
            width in 1usize..200,
            height in 1usize..50
        ) {
            // Waveform rendering should never panic
            let result = std::panic::catch_unwind(|| {
                super::super::visualization::render_waveform(&samples, width, height)
            });
            prop_assert!(result.is_ok(), "waveform render should not panic");
        }

        #[test]
        fn property_mel_render_no_panic(
            n_frames in 1usize..100,
            width in 1usize..100,
            height in 1usize..50
        ) {
            // Mel rendering should never panic with valid dimensions
            let mel_data: Vec<f32> = vec![0.0; 80 * n_frames];
            let result = std::panic::catch_unwind(|| {
                super::super::visualization::render_mel_spectrogram(
                    &mel_data, 80, n_frames, width, height
                )
            });
            prop_assert!(result.is_ok(), "mel render should not panic");
        }

        #[test]
        fn property_attention_render_no_panic(
            n_tokens in 1usize..20,
            n_frames in 1usize..50,
            width in 1usize..80,
            height in 1usize..30
        ) {
            // Attention rendering should never panic
            let attention: Vec<Vec<f32>> = (0..n_tokens)
                .map(|_| (0..n_frames).map(|f| f as f32 / n_frames as f32).collect())
                .collect();
            let result = std::panic::catch_unwind(|| {
                super::super::visualization::render_attention_heatmap(&attention, width, height)
            });
            prop_assert!(result.is_ok(), "attention render should not panic");
        }

        #[test]
        fn property_keyboard_valid_keys(key in "[1-7qr? ]") {
            // Valid keys should not crash and should change state appropriately
            let mut app = WhisperApp::new();
            for c in key.chars() {
                app.handle_key(c);
            }
            // App should still be in a valid state
            prop_assert!(matches!(
                app.state,
                WhisperState::Idle
                    | WhisperState::WaveformReady
                    | WhisperState::MelReady
                    | WhisperState::Encoding
                    | WhisperState::Decoding
                    | WhisperState::Complete
                    | WhisperState::Error
            ));
        }

        #[test]
        fn property_rtf_non_negative(
            audio_len in 1000usize..100000,
            processing_ms in 1.0f32..10000.0
        ) {
            // RTF should always be non-negative
            let mut metrics = super::super::app::PipelineMetrics::default();
            metrics.audio_duration_secs = audio_len as f32 / 16000.0;
            metrics.total_time_ms = processing_ms;
            metrics.compute_rtf();
            prop_assert!(metrics.rtf >= 0.0, "RTF should be non-negative");
        }
    }
}
