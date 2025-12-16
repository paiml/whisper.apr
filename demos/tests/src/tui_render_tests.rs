//! Probar TUI Render Tests for Benchmark TUI
//!
//! These tests verify the TUI rendering output using probar's
//! frame capture and assertion API.
//!
//! Run: cargo test -p whisper-apr-demo-tests tui_render

#![allow(missing_docs)]

use probar::tui::{expect_frame, TuiFrame};
use ratatui::{
    backend::TestBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Gauge, Paragraph},
    Frame, Terminal,
};

// =============================================================================
// Minimal App State for Testing
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepId {
    A,
    B,
    C,
    D,
    F,
    G,
    H,
}

/// Code path backend types for each pipeline step
/// Uses realizar inference primitives for world-class production performance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Scalar (naive) implementation - fallback only
    Scalar,
    /// SIMD-accelerated via trueno (shared with realizar)
    Simd,
    /// Flash Attention from realizar (SIMD + O(n) memory)
    Flash,
    /// Paged KV Cache from realizar (vLLM-style memory management)
    PagedKv,
    /// Fused LayerNorm+Linear from realizar (reduced memory)
    Fused,
    /// WebGPU compute shader
    Gpu,
    /// Disk I/O (no compute)
    DiskIo,
    /// Memory operations only
    Memory,
}

impl Backend {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Scalar => "Scalar",
            Self::Simd => "SIMD",
            Self::Flash => "Flash",
            Self::PagedKv => "PagedKV",
            Self::Fused => "Fused",
            Self::Gpu => "GPU",
            Self::DiskIo => "I/O",
            Self::Memory => "Mem",
        }
    }

    pub fn short(&self) -> &'static str {
        match self {
            Self::Scalar => "sc",
            Self::Simd => "si",
            Self::Flash => "fl",
            Self::PagedKv => "pk",
            Self::Fused => "fu",
            Self::Gpu => "gp",
            Self::DiskIo => "io",
            Self::Memory => "mm",
        }
    }

    /// Returns the source of this backend implementation
    pub fn source(&self) -> &'static str {
        match self {
            Self::Scalar => "whisper-apr",
            Self::Simd => "trueno",
            Self::Flash => "realizar",
            Self::PagedKv => "realizar",
            Self::Fused => "realizar",
            Self::Gpu => "wgpu",
            Self::DiskIo => "std",
            Self::Memory => "std",
        }
    }
}

impl StepId {
    pub fn as_char(&self) -> char {
        match self {
            Self::A => 'A',
            Self::B => 'B',
            Self::C => 'C',
            Self::D => 'D',
            Self::F => 'F',
            Self::G => 'G',
            Self::H => 'H',
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::A => "Model",
            Self::B => "Load",
            Self::C => "Parse",
            Self::D => "Resample",
            Self::F => "Mel",
            Self::G => "Encode",
            Self::H => "Decode",
        }
    }

    /// Returns the backend used for this step when realizar-inference feature is enabled
    /// Uses realizar's world-class inference primitives
    pub fn backend_simd(&self) -> Backend {
        match self {
            Self::A => Backend::DiskIo,   // Model loading is disk I/O
            Self::B => Backend::DiskIo,   // Audio loading is disk I/O
            Self::C => Backend::Memory,   // PCM parsing is memory ops
            Self::D => Backend::Simd,     // Resampling uses trueno SIMD
            Self::F => Backend::Simd,     // Mel spectrogram uses trueno SIMD FFT
            Self::G => Backend::Flash,    // Encoder: realizar Flash Attention
            Self::H => Backend::Flash,    // Decoder: realizar Flash + PagedKV cache
        }
    }

    /// Returns the backend used for this step when SIMD feature is disabled
    pub fn backend_scalar(&self) -> Backend {
        match self {
            Self::A => Backend::DiskIo,
            Self::B => Backend::DiskIo,
            Self::C => Backend::Memory,
            Self::D => Backend::Scalar,
            Self::F => Backend::Scalar,
            Self::G => Backend::Scalar,
            Self::H => Backend::Scalar,
        }
    }

    /// Returns the current backend based on feature flags
    pub fn backend(&self) -> Backend {
        if cfg!(feature = "simd") {
            self.backend_simd()
        } else {
            self.backend_scalar()
        }
    }

    /// Format step with backend annotation: "[G] Encode (SIMD)"
    pub fn label_with_backend(&self) -> String {
        format!("[{}] {} ({})", self.as_char(), self.name(), self.backend().label())
    }

    pub fn target_ms(&self) -> u64 {
        match self {
            Self::A => 500,  // Model load
            Self::B => 50,   // Audio load
            Self::C => 10,   // Parse
            Self::D => 100,  // Resample
            Self::F => 50,   // Mel
            Self::G => 500,  // Encode
            Self::H => 2000, // Decode
        }
    }

    pub fn all() -> &'static [Self] {
        &[Self::A, Self::B, Self::C, Self::D, Self::F, Self::G, Self::H]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StepStatus {
    Pending,
    Running,
    Complete,
}

#[derive(Debug, Clone)]
pub struct PipelineStep {
    pub id: StepId,
    pub status: StepStatus,
    pub elapsed_ms: f64,
}

impl PipelineStep {
    pub fn new(id: StepId) -> Self {
        Self {
            id,
            status: StepStatus::Pending,
            elapsed_ms: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AppState {
    Idle,
    Running,
    Completed,
}

/// Test App for TUI rendering verification
pub struct TestApp {
    pub state: AppState,
    pub steps: Vec<PipelineStep>,
    pub rtf: f64,
    pub memory_mb: f64,
    pub status_message: String,
    pub audio_duration_ms: f64,
}

impl Default for TestApp {
    fn default() -> Self {
        Self::new()
    }
}

impl TestApp {
    pub fn new() -> Self {
        Self {
            state: AppState::Idle,
            steps: StepId::all().iter().map(|&id| PipelineStep::new(id)).collect(),
            rtf: 0.0,
            memory_mb: 0.0,
            status_message: "Press [s] to start benchmark".to_string(),
            audio_duration_ms: 1000.0,
        }
    }

    pub fn with_running_state() -> Self {
        let mut app = Self::new();
        app.state = AppState::Running;
        app.steps[0].status = StepStatus::Running;
        app.status_message = "Loading model...".to_string();
        app
    }

    pub fn with_completed_state() -> Self {
        let mut app = Self::new();
        app.state = AppState::Completed;
        for step in &mut app.steps {
            step.status = StepStatus::Complete;
            step.elapsed_ms = 100.0;
        }
        app.rtf = 2.5;
        app.memory_mb = 145.0;
        app.status_message = "Complete! RTF: 2.50x".to_string();
        app
    }

    pub fn total_elapsed_ms(&self) -> f64 {
        self.steps.iter().map(|s| s.elapsed_ms).sum()
    }
}

// =============================================================================
// Rendering Functions (matching benchmark_tui.rs)
// =============================================================================

fn render_ui(f: &mut Frame, app: &TestApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(25),
            Constraint::Percentage(15),
            Constraint::Percentage(10),
        ])
        .split(f.area());

    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(chunks[0]);

    render_pipeline_progress(f, top_chunks[0], app);
    render_live_metrics(f, top_chunks[1], app);
    render_status(f, chunks[2], app);
    render_controls(f, chunks[3]);
}

fn render_pipeline_progress(f: &mut Frame, area: Rect, app: &TestApp) {
    let block = Block::default()
        .title(" PIPELINE PROGRESS (REAL) ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let step_height = (inner.height as usize / app.steps.len()).max(1);

    for (i, step) in app.steps.iter().enumerate() {
        if (i * step_height) as u16 >= inner.height {
            break;
        }

        let step_area = Rect {
            x: inner.x,
            y: inner.y + (i * step_height) as u16,
            width: inner.width,
            height: step_height.min(2) as u16,
        };

        let (color, modifier, progress) = match &step.status {
            StepStatus::Pending => (Color::DarkGray, Modifier::empty(), 0),
            StepStatus::Running => (Color::Yellow, Modifier::BOLD, 50),
            StepStatus::Complete => (Color::Green, Modifier::empty(), 100),
        };

        let status_char = match &step.status {
            StepStatus::Pending => ' ',
            StepStatus::Running => '▶',
            StepStatus::Complete => '✓',
        };

        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(color).add_modifier(modifier))
            .percent(progress)
            .label(format!(
                "[{}] {} {:8} ({:>6}) {:>8.1}ms",
                step.id.as_char(),
                status_char,
                step.id.name(),
                step.id.backend().label(),
                step.elapsed_ms,
            ));

        f.render_widget(gauge, step_area);
    }
}

fn render_live_metrics(f: &mut Frame, area: Rect, app: &TestApp) {
    let block = Block::default()
        .title(" LIVE METRICS ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Magenta));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let metrics_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Min(0),
        ])
        .split(inner);

    let rtf_color = if app.rtf < 2.0 {
        Color::Green
    } else if app.rtf < 4.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    let rtf_gauge = Gauge::default()
        .gauge_style(Style::default().fg(rtf_color))
        .percent(((app.rtf / 5.0) * 100.0).min(100.0) as u16)
        .label(format!("RTF: {:.2}x (target: <2.0x)", app.rtf));
    f.render_widget(rtf_gauge, metrics_layout[0]);

    let mem_gauge = Gauge::default()
        .gauge_style(Style::default().fg(Color::Cyan))
        .percent(((app.memory_mb / 200.0) * 100.0).min(100.0) as u16)
        .label(format!("Model: {:.1}MB", app.memory_mb));
    f.render_widget(mem_gauge, metrics_layout[1]);

    let elapsed = Paragraph::new(format!(
        "Elapsed: {:.0}ms / Audio: {:.0}ms",
        app.total_elapsed_ms(),
        app.audio_duration_ms
    ))
    .style(Style::default().fg(Color::White));
    f.render_widget(elapsed, metrics_layout[2]);
}

fn render_status(f: &mut Frame, area: Rect, app: &TestApp) {
    let block = Block::default()
        .title(" STATUS ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));

    let para = Paragraph::new(format!("Status: {}", app.status_message))
        .block(block)
        .style(Style::default().fg(Color::White));

    f.render_widget(para, area);
}

fn render_controls(f: &mut Frame, area: Rect) {
    let controls = Paragraph::new("[s] Start  [r] Reset  [q] Quit")
        .style(Style::default().fg(Color::DarkGray))
        .block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(Color::DarkGray)),
        );

    f.render_widget(controls, area);
}

// =============================================================================
// Helper Functions
// =============================================================================

fn capture_app_frame(app: &TestApp, width: u16, height: u16) -> TuiFrame {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("Failed to create terminal");

    terminal
        .draw(|f| render_ui(f, app))
        .expect("Failed to draw");

    // Convert TestBackend buffer to TuiFrame
    TuiFrame::from_buffer(terminal.backend().buffer(), 0)
}

// =============================================================================
// PROBAR TUI TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Initial State Rendering Tests
    // =========================================================================

    #[test]
    fn test_initial_frame_contains_pipeline_title() {
        let app = TestApp::new();
        let frame = capture_app_frame(&app, 100, 30);

        assert!(
            frame.contains("PIPELINE PROGRESS"),
            "Frame should contain PIPELINE PROGRESS title"
        );
    }

    #[test]
    fn test_initial_frame_contains_metrics_title() {
        let app = TestApp::new();
        let frame = capture_app_frame(&app, 100, 30);

        assert!(
            frame.contains("LIVE METRICS"),
            "Frame should contain LIVE METRICS title"
        );
    }

    #[test]
    fn test_initial_frame_contains_status_title() {
        let app = TestApp::new();
        let frame = capture_app_frame(&app, 100, 30);

        assert!(
            frame.contains("STATUS"),
            "Frame should contain STATUS title"
        );
    }

    #[test]
    fn test_initial_frame_shows_start_message() {
        let app = TestApp::new();
        let frame = capture_app_frame(&app, 100, 30);

        assert!(
            frame.contains("Press [s] to start"),
            "Frame should show start instruction"
        );
    }

    #[test]
    fn test_initial_frame_shows_controls() {
        let app = TestApp::new();
        let frame = capture_app_frame(&app, 100, 30);

        assert!(frame.contains("[s]"), "Frame should show [s] control");
        assert!(frame.contains("[r]"), "Frame should show [r] control");
        assert!(frame.contains("[q]"), "Frame should show [q] control");
    }

    #[test]
    fn test_initial_frame_shows_all_step_names() {
        let app = TestApp::new();
        let frame = capture_app_frame(&app, 100, 30);

        assert!(frame.contains("Model"), "Frame should show Model step");
        assert!(frame.contains("Load"), "Frame should show Load step");
        assert!(frame.contains("Parse"), "Frame should show Parse step");
        assert!(frame.contains("Resample"), "Frame should show Resample step");
        assert!(frame.contains("Mel"), "Frame should show Mel step");
        assert!(frame.contains("Encode"), "Frame should show Encode step");
        assert!(frame.contains("Decode"), "Frame should show Decode step");
    }

    #[test]
    fn test_initial_frame_shows_rtf_metric() {
        let app = TestApp::new();
        let frame = capture_app_frame(&app, 100, 30);

        assert!(
            frame.contains("RTF:"),
            "Frame should show RTF metric"
        );
    }

    // =========================================================================
    // Running State Rendering Tests
    // =========================================================================

    #[test]
    fn test_running_frame_shows_loading_status() {
        let app = TestApp::with_running_state();
        let frame = capture_app_frame(&app, 100, 30);

        assert!(
            frame.contains("Loading model"),
            "Frame should show loading status"
        );
    }

    #[test]
    fn test_running_frame_shows_running_indicator() {
        let app = TestApp::with_running_state();
        let frame = capture_app_frame(&app, 100, 30);

        // Running step shows ▶ indicator
        assert!(
            frame.contains("▶"),
            "Frame should show running indicator (▶)"
        );
    }

    // =========================================================================
    // Completed State Rendering Tests
    // =========================================================================

    #[test]
    fn test_completed_frame_shows_complete_status() {
        let app = TestApp::with_completed_state();
        let frame = capture_app_frame(&app, 100, 30);

        assert!(
            frame.contains("Complete!"),
            "Frame should show completion status"
        );
    }

    #[test]
    fn test_completed_frame_shows_rtf_value() {
        let app = TestApp::with_completed_state();
        let frame = capture_app_frame(&app, 100, 30);

        assert!(
            frame.contains("2.50"),
            "Frame should show RTF value 2.50"
        );
    }

    #[test]
    fn test_completed_frame_shows_checkmarks() {
        let app = TestApp::with_completed_state();
        let frame = capture_app_frame(&app, 100, 30);

        // Completed steps show ✓ indicator
        assert!(
            frame.contains("✓"),
            "Frame should show completion checkmark (✓)"
        );
    }

    #[test]
    fn test_completed_frame_shows_elapsed_times() {
        let app = TestApp::with_completed_state();
        let frame = capture_app_frame(&app, 100, 30);

        // Each step should show 100.0ms elapsed
        assert!(
            frame.contains("100.0"),
            "Frame should show elapsed time"
        );
    }

    // =========================================================================
    // Playwright-style Assertion Tests
    // =========================================================================

    #[test]
    fn test_frame_assertion_contains_text() {
        let app = TestApp::new();
        let frame = capture_app_frame(&app, 100, 30);
        let mut assertion = expect_frame(&frame);

        assertion
            .to_contain_text("PIPELINE")
            .expect("Should contain PIPELINE");
    }

    #[test]
    fn test_frame_assertion_not_contains_error() {
        let app = TestApp::new();
        let frame = capture_app_frame(&app, 100, 30);
        let mut assertion = expect_frame(&frame);

        assertion
            .not_to_contain_text("ERROR")
            .expect("Should not contain ERROR in idle state");
    }

    #[test]
    fn test_frame_assertion_matches_regex() {
        let app = TestApp::new();
        let frame = capture_app_frame(&app, 100, 30);
        let mut assertion = expect_frame(&frame);

        assertion
            .to_match(r"RTF: \d+\.\d+")
            .expect("Should match RTF pattern");
    }

    // =========================================================================
    // Frame Size Tests
    // =========================================================================

    #[test]
    fn test_frame_respects_dimensions() {
        let app = TestApp::new();
        let frame = capture_app_frame(&app, 80, 24);

        assert_eq!(frame.width(), 80, "Frame width should be 80");
        assert_eq!(frame.height(), 24, "Frame height should be 24");
    }

    #[test]
    fn test_frame_renders_in_small_terminal() {
        let app = TestApp::new();
        let frame = capture_app_frame(&app, 60, 20);

        // Should still render without panic
        assert!(frame.height() > 0, "Frame should have content");
    }

    #[test]
    fn test_frame_renders_in_large_terminal() {
        let app = TestApp::new();
        let frame = capture_app_frame(&app, 200, 60);

        // Should still render without panic
        assert!(
            frame.contains("PIPELINE"),
            "Large frame should still contain content"
        );
    }

    // =========================================================================
    // Frame Diff Tests
    // =========================================================================

    #[test]
    fn test_frame_diff_between_states() {
        let idle_app = TestApp::new();
        let running_app = TestApp::with_running_state();

        let idle_frame = capture_app_frame(&idle_app, 100, 30);
        let running_frame = capture_app_frame(&running_app, 100, 30);

        assert!(
            !idle_frame.is_identical(&running_frame),
            "Idle and running frames should be different"
        );
    }

    #[test]
    fn test_frame_identical_for_same_state() {
        let app1 = TestApp::new();
        let app2 = TestApp::new();

        let frame1 = capture_app_frame(&app1, 100, 30);
        let frame2 = capture_app_frame(&app2, 100, 30);

        assert!(
            frame1.is_identical(&frame2),
            "Same state should produce identical frames"
        );
    }

    // =========================================================================
    // Diagnostic Tests (run with --nocapture to see output)
    // =========================================================================

    #[test]
    fn diagnostic_dump_idle_frame() {
        let app = TestApp::new();
        let frame = capture_app_frame(&app, 80, 24);

        println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        println!("║                        IDLE STATE FRAME DUMP                                 ║");
        println!("╠══════════════════════════════════════════════════════════════════════════════╣");
        for (i, line) in frame.lines().iter().enumerate() {
            println!("║ {:2} │ {:<73} ║", i, line);
        }
        println!("╚══════════════════════════════════════════════════════════════════════════════╝");
        println!("\nFrame dimensions: {}x{}", frame.width(), frame.height());
    }

    #[test]
    fn diagnostic_dump_completed_frame() {
        let app = TestApp::with_completed_state();
        let frame = capture_app_frame(&app, 80, 24);

        println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        println!("║                      COMPLETED STATE FRAME DUMP                              ║");
        println!("╠══════════════════════════════════════════════════════════════════════════════╣");
        for (i, line) in frame.lines().iter().enumerate() {
            println!("║ {:2} │ {:<73} ║", i, line);
        }
        println!("╚══════════════════════════════════════════════════════════════════════════════╝");
        println!("\nRTF: {:.2}x | Memory: {:.1}MB | Total: {:.0}ms",
            app.rtf, app.memory_mb, app.total_elapsed_ms());
    }

    #[test]
    fn diagnostic_state_diff() {
        let idle = TestApp::new();
        let completed = TestApp::with_completed_state();

        let idle_frame = capture_app_frame(&idle, 80, 24);
        let completed_frame = capture_app_frame(&completed, 80, 24);

        let diff = idle_frame.diff(&completed_frame);

        println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        println!("║                    STATE DIFF: IDLE → COMPLETED                              ║");
        println!("╠══════════════════════════════════════════════════════════════════════════════╣");
        println!("║ Changed lines: {}                                                            ║", diff.changed_lines.len());
        println!("╠══════════════════════════════════════════════════════════════════════════════╣");
        for line_diff in diff.changed_lines.iter().take(10) {
            println!("║ Line {:2}:                                                                     ║", line_diff.line_number);
            println!("║   - {:<72} ║", truncate(&line_diff.expected, 72));
            println!("║   + {:<72} ║", truncate(&line_diff.actual, 72));
        }
        if diff.changed_lines.len() > 10 {
            println!("║ ... and {} more changed lines                                                 ║", diff.changed_lines.len() - 10);
        }
        println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    }

    #[test]
    fn diagnostic_step_timing_targets() {
        println!("\n╔═══════════════════════════════════════════════════════════════════════════════════╗");
        println!("║                    PIPELINE STEP TIMING TARGETS + BACKENDS                        ║");
        println!("╠═══════════════════════════════════════════════════════════════════════════════════╣");
        println!("║  Step  │  Name     │  Backend │  Target (ms)  │  % of Total                       ║");
        println!("╠═══════════════════════════════════════════════════════════════════════════════════╣");

        let total: u64 = StepId::all().iter().map(|s| s.target_ms()).sum();
        for step in StepId::all() {
            let target = step.target_ms();
            let pct = (target as f64 / total as f64) * 100.0;
            let backend = step.backend();
            println!("║   {}    │ {:8} │ {:>6}   │    {:>5} ms   │  {:>5.1}%                            ║",
                step.as_char(), step.name(), backend.label(), target, pct);
        }
        println!("╠═══════════════════════════════════════════════════════════════════════════════════╣");
        println!("║  TOTAL │          │          │    {:>5} ms   │  100.0%                            ║", total);
        println!("╚═══════════════════════════════════════════════════════════════════════════════════╝");

        // Show backend summary
        println!("\n┌─── Backend Summary (aligned with realizar) ────────────────────────────────────────┐");
        println!("│  SIMD Feature: {}                                                              │",
            if cfg!(feature = "simd") { "ENABLED ✓" } else { "DISABLED ✗" });
        println!("│                                                                                   │");
        println!("│  Code Paths:                                                                      │");
        println!("│    • I/O:    Model load, Audio load (disk bound)                                  │");
        println!("│    • Mem:    PCM parsing (memory bound)                                           │");
        println!("│    • SIMD:   Resample, Mel (short sequences, compute bound)                       │");
        println!("│    • Flash:  Encode, Decode (long sequences >128, O(n) memory)                    │");
        println!("│    • Scalar: Fallback when SIMD disabled                                          │");
        println!("│    • GPU:    Future WebGPU acceleration                                           │");
        println!("└───────────────────────────────────────────────────────────────────────────────────┘");

        println!("\nAmdahl's Law: Decode (H) at {:.1}% dominates optimization potential",
            (StepId::H.target_ms() as f64 / total as f64) * 100.0);
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len.saturating_sub(3)).collect();
        format!("{}...", truncated)
    }
}
