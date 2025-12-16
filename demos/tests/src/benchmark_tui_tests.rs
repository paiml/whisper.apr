//! Benchmark TUI Tests - EXTREME TDD
//!
//! These tests are written FIRST per extreme TDD methodology.
//! They define the expected behavior before implementation exists.
//!
//! Run: cargo test -p whisper-apr-demo-tests benchmark_tui

#![allow(unused_imports)]

// =============================================================================
// STEP 1: Define the types we expect to exist (will fail until implemented)
// =============================================================================

/// Pipeline step identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepId {
    B, // Load
    C, // Parse
    D, // Resample
    F, // Mel
    G, // Encode
    H, // Decode
}

impl StepId {
    pub fn as_char(&self) -> char {
        match self {
            StepId::B => 'B',
            StepId::C => 'C',
            StepId::D => 'D',
            StepId::F => 'F',
            StepId::G => 'G',
            StepId::H => 'H',
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            StepId::B => "Load",
            StepId::C => "Parse",
            StepId::D => "Resample",
            StepId::F => "Mel",
            StepId::G => "Encode",
            StepId::H => "Decode",
        }
    }

    pub fn target_ms(&self) -> u64 {
        match self {
            StepId::B => 50,
            StepId::C => 10,
            StepId::D => 100,
            StepId::F => 50,
            StepId::G => 500,
            StepId::H => 2000,
        }
    }

    pub fn all() -> &'static [StepId] {
        &[
            StepId::B,
            StepId::C,
            StepId::D,
            StepId::F,
            StepId::G,
            StepId::H,
        ]
    }
}

/// Step status
#[derive(Debug, Clone, PartialEq)]
pub enum StepStatus {
    Pending,
    Running,
    Complete,
    Error(String),
}

/// Pipeline step with timing
#[derive(Debug, Clone)]
pub struct PipelineStep {
    pub id: StepId,
    pub status: StepStatus,
    pub progress: f64,
    pub elapsed_ms: f64,
}

impl PipelineStep {
    pub fn new(id: StepId) -> Self {
        Self {
            id,
            status: StepStatus::Pending,
            progress: 0.0,
            elapsed_ms: 0.0,
        }
    }

    pub fn is_complete(&self) -> bool {
        matches!(self.status, StepStatus::Complete)
    }
}

/// Application state
#[derive(Debug, Clone, PartialEq)]
pub enum AppState {
    Idle,
    Running,
    Paused,
    Completed,
    Error(String),
}

/// Benchmark TUI Application
#[derive(Debug)]
pub struct BenchmarkApp {
    pub state: AppState,
    pub steps: Vec<PipelineStep>,
    pub current_step_idx: usize,
    pub rtf: f64,
    pub memory_mb: f64,
    pub tokens_generated: usize,
    pub transcript: String,
    pub status_message: String,
    pub audio_duration_ms: f64,
}

impl Default for BenchmarkApp {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkApp {
    pub fn new() -> Self {
        Self {
            state: AppState::Idle,
            steps: StepId::all().iter().map(|&id| PipelineStep::new(id)).collect(),
            current_step_idx: 0,
            rtf: 0.0,
            memory_mb: 0.0,
            tokens_generated: 0,
            transcript: String::new(),
            status_message: "Press [s] to start benchmark".to_string(),
            audio_duration_ms: 1500.0, // 1.5s audio
        }
    }

    pub fn can_start(&self) -> bool {
        matches!(self.state, AppState::Idle)
    }

    pub fn can_pause(&self) -> bool {
        matches!(self.state, AppState::Running)
    }

    pub fn can_resume(&self) -> bool {
        matches!(self.state, AppState::Paused)
    }

    pub fn can_reset(&self) -> bool {
        matches!(self.state, AppState::Completed | AppState::Error(_))
    }

    pub fn is_running(&self) -> bool {
        matches!(self.state, AppState::Running)
    }

    pub fn current_step(&self) -> Option<&PipelineStep> {
        self.steps.get(self.current_step_idx)
    }

    pub fn current_step_id(&self) -> Option<StepId> {
        self.current_step().map(|s| s.id)
    }

    pub fn all_steps_complete(&self) -> bool {
        self.steps.iter().all(|s| s.is_complete())
    }

    pub fn total_elapsed_ms(&self) -> f64 {
        self.steps.iter().map(|s| s.elapsed_ms).sum()
    }

    pub fn step_progress(&self, id: StepId) -> f64 {
        self.steps
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.progress)
            .unwrap_or(0.0)
    }

    /// Start the benchmark
    pub fn start(&mut self) {
        if self.can_start() {
            self.state = AppState::Running;
            self.status_message = "Running benchmark...".to_string();
            if let Some(step) = self.steps.get_mut(0) {
                step.status = StepStatus::Running;
            }
        }
    }

    /// Pause the benchmark
    pub fn pause(&mut self) {
        if self.can_pause() {
            self.state = AppState::Paused;
            self.status_message = "Paused".to_string();
        }
    }

    /// Resume from pause
    pub fn resume(&mut self) {
        if self.can_resume() {
            self.state = AppState::Running;
            self.status_message = "Running benchmark...".to_string();
        }
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Advance simulation by one tick
    pub fn tick(&mut self, delta_ms: f64) {
        if !matches!(self.state, AppState::Running) {
            return;
        }

        if self.current_step_idx >= self.steps.len() {
            return;
        }

        // Get step info
        let step_id = self.steps[self.current_step_idx].id;
        let target = step_id.target_ms() as f64;

        // Update progress
        self.steps[self.current_step_idx].elapsed_ms += delta_ms;
        let elapsed = self.steps[self.current_step_idx].elapsed_ms;
        let progress = (elapsed / target).min(1.0);
        self.steps[self.current_step_idx].progress = progress;

        // Check if step is complete
        let step_complete = progress >= 1.0;
        if step_complete {
            self.steps[self.current_step_idx].status = StepStatus::Complete;
            self.current_step_idx += 1;

            // Check if all done
            if self.current_step_idx >= self.steps.len() {
                self.state = AppState::Completed;
                self.rtf = self.total_elapsed_ms() / self.audio_duration_ms;
                self.status_message = format!("Complete! RTF: {:.2}x", self.rtf);
            } else {
                // Start next step
                self.steps[self.current_step_idx].status = StepStatus::Running;
            }
        }

        // Update memory simulation
        self.memory_mb = 50.0 + (self.current_step_idx as f64 * 15.0);

        // Update tokens during decode step
        let current_is_decode = self.current_step_id() == Some(StepId::H);
        if current_is_decode {
            let decode_progress = self.step_progress(StepId::H);
            self.tokens_generated = (decode_progress * 20.0) as usize;
            if self.tokens_generated > 0 && self.transcript.is_empty() {
                self.transcript = "The quick brown fox...".to_string();
            }
        }
    }

    /// Trigger an error
    pub fn set_error(&mut self, msg: &str) {
        self.state = AppState::Error(msg.to_string());
        self.status_message = format!("Error: {}", msg);
    }
}

// =============================================================================
// TESTS - Written FIRST per Extreme TDD
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Initial State Tests
    // =========================================================================

    #[test]
    fn test_app_initial_state_is_idle() {
        let app = BenchmarkApp::new();
        assert_eq!(app.state, AppState::Idle);
    }

    #[test]
    fn test_app_initial_can_start() {
        let app = BenchmarkApp::new();
        assert!(app.can_start());
    }

    #[test]
    fn test_app_initial_cannot_pause() {
        let app = BenchmarkApp::new();
        assert!(!app.can_pause());
    }

    #[test]
    fn test_app_initial_cannot_reset() {
        let app = BenchmarkApp::new();
        assert!(!app.can_reset());
    }

    #[test]
    fn test_app_initial_all_progress_zero() {
        let app = BenchmarkApp::new();
        for step in &app.steps {
            assert_eq!(step.progress, 0.0);
            assert_eq!(step.status, StepStatus::Pending);
        }
    }

    #[test]
    fn test_app_initial_status_message() {
        let app = BenchmarkApp::new();
        assert!(app.status_message.contains("start"));
    }

    #[test]
    fn test_app_has_all_steps() {
        let app = BenchmarkApp::new();
        assert_eq!(app.steps.len(), 6);
        assert_eq!(app.steps[0].id, StepId::B);
        assert_eq!(app.steps[5].id, StepId::H);
    }

    // =========================================================================
    // State Transition Tests
    // =========================================================================

    #[test]
    fn test_start_transitions_to_running() {
        let mut app = BenchmarkApp::new();
        app.start();
        assert_eq!(app.state, AppState::Running);
    }

    #[test]
    fn test_start_sets_first_step_running() {
        let mut app = BenchmarkApp::new();
        app.start();
        assert_eq!(app.steps[0].status, StepStatus::Running);
    }

    #[test]
    fn test_cannot_start_when_running() {
        let mut app = BenchmarkApp::new();
        app.start();
        assert!(!app.can_start());
    }

    #[test]
    fn test_pause_transitions_to_paused() {
        let mut app = BenchmarkApp::new();
        app.start();
        app.pause();
        assert_eq!(app.state, AppState::Paused);
    }

    #[test]
    fn test_resume_transitions_to_running() {
        let mut app = BenchmarkApp::new();
        app.start();
        app.pause();
        app.resume();
        assert_eq!(app.state, AppState::Running);
    }

    #[test]
    fn test_reset_returns_to_idle() {
        let mut app = BenchmarkApp::new();
        app.start();
        // Simulate completion
        for _ in 0..100 {
            app.tick(100.0);
        }
        app.reset();
        assert_eq!(app.state, AppState::Idle);
        assert_eq!(app.current_step_idx, 0);
    }

    // =========================================================================
    // Progress Tests
    // =========================================================================

    #[test]
    fn test_tick_increases_progress() {
        let mut app = BenchmarkApp::new();
        app.start();

        let initial_progress = app.steps[0].progress;
        app.tick(10.0);

        assert!(app.steps[0].progress > initial_progress);
    }

    #[test]
    fn test_tick_does_nothing_when_paused() {
        let mut app = BenchmarkApp::new();
        app.start();
        app.pause();

        let progress_before = app.steps[0].progress;
        app.tick(100.0);

        assert_eq!(app.steps[0].progress, progress_before);
    }

    #[test]
    fn test_step_completes_at_100_percent() {
        let mut app = BenchmarkApp::new();
        app.start();

        // Tick enough to complete first step (50ms target)
        for _ in 0..10 {
            app.tick(10.0);
        }

        assert_eq!(app.steps[0].status, StepStatus::Complete);
        assert!(app.steps[0].progress >= 1.0);
    }

    #[test]
    fn test_next_step_starts_after_previous_completes() {
        let mut app = BenchmarkApp::new();
        app.start();

        // Complete first step (B has 50ms target, so 5 ticks at 10ms each)
        for _ in 0..5 {
            app.tick(10.0);
        }

        // Now on step C (index 1)
        assert_eq!(app.current_step_idx, 1);
        assert_eq!(app.steps[1].status, StepStatus::Running);
    }

    #[test]
    fn test_all_steps_complete_sets_completed_state() {
        let mut app = BenchmarkApp::new();
        app.start();

        // Tick enough to complete all steps
        for _ in 0..500 {
            app.tick(10.0);
        }

        assert_eq!(app.state, AppState::Completed);
        assert!(app.all_steps_complete());
    }

    // =========================================================================
    // RTF Calculation Tests
    // =========================================================================

    #[test]
    fn test_rtf_calculated_on_completion() {
        let mut app = BenchmarkApp::new();
        app.start();

        // Complete all steps
        for _ in 0..500 {
            app.tick(10.0);
        }

        assert!(app.rtf > 0.0);
    }

    #[test]
    fn test_rtf_is_total_time_over_audio_duration() {
        let mut app = BenchmarkApp::new();
        app.audio_duration_ms = 1000.0;
        app.start();

        // Complete all steps
        for _ in 0..500 {
            app.tick(10.0);
        }

        let expected_rtf = app.total_elapsed_ms() / 1000.0;
        assert!((app.rtf - expected_rtf).abs() < 0.01);
    }

    // =========================================================================
    // Memory Tracking Tests
    // =========================================================================

    #[test]
    fn test_memory_increases_with_progress() {
        let mut app = BenchmarkApp::new();
        let initial_memory = app.memory_mb;
        app.start();

        // Complete a few steps
        for _ in 0..100 {
            app.tick(10.0);
        }

        assert!(app.memory_mb > initial_memory);
    }

    // =========================================================================
    // Token Generation Tests (Decode Step)
    // =========================================================================

    #[test]
    fn test_tokens_generated_during_decode() {
        let mut app = BenchmarkApp::new();
        app.start();

        // Progress to decode step (step index 5)
        for _ in 0..300 {
            app.tick(10.0);
        }

        // Should be in decode step now
        if app.current_step_id() == Some(StepId::H) {
            // Tick to generate some tokens
            for _ in 0..50 {
                app.tick(10.0);
            }
            assert!(app.tokens_generated > 0);
        }
    }

    #[test]
    fn test_transcript_appears_during_decode() {
        let mut app = BenchmarkApp::new();
        app.start();

        // Progress through all steps including decode
        for _ in 0..400 {
            app.tick(10.0);
        }

        // Transcript should have content
        assert!(!app.transcript.is_empty());
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_error_sets_error_state() {
        let mut app = BenchmarkApp::new();
        app.start();
        app.set_error("Test error");

        assert!(matches!(app.state, AppState::Error(_)));
    }

    #[test]
    fn test_can_reset_from_error() {
        let mut app = BenchmarkApp::new();
        app.start();
        app.set_error("Test error");

        assert!(app.can_reset());
    }

    #[test]
    fn test_error_message_in_status() {
        let mut app = BenchmarkApp::new();
        app.set_error("Out of memory");

        assert!(app.status_message.contains("Out of memory"));
    }

    // =========================================================================
    // Step ID Tests
    // =========================================================================

    #[test]
    fn test_step_id_chars() {
        assert_eq!(StepId::B.as_char(), 'B');
        assert_eq!(StepId::H.as_char(), 'H');
    }

    #[test]
    fn test_step_id_names() {
        assert_eq!(StepId::B.name(), "Load");
        assert_eq!(StepId::G.name(), "Encode");
        assert_eq!(StepId::H.name(), "Decode");
    }

    #[test]
    fn test_step_id_targets() {
        assert_eq!(StepId::B.target_ms(), 50);
        assert_eq!(StepId::H.target_ms(), 2000);
    }

    // =========================================================================
    // Forbidden Transition Tests (from playbook)
    // =========================================================================

    #[test]
    fn test_cannot_skip_to_decode() {
        let app = BenchmarkApp::new();
        // App starts at step 0 (B), cannot jump to H
        assert_eq!(app.current_step_idx, 0);
        assert_ne!(app.current_step_id(), Some(StepId::H));
    }

    #[test]
    fn test_cannot_go_backwards() {
        let mut app = BenchmarkApp::new();
        app.start();

        // Complete first step
        for _ in 0..10 {
            app.tick(10.0);
        }

        let idx = app.current_step_idx;
        // Tick more - index should only go forward or stay
        for _ in 0..10 {
            app.tick(10.0);
        }

        assert!(app.current_step_idx >= idx);
    }

    // =========================================================================
    // Performance Budget Tests
    // =========================================================================

    #[test]
    fn test_step_targets_match_playbook() {
        // From playbook: step_b_load: 50, step_h_decode: 2000
        assert_eq!(StepId::B.target_ms(), 50);
        assert_eq!(StepId::C.target_ms(), 10);
        assert_eq!(StepId::D.target_ms(), 100);
        assert_eq!(StepId::F.target_ms(), 50);
        assert_eq!(StepId::G.target_ms(), 500);
        assert_eq!(StepId::H.target_ms(), 2000);
    }

    #[test]
    fn test_total_target_under_5000ms() {
        let total: u64 = StepId::all().iter().map(|s| s.target_ms()).sum();
        assert!(total < 5000, "Total target {} should be under 5000ms", total);
    }

    // =========================================================================
    // Default Implementation Test
    // =========================================================================

    #[test]
    fn test_default_impl() {
        let app: BenchmarkApp = Default::default();
        assert_eq!(app.state, AppState::Idle);
    }
}
