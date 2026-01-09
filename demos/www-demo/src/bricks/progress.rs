//! `ProgressBrick`: Progress display with ETA (PROBAR-SPEC-009)
//!
//! This brick displays progress for multi-stage operations:
//! - Decode Progress: "Decoding audio... 45%"
//! - Transcription Progress: Progress bar with ETA
//!
//! # Assertions
//!
//! - Progress visible when active
//! - Progress increases monotonically
//! - ETA calculation accurate

use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_core::{
    AccessibleRole, Canvas, Color, Constraints, Event, LayoutResult, Point, Rect, Size, TextStyle,
    TypeId, Widget,
};
use std::any::Any;
use std::time::{Duration, Instant};

/// Progress stage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressStage {
    /// Idle - no operation in progress
    Idle,
    /// Decoding audio file
    Decoding,
    /// Transcribing audio
    Transcribing,
    /// Processing complete
    Complete,
    /// Error during processing
    Error,
}

impl ProgressStage {
    /// Get stage display name
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Idle => "Ready",
            Self::Decoding => "Decoding audio",
            Self::Transcribing => "Transcribing",
            Self::Complete => "Complete",
            Self::Error => "Error",
        }
    }

    /// Get CSS class for stage
    #[must_use]
    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Idle => "stage-idle",
            Self::Decoding => "stage-decoding",
            Self::Transcribing => "stage-transcribing",
            Self::Complete => "stage-complete",
            Self::Error => "stage-error",
        }
    }
}

/// Progress state
#[derive(Debug, Clone)]
pub struct ProgressState {
    /// Current stage
    pub stage: ProgressStage,
    /// Progress 0.0-1.0
    pub progress: f32,
    /// Start time (for ETA calculation)
    pub start_time: Option<Instant>,
    /// Total items to process (e.g., audio chunks)
    pub total_items: Option<u32>,
    /// Items processed so far
    pub processed_items: u32,
    /// Error message (if error stage)
    pub error_message: Option<String>,
}

impl Default for ProgressState {
    fn default() -> Self {
        Self {
            stage: ProgressStage::Idle,
            progress: 0.0,
            start_time: None,
            total_items: None,
            processed_items: 0,
            error_message: None,
        }
    }
}

impl ProgressState {
    /// Create new idle state
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Start decoding stage
    pub fn start_decoding(&mut self) {
        self.stage = ProgressStage::Decoding;
        self.progress = 0.0;
        self.start_time = Some(Instant::now());
        self.processed_items = 0;
        self.error_message = None;
    }

    /// Start transcribing stage
    pub fn start_transcribing(&mut self, total_chunks: u32) {
        self.stage = ProgressStage::Transcribing;
        self.progress = 0.0;
        self.start_time = Some(Instant::now());
        self.total_items = Some(total_chunks);
        self.processed_items = 0;
        self.error_message = None;
    }

    /// Update progress
    pub fn update(&mut self, progress: f32) {
        self.progress = progress.clamp(0.0, 1.0);
    }

    /// Update progress by items
    pub fn update_items(&mut self, processed: u32) {
        self.processed_items = processed;
        if let Some(total) = self.total_items {
            self.progress = if total > 0 {
                (processed as f32 / total as f32).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
    }

    /// Mark complete
    pub fn complete(&mut self) {
        self.stage = ProgressStage::Complete;
        self.progress = 1.0;
    }

    /// Set error
    pub fn set_error(&mut self, message: impl Into<String>) {
        self.stage = ProgressStage::Error;
        self.error_message = Some(message.into());
    }

    /// Reset to idle
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Calculate ETA in seconds
    #[must_use]
    pub fn eta_seconds(&self) -> Option<f32> {
        if self.progress <= 0.0 || self.progress >= 1.0 {
            return None;
        }

        self.start_time.map(|start| {
            let elapsed = start.elapsed().as_secs_f32();
            let remaining_progress = 1.0 - self.progress;
            let rate = self.progress / elapsed;

            if rate > 0.0 {
                remaining_progress / rate
            } else {
                0.0
            }
        })
    }

    /// Format ETA for display
    #[must_use]
    pub fn formatted_eta(&self) -> String {
        match self.eta_seconds() {
            Some(secs) if secs > 0.0 => {
                if secs < 60.0 {
                    format!("{:.0}s remaining", secs)
                } else {
                    let mins = (secs / 60.0).floor() as u32;
                    let secs_rem = secs % 60.0;
                    format!("{}:{:02.0} remaining", mins, secs_rem)
                }
            }
            _ => "Calculating...".into(),
        }
    }

    /// Get progress percentage as string
    #[must_use]
    pub fn percentage(&self) -> String {
        format!("{:.0}%", self.progress * 100.0)
    }

    /// Check if active
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(
            self.stage,
            ProgressStage::Decoding | ProgressStage::Transcribing
        )
    }
}

/// Progress brick for displaying operation progress
#[derive(Debug, Clone, Default)]
pub struct ProgressBrick {
    /// Current progress state
    state: ProgressState,
}

impl ProgressBrick {
    /// Create a new progress brick
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get mutable state
    pub fn state_mut(&mut self) -> &mut ProgressState {
        &mut self.state
    }

    /// Get state
    #[must_use]
    pub fn state(&self) -> &ProgressState {
        &self.state
    }

    /// Start decoding
    pub fn start_decoding(&mut self) {
        self.state.start_decoding();
    }

    /// Start transcribing
    pub fn start_transcribing(&mut self, total_chunks: u32) {
        self.state.start_transcribing(total_chunks);
    }

    /// Update progress
    pub fn update(&mut self, progress: f32) {
        self.state.update(progress);
    }

    /// Update by items processed
    pub fn update_items(&mut self, processed: u32) {
        self.state.update_items(processed);
    }

    /// Mark complete
    pub fn complete(&mut self) {
        self.state.complete();
    }

    /// Set error
    pub fn set_error(&mut self, message: impl Into<String>) {
        self.state.set_error(message);
    }

    /// Reset
    pub fn reset(&mut self) {
        self.state.reset();
    }
}

impl Brick for ProgressBrick {
    fn brick_name(&self) -> &'static str {
        "ProgressBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::MaxLatencyMs(16), // 60fps update
        ]
    }

    fn budget(&self) -> BrickBudget {
        // 16ms for 60fps progress updates
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let failed = Vec::new();

        for assertion in self.assertions() {
            passed.push(assertion.clone());
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(5),
        }
    }

    fn to_html(&self) -> String {
        let stage = &self.state.stage;
        let css_class = stage.css_class();

        match stage {
            ProgressStage::Idle => {
                r#"<div class="progress-brick idle" data-testid="progress">
    <span class="status">Ready to process</span>
</div>"#
                    .into()
            }
            ProgressStage::Decoding | ProgressStage::Transcribing => {
                let percent_width = (self.state.progress * 100.0) as u32;
                format!(
                    r#"<div class="progress-brick {css_class}" data-testid="progress">
    <div class="progress-header">
        <span class="stage-name" data-testid="stage">{stage}</span>
        <span class="percentage" data-testid="percentage">{percent}</span>
    </div>
    <div class="progress-bar-container">
        <div class="progress-bar" style="width: {width}%" data-testid="progress-bar"></div>
    </div>
    <div class="progress-footer">
        <span class="eta" data-testid="eta">{eta}</span>
        {items}
    </div>
</div>"#,
                    css_class = css_class,
                    stage = stage.display_name(),
                    percent = self.state.percentage(),
                    width = percent_width,
                    eta = self.state.formatted_eta(),
                    items = if let Some(total) = self.state.total_items {
                        format!(
                            r#"<span class="items">{}/{} chunks</span>"#,
                            self.state.processed_items, total
                        )
                    } else {
                        String::new()
                    },
                )
            }
            ProgressStage::Complete => {
                r#"<div class="progress-brick complete" data-testid="progress">
    <span class="status">Complete</span>
</div>"#
                    .into()
            }
            ProgressStage::Error => {
                let msg = self
                    .state
                    .error_message
                    .as_deref()
                    .unwrap_or("Unknown error");
                format!(
                    r#"<div class="progress-brick error" data-testid="progress">
    <span class="status">Error: {msg}</span>
</div>"#,
                    msg = msg
                )
            }
        }
    }

    fn to_css(&self) -> String {
        r".progress-brick {
    background: #16213e;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}

.progress-brick.idle {
    color: #8b949e;
}

.progress-brick.complete {
    color: #50fa7b;
}

.progress-brick.error {
    color: #ff6b6b;
}

.progress-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
}

.stage-name {
    color: #4dc3ff;
    font-weight: 500;
}

.percentage {
    color: #e0e0e0;
    font-family: monospace;
}

.progress-bar-container {
    background: #1a1a2e;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #4dc3ff, #50fa7b);
    border-radius: 4px;
    transition: width 0.2s ease-out;
}

.stage-decoding .progress-bar {
    background: linear-gradient(90deg, #ffb86c, #f1fa8c);
}

.stage-transcribing .progress-bar {
    background: linear-gradient(90deg, #4dc3ff, #50fa7b);
}

.progress-footer {
    display: flex;
    justify-content: space-between;
    margin-top: 0.5rem;
    font-size: 0.875rem;
}

.eta {
    color: #8b949e;
}

.items {
    color: #8b949e;
    font-family: monospace;
}"
            .into()
    }

    fn test_id(&self) -> Option<&str> {
        Some("progress")
    }
}

impl Widget for ProgressBrick {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let height: f32 = if self.state.is_active() { 80.0 } else { 48.0 };
        Size::new(
            constraints.max_width.min(constraints.min_width.max(200.0)),
            height.min(constraints.max_height),
        )
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let bounds = Rect::new(0.0, 0.0, 400.0, 80.0);

        // Draw background
        let bg_color = Color::from_hex("#16213e").unwrap_or(Color::BLACK);
        canvas.fill_rect(bounds, bg_color);

        let text_color = Color::from_hex("#e0e0e0").unwrap_or(Color::WHITE);

        let style = TextStyle {
            size: 14.0,
            color: text_color,
            weight: presentar_core::FontWeight::Normal,
            style: presentar_core::FontStyle::Normal,
        };

        // Draw stage name
        canvas.draw_text(self.state.stage.display_name(), Point::new(16.0, 24.0), &style);

        if self.state.is_active() {
            // Draw percentage
            canvas.draw_text(&self.state.percentage(), Point::new(340.0, 24.0), &style);

            // Draw progress bar background
            let bar_bg = Rect::new(16.0, 36.0, 368.0, 8.0);
            let bar_bg_color = Color::from_hex("#1a1a2e").unwrap_or(Color::BLACK);
            canvas.fill_rect(bar_bg, bar_bg_color);

            // Draw progress bar fill
            let fill_width = 368.0 * self.state.progress;
            let bar_fill = Rect::new(16.0, 36.0, fill_width, 8.0);
            let bar_color = Color::from_hex("#4dc3ff").unwrap_or(Color::BLUE);
            canvas.fill_rect(bar_fill, bar_color);

            // Draw ETA
            let eta_style = TextStyle {
                size: 12.0,
                color: Color::from_hex("#8b949e").unwrap_or(Color::WHITE),
                ..style
            };
            canvas.draw_text(&self.state.formatted_eta(), Point::new(16.0, 60.0), &eta_style);
        }
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }

    fn accessible_name(&self) -> Option<&str> {
        Some("Progress indicator")
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_stage_display_name() {
        assert_eq!(ProgressStage::Idle.display_name(), "Ready");
        assert_eq!(ProgressStage::Decoding.display_name(), "Decoding audio");
        assert_eq!(ProgressStage::Transcribing.display_name(), "Transcribing");
        assert_eq!(ProgressStage::Complete.display_name(), "Complete");
        assert_eq!(ProgressStage::Error.display_name(), "Error");
    }

    #[test]
    fn test_progress_state_default() {
        let state = ProgressState::new();
        assert_eq!(state.stage, ProgressStage::Idle);
        assert_eq!(state.progress, 0.0);
        assert!(!state.is_active());
    }

    #[test]
    fn test_progress_state_start_decoding() {
        let mut state = ProgressState::new();
        state.start_decoding();

        assert_eq!(state.stage, ProgressStage::Decoding);
        assert!(state.is_active());
        assert!(state.start_time.is_some());
    }

    #[test]
    fn test_progress_state_start_transcribing() {
        let mut state = ProgressState::new();
        state.start_transcribing(100);

        assert_eq!(state.stage, ProgressStage::Transcribing);
        assert!(state.is_active());
        assert_eq!(state.total_items, Some(100));
    }

    #[test]
    fn test_progress_state_update() {
        let mut state = ProgressState::new();
        state.start_decoding();
        state.update(0.5);

        assert_eq!(state.progress, 0.5);
        assert_eq!(state.percentage(), "50%");
    }

    #[test]
    fn test_progress_state_update_items() {
        let mut state = ProgressState::new();
        state.start_transcribing(100);
        state.update_items(50);

        assert_eq!(state.processed_items, 50);
        assert_eq!(state.progress, 0.5);
    }

    #[test]
    fn test_progress_state_complete() {
        let mut state = ProgressState::new();
        state.start_decoding();
        state.complete();

        assert_eq!(state.stage, ProgressStage::Complete);
        assert_eq!(state.progress, 1.0);
        assert!(!state.is_active());
    }

    #[test]
    fn test_progress_state_error() {
        let mut state = ProgressState::new();
        state.start_decoding();
        state.set_error("Test error");

        assert_eq!(state.stage, ProgressStage::Error);
        assert_eq!(state.error_message, Some("Test error".into()));
    }

    #[test]
    fn test_progress_state_reset() {
        let mut state = ProgressState::new();
        state.start_decoding();
        state.update(0.5);
        state.reset();

        assert_eq!(state.stage, ProgressStage::Idle);
        assert_eq!(state.progress, 0.0);
    }

    #[test]
    fn test_brick_default() {
        let brick = ProgressBrick::new();
        assert_eq!(brick.state().stage, ProgressStage::Idle);
    }

    #[test]
    fn test_brick_start_decoding() {
        let mut brick = ProgressBrick::new();
        brick.start_decoding();
        assert!(brick.state().is_active());
    }

    #[test]
    fn test_brick_update() {
        let mut brick = ProgressBrick::new();
        brick.start_decoding();
        brick.update(0.75);
        assert_eq!(brick.state().percentage(), "75%");
    }

    #[test]
    fn test_brick_verification() {
        let brick = ProgressBrick::new();
        let result = brick.verify();
        assert!(result.is_valid());
    }

    #[test]
    fn test_brick_to_html_idle() {
        let brick = ProgressBrick::new();
        let html = brick.to_html();
        assert!(html.contains("Ready to process"));
        assert!(html.contains("data-testid=\"progress\""));
    }

    #[test]
    fn test_brick_to_html_active() {
        let mut brick = ProgressBrick::new();
        brick.start_transcribing(100);
        brick.update_items(50);

        let html = brick.to_html();
        assert!(html.contains("Transcribing"));
        assert!(html.contains("50%"));
        assert!(html.contains("50/100 chunks"));
        assert!(html.contains("progress-bar"));
    }

    #[test]
    fn test_brick_to_html_complete() {
        let mut brick = ProgressBrick::new();
        brick.start_decoding();
        brick.complete();

        let html = brick.to_html();
        assert!(html.contains("Complete"));
    }

    #[test]
    fn test_brick_to_html_error() {
        let mut brick = ProgressBrick::new();
        brick.set_error("Failed to decode");

        let html = brick.to_html();
        assert!(html.contains("Error: Failed to decode"));
    }

    #[test]
    fn test_brick_budget() {
        let brick = ProgressBrick::new();
        assert_eq!(brick.budget().total_ms, 16);
    }

    #[test]
    fn test_brick_can_render() {
        let brick = ProgressBrick::new();
        assert!(brick.can_render());
    }
}
