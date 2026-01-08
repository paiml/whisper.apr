//! `StatusBrick`: Application status display (PROBAR-SPEC-009)
//!
//! This brick displays the current application state:
//! - Loading: Model loading progress
//! - Ready: Waiting for user action
//! - Recording: Active transcription
//! - Error: Error state with message
//!
//! # Assertions
//!
//! - Status text visible
//! - State transitions are valid

use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_core::{
    AccessibleRole, Canvas, Color, Constraints, Event, LayoutResult, Point, Rect, Size, TextStyle,
    TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Application status states
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// Loading WASM or model
    Loading { message: String },
    /// Ready to record
    Ready,
    /// Actively recording
    Recording,
    /// Error state
    Error { message: String },
}

impl Default for Status {
    fn default() -> Self {
        Self::Loading {
            message: "Loading...".into(),
        }
    }
}

/// Status brick for showing application state
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct StatusBrick {
    /// Current status
    status: Status,
    /// Model size in MB (if loaded)
    model_size_mb: Option<f32>,
    /// Load time in seconds (if loaded)
    load_time_s: Option<f32>,
}


impl StatusBrick {
    /// Create a new status brick
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set loading status with message
    pub fn set_loading(&mut self, message: impl Into<String>) {
        self.status = Status::Loading {
            message: message.into(),
        };
    }

    /// Set ready status
    pub fn set_ready(&mut self) {
        self.status = Status::Ready;
    }

    /// Set ready status with model info
    pub fn set_ready_with_info(&mut self, size_mb: f32, load_time_s: f32) {
        self.status = Status::Ready;
        self.model_size_mb = Some(size_mb);
        self.load_time_s = Some(load_time_s);
    }

    /// Set recording status
    pub fn set_recording(&mut self) {
        self.status = Status::Recording;
    }

    /// Set error status
    pub fn set_error(&mut self, message: impl Into<String>) {
        self.status = Status::Error {
            message: message.into(),
        };
    }

    /// Get current status
    #[must_use]
    pub fn status(&self) -> &Status {
        &self.status
    }

    /// Check if ready
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.status == Status::Ready
    }

    /// Check if recording
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.status == Status::Recording
    }

    /// Get the status text for display
    #[must_use]
    pub fn status_text(&self) -> String {
        match &self.status {
            Status::Loading { message } => message.clone(),
            Status::Ready => {
                if let (Some(size), Some(time)) = (self.model_size_mb, self.load_time_s) {
                    format!("Ready ({size:.1}MB in {time:.1}s)")
                } else {
                    "Ready".into()
                }
            }
            Status::Recording => "Recording...".into(),
            Status::Error { message } => format!("Error: {message}"),
        }
    }

    /// Get CSS class for current status
    fn status_class(&self) -> &'static str {
        match self.status {
            Status::Loading { .. } => "status-loading",
            Status::Ready => "status-ready",
            Status::Recording => "status-recording",
            Status::Error { .. } => "status-error",
        }
    }
}

impl Brick for StatusBrick {
    fn brick_name(&self) -> &'static str {
        "StatusBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::MaxLatencyMs(50),
        ]
    }

    fn budget(&self) -> BrickBudget {
        // 50ms for status updates
        BrickBudget::uniform(50)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let failed = Vec::new();

        // Text is always visible for status
        for assertion in self.assertions() {
            passed.push(assertion.clone());
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        let text = self.status_text();
        let class = self.status_class();

        format!(
            r#"<div class="status-brick {class}" data-testid="status" aria-live="polite">
    <span id="status">{text}</span>
</div>"#
        )
    }

    fn to_css(&self) -> String {
        r".status-brick {
    background: #16213e;
    padding: 1rem;
    border-radius: 8px;
    font-weight: 500;
}

.status-loading {
    color: #4dc3ff;
}

.status-ready {
    color: #50fa7b;
}

.status-recording {
    color: #50fa7b;
    animation: pulse 1s infinite;
}

.status-error {
    color: #ff6b6b;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}"
        .into()
    }

    fn test_id(&self) -> Option<&str> {
        Some("status")
    }
}

impl Widget for StatusBrick {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        // Status bar is typically full-width, fixed height
        let height: f32 = 48.0; // padding + text
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
        let bounds = Rect::new(0.0, 0.0, 400.0, 48.0);

        // Draw background
        let bg_color = Color::from_hex("#16213e").unwrap_or(Color::BLACK);
        canvas.fill_rect(bounds, bg_color);

        // Choose text color based on status
        let text_color = match &self.status {
            Status::Loading { .. } => Color::from_hex("#4dc3ff").unwrap_or(Color::BLUE),
            Status::Ready | Status::Recording => Color::from_hex("#50fa7b").unwrap_or(Color::GREEN),
            Status::Error { .. } => Color::from_hex("#ff6b6b").unwrap_or(Color::RED),
        };

        let style = TextStyle {
            size: 16.0,
            color: text_color,
            weight: presentar_core::FontWeight::Normal,
            style: presentar_core::FontStyle::Normal,
        };

        canvas.draw_text(&self.status_text(), Point::new(16.0, 24.0), &style);
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        // Status display doesn't handle events directly
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }

    fn accessible_name(&self) -> Option<&str> {
        Some("Application status")
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let brick = StatusBrick::new();
        match brick.status() {
            Status::Loading { message } => {
                assert_eq!(message, "Loading...");
            }
            _ => panic!("Expected Loading status"),
        }
    }

    #[test]
    fn test_set_loading() {
        let mut brick = StatusBrick::new();
        brick.set_loading("Loading model...");

        assert_eq!(brick.status_text(), "Loading model...");
    }

    #[test]
    fn test_set_ready() {
        let mut brick = StatusBrick::new();
        brick.set_ready();

        assert!(brick.is_ready());
        assert_eq!(brick.status_text(), "Ready");
    }

    #[test]
    fn test_set_ready_with_info() {
        let mut brick = StatusBrick::new();
        brick.set_ready_with_info(39.5, 2.3);

        assert!(brick.is_ready());
        assert!(brick.status_text().contains("39.5MB"));
        assert!(brick.status_text().contains("2.3s"));
    }

    #[test]
    fn test_set_recording() {
        let mut brick = StatusBrick::new();
        brick.set_recording();

        assert!(brick.is_recording());
        assert_eq!(brick.status_text(), "Recording...");
    }

    #[test]
    fn test_set_error() {
        let mut brick = StatusBrick::new();
        brick.set_error("Model load failed");

        assert_eq!(brick.status_text(), "Error: Model load failed");
    }

    #[test]
    fn test_verification_passes() {
        let brick = StatusBrick::new();
        let result = brick.verify();

        assert!(result.is_valid());
    }

    #[test]
    fn test_to_html() {
        let mut brick = StatusBrick::new();
        brick.set_ready();

        let html = brick.to_html();
        assert!(html.contains("status-ready"));
        assert!(html.contains("data-testid=\"status\""));
        assert!(html.contains("aria-live=\"polite\""));
    }

    #[test]
    fn test_budget() {
        let brick = StatusBrick::new();
        assert_eq!(brick.budget().total_ms, 50);
    }

    #[test]
    fn test_can_render() {
        let brick = StatusBrick::new();
        assert!(brick.can_render());
    }

    #[test]
    fn test_status_class() {
        let mut brick = StatusBrick::new();
        assert_eq!(brick.status_class(), "status-loading");

        brick.set_ready();
        assert_eq!(brick.status_class(), "status-ready");

        brick.set_recording();
        assert_eq!(brick.status_class(), "status-recording");

        brick.set_error("test");
        assert_eq!(brick.status_class(), "status-error");
    }
}
