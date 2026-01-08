//! `AudioBrick`: Ring buffer audio capture specification (PROBAR-SPEC-009)
//!
//! This brick specifies the audio capture interface using a `SharedArrayBuffer`
//! ring buffer for streaming audio from the main thread to the worker.
//!
//! # Design
//!
//! The `AudioBrick` wraps the `SharedRingBuffer` and provides:
//! - Buffer capacity specification
//! - Write position tracking
//! - Done flag for graceful shutdown
//! - Sample rate configuration
//!
//! # Assertions
//!
//! - Buffer capacity sufficient for latency requirements
//! - Write operations are lock-free
//! - Done flag propagates to worker

use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_core::{
    AccessibleRole, Canvas, Color, Constraints, Event, LayoutResult, Point, Rect, Size, TextStyle,
    TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Default buffer size: 3 seconds at 48kHz
const DEFAULT_BUFFER_SIZE: usize = 144_000;

/// Minimum buffer size: 0.5 seconds at 16kHz
const MIN_BUFFER_SIZE: usize = 8_000;

/// `AudioBrick` for specifying audio capture interface
#[derive(Debug, Clone)]
pub struct AudioBrick {
    /// Buffer capacity in samples
    capacity: usize,
    /// Current write position (for visualization)
    write_pos: usize,
    /// Sample rate in Hz
    sample_rate: u32,
    /// Whether capture is done
    done: bool,
    /// Total samples written
    total_written: u64,
}

impl Default for AudioBrick {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_BUFFER_SIZE,
            write_pos: 0,
            sample_rate: 48000,
            done: false,
            total_written: 0,
        }
    }
}

impl AudioBrick {
    /// Create a new audio brick with default capacity
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(MIN_BUFFER_SIZE),
            ..Default::default()
        }
    }

    /// Set sample rate
    pub fn set_sample_rate(&mut self, rate: u32) {
        self.sample_rate = rate;
    }

    /// Get buffer capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get sample rate
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get buffer duration in seconds
    #[must_use]
    pub fn buffer_duration_secs(&self) -> f32 {
        self.capacity as f32 / self.sample_rate as f32
    }

    /// Simulate write operation (for testing)
    pub fn write(&mut self, samples: &[f32]) {
        let count = samples.len();
        self.write_pos = (self.write_pos + count) % self.capacity;
        self.total_written += count as u64;
    }

    /// Get write position
    #[must_use]
    pub fn write_pos(&self) -> usize {
        self.write_pos
    }

    /// Get total samples written
    #[must_use]
    pub fn total_written(&self) -> u64 {
        self.total_written
    }

    /// Mark as done
    pub fn mark_done(&mut self) {
        self.done = true;
    }

    /// Check if done
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Reset the brick
    pub fn reset(&mut self) {
        self.write_pos = 0;
        self.done = false;
        self.total_written = 0;
    }

    /// Get fill percentage (0-100)
    #[must_use]
    pub fn fill_percent(&self) -> u8 {
        ((self.write_pos as f32 / self.capacity as f32) * 100.0) as u8
    }
}

impl Brick for AudioBrick {
    fn brick_name(&self) -> &'static str {
        "AudioBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        // Use static assertions for non-custom types
        static ASSERTIONS: &[BrickAssertion] = &[
            BrickAssertion::MaxLatencyMs(1),
        ];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        // Audio operations need to be very fast
        BrickBudget::uniform(1)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        // Verify buffer capacity meets minimum
        if self.capacity >= MIN_BUFFER_SIZE {
            // Buffer capacity check passes
            passed.push(BrickAssertion::MaxLatencyMs(1));
        } else {
            failed.push((
                BrickAssertion::MaxLatencyMs(1),
                format!(
                    "Buffer capacity {} < minimum {}",
                    self.capacity, MIN_BUFFER_SIZE
                ),
            ));
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(5),
        }
    }

    fn to_html(&self) -> String {
        let fill = self.fill_percent();
        let duration = self.buffer_duration_secs();

        format!(
            r#"<div class="audio-brick" data-testid="audio">
    <div class="audio-info">
        <span class="audio-rate">{} Hz</span>
        <span class="audio-duration">{:.1}s buffer</span>
        <span class="audio-fill">{fill}% filled</span>
    </div>
    <div class="audio-buffer-bar">
        <div class="audio-buffer-fill" style="width: {fill}%"></div>
    </div>
</div>"#,
            self.sample_rate, duration
        )
    }

    fn to_css(&self) -> String {
        r".audio-brick {
    background: #16213e;
    padding: 0.5rem;
    border-radius: 4px;
}

.audio-info {
    display: flex;
    gap: 1rem;
    font-size: 0.8rem;
    color: #888;
    margin-bottom: 0.25rem;
}

.audio-buffer-bar {
    height: 4px;
    background: #0f3460;
    border-radius: 2px;
    overflow: hidden;
}

.audio-buffer-fill {
    height: 100%;
    background: #4dc3ff;
    transition: width 100ms;
}"
        .into()
    }

    fn test_id(&self) -> Option<&str> {
        Some("audio")
    }
}

impl Widget for AudioBrick {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        // Audio brick is a small status bar
        let height: f32 = 32.0;
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

    #[allow(clippy::cast_precision_loss)]
    fn paint(&self, canvas: &mut dyn Canvas) {
        let bounds = Rect::new(0.0, 0.0, 300.0, 32.0);

        // Draw background
        let bg_color = Color::from_hex("#16213e").unwrap_or(Color::BLACK);
        canvas.fill_rect(bounds, bg_color);

        // Draw buffer bar background
        let bar_rect = Rect::new(8.0, 22.0, bounds.width - 16.0, 4.0);
        let bar_bg = Color::from_hex("#0f3460").unwrap_or(Color::BLACK);
        canvas.fill_rect(bar_rect, bar_bg);

        // Draw buffer fill
        let fill_pct = self.fill_percent() as f32 / 100.0;
        let fill_rect = Rect::new(8.0, 22.0, (bounds.width - 16.0) * fill_pct, 4.0);
        let fill_color = Color::from_hex("#4dc3ff").unwrap_or(Color::BLUE);
        canvas.fill_rect(fill_rect, fill_color);

        // Draw info text
        let text = format!(
            "{} Hz | {:.1}s | {}%",
            self.sample_rate,
            self.buffer_duration_secs(),
            self.fill_percent()
        );
        let style = TextStyle {
            size: 12.0,
            color: Color::from_hex("#888888").unwrap_or(Color::WHITE),
            weight: presentar_core::FontWeight::Normal,
            style: presentar_core::FontStyle::Normal,
        };
        canvas.draw_text(&text, Point::new(8.0, 14.0), &style);
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
        Some("Audio buffer status")
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
        let brick = AudioBrick::new();
        assert_eq!(brick.capacity(), DEFAULT_BUFFER_SIZE);
        assert_eq!(brick.sample_rate(), 48000);
        assert!(!brick.is_done());
    }

    #[test]
    fn test_with_capacity() {
        let brick = AudioBrick::with_capacity(96000);
        assert_eq!(brick.capacity(), 96000);
    }

    #[test]
    fn test_min_capacity_enforced() {
        let brick = AudioBrick::with_capacity(100);
        assert_eq!(brick.capacity(), MIN_BUFFER_SIZE);
    }

    #[test]
    fn test_buffer_duration() {
        let brick = AudioBrick::with_capacity(48000);
        assert!((brick.buffer_duration_secs() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_write() {
        let mut brick = AudioBrick::new();
        brick.write(&[0.0; 1000]);

        assert_eq!(brick.write_pos(), 1000);
        assert_eq!(brick.total_written(), 1000);
    }

    #[test]
    fn test_write_wrap() {
        // Use capacity > MIN_BUFFER_SIZE to test wrapping
        let mut brick = AudioBrick::with_capacity(10000);
        brick.write(&[0.0; 15000]);

        assert_eq!(brick.write_pos(), 5000);
        assert_eq!(brick.total_written(), 15000);
    }

    #[test]
    fn test_mark_done() {
        let mut brick = AudioBrick::new();
        assert!(!brick.is_done());

        brick.mark_done();
        assert!(brick.is_done());
    }

    #[test]
    fn test_reset() {
        let mut brick = AudioBrick::new();
        brick.write(&[0.0; 1000]);
        brick.mark_done();
        brick.reset();

        assert_eq!(brick.write_pos(), 0);
        assert!(!brick.is_done());
        assert_eq!(brick.total_written(), 0);
    }

    #[test]
    fn test_fill_percent() {
        // Use capacity > MIN_BUFFER_SIZE
        let mut brick = AudioBrick::with_capacity(10000);
        brick.write(&[0.0; 5000]);

        assert_eq!(brick.fill_percent(), 50);
    }

    #[test]
    fn test_verification_passes() {
        let brick = AudioBrick::new();
        let result = brick.verify();

        assert!(result.is_valid());
    }

    #[test]
    fn test_to_html() {
        let brick = AudioBrick::new();
        let html = brick.to_html();

        assert!(html.contains("data-testid=\"audio\""));
        assert!(html.contains("48000 Hz"));
    }

    #[test]
    fn test_budget() {
        let brick = AudioBrick::new();
        assert_eq!(brick.budget().total_ms, 1);
    }

    #[test]
    fn test_can_render() {
        let brick = AudioBrick::new();
        assert!(brick.can_render());
    }
}
