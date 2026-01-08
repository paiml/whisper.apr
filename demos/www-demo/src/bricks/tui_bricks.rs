//! TUI Brick Traits for whisper.apr (PROBAR-SPEC-009-P13)
//!
//! Implements the three-layer TUI brick architecture:
//! - CollectorBrick: Gathers system/audio metrics
//! - AnalyzerBrick: Produces insights from metrics
//! - PanelBrick: Renders TUI panels
//!
//! Based on patterns from ttop (trueno-viz).

use jugar_probar::brick::Brick;
use std::collections::VecDeque;

/// Error type for collector operations
#[derive(Debug, Clone)]
pub enum CollectorError {
    /// Feature not available on this platform
    NotAvailable,
    /// Collection failed
    Failed(String),
}

impl std::fmt::Display for CollectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAvailable => write!(f, "Feature not available"),
            Self::Failed(msg) => write!(f, "Collection failed: {}", msg),
        }
    }
}

impl std::error::Error for CollectorError {}

/// Trait for bricks that collect metrics
pub trait CollectorBrick: Brick + Send + Sync {
    /// Metrics type produced by this collector
    type Metrics;

    /// Check if collector is available on current platform
    fn is_available(&self) -> bool;

    /// Collect metrics
    fn collect(&mut self) -> Result<Self::Metrics, CollectorError>;

    /// Optional feature gate name
    fn feature_gate(&self) -> Option<&'static str> {
        None
    }
}

/// Trait for bricks that analyze metrics
pub trait AnalyzerBrick: Brick + Send + Sync {
    /// Input metrics type
    type Input;
    /// Output analysis type
    type Output;

    /// Analyze metrics and produce insights
    fn analyze(&self, input: &Self::Input) -> Self::Output;
}

/// Panel type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PanelType {
    /// Waveform display
    Waveform,
    /// Spectrogram visualization
    Spectrogram,
    /// Transcription text
    Transcription,
    /// Performance metrics
    Metrics,
    /// VU meter
    VuMeter,
    /// Status
    Status,
}

/// Panel state for focus/explode behavior
#[derive(Debug, Clone)]
pub struct PanelState {
    /// Currently focused panel
    pub focused: Option<PanelType>,
    /// Currently exploded (full-screen) panel
    pub exploded: Option<PanelType>,
    /// Visible panels in order
    pub visible: Vec<PanelType>,
}

impl Default for PanelState {
    fn default() -> Self {
        Self {
            focused: None,
            exploded: None,
            visible: vec![
                PanelType::Waveform,
                PanelType::Spectrogram,
                PanelType::Transcription,
                PanelType::Metrics,
            ],
        }
    }
}

impl PanelState {
    /// Focus next panel
    pub fn focus_next(&mut self) {
        let current_idx = self.focused.and_then(|f| {
            self.visible.iter().position(|p| *p == f)
        });

        let next_idx = current_idx
            .map(|i| (i + 1) % self.visible.len())
            .unwrap_or(0);

        self.focused = self.visible.get(next_idx).copied();
    }

    /// Focus previous panel
    pub fn focus_prev(&mut self) {
        let current_idx = self.focused.and_then(|f| {
            self.visible.iter().position(|p| *p == f)
        });

        let prev_idx = current_idx
            .map(|i| if i == 0 { self.visible.len() - 1 } else { i - 1 })
            .unwrap_or(0);

        self.focused = self.visible.get(prev_idx).copied();
    }

    /// Toggle exploded state for focused panel
    pub fn toggle_explode(&mut self) {
        if self.exploded.is_some() {
            self.exploded = None;
        } else {
            self.exploded = self.focused;
        }
    }

    /// Check if a panel is focused
    pub fn is_focused(&self, panel: PanelType) -> bool {
        self.focused == Some(panel)
    }

    /// Check if a panel is exploded
    pub fn is_exploded(&self, panel: PanelType) -> bool {
        self.exploded == Some(panel)
    }
}

/// Ring buffer for time-series data (from ttop pattern)
#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    data: VecDeque<T>,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer with given capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a value, evicting oldest if at capacity
    pub fn push(&mut self, value: T) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(value);
    }

    /// Get iterator over values (oldest first)
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Get number of elements
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get most recent value
    #[must_use]
    pub fn last(&self) -> Option<&T> {
        self.data.back()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl<T: Clone> RingBuffer<T> {
    /// Make buffer contiguous and return as slice
    #[must_use]
    pub fn make_contiguous(&self) -> Vec<T> {
        self.data.iter().cloned().collect()
    }
}

impl<T: Copy + Default> RingBuffer<T> {
    /// Fill with default values to capacity
    pub fn fill_default(&mut self) {
        while self.data.len() < self.capacity {
            self.data.push_back(T::default());
        }
    }
}

/// Audio collector metrics
#[derive(Debug, Clone, Default)]
pub struct AudioMetrics {
    /// Current audio level (0-1)
    pub level: f32,
    /// Peak level since last reset
    pub peak: f32,
    /// Number of samples processed
    pub samples_processed: u64,
    /// Buffer fill percentage
    pub buffer_fill: f32,
}

/// RTF (Real-Time Factor) analyzer output
#[derive(Debug, Clone, Default)]
pub struct RtfAnalysis {
    /// Current RTF (processing time / audio duration)
    pub current_rtf: f64,
    /// Average RTF over history
    pub average_rtf: f64,
    /// Is running faster than real-time?
    pub is_realtime: bool,
}

/// Metrics panel state
#[derive(Debug, Clone, Default)]
pub struct MetricsState {
    /// Audio level history
    pub level_history: Vec<f32>,
    /// RTF history
    pub rtf_history: Vec<f64>,
    /// Total audio processed (seconds)
    pub audio_seconds: f64,
    /// Total processing time (seconds)
    pub processing_seconds: f64,
}

/// CIELAB color for perceptual uniformity
#[derive(Debug, Clone, Copy)]
pub struct CielabColor {
    /// Lightness (0-100)
    pub l: f32,
    /// Green-red (approx -128 to 127)
    pub a: f32,
    /// Blue-yellow (approx -128 to 127)
    pub b: f32,
}

impl CielabColor {
    /// Create a CIELAB color
    #[must_use]
    pub const fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }

    /// Interpolate between two colors
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            l: self.l + (other.l - self.l) * t,
            a: self.a + (other.a - self.a) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }

    /// Convert to approximate RGB (simplified)
    #[must_use]
    pub fn to_rgb(&self) -> (u8, u8, u8) {
        // Simplified conversion - real implementation would use proper CIELAB to RGB
        let l = self.l / 100.0;
        let r = ((l + self.a / 500.0) * 255.0).clamp(0.0, 255.0) as u8;
        let g = (l * 255.0).clamp(0.0, 255.0) as u8;
        let b = ((l - self.b / 200.0) * 255.0).clamp(0.0, 255.0) as u8;
        (r, g, b)
    }

    /// Create a perceptually uniform gradient from green to red
    #[must_use]
    pub fn percent_gradient(percent: f32) -> Self {
        let green = Self::new(87.0, -86.0, 83.0);  // Bright green
        let yellow = Self::new(97.0, -21.0, 94.0); // Yellow
        let red = Self::new(53.0, 80.0, 67.0);     // Bright red

        if percent < 0.5 {
            green.lerp(&yellow, percent * 2.0)
        } else {
            yellow.lerp(&red, (percent - 0.5) * 2.0)
        }
    }
}

/// Panel dimensions
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub height: u16,
}

impl Rect {
    #[must_use]
    pub const fn new(x: u16, y: u16, width: u16, height: u16) -> Self {
        Self { x, y, width, height }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);
        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.push(4); // Should evict 1

        let values: Vec<_> = buf.iter().copied().collect();
        assert_eq!(values, vec![2, 3, 4]);
    }

    #[test]
    fn test_ring_buffer_capacity() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(5);
        for i in 0..10 {
            buf.push(i);
        }
        assert_eq!(buf.len(), 5);
        assert_eq!(*buf.last().unwrap(), 9);
    }

    #[test]
    fn test_ring_buffer_make_contiguous() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);
        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.push(4);

        let contiguous = buf.make_contiguous();
        assert_eq!(contiguous, vec![2, 3, 4]);
    }

    #[test]
    fn test_panel_state_focus() {
        let mut state = PanelState::default();
        state.focused = Some(PanelType::Waveform);

        state.focus_next();
        assert_eq!(state.focused, Some(PanelType::Spectrogram));

        state.focus_next();
        assert_eq!(state.focused, Some(PanelType::Transcription));

        state.focus_prev();
        assert_eq!(state.focused, Some(PanelType::Spectrogram));
    }

    #[test]
    fn test_panel_state_explode() {
        let mut state = PanelState::default();
        state.focused = Some(PanelType::Transcription);

        state.toggle_explode();
        assert_eq!(state.exploded, Some(PanelType::Transcription));

        state.toggle_explode();
        assert_eq!(state.exploded, None);
    }

    #[test]
    fn test_cielab_lerp() {
        let green = CielabColor::new(87.0, -86.0, 83.0);
        let red = CielabColor::new(53.0, 80.0, 67.0);

        let mid = green.lerp(&red, 0.5);
        assert!((mid.l - 70.0).abs() < 0.1);
    }

    #[test]
    fn test_cielab_gradient() {
        let start = CielabColor::percent_gradient(0.0);
        let end = CielabColor::percent_gradient(1.0);

        // Start should be greenish (negative a)
        assert!(start.a < 0.0);
        // End should be reddish (positive a)
        assert!(end.a > 0.0);
    }
}
