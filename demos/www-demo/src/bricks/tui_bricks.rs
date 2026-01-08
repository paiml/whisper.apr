//! TUI Brick Traits for whisper.apr (PROBAR-SPEC-009-P13)
//!
//! Re-exports probar TUI brick types and provides whisper.apr-specific
//! extensions for audio transcription TUI.
//!
//! Based on patterns from ttop (trueno-viz).

// Re-export probar TUI types for common use
pub use jugar_probar::brick::{
    AnalyzerBrick, CielabColor, CollectorBrick, CollectorError, PanelBrick, PanelId,
    PanelState as ProbarPanelState, RingBuffer,
};

// ============================================================================
// Whisper.apr-specific Panel Types
// ============================================================================

/// Panel type identifier for whisper.apr TUI
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

impl PanelType {
    /// Convert to generic PanelId for use with probar
    #[must_use]
    pub fn to_panel_id(self) -> PanelId {
        match self {
            Self::Waveform => PanelId::Waveform,
            Self::Spectrogram => PanelId::Custom(1),
            Self::Transcription => PanelId::Transcription,
            Self::Metrics => PanelId::Metrics,
            Self::VuMeter => PanelId::VuMeter,
            Self::Status => PanelId::Status,
        }
    }
}

/// Panel state for whisper.apr focus/explode behavior
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
        if self.visible.is_empty() {
            self.focused = None;
            return;
        }

        let current_idx = self
            .focused
            .and_then(|f| self.visible.iter().position(|p| *p == f));

        let next_idx = current_idx
            .map(|i| (i + 1) % self.visible.len())
            .unwrap_or(0);

        self.focused = self.visible.get(next_idx).copied();
    }

    /// Focus previous panel
    pub fn focus_prev(&mut self) {
        if self.visible.is_empty() {
            self.focused = None;
            return;
        }

        let current_idx = self
            .focused
            .and_then(|f| self.visible.iter().position(|p| *p == f));

        let prev_idx = current_idx
            .map(|i| {
                if i == 0 {
                    self.visible.len() - 1
                } else {
                    i - 1
                }
            })
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
    #[must_use]
    pub fn is_focused(&self, panel: PanelType) -> bool {
        self.focused == Some(panel)
    }

    /// Check if a panel is exploded
    #[must_use]
    pub fn is_exploded(&self, panel: PanelType) -> bool {
        self.exploded == Some(panel)
    }
}

// ============================================================================
// Whisper.apr-specific Metrics Types
// ============================================================================

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

/// Panel dimensions for TUI layout
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    /// X position
    pub x: u16,
    /// Y position
    pub y: u16,
    /// Width
    pub width: u16,
    /// Height
    pub height: u16,
}

impl Rect {
    /// Create a new rectangle
    #[must_use]
    pub const fn new(x: u16, y: u16, width: u16, height: u16) -> Self {
        Self { x, y, width, height }
    }
}

// ============================================================================
// Tests
// ============================================================================

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

        let contiguous = buf.to_vec();
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
