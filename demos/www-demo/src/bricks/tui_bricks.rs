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
// Whisper TUI Application (PROBAR-SPEC-009-P13h)
// ============================================================================

use super::{AudioBrick, TranscriptionBrick, VuMeterBrick, WaveformBrick};

/// RTF (Real-Time Factor) analyzer
#[derive(Debug, Clone)]
pub struct RtfAnalyzer {
    history: RingBuffer<f64>,
}

impl Default for RtfAnalyzer {
    fn default() -> Self {
        Self {
            history: RingBuffer::new(60), // 60 samples
        }
    }
}

impl RtfAnalyzer {
    /// Record an RTF measurement
    pub fn record(&mut self, audio_duration_ms: f64, processing_time_ms: f64) {
        if audio_duration_ms > 0.0 {
            let rtf = processing_time_ms / audio_duration_ms;
            self.history.push(rtf);
        }
    }

    /// Analyze RTF history
    #[must_use]
    pub fn analyze(&self) -> RtfAnalysis {
        if self.history.is_empty() {
            return RtfAnalysis::default();
        }

        let current_rtf = self.history.last().copied().unwrap_or(0.0);
        let sum: f64 = self.history.iter().sum();
        #[allow(clippy::cast_precision_loss)]
        let average_rtf = sum / self.history.len() as f64;

        RtfAnalysis {
            current_rtf,
            average_rtf,
            is_realtime: current_rtf <= 1.0,
        }
    }
}

/// Whisper TUI application integrating all bricks
///
/// This is the main application struct that follows the three-layer
/// TUI brick architecture from PROBAR-SPEC-009-P13:
/// - Collectors: Audio metrics
/// - Analyzers: RTF analysis
/// - Panels: Waveform, Transcription, Metrics
#[derive(Debug)]
pub struct WhisperTuiApp {
    /// Audio collector brick
    pub audio: AudioBrick,
    /// VU meter brick
    pub vu_meter: VuMeterBrick,
    /// Waveform panel brick
    pub waveform: WaveformBrick,
    /// Transcription panel brick
    pub transcription: TranscriptionBrick,
    /// RTF analyzer
    pub rtf_analyzer: RtfAnalyzer,
    /// Audio level history for sparkline
    pub audio_history: RingBuffer<f32>,
    /// RTF history for metrics panel
    pub rtf_history: RingBuffer<f64>,
    /// Panel navigation state
    pub panel_state: PanelState,
}

impl Default for WhisperTuiApp {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperTuiApp {
    /// Create a new whisper TUI application
    #[must_use]
    pub fn new() -> Self {
        Self {
            audio: AudioBrick::new(),
            vu_meter: VuMeterBrick::new(),
            waveform: WaveformBrick::new(),
            transcription: TranscriptionBrick::new(),
            rtf_analyzer: RtfAnalyzer::default(),
            audio_history: RingBuffer::new(120), // 2 minutes at 1Hz
            rtf_history: RingBuffer::new(60),
            panel_state: PanelState::default(),
        }
    }

    /// Update with audio samples
    pub fn push_audio(&mut self, samples: &[f32]) {
        self.audio.write(samples);

        // Update VU meter
        self.vu_meter.update_from_samples(samples);

        // Update waveform (downsample)
        for (i, &sample) in samples.iter().enumerate() {
            if i % 100 == 0 {
                self.waveform.push_sample(sample);
            }
        }

        // Record audio level in history
        self.audio_history.push(self.vu_meter.level());
    }

    /// Update with transcription
    pub fn update_transcription(&mut self, text: &str, is_final: bool) {
        if is_final {
            self.transcription.on_final(text.into());
        } else {
            self.transcription.on_partial(text.into());
        }
    }

    /// Record an inference timing
    pub fn record_inference(&mut self, audio_duration_ms: f64, processing_time_ms: f64) {
        self.rtf_analyzer.record(audio_duration_ms, processing_time_ms);
        let analysis = self.rtf_analyzer.analyze();
        self.rtf_history.push(analysis.current_rtf);
    }

    /// Get panel layout rectangles for 4-panel layout
    #[must_use]
    pub fn layout(&self, width: u16, height: u16) -> Vec<(PanelType, Rect)> {
        // 4-panel layout: 25% waveform, 25% spectrogram, 35% transcription, 15% metrics
        let h1 = height / 4;
        let h2 = height / 4;
        let h3 = (height * 35) / 100;
        let h4 = height - h1 - h2 - h3;

        vec![
            (PanelType::Waveform, Rect::new(0, 0, width, h1)),
            (PanelType::Spectrogram, Rect::new(0, h1, width, h2)),
            (PanelType::Transcription, Rect::new(0, h1 + h2, width, h3)),
            (PanelType::Metrics, Rect::new(0, h1 + h2 + h3, width, h4)),
        ]
    }

    /// Render to text lines (for TUI output)
    #[must_use]
    pub fn render(&self, width: u16, height: u16) -> Vec<String> {
        let mut lines = Vec::new();
        let layout = self.layout(width, height);

        for (panel_type, rect) in layout {
            let panel_lines = self.render_panel(panel_type, rect.width, rect.height);
            lines.extend(panel_lines);
        }

        lines
    }

    /// Render a single panel
    fn render_panel(&self, panel_type: PanelType, width: u16, height: u16) -> Vec<String> {
        match panel_type {
            PanelType::Waveform => self.render_waveform_panel(width, height),
            PanelType::Spectrogram => self.render_spectrogram_panel(width, height),
            PanelType::Transcription => self.render_transcription_panel(width, height),
            PanelType::Metrics => self.render_metrics_panel(width, height),
            PanelType::VuMeter => self.render_vu_panel(width),
            PanelType::Status => vec![String::new()],
        }
    }

    fn render_waveform_panel(&self, width: u16, _height: u16) -> Vec<String> {
        const SPARKLINE: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        let samples = self.waveform.get_display_samples();
        let bar_width = (width as usize).saturating_sub(12);

        let sparkline: String = samples
            .iter()
            .take(bar_width)
            .map(|&s| {
                let normalized = s.abs().clamp(0.0, 1.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let idx = (normalized * 7.0) as usize;
                SPARKLINE[idx.min(7)]
            })
            .collect();

        vec![
            format!("┌─ Waveform {:─<width$}┐", "", width = bar_width.saturating_sub(10)),
            format!("│ {sparkline:<bar_width$} │"),
            format!("└{:─<width$}┘", "", width = bar_width + 2),
        ]
    }

    fn render_spectrogram_panel(&self, width: u16, _height: u16) -> Vec<String> {
        let bar_width = (width as usize).saturating_sub(12);
        // Placeholder - real spectrogram would use mel filterbank data
        vec![
            format!("┌─ Spectrogram {:─<width$}┐", "", width = bar_width.saturating_sub(13)),
            format!("│ {:^bar_width$} │", "(mel filterbank visualization)"),
            format!("└{:─<width$}┘", "", width = bar_width + 2),
        ]
    }

    fn render_transcription_panel(&self, width: u16, height: u16) -> Vec<String> {
        let bar_width = (width as usize).saturating_sub(4);
        let mut lines = vec![format!(
            "┌─ Transcription {:─<width$}┐",
            "",
            width = bar_width.saturating_sub(16)
        )];

        let text = self.transcription.combined_text();
        if text.is_empty() {
            lines.push(format!("│ {:^bar_width$} │", "(waiting for speech...)"));
        } else {
            // Word wrap
            let words: Vec<&str> = text.split_whitespace().collect();
            let mut current_line = String::new();
            for word in words {
                if current_line.is_empty() {
                    current_line = word.to_string();
                } else if current_line.len() + 1 + word.len() <= bar_width {
                    current_line.push(' ');
                    current_line.push_str(word);
                } else {
                    lines.push(format!("│ {current_line:<bar_width$} │"));
                    current_line = word.to_string();
                }
            }
            if !current_line.is_empty() {
                lines.push(format!("│ {current_line:<bar_width$} │"));
            }
        }

        // Pad to height
        while lines.len() < height as usize - 1 {
            lines.push(format!("│ {:bar_width$} │", ""));
        }

        lines.push(format!("└{:─<width$}┘", "", width = bar_width + 2));
        lines
    }

    fn render_metrics_panel(&self, width: u16, _height: u16) -> Vec<String> {
        let bar_width = (width as usize).saturating_sub(4);
        let analysis = self.rtf_analyzer.analyze();

        let rtf_status = if analysis.is_realtime { "✓" } else { "⚠" };

        vec![
            format!("┌─ Metrics {:─<width$}┐", "", width = bar_width.saturating_sub(10)),
            format!(
                "│ RTF: {:.2}x (avg: {:.2}x) {} {:width$} │",
                analysis.current_rtf,
                analysis.average_rtf,
                rtf_status,
                "",
                width = bar_width.saturating_sub(30)
            ),
            format!(
                "│ Audio: {:.1}s buffered {:width$} │",
                self.audio.buffer_duration_secs(),
                "",
                width = bar_width.saturating_sub(22)
            ),
            format!("└{:─<width$}┘", "", width = bar_width + 2),
        ]
    }

    fn render_vu_panel(&self, width: u16) -> Vec<String> {
        let level = self.vu_meter.level();
        let bar_width = 20;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let filled = (level * bar_width as f32) as usize;

        let bar: String = (0..bar_width)
            .map(|i| if i < filled { '█' } else { '░' })
            .collect();

        vec![format!(
            "VU: [{bar}] {:3}% {:width$}",
            self.vu_meter.level_percent(),
            "",
            width = (width as usize).saturating_sub(30)
        )]
    }

    /// Handle keyboard navigation
    pub fn handle_key(&mut self, key: char) {
        match key {
            'j' | '\t' => self.panel_state.focus_next(),
            'k' => self.panel_state.focus_prev(),
            'z' | '\n' => self.panel_state.toggle_explode(),
            _unhandled => {} // Intentional: unknown keys are silently ignored
        }
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

    #[test]
    fn test_rtf_analyzer() {
        let mut analyzer = RtfAnalyzer::default();

        // Record some inference timings (100ms audio, 50ms processing = 0.5x RTF)
        analyzer.record(100.0, 50.0);
        let analysis = analyzer.analyze();

        assert!((analysis.current_rtf - 0.5).abs() < 0.01);
        assert!(analysis.is_realtime);
    }

    #[test]
    fn test_rtf_analyzer_not_realtime() {
        let mut analyzer = RtfAnalyzer::default();

        // Record slow inference (100ms audio, 200ms processing = 2.0x RTF)
        analyzer.record(100.0, 200.0);
        let analysis = analyzer.analyze();

        assert!((analysis.current_rtf - 2.0).abs() < 0.01);
        assert!(!analysis.is_realtime);
    }

    #[test]
    fn test_whisper_tui_app_new() {
        let app = WhisperTuiApp::new();
        assert!(app.audio_history.is_empty());
        assert!(app.rtf_history.is_empty());
        assert!(app.transcription.combined_text().is_empty());
    }

    #[test]
    fn test_whisper_tui_app_push_audio() {
        let mut app = WhisperTuiApp::new();
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 / 100.0).sin()).collect();

        app.push_audio(&samples);

        assert!(!app.audio_history.is_empty());
        assert!(app.vu_meter.level() > 0.0);
    }

    #[test]
    fn test_whisper_tui_app_transcription() {
        let mut app = WhisperTuiApp::new();

        app.update_transcription("hello", false);
        assert!(app.transcription.partial().contains("hello"));

        app.update_transcription("hello world", true);
        assert!(app.transcription.combined_text().contains("hello world"));
    }

    #[test]
    fn test_whisper_tui_app_layout() {
        let app = WhisperTuiApp::new();
        let layout = app.layout(80, 24);

        assert_eq!(layout.len(), 4);
        assert_eq!(layout[0].0, PanelType::Waveform);
        assert_eq!(layout[1].0, PanelType::Spectrogram);
        assert_eq!(layout[2].0, PanelType::Transcription);
        assert_eq!(layout[3].0, PanelType::Metrics);
    }

    #[test]
    fn test_whisper_tui_app_render() {
        let app = WhisperTuiApp::new();
        let lines = app.render(80, 24);

        assert!(!lines.is_empty());
        assert!(lines.iter().any(|l| l.contains("Waveform")));
        assert!(lines.iter().any(|l| l.contains("Transcription")));
        assert!(lines.iter().any(|l| l.contains("Metrics")));
    }

    #[test]
    fn test_whisper_tui_app_keyboard() {
        let mut app = WhisperTuiApp::new();

        app.handle_key('j'); // focus next
        assert_eq!(app.panel_state.focused, Some(PanelType::Waveform));

        app.handle_key('j');
        assert_eq!(app.panel_state.focused, Some(PanelType::Spectrogram));

        app.handle_key('k'); // focus prev
        assert_eq!(app.panel_state.focused, Some(PanelType::Waveform));

        app.handle_key('z'); // toggle explode
        assert_eq!(app.panel_state.exploded, Some(PanelType::Waveform));
    }
}
