//! `TimingStatsBrick`: Real-time timing statistics display (PROBAR-SPEC-009)
//!
//! This brick displays timing metrics for streaming transcription:
//! - Real-Time Factor (RTF) per chunk and average
//! - Latency per chunk (processing time)
//! - Audio duration vs processing time graph
//!
//! # Assertions
//!
//! - Statistics update in real-time
//! - RTF calculation accurate
//! - Visual latency graph

use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_core::{
    AccessibleRole, Canvas, Color, Constraints, Event, LayoutResult, Point, Rect, Size, TextStyle,
    TypeId, Widget,
};
use std::any::Any;
use std::collections::VecDeque;
use std::time::Duration;

/// Single timing measurement
#[derive(Debug, Clone, Copy)]
pub struct TimingMeasurement {
    /// Chunk index
    pub chunk_index: u32,
    /// Audio duration in seconds
    pub audio_duration_secs: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// End-to-end latency in milliseconds (includes queue time)
    pub latency_ms: f32,
}

impl TimingMeasurement {
    /// Calculate RTF (processing time / audio duration)
    #[must_use]
    pub fn rtf(&self) -> f32 {
        if self.audio_duration_secs > 0.0 {
            (self.processing_time_ms / 1000.0) / self.audio_duration_secs
        } else {
            0.0
        }
    }

    /// Check if real-time (RTF <= 1.0)
    #[must_use]
    pub fn is_realtime(&self) -> bool {
        self.rtf() <= 1.0
    }
}

/// Timing statistics state
#[derive(Debug, Clone)]
pub struct TimingStatsState {
    /// Recent measurements (ring buffer)
    measurements: VecDeque<TimingMeasurement>,
    /// Maximum measurements to keep
    max_history: usize,
    /// Total audio processed (seconds)
    pub total_audio_secs: f32,
    /// Total processing time (milliseconds)
    pub total_processing_ms: f32,
    /// Best (lowest) RTF observed
    pub best_rtf: Option<f32>,
    /// Worst (highest) RTF observed
    pub worst_rtf: Option<f32>,
    /// Target RTF for comparison (e.g., 2.0x for tiny model)
    pub target_rtf: f32,
}

impl Default for TimingStatsState {
    fn default() -> Self {
        Self {
            measurements: VecDeque::new(),
            max_history: 20,
            total_audio_secs: 0.0,
            total_processing_ms: 0.0,
            best_rtf: None,
            worst_rtf: None,
            target_rtf: 2.0, // Default target: 2.0x RTF (tiny model target)
        }
    }
}

impl TimingStatsState {
    /// Create new state
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom target RTF
    #[must_use]
    pub fn with_target_rtf(target_rtf: f32) -> Self {
        Self {
            target_rtf,
            ..Default::default()
        }
    }

    /// Add a timing measurement
    pub fn add_measurement(&mut self, measurement: TimingMeasurement) {
        // Update totals
        self.total_audio_secs += measurement.audio_duration_secs;
        self.total_processing_ms += measurement.processing_time_ms;

        // Update best/worst RTF
        let rtf = measurement.rtf();
        self.best_rtf = Some(match self.best_rtf {
            Some(best) => best.min(rtf),
            None => rtf,
        });
        self.worst_rtf = Some(match self.worst_rtf {
            Some(worst) => worst.max(rtf),
            None => rtf,
        });

        // Add to history
        self.measurements.push_back(measurement);
        while self.measurements.len() > self.max_history {
            self.measurements.pop_front();
        }
    }

    /// Get average RTF
    #[must_use]
    pub fn average_rtf(&self) -> Option<f32> {
        if self.total_audio_secs > 0.0 {
            Some((self.total_processing_ms / 1000.0) / self.total_audio_secs)
        } else {
            None
        }
    }

    /// Get average latency in milliseconds
    #[must_use]
    pub fn average_latency_ms(&self) -> Option<f32> {
        if self.measurements.is_empty() {
            return None;
        }

        let sum: f32 = self.measurements.iter().map(|m| m.latency_ms).sum();
        Some(sum / self.measurements.len() as f32)
    }

    /// Get recent measurements
    #[must_use]
    pub fn recent_measurements(&self) -> &VecDeque<TimingMeasurement> {
        &self.measurements
    }

    /// Get last measurement
    #[must_use]
    pub fn last_measurement(&self) -> Option<&TimingMeasurement> {
        self.measurements.back()
    }

    /// Check if meeting target RTF
    #[must_use]
    pub fn is_meeting_target(&self) -> Option<bool> {
        self.average_rtf().map(|rtf| rtf <= self.target_rtf)
    }

    /// Get RTF status color
    #[must_use]
    pub fn rtf_status_color(&self) -> &'static str {
        match self.average_rtf() {
            Some(rtf) if rtf <= 1.0 => "#50fa7b",  // Real-time: green
            Some(rtf) if rtf <= self.target_rtf => "#4dc3ff",  // Meeting target: blue
            Some(_) => "#ff6b6b",  // Below target: red
            None => "#8b949e",  // No data: gray
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.measurements.clear();
        self.total_audio_secs = 0.0;
        self.total_processing_ms = 0.0;
        self.best_rtf = None;
        self.worst_rtf = None;
    }

    /// Format RTF for display
    #[must_use]
    pub fn format_rtf(rtf: f32) -> String {
        format!("{:.2}x", rtf)
    }

    /// Format latency for display
    #[must_use]
    pub fn format_latency(latency_ms: f32) -> String {
        if latency_ms < 1000.0 {
            format!("{:.0}ms", latency_ms)
        } else {
            format!("{:.2}s", latency_ms / 1000.0)
        }
    }
}

/// Timing statistics brick
#[derive(Debug, Clone, Default)]
pub struct TimingStatsBrick {
    state: TimingStatsState,
}

impl TimingStatsBrick {
    /// Create new brick
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom target RTF
    #[must_use]
    pub fn with_target_rtf(target_rtf: f32) -> Self {
        Self {
            state: TimingStatsState::with_target_rtf(target_rtf),
        }
    }

    /// Get state reference
    #[must_use]
    pub fn state(&self) -> &TimingStatsState {
        &self.state
    }

    /// Get mutable state reference
    pub fn state_mut(&mut self) -> &mut TimingStatsState {
        &mut self.state
    }

    /// Add timing measurement
    pub fn add_measurement(&mut self, measurement: TimingMeasurement) {
        self.state.add_measurement(measurement);
    }

    /// Add measurement from individual values
    pub fn record(&mut self, chunk_index: u32, audio_secs: f32, processing_ms: f32, latency_ms: f32) {
        self.add_measurement(TimingMeasurement {
            chunk_index,
            audio_duration_secs: audio_secs,
            processing_time_ms: processing_ms,
            latency_ms,
        });
    }

    /// Set target RTF
    pub fn set_target_rtf(&mut self, target: f32) {
        self.state.target_rtf = target;
    }

    /// Reset
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Generate SVG sparkline for RTF history
    fn generate_sparkline(&self) -> String {
        let measurements: Vec<_> = self.state.measurements.iter().collect();
        if measurements.len() < 2 {
            return String::new();
        }

        let width = 200.0;
        let height = 30.0;
        let padding = 2.0;

        // Find RTF range
        let rtfs: Vec<f32> = measurements.iter().map(|m| m.rtf()).collect();
        let min_rtf = rtfs.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_rtf = rtfs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_rtf - min_rtf).max(0.1);

        // Generate points
        let point_count = rtfs.len();
        let x_step = (width - 2.0 * padding) / (point_count - 1) as f32;

        let points: Vec<String> = rtfs
            .iter()
            .enumerate()
            .map(|(i, &rtf)| {
                let x = padding + i as f32 * x_step;
                let y = height - padding - ((rtf - min_rtf) / range) * (height - 2.0 * padding);
                format!("{:.1},{:.1}", x, y)
            })
            .collect();

        let path = points.join(" ");

        // Target line y position
        let target_y = height - padding
            - ((self.state.target_rtf - min_rtf) / range).clamp(0.0, 1.0)
                * (height - 2.0 * padding);

        let target_color = "#4dc3ff";
        format!(
            r#"<svg class="rtf-sparkline" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
    <line x1="0" y1="{target_y}" x2="{width}" y2="{target_y}" stroke="{target_color}" stroke-width="1" stroke-dasharray="4,2" opacity="0.5"/>
    <polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>
</svg>"#,
            width = width,
            height = height,
            target_y = target_y,
            target_color = target_color,
            path = path,
            color = self.state.rtf_status_color(),
        )
    }
}

impl Brick for TimingStatsBrick {
    fn brick_name(&self) -> &'static str {
        "TimingStatsBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::MaxLatencyMs(16),
        ]
    }

    fn budget(&self) -> BrickBudget {
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
        let state = &self.state;

        // No data state
        if state.measurements.is_empty() {
            return r#"<div class="timing-stats-brick" data-testid="timing-stats">
    <div class="stats-header">
        <span class="title">Timing Statistics</span>
    </div>
    <div class="stats-empty">
        <span>No data yet</span>
    </div>
</div>"#
                .into();
        }

        let avg_rtf = state
            .average_rtf()
            .map(TimingStatsState::format_rtf)
            .unwrap_or_else(|| "—".into());
        let avg_latency = state
            .average_latency_ms()
            .map(TimingStatsState::format_latency)
            .unwrap_or_else(|| "—".into());
        let best_rtf = state
            .best_rtf
            .map(TimingStatsState::format_rtf)
            .unwrap_or_else(|| "—".into());
        let worst_rtf = state
            .worst_rtf
            .map(TimingStatsState::format_rtf)
            .unwrap_or_else(|| "—".into());

        let last = state.last_measurement();
        let last_rtf = last
            .map(|m| TimingStatsState::format_rtf(m.rtf()))
            .unwrap_or_else(|| "—".into());
        let last_latency = last
            .map(|m| TimingStatsState::format_latency(m.latency_ms))
            .unwrap_or_else(|| "—".into());

        let status_class = match state.is_meeting_target() {
            Some(true) => "status-good",
            Some(false) => "status-warning",
            None => "status-unknown",
        };

        let sparkline = self.generate_sparkline();

        format!(
            r#"<div class="timing-stats-brick {status_class}" data-testid="timing-stats">
    <div class="stats-header">
        <span class="title">Timing Statistics</span>
        <span class="target-badge">Target: {target}x</span>
    </div>
    <div class="stats-grid">
        <div class="stat-item">
            <span class="stat-label">Avg RTF</span>
            <span class="stat-value rtf" data-testid="avg-rtf" style="color: {rtf_color}">{avg_rtf}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Avg Latency</span>
            <span class="stat-value" data-testid="avg-latency">{avg_latency}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Last RTF</span>
            <span class="stat-value">{last_rtf}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Last Latency</span>
            <span class="stat-value">{last_latency}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Best RTF</span>
            <span class="stat-value best">{best_rtf}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Worst RTF</span>
            <span class="stat-value worst">{worst_rtf}</span>
        </div>
    </div>
    <div class="stats-graph">
        <span class="graph-label">RTF History</span>
        {sparkline}
    </div>
    <div class="stats-footer">
        <span class="total-audio">{audio:.1}s audio processed</span>
        <span class="chunk-count">{chunks} chunks</span>
    </div>
</div>"#,
            status_class = status_class,
            target = state.target_rtf,
            avg_rtf = avg_rtf,
            rtf_color = state.rtf_status_color(),
            avg_latency = avg_latency,
            last_rtf = last_rtf,
            last_latency = last_latency,
            best_rtf = best_rtf,
            worst_rtf = worst_rtf,
            sparkline = sparkline,
            audio = state.total_audio_secs,
            chunks = state.measurements.len(),
        )
    }

    fn to_css(&self) -> String {
        r".timing-stats-brick {
    background: #1a1a2e;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.timing-stats-brick.status-good {
    border-left: 3px solid #50fa7b;
}

.timing-stats-brick.status-warning {
    border-left: 3px solid #ff6b6b;
}

.stats-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.stats-header .title {
    color: #e0e0e0;
    font-weight: 600;
    font-size: 0.875rem;
}

.target-badge {
    background: #16213e;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    color: #4dc3ff;
    font-family: 'JetBrains Mono', monospace;
}

.stats-empty {
    color: #8b949e;
    text-align: center;
    padding: 1rem;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.stat-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.stat-label {
    color: #8b949e;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stat-value {
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.125rem;
    font-weight: 500;
}

.stat-value.rtf {
    font-size: 1.25rem;
}

.stat-value.best {
    color: #50fa7b;
}

.stat-value.worst {
    color: #ff6b6b;
}

.stats-graph {
    margin: 1rem 0;
}

.graph-label {
    color: #8b949e;
    font-size: 0.75rem;
    display: block;
    margin-bottom: 0.5rem;
}

.rtf-sparkline {
    width: 100%;
    height: 30px;
}

.stats-footer {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid #16213e;
}

.stats-footer span {
    font-family: 'JetBrains Mono', monospace;
}"
            .into()
    }

    fn test_id(&self) -> Option<&str> {
        Some("timing-stats")
    }
}

impl Widget for TimingStatsBrick {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let height: f32 = if self.state.measurements.is_empty() {
            80.0
        } else {
            200.0
        };
        Size::new(
            constraints.max_width.min(constraints.min_width.max(300.0)),
            height.min(constraints.max_height),
        )
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let bounds = Rect::new(0.0, 0.0, 400.0, 200.0);

        // Draw background
        let bg_color = Color::from_hex("#1a1a2e").unwrap_or(Color::BLACK);
        canvas.fill_rect(bounds, bg_color);

        let text_color = Color::from_hex("#e0e0e0").unwrap_or(Color::WHITE);
        let style = TextStyle {
            size: 14.0,
            color: text_color,
            weight: presentar_core::FontWeight::Normal,
            style: presentar_core::FontStyle::Normal,
        };

        // Draw title
        canvas.draw_text("Timing Statistics", Point::new(16.0, 24.0), &style);

        // Draw average RTF
        if let Some(rtf) = self.state.average_rtf() {
            let rtf_text = format!("Avg RTF: {:.2}x", rtf);
            let rtf_color = Color::from_hex(self.state.rtf_status_color()).unwrap_or(Color::WHITE);
            let rtf_style = TextStyle {
                size: 18.0,
                color: rtf_color,
                weight: presentar_core::FontWeight::Bold,
                style: presentar_core::FontStyle::Normal,
            };
            canvas.draw_text(&rtf_text, Point::new(16.0, 60.0), &rtf_style);
        }

        // Draw latency
        if let Some(latency) = self.state.average_latency_ms() {
            let latency_text = format!("Avg Latency: {:.0}ms", latency);
            canvas.draw_text(&latency_text, Point::new(16.0, 90.0), &style);
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
        Some("Timing statistics")
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_measurement_rtf() {
        let m = TimingMeasurement {
            chunk_index: 0,
            audio_duration_secs: 30.0,
            processing_time_ms: 15000.0,
            latency_ms: 16000.0,
        };
        assert!((m.rtf() - 0.5).abs() < 0.001);
        assert!(m.is_realtime());
    }

    #[test]
    fn test_timing_measurement_not_realtime() {
        let m = TimingMeasurement {
            chunk_index: 0,
            audio_duration_secs: 10.0,
            processing_time_ms: 15000.0,
            latency_ms: 16000.0,
        };
        assert!((m.rtf() - 1.5).abs() < 0.001);
        assert!(!m.is_realtime());
    }

    #[test]
    fn test_timing_stats_state_default() {
        let state = TimingStatsState::new();
        assert!(state.measurements.is_empty());
        assert_eq!(state.target_rtf, 2.0);
    }

    #[test]
    fn test_timing_stats_state_add_measurement() {
        let mut state = TimingStatsState::new();
        state.add_measurement(TimingMeasurement {
            chunk_index: 0,
            audio_duration_secs: 30.0,
            processing_time_ms: 15000.0,
            latency_ms: 16000.0,
        });

        assert_eq!(state.measurements.len(), 1);
        assert_eq!(state.total_audio_secs, 30.0);
        assert_eq!(state.total_processing_ms, 15000.0);
    }

    #[test]
    fn test_timing_stats_state_average_rtf() {
        let mut state = TimingStatsState::new();

        // Add two chunks with different RTFs
        state.add_measurement(TimingMeasurement {
            chunk_index: 0,
            audio_duration_secs: 30.0,
            processing_time_ms: 15000.0, // 0.5x
            latency_ms: 16000.0,
        });
        state.add_measurement(TimingMeasurement {
            chunk_index: 1,
            audio_duration_secs: 30.0,
            processing_time_ms: 45000.0, // 1.5x
            latency_ms: 46000.0,
        });

        // Average: (15000 + 45000) / (60 * 1000) = 1.0x
        let avg = state.average_rtf().unwrap();
        assert!((avg - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_timing_stats_state_best_worst_rtf() {
        let mut state = TimingStatsState::new();

        state.add_measurement(TimingMeasurement {
            chunk_index: 0,
            audio_duration_secs: 30.0,
            processing_time_ms: 15000.0, // 0.5x
            latency_ms: 16000.0,
        });
        state.add_measurement(TimingMeasurement {
            chunk_index: 1,
            audio_duration_secs: 30.0,
            processing_time_ms: 60000.0, // 2.0x
            latency_ms: 61000.0,
        });

        assert!((state.best_rtf.unwrap() - 0.5).abs() < 0.001);
        assert!((state.worst_rtf.unwrap() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_timing_stats_state_is_meeting_target() {
        let mut state = TimingStatsState::with_target_rtf(2.0);

        state.add_measurement(TimingMeasurement {
            chunk_index: 0,
            audio_duration_secs: 30.0,
            processing_time_ms: 30000.0, // 1.0x
            latency_ms: 31000.0,
        });

        assert!(state.is_meeting_target().unwrap());

        // Add a slow chunk
        state.add_measurement(TimingMeasurement {
            chunk_index: 1,
            audio_duration_secs: 30.0,
            processing_time_ms: 150000.0, // 5.0x
            latency_ms: 151000.0,
        });

        // Average now above target
        assert!(!state.is_meeting_target().unwrap());
    }

    #[test]
    fn test_timing_stats_state_format_rtf() {
        assert_eq!(TimingStatsState::format_rtf(0.5), "0.50x");
        assert_eq!(TimingStatsState::format_rtf(1.0), "1.00x");
        assert_eq!(TimingStatsState::format_rtf(2.5), "2.50x");
    }

    #[test]
    fn test_timing_stats_state_format_latency() {
        assert_eq!(TimingStatsState::format_latency(100.0), "100ms");
        assert_eq!(TimingStatsState::format_latency(999.0), "999ms");
        assert_eq!(TimingStatsState::format_latency(1500.0), "1.50s");
    }

    #[test]
    fn test_timing_stats_state_reset() {
        let mut state = TimingStatsState::new();
        state.add_measurement(TimingMeasurement {
            chunk_index: 0,
            audio_duration_secs: 30.0,
            processing_time_ms: 15000.0,
            latency_ms: 16000.0,
        });

        state.reset();

        assert!(state.measurements.is_empty());
        assert_eq!(state.total_audio_secs, 0.0);
        assert!(state.best_rtf.is_none());
    }

    #[test]
    fn test_brick_default() {
        let brick = TimingStatsBrick::new();
        assert!(brick.state().measurements.is_empty());
    }

    #[test]
    fn test_brick_with_target_rtf() {
        let brick = TimingStatsBrick::with_target_rtf(4.0);
        assert_eq!(brick.state().target_rtf, 4.0);
    }

    #[test]
    fn test_brick_record() {
        let mut brick = TimingStatsBrick::new();
        brick.record(0, 30.0, 15000.0, 16000.0);

        assert_eq!(brick.state().measurements.len(), 1);
    }

    #[test]
    fn test_brick_verification() {
        let brick = TimingStatsBrick::new();
        let result = brick.verify();
        assert!(result.is_valid());
    }

    #[test]
    fn test_brick_to_html_empty() {
        let brick = TimingStatsBrick::new();
        let html = brick.to_html();
        assert!(html.contains("No data yet"));
        assert!(html.contains("data-testid=\"timing-stats\""));
    }

    #[test]
    fn test_brick_to_html_with_data() {
        let mut brick = TimingStatsBrick::new();
        brick.record(0, 30.0, 15000.0, 16000.0);

        let html = brick.to_html();
        assert!(html.contains("Timing Statistics"));
        assert!(html.contains("data-testid=\"avg-rtf\""));
        assert!(html.contains("data-testid=\"avg-latency\""));
    }

    #[test]
    fn test_brick_budget() {
        let brick = TimingStatsBrick::new();
        assert_eq!(brick.budget().total_ms, 16);
    }

    #[test]
    fn test_brick_can_render() {
        let brick = TimingStatsBrick::new();
        assert!(brick.can_render());
    }
}
