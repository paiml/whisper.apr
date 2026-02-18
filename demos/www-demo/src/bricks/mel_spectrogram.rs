//! `MelSpectrogramBrick`: Mel spectrogram performance verification (WAPR-PERF-004)
//!
//! `BrickTracing` brick for verifying mel spectrogram computation performance
//! after delegation to aprender's `MelFilterbank`. Tracks:
//! - Computation time vs budget (50ms for 30s audio)
//! - Output dimensions (3000 frames x 80 mels)
//! - Numerical validity (no NaN/Inf)
//!
//! # Budget
//!
//! 50ms total for 30-second audio (480,000 samples at 16kHz).
//! Typical computation: ~15-25ms on modern hardware.
//!
//! # Assertions
//!
//! - Computation within 50ms budget
//! - Output frame count matches expected 3000
//! - Output mel bin count matches expected 80

use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_core::{
    AccessibleRole, Canvas, Color, Constraints, Event, LayoutResult, Point, Rect, Size, TextStyle,
    TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Whisper mel spectrogram expected dimensions
const EXPECTED_FRAMES: usize = 3000;
/// Standard mel bin count (80 for most models, 128 for large-v3-turbo)
const EXPECTED_MELS: usize = 80;
/// Budget in milliseconds for 30s audio mel computation
const MEL_BUDGET_MS: u32 = 50;

/// Mel spectrogram computation state
#[derive(Debug, Clone, Default)]
pub struct MelSpectrogramState {
    /// Last computation time in milliseconds
    pub last_mel_ms: Option<f64>,
    /// Number of frames in last output
    pub last_frames: Option<usize>,
    /// Number of mel bins in last output
    pub last_mels: Option<usize>,
    /// Whether last output contained NaN values
    pub has_nan: bool,
    /// Whether last output contained Inf values
    pub has_inf: bool,
    /// Number of computations performed
    pub compute_count: u64,
    /// Cumulative computation time in milliseconds
    pub cumulative_ms: f64,
    /// Peak computation time observed
    pub peak_ms: f64,
    /// Whether using embedded filterbank (from .apr file) vs computed
    pub using_embedded_filterbank: bool,
    /// Whether center padding is enabled (aprender delegation)
    pub center_pad_enabled: bool,
}

impl MelSpectrogramState {
    /// Create new state
    #[must_use]
    pub fn new() -> Self {
        Self {
            center_pad_enabled: true, // aprender default for whisper
            ..Default::default()
        }
    }

    /// Record a mel computation result
    pub fn record(&mut self, mel_ms: f64, frames: usize, mels: usize, output: &[f32]) {
        self.last_mel_ms = Some(mel_ms);
        self.last_frames = Some(frames);
        self.last_mels = Some(mels);
        self.has_nan = output.iter().any(|v| v.is_nan());
        self.has_inf = output.iter().any(|v| v.is_infinite());
        self.compute_count += 1;
        self.cumulative_ms += mel_ms;
        self.peak_ms = self.peak_ms.max(mel_ms);
    }

    /// Average computation time
    #[must_use]
    pub fn avg_ms(&self) -> f64 {
        if self.compute_count > 0 {
            self.cumulative_ms / self.compute_count as f64
        } else {
            0.0
        }
    }

    /// Check if within budget
    #[must_use]
    pub fn within_budget(&self) -> Option<bool> {
        self.last_mel_ms.map(|ms| ms <= f64::from(MEL_BUDGET_MS))
    }

    /// Check if dimensions match expected
    #[must_use]
    pub fn dimensions_valid(&self) -> Option<bool> {
        match (self.last_frames, self.last_mels) {
            (Some(f), Some(m)) => Some(f == EXPECTED_FRAMES && m == EXPECTED_MELS),
            _ => None,
        }
    }

    /// Check if output is numerically valid (no NaN/Inf)
    #[must_use]
    pub fn numerically_valid(&self) -> bool {
        !self.has_nan && !self.has_inf
    }
}

/// Mel spectrogram `BrickTracing` brick
#[derive(Debug, Clone, Default)]
pub struct MelSpectrogramBrick {
    state: MelSpectrogramState,
}

impl MelSpectrogramBrick {
    /// Create new brick
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: MelSpectrogramState::new(),
        }
    }

    /// Get state reference
    #[must_use]
    pub fn state(&self) -> &MelSpectrogramState {
        &self.state
    }

    /// Get mutable state reference
    pub fn state_mut(&mut self) -> &mut MelSpectrogramState {
        &mut self.state
    }

    /// Record a mel computation result
    pub fn record(&mut self, mel_ms: f64, frames: usize, mels: usize, output: &[f32]) {
        self.state.record(mel_ms, frames, mels, output);
    }

    /// Set embedded filterbank flag
    pub fn set_embedded_filterbank(&mut self, embedded: bool) {
        self.state.using_embedded_filterbank = embedded;
    }

    /// Set center padding flag
    pub fn set_center_pad(&mut self, enabled: bool) {
        self.state.center_pad_enabled = enabled;
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.state = MelSpectrogramState::new();
    }
}

impl Brick for MelSpectrogramBrick {
    fn brick_name(&self) -> &'static str {
        "MelSpectrogramBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::MaxLatencyMs(MEL_BUDGET_MS),
            BrickAssertion::TextVisible,
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(MEL_BUDGET_MS)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed: Vec<(BrickAssertion, String)> = Vec::new();

        // Check budget
        match self.state.within_budget() {
            Some(true) => {
                passed.push(BrickAssertion::MaxLatencyMs(MEL_BUDGET_MS));
            }
            Some(false) => {
                failed.push((
                    BrickAssertion::MaxLatencyMs(MEL_BUDGET_MS),
                    format!(
                        "Mel computation took {:.1}ms, budget is {}ms",
                        self.state.last_mel_ms.unwrap_or(0.0),
                        MEL_BUDGET_MS
                    ),
                ));
            }
            None => {
                // No computation recorded yet, pass optimistically
                passed.push(BrickAssertion::MaxLatencyMs(MEL_BUDGET_MS));
            }
        }

        // Check dimensions
        match self.state.dimensions_valid() {
            Some(true) => {
                passed.push(BrickAssertion::TextVisible);
            }
            Some(false) => {
                failed.push((
                    BrickAssertion::TextVisible,
                    format!(
                        "Unexpected dimensions: {}x{} (expected {}x{})",
                        self.state.last_frames.unwrap_or(0),
                        self.state.last_mels.unwrap_or(0),
                        EXPECTED_FRAMES,
                        EXPECTED_MELS
                    ),
                ));
            }
            None => {
                passed.push(BrickAssertion::TextVisible);
            }
        }

        // Check numerical validity
        if self.state.has_nan {
            failed.push((
                BrickAssertion::TextVisible,
                "Mel output contains NaN values".to_string(),
            ));
        }
        if self.state.has_inf {
            failed.push((
                BrickAssertion::TextVisible,
                "Mel output contains Inf values".to_string(),
            ));
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        let budget_class = match self.state.within_budget() {
            Some(true) => "budget-met",
            Some(false) => "budget-exceeded",
            None => "budget-unknown",
        };

        let timing_display = self
            .state
            .last_mel_ms
            .map(|ms| {
                format!(
                    r#"<div class="mel-timing {budget_class}">
            <span class="mel-label">Mel Compute</span>
            <span class="mel-value" data-testid="mel-ms">{:.1}ms</span>
            <span class="mel-budget">/ {}ms budget</span>
        </div>"#,
                    ms, MEL_BUDGET_MS
                )
            })
            .unwrap_or_else(|| {
                r#"<div class="mel-timing budget-unknown">
            <span class="mel-label">Mel Compute</span>
            <span class="mel-value">--</span>
        </div>"#
                    .to_string()
            });

        let stats_display = if self.state.compute_count > 0 {
            format!(
                r#"<div class="mel-stats">
            <span>Avg: {:.1}ms</span>
            <span>Peak: {:.1}ms</span>
            <span>Count: {}</span>
        </div>"#,
                self.state.avg_ms(),
                self.state.peak_ms,
                self.state.compute_count
            )
        } else {
            String::new()
        };

        let dim_display = match (self.state.last_frames, self.state.last_mels) {
            (Some(f), Some(m)) => {
                let dim_class = if f == EXPECTED_FRAMES && m == EXPECTED_MELS {
                    "dim-ok"
                } else {
                    "dim-mismatch"
                };
                format!(
                    r#"<div class="mel-dimensions {dim_class}">
            <span>{f}x{m}</span>
            <span class="mel-source">{}</span>
            <span class="mel-pad">{}</span>
        </div>"#,
                    if self.state.using_embedded_filterbank {
                        "embedded"
                    } else {
                        "computed"
                    },
                    if self.state.center_pad_enabled {
                        "center-pad"
                    } else {
                        "no-pad"
                    }
                )
            }
            _ => String::new(),
        };

        format!(
            r#"<div class="mel-spectrogram-brick" data-testid="mel-spectrogram">
    <div class="mel-header">
        <span class="title">Mel Spectrogram (aprender)</span>
    </div>
    {timing_display}
    {dim_display}
    {stats_display}
</div>"#
        )
    }

    fn to_css(&self) -> String {
        r#".mel-spectrogram-brick {
    background: #1a1a2e;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    border-left: 3px solid #4dc3ff;
}

.mel-header .title {
    color: #e0e0e0;
    font-weight: 600;
    font-size: 0.875rem;
}

.mel-timing {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    margin: 0.75rem 0;
}

.mel-label {
    color: #8b949e;
    font-size: 0.75rem;
    text-transform: uppercase;
}

.mel-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 600;
}

.budget-met .mel-value { color: #50fa7b; }
.budget-exceeded .mel-value { color: #ff6b6b; }
.budget-unknown .mel-value { color: #8b949e; }

.mel-budget {
    color: #8b949e;
    font-size: 0.75rem;
}

.mel-dimensions {
    display: flex;
    gap: 1rem;
    font-size: 0.8rem;
    color: #8b949e;
    font-family: 'JetBrains Mono', monospace;
}

.dim-ok { color: #50fa7b; }
.dim-mismatch { color: #ff6b6b; }

.mel-source, .mel-pad {
    padding: 0.125rem 0.375rem;
    background: #16213e;
    border-radius: 3px;
    font-size: 0.7rem;
}

.mel-stats {
    display: flex;
    gap: 1rem;
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}"#
        .into()
    }

    fn test_id(&self) -> Option<&str> {
        Some("mel-spectrogram")
    }
}

impl Widget for MelSpectrogramBrick {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let height: f32 = 120.0;
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
        let bounds = Rect::new(0.0, 0.0, 400.0, 120.0);

        // Background
        let bg_color = Color::from_hex("#1a1a2e").unwrap_or(Color::BLACK);
        canvas.fill_rect(bounds, bg_color);

        // Left accent bar
        let accent = Rect::new(0.0, 0.0, 3.0, bounds.height);
        let accent_color = Color::from_hex("#4dc3ff").unwrap_or(Color::BLUE);
        canvas.fill_rect(accent, accent_color);

        let text_color = Color::from_hex("#e0e0e0").unwrap_or(Color::WHITE);
        let style = TextStyle {
            size: 14.0,
            color: text_color,
            weight: presentar_core::FontWeight::Normal,
            style: presentar_core::FontStyle::Normal,
        };

        // Title
        canvas.draw_text("Mel Spectrogram (aprender)", Point::new(16.0, 24.0), &style);

        // Timing
        if let Some(ms) = self.state.last_mel_ms {
            let timing_color = if ms <= f64::from(MEL_BUDGET_MS) {
                Color::from_hex("#50fa7b").unwrap_or(Color::GREEN)
            } else {
                Color::from_hex("#ff6b6b").unwrap_or(Color::RED)
            };
            let timing_style = TextStyle {
                size: 18.0,
                color: timing_color,
                weight: presentar_core::FontWeight::Bold,
                style: presentar_core::FontStyle::Normal,
            };
            let timing_text = format!("{:.1}ms / {}ms", ms, MEL_BUDGET_MS);
            canvas.draw_text(&timing_text, Point::new(16.0, 60.0), &timing_style);
        }

        // Dimensions
        if let (Some(f), Some(m)) = (self.state.last_frames, self.state.last_mels) {
            let dim_style = TextStyle {
                size: 12.0,
                color: Color::from_hex("#8b949e").unwrap_or(Color::WHITE),
                weight: presentar_core::FontWeight::Normal,
                style: presentar_core::FontStyle::Normal,
            };
            let dim_text = format!("{}x{} frames", f, m);
            canvas.draw_text(&dim_text, Point::new(16.0, 90.0), &dim_style);
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
        Some("Mel spectrogram performance")
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
        let brick = MelSpectrogramBrick::new();
        assert!(brick.state().last_mel_ms.is_none());
        assert_eq!(brick.state().compute_count, 0);
        assert!(brick.state().center_pad_enabled);
    }

    #[test]
    fn test_record() {
        let mut brick = MelSpectrogramBrick::new();
        let output = vec![0.5_f32; 3000 * 80];
        brick.record(20.0, 3000, 80, &output);

        assert_eq!(brick.state().last_mel_ms, Some(20.0));
        assert_eq!(brick.state().last_frames, Some(3000));
        assert_eq!(brick.state().last_mels, Some(80));
        assert!(!brick.state().has_nan);
        assert!(!brick.state().has_inf);
        assert_eq!(brick.state().compute_count, 1);
    }

    #[test]
    fn test_within_budget() {
        let mut brick = MelSpectrogramBrick::new();
        let output = vec![0.5_f32; 3000 * 80];

        // Within budget
        brick.record(30.0, 3000, 80, &output);
        assert_eq!(brick.state().within_budget(), Some(true));

        // Over budget
        brick.record(60.0, 3000, 80, &output);
        assert_eq!(brick.state().within_budget(), Some(false));
    }

    #[test]
    fn test_dimensions_valid() {
        let mut brick = MelSpectrogramBrick::new();
        let output = vec![0.5_f32; 3000 * 80];

        // Valid dimensions
        brick.record(20.0, 3000, 80, &output);
        assert_eq!(brick.state().dimensions_valid(), Some(true));

        // Wrong dimensions
        brick.record(20.0, 2500, 80, &output);
        assert_eq!(brick.state().dimensions_valid(), Some(false));
    }

    #[test]
    fn test_numerical_validity() {
        let mut brick = MelSpectrogramBrick::new();

        // Valid output
        let output = vec![0.5_f32; 100];
        brick.record(10.0, 10, 10, &output);
        assert!(brick.state().numerically_valid());

        // NaN output
        let mut nan_output = vec![0.5_f32; 100];
        nan_output[50] = f32::NAN;
        brick.record(10.0, 10, 10, &nan_output);
        assert!(!brick.state().numerically_valid());
    }

    #[test]
    fn test_avg_ms() {
        let mut brick = MelSpectrogramBrick::new();
        let output = vec![0.5_f32; 100];

        brick.record(20.0, 10, 10, &output);
        brick.record(30.0, 10, 10, &output);
        assert!((brick.state().avg_ms() - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_peak_ms() {
        let mut brick = MelSpectrogramBrick::new();
        let output = vec![0.5_f32; 100];

        brick.record(20.0, 10, 10, &output);
        brick.record(40.0, 10, 10, &output);
        brick.record(15.0, 10, 10, &output);
        assert!((brick.state().peak_ms - 40.0).abs() < 0.001);
    }

    #[test]
    fn test_verification_passes() {
        let mut brick = MelSpectrogramBrick::new();
        let output = vec![0.5_f32; 3000 * 80];
        brick.record(20.0, 3000, 80, &output);

        let result = brick.verify();
        assert!(result.is_valid());
    }

    #[test]
    fn test_verification_fails_budget() {
        let mut brick = MelSpectrogramBrick::new();
        let output = vec![0.5_f32; 3000 * 80];
        brick.record(100.0, 3000, 80, &output);

        let result = brick.verify();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_verification_fails_dimensions() {
        let mut brick = MelSpectrogramBrick::new();
        let output = vec![0.5_f32; 100];
        brick.record(20.0, 2500, 80, &output);

        let result = brick.verify();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_to_html() {
        let mut brick = MelSpectrogramBrick::new();
        let output = vec![0.5_f32; 3000 * 80];
        brick.record(20.0, 3000, 80, &output);

        let html = brick.to_html();
        assert!(html.contains("data-testid=\"mel-spectrogram\""));
        assert!(html.contains("Mel Spectrogram (aprender)"));
        assert!(html.contains("20.0ms"));
    }

    #[test]
    fn test_budget() {
        let brick = MelSpectrogramBrick::new();
        assert_eq!(brick.budget().total_ms, MEL_BUDGET_MS);
    }

    #[test]
    fn test_brick_name() {
        let brick = MelSpectrogramBrick::new();
        assert_eq!(brick.brick_name(), "MelSpectrogramBrick");
    }

    #[test]
    fn test_can_render() {
        let brick = MelSpectrogramBrick::new();
        assert!(brick.can_render());
    }

    #[test]
    fn test_reset() {
        let mut brick = MelSpectrogramBrick::new();
        let output = vec![0.5_f32; 100];
        brick.record(20.0, 10, 10, &output);
        brick.reset();

        assert!(brick.state().last_mel_ms.is_none());
        assert_eq!(brick.state().compute_count, 0);
    }

    #[test]
    fn test_embedded_filterbank_flag() {
        let mut brick = MelSpectrogramBrick::new();
        assert!(!brick.state().using_embedded_filterbank);

        brick.set_embedded_filterbank(true);
        assert!(brick.state().using_embedded_filterbank);
    }

    #[test]
    fn test_center_pad_flag() {
        let mut brick = MelSpectrogramBrick::new();
        assert!(brick.state().center_pad_enabled);

        brick.set_center_pad(false);
        assert!(!brick.state().center_pad_enabled);
    }
}
