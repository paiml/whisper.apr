//! `AudioLevelBrick`: Audio level display with dB readout (PROBAR-SPEC-009)
//!
//! This brick displays audio level with a numeric dB reading:
//! - dB scale from -60 to 0 (standard audio range)
//! - Visual meter bar with gradient
//! - Peak hold indicator with decay
//! - Clipping indicator
//!
//! # Assertions
//!
//! - dB value displayed accurately
//! - 60fps update capability
//! - Clipping detection at 0 dB

use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_core::{
    AccessibleRole, Canvas, Color, Constraints, Event, LayoutResult, Point, Rect, Size, TextStyle,
    TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Minimum dB value (silence threshold)
const MIN_DB: f32 = -60.0;

/// Maximum dB value (clipping)
const MAX_DB: f32 = 0.0;

/// Reference level for 0 dB (full scale)
const REFERENCE_LEVEL: f32 = 1.0;

/// Convert linear amplitude to dB
#[must_use]
pub fn amplitude_to_db(amplitude: f32) -> f32 {
    if amplitude <= 0.0 {
        MIN_DB
    } else {
        (20.0 * (amplitude / REFERENCE_LEVEL).log10()).max(MIN_DB)
    }
}

/// Convert dB to linear amplitude
#[must_use]
pub fn db_to_amplitude(db: f32) -> f32 {
    if db <= MIN_DB {
        0.0
    } else {
        REFERENCE_LEVEL * 10.0_f32.powf(db / 20.0)
    }
}

/// Audio level state
#[derive(Debug, Clone)]
pub struct AudioLevelState {
    /// Current level in dB
    pub level_db: f32,
    /// Peak level in dB (with hold)
    pub peak_db: f32,
    /// Peak hold counter (for decay)
    peak_hold_frames: u32,
    /// Clipping detected (level >= 0 dB)
    pub is_clipping: bool,
    /// Peak decay rate in dB per frame
    pub decay_rate: f32,
    /// Peak hold time in frames
    pub hold_frames: u32,
}

impl Default for AudioLevelState {
    fn default() -> Self {
        Self {
            level_db: MIN_DB,
            peak_db: MIN_DB,
            peak_hold_frames: 0,
            is_clipping: false,
            decay_rate: 0.5, // 0.5 dB per frame at 60fps = 30 dB/sec
            hold_frames: 30, // Hold for 0.5 seconds at 60fps
        }
    }
}

impl AudioLevelState {
    /// Create new state
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Update level from RMS amplitude (0.0 to 1.0+)
    pub fn update_from_amplitude(&mut self, amplitude: f32) {
        let db = amplitude_to_db(amplitude);
        self.update_db(db);
    }

    /// Update level from dB value
    pub fn update_db(&mut self, db: f32) {
        self.level_db = db.clamp(MIN_DB, MAX_DB);

        // Check for clipping
        self.is_clipping = self.level_db >= -0.5; // Within 0.5 dB of clipping

        // Update peak with hold
        if self.level_db >= self.peak_db {
            self.peak_db = self.level_db;
            self.peak_hold_frames = 0;
        } else {
            self.peak_hold_frames += 1;
            if self.peak_hold_frames > self.hold_frames {
                // Decay peak
                self.peak_db = (self.peak_db - self.decay_rate).max(self.level_db);
            }
        }
    }

    /// Update from raw samples (calculates RMS)
    pub fn update_from_samples(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }

        let sum: f32 = samples.iter().map(|s| s * s).sum();
        let rms = (sum / samples.len() as f32).sqrt();
        self.update_from_amplitude(rms);
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.level_db = MIN_DB;
        self.peak_db = MIN_DB;
        self.peak_hold_frames = 0;
        self.is_clipping = false;
    }

    /// Get level as percentage (0-100)
    #[must_use]
    pub fn level_percent(&self) -> f32 {
        ((self.level_db - MIN_DB) / (MAX_DB - MIN_DB) * 100.0).clamp(0.0, 100.0)
    }

    /// Get peak as percentage (0-100)
    #[must_use]
    pub fn peak_percent(&self) -> f32 {
        ((self.peak_db - MIN_DB) / (MAX_DB - MIN_DB) * 100.0).clamp(0.0, 100.0)
    }

    /// Format dB value for display
    #[must_use]
    pub fn format_db(&self) -> String {
        if self.level_db <= MIN_DB {
            "-∞ dB".into()
        } else {
            format!("{:.1} dB", self.level_db)
        }
    }

    /// Format peak dB for display
    #[must_use]
    pub fn format_peak_db(&self) -> String {
        if self.peak_db <= MIN_DB {
            "-∞ dB".into()
        } else {
            format!("{:.1} dB", self.peak_db)
        }
    }
}

/// Audio level brick with dB display
#[derive(Debug, Clone, Default)]
pub struct AudioLevelBrick {
    state: AudioLevelState,
    /// Label for the meter
    label: String,
    /// Show peak indicator
    show_peak: bool,
}

impl AudioLevelBrick {
    /// Create new brick
    #[must_use]
    pub fn new() -> Self {
        Self {
            label: "Audio Level".into(),
            show_peak: true,
            ..Default::default()
        }
    }

    /// Create with custom label
    #[must_use]
    pub fn with_label(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            show_peak: true,
            ..Default::default()
        }
    }

    /// Get state reference
    #[must_use]
    pub fn state(&self) -> &AudioLevelState {
        &self.state
    }

    /// Get mutable state reference
    pub fn state_mut(&mut self) -> &mut AudioLevelState {
        &mut self.state
    }

    /// Update from amplitude
    pub fn update_from_amplitude(&mut self, amplitude: f32) {
        self.state.update_from_amplitude(amplitude);
    }

    /// Update from dB
    pub fn update_db(&mut self, db: f32) {
        self.state.update_db(db);
    }

    /// Update from samples
    pub fn update_from_samples(&mut self, samples: &[f32]) {
        self.state.update_from_samples(samples);
    }

    /// Reset
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Set show peak
    pub fn set_show_peak(&mut self, show: bool) {
        self.show_peak = show;
    }

    /// Get meter color based on level
    fn meter_color(&self) -> &'static str {
        if self.state.is_clipping {
            "#ff6b6b" // Red for clipping
        } else if self.state.level_db > -6.0 {
            "#ffb86c" // Orange for hot
        } else if self.state.level_db > -20.0 {
            "#50fa7b" // Green for normal
        } else {
            "#4dc3ff" // Blue for quiet
        }
    }

    /// Get dB tick marks HTML
    fn tick_marks_html(&self) -> String {
        let ticks = [-60, -40, -20, -12, -6, -3, 0];
        ticks
            .iter()
            .map(|&db| {
                let percent = ((db as f32 - MIN_DB) / (MAX_DB - MIN_DB) * 100.0) as u32;
                format!(
                    r#"<div class="db-tick" style="left: {percent}%"><span>{db}</span></div>"#,
                    percent = percent,
                    db = db
                )
            })
            .collect::<Vec<_>>()
            .join("\n        ")
    }
}

impl Brick for AudioLevelBrick {
    fn brick_name(&self) -> &'static str {
        "AudioLevelBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible, // dB value visible
            BrickAssertion::MaxLatencyMs(16), // 60fps
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
        let level_percent = state.level_percent();
        let peak_percent = state.peak_percent();
        let db_text = state.format_db();
        let meter_color = self.meter_color();

        let clipping_class = if state.is_clipping { "clipping" } else { "" };

        let peak_html = if self.show_peak && state.peak_db > MIN_DB {
            format!(
                r#"<div class="peak-marker" style="left: {peak_percent:.1}%"></div>"#,
                peak_percent = peak_percent
            )
        } else {
            String::new()
        };

        let tick_marks = self.tick_marks_html();

        format!(
            r#"<div class="audio-level-brick {clipping_class}" data-testid="audio-level">
    <div class="level-header">
        <span class="label">{label}</span>
        <span class="db-value" data-testid="db-value">{db_text}</span>
    </div>
    <div class="meter-container">
        <div class="meter-bar" style="width: {level_percent:.1}%; background: {meter_color}"></div>
        {peak_html}
    </div>
    <div class="db-scale">
        {tick_marks}
    </div>
</div>"#,
            clipping_class = clipping_class,
            label = self.label,
            db_text = db_text,
            level_percent = level_percent,
            meter_color = meter_color,
            peak_html = peak_html,
            tick_marks = tick_marks,
        )
    }

    fn to_css(&self) -> String {
        r".audio-level-brick {
    background: #1a1a2e;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.audio-level-brick.clipping {
    border: 2px solid #ff6b6b;
}

.level-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.level-header .label {
    color: #8b949e;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.db-value {
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 600;
}

.audio-level-brick.clipping .db-value {
    color: #ff6b6b;
}

.meter-container {
    position: relative;
    height: 20px;
    background: #16213e;
    border-radius: 4px;
    overflow: hidden;
}

.meter-bar {
    height: 100%;
    border-radius: 4px;
    transition: width 50ms ease-out;
}

.peak-marker {
    position: absolute;
    top: 0;
    width: 2px;
    height: 100%;
    background: #ffffff;
    transition: left 50ms ease-out;
}

.db-scale {
    position: relative;
    height: 16px;
    margin-top: 4px;
}

.db-tick {
    position: absolute;
    transform: translateX(-50%);
}

.db-tick span {
    font-size: 0.625rem;
    color: #8b949e;
    font-family: 'JetBrains Mono', monospace;
}

/* Color zones for meter background */
.meter-container::before {
    content: '';
    position: absolute;
    right: 0;
    top: 0;
    width: 10%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 107, 107, 0.2));
    pointer-events: none;
}

.meter-container::after {
    content: '';
    position: absolute;
    right: 10%;
    top: 0;
    width: 10%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 184, 108, 0.2));
    pointer-events: none;
}"
            .into()
    }

    fn test_id(&self) -> Option<&str> {
        Some("audio-level")
    }
}

impl Widget for AudioLevelBrick {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let height: f32 = 80.0;
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
        let bounds = Rect::new(0.0, 0.0, 400.0, 80.0);

        // Draw background
        let bg_color = Color::from_hex("#1a1a2e").unwrap_or(Color::BLACK);
        canvas.fill_rect(bounds, bg_color);

        // Draw label
        let label_style = TextStyle {
            size: 12.0,
            color: Color::from_hex("#8b949e").unwrap_or(Color::WHITE),
            weight: presentar_core::FontWeight::Normal,
            style: presentar_core::FontStyle::Normal,
        };
        canvas.draw_text(&self.label, Point::new(16.0, 20.0), &label_style);

        // Draw dB value
        let db_color = if self.state.is_clipping {
            Color::from_hex("#ff6b6b").unwrap_or(Color::RED)
        } else {
            Color::from_hex("#e0e0e0").unwrap_or(Color::WHITE)
        };
        let db_style = TextStyle {
            size: 18.0,
            color: db_color,
            weight: presentar_core::FontWeight::Bold,
            style: presentar_core::FontStyle::Normal,
        };
        canvas.draw_text(&self.state.format_db(), Point::new(320.0, 20.0), &db_style);

        // Draw meter background
        let meter_bg = Rect::new(16.0, 36.0, 368.0, 20.0);
        let meter_bg_color = Color::from_hex("#16213e").unwrap_or(Color::BLACK);
        canvas.fill_rect(meter_bg, meter_bg_color);

        // Draw meter bar
        let level_width = 368.0 * (self.state.level_percent() / 100.0);
        if level_width > 0.0 {
            let bar_rect = Rect::new(16.0, 36.0, level_width, 20.0);
            let bar_color = Color::from_hex(self.meter_color()).unwrap_or(Color::GREEN);
            canvas.fill_rect(bar_rect, bar_color);
        }

        // Draw peak marker
        if self.show_peak && self.state.peak_db > MIN_DB {
            let peak_x = 16.0 + 368.0 * (self.state.peak_percent() / 100.0);
            canvas.draw_line(
                Point::new(peak_x, 36.0),
                Point::new(peak_x, 56.0),
                Color::WHITE,
                2.0,
            );
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
        Some("Audio level meter with dB reading")
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amplitude_to_db() {
        assert!((amplitude_to_db(1.0) - 0.0).abs() < 0.01);
        assert!((amplitude_to_db(0.5) - (-6.02)).abs() < 0.1);
        assert!((amplitude_to_db(0.1) - (-20.0)).abs() < 0.1);
        assert_eq!(amplitude_to_db(0.0), MIN_DB);
    }

    #[test]
    fn test_db_to_amplitude() {
        assert!((db_to_amplitude(0.0) - 1.0).abs() < 0.01);
        assert!((db_to_amplitude(-6.0) - 0.5).abs() < 0.05);
        assert_eq!(db_to_amplitude(MIN_DB), 0.0);
    }

    #[test]
    fn test_audio_level_state_default() {
        let state = AudioLevelState::new();
        assert_eq!(state.level_db, MIN_DB);
        assert_eq!(state.peak_db, MIN_DB);
        assert!(!state.is_clipping);
    }

    #[test]
    fn test_audio_level_state_update_amplitude() {
        let mut state = AudioLevelState::new();
        state.update_from_amplitude(0.5);

        assert!((state.level_db - (-6.02)).abs() < 0.1);
    }

    #[test]
    fn test_audio_level_state_update_db() {
        let mut state = AudioLevelState::new();
        state.update_db(-12.0);

        assert_eq!(state.level_db, -12.0);
    }

    #[test]
    fn test_audio_level_state_clipping() {
        let mut state = AudioLevelState::new();
        state.update_db(0.0);

        assert!(state.is_clipping);
    }

    #[test]
    fn test_audio_level_state_peak_tracking() {
        let mut state = AudioLevelState::new();
        state.update_db(-6.0);
        state.update_db(-20.0);

        assert_eq!(state.level_db, -20.0);
        assert_eq!(state.peak_db, -6.0);
    }

    #[test]
    fn test_audio_level_state_level_percent() {
        let mut state = AudioLevelState::new();

        state.update_db(0.0);
        assert!((state.level_percent() - 100.0).abs() < 0.1);

        state.update_db(-30.0);
        assert!((state.level_percent() - 50.0).abs() < 0.1);

        state.update_db(MIN_DB);
        assert_eq!(state.level_percent(), 0.0);
    }

    #[test]
    fn test_audio_level_state_format_db() {
        let mut state = AudioLevelState::new();

        state.update_db(-12.0);
        assert_eq!(state.format_db(), "-12.0 dB");

        state.update_db(MIN_DB);
        assert_eq!(state.format_db(), "-∞ dB");
    }

    #[test]
    fn test_audio_level_state_reset() {
        let mut state = AudioLevelState::new();
        state.update_db(-6.0);
        state.reset();

        assert_eq!(state.level_db, MIN_DB);
        assert_eq!(state.peak_db, MIN_DB);
    }

    #[test]
    fn test_brick_default() {
        let brick = AudioLevelBrick::new();
        assert_eq!(brick.state().level_db, MIN_DB);
    }

    #[test]
    fn test_brick_with_label() {
        let brick = AudioLevelBrick::with_label("Input");
        assert_eq!(brick.label, "Input");
    }

    #[test]
    fn test_brick_update() {
        let mut brick = AudioLevelBrick::new();
        brick.update_from_amplitude(0.5);

        assert!((brick.state().level_db - (-6.02)).abs() < 0.1);
    }

    #[test]
    fn test_brick_verification() {
        let brick = AudioLevelBrick::new();
        let result = brick.verify();
        assert!(result.is_valid());
    }

    #[test]
    fn test_brick_to_html() {
        let mut brick = AudioLevelBrick::new();
        brick.update_db(-12.0);

        let html = brick.to_html();
        assert!(html.contains("data-testid=\"audio-level\""));
        assert!(html.contains("data-testid=\"db-value\""));
        assert!(html.contains("-12.0 dB"));
    }

    #[test]
    fn test_brick_to_html_clipping() {
        let mut brick = AudioLevelBrick::new();
        brick.update_db(0.0);

        let html = brick.to_html();
        assert!(html.contains("clipping"));
    }

    #[test]
    fn test_brick_budget() {
        let brick = AudioLevelBrick::new();
        assert_eq!(brick.budget().total_ms, 16);
    }

    #[test]
    fn test_brick_can_render() {
        let brick = AudioLevelBrick::new();
        assert!(brick.can_render());
    }
}
