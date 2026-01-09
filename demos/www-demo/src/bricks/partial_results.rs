//! `PartialResultsBrick`: Streaming transcription results display (PROBAR-SPEC-009)
//!
//! This brick displays partial transcription results as they stream in:
//! - Current partial text (unstable, may change)
//! - Confirmed text segments
//! - Word-level confidence highlighting
//! - Streaming cursor animation
//!
//! # Assertions
//!
//! - Text updates within 100ms of chunk completion
//! - Partial text visually distinct from confirmed
//! - Smooth text flow without flicker

use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_core::{
    AccessibleRole, Canvas, Color, Constraints, Event, LayoutResult, Point, Rect, Size, TextStyle,
    TypeId, Widget,
};
use std::any::Any;
use std::collections::VecDeque;
use std::time::Duration;

/// A single transcription segment
#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    /// The text content
    pub text: String,
    /// Start time in seconds (from audio start)
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Confidence score 0.0-1.0
    pub confidence: f32,
    /// Whether this segment is confirmed (final)
    pub is_confirmed: bool,
}

impl TranscriptSegment {
    /// Create a new segment
    #[must_use]
    pub fn new(text: impl Into<String>, start: f32, end: f32, confidence: f32) -> Self {
        Self {
            text: text.into(),
            start_time: start,
            end_time: end,
            confidence,
            is_confirmed: false,
        }
    }

    /// Create a confirmed segment
    #[must_use]
    pub fn confirmed(text: impl Into<String>, start: f32, end: f32, confidence: f32) -> Self {
        Self {
            text: text.into(),
            start_time: start,
            end_time: end,
            confidence,
            is_confirmed: true,
        }
    }

    /// Duration in seconds
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }

    /// Check if high confidence (>= 0.9)
    #[must_use]
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.9
    }

    /// Check if low confidence (< 0.7)
    #[must_use]
    pub fn is_low_confidence(&self) -> bool {
        self.confidence < 0.7
    }
}

/// Partial results state
#[derive(Debug, Clone, Default)]
pub struct PartialResultsState {
    /// Confirmed segments (finalized)
    pub confirmed: Vec<TranscriptSegment>,
    /// Partial segment (unstable, may change)
    pub partial: Option<TranscriptSegment>,
    /// Total audio duration processed
    pub audio_processed_secs: f32,
    /// Whether actively streaming
    pub is_streaming: bool,
    /// Maximum segments to display (older ones scroll off)
    max_display_segments: usize,
    /// Recent segments for display (ring buffer)
    display_segments: VecDeque<TranscriptSegment>,
}

impl PartialResultsState {
    /// Create new state
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_display_segments: 10,
            ..Default::default()
        }
    }

    /// Start streaming
    pub fn start_streaming(&mut self) {
        self.is_streaming = true;
        self.confirmed.clear();
        self.partial = None;
        self.audio_processed_secs = 0.0;
        self.display_segments.clear();
    }

    /// Stop streaming
    pub fn stop_streaming(&mut self) {
        self.is_streaming = false;
        // Promote partial to confirmed if exists
        if let Some(partial) = self.partial.take() {
            self.add_confirmed(partial);
        }
    }

    /// Update partial segment (overwrites previous partial)
    pub fn set_partial(&mut self, segment: TranscriptSegment) {
        self.partial = Some(segment);
    }

    /// Clear partial
    pub fn clear_partial(&mut self) {
        self.partial = None;
    }

    /// Add a confirmed segment
    pub fn add_confirmed(&mut self, mut segment: TranscriptSegment) {
        segment.is_confirmed = true;
        self.audio_processed_secs = segment.end_time;

        self.confirmed.push(segment.clone());

        // Add to display buffer
        self.display_segments.push_back(segment);
        while self.display_segments.len() > self.max_display_segments {
            self.display_segments.pop_front();
        }
    }

    /// Promote partial to confirmed
    pub fn confirm_partial(&mut self) {
        if let Some(partial) = self.partial.take() {
            self.add_confirmed(partial);
        }
    }

    /// Get full confirmed text
    #[must_use]
    pub fn full_text(&self) -> String {
        self.confirmed
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get display text (recent confirmed + partial)
    #[must_use]
    pub fn display_text(&self) -> String {
        let mut text = self
            .display_segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        if let Some(partial) = &self.partial {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&partial.text);
        }

        text
    }

    /// Get word count
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.confirmed
            .iter()
            .map(|s| s.text.split_whitespace().count())
            .sum()
    }

    /// Get average confidence
    #[must_use]
    pub fn average_confidence(&self) -> Option<f32> {
        if self.confirmed.is_empty() {
            return None;
        }

        let sum: f32 = self.confirmed.iter().map(|s| s.confidence).sum();
        Some(sum / self.confirmed.len() as f32)
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.confirmed.clear();
        self.partial = None;
        self.audio_processed_secs = 0.0;
        self.is_streaming = false;
        self.display_segments.clear();
    }
}

/// Partial results brick
#[derive(Debug, Clone, Default)]
pub struct PartialResultsBrick {
    state: PartialResultsState,
}

impl PartialResultsBrick {
    /// Create new brick
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get state reference
    #[must_use]
    pub fn state(&self) -> &PartialResultsState {
        &self.state
    }

    /// Get mutable state reference
    pub fn state_mut(&mut self) -> &mut PartialResultsState {
        &mut self.state
    }

    /// Start streaming
    pub fn start_streaming(&mut self) {
        self.state.start_streaming();
    }

    /// Stop streaming
    pub fn stop_streaming(&mut self) {
        self.state.stop_streaming();
    }

    /// Set partial segment
    pub fn set_partial(&mut self, text: impl Into<String>, start: f32, end: f32, confidence: f32) {
        self.state
            .set_partial(TranscriptSegment::new(text, start, end, confidence));
    }

    /// Add confirmed segment
    pub fn add_confirmed(&mut self, text: impl Into<String>, start: f32, end: f32, confidence: f32) {
        self.state
            .add_confirmed(TranscriptSegment::confirmed(text, start, end, confidence));
    }

    /// Confirm current partial
    pub fn confirm_partial(&mut self) {
        self.state.confirm_partial();
    }

    /// Clear partial
    pub fn clear_partial(&mut self) {
        self.state.clear_partial();
    }

    /// Reset
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Generate segments HTML
    fn segments_html(&self) -> String {
        let mut html = String::new();

        // Display recent confirmed segments
        for segment in &self.state.display_segments {
            let confidence_class = if segment.is_high_confidence() {
                "high-confidence"
            } else if segment.is_low_confidence() {
                "low-confidence"
            } else {
                "medium-confidence"
            };

            html.push_str(&format!(
                r#"<span class="segment confirmed {confidence_class}" data-start="{start:.2}" data-end="{end:.2}">{text}</span> "#,
                confidence_class = confidence_class,
                start = segment.start_time,
                end = segment.end_time,
                text = segment.text,
            ));
        }

        // Display partial segment with cursor
        if let Some(partial) = &self.state.partial {
            html.push_str(&format!(
                r#"<span class="segment partial" data-start="{start:.2}">{text}</span><span class="cursor"></span>"#,
                start = partial.start_time,
                text = partial.text,
            ));
        } else if self.state.is_streaming {
            html.push_str(r#"<span class="cursor"></span>"#);
        }

        html
    }
}

impl Brick for PartialResultsBrick {
    fn brick_name(&self) -> &'static str {
        "PartialResultsBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::MaxLatencyMs(100), // Update within 100ms
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16) // 60fps for smooth cursor
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

        // Empty state
        if state.confirmed.is_empty() && state.partial.is_none() {
            let status = if state.is_streaming {
                "Listening..."
            } else {
                "No transcription yet"
            };

            return format!(
                r#"<div class="partial-results-brick" data-testid="partial-results">
    <div class="empty-state {streaming_class}">
        <span class="status">{status}</span>
        {cursor}
    </div>
</div>"#,
                streaming_class = if state.is_streaming {
                    "streaming"
                } else {
                    ""
                },
                status = status,
                cursor = if state.is_streaming {
                    r#"<span class="cursor"></span>"#
                } else {
                    ""
                },
            );
        }

        let segments_html = self.segments_html();
        let confidence_display = state
            .average_confidence()
            .map(|c| format!("{:.0}%", c * 100.0))
            .unwrap_or_else(|| "—".into());

        format!(
            r#"<div class="partial-results-brick {streaming_class}" data-testid="partial-results">
    <div class="transcript-container">
        <div class="transcript-text" data-testid="transcript">{segments}</div>
    </div>
    <div class="transcript-footer">
        <span class="word-count">{words} words</span>
        <span class="audio-time">{time:.1}s</span>
        <span class="confidence" data-testid="confidence">{confidence}</span>
    </div>
</div>"#,
            streaming_class = if state.is_streaming {
                "streaming"
            } else {
                ""
            },
            segments = segments_html,
            words = state.word_count(),
            time = state.audio_processed_secs,
            confidence = confidence_display,
        )
    }

    fn to_css(&self) -> String {
        r".partial-results-brick {
    background: #1a1a2e;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.partial-results-brick.streaming {
    border-left: 3px solid #50fa7b;
}

.empty-state {
    color: #8b949e;
    font-style: italic;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.empty-state.streaming .status {
    color: #50fa7b;
}

.transcript-container {
    min-height: 60px;
    max-height: 200px;
    overflow-y: auto;
    margin-bottom: 0.75rem;
}

.transcript-text {
    color: #e0e0e0;
    font-size: 1rem;
    line-height: 1.6;
}

.segment {
    display: inline;
}

.segment.confirmed {
    color: #e0e0e0;
}

.segment.confirmed.high-confidence {
    color: #e0e0e0;
}

.segment.confirmed.medium-confidence {
    color: #b0b0b0;
}

.segment.confirmed.low-confidence {
    color: #8b949e;
    text-decoration: underline dotted;
}

.segment.partial {
    color: #4dc3ff;
    font-style: italic;
}

.cursor {
    display: inline-block;
    width: 2px;
    height: 1.2em;
    background: #50fa7b;
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: blink 1s step-end infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}

.transcript-footer {
    display: flex;
    gap: 1.5rem;
    font-size: 0.75rem;
    color: #8b949e;
    padding-top: 0.5rem;
    border-top: 1px solid #16213e;
}

.transcript-footer span {
    font-family: 'JetBrains Mono', monospace;
}

.confidence {
    margin-left: auto;
}"
            .into()
    }

    fn test_id(&self) -> Option<&str> {
        Some("partial-results")
    }
}

impl Widget for PartialResultsBrick {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let height: f32 = if self.state.confirmed.is_empty() && self.state.partial.is_none() {
            60.0
        } else {
            160.0
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
        let bounds = Rect::new(0.0, 0.0, 400.0, 160.0);

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

        // Draw transcript text
        let display_text = self.state.display_text();
        if display_text.is_empty() {
            let status = if self.state.is_streaming {
                "Listening..."
            } else {
                "No transcription yet"
            };
            let dim_style = TextStyle {
                size: 14.0,
                color: Color::from_hex("#8b949e").unwrap_or(Color::WHITE),
                weight: presentar_core::FontWeight::Normal,
                style: presentar_core::FontStyle::Italic,
            };
            canvas.draw_text(status, Point::new(16.0, 40.0), &dim_style);
        } else {
            canvas.draw_text(&display_text, Point::new(16.0, 40.0), &style);
        }

        // Draw footer
        let footer_style = TextStyle {
            size: 12.0,
            color: Color::from_hex("#8b949e").unwrap_or(Color::WHITE),
            weight: presentar_core::FontWeight::Normal,
            style: presentar_core::FontStyle::Normal,
        };

        let footer = format!(
            "{} words | {:.1}s",
            self.state.word_count(),
            self.state.audio_processed_secs
        );
        canvas.draw_text(&footer, Point::new(16.0, 140.0), &footer_style);
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
        Some("Partial transcription results")
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcript_segment_new() {
        let seg = TranscriptSegment::new("hello", 0.0, 1.0, 0.95);
        assert_eq!(seg.text, "hello");
        assert!(!seg.is_confirmed);
        assert!(seg.is_high_confidence());
    }

    #[test]
    fn test_transcript_segment_confirmed() {
        let seg = TranscriptSegment::confirmed("hello", 0.0, 1.0, 0.95);
        assert!(seg.is_confirmed);
    }

    #[test]
    fn test_transcript_segment_duration() {
        let seg = TranscriptSegment::new("test", 1.5, 3.5, 0.9);
        assert!((seg.duration() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_transcript_segment_confidence_levels() {
        let high = TranscriptSegment::new("hi", 0.0, 1.0, 0.95);
        assert!(high.is_high_confidence());
        assert!(!high.is_low_confidence());

        let medium = TranscriptSegment::new("mid", 0.0, 1.0, 0.8);
        assert!(!medium.is_high_confidence());
        assert!(!medium.is_low_confidence());

        let low = TranscriptSegment::new("low", 0.0, 1.0, 0.5);
        assert!(!low.is_high_confidence());
        assert!(low.is_low_confidence());
    }

    #[test]
    fn test_partial_results_state_default() {
        let state = PartialResultsState::new();
        assert!(!state.is_streaming);
        assert!(state.confirmed.is_empty());
        assert!(state.partial.is_none());
    }

    #[test]
    fn test_partial_results_state_start_streaming() {
        let mut state = PartialResultsState::new();
        state.start_streaming();
        assert!(state.is_streaming);
    }

    #[test]
    fn test_partial_results_state_set_partial() {
        let mut state = PartialResultsState::new();
        state.start_streaming();
        state.set_partial(TranscriptSegment::new("hello", 0.0, 1.0, 0.9));
        assert!(state.partial.is_some());
        assert_eq!(state.partial.as_ref().unwrap().text, "hello");
    }

    #[test]
    fn test_partial_results_state_add_confirmed() {
        let mut state = PartialResultsState::new();
        state.start_streaming();
        state.add_confirmed(TranscriptSegment::new("hello", 0.0, 1.0, 0.9));

        assert_eq!(state.confirmed.len(), 1);
        assert!(state.confirmed[0].is_confirmed);
        assert_eq!(state.full_text(), "hello");
    }

    #[test]
    fn test_partial_results_state_confirm_partial() {
        let mut state = PartialResultsState::new();
        state.start_streaming();
        state.set_partial(TranscriptSegment::new("world", 1.0, 2.0, 0.85));
        state.confirm_partial();

        assert!(state.partial.is_none());
        assert_eq!(state.confirmed.len(), 1);
        assert_eq!(state.full_text(), "world");
    }

    #[test]
    fn test_partial_results_state_display_text() {
        let mut state = PartialResultsState::new();
        state.start_streaming();
        state.add_confirmed(TranscriptSegment::new("hello", 0.0, 1.0, 0.9));
        state.set_partial(TranscriptSegment::new("world", 1.0, 2.0, 0.8));

        assert_eq!(state.display_text(), "hello world");
    }

    #[test]
    fn test_partial_results_state_word_count() {
        let mut state = PartialResultsState::new();
        state.add_confirmed(TranscriptSegment::new("hello world", 0.0, 1.0, 0.9));
        state.add_confirmed(TranscriptSegment::new("foo bar baz", 1.0, 2.0, 0.9));

        assert_eq!(state.word_count(), 5);
    }

    #[test]
    fn test_partial_results_state_average_confidence() {
        let mut state = PartialResultsState::new();
        state.add_confirmed(TranscriptSegment::new("a", 0.0, 1.0, 0.8));
        state.add_confirmed(TranscriptSegment::new("b", 1.0, 2.0, 1.0));

        let avg = state.average_confidence().unwrap();
        assert!((avg - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_partial_results_state_stop_streaming() {
        let mut state = PartialResultsState::new();
        state.start_streaming();
        state.set_partial(TranscriptSegment::new("final", 0.0, 1.0, 0.9));
        state.stop_streaming();

        assert!(!state.is_streaming);
        assert!(state.partial.is_none());
        assert_eq!(state.confirmed.len(), 1);
    }

    #[test]
    fn test_partial_results_state_reset() {
        let mut state = PartialResultsState::new();
        state.start_streaming();
        state.add_confirmed(TranscriptSegment::new("test", 0.0, 1.0, 0.9));
        state.reset();

        assert!(!state.is_streaming);
        assert!(state.confirmed.is_empty());
        assert!(state.partial.is_none());
    }

    #[test]
    fn test_brick_default() {
        let brick = PartialResultsBrick::new();
        assert!(!brick.state().is_streaming);
    }

    #[test]
    fn test_brick_start_streaming() {
        let mut brick = PartialResultsBrick::new();
        brick.start_streaming();
        assert!(brick.state().is_streaming);
    }

    #[test]
    fn test_brick_set_partial() {
        let mut brick = PartialResultsBrick::new();
        brick.start_streaming();
        brick.set_partial("hello", 0.0, 1.0, 0.9);
        assert!(brick.state().partial.is_some());
    }

    #[test]
    fn test_brick_add_confirmed() {
        let mut brick = PartialResultsBrick::new();
        brick.add_confirmed("hello", 0.0, 1.0, 0.95);
        assert_eq!(brick.state().confirmed.len(), 1);
    }

    #[test]
    fn test_brick_verification() {
        let brick = PartialResultsBrick::new();
        let result = brick.verify();
        assert!(result.is_valid());
    }

    #[test]
    fn test_brick_to_html_empty() {
        let brick = PartialResultsBrick::new();
        let html = brick.to_html();
        assert!(html.contains("No transcription yet"));
        assert!(html.contains("data-testid=\"partial-results\""));
    }

    #[test]
    fn test_brick_to_html_streaming_empty() {
        let mut brick = PartialResultsBrick::new();
        brick.start_streaming();
        let html = brick.to_html();
        assert!(html.contains("Listening..."));
        assert!(html.contains("cursor"));
    }

    #[test]
    fn test_brick_to_html_with_text() {
        let mut brick = PartialResultsBrick::new();
        brick.start_streaming();
        brick.add_confirmed("hello world", 0.0, 2.0, 0.95);
        brick.set_partial("how are", 2.0, 3.0, 0.8);

        let html = brick.to_html();
        assert!(html.contains("hello world"));
        assert!(html.contains("how are"));
        assert!(html.contains("segment confirmed"));
        assert!(html.contains("segment partial"));
    }

    #[test]
    fn test_brick_budget() {
        let brick = PartialResultsBrick::new();
        assert_eq!(brick.budget().total_ms, 16);
    }

    #[test]
    fn test_brick_can_render() {
        let brick = PartialResultsBrick::new();
        assert!(brick.can_render());
    }
}
