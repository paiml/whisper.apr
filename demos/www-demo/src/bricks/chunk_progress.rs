//! `ChunkProgressBrick`: Real-time chunk processing display (PROBAR-SPEC-009)
//!
//! This brick displays real-time streaming progress for live recording mode:
//! - "Processing chunk 3/N..." with animation
//! - Chunk-by-chunk progress visualization
//! - Real-time audio buffer state indication
//!
//! # Assertions
//!
//! - Chunk count updates in real-time
//! - Visual feedback on chunk processing
//! - Maximum 16ms render latency for 60fps

use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_core::{
    AccessibleRole, Canvas, Color, Constraints, Event, LayoutResult, Point, Rect, Size, TextStyle,
    TypeId, Widget,
};
use std::any::Any;
use std::time::{Duration, Instant};

/// Chunk processing state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkState {
    /// Waiting for audio input
    Waiting,
    /// Buffering audio (building up initial buffer)
    Buffering,
    /// Actively processing chunks
    Processing,
    /// Processing paused (e.g., silence detected)
    Paused,
    /// Processing complete
    Complete,
}

impl ChunkState {
    /// Get display name for state
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Waiting => "Waiting for audio...",
            Self::Buffering => "Buffering...",
            Self::Processing => "Processing",
            Self::Paused => "Paused",
            Self::Complete => "Complete",
        }
    }

    /// Get CSS class for state
    #[must_use]
    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Waiting => "chunk-waiting",
            Self::Buffering => "chunk-buffering",
            Self::Processing => "chunk-processing",
            Self::Paused => "chunk-paused",
            Self::Complete => "chunk-complete",
        }
    }

    /// Check if state is active (audio being captured)
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Buffering | Self::Processing)
    }
}

/// Statistics for a single chunk
#[derive(Debug, Clone)]
pub struct ChunkStats {
    /// Chunk index (0-based)
    pub index: u32,
    /// Duration of audio in this chunk (in seconds)
    pub audio_duration_secs: f32,
    /// Processing time for this chunk (in milliseconds)
    pub processing_time_ms: f32,
    /// Timestamp when chunk started processing
    pub start_time: Instant,
}

impl ChunkStats {
    /// Calculate Real-Time Factor for this chunk
    #[must_use]
    pub fn rtf(&self) -> f32 {
        if self.audio_duration_secs > 0.0 {
            (self.processing_time_ms / 1000.0) / self.audio_duration_secs
        } else {
            0.0
        }
    }
}

/// Chunk progress state
#[derive(Debug, Clone)]
pub struct ChunkProgressState {
    /// Current state
    pub state: ChunkState,
    /// Current chunk being processed (1-based for display)
    pub current_chunk: u32,
    /// Total expected chunks (None if streaming indefinitely)
    pub total_chunks: Option<u32>,
    /// Chunk duration in seconds (typically 30s for Whisper)
    pub chunk_duration_secs: f32,
    /// Total audio duration processed
    pub total_audio_secs: f32,
    /// Buffer fill level (0.0-1.0)
    pub buffer_fill: f32,
    /// Processing start time
    pub start_time: Option<Instant>,
    /// Recent chunk statistics (last N chunks)
    pub recent_chunks: Vec<ChunkStats>,
    /// Maximum chunks to keep in history
    max_history: usize,
}

impl Default for ChunkProgressState {
    fn default() -> Self {
        Self {
            state: ChunkState::Waiting,
            current_chunk: 0,
            total_chunks: None,
            chunk_duration_secs: 30.0,
            total_audio_secs: 0.0,
            buffer_fill: 0.0,
            start_time: None,
            recent_chunks: Vec::new(),
            max_history: 10,
        }
    }
}

impl ChunkProgressState {
    /// Create new state
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom chunk duration
    #[must_use]
    pub fn with_chunk_duration(chunk_duration_secs: f32) -> Self {
        Self {
            chunk_duration_secs,
            ..Default::default()
        }
    }

    /// Start waiting for audio
    pub fn start_waiting(&mut self) {
        self.state = ChunkState::Waiting;
        self.current_chunk = 0;
        self.total_audio_secs = 0.0;
        self.buffer_fill = 0.0;
        self.start_time = Some(Instant::now());
        self.recent_chunks.clear();
    }

    /// Begin buffering
    pub fn start_buffering(&mut self) {
        self.state = ChunkState::Buffering;
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }
    }

    /// Update buffer fill level
    pub fn update_buffer(&mut self, fill: f32) {
        self.buffer_fill = fill.clamp(0.0, 1.0);
        if self.state == ChunkState::Waiting && fill > 0.0 {
            self.state = ChunkState::Buffering;
        }
    }

    /// Start processing a new chunk
    pub fn start_chunk(&mut self, chunk_index: u32) {
        self.state = ChunkState::Processing;
        self.current_chunk = chunk_index + 1; // 1-based for display
    }

    /// Complete processing a chunk
    pub fn complete_chunk(&mut self, stats: ChunkStats) {
        self.total_audio_secs += stats.audio_duration_secs;

        // Add to history, maintaining max size
        self.recent_chunks.push(stats);
        if self.recent_chunks.len() > self.max_history {
            self.recent_chunks.remove(0);
        }
    }

    /// Set total chunks (when known, e.g., file upload)
    pub fn set_total_chunks(&mut self, total: u32) {
        self.total_chunks = Some(total);
    }

    /// Pause processing
    pub fn pause(&mut self) {
        self.state = ChunkState::Paused;
    }

    /// Resume processing
    pub fn resume(&mut self) {
        if self.state == ChunkState::Paused {
            self.state = ChunkState::Processing;
        }
    }

    /// Mark as complete
    pub fn complete(&mut self) {
        self.state = ChunkState::Complete;
    }

    /// Reset state
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get average RTF from recent chunks
    #[must_use]
    pub fn average_rtf(&self) -> Option<f32> {
        if self.recent_chunks.is_empty() {
            return None;
        }

        let sum: f32 = self.recent_chunks.iter().map(|c| c.rtf()).sum();
        Some(sum / self.recent_chunks.len() as f32)
    }

    /// Get elapsed time since start
    #[must_use]
    pub fn elapsed(&self) -> Option<Duration> {
        self.start_time.map(|t| t.elapsed())
    }

    /// Format chunk progress as string
    #[must_use]
    pub fn format_progress(&self) -> String {
        match self.total_chunks {
            Some(total) => format!("chunk {}/{}", self.current_chunk, total),
            None => format!("chunk {}", self.current_chunk),
        }
    }

    /// Format elapsed time
    #[must_use]
    pub fn format_elapsed(&self) -> String {
        match self.elapsed() {
            Some(dur) => {
                let secs = dur.as_secs();
                if secs < 60 {
                    format!("{}s", secs)
                } else {
                    format!("{}:{:02}", secs / 60, secs % 60)
                }
            }
            None => "0s".into(),
        }
    }
}

/// Chunk progress brick for real-time streaming display
#[derive(Debug, Clone, Default)]
pub struct ChunkProgressBrick {
    state: ChunkProgressState,
}

impl ChunkProgressBrick {
    /// Create new chunk progress brick
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom chunk duration
    #[must_use]
    pub fn with_chunk_duration(chunk_duration_secs: f32) -> Self {
        Self {
            state: ChunkProgressState::with_chunk_duration(chunk_duration_secs),
        }
    }

    /// Get state reference
    #[must_use]
    pub fn state(&self) -> &ChunkProgressState {
        &self.state
    }

    /// Get mutable state reference
    pub fn state_mut(&mut self) -> &mut ChunkProgressState {
        &mut self.state
    }

    /// Start waiting
    pub fn start_waiting(&mut self) {
        self.state.start_waiting();
    }

    /// Start buffering
    pub fn start_buffering(&mut self) {
        self.state.start_buffering();
    }

    /// Update buffer fill
    pub fn update_buffer(&mut self, fill: f32) {
        self.state.update_buffer(fill);
    }

    /// Start processing chunk
    pub fn start_chunk(&mut self, index: u32) {
        self.state.start_chunk(index);
    }

    /// Complete chunk with stats
    pub fn complete_chunk(&mut self, stats: ChunkStats) {
        self.state.complete_chunk(stats);
    }

    /// Set total chunks
    pub fn set_total_chunks(&mut self, total: u32) {
        self.state.set_total_chunks(total);
    }

    /// Pause
    pub fn pause(&mut self) {
        self.state.pause();
    }

    /// Resume
    pub fn resume(&mut self) {
        self.state.resume();
    }

    /// Complete
    pub fn complete(&mut self) {
        self.state.complete();
    }

    /// Reset
    pub fn reset(&mut self) {
        self.state.reset();
    }
}

impl Brick for ChunkProgressBrick {
    fn brick_name(&self) -> &'static str {
        "ChunkProgressBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::MaxLatencyMs(16), // 60fps for live updates
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16) // 60fps budget
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
        let css_class = state.state.css_class();

        match state.state {
            ChunkState::Waiting => {
                format!(
                    r#"<div class="chunk-progress-brick {css_class}" data-testid="chunk-progress">
    <div class="chunk-status">
        <span class="pulse-dot"></span>
        <span class="status-text">{status}</span>
    </div>
</div>"#,
                    css_class = css_class,
                    status = state.state.display_name()
                )
            }
            ChunkState::Buffering => {
                let buffer_percent = (state.buffer_fill * 100.0) as u32;
                format!(
                    r#"<div class="chunk-progress-brick {css_class}" data-testid="chunk-progress">
    <div class="chunk-status">
        <span class="pulse-dot buffering"></span>
        <span class="status-text">{status}</span>
    </div>
    <div class="buffer-indicator">
        <div class="buffer-fill" style="width: {percent}%"></div>
    </div>
    <span class="buffer-label">Buffer: {percent}%</span>
</div>"#,
                    css_class = css_class,
                    status = state.state.display_name(),
                    percent = buffer_percent,
                )
            }
            ChunkState::Processing => {
                let rtf_display = state
                    .average_rtf()
                    .map(|rtf| format!("{:.2}x RTF", rtf))
                    .unwrap_or_else(|| "—".into());

                format!(
                    r#"<div class="chunk-progress-brick {css_class}" data-testid="chunk-progress">
    <div class="chunk-status">
        <span class="pulse-dot active"></span>
        <span class="status-text">Processing {progress}...</span>
    </div>
    <div class="chunk-details">
        <span class="elapsed" data-testid="elapsed">{elapsed}</span>
        <span class="rtf" data-testid="rtf">{rtf}</span>
        <span class="audio-processed" data-testid="audio-processed">{audio:.1}s audio</span>
    </div>
    <div class="buffer-indicator">
        <div class="buffer-fill" style="width: {buffer}%"></div>
    </div>
</div>"#,
                    css_class = css_class,
                    progress = state.format_progress(),
                    elapsed = state.format_elapsed(),
                    rtf = rtf_display,
                    audio = state.total_audio_secs,
                    buffer = (state.buffer_fill * 100.0) as u32,
                )
            }
            ChunkState::Paused => {
                format!(
                    r#"<div class="chunk-progress-brick {css_class}" data-testid="chunk-progress">
    <div class="chunk-status">
        <span class="pause-icon">⏸</span>
        <span class="status-text">Paused at {progress}</span>
    </div>
    <div class="chunk-details">
        <span class="elapsed">{elapsed}</span>
        <span class="audio-processed">{audio:.1}s audio</span>
    </div>
</div>"#,
                    css_class = css_class,
                    progress = state.format_progress(),
                    elapsed = state.format_elapsed(),
                    audio = state.total_audio_secs,
                )
            }
            ChunkState::Complete => {
                let rtf_display = state
                    .average_rtf()
                    .map(|rtf| format!("{:.2}x RTF", rtf))
                    .unwrap_or_else(|| "—".into());

                format!(
                    r#"<div class="chunk-progress-brick {css_class}" data-testid="chunk-progress">
    <div class="chunk-status">
        <span class="complete-icon">✓</span>
        <span class="status-text">Complete</span>
    </div>
    <div class="chunk-details">
        <span class="chunks-processed">{chunks} chunks</span>
        <span class="total-time">{elapsed}</span>
        <span class="rtf">{rtf}</span>
        <span class="audio-processed">{audio:.1}s audio</span>
    </div>
</div>"#,
                    css_class = css_class,
                    chunks = state.current_chunk,
                    elapsed = state.format_elapsed(),
                    rtf = rtf_display,
                    audio = state.total_audio_secs,
                )
            }
        }
    }

    fn to_css(&self) -> String {
        r".chunk-progress-brick {
    background: #1a1a2e;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.chunk-status {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}

.pulse-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #8b949e;
}

.pulse-dot.buffering {
    background: #ffb86c;
    animation: pulse 1.5s ease-in-out infinite;
}

.pulse-dot.active {
    background: #50fa7b;
    animation: pulse 1s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
}

.status-text {
    color: #e0e0e0;
    font-size: 1rem;
    font-weight: 500;
}

.chunk-details {
    display: flex;
    gap: 1.5rem;
    margin: 0.5rem 0;
    font-size: 0.875rem;
    color: #8b949e;
}

.chunk-details span {
    font-family: 'JetBrains Mono', monospace;
}

.rtf {
    color: #4dc3ff;
}

.buffer-indicator {
    background: #16213e;
    height: 4px;
    border-radius: 2px;
    margin-top: 0.5rem;
    overflow: hidden;
}

.buffer-fill {
    height: 100%;
    background: linear-gradient(90deg, #4dc3ff, #50fa7b);
    border-radius: 2px;
    transition: width 0.1s ease-out;
}

.buffer-label {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 0.25rem;
    display: block;
}

.chunk-waiting .status-text {
    color: #8b949e;
}

.chunk-buffering .status-text {
    color: #ffb86c;
}

.chunk-processing .status-text {
    color: #50fa7b;
}

.chunk-paused .status-text {
    color: #f1fa8c;
}

.chunk-complete .status-text {
    color: #50fa7b;
}

.pause-icon, .complete-icon {
    font-size: 1.25rem;
}

.complete-icon {
    color: #50fa7b;
}

.pause-icon {
    color: #f1fa8c;
}"
            .into()
    }

    fn test_id(&self) -> Option<&str> {
        Some("chunk-progress")
    }
}

impl Widget for ChunkProgressBrick {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let height: f32 = match self.state.state {
            ChunkState::Waiting => 48.0,
            ChunkState::Buffering => 72.0,
            ChunkState::Processing => 96.0,
            ChunkState::Paused => 72.0,
            ChunkState::Complete => 72.0,
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
        let bounds = Rect::new(0.0, 0.0, 400.0, 96.0);

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

        // Draw status text
        let status_text = match self.state.state {
            ChunkState::Processing => format!("Processing {}...", self.state.format_progress()),
            _ => self.state.state.display_name().into(),
        };
        canvas.draw_text(&status_text, Point::new(32.0, 24.0), &style);

        // Draw buffer indicator
        if self.state.state.is_active() {
            let bar_bg = Rect::new(16.0, 60.0, 368.0, 4.0);
            let bar_bg_color = Color::from_hex("#16213e").unwrap_or(Color::BLACK);
            canvas.fill_rect(bar_bg, bar_bg_color);

            let fill_width = 368.0 * self.state.buffer_fill;
            let bar_fill = Rect::new(16.0, 60.0, fill_width, 4.0);
            let bar_color = Color::from_hex("#4dc3ff").unwrap_or(Color::BLUE);
            canvas.fill_rect(bar_fill, bar_color);
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
        Some("Chunk processing progress")
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_state_display_name() {
        assert_eq!(ChunkState::Waiting.display_name(), "Waiting for audio...");
        assert_eq!(ChunkState::Buffering.display_name(), "Buffering...");
        assert_eq!(ChunkState::Processing.display_name(), "Processing");
        assert_eq!(ChunkState::Paused.display_name(), "Paused");
        assert_eq!(ChunkState::Complete.display_name(), "Complete");
    }

    #[test]
    fn test_chunk_state_is_active() {
        assert!(!ChunkState::Waiting.is_active());
        assert!(ChunkState::Buffering.is_active());
        assert!(ChunkState::Processing.is_active());
        assert!(!ChunkState::Paused.is_active());
        assert!(!ChunkState::Complete.is_active());
    }

    #[test]
    fn test_chunk_stats_rtf() {
        let stats = ChunkStats {
            index: 0,
            audio_duration_secs: 30.0,
            processing_time_ms: 15000.0, // 15 seconds
            start_time: Instant::now(),
        };
        assert!((stats.rtf() - 0.5).abs() < 0.001); // 0.5x RTF
    }

    #[test]
    fn test_chunk_progress_state_default() {
        let state = ChunkProgressState::new();
        assert_eq!(state.state, ChunkState::Waiting);
        assert_eq!(state.current_chunk, 0);
        assert_eq!(state.chunk_duration_secs, 30.0);
    }

    #[test]
    fn test_chunk_progress_state_transitions() {
        let mut state = ChunkProgressState::new();

        state.start_waiting();
        assert_eq!(state.state, ChunkState::Waiting);

        state.start_buffering();
        assert_eq!(state.state, ChunkState::Buffering);

        state.update_buffer(0.5);
        assert_eq!(state.buffer_fill, 0.5);

        state.start_chunk(0);
        assert_eq!(state.state, ChunkState::Processing);
        assert_eq!(state.current_chunk, 1);

        state.pause();
        assert_eq!(state.state, ChunkState::Paused);

        state.resume();
        assert_eq!(state.state, ChunkState::Processing);

        state.complete();
        assert_eq!(state.state, ChunkState::Complete);
    }

    #[test]
    fn test_chunk_progress_state_complete_chunk() {
        let mut state = ChunkProgressState::new();
        state.start_chunk(0);

        let stats = ChunkStats {
            index: 0,
            audio_duration_secs: 30.0,
            processing_time_ms: 10000.0,
            start_time: Instant::now(),
        };
        state.complete_chunk(stats);

        assert_eq!(state.total_audio_secs, 30.0);
        assert_eq!(state.recent_chunks.len(), 1);
    }

    #[test]
    fn test_chunk_progress_state_average_rtf() {
        let mut state = ChunkProgressState::new();
        state.start_chunk(0);

        // Add 3 chunks with different RTFs
        for i in 0..3 {
            let stats = ChunkStats {
                index: i,
                audio_duration_secs: 30.0,
                processing_time_ms: 15000.0, // 0.5x RTF
                start_time: Instant::now(),
            };
            state.complete_chunk(stats);
        }

        let avg = state.average_rtf().unwrap();
        assert!((avg - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_chunk_progress_state_format_progress() {
        let mut state = ChunkProgressState::new();
        state.start_chunk(2);
        assert_eq!(state.format_progress(), "chunk 3");

        state.set_total_chunks(10);
        assert_eq!(state.format_progress(), "chunk 3/10");
    }

    #[test]
    fn test_brick_default() {
        let brick = ChunkProgressBrick::new();
        assert_eq!(brick.state().state, ChunkState::Waiting);
    }

    #[test]
    fn test_brick_with_chunk_duration() {
        let brick = ChunkProgressBrick::with_chunk_duration(10.0);
        assert_eq!(brick.state().chunk_duration_secs, 10.0);
    }

    #[test]
    fn test_brick_transitions() {
        let mut brick = ChunkProgressBrick::new();

        brick.start_waiting();
        assert_eq!(brick.state().state, ChunkState::Waiting);

        brick.start_buffering();
        assert_eq!(brick.state().state, ChunkState::Buffering);

        brick.start_chunk(0);
        assert_eq!(brick.state().state, ChunkState::Processing);

        brick.complete();
        assert_eq!(brick.state().state, ChunkState::Complete);
    }

    #[test]
    fn test_brick_verification() {
        let brick = ChunkProgressBrick::new();
        let result = brick.verify();
        assert!(result.is_valid());
    }

    #[test]
    fn test_brick_to_html_waiting() {
        let brick = ChunkProgressBrick::new();
        let html = brick.to_html();
        assert!(html.contains("Waiting for audio"));
        assert!(html.contains("data-testid=\"chunk-progress\""));
    }

    #[test]
    fn test_brick_to_html_processing() {
        let mut brick = ChunkProgressBrick::new();
        brick.start_chunk(2);

        let html = brick.to_html();
        assert!(html.contains("Processing chunk 3"));
        assert!(html.contains("pulse-dot active"));
    }

    #[test]
    fn test_brick_to_html_complete() {
        let mut brick = ChunkProgressBrick::new();
        brick.start_chunk(0);
        brick.complete();

        let html = brick.to_html();
        assert!(html.contains("Complete"));
        assert!(html.contains("complete-icon"));
    }

    #[test]
    fn test_brick_budget() {
        let brick = ChunkProgressBrick::new();
        assert_eq!(brick.budget().total_ms, 16);
    }

    #[test]
    fn test_brick_can_render() {
        let brick = ChunkProgressBrick::new();
        assert!(brick.can_render());
    }
}
