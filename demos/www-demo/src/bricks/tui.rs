//! TUI Renderer for Bricks (PROBAR-SPEC-009)
//!
//! This module provides ratatui-based TUI rendering of bricks.
//! The same brick definitions render to both TUI and WASM targets.
//!
//! # Dual-Target Architecture
//!
//! ```text
//! TranscriptionBrick
//!     ├── to_html() → WASM/Browser
//!     └── to_tui()  → Terminal/TUI
//! ```
//!
//! # Integration with trueno-viz
//!
//! This module uses patterns from trueno-viz/ttop:
//! - Ring buffers for sparkline data
//! - SIMD-optimized statistics
//! - percent_color for meter gradients

use jugar_probar::brick::Brick;

use super::{
    AudioBrick, ScoreBrick, StatusBrick, TranscriptionBrick, VuMeterBrick, WaveformBrick,
};

/// TUI rendering output (text-based)
#[derive(Debug, Clone)]
pub struct TuiOutput {
    /// Lines of text to render
    pub lines: Vec<String>,
    /// Width in characters
    pub width: u16,
    /// Height in lines
    pub height: u16,
}

impl TuiOutput {
    /// Create empty output
    #[must_use]
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            lines: Vec::new(),
            width,
            height,
        }
    }

    /// Add a line
    pub fn push_line(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
    }

    /// Render to string
    #[must_use]
    pub fn to_string(&self) -> String {
        self.lines.join("\n")
    }
}

/// TUI renderer for bricks
pub struct TuiRenderer {
    /// Terminal width
    width: u16,
    /// Terminal height
    height: u16,
}

impl Default for TuiRenderer {
    fn default() -> Self {
        Self {
            width: 80,
            height: 24,
        }
    }
}

impl TuiRenderer {
    /// Create a new TUI renderer
    #[must_use]
    pub fn new(width: u16, height: u16) -> Self {
        Self { width, height }
    }

    /// Render a transcription brick
    #[must_use]
    pub fn render_transcription(&self, brick: &TranscriptionBrick) -> TuiOutput {
        let mut output = TuiOutput::new(self.width, 6);

        output.push_line("┌─ Transcription ─────────────────────────────────────┐");

        if !brick.partial().is_empty() {
            let partial = truncate(brick.partial(), (self.width - 4) as usize);
            output.push_line(format!("│ \x1b[3m\x1b[90m{partial}\x1b[0m"));
        }

        let combined = brick.combined_text();
        if !combined.is_empty() {
            // Word wrap the text
            for line in word_wrap(&combined, (self.width - 4) as usize) {
                output.push_line(format!("│ {line}"));
            }
        } else if brick.partial().is_empty() {
            output.push_line("│ (waiting for speech...)");
        }

        output.push_line("└─────────────────────────────────────────────────────┘");
        output
    }

    /// Render a VU meter brick
    #[must_use]
    pub fn render_vu_meter(&self, brick: &VuMeterBrick) -> TuiOutput {
        let mut output = TuiOutput::new(self.width, 1);

        let level = brick.level();
        let bar_width = 20;
        let filled = (level * bar_width as f32) as usize;

        let bar: String = (0..bar_width)
            .map(|i| {
                if i < filled {
                    percent_color(i as f32 / bar_width as f32, '█')
                } else {
                    '░'
                }
            })
            .collect();

        output.push_line(format!("VU: [{}] {:3}%", bar, brick.level_percent()));
        output
    }

    /// Render a waveform brick as sparkline
    #[must_use]
    pub fn render_waveform(&self, brick: &WaveformBrick) -> TuiOutput {
        let mut output = TuiOutput::new(self.width, 2);

        let samples = brick.get_display_samples();
        let sparkline = samples_to_sparkline(&samples, (self.width - 12) as usize);

        output.push_line(format!("Waveform: {sparkline}"));
        output
    }

    /// Render a status brick
    #[must_use]
    pub fn render_status(&self, brick: &StatusBrick) -> TuiOutput {
        let mut output = TuiOutput::new(self.width, 1);

        let (icon, color) = match brick.status() {
            super::status::Status::Loading { .. } => ("⏳", "\x1b[36m"),
            super::status::Status::Ready => ("✓", "\x1b[32m"),
            super::status::Status::Recording => ("●", "\x1b[31m"),
            super::status::Status::Error { .. } => ("✗", "\x1b[91m"),
        };

        output.push_line(format!(
            "{color}{icon}\x1b[0m {}",
            brick.status_text()
        ));
        output
    }

    /// Render an audio brick
    #[must_use]
    pub fn render_audio(&self, brick: &AudioBrick) -> TuiOutput {
        let mut output = TuiOutput::new(self.width, 1);

        let fill = brick.fill_percent();
        let bar_width = 30;
        let filled = (fill as usize * bar_width) / 100;

        let bar: String = (0..bar_width)
            .map(|i| if i < filled { '█' } else { '░' })
            .collect();

        output.push_line(format!(
            "Audio: [{}] {}Hz {:.1}s",
            bar,
            brick.sample_rate(),
            brick.buffer_duration_secs()
        ));
        output
    }

    /// Render a score brick (falsification dashboard)
    #[must_use]
    pub fn render_score(&self, brick: &ScoreBrick) -> TuiOutput {
        let lines = brick.to_tui_lines(self.width);
        let mut output = TuiOutput::new(self.width, lines.len() as u16);

        for line in lines {
            output.push_line(line);
        }

        output
    }
}

/// Render any brick to TUI output
#[must_use]
pub fn render_brick_to_tui(brick: &dyn Brick) -> TuiOutput {
    let _renderer = TuiRenderer::default();
    let mut output = TuiOutput::new(80, 1);

    // Use brick name to determine rendering
    match brick.brick_name() {
        "TranscriptionBrick" => {
            // Can't downcast dyn Brick, so provide generic rendering
            output.push_line(format!("[{}]", brick.brick_name()));
        }
        "VuMeterBrick" => {
            output.push_line(format!("[{}]", brick.brick_name()));
        }
        "WaveformBrick" => {
            output.push_line(format!("[{}]", brick.brick_name()));
        }
        "StatusBrick" => {
            output.push_line(format!("[{}]", brick.brick_name()));
        }
        "AudioBrick" => {
            output.push_line(format!("[{}]", brick.brick_name()));
        }
        _ => {
            output.push_line(format!("[Unknown: {}]", brick.brick_name()));
        }
    }

    output
}

/// Convert samples to sparkline characters
fn samples_to_sparkline(samples: &[f32], width: usize) -> String {
    const SPARKLINE_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    if samples.is_empty() || width == 0 {
        return String::new();
    }

    // Downsample if needed
    let step = samples.len().max(1) / width.max(1);
    let step = step.max(1);

    samples
        .iter()
        .step_by(step)
        .take(width)
        .map(|&s| {
            // Normalize to 0-1 range (assuming samples are -1 to 1)
            let normalized = (s.abs()).clamp(0.0, 1.0);
            let idx = (normalized * 7.0) as usize;
            SPARKLINE_CHARS[idx.min(7)]
        })
        .collect()
}

/// Get color character based on percentage (ttop-style)
fn percent_color(_percent: f32, ch: char) -> char {
    // In a real implementation, this would return ANSI colored output
    // For now, just return the character
    ch
}

/// Truncate string to max length with ellipsis
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        "...".to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Word wrap text to max width
fn word_wrap(text: &str, max_width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in text.split_whitespace() {
        if current_line.is_empty() {
            current_line = word.to_string();
        } else if current_line.len() + 1 + word.len() <= max_width {
            current_line.push(' ');
            current_line.push_str(word);
        } else {
            lines.push(current_line);
            current_line = word.to_string();
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tui_output() {
        let mut output = TuiOutput::new(80, 24);
        output.push_line("Hello");
        output.push_line("World");

        assert_eq!(output.lines.len(), 2);
        assert_eq!(output.to_string(), "Hello\nWorld");
    }

    #[test]
    fn test_render_transcription() {
        let mut brick = TranscriptionBrick::new();
        brick.on_final("Hello world".into());

        let renderer = TuiRenderer::default();
        let output = renderer.render_transcription(&brick);

        assert!(output.to_string().contains("Hello world"));
    }

    #[test]
    fn test_render_vu_meter() {
        let mut brick = VuMeterBrick::new();
        brick.update_level(0.5);

        let renderer = TuiRenderer::default();
        let output = renderer.render_vu_meter(&brick);

        assert!(output.to_string().contains("VU:"));
        assert!(output.to_string().contains("50%"));
    }

    #[test]
    fn test_render_waveform() {
        let mut brick = WaveformBrick::new();
        for i in 0..100 {
            brick.push_sample((i as f32 / 100.0).sin());
        }

        let renderer = TuiRenderer::default();
        let output = renderer.render_waveform(&brick);

        assert!(output.to_string().contains("Waveform:"));
    }

    #[test]
    fn test_render_status() {
        let mut brick = StatusBrick::new();
        brick.set_ready();

        let renderer = TuiRenderer::default();
        let output = renderer.render_status(&brick);

        assert!(output.to_string().contains("Ready"));
    }

    #[test]
    fn test_render_audio() {
        let brick = AudioBrick::new();

        let renderer = TuiRenderer::default();
        let output = renderer.render_audio(&brick);

        assert!(output.to_string().contains("Audio:"));
        assert!(output.to_string().contains("48000Hz"));
    }

    #[test]
    fn test_samples_to_sparkline() {
        let samples = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let sparkline = samples_to_sparkline(&samples, 5);

        assert_eq!(sparkline.chars().count(), 5);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello world", 20), "hello world");
        assert_eq!(truncate("hello world", 8), "hello...");
        assert_eq!(truncate("hi", 2), "hi");
    }

    #[test]
    fn test_word_wrap() {
        let lines = word_wrap("hello world this is a test", 12);
        assert!(lines.len() >= 2);
    }
}
