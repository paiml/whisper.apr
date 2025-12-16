//! ASCII Visualization Utilities
//!
//! Renders waveforms, spectrograms, and heatmaps as ASCII art.
//!
//! References:
//! - Davis & Mermelstein (1980): Mel filterbank fundamentals
//! - Bahdanau et al. (2014): Attention visualization

use std::fmt::Write;

/// Waveform display configuration
#[derive(Debug, Clone)]
pub struct WaveformDisplay {
    samples: Vec<f32>,
    width: usize,
    height: usize,
}

impl WaveformDisplay {
    /// Create new waveform display
    pub fn new(samples: &[f32], width: usize, height: usize) -> Self {
        Self {
            samples: samples.to_vec(),
            width,
            height,
        }
    }

    /// Get display width
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get display height
    pub fn height(&self) -> usize {
        self.height
    }

    /// Render to ASCII art
    pub fn render(&self) -> String {
        if self.samples.is_empty() {
            return "No data".to_string();
        }

        render_waveform(&self.samples, self.width, self.height)
    }
}

/// Mel spectrogram display configuration
#[derive(Debug, Clone)]
pub struct MelDisplay {
    mel_data: Vec<f32>,
    n_mels: usize,
    n_frames: usize,
    width: usize,
    height: usize,
}

impl MelDisplay {
    /// Create new mel display
    pub fn new(mel_data: &[f32], n_mels: usize, n_frames: usize, width: usize, height: usize) -> Self {
        Self {
            mel_data: mel_data.to_vec(),
            n_mels,
            n_frames,
            width,
            height,
        }
    }

    /// Render to ASCII heatmap
    pub fn render(&self) -> String {
        if self.mel_data.is_empty() || self.n_frames == 0 {
            return "No data".to_string();
        }

        render_mel_spectrogram(&self.mel_data, self.n_mels, self.n_frames, self.width, self.height)
    }
}

/// Render waveform as ASCII art
#[allow(clippy::no_effect_underscore_binding)]
pub fn render_waveform(samples: &[f32], width: usize, height: usize) -> String {
    let _span = crate::trace_enter!("tui.render_waveform");
    if samples.is_empty() || width == 0 || height == 0 {
        return String::new();
    }

    // Find min/max for normalization
    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
    let scale = if max_abs > 0.0 { max_abs } else { 1.0 };

    // Downsample to fit width
    let samples_per_col = samples.len() / width.max(1);
    let samples_per_col = samples_per_col.max(1);

    // Create grid
    let mut grid = vec![vec![' '; width]; height];
    let mid_row = height / 2;

    // Plot waveform
    for col in 0..width {
        let start = col * samples_per_col;
        let end = ((col + 1) * samples_per_col).min(samples.len());

        if start >= samples.len() {
            break;
        }

        // Get max value in this column's range
        let chunk = &samples[start..end];
        let max_val = chunk.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);

        // Map to row (centered on mid_row)
        let normalized = max_val / scale;
        let half_height = height / 2;
        let row_offset = (normalized * half_height as f32) as usize;

        // Draw vertical line from center
        for row in (mid_row.saturating_sub(row_offset))..=(mid_row + row_offset).min(height - 1) {
            let char = if row == mid_row {
                '─'
            } else if chunk.iter().any(|&s| s > 0.0) && row < mid_row {
                '│'
            } else if chunk.iter().any(|&s| s < 0.0) && row > mid_row {
                '│'
            } else {
                '│'
            };
            grid[row][col] = char;
        }
    }

    // Add axis labels
    let mut output = String::new();
    output.push_str(&format!("+{:.2}\n", scale));
    for row in grid {
        output.push_str(&row.iter().collect::<String>());
        output.push('\n');
    }
    output.push_str(&format!("-{:.2}\n", scale));

    output
}

/// Render mel spectrogram as ASCII heatmap
#[allow(clippy::no_effect_underscore_binding)]
pub fn render_mel_spectrogram(
    mel_data: &[f32],
    n_mels: usize,
    n_frames: usize,
    width: usize,
    height: usize,
) -> String {
    let _span = crate::trace_enter!("tui.render_mel_spectrogram");
    if mel_data.is_empty() || width == 0 || height == 0 || n_frames == 0 {
        return String::new();
    }

    // Heatmap characters (low to high intensity)
    const HEATMAP_CHARS: [char; 10] = [' ', '░', '▒', '▓', '█', '█', '█', '█', '█', '█'];

    // Find min/max for normalization
    let min_val = mel_data.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = mel_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_val - min_val).max(1e-6);

    // Calculate scaling
    let frames_per_col = (n_frames / width).max(1);
    let mels_per_row = (n_mels / height).max(1);

    let mut output = String::new();

    // Render from high frequency (top) to low frequency (bottom)
    for row in 0..height {
        let mel_start = (n_mels - 1).saturating_sub((row + 1) * mels_per_row);
        let mel_end = (n_mels - 1).saturating_sub(row * mels_per_row);

        for col in 0..width {
            let frame_start = col * frames_per_col;
            let frame_end = ((col + 1) * frames_per_col).min(n_frames);

            // Average value in this cell
            let mut sum = 0.0;
            let mut count = 0;

            for frame in frame_start..frame_end {
                for mel in mel_start..=mel_end.min(n_mels - 1) {
                    let idx = frame * n_mels + mel;
                    if idx < mel_data.len() {
                        sum += mel_data[idx];
                        count += 1;
                    }
                }
            }

            let avg = if count > 0 { sum / count as f32 } else { min_val };
            let normalized = ((avg - min_val) / range).clamp(0.0, 1.0);
            let char_idx = (normalized * (HEATMAP_CHARS.len() - 1) as f32) as usize;

            output.push(HEATMAP_CHARS[char_idx.min(HEATMAP_CHARS.len() - 1)]);
        }
        output.push('\n');
    }

    output
}

/// Render attention weights as ASCII heatmap
#[allow(clippy::no_effect_underscore_binding)]
pub fn render_attention_heatmap(
    attention_weights: &[Vec<f32>],
    width: usize,
    height: usize,
) -> String {
    const HEATMAP_CHARS: [char; 10] = [' ', '·', ':', '∴', '▪', '▫', '■', '□', '▣', '█'];

    let _span = crate::trace_enter!("tui.render_attention_heatmap");
    if attention_weights.is_empty() || width == 0 || height == 0 {
        return String::new();
    }

    let n_tokens = attention_weights.len();
    let n_frames = attention_weights.first().map_or(0, |a| a.len());

    if n_frames == 0 {
        return String::new();
    }

    // Find max for normalization
    let max_val = attention_weights
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .fold(0.0_f32, f32::max);
    let scale = if max_val > 0.0 { max_val } else { 1.0 };

    // Calculate scaling
    let tokens_per_row = (n_tokens / height).max(1);
    let frames_per_col = (n_frames / width).max(1);

    let mut output = String::new();

    // Column header (frame indices)
    output.push_str("     ");
    for col in 0..width.min(20) {
        let frame = col * frames_per_col;
        let _ = write!(output, "{:>3}", frame % 1000);
    }
    output.push_str("\n     ");
    for _ in 0..width.min(20) {
        output.push_str("───");
    }
    output.push('\n');

    for row in 0..height.min(n_tokens) {
        let token_start = row * tokens_per_row;
        let token_end = ((row + 1) * tokens_per_row).min(n_tokens);

        // Row label
        let _ = write!(output, "{row:>3} │");

        for col in 0..width {
            let frame_start = col * frames_per_col;
            let frame_end = ((col + 1) * frames_per_col).min(n_frames);

            // Average attention in this cell
            let mut sum = 0.0;
            let mut count = 0;

            for token_idx in token_start..token_end {
                if token_idx < attention_weights.len() {
                    for frame_idx in frame_start..frame_end {
                        if frame_idx < attention_weights[token_idx].len() {
                            sum += attention_weights[token_idx][frame_idx];
                            count += 1;
                        }
                    }
                }
            }

            let avg = if count > 0 { sum / count as f32 } else { 0.0 };
            let normalized = (avg / scale).clamp(0.0, 1.0);
            let char_idx = (normalized * (HEATMAP_CHARS.len() - 1) as f32) as usize;

            output.push(HEATMAP_CHARS[char_idx.min(HEATMAP_CHARS.len() - 1)]);
        }
        output.push('\n');
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_waveform_render_basic() {
        let samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let output = render_waveform(&samples, 20, 5);

        assert!(!output.is_empty());
        assert!(output.contains('│') || output.contains('─'));
    }

    #[test]
    fn test_waveform_empty_input() {
        let output = render_waveform(&[], 20, 5);
        assert!(output.is_empty());
    }

    #[test]
    fn test_mel_render_basic() {
        let mel: Vec<f32> = (0..800).map(|i| -4.0 + (i as f32 / 200.0)).collect();
        let output = render_mel_spectrogram(&mel, 80, 10, 20, 10);

        assert!(!output.is_empty());
    }

    #[test]
    fn test_attention_render_basic() {
        let attention: Vec<Vec<f32>> = (0..5)
            .map(|i| (0..10).map(|j| if i == j / 2 { 1.0 } else { 0.1 }).collect())
            .collect();

        let output = render_attention_heatmap(&attention, 10, 5);

        assert!(!output.is_empty());
        assert!(output.contains('█') || output.contains('·'));
    }

    #[test]
    fn test_waveform_display_struct() {
        let samples: Vec<f32> = vec![0.5, -0.5, 0.3, -0.3];
        let display = WaveformDisplay::new(&samples, 40, 10);

        assert_eq!(display.width(), 40);
        assert_eq!(display.height(), 10);

        let output = display.render();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_mel_display_struct() {
        let mel: Vec<f32> = vec![0.0; 800];
        let display = MelDisplay::new(&mel, 80, 10, 20, 10);

        let output = display.render();
        assert!(!output.is_empty());
    }
}
