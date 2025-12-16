//! TUI Panel Rendering
//!
//! Renders the whisper dashboard panels using ratatui.

use super::app::{WhisperApp, WhisperPanel, WhisperState};
use super::visualization::{render_attention_heatmap, render_mel_spectrogram, render_waveform};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, Tabs},
    Frame,
};

/// Render the main whisper dashboard
pub fn render_whisper_dashboard(f: &mut Frame, app: &WhisperApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Tabs
            Constraint::Min(10),     // Main content
            Constraint::Length(3),  // Status bar
        ])
        .split(f.area());

    // Render tab bar
    render_tabs(f, app, chunks[0]);

    // Render main panel
    render_main_panel(f, app, chunks[1]);

    // Render status bar
    render_status_bar(f, app, chunks[2]);
}

/// Render tab navigation
fn render_tabs(f: &mut Frame, app: &WhisperApp, area: Rect) {
    let titles: Vec<Line> = WhisperPanel::titles()
        .iter()
        .map(|t| Line::from(*t))
        .collect();

    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title("Navigation"))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .select(app.current_panel.index());

    f.render_widget(tabs, area);
}

/// Render main content panel
fn render_main_panel(f: &mut Frame, app: &WhisperApp, area: Rect) {
    match app.current_panel {
        WhisperPanel::Waveform => render_waveform_panel(f, app, area),
        WhisperPanel::Mel => render_mel_panel(f, app, area),
        WhisperPanel::Encoder => render_encoder_panel(f, app, area),
        WhisperPanel::Decoder => render_decoder_panel(f, app, area),
        WhisperPanel::Attention => render_attention_panel(f, app, area),
        WhisperPanel::Transcription => render_transcription_panel(f, app, area),
        WhisperPanel::Metrics => render_metrics_panel(f, app, area),
        WhisperPanel::Help => render_help_panel(f, app, area),
    }
}

/// Render waveform visualization panel
fn render_waveform_panel(f: &mut Frame, app: &WhisperApp, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title("WAVEFORM - Audio Signal");

    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.audio_data.is_empty() {
        let text = Paragraph::new("No audio loaded. Load audio to visualize waveform.")
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(text, inner);
        return;
    }

    // Render waveform ASCII art
    let waveform = render_waveform(&app.audio_data, inner.width as usize, inner.height as usize);
    let info = format!(
        "{} samples @ {}Hz ({:.2}s)",
        app.audio_data.len(),
        app.sample_rate,
        app.metrics.audio_duration_secs
    );

    let content = format!("{}\n\n{}", info, waveform);
    let paragraph = Paragraph::new(content);
    f.render_widget(paragraph, inner);
}

/// Render mel spectrogram panel
fn render_mel_panel(f: &mut Frame, app: &WhisperApp, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title("MEL SPECTROGRAM - 80 Mel Bins");

    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.mel_data.is_empty() {
        let text = Paragraph::new("No mel spectrogram computed. Compute mel to visualize.")
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(text, inner);
        return;
    }

    // Render mel heatmap
    let heatmap = render_mel_spectrogram(
        &app.mel_data,
        80,
        app.mel_frames,
        inner.width as usize,
        inner.height as usize,
    );
    let info = format!("80 bins x {} frames", app.mel_frames);

    let content = format!("{}\n{}", info, heatmap);
    let paragraph = Paragraph::new(content);
    f.render_widget(paragraph, inner);
}

/// Render encoder panel
fn render_encoder_panel(f: &mut Frame, app: &WhisperApp, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title("ENCODER - Layer Activations");

    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.encoder_metrics.is_empty() {
        let text = Paragraph::new("No encoder data. Start encoding to see layer activations.")
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(text, inner);
        return;
    }

    // Create table of layer metrics
    let header = Row::new(vec!["Layer", "Mean Act.", "Max Act.", "Attn Entropy"])
        .style(Style::default().add_modifier(Modifier::BOLD))
        .bottom_margin(1);

    let rows: Vec<Row> = app
        .encoder_metrics
        .iter()
        .map(|m| {
            Row::new(vec![
                Cell::from(format!("Layer {}", m.layer)),
                Cell::from(format!("{:.3}", m.mean_activation)),
                Cell::from(format!("{:.3}", m.max_activation)),
                Cell::from(format!("{:.3}", m.attention_entropy)),
            ])
        })
        .collect();

    let table = Table::new(rows, [
        Constraint::Length(10),
        Constraint::Length(12),
        Constraint::Length(12),
        Constraint::Length(14),
    ])
    .header(header)
    .block(Block::default());

    f.render_widget(table, inner);
}

/// Render decoder panel
fn render_decoder_panel(f: &mut Frame, app: &WhisperApp, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title("DECODER - Token Generation");

    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.decoder_tokens.is_empty() {
        let text = Paragraph::new("No tokens generated. Start decoding to see tokens.")
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(text, inner);
        return;
    }

    // Create table of tokens
    let header = Row::new(vec!["Idx", "Token ID", "Text", "Log P"])
        .style(Style::default().add_modifier(Modifier::BOLD))
        .bottom_margin(1);

    let rows: Vec<Row> = app
        .decoder_tokens
        .iter()
        .enumerate()
        .map(|(i, t)| {
            Row::new(vec![
                Cell::from(format!("{}", i)),
                Cell::from(format!("{}", t.id)),
                Cell::from(t.text.clone()),
                Cell::from(format!("{:.3}", t.log_prob)),
            ])
        })
        .collect();

    let table = Table::new(rows, [
        Constraint::Length(5),
        Constraint::Length(10),
        Constraint::Length(20),
        Constraint::Length(10),
    ])
    .header(header)
    .block(Block::default());

    f.render_widget(table, inner);
}

/// Render attention panel
fn render_attention_panel(f: &mut Frame, app: &WhisperApp, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title("ATTENTION - Cross-Attention Weights");

    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.attention_weights.is_empty() {
        let text = Paragraph::new("No attention data. Decode to see cross-attention.")
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(text, inner);
        return;
    }

    // Render attention heatmap
    let heatmap = render_attention_heatmap(
        &app.attention_weights,
        inner.width as usize,
        inner.height as usize,
    );

    let info = format!(
        "Cross-attention: {} tokens x {} frames",
        app.attention_weights.len(),
        app.attention_weights.first().map(|a| a.len()).unwrap_or(0)
    );

    let content = format!("{}\n{}", info, heatmap);
    let paragraph = Paragraph::new(content);
    f.render_widget(paragraph, inner);
}

/// Render transcription panel
fn render_transcription_panel(f: &mut Frame, app: &WhisperApp, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title("TRANSCRIPTION - Final Output");

    let inner = block.inner(area);
    f.render_widget(block, area);

    let content = if app.transcription.is_empty() {
        "No transcription yet. Complete pipeline to see result.".to_string()
    } else {
        let mut text = String::new();
        text.push_str(&format!("Result: {}\n\n", app.transcription));
        text.push_str("Token Details:\n");

        for (i, token) in app.decoder_tokens.iter().enumerate() {
            text.push_str(&format!(
                "  [{}] {} (p={:.3})\n",
                i, token.text, token.log_prob.exp()
            ));
        }

        text
    };

    let paragraph = Paragraph::new(content);
    f.render_widget(paragraph, inner);
}

/// Render metrics panel
fn render_metrics_panel(f: &mut Frame, app: &WhisperApp, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title("METRICS - Performance Data");

    let inner = block.inner(area);
    f.render_widget(block, area);

    let m = &app.metrics;

    let content = format!(
        r#"Pipeline Performance Metrics

Audio Duration:    {:.2} seconds
Sample Rate:       {} Hz
Mel Frames:        {}

Timing Breakdown:
  Mel Compute:     {:.2} ms
  Encoder:         {:.2} ms
  Decoder:         {:.2} ms
  ─────────────────────
  Total:           {:.2} ms

RTF (Real-Time Factor): {:.3}x
  (< 1.0 = faster than real-time)

Tokens Generated:  {}
Tokens/Second:     {:.1}
"#,
        m.audio_duration_secs,
        app.sample_rate,
        app.mel_frames,
        m.mel_time_ms,
        m.encoder_time_ms,
        m.decoder_time_ms,
        m.total_time_ms,
        m.rtf,
        m.tokens_generated,
        if m.total_time_ms > 0.0 {
            m.tokens_generated as f32 / (m.total_time_ms / 1000.0)
        } else {
            0.0
        }
    );

    let paragraph = Paragraph::new(content);
    f.render_widget(paragraph, inner);
}

/// Render help panel
fn render_help_panel(f: &mut Frame, _app: &WhisperApp, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title("HELP - Keyboard Shortcuts");

    let inner = block.inner(area);
    f.render_widget(block, area);

    let content = r#"Keyboard Bindings

Panel Navigation:
  1  Waveform panel
  2  Mel spectrogram panel
  3  Encoder panel
  4  Decoder panel
  5  Attention panel
  6  Transcription panel
  7  Metrics panel
  ?  This help panel

Controls:
  Space  Pause/Resume
  r      Reset pipeline
  q      Quit

Scroll (where applicable):
  ←/→    Scroll horizontally
  ↑/↓    Scroll vertically

References:
  Radford et al. (2022) - Whisper architecture
  Davis & Mermelstein (1980) - Mel filterbank
  Vaswani et al. (2017) - Transformer attention
"#;

    let paragraph = Paragraph::new(content);
    f.render_widget(paragraph, inner);
}

/// Render status bar
fn render_status_bar(f: &mut Frame, app: &WhisperApp, area: Rect) {
    let state_color = match app.state {
        WhisperState::Idle => Color::DarkGray,
        WhisperState::WaveformReady => Color::Blue,
        WhisperState::MelReady => Color::Cyan,
        WhisperState::Encoding => Color::Yellow,
        WhisperState::Decoding => Color::Magenta,
        WhisperState::Complete => Color::Green,
        WhisperState::Error => Color::Red,
    };

    let state_text = app.state_description();
    let status = app.status_message.as_deref().unwrap_or("");
    let paused = if app.paused { " [PAUSED]" } else { "" };

    let spans = vec![
        Span::styled(
            format!(" {} ", state_text),
            Style::default().fg(Color::Black).bg(state_color),
        ),
        Span::raw(format!(" {} {}", status, paused)),
    ];

    let paragraph = Paragraph::new(Line::from(spans))
        .block(Block::default().borders(Borders::ALL));

    f.render_widget(paragraph, area);
}
