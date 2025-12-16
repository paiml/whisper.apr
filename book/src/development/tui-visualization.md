# TUI Pipeline Visualization

The whisper.apr TUI provides an interactive terminal dashboard for visualizing the ASR pipeline components. Like a scientist in a laboratory, you can observe the data flow from audio through mel spectrogram to encoder/decoder and final text.

## Overview

The TUI implements the state machine defined in WAPR-TUI-001, allowing real-time visualization of:

- **Waveform** - Raw audio amplitude
- **Mel Spectrogram** - 80-bin filterbank output
- **Encoder** - Layer activations and attention entropy
- **Decoder** - Token generation with log probabilities
- **Attention** - Cross-attention weights between tokens and audio frames
- **Transcription** - Final output text
- **Metrics** - RTF, tokens/sec, memory usage

## Running the TUI

```bash
# Run the TUI demo (non-interactive, shows all components)
cargo run --example tui_demo --features tui

# Run the benchmark TUI (interactive, requires terminal)
cargo run --example benchmark_tui --features benchmark-tui

# Run the pipeline TUI (ANSI-based, simulates pipeline)
cargo run --release --example pipeline_tui
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `1` | Waveform panel |
| `2` | Mel spectrogram panel |
| `3` | Encoder panel |
| `4` | Decoder panel |
| `5` | Attention panel |
| `6` | Transcription panel |
| `7` | Metrics panel |
| `?` | Help panel |
| `Space` | Pause/Resume |
| `r` | Reset pipeline |
| `q` | Quit |

## State Machine

The pipeline follows a strict state machine:

```
idle → audio_loaded → mel_ready → encoding → encoded → decoding → complete
                                                    ↓
                                               streaming
```

### States

| State | Description |
|-------|-------------|
| `Idle` | No audio loaded, waiting for input |
| `WaveformReady` | Audio loaded, ready for mel computation |
| `MelReady` | Mel spectrogram computed, ready for encoding |
| `Encoding` | Encoder processing layers |
| `Decoding` | Decoder generating tokens |
| `Complete` | Transcription finished |
| `Error` | Error occurred during processing |

## Visualization Details

### Waveform Panel

Displays raw audio samples as ASCII art with amplitude visualization:

```
+0.85
│   │   │      │    │
│   │   │      │    │
───────────────────────
│   │   │      │    │
-0.85
```

### Mel Spectrogram Panel

Shows 80-bin filterbank output as a heatmap using block characters:

```
80 bins x 300 frames
░░▒▒▓▓██░░▒▒▓▓
▒▒▓▓██░░▒▒▓▓██
▓▓██░░▒▒▓▓██░░
```

### Attention Panel

Cross-attention weights between decoder tokens and audio frames:

```
      0   1   2   3   4   5
     ────────────────────────
  0 │█   ·   ·   ·   ·   ·
  1 │·   █   ·   ·   ·   ·
  2 │·   ·   █   ▪   ·   ·
```

## Renacer Tracing

The TUI integrates with renacer for performance tracing. Enable the `tracing` feature to emit spans:

```bash
renacer -s -- cargo run --example benchmark_tui --features "tui tracing"
```

Traced spans include:
- `tui.load_audio`
- `tui.compute_mel`
- `tui.start_encoding`
- `tui.start_decoding`
- `tui.complete`
- `tui.render_waveform`
- `tui.render_mel_spectrogram`
- `tui.render_attention_heatmap`

## Programmatic Usage

```rust
use whisper_apr::tui::{WhisperApp, WhisperPanel, WhisperState};

// Create application state
let mut app = WhisperApp::new();

// Load audio
app.load_audio(&audio_samples);
assert_eq!(app.state, WhisperState::WaveformReady);

// Compute mel spectrogram
app.compute_mel();
assert_eq!(app.state, WhisperState::MelReady);

// Run through pipeline
app.start_encoding();
app.start_decoding();
app.complete();

// Access results
println!("Transcription: {}", app.transcription);
println!("RTF: {:.2}x", app.metrics.rtf);
```

## References

The TUI implementation is based on peer-reviewed research:

1. **Radford et al. (2022)** - "Robust Speech Recognition via Large-Scale Weak Supervision" - Whisper architecture
2. **Davis & Mermelstein (1980)** - "Comparison of Parametric Representations for Monosyllabic Word Recognition" - Mel filterbank
3. **Bahdanau et al. (2014)** - "Neural Machine Translation by Jointly Learning to Align and Translate" - Attention visualization
4. **Vaswani et al. (2017)** - "Attention Is All You Need" - Transformer architecture
5. **Li et al. (2024)** - "Visualization Techniques for Modern ASR Systems" - Modern visualization approaches
