# WAPR-TUI-001: Whisper Pipeline Visualization TUI

## Overview

Interactive terminal dashboard for visualizing the Whisper ASR pipeline components like a scientist in a laboratory. Enables real-time observation of audio processing, mel spectrogram transformation, encoder/decoder states, and transcription output.

## Peer-Reviewed Citations

1. **Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022)**.
   "Robust Speech Recognition via Large-Scale Weak Supervision."
   *arXiv preprint arXiv:2212.04356*.
   **Relevance**: Foundational Whisper architecture - encoder-decoder transformer with mel spectrogram input.

2. **Davis, S., & Mermelstein, P. (1980)**.
   "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences."
   *IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357-366*.
   **Relevance**: Mel-frequency cepstral coefficients - the mel filterbank visualization basis.

3. **Bahdanau, D., Cho, K., & Bengio, Y. (2014)**.
   "Neural Machine Translation by Jointly Learning to Align and Translate."
   *arXiv preprint arXiv:1409.0473*.
   **Relevance**: Attention mechanism visualization - cross-attention between audio and text.

4. **Vaswani, A., Shazeer, N., Parmar, N., et al. (2017)**.
   "Attention Is All You Need."
   *Advances in Neural Information Processing Systems, 30*.
   **Relevance**: Transformer architecture - multi-head attention visualization.

5. **Li, J., Ye, Z., Zhang, Z., & Li, T. (2024)**.
   "Audio-Visual Speech Recognition: Progress and Challenges."
   *IEEE/ACM Transactions on Audio, Speech, and Language Processing*.
   **Relevance**: Modern ASR visualization techniques for debugging and analysis.

## TUI Panels

### 1. Waveform Panel
Displays raw audio waveform with:
- Amplitude over time (ASCII art visualization)
- Sample rate indicator
- Duration and position markers
- VAD (Voice Activity Detection) regions highlighted

### 2. Mel Spectrogram Panel
Visualizes 80-bin mel filterbank output:
- Time x Frequency heatmap (block characters)
- Log-mel scale coloring
- 3000-frame standard view
- Zoom/scroll support

### 3. Encoder Panel
Shows encoder transformer state:
- Layer-by-layer activation magnitudes
- Self-attention patterns (condensed view)
- Positional encoding visualization
- Feature extraction progress

### 4. Decoder Panel
Displays decoder generation:
- Token-by-token output
- Cross-attention weights to audio frames
- KV cache status
- Beam/greedy indicator

### 5. Attention Panel
Detailed attention visualization:
- Cross-attention heatmap (tokens x frames)
- Peak attention positions
- Alignment quality metrics

### 6. Transcription Panel
Final output display:
- Generated text with word timestamps
- Confidence scores
- Language detection
- RTF (Real-Time Factor) metrics

### 7. Metrics Panel
Performance instrumentation:
- Component timing breakdown
- Memory usage
- GPU utilization (if applicable)
- Throughput (tokens/sec)

## State Machine

```
┌─────────────┐
│    Idle     │
└──────┬──────┘
       │ load_audio
       ▼
┌─────────────┐
│  Waveform   │──────► Shows raw audio
└──────┬──────┘
       │ compute_mel
       ▼
┌─────────────┐
│    Mel      │──────► Shows spectrogram
└──────┬──────┘
       │ encode
       ▼
┌─────────────┐
│  Encoding   │──────► Shows encoder layers
└──────┬──────┘
       │ decode_start
       ▼
┌─────────────┐
│  Decoding   │──────► Token-by-token
└──────┬──────┘
       │ complete
       ▼
┌─────────────┐
│  Complete   │──────► Full transcription
└─────────────┘
```

## Keyboard Bindings

| Key | Action |
|-----|--------|
| `1` | Waveform panel |
| `2` | Mel spectrogram |
| `3` | Encoder view |
| `4` | Decoder view |
| `5` | Attention view |
| `6` | Transcription |
| `7` | Metrics |
| `Tab` | Next panel |
| `←/→` | Scroll time |
| `↑/↓` | Scroll frequency/layer |
| `Space` | Pause/resume |
| `r` | Reset |
| `q` | Quit |
| `?` | Help |

## Probar UX Coverage Requirements

100% coverage of:
- All 7 panels visited
- All keyboard bindings exercised
- All state transitions tested
- Panel content assertions verified

## Renacer Tracing Spans

```rust
#[instrument(name = "whisper.compute_mel")]
fn compute_mel(&self, audio: &[f32]) -> Result<Vec<f32>>

#[instrument(name = "whisper.encode")]
fn encode(&self, mel: &[f32]) -> Result<EncoderOutput>

#[instrument(name = "whisper.decode_step")]
fn decode_step(&mut self, encoder_output: &[f32]) -> Result<Token>
```

## Implementation Requirements

1. **Test-First**: All TUI tests written before implementation
2. **Probar Integration**: Frame assertions, snapshots, UX coverage
3. **Renacer Spans**: All pipeline components instrumented
4. **PMAT Tracking**: Work tracked with pmat work start/continue/complete
5. **Zero Warnings**: Clippy clean, no dead code

## Files

- `src/tui/mod.rs` - TUI module entry
- `src/tui/app.rs` - Application state
- `src/tui/panels.rs` - Panel rendering
- `src/tui/waveform.rs` - Waveform visualization
- `src/tui/mel.rs` - Mel spectrogram visualization
- `src/tui/attention.rs` - Attention heatmap
- `src/tui/tests.rs` - Probar TUI tests
