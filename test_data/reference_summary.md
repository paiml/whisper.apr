# Ground Truth Extraction Summary

**Audio**: `demos/test-audio/test-speech-1.5s.wav`
**Model**: `../whisper.cpp/models/ggml-tiny.bin`

## Steps Extracted

| Step | Shape | Range | Mean |
|------|-------|-------|------|
| step_a_audio | 24000 | [-0.1983, 0.2980] | 0.0002 |
| step_b_filterbank | 80x201 | [0.0000, 0.0259] | 0.0001 |
| step_c_mel_numpy | 148x80 | [-0.7658, 1.2342] | -0.2148 |

## Usage

```rust
// Load ground truth
let gt_mel = load_f32_binary("golden_traces/step_c_mel_numpy.bin")?;

// Compare with our mel
let our_mel = model.compute_mel(&audio)?;
let cosine = cosine_similarity(&gt_mel, &our_mel);
assert!(cosine > 0.99, "Mel divergence: {}", 1.0 - cosine);
```