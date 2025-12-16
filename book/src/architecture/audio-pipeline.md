# Audio Pipeline

The audio pipeline transforms raw audio into mel spectrograms that Whisper's encoder can process. This is a critical component where numerical precision directly impacts transcription quality.

## Pipeline Overview

```
Audio (f32[]) → Resampler (16kHz) → MelFilterbank (80 bins) → Encoder
```

## Mel Filterbank: The Critical Component

The mel filterbank converts power spectra to mel-scale representations. Whisper was trained with a specific filterbank from librosa using **slaney normalization**.

### The Filterbank Mismatch Problem

Computing filterbanks from scratch produces **different** values than OpenAI's training filterbank:

| Source | Row Sum | Normalization |
|--------|---------|---------------|
| OpenAI (slaney) | ~0.025 | Area-normalized |
| Computed from scratch | ~1.0+ | Peak-normalized |

This mismatch causes the infamous "rererer" hallucination bug where the model produces repetitive nonsense instead of actual transcription.

### The Solution: Embedded Filterbank

The `.apr` format embeds OpenAI's exact filterbank directly in the model file:

```rust
// Load model with embedded filterbank
let reader = AprReader::new(model_bytes)?;

// Get the slaney-normalized filterbank from model
let filterbank = if let Some(fb_data) = reader.read_mel_filterbank() {
    // Use exact OpenAI filterbank (recommended)
    MelFilterbank::from_apr_data(fb_data, 16000)
} else {
    // Fallback to computed (may cause issues)
    MelFilterbank::new(80, 400, 16000)
};
```

### Verifying Filterbank Correctness

Run the comparison example to verify filterbank alignment:

```bash
cargo run --example filterbank_embedding
```

This shows the cosine similarity between embedded and computed filterbanks. A value of 1.0 means exact match; values below 0.99 indicate potential transcription issues.

## Mel Spectrogram Computation

### Parameters (Whisper Standard)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_mels` | 80 (or 128 for large-v3) | Mel frequency bands |
| `n_fft` | 400 | FFT window size |
| `hop_length` | 160 | Samples between frames |
| `sample_rate` | 16000 Hz | Required sample rate |

### Processing Steps

1. **Windowing**: Apply Hann window to each frame
2. **FFT**: Compute power spectrum via FFT
3. **Mel Filterbank**: Apply filterbank to get mel energies
4. **Log Compression**: Apply `log10(max(energy, 1e-10))`
5. **Normalization**: Whisper-specific clamping and scaling

```rust
use whisper_apr::audio::MelFilterbank;

// Create from embedded filterbank (preferred)
let mel = MelFilterbank::from_apr_data(filterbank_data, 16000);

// Compute mel spectrogram
let mel_spec = mel.compute(&audio_samples, 160)?;

// SIMD-optimized version
let mel_spec = mel.compute_simd(&audio_samples, 160)?;
```

## Example: Full Pipeline

```rust
use whisper_apr::audio::{MelFilterbank, Resampler};
use whisper_apr::format::AprReader;

fn preprocess_audio(audio: &[f32], sample_rate: u32, model: &AprReader) -> Vec<f32> {
    // Resample to 16kHz if needed
    let audio_16k = if sample_rate != 16000 {
        Resampler::new(sample_rate, 16000).resample(audio)
    } else {
        audio.to_vec()
    };

    // Get mel filterbank from model (ensures exact match)
    let mel = model.read_mel_filterbank()
        .map(|fb| MelFilterbank::from_apr_data(fb, 16000))
        .unwrap_or_else(|| MelFilterbank::new(80, 400, 16000));

    // Compute mel spectrogram
    mel.compute_simd(&audio_16k, 160).expect("mel computation")
}
```

## File Format: Filterbank Section

The `.apr` format stores filterbank data after the vocabulary section:

```
[Header byte 7, bit 1] = has_filterbank flag

Filterbank section:
  - size (u32): Total bytes of filterbank data
  - n_mels (u32): Number of mel bands (80 or 128)
  - n_freqs (u32): Number of frequency bins (201)
  - data (f32[]): Raw filterbank matrix (n_mels × n_freqs)
```

Total size: `8 + (n_mels × n_freqs × 4)` bytes

- mel_80: 64,328 bytes
- mel_128: 102,920 bytes

## Related Resources

- [.apr Model Format](./apr-format.md) - Full format specification
- [Filterbank Embedding Example](../examples/filterbank-embedding.md)
- [Model Conversion](../advanced/model-conversion.md) - How filterbank is embedded during conversion
