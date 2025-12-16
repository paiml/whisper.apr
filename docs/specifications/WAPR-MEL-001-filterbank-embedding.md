# WAPR-MEL-001: Embed Mel Filterbank in .apr Model Files

**Status:** Draft
**Author:** Claude Code
**Created:** 2025-12-16
**GitHub Issue:** https://github.com/paiml/whisper.apr/issues/7

## Abstract

This specification defines how mel filterbank weights are embedded in `.apr` model files to ensure exact numerical reproducibility with OpenAI's Whisper implementation.

## Problem Statement

### Root Cause Analysis (5 Whys)

1. **Why does transcription produce "rererer"?**
   Token 265 ("re") dominates logits in the decoder.

2. **Why does token 265 dominate?**
   Encoder output statistics are wrong (mean/var don't match whisper.cpp).

3. **Why are encoder outputs wrong?**
   Mel spectrogram input to encoder is incorrect.

4. **Why is mel spectrogram incorrect?**
   Filterbank weights don't match OpenAI's implementation.

5. **Why don't filterbank weights match?**
   We compute filterbank from scratch; OpenAI loads pre-computed weights.

### Evidence

| Metric | Our Implementation | OpenAI/whisper.cpp |
|--------|-------------------|-------------------|
| Cosine similarity | 0.13 | 1.0 (reference) |
| Row sum (mel band 0) | 1.0+ | ~0.025 |
| Total filterbank sum | 197 | 2 |
| Normalization | None | Slaney (area-normalized) |

## Solution

Store the mel filterbank in `.apr` model metadata, following the existing pattern for vocabulary.

### Format

The filterbank is stored as two metadata keys:

```json
{
  "mel_filterbank": [0.0, 0.0, ..., 0.0234, ...],  // 80 * 201 = 16,080 f32 values
  "mel_filterbank_shape": [80, 201]
}
```

### Size Analysis

- Elements: 80 (mel bands) × 201 (frequency bins) = 16,080
- JSON size: ~200KB (with full precision)
- Binary size: 64KB (as tensor)

**Recommendation:** Store as JSON metadata for simplicity. 200KB overhead is negligible for model files.

## Implementation

### 1. Model Converter Changes

```rust
// In model converter (e.g., tools/convert_ggml.rs)
fn extract_filterbank(ggml_path: &Path) -> Vec<f32> {
    // Read filterbank from ggml file at known offset
    // whisper.cpp stores at: model.filters.data (80 * 201 floats)
    let file = File::open(ggml_path)?;
    // ... extraction logic
}

fn convert_model(ggml_path: &Path, apr_path: &Path) -> Result<()> {
    let mut writer = AprWriter::new();

    // Extract and embed filterbank
    let filterbank = extract_filterbank(ggml_path)?;
    writer.set_metadata("mel_filterbank",
        JsonValue::Array(filterbank.iter().map(|&f| json!(f)).collect()));
    writer.set_metadata("mel_filterbank_shape", json!([80, 201]));

    // ... rest of conversion
}
```

### 2. Runtime Loading Changes

```rust
// In src/audio/mel.rs
impl MelFilterbank {
    /// Load filterbank from .apr model metadata
    pub fn from_apr_metadata(metadata: &AprMetadata) -> Result<Self, WhisperError> {
        let filters = metadata.get("mel_filterbank")
            .ok_or(WhisperError::MissingFilterbank)?
            .as_array()
            .ok_or(WhisperError::InvalidFilterbank)?
            .iter()
            .map(|v| v.as_f64().map(|f| f as f32))
            .collect::<Option<Vec<f32>>>()
            .ok_or(WhisperError::InvalidFilterbank)?;

        let shape = metadata.get("mel_filterbank_shape")
            .and_then(|v| v.as_array())
            .map(|arr| (arr[0].as_u64()? as usize, arr[1].as_u64()? as usize))
            .ok_or(WhisperError::InvalidFilterbank)?;

        Ok(Self::from_filters(filters, shape.0, shape.1 * 2 - 2, 16000))
    }
}
```

### 3. Transcription Pipeline Changes

```rust
// In src/lib.rs or wherever model is loaded
fn load_model(apr_path: &Path) -> Result<WhisperModel, WhisperError> {
    let reader = AprReader::open(apr_path)?;

    // Load filterbank from metadata (not computed)
    let mel_filterbank = MelFilterbank::from_apr_metadata(&reader.metadata)?;

    // ... load tensors, build model
}
```

## Alternative: Store as Tensor

For larger auxiliary data, storing as a named tensor may be preferable:

```rust
// Writer
writer.add_tensor_f32("audio.mel_filterbank", vec![80, 201], &filterbank);

// Reader
let filterbank = reader.read_tensor_f32("audio.mel_filterbank")?;
```

**Trade-offs:**

| Approach | Pros | Cons |
|----------|------|------|
| JSON metadata | Simple, human-readable | Larger file size |
| Tensor | Compact, binary | Requires tensor reading |

**Decision:** Use JSON metadata for v1. Can migrate to tensor in v2 if needed.

## Verification

### Test Case

```rust
#[test]
fn test_filterbank_from_apr_matches_whisper_cpp() {
    let reader = AprReader::open("models/whisper-tiny.apr").unwrap();
    let mel = MelFilterbank::from_apr_metadata(&reader.metadata).unwrap();

    // Load whisper.cpp filterbank for comparison
    let expected = load_whisper_cpp_filterbank("models/ggml-tiny.bin");

    // Cosine similarity must be > 0.9999
    let similarity = cosine_similarity(mel.filters(), &expected);
    assert!(similarity > 0.9999, "Filterbank mismatch: {}", similarity);
}

#[test]
fn test_transcription_produces_correct_output() {
    let model = WhisperModel::load("models/whisper-tiny.apr").unwrap();
    let audio = load_audio("test-audio/the-birds-can-use.wav");

    let result = model.transcribe(&audio, Default::default()).unwrap();

    assert!(result.text.to_lowercase().contains("the birds can use"));
}
```

## Migration Path

1. **Phase 1:** Update model converter to embed filterbank
2. **Phase 2:** Update runtime to load filterbank from metadata
3. **Phase 3:** Fall back to computed filterbank if metadata missing (backward compat)
4. **Phase 4:** Remove computed filterbank code path (breaking change)

## References

- OpenAI Whisper `audio.py`: Lines 91-107 (mel_filters function)
- whisper.cpp: Line 1584 (filterbank loading)
- librosa: `librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)`
- `examples/compare_filterbank.rs`: Statistical comparison tool
