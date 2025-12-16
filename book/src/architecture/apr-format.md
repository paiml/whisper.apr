# .apr Model Format

The `.apr` (Aprender) format is a binary model format optimized for WASM delivery. It stores model weights, vocabulary, and mel filterbank in a single file with CRC32 integrity verification.

## Format Overview

```
┌─────────────────────────────────────────────────┐
│ Magic (4 bytes): "APR1"                         │
├─────────────────────────────────────────────────┤
│ Header (48 bytes)                               │
│   - version, model_type, quantization           │
│   - n_tensors, has_vocab, has_filterbank        │
│   - model dimensions (n_vocab, n_layers, etc.)  │
├─────────────────────────────────────────────────┤
│ Tensor Index (96 bytes × n_tensors)             │
│   - name, shape, offset, size per tensor        │
├─────────────────────────────────────────────────┤
│ [Scale Table - int8 only] (4 bytes × n_tensors) │
├─────────────────────────────────────────────────┤
│ Tensor Data (variable)                          │
├─────────────────────────────────────────────────┤
│ [Vocabulary Section] (optional)                 │
│   - size (u32) + vocab_bytes                    │
├─────────────────────────────────────────────────┤
│ [Filterbank Section] (optional)                 │
│   - size (u32) + filterbank_bytes               │
├─────────────────────────────────────────────────┤
│ CRC32 (4 bytes)                                 │
└─────────────────────────────────────────────────┘
```

## Header Structure (48 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 2 | version | Format version (currently 1) |
| 2 | 1 | model_type | 0=tiny, 1=tiny.en, 2=base, ... |
| 3 | 1 | quantization | 0=f32, 1=f16, 2=int8 |
| 4 | 1 | compressed | 0=no, 1=LZ4 compressed |
| 5 | 2 | n_tensors | Number of tensors (u16) |
| 7 | 1 | flags | bit 0=has_vocab, bit 1=has_filterbank |
| 8 | 4 | n_vocab | Vocabulary size |
| 12 | 4 | n_audio_ctx | Audio context length |
| 16 | 4 | n_audio_state | Audio hidden dimension |
| 20 | 4 | n_audio_head | Audio attention heads |
| 24 | 4 | n_audio_layer | Audio encoder layers |
| 28 | 4 | n_text_ctx | Text context length |
| 32 | 4 | n_text_state | Text hidden dimension |
| 36 | 4 | n_text_head | Text attention heads |
| 40 | 4 | n_text_layer | Text decoder layers |
| 44 | 4 | n_mels | Mel frequency bands |

## Filterbank Section

When `has_filterbank` flag is set (byte 7, bit 1), the filterbank section follows the vocabulary section:

```
Filterbank Section:
  ├── size (u32): Total bytes of data following
  ├── n_mels (u32): Number of mel bands (80 or 128)
  ├── n_freqs (u32): Number of frequency bins (201)
  └── data (f32[]): Raw filterbank matrix in row-major order
```

### Filterbank Sizes

| Model Type | n_mels | n_freqs | Data Size | Total Section |
|------------|--------|---------|-----------|---------------|
| tiny-large-v2 | 80 | 201 | 64,320 bytes | 64,328 bytes |
| large-v3 | 128 | 201 | 102,912 bytes | 102,920 bytes |

### Why Embed the Filterbank?

The mel filterbank is **not just a preprocessing parameter** - it directly affects model behavior:

1. **Training-Inference Match**: Whisper was trained with librosa's slaney-normalized filterbank
2. **Numerical Precision**: Computing filterbanks from scratch produces different values
3. **Avoiding Hallucinations**: Mismatched filterbanks cause the "rererer" bug

## Reading .apr Files

```rust
use whisper_apr::format::{AprReader, MelFilterbankData};
use whisper_apr::audio::MelFilterbank;

// Load model
let model_bytes = std::fs::read("model.apr")?;
let reader = AprReader::new(model_bytes)?;

// Check what's embedded
println!("Has vocabulary: {}", reader.has_vocabulary());
println!("Has filterbank: {}", reader.has_mel_filterbank());

// Read filterbank
if let Some(fb_data) = reader.read_mel_filterbank() {
    println!("Filterbank: {}x{}", fb_data.n_mels, fb_data.n_freqs);
    let mel = MelFilterbank::from_apr_data(fb_data, 16000);
}

// Read vocabulary
if let Some(vocab) = reader.read_vocabulary() {
    println!("Vocabulary: {} tokens", vocab.len());
}
```

## Writing .apr Files

```rust
use whisper_apr::format::{AprWriter, AprHeader, MelFilterbankData};

// Create writer
let mut writer = AprWriter::from_config(&model_config);

// Add tensors
writer.add("encoder.conv1.weight", vec![384, 80, 3], weights);
writer.add("encoder.conv1.bias", vec![384], bias);

// Embed vocabulary
writer.set_vocabulary(vocab);

// Embed mel filterbank (critical for correct transcription)
let filterbank = MelFilterbankData::mel_80(filterbank_data);
writer.set_mel_filterbank(filterbank);

// Write to file
writer.write_to_file("model.apr")?;
```

## Quantization Support

The format supports multiple quantization levels:

| Type | Bytes/Element | Description |
|------|---------------|-------------|
| f32 | 4 | Full precision |
| f16 | 2 | Half precision |
| int8 | 1 | 8-bit quantized with per-tensor scale |

For int8 models, a scale table is stored between the tensor index and tensor data.

## Model Conversion

Convert HuggingFace models to .apr format:

```bash
# Full precision with filterbank
cargo run --release --bin whisper-convert -- tiny --output whisper-tiny.apr

# Int8 quantized
cargo run --release --bin whisper-convert -- tiny --quantize int8 --output whisper-tiny-q8.apr
```

The converter automatically:
1. Downloads weights from HuggingFace
2. Downloads vocab and tokenizer
3. Downloads mel filterbank from OpenAI
4. Embeds everything in a single .apr file

## File Size Examples

| Model | f32 Size | int8 Size | With Filterbank |
|-------|----------|-----------|-----------------|
| tiny | 150 MB | 39 MB | +64 KB |
| base | 290 MB | 74 MB | +64 KB |
| small | 960 MB | 244 MB | +64 KB |
| large-v3 | 6.2 GB | 1.5 GB | +103 KB |

## Example: Inspect .apr File

```bash
cargo run --example filterbank_embedding -- models/whisper-tiny.apr
```

Output:
```
=== APR File Inspection ===
Model: whisper-tiny.apr
Size: 150.2 MB
Tensors: 167
Has Vocabulary: true (51865 tokens)
Has Filterbank: true (80x201)
Filterbank row 0 sum: 0.024891 (slaney normalized)
```

## Related Resources

- [Audio Pipeline](./audio-pipeline.md) - How filterbank is used
- [Model Conversion](../advanced/model-conversion.md) - Converting models
- [Quantization](./quantization.md) - Quantization details
