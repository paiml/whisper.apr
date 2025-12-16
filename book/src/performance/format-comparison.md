# Model Format Comparison

The `.apr` format is designed specifically for efficient WASM delivery. This page compares APR against other common model formats.

## Format Overview

| Format | Description | Use Case |
|--------|-------------|----------|
| **SafeTensors** | HuggingFace standard format | Training, Python ecosystem |
| **GGML/GGUF** | whisper.cpp native format | Desktop inference |
| **APR (f32)** | Unquantized APR | Maximum accuracy |
| **APR (int8)** | Quantized APR | WASM delivery ✨ |

## File Size Comparison

Benchmark results for Whisper Tiny model:

| Format | Size | vs SafeTensors | WASM Suitability |
|--------|------|----------------|------------------|
| SafeTensors | 145 MB | 100% (baseline) | ❌ Too large |
| GGML | 75 MB | 52% | ⚠️ Moderate |
| APR-f32 | 145 MB | 100% | ❌ Too large |
| **APR-int8** | **37 MB** | **25%** | ✅ Excellent |

**Key insight**: APR-int8 achieves **75% compression** with minimal quality loss.

## Loading Performance

Measured on native Rust (representative of WASM performance):

| Operation | APR-f32 | APR-int8 | Speedup |
|-----------|---------|----------|---------|
| File Read | 87ms | **21ms** | 4.1x |
| Format Parse | 73ms | **19ms** | 3.8x |
| Full Model Load | 490ms | **416ms** | 1.2x |
| Single Tensor Load | 65µs | **6.5µs** | 10x |

### Why APR-int8 is Faster

1. **Smaller file**: 37MB reads faster than 145MB
2. **Simple dequantization**: `int8 * scale` is very fast
3. **Cache-friendly**: Smaller data fits in CPU cache
4. **Streaming-ready**: LZ4 block compression (coming soon)

## First Token Latency

Time from audio input to first transcription token:

| Component | Time | Notes |
|-----------|------|-------|
| Mel Spectrogram | ~10ms | 1 second of audio |
| Encoder | ~122ms | Full audio context |
| Decoder (1 step) | ~156ms | Initial tokens |
| **Total** | **~288ms** | First word appears |

## Running the Benchmark

```bash
# Run format comparison benchmark
cargo bench --bench format_comparison

# Run with baseline comparison
cargo bench --bench format_comparison -- --save-baseline main
```

## Example: Format Comparison

```bash
# Run the format comparison example
cargo run --example format_comparison --release
```

Output:
```
=== Model Format Comparison ===

Format          Size         Compression
─────────────── ──────────── ────────────
SafeTensors     145.0 MB     100.0%
GGML            75.0 MB      51.7%
APR-f32         145.0 MB     100.0%
APR-int8        37.0 MB      25.5%

=== Loading Performance ===

Format          Read Time    Parse Time   Total
─────────────── ──────────── ──────────── ────────────
APR-f32         87ms         73ms         490ms
APR-int8        21ms         19ms         416ms
```

## Recommendations

### For WASM/Browser Deployment
Use **APR-int8** for:
- Smallest download size (37MB)
- Fastest loading time
- Acceptable quality for most use cases

### For Maximum Accuracy
Use **APR-f32** when:
- Accuracy is critical
- Network bandwidth is not a concern
- Running on native (non-WASM)

### Migration from Other Formats

Convert from SafeTensors:
```bash
cargo run --bin convert -- --model tiny --quantize int8 -o whisper-tiny-int8.apr
```

## Technical Details

### APR Format Structure

```
┌─────────────────────────────────────┐
│ Header (32 bytes)                   │
│   - Magic: "APR1"                   │
│   - Version, quantization, n_tensors│
│   - Model config (d_model, n_heads) │
├─────────────────────────────────────┤
│ Tensor Index (32 bytes × n_tensors) │
│   - Name (24 bytes, null-padded)    │
│   - Offset, size, shape             │
├─────────────────────────────────────┤
│ Scale Table (int8 only)             │
│   - 4 bytes per tensor              │
├─────────────────────────────────────┤
│ Tensor Data                         │
│   - Contiguous, aligned             │
│   - f32 or int8 depending on quant  │
├─────────────────────────────────────┤
│ Vocabulary (optional)               │
│   - Token count + token data        │
└─────────────────────────────────────┘
```

### Int8 Quantization

Per-tensor symmetric quantization:
```
scale = max(abs(tensor)) / 127
quantized = round(tensor / scale)
dequantized = quantized * scale
```

Benefits:
- Simple, fast dequantization
- No zero-point offset needed
- Compatible with SIMD operations
