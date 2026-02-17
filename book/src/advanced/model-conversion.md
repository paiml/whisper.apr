# Custom Model Conversion

whisper.apr supports multiple model formats with different conversion workflows.

## Model Format Overview

| Format | Loading | Notes |
|--------|---------|-------|
| **.apr** | `WhisperApr::load_from_apr()` | Native format, optimized for WASM streaming |
| **.gguf** | Direct via `--model-path` | Pre-quantized, no conversion needed |
| **SafeTensors** | Convert to .apr first | HuggingFace format |

## GGUF Models (No Conversion Needed)

Pre-quantized GGUF models from HuggingFace can be loaded directly:

```bash
# Load GGUF model directly
whisper-apr transcribe -f audio.wav --model-path whisper-tiny.gguf
```

### How GGUF Loading Works

1. Detects GGUF magic bytes (`0x46554747`)
2. Parses tensors via aprender's GGUF parser (supports Q4_0 through Q6_K, F16, F32)
3. Remaps whisper.cpp tensor names to internal names
4. Infers model config from tensor shapes
5. Builds APR representation in memory
6. Loads via standard APR pipeline

### Supported Quantization Levels

| Type | Bits/Weight | Size Reduction |
|------|------------|----------------|
| Q4_0 | 4.5 | ~4x |
| Q4_1 | 5.0 | ~3.5x |
| Q5_0 | 5.5 | ~3x |
| Q5_1 | 6.0 | ~2.7x |
| Q6_K | 6.6 | ~2.4x |
| Q8_0 | 8.5 | ~1.9x |
| F16 | 16 | ~2x |
| F32 | 32 | 1x (baseline) |

## SafeTensors to APR Conversion

Convert HuggingFace SafeTensors models to the optimized .apr format:

```bash
# Using the standalone converter
cargo run --bin whisper-convert --features converter -- \
  --model tiny --output whisper-tiny.apr

# Available models: tiny, base, small, medium, large, large-v3-turbo
cargo run --bin whisper-convert --features converter -- \
  --model large-v3-turbo --output whisper-large-v3-turbo.apr
```

### Moonshine Conversion

Moonshine models are also converted from SafeTensors:

```bash
# Convert Moonshine models
cargo run --bin whisper-convert --features converter -- \
  --model moonshine-tiny --output moonshine-tiny.apr

cargo run --bin whisper-convert --features converter -- \
  --model moonshine-base --output moonshine-base.apr
```

## APR Format Details

The .apr format features:

- **Streaming load**: Start inference before full download completes
- **Zstd compression**: 30-50% smaller than raw weights
- **Embedded vocabulary**: BPE tokens included in the file
- **Mel filterbank**: Pre-computed filterbank embedded
- **Quantization metadata**: Scales and zero-points for dequantization

## Programmatic Conversion

```rust
use aprender::format::apr::AprWriter;

// Build APR file programmatically
let mut writer = AprWriter::new();
writer.set_config(&model_config);
writer.add_tensor("encoder.conv1.weight", &tensor_data);
// ... add all tensors
let apr_bytes = writer.finalize();
```
