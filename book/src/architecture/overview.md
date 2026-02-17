# Architecture Overview

Whisper.apr implements OpenAI's Whisper architecture and Moonshine ASR in pure Rust, optimized for WASM deployment.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Whisper.apr                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │    Audio     │    │     Mel      │    │    Transformer   │  │
│  │ Preprocessor │───►│  Spectrogram │───►│     Encoder      │  │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘  │
│                                                    │            │
│                                                    ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Output     │◄───│   Tokenizer  │◄───│    Transformer   │  │
│  │    Text      │    │    (BPE)     │    │     Decoder      │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Supported Model Architectures

### Whisper (OpenAI)

Standard encoder-decoder transformer with multi-head attention (MHA) and cross-attention.

### Moonshine

Lightweight alternative with Grouped Query Attention (GQA) decoder and ConvStem encoder. Smaller model sizes with competitive accuracy.

## Key Components

### Audio Pipeline (`src/audio/`)

1. **Symphonia Decoder** - Decodes MP3, FLAC, OGG, AAC, M4A, WAV
2. **Resampler** - Converts input audio to 16kHz mono
3. **MelFilterbank** - Computes 80-bin or 128-bin mel spectrogram
4. **Normalization** - Standardizes input for the encoder

### Transformer (`src/model/`)

1. **Encoder** - Processes mel spectrogram into audio features
   - Convolutional stem (2 layers for Whisper, ConvStem for Moonshine)
   - Transformer blocks with self-attention
   - Sinusoidal positional encoding

2. **Decoder** - Generates text tokens autoregressively
   - Masked self-attention (MHA for Whisper, GQA for Moonshine)
   - Cross-attention to encoder output
   - Linear projection to vocabulary

3. **Attention** - Multi-head attention with SIMD optimization via trueno
   - Query, Key, Value projections
   - Scaled dot-product attention (SDPA)
   - Output projection
   - Tiled MatVec fast path for single-token decoding (3.5x speedup)

### Tokenizer (`src/tokenizer/`)

- BPE (Byte Pair Encoding) tokenization
- 51,865 token vocabulary
- Special tokens for language (99 languages), task, timestamps

### Inference (`src/inference/`)

1. **Greedy** - Fast, memory-efficient decoding
2. **BeamSearch** - Higher quality with configurable beam width

### Format (`src/format/`)

1. **.apr Loader** - Streaming, compressed model loading
2. **GGUF Loader** - Direct loading of pre-quantized GGUF models
3. **Validation** - 25-point model QA checklist

## Data Flow

```
Audio (f32[] or MP3/FLAC/OGG/M4A)
    │
    ▼
Symphonia Decode + Resample to 16kHz
    │
    ▼
Mel Spectrogram [T, 80] or [T, 128]
    │
    ▼
Encoder (Transformer)
    │
    ▼
Audio Features [T/2, d_model]
    │
    ▼
Decoder (Autoregressive)
    │
    ▼
Token IDs [N]
    │
    ▼
BPE Decode
    │
    ▼
Text Output
```

## Model Configurations

### Whisper Models

| Model | d_model | n_heads | Enc Layers | Dec Layers | Mels | Parameters |
|-------|---------|---------|------------|------------|------|------------|
| tiny  | 384     | 6       | 4          | 4          | 80   | 39M        |
| base  | 512     | 8       | 6          | 6          | 80   | 74M        |
| small | 768     | 12      | 12         | 12         | 80   | 244M       |
| medium | 1024   | 16      | 24         | 24         | 80   | 769M       |
| large | 1280    | 20      | 32         | 32         | 128  | 1.5B       |
| large-v3-turbo | 1280 | 20 | 32       | 4          | 128  | 809M       |

### Moonshine Models

| Model | d_model | n_heads | Enc Layers | Dec Layers | Parameters |
|-------|---------|---------|------------|------------|------------|
| moonshine-tiny | 288 | 6    | 6          | 6          | 27M        |
| moonshine-base | 416 | 8    | 8          | 8          | 61M        |

## WASM Considerations

- **Memory Limits**: Safari iOS ~1GB, other browsers ~4GB
- **SIMD**: WASM SIMD 128-bit for 2-4x speedup
- **Streaming**: Progressive model loading via 64KB blocks
- **Web Workers**: Offload inference from main thread

## Trueno Integration

All matrix operations dispatch through Trueno for automatic SIMD acceleration:

```rust
use trueno::{Vector, Matrix};

// Trueno selects optimal backend (Scalar, SIMD, WASM SIMD)
let attention_scores = trueno::matmul(&query, &key.transpose());
let softmax_weights = trueno::softmax(&attention_scores);
let output = trueno::matmul(&softmax_weights, &value);
```
