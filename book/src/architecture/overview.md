# Architecture Overview

Whisper.apr implements OpenAI's Whisper architecture in pure Rust, optimized for WASM deployment.

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

## Key Components

### Audio Pipeline (`src/audio/`)

1. **Resampler** - Converts input audio to 16kHz mono
2. **MelFilterbank** - Computes 80-bin mel spectrogram
3. **Normalization** - Standardizes input for the encoder

### Transformer (`src/model/`)

1. **Encoder** - Processes mel spectrogram into audio features
   - Convolutional stem (2 layers)
   - Transformer blocks with self-attention
   - Sinusoidal positional encoding

2. **Decoder** - Generates text tokens autoregressively
   - Masked self-attention
   - Cross-attention to encoder output
   - Linear projection to vocabulary

3. **Attention** - Multi-head attention with SIMD optimization
   - Query, Key, Value projections
   - Scaled dot-product attention
   - Output projection

### Tokenizer (`src/tokenizer/`)

- BPE (Byte Pair Encoding) tokenization
- 51,865 token vocabulary
- Special tokens for language, task, timestamps

### Inference (`src/inference/`)

1. **Greedy** - Fast, memory-efficient decoding
2. **BeamSearch** - Higher quality with configurable beam width

## Data Flow

```
Audio (f32[])
    │
    ▼
Resample to 16kHz
    │
    ▼
Mel Spectrogram [T, 80]
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

| Model | d_model | n_heads | n_layers | Parameters |
|-------|---------|---------|----------|------------|
| tiny  | 384     | 6       | 4        | 39M        |
| base  | 512     | 8       | 6        | 74M        |
| small | 768     | 12      | 12       | 244M       |

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
