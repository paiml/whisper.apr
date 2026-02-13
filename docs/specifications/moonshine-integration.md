# Moonshine Integration Specification

**Document:** WAPR-MOONSHINE-001
**Status:** Draft
**Version:** 1.0.0
**Date:** 2026-02-13
**License:** Moonshine is MIT-licensed (Useful Sensors, Inc.)

---

## 1. Executive Summary

Moonshine is a speech recognition model by Useful Sensors that is architecturally
optimized for short-form audio. Its key advantage over OpenAI Whisper is
**variable-length input**: Moonshine processes only the audio it receives, while
Whisper zero-pads all input to 30 seconds regardless of actual duration. For a 1.5s
utterance, Whisper computes over 1500 mel frames; Moonshine computes ~7 encoder
frames.

This specification makes **Moonshine the default model family** in whisper.apr while
retaining full Whisper support as an alternative. The change is justified by:

- **Smaller models**: Moonshine tiny is 27.1M params vs Whisper tiny's 39M (30% smaller)
- **Proportional compute**: Processing time scales with audio length, not a fixed 30s window
- **Architecture reuse**: Moonshine uses RoPE, SwiGLU, and GQA — all already implemented in whisper.apr for LFM2
- **MIT license**: No usage restrictions, unlike Whisper's MIT license with OpenAI's model weights under their own terms

No code changes are included in this specification. This is a design document only.

---

## 2. Architecture Comparison

| Aspect | Whisper | Moonshine |
|--------|---------|-----------|
| **Audio frontend** | 80-mel filterbank (FFT + Hann window + triangular filters) | Learned conv stem (3 Conv1d layers, stride 441->4->2) |
| **Positional encoding** | Sinusoidal (fixed 1500 frames max) | RoPE (variable length, no max) |
| **FFN activation** | GELU (4x expansion, 2 projections) | SwiGLU (3-projection gated, ~2.67x expansion) |
| **Attention** | Standard multi-head attention (MHA) | Grouped Query Attention (GQA) |
| **Input length** | Fixed 30s (zero-padded to 480,000 samples) | Variable (proportional to audio duration) |
| **Vocabulary** | 51,865 BPE tokens | 32,768 SentencePiece tokens |
| **Encoder layers (tiny)** | 4 | 6 |
| **Encoder layers (base)** | 6 | 12 |
| **Decoder layers (tiny)** | 4 | 6 |
| **Decoder layers (base)** | 6 | 6 |
| **Params (tiny)** | 39M | 27.1M |
| **Params (base)** | 74M | 61.5M |
| **Encoder-decoder arch** | Yes (cross-attention) | Yes (cross-attention) |
| **Autoregressive decoding** | Yes | Yes |

Both models are encoder-decoder transformers with cross-attention. The decoder
architecture is structurally identical: autoregressive text generation conditioned on
encoder output via cross-attention. The differences are confined to the encoder's
audio frontend, positional encoding, and transformer block internals.

---

## 3. Components Reused from whisper.apr

Moonshine's architecture overlaps heavily with the LFM2 components already
implemented in whisper.apr. The following modules carry over directly:

### 3.1 RoPE — `src/model/lfm2/rope.rs`

- `RopeConfig` struct with `head_dim`, `base`, `max_seq_len`
- `RotaryEmbedding` struct with precomputed sin/cos tables
- `forward()` and `forward_inplace()` methods
- Moonshine uses RoPE in both encoder self-attention and decoder self-attention
- Existing `lfm2_2_6b()` factory proves the implementation works; Moonshine needs
  its own factory (e.g., `moonshine_tiny()`) with appropriate `head_dim` and `base`

### 3.2 SwiGLU — `src/model/lfm2/swiglu.rs`

- `SwiGluConfig` struct with `hidden_size`, `intermediate_size`, `bias`
- `SwiGluFfn` struct with gate/up/down projections
- Formula: `output = (Swish(x @ W_gate) * (x @ W_up)) @ W_down`
- Moonshine tiny: hidden_size=288, intermediate_size ~= 768 (2.67x)
- Moonshine base: hidden_size=416, intermediate_size ~= 1110

### 3.3 GQA — `src/model/lfm2/gqa.rs`

- `GqaConfig` struct with `num_q_heads`, `num_kv_heads`, `head_dim`
- `GroupedQueryAttention` struct with Q/K/V/O projections
- `forward_with_rope()` method integrates RoPE directly
- Built-in `KvCache` struct with `append()` and `reset()` for incremental decoding
- Moonshine tiny: 8 Q heads, 2 KV heads (4:1 ratio), head_dim=36
- Moonshine base: 8 Q heads, 2 KV heads (4:1 ratio), head_dim=52

### 3.4 KV Cache — `src/model/decoder_generated.rs`

- `LayerKVCache` with `append()`, `clear()`, `reset()`, `remaining_capacity()`
- `LayerKVCacheTransposed` for optimized value access patterns
- Both variants support dynamic sizing — critical for Moonshine's variable-length
  encoder output (cross-attention KV cache sized to actual encoder frames, not
  a fixed 1500)

### 3.5 APR Format — `src/format/apr2_generated.rs`

- Same binary container (magic `APR1`, LZ4 block compression)
- Same `Apr2Quantization` enum (F32, F16, Int8, Int4, etc.)
- Same `QuantConfig` for per-tensor quantization parameters
- Only difference: weight tensor names and `ModelFamily` discriminant

### 3.6 Inference Strategies

- Greedy decoding (`src/inference/greedy.rs`) — works unchanged
- Beam search (`src/inference/beam.rs`) — works unchanged
- Both strategies operate on decoder logits, which are model-agnostic

### 3.7 Cross-Attention Decoder

- Same encoder-output-conditioned autoregressive generation
- Decoder cross-attention keys/values come from encoder output
- Token embedding + positional encoding + masked self-attention + cross-attention + FFN
- The decoder structure is identical; only internal block components differ
  (SwiGLU vs GELU, GQA vs MHA)

---

## 4. Components Moonshine Replaces

### 4.1 Mel Filterbank -> Learned Conv Stem

**Current (Whisper):** `src/audio/mel/mod.rs`
- `MelFilterbank` struct: 80-mel filterbank with FFT, Hann window, triangular filters
- Input: raw audio samples (f32)
- Output: 80-channel mel spectrogram, fixed 1500 frames (30s at 16kHz)
- Pipeline: audio -> STFT (n_fft=400, hop=160) -> mel filter -> log scale

**Replacement (Moonshine):** Learned convolutional stem
- 3 Conv1d layers applied directly to raw audio waveform
- Layer 1: in=1, out=C, kernel=441, stride=441 (captures ~27.6ms per frame at 16kHz)
- Layer 2: in=C, out=C, kernel=7, stride=4
- Layer 3: in=C, out=D, kernel=7, stride=2
- Total stride: 441 x 4 x 2 = 3528 samples per output frame (~220ms)
- No FFT, no Hann window, no mel triangular filters
- Output length is `ceil(audio_samples / 3528)`, proportional to input duration
- Weights are learned during training (stored in APR file)

The `MelFilterbank` is bypassed entirely for Moonshine models. The `ConvFrontend`
in `src/model/encoder/conv.rs` (Whisper's 2-layer conv after mel) is also bypassed
since Moonshine's learned conv stem subsumes both audio feature extraction and the
initial projection.

### 4.2 Sinusoidal Positional Encoding -> RoPE

**Current (Whisper):** `src/model/encoder/mod.rs` lines 78-100
- `create_positional_embedding(max_len, d_model)` generates fixed sinusoidal PE
- Formula: `pe[pos, 2i] = sin(pos / 10000^(2i/d))`, `pe[pos, 2i+1] = cos(...)`
- Fixed to `max_len=1500` (30s / 20ms per frame)
- Added to encoder input before transformer blocks

**Replacement (Moonshine):** RoPE (reuse from `src/model/lfm2/rope.rs`)
- Applied within each attention layer, not as an additive embedding
- Variable length: no hard max (limited only by `max_seq_len` in `RopeConfig`)
- Relative positional encoding enables length generalization

### 4.3 Encoder Transformer Blocks

**Current (Whisper):**
- Standard MHA with `n_audio_head` heads (6 for tiny, 8 for base)
- GELU FFN with 4x expansion
- Pre-LayerNorm

**Replacement (Moonshine):**
- GQA with 8 Q heads, 2 KV heads (reuse `GroupedQueryAttention`)
- SwiGLU FFN with ~2.67x expansion (reuse `SwiGluFfn`)
- Pre-RMSNorm (standard for RoPE-based architectures)

### 4.4 Tokenizer Vocabulary

**Current (Whisper):** 51,865 BPE tokens (`MULTILINGUAL_VOCAB_THRESHOLD` in `src/tokenizer/vocab/mod.rs`)
- Byte-level BPE with special tokens for language/task/timestamps

**Replacement (Moonshine):** 32,768 SentencePiece tokens
- Unigram model (not BPE)
- Smaller vocabulary reduces embedding matrix size
- English-only (no multilingual special tokens)
- Requires loading a SentencePiece model file or equivalent vocabulary mapping

---

## 5. ModelFamily Extension

Add `Moonshine = 3` to the `ModelFamily` enum in `src/format/apr2_generated.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ModelFamily {
    /// OpenAI Whisper (ASR)
    Whisper = 0,
    /// LiquidAI LFM2 (LLM for transcript summarization)
    Lfm2 = 1,
    /// Meta Llama-style architecture
    Llama = 2,
    /// Useful Sensors Moonshine (ASR, variable-length)
    Moonshine = 3,  // NEW
    /// Generic transformer
    Generic = 255,
}
```

The `TryFrom<u8>` and `Display` implementations must be extended to handle `3 => Moonshine`.

The `ModelFamily` enum in `src/model/download.rs` must also be extended:

```rust
pub enum ModelFamily {
    /// Whisper ASR models
    Whisper,
    /// LFM2 transcript summarization
    Lfm2,
    /// Moonshine ASR models
    Moonshine,
}
```

---

## 6. ModelConfig for Moonshine

### 6.1 New Configuration Enums

Introduce discriminant enums to make `ModelConfig` model-family-aware. These
complement the existing `FfnActivation` enum in `src/format/apr2_generated.rs`:

```rust
/// Audio frontend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFrontend {
    /// 80-mel filterbank (Whisper)
    MelFilterbank,
    /// Learned convolutional stem (Moonshine)
    LearnedConv,
}

/// Positional encoding type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionalEncoding {
    /// Fixed sinusoidal (Whisper)
    Sinusoidal,
    /// Rotary position embedding (Moonshine, LFM2, Llama)
    Rotary,
}

/// Attention mechanism type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    /// Standard multi-head attention (Whisper)
    Mha,
    /// Grouped query attention (Moonshine, LFM2, Llama)
    Gqa { kv_heads: u32 },
}
```

The existing `FfnActivation` enum already covers FFN types (`Gelu` for Whisper,
`Swiglu` for Moonshine).

### 6.2 Extended ModelConfig

Add optional fields to `ModelConfig` (in `src/model/mod.rs`) with defaults that
preserve backward compatibility for Whisper:

```rust
pub struct ModelConfig {
    // --- Existing fields (unchanged) ---
    pub model_type: ModelType,
    pub n_vocab: u32,
    pub n_audio_ctx: u32,
    pub n_audio_state: u32,
    pub n_audio_head: u32,
    pub n_audio_layer: u32,
    pub n_text_ctx: u32,
    pub n_text_state: u32,
    pub n_text_head: u32,
    pub n_text_layer: u32,
    pub n_mels: u32,

    // --- New fields for multi-model support ---
    pub audio_frontend: AudioFrontend,
    pub positional_encoding: PositionalEncoding,
    pub ffn_activation: FfnActivation,
    pub attention_type: AttentionType,
    pub model_family: ModelFamily,
}
```

Existing `tiny()`, `base()`, and `small()` factory methods set defaults:
`audio_frontend: MelFilterbank`, `positional_encoding: Sinusoidal`,
`ffn_activation: Gelu`, `attention_type: Mha`, `model_family: Whisper`.

### 6.3 Moonshine Factory Methods

```rust
impl ModelConfig {
    pub const fn moonshine_tiny() -> Self {
        Self {
            model_type: ModelType::Tiny,
            n_vocab: 32768,
            n_audio_ctx: 0,        // variable length (no fixed max)
            n_audio_state: 288,
            n_audio_head: 8,
            n_audio_layer: 6,
            n_text_ctx: 448,
            n_text_state: 288,
            n_text_head: 8,
            n_text_layer: 6,
            n_mels: 0,            // no mel filterbank
            audio_frontend: AudioFrontend::LearnedConv,
            positional_encoding: PositionalEncoding::Rotary,
            ffn_activation: FfnActivation::Swiglu,
            attention_type: AttentionType::Gqa { kv_heads: 2 },
            model_family: ModelFamily::Moonshine,
        }
    }

    pub const fn moonshine_base() -> Self {
        Self {
            model_type: ModelType::Base,
            n_vocab: 32768,
            n_audio_ctx: 0,
            n_audio_state: 416,
            n_audio_head: 8,
            n_audio_layer: 12,
            n_text_ctx: 448,
            n_text_state: 416,
            n_text_head: 8,
            n_text_layer: 6,
            n_mels: 0,
            audio_frontend: AudioFrontend::LearnedConv,
            positional_encoding: PositionalEncoding::Rotary,
            ffn_activation: FfnActivation::Swiglu,
            attention_type: AttentionType::Gqa { kv_heads: 2 },
            model_family: ModelFamily::Moonshine,
        }
    }
}
```

---

## 7. Dual-Model Runtime

### 7.1 Model Loading

`WhisperApr::load_from_apr()` reads the APR file header which contains the
`ModelFamily` discriminant byte. Dispatch:

```
APR header -> ModelFamily byte
  0 (Whisper)   -> ModelConfig::tiny() / base() / small() based on dims
  3 (Moonshine) -> ModelConfig::moonshine_tiny() / moonshine_base() based on dims
```

The `WhisperApr` struct (in `src/core_generated.rs`) gains a new field:

```rust
pub struct WhisperApr {
    config: model::ModelConfig,
    encoder: model::Encoder,         // dispatches internally based on config
    decoder: model::Decoder,         // dispatches internally based on config
    tokenizer: tokenizer::Tokenizer, // enum: BPE or SentencePiece
    mel_filters: Option<audio::MelFilterbank>,  // None for Moonshine
    conv_stem: Option<audio::ConvStem>,         // None for Whisper
    resampler: Option<audio::SincResampler>,
    weights_loaded: bool,
}
```

### 7.2 Transcription Dispatch

```
WhisperApr::transcribe(audio, options)
  |
  +-- resample to 16kHz (shared)
  |
  +-- audio frontend (dispatched by config.audio_frontend)
  |     MelFilterbank -> mel_filters.forward(audio) -> [batch, 80, T=1500]
  |     LearnedConv   -> conv_stem.forward(audio)   -> [batch, D, T=variable]
  |
  +-- encoder.forward(features) (internal blocks use config to select MHA/GQA, GELU/SwiGLU)
  |
  +-- decoder.generate(encoder_output, options) (shared autoregressive loop)
  |
  +-- tokenizer.decode(token_ids) (dispatched by tokenizer type)
```

### 7.3 Encoder Dispatch

The `Encoder` is constructed based on `ModelConfig` and internally uses:
- `PositionalEncoding::Sinusoidal` -> additive sinusoidal PE
- `PositionalEncoding::Rotary` -> RoPE applied in each attention layer
- `AttentionType::Mha` -> standard multi-head attention
- `AttentionType::Gqa { kv_heads }` -> grouped query attention (reuse from lfm2)
- `FfnActivation::Gelu` -> standard GELU FFN
- `FfnActivation::Swiglu` -> SwiGLU FFN (reuse from lfm2)

### 7.4 Decoder Sharing

The decoder follows the same dispatch pattern. Both Whisper and Moonshine decoders
are autoregressive with:
1. Token embedding lookup
2. Positional encoding (sinusoidal for Whisper, RoPE for Moonshine)
3. Masked self-attention (MHA or GQA)
4. Cross-attention to encoder output (MHA or GQA)
5. FFN (GELU or SwiGLU)
6. Linear projection to vocabulary logits

The inference loop (greedy/beam search) is identical.

---

## 8. Weight Conversion Pipeline

### 8.1 Source Format

Moonshine weights are published on HuggingFace in ONNX format:
- `usefulsensors/moonshine-tiny` (ONNX, ~108MB fp32)
- `usefulsensors/moonshine-base` (ONNX, ~246MB fp32)

### 8.2 Conversion Tool

Extend the existing `whisper-convert` tool to handle Moonshine ONNX graphs:

```
whisper-convert moonshine \
  --input usefulsensors/moonshine-tiny \
  --output moonshine-tiny.apr \
  --quantize fp16
```

### 8.3 Weight Naming Convention

ONNX node names map to APR tensor names:

| ONNX Node Pattern | APR Tensor Name |
|-------------------|-----------------|
| `preprocess/conv1/weight` | `encoder.conv_stem.0.weight` |
| `preprocess/conv1/bias` | `encoder.conv_stem.0.bias` |
| `preprocess/conv2/weight` | `encoder.conv_stem.1.weight` |
| `preprocess/conv3/weight` | `encoder.conv_stem.2.weight` |
| `encode/layers.{i}/attn/q_proj/weight` | `encoder.blocks.{i}.attn.q.weight` |
| `encode/layers.{i}/attn/k_proj/weight` | `encoder.blocks.{i}.attn.k.weight` |
| `encode/layers.{i}/attn/v_proj/weight` | `encoder.blocks.{i}.attn.v.weight` |
| `encode/layers.{i}/attn/o_proj/weight` | `encoder.blocks.{i}.attn.o.weight` |
| `encode/layers.{i}/ffn/gate/weight` | `encoder.blocks.{i}.ffn.w_gate.weight` |
| `encode/layers.{i}/ffn/up/weight` | `encoder.blocks.{i}.ffn.w_up.weight` |
| `encode/layers.{i}/ffn/down/weight` | `encoder.blocks.{i}.ffn.w_down.weight` |
| `decode/layers.{i}/...` | `decoder.blocks.{i}/...` (same pattern) |
| `decode/token_embedding/weight` | `decoder.token_embedding.weight` |
| `decode/ln_final/weight` | `decoder.ln_final.weight` |

### 8.4 Quantization

| Format | moonshine-tiny size | moonshine-base size |
|--------|--------------------|--------------------|
| F32 | ~108MB | ~246MB |
| F16 (default) | ~54MB | ~123MB |
| Int8 | ~27MB | ~62MB |
| Int4 | ~14MB | ~31MB |

F16 is the default for WASM delivery. Int8/Int4 for memory-constrained environments.

---

## 9. Variable-Length Input Design

This is Moonshine's primary performance advantage over Whisper.

### 9.1 The Problem with Whisper's Fixed Input

Whisper always processes exactly 480,000 samples (30 seconds at 16kHz):
- 1.5s audio -> 24,000 samples + 456,000 zeros = 480,000 total
- Mel spectrogram: always 1500 frames regardless of content
- Encoder processes all 1500 frames with full attention (O(n^2))
- ~95% of compute is wasted on padding for a 1.5s utterance

### 9.2 Moonshine's Variable-Length Solution

Moonshine's learned conv stem produces output proportional to input:

```
encoder_frames = ceil(audio_samples / total_stride)
total_stride   = 441 * 4 * 2 = 3528 samples per frame
frame_duration = 3528 / 16000 = 0.22 seconds per frame
```

Examples:

| Audio Duration | Samples (16kHz) | Moonshine Frames | Whisper Frames | Ratio |
|----------------|-----------------|------------------|----------------|-------|
| 1.5s | 24,000 | 7 | 1500 | 214x fewer |
| 3.0s | 48,000 | 14 | 1500 | 107x fewer |
| 5.0s | 80,000 | 23 | 1500 | 65x fewer |
| 10.0s | 160,000 | 46 | 1500 | 33x fewer |
| 30.0s | 480,000 | 136 | 1500 | 11x fewer |

Self-attention cost is O(n^2) in sequence length. For a 1.5s utterance:
- Whisper: O(1500^2) = 2,250,000 attention elements per layer
- Moonshine: O(7^2) = 49 attention elements per layer
- **45,918x fewer attention computations**

### 9.3 Implementation Impact

- `Encoder::forward()` no longer assumes fixed `seq_len=1500`
- Cross-attention KV cache in decoder sized to actual encoder output length
- `TranscribeOptions` and `TranscriptionResult` are unchanged (audio duration
  metadata already tracked)
- Streaming/chunked processing: chunks can be variable-length (no need to
  accumulate 30s before processing)

### 9.4 WASM Memory Implications

Whisper tiny allocates attention buffers for 1500 frames regardless of input:
- Per-layer self-attention: 1500 x 1500 x sizeof(f32) = 9MB
- 4 layers: 36MB just for attention matrices

Moonshine tiny for a 3s utterance (14 frames):
- Per-layer self-attention: 14 x 14 x sizeof(f32) = 784 bytes
- 6 layers: 4.7KB for attention matrices

This dramatically reduces peak WASM memory for short utterances.

---

## 10. Default Model Behavior

### 10.1 CLI Defaults

```bash
# Default: Moonshine tiny (downloads on first use)
whisper-apr transcribe audio.wav

# Explicit Whisper
whisper-apr transcribe --model whisper-tiny audio.wav
whisper-apr transcribe --model whisper-base audio.wav

# Explicit Moonshine
whisper-apr transcribe --model moonshine-tiny audio.wav
whisper-apr transcribe --model moonshine-base audio.wav

# Auto-detect from .apr file
whisper-apr transcribe --model-path /path/to/model.apr audio.wav
```

### 10.2 Model Registry Update

Add Moonshine entries to `MODELS` in `src/model/download.rs`:

```rust
pub const MODELS: &[ModelInfo] = &[
    // Moonshine models (default family)
    ModelInfo {
        name: "moonshine-tiny",
        repo_id: "usefulsensors/moonshine-tiny",
        family: ModelFamily::Moonshine,
        params: "27.1M",
        description: "Default. Fast, variable-length, efficient for short audio",
        wasm_quant: "fp16",
        size_fp16: "54MB",
        size_int4: "14MB",
    },
    ModelInfo {
        name: "moonshine-base",
        repo_id: "usefulsensors/moonshine-base",
        family: ModelFamily::Moonshine,
        params: "61.5M",
        description: "Higher accuracy, variable-length",
        wasm_quant: "fp16",
        size_fp16: "123MB",
        size_int4: "31MB",
    },
    // Whisper models (alternative family)
    ModelInfo {
        name: "whisper-tiny",
        repo_id: "openai/whisper-tiny",
        family: ModelFamily::Whisper,
        params: "39M",
        description: "Whisper: fixed 30s input, multilingual",
        wasm_quant: "fp16",
        size_fp16: "78MB",
        size_int4: "20MB",
    },
    // ... existing Whisper entries ...
];
```

### 10.3 Default Resolution

When no `--model` flag is provided:
1. Check if a model is already downloaded in the cache directory
2. If multiple cached models exist, prefer Moonshine over Whisper, tiny over base
3. If no model is cached, download `moonshine-tiny` (F16, ~54MB)

---

## 11. Performance Targets

| Model | Target RTF (WASM, short audio) | Target RTF (WASM, 30s) | Memory Peak | Download Size (F16) |
|-------|-------------------------------|------------------------|-------------|---------------------|
| moonshine-tiny | ≤1.5x | ≤3.0x | ≤100MB | ~54MB |
| moonshine-base | ≤2.0x | ≤4.0x | ≤200MB | ~123MB |
| whisper-tiny | ≤2.0x (constant) | ≤2.0x | ≤150MB | ~78MB |
| whisper-base | ≤2.5x (constant) | ≤2.5x | ≤350MB | ~148MB |

RTF = Real-Time Factor (processing time / audio duration). Lower is better.

Key insight: Moonshine RTF varies with audio length. For short audio (<5s),
Moonshine is significantly faster than Whisper. For 30s audio, Moonshine processes
more encoder frames (136 vs 1500 for Whisper after conv frontend downsampling)
but each frame is cheaper due to GQA/SwiGLU efficiency. Whisper's RTF is constant
regardless of content duration due to fixed padding.

---

## 12. Migration and Compatibility

### 12.1 No Breaking Changes

- **`TranscribeOptions`** (`src/core_generated.rs`): Unchanged. Fields `language`,
  `task`, `strategy`, `word_timestamps`, `profile` are model-agnostic.
- **`TranscriptionResult`** (`src/core_generated.rs`): Unchanged. Fields `text`,
  `language`, `segments`, `profiling` are model-agnostic.
- **`DecodingStrategy`** enum: `Greedy` and `BeamSearch` work for both families.
- **`Segment`** struct: `start`, `end`, `text` are model-agnostic.

### 12.2 Backward Compatibility

- Existing Whisper `.apr` files continue to load and run without modification
- `ModelFamily::Whisper = 0` discriminant is unchanged
- All existing tests pass without modification
- `WhisperApr` struct name is retained (it's the crate name, not model-specific)

### 12.3 Feature Flags

No new feature flags required. Both model families are always compiled in. The
runtime cost of unused code paths is zero (no Moonshine weights loaded = no
Moonshine compute). The `Conv1d` implementation in `src/model/encoder/conv.rs`
is already generic enough to support both Whisper's 2-layer conv frontend and
Moonshine's 3-layer learned stem.

### 12.4 API Surface

Public API changes are additive only:
- `ModelConfig::moonshine_tiny()` — new factory method
- `ModelConfig::moonshine_base()` — new factory method
- `AudioFrontend` enum — new type
- `PositionalEncoding` enum — new type
- `AttentionType` enum — new type
- `ModelFamily::Moonshine` — new variant

No existing public API is removed or modified.

---

## 13. SentencePiece Tokenizer

### 13.1 Vocabulary Differences

| Aspect | Whisper BPE | Moonshine SentencePiece |
|--------|-------------|------------------------|
| Vocab size | 51,865 | 32,768 |
| Algorithm | Byte-level BPE | Unigram |
| Languages | Multilingual (99 languages) | English only |
| Special tokens | Language, task, timestamps, translate | BOS, EOS, pad |
| Model file | Embedded in APR | Embedded in APR |

### 13.2 Tokenizer Abstraction

The current `BpeTokenizer` in `src/tokenizer/mod.rs` is Whisper-specific. A
tokenizer trait or enum is needed:

```rust
pub enum Tokenizer {
    /// Whisper BPE tokenizer (51,865 tokens)
    Bpe(BpeTokenizer),
    /// Moonshine SentencePiece tokenizer (32,768 tokens)
    SentencePiece(SentencePieceTokenizer),
}

impl Tokenizer {
    pub fn encode(&self, text: &str) -> WhisperResult<Vec<u32>> { ... }
    pub fn decode(&self, ids: &[u32]) -> WhisperResult<String> { ... }
    pub fn vocab_size(&self) -> usize { ... }
}
```

### 13.3 Special Token Mapping

| Function | Whisper Token | Moonshine Token |
|----------|---------------|-----------------|
| Start of transcript | `<\|startoftranscript\|>` (50258) | `<s>` (1) |
| End of transcript | `<\|endoftext\|>` (50257) | `</s>` (2) |
| Padding | N/A | `<pad>` (0) |
| No timestamps | `<\|notimestamps\|>` (50363) | N/A |

---

## 14. Learned Conv Stem Implementation

### 14.1 Architecture Detail

The Moonshine conv stem replaces both Whisper's mel filterbank AND the encoder's
`ConvFrontend`. Three 1D convolution layers:

```
Layer 1: Conv1d(in=1, out=C, kernel=441, stride=441, padding=0) + GELU
Layer 2: Conv1d(in=C, out=C, kernel=7, stride=4, padding=3) + GELU
Layer 3: Conv1d(in=C, out=D, kernel=7, stride=2, padding=3) + GELU
```

Where:
- C = intermediate channels (model-specific, ~64 for tiny)
- D = model dimension (288 for tiny, 416 for base)
- Total downsampling: 441 x 4 x 2 = 3528x

### 14.2 Code Location

New module: `src/audio/conv_stem.rs` (or extend existing `src/model/encoder/conv.rs`)

The existing `Conv1d` in `src/model/encoder/conv.rs` already supports configurable
`kernel_size`, `stride`, and `padding`. It can be reused directly for all three
layers. The `ConvStem` struct wraps three `Conv1d` instances:

```rust
pub struct ConvStem {
    conv1: Conv1d,  // raw audio -> intermediate (stride 441)
    conv2: Conv1d,  // intermediate -> intermediate (stride 4)
    conv3: Conv1d,  // intermediate -> d_model (stride 2)
}

impl ConvStem {
    pub fn forward(&self, audio: &[f32]) -> WhisperResult<Vec<f32>> {
        let x = self.conv1.forward(audio)?;
        let x = gelu_inplace(x);
        let x = self.conv2.forward(&x)?;
        let x = gelu_inplace(x);
        let x = self.conv3.forward(&x)?;
        gelu_inplace(x);
        Ok(x)
    }
}
```

---

## 15. Testing Strategy

### 15.1 Component Tests (reuse verification)

Verify that existing LFM2 components work with Moonshine-sized configs:
- `RopeConfig { head_dim: 36, base: 10000.0, max_seq_len: 2048 }` (moonshine-tiny)
- `GqaConfig { hidden_size: 288, num_q_heads: 8, num_kv_heads: 2, head_dim: 36 }`
- `SwiGluConfig { hidden_size: 288, intermediate_size: 768, bias: false }`

### 15.2 Conv Stem Tests

- Forward pass with known input produces expected output shape
- Stride calculation: `ceil(input_len / 3528)` output frames
- Short audio (0.1s = 1600 samples) produces at least 1 frame
- Empty audio returns error

### 15.3 End-to-End Parity

Same approach as `docs/specifications/whisper-apr-cpp-parity.md`:
- Compare Moonshine whisper.apr output against Moonshine ONNX reference
- Greedy decoding, same audio clips
- WER target: 0% on test-speech-1.5s.wav (exact token match)

### 15.4 Variable-Length Correctness

- Verify that transcription output is identical regardless of trailing silence padding
- 1.5s audio with 0s padding == 1.5s audio with 28.5s padding (Moonshine should
  produce same output; Whisper may differ slightly due to attention over padding)

---

## 16. File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `src/format/apr2_generated.rs` | MODIFY | Add `Moonshine = 3` to `ModelFamily` |
| `src/model/mod.rs` | MODIFY | Add `AudioFrontend`, `PositionalEncoding`, `AttentionType` enums; extend `ModelConfig` with new fields and `moonshine_tiny()`/`moonshine_base()` factories |
| `src/model/download.rs` | MODIFY | Add `Moonshine` to `ModelFamily`, add registry entries |
| `src/model/encoder/mod.rs` | MODIFY | Dispatch based on `AudioFrontend` and `PositionalEncoding` |
| `src/model/encoder/conv.rs` | MODIFY | Add `ConvStem` struct (3-layer learned conv) |
| `src/model/decoder_generated.rs` | MODIFY | Dispatch based on attention/FFN type in decoder blocks |
| `src/core_generated.rs` | MODIFY | `WhisperApr` gains `conv_stem` field, `mel_filters` becomes `Option` |
| `src/tokenizer/mod.rs` | MODIFY | Add `Tokenizer` enum wrapping `BpeTokenizer` and `SentencePieceTokenizer` |
| `src/tokenizer/sentencepiece.rs` | CREATE | SentencePiece tokenizer for Moonshine's 32K vocab |
| `src/audio/conv_stem.rs` | CREATE | Learned conv stem (wraps 3x Conv1d) |
| `tools/whisper-convert/` | MODIFY | Add Moonshine ONNX -> APR conversion path |
