# Ground Truth Validation Specification

**WAPR-SPEC-GT-001: Whisper.apr vs whisper.cpp vs HuggingFace**

| Field | Value |
|-------|-------|
| Status | FIXED - Root Causes Identified (EOT Token + H35 Attention Masking) |
| Author | Claude Code |
| Created | 2025-12-16 |
| Updated | 2025-12-20 |
| Toyota Way Phase | Kaizen (改善) - Continuous Improvement |
| Pipeline Position | 25/25 (100%) - Fixes Implemented |
| Upstream Issue | [RLZR-GEN-001](https://github.com/paiml/realizar/issues/23) |

---

## Executive Summary

This specification defines the systematic approach to validate whisper.apr transcription accuracy against two independent ground truth implementations: whisper.cpp (C++) and OpenAI Whisper via HuggingFace (Python). The current state shows **critical hallucination bugs** requiring root cause analysis.

### Current State (Observed)

| Implementation | Output | RTF | Status |
|----------------|--------|-----|--------|
| whisper.cpp | "The birds can use." | 0.45x | ✓ Ground Truth |
| HuggingFace | "The birds can use" | 1.25x | ✓ Ground Truth |
| whisper.apr | "the other one of..." × 100 | 6.74x | ✗ HALLUCINATION |

---

## Root Cause Analysis Results (2025-12-20)

### Summary of Discovered Bugs

Two critical bugs were identified through systematic Popperian falsification:

| Bug ID | Description | Impact | Status |
|--------|-------------|--------|--------|
| **EOT-001** | EOT token ID off-by-one (50256 → 50257) | Infinite repetition loops | ✅ FIXED |
| **H35** | Positional Singularity - Attention to padding | Decoder attends to padding positions | ✅ FIXED |

### Bug 1: EOT Token Off-by-One (EOT-001)

**Discovery:** Whisper has two tokenizer variants with different EOT token IDs:

| Model Type | Vocabulary Size | EOT Token | SOT Token | LANG_BASE |
|------------|-----------------|-----------|-----------|-----------|
| English-only (tiny.en, base.en) | < 51865 | **50256** | 50257 | 50258 |
| Multilingual (tiny, base, small) | ≥ 51865 | **50257** | 50258 | 50259 |

**Root Cause:** The codebase hardcoded `EOT = 50256` (English-only value) but was using multilingual models (`whisper-tiny.apr` with vocab size 51865).

**Evidence from Debug Session:**
```
$ cargo run --example trace_decoder
Top 20 tokens after initial sequence:
1. token 50257 (EOT) = 12.4523   ← EOT is winning!
2. token   440 (BPE) = 11.2341   ← " The"
3. token   383 (BPE) = 10.9812   ← " the"
...

But code checked for EOT = 50256, so loop continued forever!
```

**Fix Applied:** Dynamic `SpecialTokens::for_vocab_size(n_vocab)` lookup in `src/tokenizer/vocab.rs`:

```rust
pub struct SpecialTokens {
    pub eot: u32,      // 50256 for English-only, 50257 for multilingual
    pub sot: u32,      // 50257 for English-only, 50258 for multilingual
    pub lang_base: u32, // 50258 for English-only, 50259 for multilingual
    // ...
}

impl SpecialTokens {
    pub fn for_vocab_size(n_vocab: usize) -> Self {
        if n_vocab >= 51865 {
            Self::multilingual()  // EOT = 50257
        } else {
            Self::english_only()  // EOT = 50256
        }
    }
}
```

### Bug 2: Positional Singularity - H35 Attention Masking

**Discovery:** Decoder cross-attention was attending to padding positions in the encoder output, causing positional embedding singularities at sequence boundaries.

**Hypothesis (H35):** End-of-sequence positional embeddings create attention attractors that distort the decoder's attention distribution.

**Evidence:**
```
Encoder output shape: [1, 1500, 384]
Audio length: ~47 frames (1.5s audio)
Padding positions: frames 48-1500 (all zeros + sinusoidal pos embed)

Cross-attention without masking:
  - Query sees 1500 key positions
  - Positions 48-1500 have same content (zeros) but unique positional encodings
  - Creates artificial "peaks" in attention at padding boundaries

Cross-attention with masking:
  - Mask sets padding positions to -inf before softmax
  - Query only attends to valid audio positions (0-47)
  - Attention correctly peaks at audio content boundaries
```

**Fix Applied:** Added `audio_encoder_len` parameter to mask padding positions:

```rust
// In decoder cross-attention
fn cross_attention(
    query: &[f32],
    encoder_output: &[f32],
    audio_encoder_len: usize,  // NEW: actual content length
) -> Vec<f32> {
    // ... compute attention scores ...

    // Mask padding positions
    for pos in audio_encoder_len..1500 {
        attention_scores[pos] = f32::NEG_INFINITY;
    }

    // Softmax only sees valid positions
    softmax(&mut attention_scores);
}
```

### Verification

After applying both fixes:

| Implementation | Output | Status |
|----------------|--------|--------|
| whisper.cpp | "The birds can use." | ✓ Ground Truth |
| HuggingFace | "The birds can use" | ✓ Ground Truth |
| whisper.apr | "The birds can use." | ✅ **MATCHES** |

---

## Toyota Way Framework

### 1. Genchi Genbutsu (現地現物) - Go and See

**Observation:** Direct comparison reveals whisper.apr produces repetitive hallucinations instead of correct transcription.

```
Expected: "The birds can use"
Actual:   "the other one of the other one of the other one of..."
```

**Verification Log (2025-12-16):**
Executed `whisper-apr-cli` with `whisper-tiny.apr` on `test-speech-1.5s.wav`.
- **Result:** Confirmed Hallucination.
- **Output:** "the other one of the other one of..." (repeated >20 times).
- **RTF:** 5.53x (Observed) vs 0.45x (Target).
- **Conclusion:** Critical defect confirmed. Hypothesis of EOT detection failure is supported.

### 2. Five Whys Analysis

| Level | Question | Answer |
|-------|----------|--------|
| Why 1 | Why does whisper.apr hallucinate? | Decoder loop doesn't terminate at EOT |
| Why 2 | Why doesn't decoder terminate? | EOT token probability never becomes highest |
| Why 3 | Why is EOT probability low? | Cross-attention weights may be incorrect |
| Why 4 | Why are attention weights wrong? | Possible encoder output or weight loading issue |
| Why 5 | Why might weights be wrong? | .apr format conversion or attention mask computation |

### 3. Jidoka (自働化) - Automation with Human Touch

Implement automated quality gates that stop the line when defects are detected:

```yaml
quality_gates:
  - name: "hallucination_detection"
    trigger: "repetitive pattern detected"
    action: "STOP - investigate root cause"

  - name: "ground_truth_mismatch"
    trigger: "WER > 50% vs reference"
    action: "STOP - compare with whisper.cpp"
```

### 4. Kaizen (改善) - Continuous Improvement

100-point checklist below implements systematic falsification testing.

### 5. Heijunka (平準化) - Level Loading

Prioritize fixes by impact:
1. **Critical:** Hallucination/EOT detection (blocks all usage)
2. **High:** Performance (6.74x RTF vs 0.45x target)
3. **Medium:** Memory usage optimization
4. **Low:** API ergonomics

---

## Falsification-Style 100-Point Checklist

The scientific method requires attempting to **falsify** hypotheses, not confirm them. Each checkpoint attempts to prove whisper.apr is broken.

### Section A: Model Loading & Format (Points 1-15)

| # | Falsification Test | Method | Pass Criteria |
|---|-------------------|--------|---------------|
| 1 | APR magic bytes incorrect | `hexdump -C models/*.apr \| head -1` | Must show `APR1` |
| 2 | Weight dimensions mismatch | Compare tensor shapes vs whisper.cpp | All shapes match |
| 3 | Quantization corrupts weights | Dequantize and compare L2 norm | < 1% deviation |
| 4 | LZ4 decompression lossy | Decompress and recompress, compare | Bit-identical |
| 5 | Embedding weights wrong | Compare first 10 embedding vectors | Cosine sim > 0.99 |
| 6 | Positional encoding wrong | Compare sinusoidal patterns | Exact match |
| 7 | Layer norm weights wrong | Compare gamma/beta per layer | L2 < 1e-5 |
| 8 | Attention weights wrong | Compare QKV projections | L2 < 1e-5 |
| 9 | MLP weights wrong | Compare fc1/fc2 per layer | L2 < 1e-5 |
| 10 | Output projection wrong | Compare final linear layer | L2 < 1e-5 |
| 11 | Vocab size mismatch | Count tokens in vocab | Must be 51865 |
| 12 | Special tokens wrong | Check SOT/EOT/LANG token IDs | ✅ FIXED - Dynamic lookup via `SpecialTokens::for_vocab_size()` |
| 12a | EOT token ID wrong for model type | `SpecialTokens::for_vocab_size(n_vocab).eot` | EOT=50257 (multilingual) or 50256 (english-only) |
| 12b | Tokenizer variant detection wrong | Check `n_vocab >= 51865` threshold | Multilingual if vocab ≥ 51865 |
| 12c | Initial tokens sequence wrong | Compare `[SOT, LANG, TRANSCRIBE, NO_TIMESTAMPS]` | Tokens match model variant |
| 13 | BPE merges wrong | Compare merge rules with tiktoken | Exact match |
| 14 | Model config wrong | Compare n_layers, n_heads, d_model | Must match |
| 15 | Byte order wrong | Check endianness in weight loading | Must be little-endian |

### Section B: Audio Preprocessing (Points 16-30)

| # | Falsification Test | Method | Pass Criteria |
|---|-------------------|--------|---------------|
| 16 | Sample rate wrong | Check resampling to 16kHz | Exact 16000 Hz |
| 17 | Mel filterbank wrong | Compare 80 mel bins vs librosa | L2 < 1e-4 |
| 18 | FFT size wrong | Must be 400 samples (25ms @ 16kHz) | Exact match |
| 19 | Hop length wrong | Must be 160 samples (10ms @ 16kHz) | Exact match |
| 20 | Window function wrong | Compare Hann window coefficients | Exact match |
| 21 | Mel frequency range wrong | 0-8000 Hz for Whisper | Exact range |
| 22 | Log mel scaling wrong | Compare log1p vs log10 | Must be log1p |
| 23 | Normalization wrong | Compare mean/std normalization | Per-channel match |
| 24 | Padding wrong | Compare zero-padding behavior | 30s max, centered |
| 25 | Channel handling wrong | Stereo to mono conversion | Mean of channels |
| 26 | Clipping handling wrong | Audio clipping detection | No saturation |
| 27 | DC offset wrong | Remove DC component | Zero mean |
| 28 | Pre-emphasis wrong | Check high-frequency boost | If applied, correct |
| 29 | Frame extraction wrong | Compare frame boundaries | Aligned to spec |
| 30 | Mel spectrogram shape wrong | Must be (80, 3000) for 30s | Exact shape |

### Section C: Encoder (Points 31-50)

| # | Falsification Test | Method | Pass Criteria |
|---|-------------------|--------|---------------|
| 31 | Conv1 output wrong | Compare first conv layer output | L2 < 1e-3 |
| 32 | Conv2 output wrong | Compare second conv layer output | L2 < 1e-3 |
| 33 | GELU activation wrong | Compare GELU vs ReLU | Must be GELU |
| 34 | Layer norm epsilon wrong | Must be 1e-5 | Exact value |
| 35 | Attention mask wrong | Check causal vs bidirectional | ✅ FIXED (H35) - Cross-attention now masks padding positions |
| 35a | Cross-attn padding mask missing | Verify `audio_encoder_len` passed to decoder | Padding positions masked to -inf |
| 35b | Positional singularity at boundary | Check attention entropy at seq boundary | No artificial peaks at padding positions |
| 36 | QKV projection wrong | Compare attention input projections | L2 < 1e-4 |
| 37 | Attention scaling wrong | Must be 1/sqrt(d_head) | Exact formula |
| 38 | Softmax numerics wrong | Check for overflow/underflow | Stable softmax |
| 39 | Multi-head concat wrong | Head dimension ordering | Correct reshape |
| 40 | Output projection wrong | Post-attention linear | L2 < 1e-4 |
| 41 | Residual connection wrong | x + attention(x) | Correct addition |
| 42 | FFN intermediate wrong | 4x expansion factor | Exact ratio |
| 43 | FFN activation wrong | Must be GELU | Not ReLU |
| 44 | FFN output wrong | Compare MLP output | L2 < 1e-4 |
| 45 | Layer ordering wrong | Pre-norm vs post-norm | Pre-norm (GPT-2) |
| 46 | Final layer norm wrong | Compare encoder output norm | L2 < 1e-5 |
| 47 | Encoder output shape wrong | (batch, seq, d_model) | Correct dims |
| 48 | Positional encoding wrong | Sinusoidal vs learned | Sinusoidal |
| 49 | Attention entropy wrong | Compare entropy distribution | Similar pattern |
| 50 | Encoder deterministic | Same input = same output | Bit-identical |

### Section D: Decoder (Points 51-75)

| # | Falsification Test | Method | Pass Criteria |
|---|-------------------|--------|---------------|
| 51 | Token embedding wrong | Compare token embeddings | L2 < 1e-5 |
| 52 | Positional embedding wrong | Compare learned positions | L2 < 1e-5 |
| 53 | Causal mask wrong | Future tokens must be masked | -inf for future |
| 54 | Self-attention wrong | Compare decoder self-attn | L2 < 1e-4 |
| 55 | Cross-attention wrong | Compare encoder-decoder attn | L2 < 1e-4 |
| 56 | Cross-attention keys wrong | Keys from encoder output | Correct source |
| 57 | Cross-attention values wrong | Values from encoder output | Correct source |
| 58 | KV cache wrong | Compare cached keys/values | Exact match |
| 59 | KV cache update wrong | Incremental update logic | Correct append |
| 60 | Layer norm wrong | Per-layer normalization | L2 < 1e-5 |
| 61 | FFN wrong | Compare decoder MLP | L2 < 1e-4 |
| 62 | Output logits wrong | Compare final logits | L2 < 1e-3 |
| 63 | Softmax temperature wrong | Must be configurable | Default 1.0 |
| 64 | Top-k sampling wrong | If used, correct implementation | k tokens only |
| 65 | Top-p sampling wrong | If used, correct nucleus | Cumulative prob |
| 66 | Greedy decoding wrong | argmax selection | Highest prob |
| 67 | Beam search wrong | If used, correct beam management | Width maintained |
| 68 | EOT detection wrong | **CRITICAL** - Stop at EOT | ✅ FIXED - EOT=50257 for multilingual, 50256 for English-only |
| 69 | SOT handling wrong | Start with correct token | ✅ FIXED - SOT=50258 for multilingual, 50257 for English-only |
| 70 | Language token wrong | Correct language prefix | ✅ FIXED - LANG_BASE=50259 for multilingual, 50258 for English-only |
| 71 | Task token wrong | Transcribe vs translate | Correct task |
| 72 | No-speech detection wrong | Silence handling | Correct threshold |
| 73 | Timestamp tokens wrong | If used, correct format | <\|0.00\|> style |
| 74 | Max tokens wrong | 448 token limit | Enforce limit |
| 75 | Repetition penalty wrong | If used, correct penalty | Configurable |

### Section E: Inference Pipeline (Points 76-90)

| # | Falsification Test | Method | Pass Criteria |
|---|-------------------|--------|---------------|
| 76 | Batch dimension wrong | Batch processing correct | Dim 0 is batch |
| 77 | Sequence length wrong | Correct padding/truncation | ≤ max_length |
| 78 | Memory leak | Check memory growth over time | No growth |
| 79 | Thread safety wrong | Concurrent inference | No race conditions |
| 80 | Determinism broken | Same input = same output | Reproducible |
| 81 | Streaming broken | Chunk-by-chunk processing | Correct boundaries |
| 82 | VAD integration wrong | Voice activity detection | Correct segments |
| 83 | Long audio wrong | > 30s audio handling | Correct chunking |
| 84 | Short audio wrong | < 1s audio handling | Correct padding |
| 85 | Silence handling wrong | All-silence input | No crash/hang |
| 86 | Noise handling wrong | High-noise input | Graceful degradation |
| 87 | Multi-language wrong | Non-English detection | Correct language |
| 88 | Unicode output wrong | Non-ASCII characters | Correct encoding |
| 89 | Punctuation wrong | Sentence boundaries | Reasonable punct |
| 90 | Capitalization wrong | Sentence starts | Correct casing |

### Section F: Performance & Quality (Points 91-100)

| # | Falsification Test | Method | Pass Criteria |
|---|-------------------|--------|---------------|
| 91 | RTF too high | Real-time factor | ≤ 2.0x for tiny |
| 92 | Memory too high | Peak memory usage | ≤ 150MB for tiny |
| 93 | SIMD not used | Check vectorization | trueno SIMD active |
| 94 | WER too high | Word error rate | ≤ 10% on LibriSpeech |
| 95 | CER too high | Character error rate | ≤ 5% on LibriSpeech |
| 96 | Latency too high | First token latency | ≤ 500ms |
| 97 | Throughput too low | Tokens per second | ≥ 50 tok/s |
| 98 | GPU not used | WebGPU acceleration | If available, used |
| 99 | WASM size too large | Binary size | ≤ 2MB core |
| 100 | Load time too slow | Model initialization | ≤ 2s for tiny |

---

## APR Format vs Custom Code Analysis

### Decision Framework

| Factor | APR Format (aprender) | Custom Code |
|--------|----------------------|-------------|
| **Maintenance** | Single codebase (aprender) | Duplicate logic |
| **Correctness** | Battle-tested across projects | Whisper-specific bugs |
| **Performance** | Optimized tensor ops | Custom optimizations |
| **Flexibility** | Standard format | Full control |
| **Debugging** | Shared tooling | Custom tooling |

### Recommendation

**Use APR format (aprender) for weight storage and loading.**

Rationale:
1. aprender is already a dependency
2. Tensor operations go through trueno (shared SIMD)
3. LZ4 compression is well-tested
4. Reduces maintenance burden
5. Enables cross-project tooling (e.g., model inspection)

**Keep custom code for:**
1. Mel spectrogram (audio-specific)
2. Tokenization (BPE-specific)
3. Inference loop (Whisper-specific)
4. Streaming (latency-sensitive)

### Migration Path

```
Phase 1: Verify current .apr files match whisper.cpp weights
Phase 2: Add aprender weight inspection tooling
Phase 3: Compare inference outputs layer-by-layer
Phase 4: Bisect to find divergence point
```

---

## LogitProcessor Architecture (RLZR-GEN-001)

### Root Cause Analysis

The hallucination bug stems from missing logit processing between model forward pass and token sampling:

```
CURRENT (Broken):
  decoder.forward() → [logits] → argmax → token
                                    ↑
                            NO PROCESSING!
                    (SOT token has highest logit, wins every time)

FIXED (With LogitProcessor):
  decoder.forward() → [logits] → LogitProcessors → Sampler → token
                                       ↓
                            ├── TokenSuppressor (suppress SOT)
                            ├── RepetitionPenalty
                            └── TemperatureScaler
```

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1: aprender (Storage)                                         │
├─────────────────────────────────────────────────────────────────────┤
│  • .apr file format (tensor serialization)                           │
│  • LZ4 compression in 64KB blocks                                    │
│  • Weight quantization (fp32, fp16, int8)                            │
│  • Model metadata and versioning                                     │
│  → Pure storage, NO inference logic                                  │
└──────────────────────────────────────┬──────────────────────────────┘
                                       │ weights
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 2: realizar (Inference Runtime)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  pub trait LogitProcessor: Send + Sync {                            │
│      fn process(&self, logits: &mut [f32], ctx: &GenerationContext);│
│  }                                                                   │
│                                                                      │
│  pub struct GenerationPipeline<M: Model> {                          │
│      model: M,                                                       │
│      processors: Vec<Box<dyn LogitProcessor>>,                      │
│      sampler: SamplingStrategy,                                      │
│      eos_token: Option<u32>,                                         │
│  }                                                                   │
│                                                                      │
│  Built-in processors:                                                │
│  ├── TokenSuppressor       - Suppress specific tokens                │
│  ├── RepetitionPenalty     - Penalize repeated n-grams               │
│  ├── TemperatureScaler     - Scale logits by temperature             │
│  ├── TopKFilter            - Keep only top-k logits                  │
│  └── TopPFilter            - Keep nucleus by cumulative prob         │
│                                                                      │
└──────────────────────────────────────┬──────────────────────────────┘
                                       │ trait Model
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 3: whisper.apr (Application)                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  impl realizar::Model for WhisperDecoder {                          │
│      fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>>;         │
│      fn vocab_size(&self) -> usize { 51865 }                        │
│  }                                                                   │
│                                                                      │
│  // Whisper-specific processors                                      │
│  pub struct WhisperTokenSuppressor;  // Suppress SOT, PREV, SOLM    │
│  pub struct TimestampConstrainer;    // Enforce timestamp grammar    │
│  pub struct LanguageForcer;          // Force language token         │
│                                                                      │
│  let pipeline = GenerationPipeline::new(decoder)                    │
│      .add_processor(WhisperTokenSuppressor::new())                  │
│      .add_processor(TimestampConstrainer::new())                    │
│      .with_eos_token(EOT)  // 50256                                 │
│      .with_sampler(SamplingStrategy::Greedy);                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Contract Definition

The APR serving contract between layers:

```rust
// realizar/src/generate.rs - Core traits

/// Context available during generation
pub struct GenerationContext<'a> {
    /// Previously generated tokens
    pub tokens: &'a [u32],
    /// Current generation step (0-indexed)
    pub step: usize,
    /// Vocabulary size
    pub n_vocab: usize,
    /// Model-specific metadata
    pub metadata: Option<&'a dyn std::any::Any>,
}

/// Logit processor - composable pre-sampling transform
pub trait LogitProcessor: Send + Sync {
    /// Process logits in-place before sampling
    ///
    /// Processors can:
    /// - Set logits to -inf to suppress tokens
    /// - Add penalties (repetition, length)
    /// - Scale logits (temperature)
    fn process(&self, logits: &mut [f32], ctx: &GenerationContext);

    /// Human-readable name for debugging
    fn name(&self) -> &str { "unnamed" }
}

/// Model trait for generation pipeline
pub trait GenerativeModel {
    /// Forward pass producing logits
    fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Vocabulary size
    fn vocab_size(&self) -> usize;

    /// Optional: reset KV cache
    fn reset_cache(&mut self) {}
}
```

### Why realizar (Not aprender)

| Concern | aprender | realizar | Decision |
|---------|----------|----------|----------|
| Tensor storage | ✓ Core competency | Uses for weights | aprender |
| Model format | ✓ .apr format | Consumes | aprender |
| Forward pass | Too low-level | ✓ Runtime | realizar |
| Logit processing | Not applicable | ✓ Inference | realizar |
| Sampling | Not applicable | ✓ Generation | realizar |
| Pipeline orchestration | Not applicable | ✓ Runtime | realizar |

**aprender** = "to learn" → Training, model storage
**realizar** = "to accomplish" → Inference, generation

---

## Peer-Reviewed Citations

### Speech Recognition & Transformer Architecture (1-10)

1. **Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022).** "Robust Speech Recognition via Large-Scale Weak Supervision." *arXiv preprint arXiv:2212.04356*. [OpenAI Whisper architecture]

2. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., & Polosukhin, I. (2017).** "Attention Is All You Need." *Advances in Neural Information Processing Systems, 30*. [Transformer architecture foundation]

3. **Bahdanau, D., Cho, K., & Bengio, Y. (2014).** "Neural Machine Translation by Jointly Learning to Align and Translate." *arXiv preprint arXiv:1409.0473*. [Attention mechanism for sequence-to-sequence]

4. **Davis, S., & Mermelstein, P. (1980).** "Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences." *IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357-366*. [Mel-frequency cepstral coefficients]

5. **Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006).** "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks." *ICML 2006*. [CTC loss for ASR]

6. **Gulati, A., Qin, J., Chiu, C.C., Parmar, N., Zhang, Y., Yu, J., Han, W., Wang, S., Zhang, Z., Wu, Y., & Pang, R. (2020).** "Conformer: Convolution-augmented Transformer for Speech Recognition." *Interspeech 2020*. [Conformer architecture]

7. **Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D., & Le, Q.V. (2019).** "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition." *Interspeech 2019*. [Data augmentation for ASR]

8. **Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015).** "LibriSpeech: An ASR Corpus Based on Public Domain Audio Books." *ICASSP 2015*. [LibriSpeech benchmark dataset]

9. **Liker, J.K. (2004).** "The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer." *McGraw-Hill*. [Toyota Production System principles]

10. **Popper, K. (1959).** "The Logic of Scientific Discovery." *Hutchinson & Co*. [Falsificationism methodology]

### Text Generation & Logit Processing (11-15)

11. **Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020).** "The Curious Case of Neural Text Degeneration." *ICLR 2020*. [Nucleus (top-p) sampling - demonstrates that greedy/beam search leads to repetitive, degenerate text; proposes top-p sampling as solution. **Key insight: token suppression and probability shaping are essential for coherent generation.**]

12. **Keskar, N.S., McCann, B., Varshney, L.R., Xiong, C., & Socher, R. (2019).** "CTRL: A Conditional Transformer Language Model for Controllable Generation." *arXiv preprint arXiv:1909.05858*. [Controllable generation with token-level constraints - shows how to influence generation via logit manipulation. **Validates composable processor architecture.**]

13. **Su, Y., Lan, T., Wang, Y., Yogatama, D., Kong, L., & Collier, N. (2022).** "A Contrastive Framework for Neural Text Generation." *NeurIPS 2022*. [Contrastive decoding - applies penalties to reduce repetition. **Demonstrates that logit processors can prevent hallucination/repetition.**]

14. **Fan, A., Lewis, M., & Dauphin, Y. (2018).** "Hierarchical Neural Story Generation." *ACL 2018*. [Top-k sampling - shows that restricting sampling to top-k tokens improves coherence. **Foundation for TopKFilter processor.**]

15. **Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., & Rush, A.M. (2020).** "Transformers: State-of-the-Art Natural Language Processing." *EMNLP 2020 System Demonstrations*. [HuggingFace Transformers library - defines LogitsProcessor pattern adopted industry-wide. **Direct precedent for realizar's LogitProcessor trait.**]

---

## Testing Infrastructure

### Probar Integration

```yaml
# demos/playbooks/ground-truth-validation.yaml
test_matrix:
  audio_samples:
    - "test-speech-1.5s.wav"
    - "test-speech-3s.wav"
    - "test-speech-full.wav"

  implementations:
    - name: "whisper.apr"
      command: "cargo run --release --bin whisper-apr-cli"
    - name: "whisper.cpp"
      command: "/home/noah/.local/bin/main"
    - name: "huggingface"
      command: "uv run scripts/hf_transcribe.py"

  assertions:
    - type: "consensus"
      description: "whisper.cpp and HuggingFace must agree"
    - type: "match"
      description: "whisper.apr must match consensus"
    - type: "no_hallucination"
      pattern: "(.{5,})\\1{3,}"
```

### TUI Visualization

```bash
# Run TUI to observe pipeline state
cargo run --example tui_demo --features tui

# States to monitor:
# - Idle → WaveformReady → MelReady → Encoding → Decoding → Complete
# - Watch for: stuck in Decoding (hallucination indicator)
```

### Renacer Tracing

```bash
# Trace decoder loop iterations
renacer -s -- cargo test test_decoder_eot

# Expected spans:
# - decoder.step (should terminate < 448 iterations)
# - decoder.logits (check EOT probability)
# - decoder.sample (verify token selection)
```

---

## Linear Progress Scale (25-Step Pipeline)

### Methodology

Divide inference pipeline into 25 atomic steps. Each iteration:
1. Identify current step position for each implementation
2. Compare intermediate outputs at that step
3. If stuck (same step 3+ iterations), apply Five Whys

### Step Definitions

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUDIO PREPROCESSING (1-5)                      │
├─────────────────────────────────────────────────────────────────┤
│ Step 1:  Load WAV → raw samples (f32)                            │
│ Step 2:  Resample to 16kHz                                       │
│ Step 3:  Pad/truncate to 30s (480000 samples)                    │
│ Step 4:  Apply Hann window (400 samples, hop 160)                │
│ Step 5:  Compute FFT (201 bins)                                  │
├─────────────────────────────────────────────────────────────────┤
│                    MEL SPECTROGRAM (6-10)                         │
├─────────────────────────────────────────────────────────────────┤
│ Step 6:  Apply 80-mel filterbank                                 │
│ Step 7:  Log mel scaling (log1p)                                 │
│ Step 8:  Normalize (mean/std per channel)                        │
│ Step 9:  Reshape to (1, 80, 3000)                                │
│ Step 10: Verify mel checksum vs reference                        │
├─────────────────────────────────────────────────────────────────┤
│                    ENCODER (11-15)                                │
├─────────────────────────────────────────────────────────────────┤
│ Step 11: Conv1 + GELU (80→d_model)                               │
│ Step 12: Conv2 + GELU (d_model→d_model)                          │
│ Step 13: Add sinusoidal positional encoding                      │
│ Step 14: Encoder transformer layers (N layers)                   │
│ Step 15: Final layer norm → encoder output (1, 1500, d_model)    │
├─────────────────────────────────────────────────────────────────┤
│                    DECODER INIT (16-18)                           │
├─────────────────────────────────────────────────────────────────┤
│ Step 16: Token embedding lookup                                  │
│ Step 17: Add positional embedding                                │
│ Step 18: Initialize KV cache                                     │
├─────────────────────────────────────────────────────────────────┤
│                    DECODER LOOP (19-22)                           │
├─────────────────────────────────────────────────────────────────┤
│ Step 19: Self-attention (causal mask)                            │
│ Step 20: Cross-attention (attend to encoder)          ← SUSPECT  │
│ Step 21: FFN + residual                                          │
│ Step 22: Output projection → logits (51865)                      │
├─────────────────────────────────────────────────────────────────┤
│                    TOKEN GENERATION (23-25)                       │
├─────────────────────────────────────────────────────────────────┤
│ Step 23: Apply LogitProcessor (suppress SOT, etc.)               │
│ Step 24: Sample next token (greedy/beam)                         │
│ Step 25: Check EOT → terminate or loop to Step 19    ← SUSPECT   │
└─────────────────────────────────────────────────────────────────┘
```

### Current Position

| Implementation | Current Step | Status | Notes |
|----------------|--------------|--------|-------|
| whisper.cpp    | 25 ✓         | PASS   | Full pipeline works |
| HuggingFace    | 25 ✓         | PASS   | Full pipeline works |
| whisper.apr    | **25** ✓     | PASS   | ✅ Fixed: EOT token ID + H35 attention masking |

### Hypothesis Testing Results (2025-12-16)

Debug tooling created in `examples/debug_cross_attn.rs` to systematically test 5 hypotheses:

| Hypothesis | Description | Test Method | Result |
|------------|-------------|-------------|--------|
| H1 | Cross-attn K/V not connected to encoder | Different inputs → different outputs? | ✅ PASSED |
| H2 | Encoder output shape wrong | Check 1500 × d_model = 576000 | ✅ PASSED |
| H3 | Cross-attn weights never loaded | Check weight norms non-zero | ✅ PASSED |
| H4 | Attention scaling factor wrong | Verify 1/sqrt(d_head) | ✅ PASSED |
| H5 | KV cache overwrites encoder context | Code review | ✅ PASSED (code review) |

**Key Observation:** All structural hypotheses PASSED, yet model still produced repetitive garbage. This led to deeper investigation.

### Additional Hypothesis Testing (2025-12-20)

| Hypothesis | Description | Test Method | Result |
|------------|-------------|-------------|--------|
| H6 | Weight values incorrectly converted | Compare L2 norms with whisper.cpp | ✅ PASSED (weights correct) |
| **H35** | Positional singularity at padding | Mask cross-attention to valid frames | ✅ **ROOT CAUSE FOUND** |
| **EOT-001** | EOT token ID wrong for multilingual | Check vocab size → token ID mapping | ✅ **ROOT CAUSE FOUND** |

**Root Cause Discovery:**

Two bugs combined to cause infinite hallucination loops:

1. **EOT Token Off-by-One:** Code checked for `EOT = 50256` but multilingual models use `EOT = 50257`. The model correctly predicted EOT but the termination check failed.

2. **H35 Attention Masking:** Decoder cross-attention attended to padding positions (frames 48-1500 for 1.5s audio), where sinusoidal positional embeddings created artificial attention peaks at sequence boundaries.

**Evidence from Trace:**
```
Before fixes:
  - EOT (50257) has highest logit but code checks 50256
  - Attention entropy high due to padding positions
  - Loop continues, model enters repetition mode

After fixes:
  - EOT (50257) correctly triggers termination
  - Cross-attention masked to valid frames only
  - Clean transcription: "The birds can use."
```

### Model Size Elimination Test

| Model | whisper.cpp | HuggingFace | whisper.apr | Conclusion |
|-------|-------------|-------------|-------------|------------|
| tiny  | "The birds can use" | "The birds can use" | "the other one of..." ×100 | ✗ Hallucination |
| base  | "the birch canoes" | "The Birch can do" | ". The. The..." ×200 | ✗ Hallucination |

**Finding:** Both models hallucinate differently but with same root cause. Issue is NOT model-specific - confirms Step 20 (cross-attention) divergence.

### Divergence Analysis

**Step 20 (Cross-Attention) is the likely divergence point:**

```
whisper.cpp Step 20 output:
  - Query attends to encoder positions 0-1500
  - Attention weights peak at audio boundaries
  - EOT probability rises when audio ends

whisper.apr Step 20 output:
  - Query ignores encoder (flat attention?)
  - No correlation with audio content
  - EOT probability stays at rank ~50000
```

### Five Whys (Triggered: Stuck at Step 20)

| Level | Question | Answer |
|-------|----------|--------|
| Why 1 | Why is cross-attention output wrong? | Attention weights don't match reference |
| Why 2 | Why don't attention weights match? | Keys/Values from encoder may be wrong |
| Why 3 | Why might encoder output be wrong? | Weights may be incorrectly loaded |
| Why 4 | Why might weights be wrong? | APR→tensor mapping or quantization |
| Why 5 | Why check APR format first? | It's the untested interface between aprender and whisper.apr |

### Iteration Protocol

```bash
# Each debugging iteration:
1. cargo run --example debug_step_N  # Compare at step N
2. diff output_whisper_apr.json output_whisper_cpp.json
3. if divergence > threshold:
     record_divergence(step=N)
     five_whys(step=N)
   else:
     advance(step=N+1)
```

### Step Validation Commands

| Step | Validation Command | Pass Criteria |
|------|-------------------|---------------|
| 1-5  | `cargo run --example debug_audio` | Samples match ±1e-6 |
| 6-10 | `cargo run --example debug_mel` | Mel L2 < 1e-4 |
| 11-15| `cargo run --example debug_encoder` | Encoder L2 < 1e-3 |
| 16-18| `cargo run --example debug_decoder_init` | Embeddings match |
| 19-22| `cargo run --example debug_decoder_step` | Logits L2 < 1e-2 |
| 23-25| `cargo run --example debug_generation` | Same tokens |

---

## Implementation Plan

### Phase 1: Diagnosis (Week 1)

| Task | Owner | Validation |
|------|-------|------------|
| Run 100-point checklist | AI Engineering | Checklist complete |
| Identify failing tests | AI Engineering | List of failures |
| Root cause analysis | AI Engineering | Five Whys document |
| Weight comparison | AI Engineering | L2 norm report |

### Phase 2: Fix Critical Bugs (Week 2)

| Task | Owner | Validation |
|------|-------|------------|
| Fix EOT detection | AI Engineering | Probar test passes |
| Fix attention weights | AI Engineering | Layer comparison |
| Fix KV cache | AI Engineering | Determinism test |
| Verify on all samples | AI Engineering | 3-column match |

### Phase 3: Performance (Week 3)

| Task | Owner | Validation |
|------|-------|------------|
| Profile with renacer | AI Engineering | Trace analysis |
| Optimize hot paths | AI Engineering | RTF ≤ 2.0x |
| Memory optimization | AI Engineering | Peak ≤ 150MB |
| SIMD verification | AI Engineering | trueno active |

### Phase 4: Quality Gates (Week 4)

| Task | Owner | Validation |
|------|-------|------------|
| Add probar tests | AI Engineering | CI green |
| Add mutation tests | AI Engineering | 85% score |
| Add property tests | AI Engineering | 50 properties |
| Documentation | AI Engineering | Book updated |

---

## Approval

| Role | Name | Date | Status |
|------|------|------|--------|
| Author | Claude Code | 2025-12-16 | Complete |
| AI Engineering Lead | | | **PENDING** |
| Quality Assurance | | | **PENDING** |
| Technical Review | Gemini | 2025-12-16 | **APPROVED** |

---

## Appendix A: Reproduction Commands

```bash
# Ground truth comparison
./scripts/ground_truth_compare.sh

# Individual implementations
/home/noah/.local/bin/main -m /home/noah/src/whisper.cpp/models/ggml-tiny.bin \
  -f demos/test-audio/test-speech-1.5s.wav

uv run scripts/hf_transcribe.py demos/test-audio/test-speech-1.5s.wav

cargo run --release --bin whisper-apr-cli --features cli -- transcribe \
  --model-path models/whisper-tiny.apr \
  -v demos/test-audio/test-speech-1.5s.wav
```

## Appendix B: Expected Outputs

| Audio | Expected Transcription |
|-------|----------------------|
| test-speech-1.5s.wav | "The birds can use" |
| test-speech-3s.wav | TBD (run whisper.cpp) |
| test-speech-full.wav | TBD (run whisper.cpp) |

---

## Appendix C: Self-Diagnostic CLI Command

The `whisper-apr diagnose` (alias: `doctor`) command validates tokenizer configuration and checks for known issues.

### Usage

```bash
# Basic tokenizer check
whisper-apr diagnose

# Check with specific model
whisper-apr diagnose --model models/whisper-tiny.apr

# Full check including model validation
whisper-apr diagnose --model models/whisper-tiny.apr --full

# JSON output for scripting
whisper-apr diagnose --json

# Tokenizer-only (skip model and known issues sections)
whisper-apr diagnose --tokenizer-only
```

### Checks Performed

| Check ID | Name | Description |
|----------|------|-------------|
| TOK-001 | EOT token (multilingual) | Verifies EOT=50257 for vocab >= 51865 |
| TOK-002 | EOT token (English-only) | Verifies EOT=50256 for vocab < 51865 |
| TOK-003 | SOT token (multilingual) | Verifies SOT=50258 for multilingual |
| TOK-004 | LANG_BASE token | Verifies LANG_BASE=50259 for multilingual |
| TOK-005 | English language token | Verifies language_token("en") = 50259 |
| TOK-006 | Initial tokens sequence | Verifies [SOT, LANG, TRANSCRIBE, NO_TIMESTAMPS] |
| TOK-007 | TIMESTAMP_BASE | Verifies timestamp base = 50364 for multilingual |
| MDL-001 | Model file exists | Checks model file is present |
| MDL-002 | APR magic bytes | Verifies "APR1" header |

### Example Output

```
═══════════════════════════════════════════════════════════════════
                    whisper-apr Self-Diagnostic
═══════════════════════════════════════════════════════════════════

1. Tokenizer Configuration
───────────────────────────────────────────────────────────────────
  ✓ [TOK-001] EOT token (multilingual): EOT for multilingual models (expected: 50257, got: 50257)
  ✓ [TOK-002] EOT token (English-only): EOT for English-only models (expected: 50256, got: 50256)
  ✓ [TOK-003] SOT token (multilingual): SOT for multilingual models (expected: 50258, got: 50258)
  ...

3. Known Issues Status
───────────────────────────────────────────────────────────────────
  ✓ EOT-001: EOT token off-by-one - FIXED (2025-12-20)
  ✓ H35: Cross-attention padding mask - FIXED (2025-12-20)

═══════════════════════════════════════════════════════════════════
RESULT: 7/7 checks passed ✓
═══════════════════════════════════════════════════════════════════
```

---

## Appendix D: Complete Special Token Reference

### Token ID Mapping by Model Type

| Token | English-only (vocab < 51865) | Multilingual (vocab ≥ 51865) | Purpose |
|-------|------------------------------|------------------------------|---------|
| EOT | 50256 | 50257 | End of transcript |
| SOT | 50257 | 50258 | Start of transcript |
| LANG_BASE | 50258 | 50259 | Language token base (+ offset for language) |
| TRANSLATE | 50357 | 50358 | Translate task |
| TRANSCRIBE | 50358 | 50359 | Transcribe task |
| SPEAKER_TURN | 50359 | 50360 | Speaker diarization marker |
| PREV | 50360 | 50361 | Previous context token |
| NO_SPEECH | 50361 | 50362 | No speech detected |
| NO_TIMESTAMPS | 50362 | 50363 | Disable timestamp generation |
| TIMESTAMP_BASE | 50363 | 50364 | First timestamp token (0.00s) |

### Language Offsets (from LANG_BASE)

| Offset | Language | Code | Token (Multilingual) |
|--------|----------|------|---------------------|
| 0 | English | en | 50259 |
| 1 | Chinese | zh | 50260 |
| 2 | German | de | 50261 |
| 3 | Spanish | es | 50262 |
| 4 | Russian | ru | 50263 |
| 5 | Korean | ko | 50264 |
| 6 | French | fr | 50265 |
| 7 | Japanese | ja | 50266 |
| 8 | Portuguese | pt | 50267 |
| ... | ... | ... | ... |

### Initial Token Sequence

For multilingual transcription (English):
```
[50258, 50259, 50359, 50363]
   │      │      │      └── NO_TIMESTAMPS
   │      │      └── TRANSCRIBE
   │      └── LANG_EN (LANG_BASE + 0)
   └── SOT
```

For English-only transcription:
```
[50257, 50358, 50362]
   │      │      └── NO_TIMESTAMPS
   │      └── TRANSCRIBE
   └── SOT
```

### Timestamp Token Range

- Multilingual: 50364 to 51864 (1501 tokens, 0.00s to 30.00s in 0.02s increments)
- English-only: 50363 to 51863 (same range, offset by 1)

---

*This specification follows Toyota Way principles and Popperian falsificationism to systematically identify and resolve quality issues in whisper.apr.*
