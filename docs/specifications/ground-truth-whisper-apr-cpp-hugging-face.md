# Ground Truth Validation Specification

**WAPR-SPEC-GT-001: Whisper.apr vs whisper.cpp vs HuggingFace**

| Field | Value |
|-------|-------|
| Status | DRAFT - Awaiting AI Engineering Review |
| Author | Claude Code |
| Created | 2025-12-16 |
| Toyota Way Phase | Genchi Genbutsu (現地現物) - Go and See |

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
| 12 | Special tokens wrong | Check SOT/EOT/LANG token IDs | Must match spec |
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
| 35 | Attention mask wrong | Check causal vs bidirectional | Bidirectional |
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
| 68 | EOT detection wrong | **CRITICAL** - Stop at EOT | Token 50257 |
| 69 | SOT handling wrong | Start with correct token | Token 50258 |
| 70 | Language token wrong | Correct language prefix | e.g., 50259 for en |
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

## Peer-Reviewed Citations

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

*This specification follows Toyota Way principles and Popperian falsificationism to systematically identify and resolve quality issues in whisper.apr.*
