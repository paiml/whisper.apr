# Real-Time Factor Analysis

Real-Time Factor (RTF) measures how fast transcription runs relative to audio duration.

## Definition

```
RTF = processing_time / audio_duration
```

- **RTF < 1.0x**: Faster than real-time (streaming capable)
- **RTF = 1.0x**: Real-time (keeps up with live audio)
- **RTF > 1.0x**: Slower than real-time (batch processing only)

## Achieved Results

### whisper-tiny Q4K (Release Mode)

| Metric | Value |
|--------|-------|
| **RTF** | **0.47x** |
| tokens/sec | 21.20 |
| ms/token | 47.17ms |

**Sub-real-time transcription achieved!** Audio is transcribed 2.1x faster than it plays.

### Debug vs Release Mode

| Mode | RTF | ms/token | Speedup |
|------|-----|----------|---------|
| Debug | 4.18x | 418.00ms | baseline |
| Release | 0.47x | 47.17ms | **8.9x** |

Release mode optimization is critical for performance.

## Component Breakdown

Where time is spent during decoding:

```
Decoder Component Profiling:
├── Token embedding:     1% (lookup only)
├── Position embedding:  1% (lookup only)
├── Self-attention:     28% (Q4K projections + softmax)
├── Cross-attention:    28% (Q4K projections + softmax)
├── FFN:                32% (two Q4K matmuls + GELU) ← BOTTLENECK
├── LayerNorm:           4% (SIMD-accelerated)
└── VocabProjection:     6% (final logits over 51865 vocab)
```

The Feed-Forward Network (FFN) is the primary bottleneck at 32% of decode time, which is typical for transformer architectures.

## Decoder Latency by Audio Length

| Audio Duration | Tokens | Latency | Status |
|----------------|--------|---------|--------|
| 1.5s | ~15 | 707ms | ✅ Interactive |
| 3.0s | ~30 | 1,415ms | ✅ Acceptable |
| 30s | ~300 | 14,151ms | Expected |

For short audio chunks (1.5s), latency is under 1 second, enabling responsive streaming applications.

## Amdahl's Law Analysis

Since the decoder dominates runtime (~80%), optimization focus must be there:

```
Maximum Speedup = 1 / ((1-P) + P/S)
Where P = 0.80 (decoder fraction), S = speedup factor

With 2x decoder speedup: Max total = 1.67x
With 4x decoder speedup: Max total = 2.50x
```

## Optimization Stack

The 0.47x RTF was achieved through:

1. **Q4K Quantization**: 86% weight reduction, 4.5-bit precision
2. **SIMD Acceleration**: 2.0-3.15x speedup via trueno
3. **Flash Attention**: O(n) memory for long sequences
4. **Fused Operations**: LayerNorm + Linear fusion
5. **Release Build**: 8.9x faster than debug

## Programmatic Benchmark API

```rust
use whisper_apr::benchmark::{
    generate_whisper_tiny_summary,
    RtfBenchmarkConfig,
    run_rtf_benchmark,
};

// Get pre-configured validation
let summary = generate_whisper_tiny_summary();
assert!(summary.all_targets_met());
println!("Targets met: {}/{}", summary.targets_met_count().0, summary.targets_met_count().1);

// Run custom benchmark
let config = RtfBenchmarkConfig::whisper_tiny(5.0); // 5-second audio
let result = run_rtf_benchmark(&config);
println!("RTF: {:.2}x", result.rtf);
```
