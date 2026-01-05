# Measurable Thresholds

This document defines all falsifiable performance and quality thresholds for whisper.apr.
Each threshold is measurable, reproducible, and has explicit pass/fail criteria.

## Performance Thresholds

### Real-Time Factor (RTF)

| Model | Threshold | Measurement | Command |
|-------|-----------|-------------|---------|
| tiny | RTF < 2.0x | Processing time / Audio duration | `cargo bench --bench inference -- rtf_tiny` |
| base | RTF < 2.5x | Processing time / Audio duration | `cargo bench --bench inference -- rtf_base` |
| small | RTF < 4.0x | Processing time / Audio duration | `cargo bench --bench inference -- rtf_small` |

**Pass Criteria**: RTF must be below threshold for 95% of test runs (n≥10).

### Memory Usage

| Model | Peak RSS | Measurement | Command |
|-------|----------|-------------|---------|
| tiny | < 150 MB | Peak resident set size | `heaptrack cargo run --release` |
| base | < 350 MB | Peak resident set size | `heaptrack cargo run --release` |
| small | < 800 MB | Peak resident set size | `heaptrack cargo run --release` |

**Pass Criteria**: Peak RSS must not exceed threshold during transcription of 30-second audio.

### SIMD Acceleration

| Metric | Threshold | Measurement | Command |
|--------|-----------|-------------|---------|
| Speedup vs scalar | > 2.0x | SIMD time / Scalar time | `cargo bench --bench inference -- attention` |
| MatMul speedup | > 3.0x | SIMD matmul / Scalar matmul | `cargo bench --bench wasm_simd` |

**Pass Criteria**: SIMD implementation must be at least 2x faster than scalar baseline.

### Latency

| Metric | Threshold | Measurement | Command |
|--------|-----------|-------------|---------|
| First token | < 500 ms | Time to first decoder output | `cargo bench --bench inference -- first_token` |
| Token throughput | > 20 tok/s | Tokens generated per second | `cargo bench --bench inference -- throughput` |
| Model load (tiny) | < 500 ms | Time to load model from disk | `cargo bench --bench inference -- load` |

**Pass Criteria**: 95th percentile latency must be below threshold.

## Quality Thresholds

### Test Coverage

| Metric | Threshold | Tool | Command |
|--------|-----------|------|---------|
| Line coverage | ≥ 95% | cargo-llvm-cov | `make coverage` |
| Branch coverage | ≥ 80% | cargo-llvm-cov | `make coverage` |
| Mutation score | ≥ 85% | cargo-mutants | `cargo mutants` |

**Pass Criteria**: Coverage must meet or exceed threshold for merge to main.

### Code Quality

| Metric | Threshold | Tool | Command |
|--------|-----------|------|---------|
| TDG Grade | ≥ A (90+) | pmat tdg | `pmat tdg .` |
| Clippy warnings | 0 | cargo clippy | `cargo clippy -- -D warnings` |
| Cyclomatic complexity | ≤ 10/function | pmat analyze | `pmat analyze complexity` |
| SATD comments | 0 | pmat analyze | `pmat analyze satd` |

**Pass Criteria**: All quality gates must pass before commit.

### Accuracy

| Metric | Threshold | Measurement | Command |
|--------|-----------|-------------|---------|
| WER (LibriSpeech) | < 10% (tiny) | Word Error Rate | `cargo test ground_truth` |
| WER vs whisper.cpp | < 1% difference | Parity comparison | `cargo run -- parity` |
| Hallucination rate | < 1% | Manual review | N/A |

**Pass Criteria**: WER must not exceed threshold on standard test set.

## Quantization Thresholds

| Format | Size Reduction | Quality Loss | Command |
|--------|---------------|--------------|---------|
| Q8_0 | > 50% | < 1% WER increase | `cargo test quantization` |
| Q5_K | > 70% | < 2% WER increase | `cargo test quantization` |
| Q4_K | > 80% | < 5% WER increase | `cargo test quantization` |

**Pass Criteria**: Size reduction must be achieved without exceeding quality loss threshold.

## Statistical Requirements

### Benchmark Rigor

- **Minimum iterations**: 10 per benchmark
- **Warm-up iterations**: 3 (excluded from statistics)
- **Confidence level**: 95%
- **Outlier handling**: Report but don't exclude

### Reporting Format

All benchmark results must include:
- Mean ± standard deviation
- Min/max values
- 95% confidence interval
- Sample size (n)
- Hardware specification

Example:
```
RTF (tiny): 0.47 ± 0.03 [0.42, 0.53] (n=10, 95% CI)
Hardware: AMD Ryzen 9 5900X, 64GB RAM
```

## Falsification Protocol

To falsify any threshold claim:

1. **Setup**: Clone repository, run `nix develop` or follow CONTRIBUTING.md
2. **Execute**: Run the specified command from the threshold table
3. **Verify**: Compare result against threshold
4. **Report**: If threshold is not met, open GitHub issue with:
   - Full command output
   - Hardware specification (`lscpu`, `free -h`)
   - Software versions (`rustc --version`, `cargo --version`)
   - Environment details (`uname -a`)

## Threshold History

| Date | Change | Justification |
|------|--------|---------------|
| 2024-12-01 | Initial thresholds | Based on whisper.cpp baseline |
| 2025-01-05 | Added quantization thresholds | Q4K/Q5K/Q6K support added |
| 2026-01-05 | Updated RTF targets | Performance improvements achieved |

## Related Documents

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) - Full reproducibility guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development setup and workflow
- [benches/README.md](benches/README.md) - Benchmark documentation
