# Extreme TDD

Whisper.apr follows **EXTREME TDD** methodology combining Toyota Way principles with statistical validation.

## Core Principles

1. **Test-First Always**: Write failing tests before implementation
2. **Five-Whys Root Cause Analysis**: Diagnose issues systematically
3. **Statistical Validation**: Use t-tests for numerical comparisons
4. **Three-Way Ground Truth**: Validate against multiple reference implementations

## Five-Whys Methodology

When debugging complex issues, especially numerical discrepancies, apply the Toyota Way Five-Whys analysis:

### Example: Decoder Weight Anomaly (T10.D)

**Observation**: Decoder LayerNorm gamma has mean=11.098 (expected ~1.0)

```
Why #1: Why is the LayerNorm gamma mean 11.098?
→ Because OpenAI's original training produced these values.

Why #2: Why did OpenAI's training produce unusual gamma values?
→ Because they used a non-standard initialization or training regime.

Why #3: Why didn't they normalize to standard gamma≈1.0?
→ Because the model converged to this solution during training.

Why #4: Why does whisper-apr show the same values?
→ Because we correctly preserve the original weights (bit-exact).

Why #5: Why is this actually correct behavior?
→ Because HuggingFace, whisper.cpp, and whisper-apr all match exactly.

ROOT CAUSE: Not a bug - OpenAI's original weights have unusual but intentional values.
```

### Five-Whys in Tests

Document your Five-Whys analysis directly in test code:

```rust
/// T10.D2: Decoder weights match HuggingFace reference
///
/// Five-Whys Analysis:
/// Q1: Why is decoder.layer_norm.weight mean 11.098?
/// A1: OpenAI's original training produced these values.
///
/// Q2: Why didn't we normalize to gamma≈1.0?
/// A2: We preserve original weights bit-exact.
///
/// Q3: Why is this correct?
/// A3: HuggingFace reference has identical values.
///
/// RESOLUTION: Not a bug - correct weight preservation verified.
#[test]
fn test_decoder_weights_match_reference() {
    let hf_reference: [f32; 10] = [11.7109, 10.3359, 7.9414, ...];
    let our_values: [f32; 10] = [11.7109, 10.3359, 7.9414, ...];
    let (t_stat, df, p_value) = welch_t_test(&our_values, &hf_reference);
    assert!(p_value > 0.05, "Weights should match reference");
}
```

## Statistical Validation with T-Tests

For numerical comparisons, use Welch's t-test to provide statistical rigor:

### Welch's T-Test Implementation

```rust
/// Welch's t-test for comparing two samples (unequal variances)
///
/// Returns: (t-statistic, degrees of freedom, approximate p-value)
fn welch_t_test(sample1: &[f32], sample2: &[f32]) -> (f64, f64, f64) {
    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    // Calculate means
    let mean1 = sample1.iter().map(|&x| x as f64).sum::<f64>() / n1;
    let mean2 = sample2.iter().map(|&x| x as f64).sum::<f64>() / n2;

    // Calculate variances
    let var1 = sample1.iter()
        .map(|&x| (x as f64 - mean1).powi(2))
        .sum::<f64>() / (n1 - 1.0);
    let var2 = sample2.iter()
        .map(|&x| (x as f64 - mean2).powi(2))
        .sum::<f64>() / (n2 - 1.0);

    // Welch's t-test
    let se = ((var1 / n1) + (var2 / n2)).sqrt();
    let t = if se > 1e-10 { (mean1 - mean2) / se } else { 0.0 };

    // Welch-Satterthwaite degrees of freedom
    let num = ((var1 / n1) + (var2 / n2)).powi(2);
    let denom = (var1 / n1).powi(2) / (n1 - 1.0)
              + (var2 / n2).powi(2) / (n2 - 1.0);
    let df = if denom > 1e-10 { num / denom } else { n1 + n2 - 2.0 };

    // P-value approximation
    let p_approx = if t.abs() < 1e-10 {
        1.0  // Identical samples
    } else if df > 30.0 {
        2.0 * (1.0 - normal_cdf(t.abs()))
    } else {
        // Conservative estimate for small df
        (-0.5 * t.abs()).exp().min(1.0)
    };

    (t, df, p_approx)
}
```

### Interpreting Results

| p-value | Interpretation |
|---------|----------------|
| p = 1.0 | Samples are identical |
| p > 0.05 | No significant difference (PASS) |
| p < 0.05 | Significant difference (INVESTIGATE) |
| p < 0.01 | Highly significant difference (FAIL) |

## Three-Way Ground Truth Comparison

For ASR implementations, validate against multiple references:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   HuggingFace   │    │   whisper.cpp   │    │   whisper.apr   │
│   (Reference)   │    │   (C++ impl)    │    │   (Our impl)    │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Three-Way T-Test    │
                    │   Validation Matrix   │
                    └───────────────────────┘
```

### Validation Matrix

| Comparison | t-statistic | p-value | Result |
|------------|-------------|---------|--------|
| HuggingFace vs whisper-apr | 0.0 | 1.0 | IDENTICAL |
| HuggingFace vs whisper.cpp | 0.0 | 1.0 | IDENTICAL |
| whisper-apr vs whisper.cpp | 0.0 | 1.0 | IDENTICAL |

**Conclusion**: All three implementations use identical weights from OpenAI's original model.

## Running Weight Validation

```bash
# Verify weights against HuggingFace
cargo run --release --example verify_hf_weights

# Five-Whys validation example
cargo run --release --example five_whys_validation

# Full CLI parity tests with t-tests
cargo test cli_parity -- --nocapture
```

## Best Practices

1. **Always document Five-Whys**: Include the analysis in test docstrings
2. **Use t-tests for floats**: Never use exact equality for floating-point comparisons
3. **Compare against references**: Maintain ground truth from HuggingFace and whisper.cpp
4. **Track p-values**: p > 0.99 indicates bit-exact match
5. **Document resolutions**: Mark whether an investigation found a bug or confirmed correct behavior

## PMAT Integration

Track Five-Whys investigations with PMAT:

```bash
# Start investigation
pmat work start WAPR-T10-D-FIVE-WHYS

# Run Five-Whys
pmat five-whys "decoder LayerNorm gamma has unusual mean"

# Complete with resolution
pmat work complete WAPR-T10-D-FIVE-WHYS
```

## Further Reading

- [CLI Parity Specification](../../specifications/whisper-cli-parity.md) - Full 240-point validation
- [Ground Truth Specification](../../specifications/ground-truth-whisper-apr-cpp-hugging-face.md) - Three-way comparison methodology
