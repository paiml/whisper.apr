# Testing

Whisper.apr follows **EXTREME TDD** methodology with comprehensive testing at multiple levels.

## Test Distribution

The project targets the following test distribution:
- **60%** Unit tests
- **30%** Property-based tests
- **10%** Integration tests

## Running Tests

```bash
# All tests (unit + property + integration + doc)
cargo test

# Fast tests with nextest (recommended)
make test-fast

# Unit tests only
cargo test --lib

# Property-based tests only
cargo test property_ --lib

# Doc tests
cargo test --doc
```

## Coverage

The project maintains **≥95% line coverage** using `cargo-llvm-cov`:

```bash
# Generate coverage report
make coverage

# View HTML report
open target/coverage/html/index.html
```

### Current Coverage Stats

| Module | Line Coverage |
|--------|---------------|
| audio/streaming.rs | 99.45% |
| audio/ring_buffer.rs | 98.94% |
| format/checksum.rs | 100.00% |
| simd.rs | 95.19% |
| **TOTAL** | **95.19%** |

## Test Categories

### Unit Tests

Located alongside the code in `#[cfg(test)]` modules:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_filterbank_new() {
        let mel = MelFilterbank::new(80, 400, 16000);
        assert_eq!(mel.n_mels(), 80);
    }
}
```

### Property-Based Tests

Using `proptest` for invariant validation:

```rust
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn property_softmax_sums_to_one(len in 4usize..128) {
            let logits: Vec<f32> = (0..len)
                .map(|i| (i as f32 * 0.1).sin())
                .collect();
            let probs = simd::softmax(&logits);
            let sum: f32 = probs.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-5);
        }
    }
}
```

### Integration Tests

End-to-end tests in `tests/` directory testing the full transcription pipeline.

## Makefile Targets

```bash
make test-fast    # Fast tests with nextest
make coverage     # Coverage with HTML report
make tier1        # Quick validation (<1s)
make tier2        # Pre-commit (<5s)
make tier3        # Pre-push (1-5min)
```

## Mutation Testing

Validate test quality with `cargo-mutants`:

```bash
# Run mutation tests
make mutants

# Quick mutation test on changed files
make mutants-quick

# List mutants for a specific file
cargo mutants --list --file src/simd.rs
```
