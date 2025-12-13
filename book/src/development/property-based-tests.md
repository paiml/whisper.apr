# Property-Based Tests

Whisper.apr uses **proptest** for property-based testing to validate invariants across randomly generated inputs.

## Why Property-Based Testing?

Traditional unit tests check specific examples:
```rust
assert_eq!(softmax(&[1.0, 2.0, 3.0]).iter().sum::<f32>(), 1.0);
```

Property tests verify **invariants** across many random inputs:
```rust
proptest! {
    #[test]
    fn property_softmax_sums_to_one(logits in vec(any::<f32>(), 4..128)) {
        let probs = softmax(&logits);
        prop_assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}
```

## Property Test Modules

### Mel Filterbank (`audio/mel.rs`)

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    // Mel scale is monotonically increasing
    #[test]
    fn property_mel_scale_monotonic(freq in 0.0f32..20000.0) {
        let mel1 = MelFilterbank::hz_to_mel(freq);
        let mel2 = MelFilterbank::hz_to_mel(freq + 1.0);
        prop_assert!(mel2 >= mel1);
    }

    // Hz -> Mel -> Hz roundtrip
    #[test]
    fn property_mel_hz_roundtrip(freq in 20.0f32..8000.0) {
        let mel = MelFilterbank::hz_to_mel(freq);
        let back = MelFilterbank::mel_to_hz(mel);
        prop_assert!((freq - back).abs() < 1e-2);
    }

    // Filterbank values are non-negative
    #[test]
    fn property_filterbank_nonnegative(n_mels in 20usize..128, n_fft in 256usize..1024) {
        let mel = MelFilterbank::new(n_mels, n_fft, 16000);
        for val in &mel.filters {
            prop_assert!(*val >= 0.0);
        }
    }
}
```

### Ring Buffer (`audio/ring_buffer.rs`)

```rust
proptest! {
    // Write/read preserves data integrity
    #[test]
    fn property_write_read_preserves_data(
        capacity in 64usize..1024,
        data_len in 1usize..512
    ) {
        let capacity = capacity.max(data_len + 1);
        let mut buffer = RingBuffer::new(capacity);
        let data: Vec<f32> = (0..data_len).map(|i| i as f32).collect();

        buffer.write(&data);
        let mut output = vec![0.0; data_len];
        let read = buffer.read(&mut output);

        prop_assert_eq!(read, data_len);
        prop_assert_eq!(output, data);
    }

    // Clear empties the buffer
    #[test]
    fn property_clear_empties_buffer(capacity in 32usize..256) {
        let mut buffer = RingBuffer::new(capacity);
        buffer.write(&[1.0, 2.0, 3.0]);
        buffer.clear();

        prop_assert!(buffer.is_empty());
        prop_assert_eq!(buffer.available_read(), 0);
    }
}
```

### SIMD Operations (`simd.rs`)

```rust
proptest! {
    // Dot product is commutative
    #[test]
    fn property_dot_commutative(len in 4usize..256) {
        let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2).cos()).collect();

        let ab = simd::dot(&a, &b);
        let ba = simd::dot(&b, &a);
        prop_assert!((ab - ba).abs() < 1e-5);
    }

    // Softmax output sums to 1.0
    #[test]
    fn property_softmax_sums_to_one(len in 4usize..128) {
        let logits: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin()).collect();
        let probs = simd::softmax(&logits);
        let sum: f32 = probs.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5);
    }

    // GELU is bounded for reasonable inputs
    #[test]
    fn property_gelu_bounded(len in 4usize..128) {
        let input: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1 - 5.0)).collect();
        let output = simd::gelu(&input);
        for (x, y) in input.iter().zip(output.iter()) {
            prop_assert!(y.is_finite());
            prop_assert!(*y >= -0.5 && *y <= *x + 0.5);
        }
    }

    // Layer norm produces mean ~0, std ~1
    #[test]
    fn property_layer_norm_mean_zero(len in 8usize..256) {
        let data: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin()).collect();
        let gamma = vec![1.0; len];
        let beta = vec![0.0; len];
        let normalized = simd::layer_norm(&data, &gamma, &beta, 1e-5);
        let mean: f32 = normalized.iter().sum::<f32>() / len as f32;
        prop_assert!(mean.abs() < 1e-5);
    }
}
```

### Tokenizer (`tokenizer/mod.rs`)

```rust
proptest! {
    // Encode produces valid token IDs
    #[test]
    fn property_encode_produces_valid_tokens(s in "[a-zA-Z0-9 ]{1,50}") {
        let tokenizer = BpeTokenizer::with_base_tokens();
        if let Ok(tokens) = tokenizer.encode(&s) {
            for token in &tokens {
                prop_assert!((*token as usize) < tokenizer.vocab_size());
            }
        }
    }

    // Whisper special tokens are in expected range
    #[test]
    fn property_special_tokens_reasonable(_dummy in 0u8..1) {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let special_tokens = [SOT, EOT, TRANSCRIBE, TRANSLATE, NO_TIMESTAMPS];

        for token in special_tokens {
            prop_assert!(
                (token as usize) >= 50257 && (token as usize) < 60000
            );
            prop_assert!(tokenizer.vocab_size() > 0);
        }
    }
}
```

## Configuration

Property tests use a reduced case count (50) for fast CI execution:

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]
    // tests...
}
```

For thorough local testing, increase cases:
```bash
PROPTEST_CASES=1000 cargo test property_
```

## Running Property Tests

```bash
# Run all property tests
cargo test property_ --lib

# Run with verbose output
cargo test property_ --lib -- --nocapture

# Run specific module's property tests
cargo test audio::mel::tests::property_tests --lib
```

## Best Practices

1. **Test invariants, not examples**: Focus on properties that should hold for ALL inputs
2. **Use meaningful ranges**: `freq in 20.0f32..8000.0` (audible range) instead of `any::<f32>()`
3. **Keep tests fast**: Use 50 cases for CI, more for local testing
4. **Handle edge cases**: Property tests often find unexpected edge cases
5. **Save regression seeds**: Proptest saves failing seeds in `proptest-regressions/`
