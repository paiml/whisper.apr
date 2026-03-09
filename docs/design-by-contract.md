# Design by Contract -- whisper.apr

whisper.apr enforces contracts at three levels: audio preprocessing (mel
spectrogram shape), model architecture (encoder/decoder dimension consistency),
and format validation (APR metadata plausibility). Violations are caught at
construction time or load time, never deferred to inference.

## Mel Spectrogram Shape Contracts

The mel filterbank is the first tensor in the pipeline. Shape errors here
propagate silently through the entire encoder, producing garbage.

| Parameter    | Whisper v1-v2 / tiny-large | large-v3 / large-v3-turbo |
|-------------|---------------------------|--------------------------|
| `n_mels`     | 80                        | 128                      |
| `n_fft`      | 400                       | 400                      |
| `hop_length` | 160                       | 160                      |
| `n_freqs`    | 201 (`n_fft/2 + 1`)       | 201                      |
| `n_frames`   | 3000 (30s at 16kHz)       | 3000                     |
| sample_rate  | 16000 Hz                  | 16000 Hz                 |

The `MelConfig` struct (from `aprender::audio`) is the source of truth.
`MelConfig::default()` returns the Whisper-standard 80-mel configuration.
`ModelConfig::n_mels` must match the `MelConfig` used to build the filterbank.

The conv1 weight tensor has shape `[d_model, n_mels, 3]` -- a mismatch between
`n_mels` in the config and the actual filterbank width produces a matmul
dimension error at encoder entry.

## Encoder/Decoder Dimension Consistency

Each `ModelConfig` factory method (`tiny()`, `base()`, `small()`, `medium()`,
`large()`, `large_v3_turbo()`) is a `const fn` returning a known-good
configuration. The invariants enforced:

| Invariant                              | Rationale                               |
|----------------------------------------|-----------------------------------------|
| `n_audio_state == n_text_state`        | Cross-attention requires matching dims  |
| `n_audio_state % n_audio_head == 0`   | Head dimension must divide evenly       |
| `n_text_state % n_text_head == 0`     | Head dimension must divide evenly       |
| `n_audio_layer > 0 && n_text_layer > 0` | At least one layer each              |
| `n_audio_head == n_text_head` (standard Whisper) | Symmetric attention     |

The `large_v3_turbo` variant is intentionally asymmetric: 32 encoder layers,
4 decoder layers, 128 mels. This is validated in `core_generated.rs` tests.

## Architecture-Specific Configurations

| Variant        | d_model | enc layers | dec layers | heads | n_mels | params |
|----------------|---------|-----------|-----------|-------|--------|--------|
| `tiny`         | 384     | 4         | 4         | 6     | 80     | ~39M   |
| `base`         | 512     | 6         | 6         | 8     | 80     | ~74M   |
| `small`        | 768     | 12        | 12        | 12    | 80     | ~244M  |
| `medium`       | 1024    | 24        | 24        | 16    | 80     | ~769M  |
| `large`        | 1280    | 32        | 32        | 20    | 80     | ~1.5B  |
| `large_v3_turbo` | 1280  | 32        | 4         | 20    | 128    | ~809M  |

## APR Format Validation

When loading via `WhisperApr::load_from_apr()`, the APR header metadata is
validated against the architecture table above. The GGUF loader
(`format::gguf_loader`) infers `d_model`, `n_mels`, `n_encoder_layers`, and
`n_decoder_layers` from tensor shapes and rejects unknown combinations.

## Generated Code Enforcement

`src/core_generated.rs` contains tests that assert config consistency:

```rust
assert_eq!(whisper.config().n_audio_layer, 4);  // tiny
assert_eq!(whisper.config().n_audio_state, 384);
assert_eq!(whisper.config().n_text_state, 384);
```

These tests run via `cargo test --lib` and serve as regression guards against
config drift.

## Running the Contracts

```bash
cargo test --lib                              # All unit tests including config assertions
cargo run --example design_by_contract        # Standalone contract demonstration
```

## Cross-References

- `src/model/mod.rs` -- `ModelConfig` struct and factory methods
- `src/core_generated.rs` -- `WhisperApr` integration tests
- `src/audio/mod.rs` -- `MelConfig` re-export and default assertions
- `src/format/gguf_loader.rs` -- tensor-shape-based config inference
- `src/format/whisper_metadata.rs` -- APR metadata validation
