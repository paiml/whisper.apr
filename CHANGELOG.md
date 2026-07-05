# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING (build-only): `cli` is now in default features.** `cargo install whisper-apr` (and
  `cargo install --path .` from a checkout) lands the `whisper-apr` binary on PATH without
  requiring `--features cli`. Library-only consumers (e.g. WASM targets) should opt out with
  `default-features = false` and pick the subset they need (e.g. `["std", "simd", "parallel"]`).
  Closes [#55](https://github.com/paiml/whisper.apr/issues/55).
- **BREAKING (target name): the `whisper-apr-cli` bin alias was removed.** Both targets pointed
  at the same source file (`src/bin/whisper-apr-cli.rs`), generating a `cargo` build-target
  collision warning and confusing users about which name to invoke. The canonical name is
  `whisper-apr`, matching the crate name and `default-run`. Users invoking the old alias should
  switch to the canonical name.

### Fixed
- `load_audio_samples` now preserves `CliError::UnsupportedFormat` and `CliError::NotImplemented`
  through the symphonia → ffmpeg fallback chain, instead of wrapping every failure as
  `InvalidArgument`. The fallback was added later for HE-AAC codecs symphonia can't handle, but
  it masked the structural error variants the test suite asserts on. Three latent
  `test_load_audio_*` tests (no_extension / unsupported_format / unknown_extension) now pass
  under the default lint config (#55).

### Added
- `--hotwords` CLI flag for comma-separated logit biasing during decoding (WAPR-170). Boosts
  domain-specific vocabulary (e.g. `--hotwords "Databricks,PySpark,SparkSQL"`) without the
  hallucination risk of `--prompt`. Safe with all model sizes including tiny/base. The existing
  `HotwordBooster` engine applies logit biasing during decoding — this flag exposes it to CLI users.

### Changed
- `--prompt` flag now includes warning about hallucination risk with tiny/base models (<100M params).
  With these models, `--prompt` feeds forced decoder tokens via `<|startofprev|>`, causing a degenerate
  loop during silence segments where the model regurgitates the prompt as literal subtitle text. Prefer
  `--hotwords` for domain vocabulary biasing.
- `make install` prevents duplicate binary installs: removes stale copies outside `~/.cargo/bin/` before
  install, verifies single binary post-install (exits non-zero if duplicates detected).

## [0.3.2] - 2026-07-05

### Fixed
- **WASM / minimal-feature builds now compile.** `cargo build --target wasm32-unknown-unknown
  --no-default-features --features wasm` (and any `default-features = false, features = ["std"]`
  library consumer) failed to compile because `simd::optimized::tiled_matmul_into` and
  `simd::matrix::{matmul, matmul_with_prepacked}` called `rayon` (`current_num_threads`,
  `into_par_iter`, `par_chunks_mut`) without `#[cfg(feature = "parallel")]` guards, but `rayon`
  is gated behind the `parallel` feature (not implied by `std`/`wasm`). Each site now gates the
  parallel path and adds a single-threaded serial fallback, matching the other SIMD functions.
  (The wasm/minimal build still emits a few benign dead-code / unused-import warnings from
  feature-gated and generated code — pre-existing and non-blocking; this release fixes the hard
  compile *errors* that previously stopped the build entirely.)

### Changed
- `make build-wasm` now builds with `--no-default-features` (it previously used the default
  feature set, which pulls wasm-incompatible deps — masking the breakage above).
- Auto-download error message in `load_or_download_model` is now actionable (points to
  `cargo install whisper-apr --features converter` or `--model-path`), and the README no longer
  implies the default install auto-downloads models (it requires the `converter` feature, kept
  opt-in to preserve the small default dependency tree).

### Added
- `make check-features` — a feature-matrix regression guard that compiles the crate for wasm and
  the minimal `std`-only library config. Wired into `make tier2` so this class of feature-gating
  regression is caught before merge (Jidoka: build quality in).

## [0.3.1] - 2026-07-05

### Changed
- **Dependencies consolidated onto the aprender monorepo.** `trueno` → `aprender-compute`,
  `trueno-gpu` → `aprender-gpu`, `realizar` → `aprender-serve`, and `provable-contracts-macros`
  → `aprender-contracts-macros`, all unified at `0.51`. Library names are unchanged, so there is
  no source or public-API change. Removes three duplicate `trueno` versions and the legacy `0.29`
  line.
- `hf-hub` now uses the `ureq`/rustls backend (`default-features = false, features = ["ureq"]`),
  dropping `reqwest`, `native-tls`, and `openssl` from the dependency tree.

### Removed
- **`apr-cli` dependency dropped.** The default `cli` feature pulled it in only to proxy
  `apr pull` / `apr ls`, but it transitively dragged in ~230 crates (batuta/orchestrate →
  arrow+parquet, tonic, axum, rusqlite, opentelemetry, reqwest, a wgpu stack, and the `quick-xml`
  advisories). `apr pull` / `apr ls` are reimplemented on the in-tree `ModelDownloader` (new
  `cli::apr_commands::model_pull` module) under a provable-contracts contract with offline
  falsification tests. Net effect: `Cargo.lock` **809 → 567** crates.

### Fixed
- Eliminated the `quick-xml` `RUSTSEC-2026-0194` / `RUSTSEC-2026-0195` advisories at the source
  (they arrived transitively via `apr-cli`), rather than suppressing them in `audit.toml`.
- Removed `panic = "abort"` from the release profile, restoring stack unwinding in release builds.

## [0.2.2] - 2025-01-26

### Changed
- Switched to crates.io dependencies for reproducible builds
  - trueno 0.14.3, trueno-gpu 0.4.12, aprender 0.24.1, realizar 0.6.10
- Fixed `realizar-gpu` feature to enable both `gpu` and `cuda` features
- Fixed CUDA module import in core_generated.rs

### Removed
- Removed `inference-monitoring` feature due to cyclic dependency with entrenar

## [0.2.1] - 2025-01-26

### Changed
- Refactored 20+ modules into modular directory structure (file.rs → file/{mod.rs, tests.rs})
- Improved code organization for TDG (Technical Debt Grade) scoring
- Split test code from implementation across all major modules

### Modules Refactored
- `audio/mel`, `audio/vad`, `audio/wav`, `audio/resampler`, `audio/ring_buffer`
- `backend/selector`, `backend/traits`
- `cli/parity`, `cli/output`
- `diarization/detection`, `diarization/clustering`, `diarization/embedding`, `diarization/segmentation`
- `format/compress`, `format/validation`
- `gpu/detect`, `gpu/pipeline`
- `inference/beam`, `inference/streaming`
- `memory/mmap`
- `model/lfm2/model`, `model/lfm2/tokenizer`
- `publish`
- `timestamps/alignment`, `timestamps/interpolation`, `timestamps/boundaries`
- `vocabulary/adapter`, `vocabulary/hotwords`, `vocabulary/trie`
- `wasm/capabilities`, `wasm/timestamps`, `wasm/worker`

## [0.2.0] - 2025-01-09

### Added
- LFM2-2.6B-Transcript Post-Transcription Benchmark (GitHub Issue #10) (#10)
- WAPR-182: Phase 3 - Transcription in Worker (#6)
- WAPR-181: Phase 2 - Model Loading in Worker (#2)
- WAPR-180: Phase 1 - Async Worker Foundation (#1)
- WAPR-184: Probar GUI & Pixel Regression Tests (#5)
- WAPR-183: Phase 4 - Robustness & Testing (#4)
- SIMD-optimized Conv1d with im2col transformation
- SIMD-optimized LayerNorm with batch processing
- SIMD-optimized attention with unified dispatch pattern
- Transposed weight caching for LinearWeights (Phase 2 memory optimization)
- `finalize_weights()` method cascade through Encoder/Decoder/Attention/FeedForward
- Flash Attention implementation with O(n) memory (Phase 3 attention optimization)
- `FlashAttentionConfig` struct for configurable block-based attention
- `forward_cross_flash()` and `forward_cross_auto()` methods in MultiHeadAttention
- SIMD helpers: `max_element()`, `scale_inplace()`, `axpy()`, `add_inplace()`
- `CircularKVBuffer` for memory-efficient sliding window KV caching (Phase 3)
- Streaming attention support: `forward_streaming()` and `forward_self_streaming()` methods
- End-to-end transcription benchmarks with Flash Attention comparison (Sprint 4)
- Streaming attention benchmarks for incremental decoding (Sprint 4)
- Regression test suite for SIMD optimizations (Sprint 4)
- Public exports for Flash Attention API (`flash_attention`, `flash_attention_simd`, `FlashAttentionConfig`)
- Custom vocabulary fine-tuning support (WAPR-170 to WAPR-173)
- Improved word-level timestamps (WAPR-160 to WAPR-163)
- Speaker diarization foundation (WAPR-150 to WAPR-153)
- WebGPU backend for matrix operations (WAPR-130 to WAPR-143)

### Changed
- Unified SIMD dispatch pattern using `cfg!(feature = "simd")`
- Refactored complex functions to reduce cyclomatic complexity
- Improved code organization in demo applications

### Fixed
- Input size mismatch errors in encoder/decoder
- SATD violations in GPU detection code

## [0.1.0] - 2024-01-01

### Added
- Initial release of whisper.apr
- Pure Rust implementation of OpenAI Whisper
- WASM-first architecture with `wasm32-unknown-unknown` target
- WASM SIMD 128-bit intrinsics support via trueno
- Mel spectrogram computation with 80-mel filterbank
- Audio resampling to 16kHz
- BPE tokenization with 51,865 token vocabulary
- Greedy and beam search decoding strategies
- .apr model format with LZ4 compression
- Support for tiny, base, and small model sizes

[Unreleased]: https://github.com/paiml/whisper.apr/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/paiml/whisper.apr/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/paiml/whisper.apr/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/paiml/whisper.apr/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/paiml/whisper.apr/releases/tag/v0.1.0
