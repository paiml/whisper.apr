# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
