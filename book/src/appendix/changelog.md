# Changelog

All notable changes to whisper.apr are documented here.

## [Unreleased]

### Planned
- WebGPU acceleration
- Word-level timestamps
- Distil-Whisper model support

## [0.2.4] - 2026-02-17

### Added
- **Moonshine ASR support** - moonshine-tiny (27M) and moonshine-base (61M) models with GQA decoder and ConvStem encoder
- **GGUF model loading** - Load pre-quantized GGUF models from HuggingFace directly, no conversion step needed
- **Large v3 Turbo model** - 809M params (1280 dim, 32 encoder + 4 decoder layers, 128 mels)
- **Multi-format audio** - MP3, FLAC, OGG/Vorbis, AAC, M4A, MKV/WebM via symphonia
- **CLI commands** - `probe` (forward-pass debugging), `config-check` (model validation), `selftest` (install verification), `parity` (reference comparison)
- **Browser E2E tests** via Probar for WASM deployment validation

### Performance
- **3.5x single-token decoding speedup** via tiled_matvec fast path in matmul_raw
- Moonshine GQA/MLP routed through trueno SIMD matmul + SDPA

### Fixed
- WASM compilation with split parallel/wasm-threads features (#11)
- Model download URL parsing failure by updating hf-hub 0.3 to 0.4 (#13)
- Moonshine encoder LayerNorm placement (post-block only)
- 3 Moonshine forward-pass bugs causing garbage transcription
- NaN in beam search log_softmax when all logits suppressed
- Production unwrap() violations (zero-unwrap policy enforced)

### Quality
- **TDG Score: 99.5/100 (A+)** (up from 90.9)
- **2,885 unit tests**, 0 failures
- **96%+ line coverage** (above 95% target)
- pmat compliance: COMPLIANT, all quality gates passing
- All 15 GitHub issues closed

### Dependencies
- trueno 0.14.6 (SIMD compute)
- aprender 0.25.9 (model format + GGUF parsing)
- realizar 0.6.13 (inference primitives)

## [0.2.0] - 2026-01-22

### Added
- GPU-resident tensor architecture via trueno-gpu 0.4.10
- CUDA acceleration with 5.8x speedup over whisper.cpp
- `BenchmarkSummary` struct for comprehensive performance validation
- JSON export for all benchmark results
- 100+ examples in `examples/` directory

### Performance
- **RTF: 0.47x** - Sub-real-time transcription achieved
- **Memory: 90.45MB** - 40% under 150MB target
- **Latency: 707ms** for 1.5s audio (53% under target)
- **SIMD: 2.12x** average speedup
- **Q4K: 86%** weight reduction
- **CUDA: 5.8x** speedup with GPU-resident tensors

### Validation
- All 7/7 performance targets met
- Average achievement ratio: 1.76x
- 2,125 tests passing
- 95% test coverage

### Dependencies
- trueno 0.13.0
- trueno-gpu 0.4.10 (optional, for CUDA)
- realizar 0.6.8 (optional, for advanced inference)

## [0.1.1] - 2026-01-15

### Fixed
- Minor bug fixes and stability improvements

## [0.1.0] - 2025-12-15

### Added
- Initial whisper.apr implementation
- whisper-tiny model support
- Q4K quantization (4.5-bit precision)
- SIMD acceleration via trueno
- Flash Attention for long sequences
- Greedy and beam search decoding
- Mel spectrogram computation
- BPE tokenization (51,865 tokens)
- Streaming audio support
- WASM build target

### Architecture
- Pure Rust implementation
- `wasm32-unknown-unknown` target (no Emscripten)
- WASM SIMD 128-bit intrinsics
- LZ4-compressed .apr model format

### Performance Targets Met
- RTF < 2.0x for whisper-tiny
- Memory < 150MB peak
- Decoder latency < 1500ms for short audio
