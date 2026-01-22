# Changelog

All notable changes to whisper.apr are documented here.

## [Unreleased]

### Planned
- WebGPU acceleration
- Turbo model support
- Word-level timestamps
- Voice activity detection

## [0.2.0] - 2026-01-22

### Added
- GPU-resident tensor architecture via trueno-gpu 0.4.10
- CUDA acceleration with 5.8x speedup over whisper.cpp
- `BenchmarkSummary` struct for comprehensive performance validation
- `PerformanceTarget` with `is_met()` and `achievement_ratio()` methods
- `generate_whisper_tiny_summary()` for pre-configured validation
- `estimate_memory_usage()` function for memory profiling
- `estimate_decoder_latency_ms()` for latency predictions
- `run_rtf_benchmark_instrumented()` with component breakdown
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
- 2125 tests passing (up from 1823)
- 95% test coverage
- Golden tests locked as immutable guardians

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
