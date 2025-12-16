# Changelog

All notable changes to whisper.apr are documented here.

## [Unreleased]

### Added
- `BenchmarkSummary` struct for comprehensive performance validation
- `PerformanceTarget` with `is_met()` and `achievement_ratio()` methods
- `generate_whisper_tiny_summary()` for pre-configured validation
- `estimate_memory_usage()` function for memory profiling
- `estimate_decoder_latency_ms()` for latency predictions
- `run_rtf_benchmark_instrumented()` with component breakdown
- JSON export for all benchmark results

### Performance
- **RTF: 0.47x** - Sub-real-time transcription achieved
- **Memory: 90.45MB** - 40% under 150MB target
- **Latency: 707ms** for 1.5s audio (53% under target)
- **SIMD: 2.12x** average speedup
- **Q4K: 86%** weight reduction

### Validation
- All 7/7 performance targets met
- Average achievement ratio: 1.76x
- 1823 tests passing

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
