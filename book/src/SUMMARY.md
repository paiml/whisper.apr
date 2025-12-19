# Whisper.apr - WASM-First Speech Recognition

[Introduction](./introduction.md)

# Getting Started

- [Installation](./getting-started/installation.md)
- [Quick Start](./getting-started/quick-start.md)
- [Command Line Interface](./getting-started/cli.md)
- [Browser Integration](./getting-started/browser-integration.md)
- [Core Concepts](./getting-started/core-concepts.md)

# Architecture

- [Overview](./architecture/overview.md)
- [WASM-First Design](./architecture/wasm-first.md)
- [Audio Pipeline](./architecture/audio-pipeline.md)
- [Transformer Architecture](./architecture/transformer.md)
  - [Encoder](./architecture/encoder.md)
  - [Decoder](./architecture/decoder.md)
  - [Multi-Head Attention](./architecture/attention.md)
- [.apr Model Format](./architecture/apr-format.md)
- [Quantization](./architecture/quantization.md)
- [Trueno Integration](./architecture/trueno-integration.md)

# API Reference

- [WhisperApr](./api-reference/whisper-apr.md)
- [TranscribeOptions](./api-reference/transcribe-options.md)
- [TranscriptionResult](./api-reference/transcription-result.md)
- [DecodingStrategy](./api-reference/decoding-strategy.md)
- [Audio Processing](./api-reference/audio-processing.md)
- [Error Handling](./api-reference/error-handling.md)
- [JavaScript Bindings](./api-reference/javascript-bindings.md)

# Performance

- [Benchmarks Overview](./performance/benchmarks.md)
- [Format Comparison](./performance/format-comparison.md)
- [Real-Time Factor Analysis](./performance/rtf-analysis.md)
- [WASM SIMD Performance](./performance/wasm-simd.md)
- [Memory Optimization](./performance/memory.md)
- [Model Quantization Impact](./performance/quantization-impact.md)
- [Browser Comparison](./performance/browser-comparison.md)
- [Profiling with Renacer](./performance/renacer-profiling.md)

# Examples

- [Basic Transcription](./examples/basic-transcription.md)
- [Real-Time Microphone](./examples/real-time-microphone.md)
- [File Upload](./examples/file-upload.md)
- [Language Detection](./examples/language-detection.md)
- [Translation](./examples/translation.md)
- [Web Worker Integration](./examples/web-workers.md)
- [React Integration](./examples/react-integration.md)
- [Progressive Web App](./examples/pwa.md)

# Development Guide

- [Contributing](./development/contributing.md)
- [Extreme TDD](./development/extreme-tdd.md)
- [Testing](./development/testing.md)
  - [Unit Tests](./development/unit-tests.md)
  - [Property-Based Tests](./development/property-based-tests.md)
  - [WASM Tests](./development/wasm-tests.md)
  - [Backend E2E Tests](./development/backend-testing.md)
  - [WER Validation](./development/wer-validation.md)
- [TUI Pipeline Visualization](./development/tui-visualization.md)
- [Benchmarking](./development/benchmarking.md)
- [Quality Gates](./development/quality-gates.md)
- [PMAT Integration](./development/pmat-integration.md)
- [Debugging Models with Hex Dumps](./development/hex-debugging.md)

# Advanced Topics

- [Custom Model Conversion](./advanced/model-conversion.md)
- [Streaming Inference](./advanced/streaming.md)
- [Voice Activity Detection](./advanced/vad.md)
- [WebGPU Backend](./advanced/webgpu.md)
- [Server-Side Deployment](./advanced/server-side.md)
- [Edge Deployment](./advanced/edge-deployment.md)
- [Optimizing for Mobile](./advanced/mobile-optimization.md)

# Appendix

- [Glossary](./appendix/glossary.md)
- [References](./appendix/references.md)
- [FAQ](./appendix/faq.md)
- [Changelog](./appendix/changelog.md)
- [Whisper Model Comparison](./appendix/model-comparison.md)
- [Browser Compatibility](./appendix/browser-compatibility.md)
