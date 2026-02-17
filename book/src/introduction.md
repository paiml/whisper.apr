# Introduction

Whisper.apr is a WASM-first automatic speech recognition (ASR) engine implementing OpenAI's Whisper architecture in pure Rust. Unlike whisper.cpp (C++ with Emscripten) or Python implementations, Whisper.apr is designed from inception for browser deployment via `wasm32-unknown-unknown`. It also supports Moonshine ASR models and direct GGUF model loading.

## Why Whisper.apr?

### Privacy-First Transcription

Traditional speech recognition requires sending audio to cloud servers. Whisper.apr runs entirely in the browser:

- **No server roundtrips** - Zero latency from network requests
- **Complete privacy** - Audio never leaves the user's device
- **Offline capable** - Works without internet connection
- **HIPAA/GDPR friendly** - Simplifies compliance for sensitive applications

### Pure Rust Advantages

Building on Rust's superior WASM toolchain delivers:

- **30-40% smaller binaries** through tree-shaking (vs Emscripten)
- **Native WASM SIMD** - 128-bit intrinsics without wrapper overhead
- **Zero-copy buffers** - Shared memory with JavaScript
- **Type safety** - Catch errors at compile time

### Real-Time Performance

Whisper.apr achieves practical transcription speeds:

| Model | Parameters | Target RTF | Memory |
|-------|------------|------------|--------|
| tiny  | 39M        | 2.0x       | 150MB  |
| base  | 74M        | 2.5x       | 300MB  |
| small | 244M       | 4.0x       | 800MB  |
| moonshine-tiny | 27M | 1.0x    | 50MB   |
| moonshine-base | 61M | 1.5x    | 100MB  |

*RTF = Real-Time Factor (2.0x means 60s audio takes 120s to process)*

## Quick Example

```javascript
import init, { WhisperApr } from 'whisper-apr';

// Initialize WASM module
await init();

// Load model (streams from CDN)
const whisper = await WhisperApr.load('/models/base.apr');

// Transcribe audio
const result = await whisper.transcribe(audioBuffer, {
  language: 'auto',
  task: 'transcribe',
});

console.log(result.text);
```

## Design Philosophy

Whisper.apr follows Toyota Way principles:

1. **Kaizen** - Continuous improvement through iterative sprints
2. **Jidoka** - Quality built in via PMAT gates and mutation testing
3. **Genchi Genbutsu** - Reality-based performance targets from browser benchmarks

## Project Status (v0.2.4)

- [x] Core transformer architecture (encoder-decoder)
- [x] Audio preprocessing (mel spectrogram, symphonia multi-format)
- [x] BPE tokenization (51,865 tokens, 99 languages)
- [x] Greedy and beam search decoding
- [x] .apr model format with streaming load
- [x] GGUF model loading (pre-quantized from HuggingFace)
- [x] Moonshine ASR (GQA decoder, ConvStem encoder)
- [x] Large v3 Turbo support (809M params)
- [x] JavaScript/WASM bindings
- [x] Int4/Int8 quantization
- [x] CLI with transcribe, probe, parity, config-check, selftest
- [x] 2,885 tests, 96%+ coverage, TDG 99.5/100 A+

## Next Steps

- [Installation](./getting-started/installation.md) - Set up your development environment
- [Quick Start](./getting-started/quick-start.md) - Transcribe your first audio
- [Architecture](./architecture/overview.md) - Understand the system design
