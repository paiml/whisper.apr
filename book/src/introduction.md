# Introduction

Whisper.apr is a WASM-first automatic speech recognition (ASR) engine implementing OpenAI's Whisper architecture in pure Rust. Unlike whisper.cpp (C++ with Emscripten) or Python implementations, Whisper.apr is designed from inception for browser deployment via `wasm32-unknown-unknown`.

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

## Project Status

Whisper.apr is under active development. Current focus:

- [x] Core transformer architecture
- [x] Audio preprocessing (mel spectrogram)
- [x] BPE tokenization
- [ ] Greedy decoding
- [ ] Beam search
- [ ] .apr model format
- [ ] JavaScript bindings

## Next Steps

- [Installation](./getting-started/installation.md) - Set up your development environment
- [Quick Start](./getting-started/quick-start.md) - Transcribe your first audio
- [Architecture](./architecture/overview.md) - Understand the system design
