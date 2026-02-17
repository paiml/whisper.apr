# Installation

## CLI (Recommended)

Install the whisper-apr CLI for quick transcription:

```bash
cargo install whisper-apr --features cli
```

Verify installation:

```bash
whisper-apr selftest
```

## Browser Usage (npm)

For web applications, install via npm:

```bash
npm install whisper-apr
```

Or with yarn:

```bash
yarn add whisper-apr
```

## Rust Library

Add to your `Cargo.toml`:

```toml
[dependencies]
whisper-apr = "0.2"
```

### Feature Flags

```toml
[dependencies]
whisper-apr = { version = "0.2", features = ["wasm", "simd"] }
```

Available features:
- `std` (default) - Standard library support
- `simd` (default) - SIMD optimization paths
- `parallel` (default) - Multi-threaded inference via rayon
- `realizar-inference` (default) - Advanced inference primitives
- `wasm` - WASM bindings via wasm-bindgen
- `cli` - Command-line interface
- `converter` - Model converter tool
- `tracing` - Performance tracing via renacer
- `tui` - Terminal UI benchmark visualization
- `symphonia` - Multi-format audio (MP3, FLAC, OGG, AAC, M4A)

## Building from Source

### Prerequisites

- Rust 1.75+ (edition 2021)
- wasm-pack (for WASM builds)
- Node.js 18+ (for running browser tests)

### Clone and Build

```bash
git clone https://github.com/paiml/whisper.apr
cd whisper.apr

# Native build
cargo build --release

# WASM build
cargo build --target wasm32-unknown-unknown --features wasm --release

# Or use wasm-pack for npm package
wasm-pack build --target web --release
```

### Running Tests

```bash
# All unit tests
cargo test --lib

# Full test suite (including integration tests)
cargo test --features integration-tests

# WASM tests (requires Chrome)
wasm-pack test --headless --chrome
```

## Model Setup

### Auto-Download (CLI)

The CLI auto-downloads models from HuggingFace on first use:

```bash
# Uses whisper-tiny by default (auto-downloads ~39MB)
whisper-apr transcribe -f audio.wav

# Specify model size
whisper-apr transcribe -f audio.wav --model base

# Use Moonshine models
whisper-apr transcribe -f audio.wav --model moonshine-tiny
```

### GGUF Models (Direct Loading)

Load pre-quantized GGUF models from HuggingFace directly:

```bash
# Download a GGUF model manually
# Then load it directly (no conversion needed)
whisper-apr transcribe -f audio.wav --model-path whisper-tiny.gguf
```

### Model Conversion

Convert SafeTensors models to .apr format:

```bash
cargo run --bin whisper-convert --features converter -- \
  --model tiny --output whisper-tiny.apr
```

## Verifying Installation

```rust
use whisper_apr::{WhisperApr, TranscribeOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    let model_data = std::fs::read("whisper-tiny.apr")?;
    let whisper = WhisperApr::load_from_apr(&model_data)?;

    println!("Model loaded successfully!");
    Ok(())
}
```
