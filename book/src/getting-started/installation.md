# Installation

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
whisper-apr = "0.1"
```

### Feature Flags

```toml
[dependencies]
whisper-apr = { version = "0.1", features = ["wasm", "simd"] }
```

Available features:
- `std` (default) - Standard library support
- `wasm` - WASM bindings via wasm-bindgen
- `simd` - Explicit SIMD optimization paths
- `tracing` - Performance tracing via renacer

## Building from Source

### Prerequisites

- Rust 1.85+ (edition 2024)
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
# All tests
cargo test

# WASM tests (requires Chrome)
wasm-pack test --headless --chrome
```

## Model Download

Download pre-converted `.apr` models:

```bash
# tiny model (~40MB)
curl -O https://models.paiml.com/whisper/tiny.apr

# base model (~75MB)
curl -O https://models.paiml.com/whisper/base.apr
```

Or use the model converter to create `.apr` from PyTorch weights:

```bash
cd models/converter
python convert.py --model base --output base.apr
```

## Verifying Installation

```rust
use whisper_apr::{WhisperApr, TranscribeOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    let model_data = std::fs::read("base.apr")?;
    let whisper = WhisperApr::load(&model_data)?;

    println!("Model loaded successfully!");
    Ok(())
}
```
