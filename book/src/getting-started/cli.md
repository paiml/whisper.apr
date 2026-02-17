# Command Line Interface

whisper-apr provides a CLI for transcription and model debugging:

```bash
cargo install whisper-apr --features cli
```

## Quick Start

```bash
# Basic transcription (auto-downloads model)
whisper-apr transcribe -f audio.wav

# Choose model size
whisper-apr transcribe -f audio.wav --model base

# Use Moonshine model
whisper-apr transcribe -f audio.wav --model moonshine-tiny

# Load GGUF model directly
whisper-apr transcribe -f audio.wav --model-path whisper-tiny.gguf

# Multi-format audio (MP3, FLAC, OGG, AAC, M4A)
whisper-apr transcribe -f podcast.mp3
```

## Commands

### transcribe

Transcribe audio/video files to text:

```bash
whisper-apr transcribe -f input.wav [OPTIONS]

Options:
  -f, --file <FILE>         Input audio/video file (required)
  -m, --model <MODEL>       Model size [tiny|base|small|large-v3-turbo|moonshine-tiny|moonshine-base]
  --model-path <PATH>       Path to .apr or .gguf model file
  -l, --language <LANG>     Source language (ISO 639-1) or 'auto'
  --beam-size <N>           Beam search size (1 = greedy)
  --temperature <F32>       Sampling temperature
  --translate               Translate to English
  -v, --verbose             Show timing info and debug output
```

### probe

Inspect model internals and run forward-pass debugging:

```bash
whisper-apr probe --model-path model.apr [OPTIONS]

Options:
  --model-path <PATH>       Path to model file
  --layer <N>               Inspect specific layer
  -v, --verbose             Detailed output
```

### config-check

Validate model configuration and tensor shapes:

```bash
whisper-apr config-check --model-path model.apr
```

### parity

Compare output against reference implementations:

```bash
whisper-apr parity --model-path model.apr -f audio.wav [OPTIONS]

Options:
  --model-path <PATH>       Path to model file
  -f, --file <FILE>         Audio file for comparison
  -v, --verbose             Detailed output
```

### selftest

Verify installation and basic functionality:

```bash
whisper-apr selftest
```

## Supported Models

| Model | Size | Type | Notes |
|-------|------|------|-------|
| `tiny` | 39M | Whisper | Default, fastest |
| `base` | 74M | Whisper | Good balance |
| `small` | 244M | Whisper | Higher accuracy |
| `large-v3-turbo` | 809M | Whisper | 32 enc + 4 dec layers |
| `moonshine-tiny` | 27M | Moonshine | Ultra-lightweight |
| `moonshine-base` | 61M | Moonshine | Lightweight alternative |

## Supported Audio Formats

Via symphonia (pure Rust, no system dependencies):

- WAV (native, preferred)
- MP3
- FLAC
- OGG (Vorbis)
- AAC / M4A
- MKV / WebM

## Building from Source

```bash
# Build CLI
cargo build --features cli --release

# Run from source
cargo run --features cli -- transcribe -f audio.wav
```
