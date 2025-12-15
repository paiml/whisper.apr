# Command Line Interface

whisper-apr provides a powerful CLI installable via Cargo:

```bash
cargo install whisper-apr
```

## Running from Source

```bash
# Build CLI
cargo build --features cli --bin whisper-apr-cli

# Run CLI
cargo run --features cli --bin whisper-apr-cli -- --help

# Run CLI example
cargo run --features cli --example cli_usage
```

## Quick Start

```bash
# Basic transcription
whisper-apr transcribe audio.wav

# With options
whisper-apr transcribe audio.mp3 \
  --model base \
  --language auto \
  --output transcript.srt \
  --format srt

# Translation (any language → English)
whisper-apr translate audio.wav --output english.txt

# Real-time recording + transcription
whisper-apr record --live

# Interactive TUI
whisper-apr tui
```

## Commands

### transcribe

Transcribe audio/video files to text:

```bash
whisper-apr transcribe input.wav [OPTIONS]

Options:
  -m, --model <MODEL>       Model size [tiny|base|small|medium|large]
  -l, --language <LANG>     Source language (ISO 639-1) or 'auto'
  -o, --output <FILE>       Output file path
  -f, --format <FORMAT>     Output format [txt|srt|vtt|json|csv|md]
  --timestamps              Include timestamps
  --word-timestamps         Word-level timestamps
  --vad                     Enable voice activity detection
  --gpu                     Use GPU acceleration
```

### translate

Translate speech from any language to English:

```bash
whisper-apr translate german.wav --output english.txt
```

### record

Record audio from microphone:

```bash
# Record to file
whisper-apr record --duration 30 --output recording.wav

# List audio devices
whisper-apr record --list-devices

# Real-time transcription
whisper-apr record --live
```

### batch

Process multiple files in parallel:

```bash
whisper-apr batch *.mp4 --output-dir ./transcripts --parallel 4
```

### tui

Launch interactive terminal UI:

```bash
whisper-apr tui
```

The TUI provides:
- Real-time progress visualization
- Timing breakdown (mel/encode/decode)
- Memory and GPU monitoring
- Interactive controls

### test

Run backend end-to-end tests:

```bash
# Test all backends
whisper-apr test --backend all

# Test specific backend
whisper-apr test --backend simd
whisper-apr test --backend wasm
whisper-apr test --backend cuda
```

### model

Manage models:

```bash
# List available models
whisper-apr model list

# Download model
whisper-apr model download tiny

# Convert PyTorch model to .apr format
whisper-apr model convert whisper-tiny.pt --output tiny.apr

# Show model information
whisper-apr model info tiny.apr
```

### benchmark

Performance benchmarking:

```bash
# Benchmark tiny model with SIMD backend
whisper-apr benchmark tiny --backend simd --iterations 5

# Benchmark with verbose output
whisper-apr benchmark base --iterations 3 -v
```

Output includes:
- Average, min, max processing times
- Real-Time Factor (RTF)

## Global Options

All commands support these global flags:

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Show timing info and debug output |
| `-q, --quiet` | Suppress non-essential output |
| `--json` | Output as JSON (machine-readable) |
| `--trace <FILE>` | Export performance trace (Chrome format) |
| `--no-color` | Disable colored output |

## Output Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| txt | .txt | Plain text |
| srt | .srt | Video subtitles |
| vtt | .vtt | Web subtitles |
| json | .json | Machine processing |
| csv | .csv | Spreadsheets |
| md | .md | Documentation |

## Supported Input Formats

- WAV (native, preferred)
- MP3
- FLAC
- OGG (Vorbis/Opus)
- MP4/M4A
- WebM
- MKV/AVI

## Performance Tips

1. **Use GPU when available**: `--gpu` flag enables wgpu acceleration
2. **Enable VAD for noisy audio**: `--vad` skips silence
3. **Batch processing**: Use `batch` command for multiple files
4. **Model selection**: Start with `tiny` for testing, `base` for production

## Environment Variables

```bash
# Model cache directory
export WHISPER_APR_CACHE=~/.cache/whisper-apr

# Default model
export WHISPER_APR_MODEL=base

# GPU preference
export WHISPER_APR_GPU=1
```

## Shell Completions

```bash
# Bash
whisper-apr completions bash > ~/.local/share/bash-completion/completions/whisper-apr

# Zsh
whisper-apr completions zsh > ~/.zfunc/_whisper-apr

# Fish
whisper-apr completions fish > ~/.config/fish/completions/whisper-apr.fish
```
