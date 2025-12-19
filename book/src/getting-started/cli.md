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
whisper-apr transcribe -f audio.wav

# With options
whisper-apr transcribe -f audio.mp3 \
  --model base \
  --language auto \
  --output-file transcript.srt \
  --format srt

# Translation (any language → English)
whisper-apr translate -f audio.wav --output-file english.txt

# Real-time recording + transcription
whisper-apr record --live

# Interactive TUI
whisper-apr tui
```

## Commands

### transcribe

Transcribe audio/video files to text:

```bash
whisper-apr transcribe -f input.wav [OPTIONS]

Options:
  -f, --file <FILE>         Input audio/video file (required)
  -m, --model <MODEL>       Model size [tiny|base|small|medium|large]
  --model-path <PATH>       Path to custom .apr model file
  -l, --language <LANG>     Source language (ISO 639-1) or 'auto'
  --output-file <FILE>      Output file path
  -o, --format <FORMAT>     Output format [txt|srt|vtt|json|csv|lrc|md]
  --timestamps              Include timestamps
  --word-timestamps         Word-level timestamps
  --vad                     Enable voice activity detection
  --vad-threshold <F32>     VAD threshold (0.0-1.0, default: 0.5)
  --gpu                     Use GPU acceleration
  -t, --threads <N>         Number of threads
  --beam-size <N>           Beam search size (-1 for greedy)
  --temperature <F32>       Sampling temperature
  --best-of <N>             Best-of candidates
  --hallucination-filter    Filter repeated hallucinations
  --translate               Translate to English
```

### translate

Translate speech from any language to English:

```bash
whisper-apr translate -f german.wav --output-file english.txt
```

### stream

Real-time streaming transcription from microphone:

```bash
whisper-apr stream [OPTIONS]

Options:
  --step <MS>         Step size in ms (default: 3000)
  --length <MS>       Audio length in ms (default: 10000)
  --keep <MS>         Audio to keep in ms (default: 200)
  --capture <ID>      Capture device ID
  --max-tokens <N>    Max tokens per segment (default: 32)
  --vad-thold <F32>   VAD threshold (default: 0.6)
  --keep-context      Keep previous context
  --save-audio        Save audio to file
  --translate         Translate to English
```

### serve

Start HTTP API server (whisper.cpp server compatibility):

```bash
whisper-apr serve [OPTIONS]

Options:
  --host <HOST>           Host address (default: 127.0.0.1)
  --port <PORT>           Port number (default: 8080)
  --public <PATH>         Static files directory
  --inference-path <PATH> Inference endpoint (default: /inference)
  -m, --model <MODEL>     Model size to use
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
whisper-apr batch --pattern "*.mp4" --output-dir ./transcripts --parallel 4
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

### validate

Validate .apr model file (25-point QA checklist):

```bash
whisper-apr validate model.apr [OPTIONS]

Options:
  --quick           Quick validation (skip detailed checks)
  --detailed        Show detailed breakdown
  --min-score <N>   Minimum passing score (default: 80)
  --format <FMT>    Output format [text|json]
```

### parity

Compare output against whisper.cpp (parity testing):

```bash
whisper-apr parity -f audio.wav [OPTIONS]

Options:
  --cpp-output <FILE>  whisper.cpp output for comparison
  --tolerance <F32>    WER tolerance (default: 0.01 = 1%)
  --timestamp-ms <N>   Timestamp tolerance in ms (default: 50)
```

### quantize

Quantize model to smaller size:

```bash
whisper-apr quantize input.bin output.apr [OPTIONS]

Options:
  -Q, --quantize <TYPE>  Quantization type [f32|f16|q8-0|q5-0|q4-0]
  -v, --verbose          Verbose output
```

### command

Voice command recognition (grammar-constrained):

```bash
whisper-apr command [OPTIONS]

Options:
  --grammar <FILE>   GBNF grammar file
  --commands <FILE>  Commands file
  -m, --model <MODEL> Model size
```

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
