# whisper-apr CLI Parity Specification

**Version**: 1.4.0-draft
**Status**: Under Review
**Created**: 2025-12-18
**Methodology**: EXTREME TDD + Toyota Way + Popperian Falsification
**Target Coverage**: ≥95% line coverage
**Parity Reference**: whisper.cpp (ggerganov/whisper.cpp)

---

## Table of Contents

- [§1. Executive Summary](#1-executive-summary)
- [§2. Design Principles](#2-design-principles)
- [§3. Toyota Way Alignment](#3-toyota-way-alignment)
- [§4. CLI Architecture](#4-cli-architecture)
- [§5. Command Parity Matrix](#5-command-parity-matrix)
- [§6. Argument Parity Specification](#6-argument-parity-specification)
- [§7. Output Format Parity](#7-output-format-parity)
- [§8. Performance Parity Requirements](#8-performance-parity-requirements)
- [§9. EXTREME TDD Methodology](#9-extreme-tdd-methodology)
- [§10. Parity Testing Framework](#10-parity-testing-framework)
- [§11. Error Handling & Security](#11-error-handling-security)
- [§12. Quality Gates](#12-quality-gates)
- [§13. Peer-Reviewed Citations](#13-peer-reviewed-citations)
- [§14. References](#14-references)
- [§15. Ecosystem Dependencies](#15-ecosystem-dependencies)
- [§16. 100-Point CLI Transcription Falsification Checklist](#16-100-point-cli-transcription-falsification-checklist)

---

## §1. Executive Summary

This specification defines **complete CLI parity** between `whisper-apr` and `whisper.cpp`, ensuring:

1. **Argument-level compatibility** - All whisper.cpp CLI flags have whisper-apr equivalents
2. **Output format compatibility** - Identical TXT/SRT/VTT/JSON/CSV/LRC output
3. **Performance parity** - Real-time factor (RTF) within 10% of whisper.cpp on equivalent hardware
4. **Behavioral equivalence** - Same audio input produces semantically identical transcriptions

The specification follows the **aprender ecosystem** conventions with apr-cli patterns, **realizar-style parity testing**, and **Popperian falsification methodology** to scientifically verify claims.

---

## §2. Design Principles

### §2.1 Core Tenets

| Principle | Description | Citation |
|-----------|-------------|----------|
| **Testable Logic Separation** | All CLI logic resides in library (`src/cli/`), binary is thin shell | [1] Martin, Clean Architecture |
| **Popperian Falsifiability** | Every claim must be testable and disprovable | [2] Popper, Logic of Scientific Discovery |
| **Zero-Overhead Abstraction** | CLI wrapper adds <1% latency vs direct library call | [3] Stroustrup, C++ Design |
| **Fail-Fast Error Handling** | No silent failures; explicit error messages with exit codes | [4] Shore, Fail Fast |
| **Deterministic Reproducibility** | Same input + seed → identical output across runs | [5] Sculley, ML Systems |
| **Fail-Safe Defaults** | Security configuration defaults to highest safety (e.g., path restrictions) | [14] Saltzer & Schroeder, Protection |

### §2.2 Aprender Ecosystem Integration

```
┌────────────────────────────────────────────────────────────┐
│  whisper-apr CLI (this specification)                      │
├────────────────────────────────────────────────────────────┤
│  whisper-apr library (src/lib.rs)                          │
├────────────────────────────────────────────────────────────┤
│  aprender (.apr format) + trueno (tensor ops)              │
├────────────────────────────────────────────────────────────┤
│  realizar (inference) + pmat (quality gates)               │
└────────────────────────────────────────────────────────────┘
```

---

## §3. Toyota Way Alignment

This specification embodies the **14 Principles of the Toyota Production System** [6]:

| TPS Principle | CLI Implementation | Verification |
|---------------|-------------------|--------------|
| **Genchi Genbutsu** (Go and See) | Compare actual whisper.cpp output byte-for-byte | Parity tests §10 |
| **Jidoka** (Built-in Quality) | Compilation fails if parity tests fail | CI gates §12 |
| **Kaizen** (Continuous Improvement) | Performance regression alerts on every PR | Benchmarks §8 |
| **Heijunka** (Level Loading) | Batch processing distributes work evenly | §6.3 batch command |
| **Poka-Yoke** (Error Proofing) | Type-safe argument parsing prevents invalid states | clap derive macros |
| **Andon** (Visual Control) | Progress bars, color output, timing displays | §7 output formats |
| **Muda Elimination** (Waste Reduction) | Zero-copy audio processing where possible | §8 performance |
| **Standardized Work** | Every command follows identical dispatch pattern | §4 architecture |

### §3.1 Jidoka Quality Gates

```
STOP THE LINE if:
├── Parity test fails (any whisper.cpp divergence)
├── Coverage drops below 95%
├── RTF exceeds whisper.cpp by >10%
├── Any unwrap() call in src/cli/
└── SATD comment (TODO/FIXME/HACK) introduced
```

---

## §4. CLI Architecture

### §4.1 Testable Logic Separation (EXTREME TDD)

**Critical Requirement**: ALL logic MUST reside in the library, NOT the binary.

```
src/
├── bin/
│   └── whisper-apr-cli.rs    # THIN SHELL ONLY (~20 LOC)
│       │
│       └── fn main() {
│             cli::run(std::env::args()).exit()
│           }
│
└── cli/                       # ALL LOGIC HERE (testable)
    ├── mod.rs                # Module exports
    ├── args.rs               # Argument parsing (clap)
    ├── commands/             # Command implementations
    │   ├── mod.rs
    │   ├── transcribe.rs
    │   ├── translate.rs
    │   ├── stream.rs
    │   ├── server.rs
    │   ├── bench.rs
    │   ├── quantize.rs
    │   └── command.rs        # Voice command recognition
    ├── output.rs             # Output formatters
    ├── error.rs              # Error types with exit codes
    └── parity.rs             # Parity validation utilities
```

### §4.2 Binary Shell Pattern

```rust
// src/bin/whisper-apr-cli.rs - MAXIMUM 20 LINES
//! Thin shell binary for whisper-apr CLI.
//! ALL logic is in src/cli/ for testability.

fn main() {
    whisper_apr::cli::run(std::env::args()).exit()
}
```

### §4.3 Library Entry Point

```rust
// src/cli/mod.rs
pub fn run<I, T>(args: I) -> CliResult
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let args = Args::try_parse_from(args)?;
    dispatch(args)
}

pub struct CliResult {
    exit_code: i32,
    output: Option<String>,
}

impl CliResult {
    pub fn exit(self) -> ! {
        if let Some(output) = self.output {
            println!("{}", output);
        }
        std::process::exit(self.exit_code)
    }
}
```

---

## §5. Command Parity Matrix

Complete mapping from whisper.cpp commands to whisper-apr equivalents:

| whisper.cpp Binary | whisper-apr Command | Status | Notes |
|--------------------|---------------------|--------|-------|
| `whisper-cli` | `whisper-apr transcribe` | 🎯 Target | Primary transcription |
| `whisper-cli --translate` | `whisper-apr translate` | 🎯 Target | X→English translation |
| `whisper-server` | `whisper-apr serve` | 🎯 Target | HTTP API server |
| `whisper-stream` | `whisper-apr stream` | 🎯 Target | Real-time microphone |
| `whisper-command` | `whisper-apr command` | 🎯 Target | Voice command recognition |
| `whisper-bench` | `whisper-apr bench` | 🎯 Target | Performance benchmarking |
| `whisper-quantize` | `whisper-apr quantize` | 🎯 Target | Model quantization |
| N/A (new) | `whisper-apr batch` | ✅ Extension | Parallel batch processing |
| N/A (new) | `whisper-apr tui` | ✅ Extension | Interactive terminal UI |
| N/A (new) | `whisper-apr validate` | ✅ Extension | Model QA checklist |
| N/A (new) | `whisper-apr parity` | ✅ Extension | whisper.cpp comparison |
| N/A (new) | `whisper-apr test` | ✅ Extension | Backend E2E tests |
| N/A (new) | `whisper-apr model` | ✅ Extension | Model management (DL/Convert) |
| N/A (new) | `whisper-apr record` | ✅ Extension | Audio capture to file |

---

## §6. Argument Parity Specification

### §6.1 Global Arguments (All Commands)

Defined in the `Args` struct, these flags apply to the entire CLI application.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-v, --verbose` | `bool` | false | Show detailed timing and info |
| `-q, --quiet` | `bool` | false | Suppress non-essential output |
| `--json` | `bool` | false | Machine-readable output |
| `--trace <PATH>` | `PathBuf` | - | Export performance trace (Chrome format) |
| `--no-color` | `bool` | false | Disable ANSI color output |
| `-h, --help` | - | - | Show help |
| `-V, --version` | - | - | Show version |

### §6.2 Common Execution Arguments (Command-Specific)

These arguments are shared across most inference-related commands (`transcribe`, `translate`, `stream`, etc.) but are not global.

| whisper.cpp Flag | whisper-apr Flag | Type | Default | Description |
|------------------|------------------|------|---------|-------------|
| `-t, --threads` | `-t, --threads` | `u32` | auto | Thread count |
| `-p, --processors` | `-p, --processors` | `u32` | 1 | Processor count |
| `-m, --model` | `-m, --model` | `ModelSize` | tiny | Predefined model size |
| N/A | `--model-path` | `PathBuf` | - | Direct path to .apr file |
| `-ng, --no-gpu` | `--no-gpu` | `bool` | false | Disable GPU |
| `-fa, --flash-attn` | `--flash-attn` | `bool` | true | Flash attention |
| `-nfa, --no-flash-attn` | `--no-flash-attn` | `bool` | false | Disable flash attn |

### §6.3 Transcription Arguments (`whisper-apr transcribe`)

| whisper.cpp Flag | whisper-apr Flag | Type | Default | Description |
|------------------|------------------|------|---------|-------------|
| `-f, --file` | `-f, --file` | `PathBuf` | - | Input audio file |
| `-l, --language` | `-l, --language` | `String` | "auto" | Language code |
| `-dl, --detect-language` | `--detect-language` | `bool` | false | Detect and exit |
| `-tr, --translate` | `--translate` | `bool` | false | Translate to English |
| `-ot, --offset-t` | `--offset-t` | `u32` | 0 | Time offset (ms) |
| `-on, --offset-n` | `--offset-n` | `u32` | 0 | Segment offset |
| `-d, --duration` | `-d, --duration` | `u32` | 0 | Duration (ms) |
| `-mc, --max-context` | `--max-context` | `i32` | -1 | Max context tokens |
| `-ml, --max-len` | `--max-len` | `u32` | 0 | Max segment length |
| `-ac, --audio-ctx` | `--audio-ctx` | `u32` | 0 | Audio context size |
| `-bo, --best-of` | `--best-of` | `u32` | 2 | Best-of candidates |
| `-bs, --beam-size` | `--beam-size` | `i32` | -1 | Beam search size |
| `-sow, --split-on-word` | `--split-on-word` | `bool` | false | Word-level split |
| `-wt, --word-thold` | `--word-thold` | `f32` | 0.01 | Word threshold |
| `-et, --entropy-thold` | `--entropy-thold` | `f32` | 2.40 | Entropy threshold |
| `-lpt, --logprob-thold` | `--logprob-thold` | `f32` | -1.0 | Logprob threshold |
| `-nth, --no-speech-thold` | `--no-speech-thold` | `f32` | 0.6 | No-speech threshold |
| `-tp, --temperature` | `--temperature` | `f32` | 0.0 | Sampling temperature |
| `-tpi, --temperature-inc` | `--temperature-inc` | `f32` | 0.2 | Temperature increment |
| `-nf, --no-fallback` | `--no-fallback` | `bool` | false | No temp fallback |
| `--prompt` | `--prompt` | `String` | "" | Initial prompt |
| `--suppress-regex` | `--suppress-regex` | `String` | "" | Suppress pattern |
| `--grammar` | `--grammar` | `String` | "" | GBNF grammar |
| `--grammar-rule` | `--grammar-rule` | `String` | "" | Grammar rule |
| `--grammar-penalty` | `--grammar-penalty` | `f32` | 100.0 | Grammar penalty |

### §6.3 Output Format Arguments

| whisper.cpp Flag | whisper-apr Flag | Description |
|------------------|------------------|-------------|
| `-otxt, --output-txt` | `-o txt` / `--format txt` | Plain text |
| `-ovtt, --output-vtt` | `-o vtt` / `--format vtt` | WebVTT subtitles |
| `-osrt, --output-srt` | `-o srt` / `--format srt` | SRT subtitles |
| `-olrc, --output-lrc` | `-o lrc` / `--format lrc` | LRC lyrics |
| `-ocsv, --output-csv` | `-o csv` / `--format csv` | CSV format |
| `-oj, --output-json` | `-o json` / `--format json` | JSON output |
| `-ojf, --output-json-full` | `--format json-full` | Extended JSON |
| `-owts, --output-words` | `--format wts` | Karaoke script |
| `-of, --output-file` | `--output` | Output file path |

### §6.4 Display Arguments

| whisper.cpp Flag | whisper-apr Flag | Description |
|------------------|------------------|-------------|
| `-np, --no-prints` | `--no-prints` | Suppress transcription output |
| N/A | `-q, --quiet` | Global: Suppress all non-essential output |
| `-ps, --print-special` | `--print-special` | Show special tokens |
| `-pc, --print-colors` | `--colors` | Color-coded confidence |
| `--print-confidence` | `--confidence` | Show confidence |
| `-pp, --print-progress` | `--progress` | Progress percentage |
| `-nt, --no-timestamps` | `--no-timestamps` | Omit timestamps |
| N/A | `--word-timestamps`| Enable word-level timestamps |
| N/A | `--hallucination-filter` | Filter repeated hallucinations |

### §6.5 Voice Activity Detection (VAD) Arguments

| whisper.cpp Flag | whisper-apr Flag | Type | Default |
|------------------|------------------|------|---------|
| `--vad` | `--vad` | `bool` | false |
| `-vm, --vad-model` | `--vad-model` | `PathBuf` | - |
| `-vt, --vad-threshold` | `--vad-threshold` | `f32` | 0.5 |
| `-vspd, --vad-min-speech-duration-ms` | `--vad-min-speech-ms` | `u32` | 250 |
| `-vsd, --vad-min-silence-duration-ms` | `--vad-min-silence-ms` | `u32` | 100 |
| `-vmsd, --vad-max-speech-duration-s` | `--vad-max-speech-s` | `f32` | ∞ |
| `-vp, --vad-speech-pad-ms` | `--vad-pad-ms` | `u32` | 30 |
| `-vo, --vad-samples-overlap` | `--vad-overlap` | `f32` | 0.1 |

### §6.6 Server Arguments (`whisper-apr serve`)

| whisper.cpp Flag | whisper-apr Flag | Type | Default |
|------------------|------------------|------|---------|
| `--host` | `--host` | `String` | "127.0.0.1" |
| `--port` | `--port` | `u16` | 8080 |
| `--public` | `--public` | `PathBuf` | - |
| `--request-path` | `--request-path` | `String` | "" |
| `--inference-path` | `--inference-path` | `String` | "/inference" |
| `--convert` | `--convert` | `bool` | false |
| `--tmp-dir` | `--tmp-dir` | `PathBuf` | "." |

### §6.7 Stream Arguments (`whisper-apr stream`)

| whisper.cpp Flag | whisper-apr Flag | Type | Default |
|------------------|------------------|------|---------|
| `--step` | `--step` | `u32` | 3000 |
| `--length` | `--length` | `u32` | 10000 |
| `--keep` | `--keep` | `u32` | 200 |
| `-c, --capture` | `--capture` | `i32` | -1 |
| `-mt, --max-tokens` | `--max-tokens` | `u32` | 32 |
| `-vth, --vad-thold` | `--vad-thold` | `f32` | 0.6 |
| `-fth, --freq-thold` | `--freq-thold` | `f32` | 100.0 |
| `-kc, --keep-context` | `--keep-context` | `bool` | false |
| `-sa, --save-audio` | `--save-audio` | `bool` | false |
| `-t, --threads` | `-t, --threads` | `u32` | auto |
| `-tr, --translate` | `--translate` | `bool` | false |

### §6.8 Benchmark Arguments (`whisper-apr bench`)

| whisper.cpp Flag | whisper-apr Flag | Type | Default | Description |
|------------------|------------------|------|---------|-------------|
| `-w, --what` | `--what` | `u8` | 0 | What to benchmark |
| N/A | `--iterations` | `u32` | 3 | Number of iterations |
| N/A | `--backend` | `Backend` | simd | Backend to use |

### §6.9 Diarization Arguments (Planned)

Arguments for speaker diarization (who spoke when), following WAPR-150.

| whisper-apr Flag | Type | Default | Description |
|------------------|------|---------|-------------|
| `--diarize` | `bool` | false | Enable speaker diarization |
| `--min-speakers` | `u32` | 1 | Minimum number of speakers |
| `--max-speakers` | `u32` | - | Maximum number of speakers |
| `--diarize-thold` | `f32` | 0.5 | Clustering threshold |
| `--diarize-model` | `PathBuf` | - | Diarization embedding model |

### §6.10 Record Arguments (`whisper-apr record`)

Arguments for direct audio recording to file (WAPR-180).

| whisper-apr Flag | Type | Default | Description |
|------------------|------|---------|-------------|
| `-d, --duration` | `u32` | - | Recording duration (seconds) |
| `-o, --output` | `PathBuf` | - | Output WAV file path |
| `--device` | `String` | default | Audio input device ID/name |
| `--sample-rate` | `u32` | 16000 | Sample rate in Hz |
| `--live` | `bool` | false | Real-time transcription (preview) |
| `--list-devices` | `bool` | false | List available input devices |

---

## §7. Output Format Parity

### §7.1 Text Output (TXT)

**Requirement**: Byte-for-byte identical to whisper.cpp when normalized.

```
[00:00:00.000 --> 00:00:05.120]  Hello, world.
[00:00:05.120 --> 00:00:08.960]  This is a test.
```

### §7.2 SRT Output

```srt
1
00:00:00,000 --> 00:00:05,120
Hello, world.

2
00:00:05,120 --> 00:00:08,960
This is a test.
```

### §7.3 VTT Output

```vtt
WEBVTT

00:00:00.000 --> 00:00:05.120
Hello, world.

00:00:05.120 --> 00:00:08.960
This is a test.
```

### §7.4 JSON Output

```json
{
  "text": "Hello, world. This is a test.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.12,
      "text": "Hello, world.",
      "tokens": [50364, 2425, 11, 1002, 13, 50620],
      "avg_logprob": -0.234,
      "no_speech_prob": 0.012
    }
  ],
  "language": "en"
}
```

### §7.5 JSON-Full Output (Extended)

Includes token-level timing, word-level timestamps, and per-token probabilities.

### §7.6 CSV Output

```csv
start,end,speaker,text
0.000,5.120,SPEAKER_00,"Hello, world."
5.120,8.960,SPEAKER_00,"This is a test."
```

### §7.7 LRC Output

```lrc
[00:00.00]Hello, world.
[00:05.12]This is a test.
```

---

## §8. Performance Parity Requirements

### §8.1 Performance Targets

| Metric | whisper.cpp Baseline | whisper-apr Target | Tolerance |
|--------|---------------------|-------------------|-----------|
| **RTF (tiny)** | 0.5x | ≤0.55x | +10% |
| **RTF (base)** | 0.8x | ≤0.88x | +10% |
| **RTF (small)** | 1.5x | ≤1.65x | +10% |
| **RTF (medium)** | 3.0x | ≤3.3x | +10% |
| **RTF (large-v3)** | 5.0x | ≤5.5x | +10% |
| **Memory (tiny)** | 273 MB | ≤300 MB | +10% |
| **Memory (base)** | 388 MB | ≤427 MB | +10% |
| **Startup Time** | <500ms | <550ms | +10% |
| **First Token Latency** | <100ms | <110ms | +10% |

### §8.2 Benchmark Methodology

Following realizar's PARITY-114 pattern [7]:

```rust
#[derive(Debug)]
pub struct ParityBenchmark {
    /// whisper.cpp measurement
    pub cpp_rtf: f64,
    /// whisper-apr measurement
    pub apr_rtf: f64,
    /// Ratio (apr/cpp, should be ≤1.1)
    pub ratio: f64,
    /// PASS if ratio ≤ 1.1
    pub parity: bool,
}

impl ParityBenchmark {
    pub fn verify(&self) -> Result<(), ParityError> {
        if self.ratio > 1.1 {
            return Err(ParityError::PerformanceRegression {
                cpp: self.cpp_rtf,
                apr: self.apr_rtf,
                ratio: self.ratio,
            });
        }
        Ok(())
    }
}
```

### §8.3 Hardware Reference Configurations

| Configuration | CPU | GPU | RAM | Expected RTF (base) |
|--------------|-----|-----|-----|-------------------|
| M1 MacBook Air | Apple M1 | Metal | 8GB | 0.6x |
| Desktop (NVIDIA) | i7-12700K | RTX 3080 | 32GB | 0.4x |
| Desktop (AMD) | R9 5900X | RX 6800 | 32GB | 0.5x |
| Cloud (AWS) | c6i.xlarge | - | 8GB | 1.2x |
| Raspberry Pi 4 | BCM2711 | - | 8GB | 8.0x |

### §8.4 Ground Truth Benchmark Infrastructure

Following realizar's proven methodology for ground truth comparison [14]:

#### §8.4.1 Comparison Targets

| Ground Truth | Binary | Purpose | Priority |
|--------------|--------|---------|----------|
| **whisper.cpp** | `whisper-cli` | Primary reference (C++) | P0 - Required |
| **OpenAI Whisper** | `whisper` (Python) | Original implementation | P1 - Recommended |
| **HuggingFace Transformers** | `transformers` | Alternative reference | P2 - Optional |

#### §8.4.2 Side-by-Side Benchmark Script

**File**: `scripts/bench-ground-truth.sh`

```bash
#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# bashrs: compliant
# Ground Truth Benchmark: whisper-apr vs whisper.cpp
# Following realizar methodology (Hoefler & Belli SC'15)
#
# Description: Side-by-side performance comparison against whisper.cpp
# Usage: ./scripts/bench-ground-truth.sh [audio_file] [model_cpp] [model_apr]
# Dependencies: bc, whisper-cli (whisper.cpp), whisper-apr
set -euo pipefail

# Configuration
AUDIO_FILE="${1:-test_data/jfk.wav}"
MODEL_CPP="${2:-models/ggml-base.bin}"
MODEL_APR="${3:-models/whisper-base.apr}"
WARMUP_ITERATIONS=10
MIN_SAMPLES=30
MAX_SAMPLES=200
CV_THRESHOLD=0.05  # 5% coefficient of variation

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "═══════════════════════════════════════════════════════════"
echo "  Ground Truth Benchmark: whisper-apr vs whisper.cpp"
echo "═══════════════════════════════════════════════════════════"
echo "Audio: $AUDIO_FILE"
echo "Model (cpp): $MODEL_CPP"
echo "Model (apr): $MODEL_APR"
echo ""

# Function: Calculate statistics
calc_stats() {
    local -n arr=$1
    local n=${#arr[@]}

    # Mean
    local sum=0
    for val in "${arr[@]}"; do
        sum=$(echo "$sum + $val" | bc -l)
    done
    local mean=$(echo "scale=6; $sum / $n" | bc -l)

    # Standard deviation
    local sq_sum=0
    for val in "${arr[@]}"; do
        local diff=$(echo "$val - $mean" | bc -l)
        sq_sum=$(echo "$sq_sum + ($diff * $diff)" | bc -l)
    done
    local std=$(echo "scale=6; sqrt($sq_sum / ($n - 1))" | bc -l)

    # Coefficient of variation
    local cv=$(echo "scale=6; $std / $mean" | bc -l)

    # Percentiles (sorted)
    IFS=$'\n' sorted=($(sort -n <<<"${arr[*]}")); unset IFS
    local p50=${sorted[$((n / 2))]}
    local p95=${sorted[$((n * 95 / 100))]}
    local p99=${sorted[$((n * 99 / 100))]}

    echo "$mean $std $cv $p50 $p95 $p99"
}

# Function: Run benchmark with CV-based stopping
run_benchmark() {
    local cmd="$1"
    local name="$2"
    local -a latencies=()

    echo -n "  Warming up $name... "
    for ((i=1; i<=WARMUP_ITERATIONS; i++)); do
        eval "$cmd" > /dev/null 2>&1
    done
    echo "done"

    echo -n "  Benchmarking $name"
    local iteration=0
    while [[ $iteration -lt $MAX_SAMPLES ]]; do
        # Measure with nanosecond precision
        local start=$(date +%s%N)
        eval "$cmd" > /dev/null 2>&1
        local end=$(date +%s%N)
        local latency_ms=$(echo "scale=3; ($end - $start) / 1000000" | bc -l)
        latencies+=("$latency_ms")
        ((iteration++))

        # Check CV after MIN_SAMPLES
        if [[ $iteration -ge $MIN_SAMPLES ]]; then
            local stats=($(calc_stats latencies))
            local cv=${stats[2]}
            if (( $(echo "$cv < $CV_THRESHOLD" | bc -l) )); then
                echo " (CV=${cv}, n=${iteration})"
                break
            fi
        fi
        echo -n "."
    done

    # Final statistics
    local stats=($(calc_stats latencies))
    echo "${stats[@]}"
}

# Run whisper.cpp benchmark
echo ""
echo "Running whisper.cpp benchmark..."
CPP_CMD="./whisper-cli -m $MODEL_CPP -f $AUDIO_FILE --no-prints"
CPP_STATS=($(run_benchmark "$CPP_CMD" "whisper.cpp"))

# Run whisper-apr benchmark
echo ""
echo "Running whisper-apr benchmark..."
APR_CMD="whisper-apr transcribe -m $MODEL_APR -f $AUDIO_FILE --quiet"
APR_STATS=($(run_benchmark "$APR_CMD" "whisper-apr"))

# Calculate ratios
CPP_MEAN=${CPP_STATS[0]}
APR_MEAN=${APR_STATS[0]}
RATIO=$(echo "scale=4; $APR_MEAN / $CPP_MEAN" | bc -l)
SPEEDUP=$(echo "scale=4; $CPP_MEAN / $APR_MEAN" | bc -l)

# Results
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  RESULTS"
echo "═══════════════════════════════════════════════════════════"
printf "%-20s %12s %12s %12s\n" "Metric" "whisper.cpp" "whisper-apr" "Ratio"
printf "%-20s %12.3f %12.3f %12.4f\n" "Mean (ms)" "$CPP_MEAN" "$APR_MEAN" "$RATIO"
printf "%-20s %12.3f %12.3f\n" "Std Dev (ms)" "${CPP_STATS[1]}" "${APR_STATS[1]}"
printf "%-20s %12.3f %12.3f\n" "p50 (ms)" "${CPP_STATS[3]}" "${APR_STATS[3]}"
printf "%-20s %12.3f %12.3f\n" "p95 (ms)" "${CPP_STATS[4]}" "${APR_STATS[4]}"
printf "%-20s %12.3f %12.3f\n" "p99 (ms)" "${CPP_STATS[5]}" "${APR_STATS[5]}"

echo ""
# Parity check (Jidoka gate)
if (( $(echo "$RATIO <= 1.1" | bc -l) )); then
    echo -e "${GREEN}✓ PARITY ACHIEVED${NC}: whisper-apr is within 10% of whisper.cpp"
    echo "  Ratio: ${RATIO}x (target: ≤1.1x)"
    exit 0
else
    echo -e "${RED}✗ PARITY FAILED${NC}: whisper-apr exceeds 10% tolerance"
    echo "  Ratio: ${RATIO}x (target: ≤1.1x)"
    echo "  Regression: $(echo "scale=1; ($RATIO - 1.0) * 100" | bc -l)%"
    exit 1
fi
```

#### §8.4.3 Criterion.rs Benchmark Suite

**File**: `benches/ground_truth_parity.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::process::Command;
use std::time::Duration;

/// Ground truth parity benchmark comparing whisper-apr against whisper.cpp
fn ground_truth_benchmark(c: &mut Criterion) {
    let audio_file = "test_data/jfk.wav";
    let model_cpp = "models/ggml-base.bin";
    let model_apr = "models/whisper-base.apr";

    let mut group = c.benchmark_group("ground_truth_parity");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(60));
    group.confidence_level(0.95);

    // whisper.cpp baseline (ground truth)
    group.bench_function("whisper_cpp", |b| {
        b.iter(|| {
            Command::new("./whisper-cli")
                .args(["-m", model_cpp, "-f", audio_file, "--no-prints"])
                .output()
                .expect("whisper.cpp failed")
        })
    });

    // whisper-apr under test
    group.bench_function("whisper_apr", |b| {
        b.iter(|| {
            whisper_apr::transcribe(
                black_box(audio_file),
                black_box(model_apr),
                Default::default(),
            )
        })
    });

    group.finish();
}

/// Encoder-only benchmark (isolated component)
fn encoder_parity_benchmark(c: &mut Criterion) {
    // ... encoder-specific benchmarks
}

/// Memory usage comparison
fn memory_parity_benchmark(c: &mut Criterion) {
    // ... memory tracking benchmarks
}

criterion_group!(
    benches,
    ground_truth_benchmark,
    encoder_parity_benchmark,
    memory_parity_benchmark,
);
criterion_main!(benches);
```

### §8.5 Statistical Methodology

Following scientific benchmarking standards [14, 15]:

#### §8.5.1 Sample Size Determination

Power analysis for detecting 10% difference with 95% confidence:

```
n = 2 × (Z_α/2 + Z_β)² × (CV/δ)²
n = 2 × (1.96 + 0.84)² × (0.05/0.10)²
n ≈ 4 minimum → 100 used (25× safety margin)
```

#### §8.5.2 CV-Based Stopping Criterion

Per Hoefler & Belli SC'15 methodology [14]:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Warmup iterations | 10 | Cache warming, JIT stabilization |
| Minimum samples | 30 | Central limit theorem |
| Maximum samples | 200 | Practical time limit |
| CV threshold | 5% | Statistical stability |
| Confidence level | 95% | Industry standard |

```rust
/// CV-based stopping criterion
pub struct BenchmarkController {
    warmup: usize,
    min_samples: usize,
    max_samples: usize,
    cv_threshold: f64,
}

impl BenchmarkController {
    pub fn should_stop(&self, samples: &[f64]) -> bool {
        if samples.len() < self.min_samples {
            return false;
        }
        if samples.len() >= self.max_samples {
            return true;
        }

        let cv = coefficient_of_variation(samples);
        cv < self.cv_threshold
    }
}

fn coefficient_of_variation(samples: &[f64]) -> f64 {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (samples.len() - 1) as f64;
    variance.sqrt() / mean
}
```

#### §8.5.3 Statistical Tests

| Test | Purpose | Threshold | Citation |
|------|---------|-----------|----------|
| **Welch's t-test** | Significance (unequal variance) | p < 0.001 | [15] |
| **Cohen's d** | Effect size magnitude | Report all | [15] |
| **Mann-Whitney U** | Non-parametric robustness | p < 0.001 | [15] |
| **Bootstrap CI** | Confidence intervals | 95%, 10K resamples | [15] |

```rust
use statrs::distribution::{StudentsT, ContinuousCDF};

/// Welch's t-test for unequal variances
pub fn welchs_t_test(a: &[f64], b: &[f64]) -> TTestResult {
    let n1 = a.len() as f64;
    let n2 = b.len() as f64;
    let mean1 = mean(a);
    let mean2 = mean(b);
    let var1 = variance(a);
    let var2 = variance(b);

    let se = ((var1 / n1) + (var2 / n2)).sqrt();
    let t = (mean1 - mean2) / se;

    // Welch-Satterthwaite degrees of freedom
    let df = ((var1/n1 + var2/n2).powi(2)) /
             ((var1/n1).powi(2)/(n1-1.0) + (var2/n2).powi(2)/(n2-1.0));

    let dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let p_value = 2.0 * (1.0 - dist.cdf(t.abs()));

    TTestResult { t, df, p_value, significant: p_value < 0.001 }
}

/// Cohen's d effect size
pub fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let pooled_std = ((variance(a) + variance(b)) / 2.0).sqrt();
    (mean(a) - mean(b)).abs() / pooled_std
}

/// Effect size interpretation
pub fn interpret_cohens_d(d: f64) -> &'static str {
    match d {
        d if d < 0.2 => "negligible",
        d if d < 0.5 => "small",
        d if d < 0.8 => "medium",
        _ => "large",
    }
}
```

### §8.6 CI Integration (Jidoka Gates)

**Principle**: Stop the line on regression (Toyota Way Jidoka) [6].

#### §8.6.1 GitHub Actions Workflow

**File**: `.github/workflows/benchmark.yml`

```yaml
name: Ground Truth Parity Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Nightly at 2 AM UTC

env:
  CARGO_TERM_COLOR: always
  WHISPER_CPP_VERSION: "1.7.2"

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-action@stable

      - name: Cache whisper.cpp
        uses: actions/cache@v4
        with:
          path: ~/whisper.cpp
          key: whisper-cpp-${{ env.WHISPER_CPP_VERSION }}

      - name: Build whisper.cpp (ground truth)
        run: |
          if [ ! -d ~/whisper.cpp ]; then
            git clone --depth 1 --branch v${{ env.WHISPER_CPP_VERSION }} \
              https://github.com/ggerganov/whisper.cpp ~/whisper.cpp
            cd ~/whisper.cpp && make -j$(nproc)
          fi
          ln -sf ~/whisper.cpp/main ./whisper-cli

      - name: Download test models
        run: |
          ./scripts/download-models.sh ggml-base.bin
          ./scripts/convert-model.sh models/ggml-base.bin models/whisper-base.apr

      - name: Download baseline
        uses: actions/download-artifact@v4
        with:
          name: benchmark-baseline
          path: target/criterion-baseline/
        continue-on-error: true  # First run won't have baseline

      - name: Run ground truth benchmarks
        run: |
          cargo bench --bench ground_truth_parity -- \
            --save-baseline current \
            --baseline main 2>&1 | tee benchmark-output.txt

      - name: Check for regressions (Jidoka gate)
        run: |
          if grep -q "Performance has regressed" benchmark-output.txt; then
            REGRESSION=$(grep -oP 'regressed by \K[\d.]+%' benchmark-output.txt | head -1)
            if (( $(echo "$REGRESSION > 10" | bc -l) )); then
              echo "::error::JIDOKA STOP: Performance regression of ${REGRESSION} exceeds 10% threshold"
              exit 1
            fi
          fi
          echo "✓ Parity maintained within 10% tolerance"

      - name: Generate comparison report
        run: |
          cargo bench --bench ground_truth_parity -- --list > bench-list.txt
          python3 scripts/generate_benchmark_report.py \
            --baseline target/criterion-baseline \
            --current target/criterion \
            --output benchmark-report.md

      - name: Upload benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results-${{ github.sha }}
          path: |
            target/criterion/
            benchmark-report.md
            benchmark-output.txt
          retention-days: 30

      - name: Update baseline (main branch only)
        if: github.ref == 'refs/heads/main' && github.event_name == 'push'
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-baseline
          path: target/criterion/
          retention-days: 90

      - name: Post results to PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('benchmark-report.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '## Benchmark Results\n\n' + report
            });

  memory-benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Memory parity check
        run: |
          # Run with memory tracking
          /usr/bin/time -v ./whisper-cli -m models/ggml-base.bin \
            -f test_data/jfk.wav 2>&1 | grep "Maximum resident" > cpp-memory.txt

          /usr/bin/time -v whisper-apr transcribe -m models/whisper-base.apr \
            -f test_data/jfk.wav 2>&1 | grep "Maximum resident" > apr-memory.txt

          # Compare (Jidoka gate: ≤110% of whisper.cpp)
          CPP_MEM=$(grep -oP '\d+' cpp-memory.txt)
          APR_MEM=$(grep -oP '\d+' apr-memory.txt)
          RATIO=$(echo "scale=4; $APR_MEM / $CPP_MEM" | bc -l)

          if (( $(echo "$RATIO > 1.1" | bc -l) )); then
            echo "::error::Memory parity failed: ${RATIO}x (target: ≤1.1x)"
            exit 1
          fi
```

#### §8.6.2 Jidoka Decision Matrix

| Regression | Action | Notification |
|------------|--------|--------------|
| 0-5% | ✅ Pass | None |
| 5-10% | ⚠️ Warning | PR comment |
| >10% | 🛑 **STOP THE LINE** | Block merge, alert team |

### §8.7 Result Schema

#### §8.7.1 JSON Output Format

**File**: `benchmark-results.json`

```json
{
  "metadata": {
    "timestamp": "2025-12-18T14:30:00Z",
    "git_sha": "abc123def456",
    "git_branch": "main",
    "whisper_cpp_version": "1.7.2",
    "whisper_apr_version": "1.0.0",
    "platform": {
      "os": "Linux 6.8.0-90-generic",
      "cpu": "AMD Ryzen 9 5900X 12-Core",
      "gpu": "NVIDIA RTX 3080",
      "ram_gb": 32
    }
  },
  "config": {
    "audio_file": "test_data/jfk.wav",
    "audio_duration_sec": 11.0,
    "model": "base",
    "warmup_iterations": 10,
    "min_samples": 30,
    "max_samples": 200,
    "cv_threshold": 0.05
  },
  "results": {
    "whisper_cpp": {
      "samples": 47,
      "mean_ms": 892.34,
      "std_ms": 23.45,
      "cv": 0.026,
      "p50_ms": 889.12,
      "p95_ms": 934.56,
      "p99_ms": 952.78,
      "ci_95_lower": 885.67,
      "ci_95_upper": 899.01,
      "rtf": 0.081,
      "memory_peak_mb": 388
    },
    "whisper_apr": {
      "samples": 52,
      "mean_ms": 934.67,
      "std_ms": 28.91,
      "cv": 0.031,
      "p50_ms": 931.23,
      "p95_ms": 989.45,
      "p99_ms": 1012.34,
      "ci_95_lower": 926.78,
      "ci_95_upper": 942.56,
      "rtf": 0.085,
      "memory_peak_mb": 412
    }
  },
  "comparison": {
    "latency_ratio": 1.047,
    "memory_ratio": 1.062,
    "rtf_ratio": 1.049,
    "parity_achieved": true,
    "statistical_tests": {
      "welchs_t": {
        "t_statistic": 8.234,
        "df": 94.56,
        "p_value": 0.0000001,
        "significant": true
      },
      "cohens_d": {
        "value": 1.612,
        "interpretation": "large"
      },
      "mann_whitney_u": {
        "u_statistic": 892,
        "p_value": 0.0000003
      }
    }
  },
  "verdict": {
    "status": "PASS",
    "message": "Parity achieved: whisper-apr is within 10% of whisper.cpp",
    "latency_delta_pct": 4.7,
    "memory_delta_pct": 6.2
  }
}
```

#### §8.7.2 Rust Schema Types

```rust
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub metadata: Metadata,
    pub config: BenchmarkConfig,
    pub results: BenchmarkResults,
    pub comparison: Comparison,
    pub verdict: Verdict,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Metadata {
    pub timestamp: DateTime<Utc>,
    pub git_sha: String,
    pub git_branch: String,
    pub whisper_cpp_version: String,
    pub whisper_apr_version: String,
    pub platform: Platform,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Platform {
    pub os: String,
    pub cpu: String,
    pub gpu: Option<String>,
    pub ram_gb: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub audio_file: String,
    pub audio_duration_sec: f64,
    pub model: String,
    pub warmup_iterations: usize,
    pub min_samples: usize,
    pub max_samples: usize,
    pub cv_threshold: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Metrics {
    pub samples: usize,
    pub mean_ms: f64,
    pub std_ms: f64,
    pub cv: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub ci_95_lower: f64,
    pub ci_95_upper: f64,
    pub rtf: f64,
    pub memory_peak_mb: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Comparison {
    pub latency_ratio: f64,
    pub memory_ratio: f64,
    pub rtf_ratio: f64,
    pub parity_achieved: bool,
    pub statistical_tests: StatisticalTests,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Verdict {
    pub status: VerdictStatus,
    pub message: String,
    pub latency_delta_pct: f64,
    pub memory_delta_pct: f64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum VerdictStatus {
    Pass,
    Warning,
    Fail,
}
```

### §8.8 Baseline Management

#### §8.8.1 Artifact Retention Policy

| Artifact Type | Retention | Purpose |
|---------------|-----------|---------|
| **Benchmark baseline** | 90 days | Historical comparison |
| **PR results** | 30 days | Review and debugging |
| **Nightly results** | 14 days | Trend analysis |
| **Release results** | Permanent | Version comparison |

#### §8.8.2 Time-Series Tracking

```rust
/// Baseline manager for historical comparison
pub struct BaselineManager {
    storage_path: PathBuf,
    retention_days: u32,
}

impl BaselineManager {
    /// Load baseline for comparison
    pub fn load_baseline(&self, branch: &str) -> Option<BenchmarkReport> {
        let path = self.storage_path.join(format!("{}-baseline.json", branch));
        if path.exists() {
            let content = std::fs::read_to_string(&path).ok()?;
            serde_json::from_str(&content).ok()
        } else {
            None
        }
    }

    /// Save new baseline (only on main branch)
    pub fn save_baseline(&self, report: &BenchmarkReport) -> Result<(), Error> {
        let path = self.storage_path.join("main-baseline.json");
        let content = serde_json::to_string_pretty(report)?;
        std::fs::write(&path, content)?;
        Ok(())
    }

    /// Compare against baseline and detect regressions
    pub fn compare(&self, current: &BenchmarkReport, baseline: &BenchmarkReport)
        -> ComparisonResult
    {
        let latency_delta = (current.results.whisper_apr.mean_ms
                           / baseline.results.whisper_apr.mean_ms) - 1.0;
        let memory_delta = (current.results.whisper_apr.memory_peak_mb as f64
                          / baseline.results.whisper_apr.memory_peak_mb as f64) - 1.0;

        ComparisonResult {
            latency_regression_pct: latency_delta * 100.0,
            memory_regression_pct: memory_delta * 100.0,
            jidoka_triggered: latency_delta > 0.10 || memory_delta > 0.10,
        }
    }

    /// Cleanup old baselines
    pub fn cleanup(&self) -> Result<(), Error> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(self.retention_days as i64);
        // ... cleanup logic
        Ok(())
    }
}
```

#### §8.8.3 Regression Trend Visualization

**File**: `scripts/plot_benchmark_history.py`

```python
#!/usr/bin/env python3
"""Generate benchmark trend visualization."""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def load_history(results_dir: Path) -> list[dict]:
    """Load all benchmark results sorted by timestamp."""
    results = []
    for file in results_dir.glob("benchmark-results-*.json"):
        with open(file) as f:
            results.append(json.load(f))
    return sorted(results, key=lambda r: r["metadata"]["timestamp"])

def plot_parity_trend(history: list[dict], output: Path):
    """Plot latency ratio trend over time."""
    timestamps = [datetime.fromisoformat(r["metadata"]["timestamp"]) for r in history]
    ratios = [r["comparison"]["latency_ratio"] for r in history]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timestamps, ratios, 'b-o', label='Latency Ratio (apr/cpp)')
    ax.axhline(y=1.0, color='g', linestyle='--', label='Perfect Parity')
    ax.axhline(y=1.1, color='r', linestyle='--', label='10% Threshold')
    ax.fill_between(timestamps, 1.0, 1.1, alpha=0.2, color='yellow', label='Acceptable Range')

    ax.set_xlabel('Date')
    ax.set_ylabel('Latency Ratio')
    ax.set_title('Ground Truth Parity Trend: whisper-apr vs whisper.cpp')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved trend plot to {output}")

if __name__ == "__main__":
    history = load_history(Path("benchmark-history/"))
    plot_parity_trend(history, Path("benchmark-trend.png"))
```

### §8.9 Model Optimization Verification Standards

Following the methodology of Hinton [15] and Jacob [16], model optimizations (quantization, distillation, pruning) must pass specific verification gates:

| Optimization | Metric | Target Threshold | Validation Method |
|--------------|--------|------------------|-------------------|
| **Quantization (Int8)** | Accuracy | WER ≤ FP16 + 1.0% | `whisper-apr parity --quantized` |
| **Quantization (Int8)** | Speedup | RTF ≤ FP16 × 0.8 | `whisper-apr bench --quantized` |
| **Quantization (Int8)** | Memory | RAM ≤ FP16 × 0.6 | Peak RSS measurement |
| **Distillation** | Semantic | Cosine Sim ≥ 0.95 | Encoder embedding comparison |
| **Distillation** | Accuracy | WER ≤ Teacher + 2.0% | Student vs Teacher transcription |
| **Pruning** | Sparsity | Actual Zeros ≥ Target | Tensor inspection |

---

## §9. EXTREME TDD Methodology

### §9.1 Test Pyramid

```
                    ┌─────────────┐
                    │ E2E Parity  │  10% - whisper.cpp comparison
                    │    Tests    │
                    ├─────────────┤
                    │ Integration │  20% - CLI command tests
                    │    Tests    │
                    ├─────────────┤
                    │  Property   │  30% - Fuzzing, edge cases
                    │   Tests     │
                    ├─────────────┤
                    │    Unit     │  40% - Function-level
                    │   Tests     │
                    └─────────────┘
```

### §9.2 Coverage Requirements

| Metric | Target | Enforcement |
|--------|--------|-------------|
| Line Coverage | ≥95% | CI gate (cargo-llvm-cov) |
| Branch Coverage | ≥90% | CI gate |
| Function Coverage | ≥98% | CI gate |
| Mutation Score | ≥85% | CI gate (cargo-mutants) |

### §9.3 Test Categories

#### §9.3.1 Unit Tests (src/cli/*.rs)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_language_code() {
        assert_eq!(parse_language("en"), Ok(Language::English));
        assert_eq!(parse_language("auto"), Ok(Language::Auto));
        assert!(parse_language("invalid").is_err());
    }

    #[test]
    fn test_output_format_from_extension() {
        assert_eq!(OutputFormat::from_path("out.srt"), OutputFormat::Srt);
        assert_eq!(OutputFormat::from_path("out.vtt"), OutputFormat::Vtt);
    }
}
```

#### §9.3.2 Property Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_timestamp_roundtrip(ms in 0u64..86400000) {
        let ts = Timestamp::from_millis(ms);
        let formatted = ts.to_string();
        let parsed = Timestamp::parse(&formatted).unwrap();
        prop_assert_eq!(ts, parsed);
    }

    #[test]
    fn prop_args_valid_combinations(
        threads in 1u32..128,
        beam_size in -1i32..16,
        temperature in 0.0f32..1.0,
    ) {
        let args = TranscribeArgs {
            threads,
            beam_size,
            temperature,
            ..Default::default()
        };
        prop_assert!(args.validate().is_ok());
    }
}
```

#### §9.3.3 Integration Tests

```rust
#[test]
fn test_cli_transcribe_wav() {
    let result = cli::run(&[
        "whisper-apr", "transcribe",
        "-m", "models/ggml-tiny.bin",
        "-f", "test_data/jfk.wav",
        "--format", "txt",
    ]);

    assert!(result.is_ok());
    assert!(result.output().contains("ask not what your country"));
}
```

#### §9.3.4 Parity Tests

```rust
#[test]
fn test_parity_with_whisper_cpp() {
    let cpp_output = run_whisper_cpp(&["--model", "tiny", "-f", "jfk.wav"]);
    let apr_output = cli::run(&[
        "whisper-apr", "transcribe",
        "-m", "models/ggml-tiny.bin",
        "-f", "test_data/jfk.wav",
    ]);

    // Normalize whitespace and compare
    let cpp_text = normalize_text(&cpp_output);
    let apr_text = normalize_text(&apr_output);

    assert_eq!(cpp_text, apr_text, "Text parity failed");
}
```

### §9.4 RED-GREEN-REFACTOR Cycle

```
┌─────────────────────────────────────────────────────────┐
│  1. RED: Write failing test for new feature             │
│     → Test MUST fail initially                          │
│     → Test must be specific and falsifiable            │
├─────────────────────────────────────────────────────────┤
│  2. GREEN: Minimal implementation to pass               │
│     → No gold-plating                                   │
│     → Just enough code to make test green              │
├─────────────────────────────────────────────────────────┤
│  3. REFACTOR: Clean up while tests stay green           │
│     → Extract abstractions                              │
│     → Improve readability                               │
│     → Maintain 100% test passage                        │
└─────────────────────────────────────────────────────────┘
```

---

## §10. Parity Testing Framework

### §10.1 Three-Way Comparison (realizar pattern)

Following realizar's PARITY-114 methodology [7]:

```rust
pub struct ParityTest {
    /// Input audio file
    pub input: PathBuf,
    /// whisper.cpp output (ground truth)
    pub cpp_output: String,
    /// whisper-apr output (under test)
    pub apr_output: String,
    /// HuggingFace Transformers output (reference)
    pub hf_output: Option<String>,
}

impl ParityTest {
    /// FALSIFIABLE: Outputs must match within tolerance
    pub fn verify_text_parity(&self) -> ParityResult {
        let cpp_normalized = normalize(&self.cpp_output);
        let apr_normalized = normalize(&self.apr_output);

        // Word Error Rate must be < 1%
        let wer = calculate_wer(&cpp_normalized, &apr_normalized);

        if wer > 0.01 {
            ParityResult::Fail {
                wer,
                cpp: cpp_normalized,
                apr: apr_normalized,
            }
        } else {
            ParityResult::Pass { wer }
        }
    }

    /// FALSIFIABLE: Timestamps must match within 50ms
    pub fn verify_timestamp_parity(&self) -> ParityResult {
        let cpp_segments = parse_segments(&self.cpp_output);
        let apr_segments = parse_segments(&self.apr_output);

        for (cpp, apr) in cpp_segments.iter().zip(apr_segments.iter()) {
            let start_diff = (cpp.start - apr.start).abs();
            let end_diff = (cpp.end - apr.end).abs();

            if start_diff > 0.050 || end_diff > 0.050 {
                return ParityResult::Fail {
                    message: format!(
                        "Timestamp mismatch: cpp={:?}, apr={:?}",
                        cpp, apr
                    ),
                };
            }
        }

        ParityResult::Pass { tolerance_ms: 50 }
    }
}
```

### §10.2 Parity Test Matrix

| Test Case | Audio | Model | whisper.cpp | whisper-apr | Verification |
|-----------|-------|-------|-------------|-------------|--------------|
| PARITY-001 | jfk.wav | tiny | ✓ | ✓ | Text exact |
| PARITY-002 | jfk.wav | base | ✓ | ✓ | Text exact |
| PARITY-003 | jfk.wav | small | ✓ | ✓ | Text exact |
| PARITY-010 | multi-speaker.wav | tiny | ✓ | ✓ | Diarization |
| PARITY-020 | noisy.wav | base | ✓ | ✓ | VAD behavior |
| PARITY-030 | long-form.wav | small | ✓ | ✓ | Chunking |
| PARITY-040 | multilingual.wav | large | ✓ | ✓ | Language detect |
| PARITY-050 | silence.wav | tiny | ✓ | ✓ | Empty handling |
| PARITY-060 | 8khz.wav | base | ✓ | ✓ | Resampling |
| PARITY-070 | stereo.wav | tiny | ✓ | ✓ | Channel mixing |

### §10.3 Ratio Analysis for Failure Diagnosis

Following realizar's ratio analysis pattern:

| Ratio (apr/cpp) | Diagnosis | Root Cause |
|-----------------|-----------|------------|
| 1.0x | ✓ Pass | Parity achieved |
| 2.0x | Half iterations | Loop counter bug |
| 4.0x | Quarter tiles | Tile accumulation bug |
| 8.0x | Accumulator | Reduction bug |
| Variable | State leak | Buffer not cleared |

---

## §11. 100-Point Popperian Falsification Checklist

### Scoring Methodology

Following Popper's falsificationism [2]: each check is designed to **disprove** a claim. Surviving checks indicate verified functionality.

**Grading Scale**:
- 95-100 points: A+ (Production Ready)
- 90-94 points: A (Release Candidate)
- 85-89 points: B (Beta Quality)
- 80-84 points: C (Alpha Quality)
- <80 points: F (Not Ready)

---

### Section A: Argument Parsing (15 points)

| # | Check | Command/Action | Expected | Pass | Fail |
|---|-------|----------------|----------|------|------|
| A.1 | `--help` displays all options | `whisper-apr --help` | All flags listed | [x] | [ ] |
| A.2 | `-h` short form works | `whisper-apr -h` | Same as --help | [x] | [ ] |
| A.3 | `--version` shows version | `whisper-apr --version` | Semver format | [x] | [ ] |
| A.4 | Unknown flag errors | `whisper-apr --invalid` | Exit code ≠ 0 | [x] | [ ] |
| A.5 | Missing required arg errors | `whisper-apr transcribe` | Error: -m required | [x] | [ ] |
| A.6 | Invalid type rejected | `--threads abc` | Type error | [x] | [ ] |
| A.7 | Negative threads rejected | `--threads -1` | Range error | [x] | [ ] |
| A.8 | Temperature range validated | `--temperature 2.0` | Range error | [x] | [ ] |
| A.9 | Model file not found | `-m nonexistent.bin` | File error | [x] | [ ] |
| A.10 | Audio file not found | `-f nonexistent.wav` | File error | [x] | [ ] |
| A.11 | Response file works | `@args.txt` | Args from file | [ ] | [x] |
| A.12 | Conflicting flags error | `--quiet --verbose` | Conflict error | [x] | [ ] |
| A.13 | Language code validated | `-l invalid` | Language error | [ ] | [ ] |
| A.14 | Output format validated | `--format xyz` | Format error | [ ] | [ ] |
| A.15 | Multiple files accepted | `-f a.wav -f b.wav` | Both processed | [ ] | [ ] |

**Status (2025-12-19):**
- **A.8**: ✅ Fixed - Temperature now validated (0.0-1.0 range).
- **A.9**: ✅ Fixed - File existence checked at runtime.
- **A.11**: ❌ Not Implemented - Response file `@args.txt` support deferred.
- **A.12**: ✅ Fixed - `--quiet` and `--verbose` conflict enforced via clap.

**Remaining:**
- **A.11**: Response file support not implemented (low priority).

---

### Section B: Core Transcription (20 points)

| # | Check | Command/Action | Expected | Pass | Fail |
|---|-------|----------------|----------|------|------|
| B.1 | WAV 16kHz mono works | Standard test file | Transcription | [ ] | [ ] |
| B.2 | WAV 44.1kHz resampled | Higher sample rate | Correct resample | [ ] | [ ] |
| B.3 | WAV 8kHz upsampled | Lower sample rate | Correct upsample | [ ] | [ ] |
| B.4 | WAV stereo mixed | Two channels | Mono mixdown | [ ] | [ ] |
| B.5 | FLAC input works | .flac file | Transcription | [ ] | [ ] |
| B.6 | MP3 input works | .mp3 file | Transcription | [ ] | [ ] |
| B.7 | OGG input works | .ogg file | Transcription | [ ] | [ ] |
| B.8 | Empty audio handled | 0 samples | Empty output | [ ] | [ ] |
| B.9 | Silent audio handled | All zeros | No speech detected | [ ] | [ ] |
| B.10 | Very short audio (<1s) | Brief audio | Correct handling | [ ] | [ ] |
| B.11 | Long audio (>10min) | Extended audio | Chunked correctly | [ ] | [ ] |
| B.12 | Unicode text output | Non-ASCII speech | Correct encoding | [ ] | [ ] |
| B.13 | Punctuation preserved | Speech with pauses | Periods, commas | [ ] | [ ] |
| B.14 | Numbers transcribed | "One two three" | As spoken | [ ] | [ ] |
| B.15 | Language auto-detect | Multi-language | Correct detection | [ ] | [ ] |
| B.16 | Timestamp accuracy | Known segments | ±50ms of reference | [ ] | [ ] |
| B.17 | Word-level timestamps | `--dtw` flag | Per-word timing | [ ] | [ ] |
| B.18 | Greedy decoding | Default strategy | Deterministic | [ ] | [ ] |
| B.19 | Beam search works | `--beam-size 5` | Better quality | [ ] | [ ] |
| B.20 | Temperature sampling | `--temperature 0.5` | Varied output | [ ] | [ ] |

---

### Section C: Output Formats (10 points)

| # | Check | Command/Action | Expected | Pass | Fail |
|---|-------|----------------|----------|------|------|
| C.1 | TXT/SRT/VTT valid | `--format txt,srt,vtt` | All 3 valid | [ ] | [ ] |
| C.2 | JSON/JSON-Full valid | `--format json,json-full` | Valid schemas | [ ] | [ ] |
| C.3 | CSV/LRC valid | `--format csv,lrc` | Valid structures | [ ] | [ ] |
| C.4 | Output file creation | `--output out.txt` | File created | [ ] | [ ] |
| C.5 | Stdout fallback | No --output flag | Prints to stdout | [ ] | [ ] |
| C.6 | Multiple formats | `--format txt,json` | Both created | [ ] | [ ] |
| C.7 | Auto-extension | `--output out` | .txt added | [ ] | [ ] |
| C.8 | Overwrite protection | Existing file | Overwrites/Fails | [ ] | [ ] |
| C.9 | UTF-8 Correctness | Unicode output | Valid UTF-8 | [ ] | [ ] |
| C.10 | Line endings | Platform check | CRLF/LF correct | [ ] | [ ] |

---

### Section D: whisper.cpp Parity (20 points)

| # | Check | Command/Action | Expected | Pass | Fail |
|---|-------|----------------|----------|------|------|
| D.1 | Text parity (tiny) | jfk.wav, tiny model | WER < 1% | [ ] | [ ] |
| D.2 | Text parity (base) | jfk.wav, base model | WER < 1% | [ ] | [ ] |
| D.3 | Text parity (small) | jfk.wav, small model | WER < 1% | [ ] | [ ] |
| D.4 | Timestamp parity | Same inputs | ±50ms | [ ] | [ ] |
| D.5 | SRT output identical | Compare SRT files | Byte-equal* | [ ] | [ ] |
| D.6 | VTT output identical | Compare VTT files | Byte-equal* | [ ] | [ ] |
| D.7 | JSON structure match | Compare JSON | Schema match | [ ] | [ ] |
| D.8 | Language detection match | Auto-detect | Same language | [ ] | [ ] |
| D.9 | VAD behavior match | --vad flag | Same segments | [ ] | [ ] |
| D.10 | Translate output match | --translate | Same English | [ ] | [ ] |
| D.11 | Beam search match | --beam-size 5 | Same output | [ ] | [ ] |
| D.12 | Temperature fallback | Complex audio | Same behavior | [ ] | [ ] |
| D.13 | Prompt handling match | --prompt "..." | Same effect | [ ] | [ ] |
| D.14 | Special tokens match | --print-special | Same tokens | [ ] | [ ] |
| D.15 | No-speech detection | Silent segments | Same behavior | [ ] | [ ] |
| D.16 | Offset handling match | --offset-t 5000 | Same start | [ ] | [ ] |
| D.17 | Duration handling match | --duration 10000 | Same length | [ ] | [ ] |
| D.18 | Max-context match | --max-context 128 | Same context | [ ] | [ ] |
| D.19 | Grammar support match | --grammar "..." | Same constraint | [ ] | [ ] |
| D.20 | Diarization parity | --diarize | Same speakers | [ ] | [ ] |

*After whitespace normalization

---

### Section E: Performance (15 points)

| # | Check | Command/Action | Expected | Pass | Fail |
|---|-------|----------------|----------|------|------|
| E.1 | RTF ≤1.1× (tiny) | Benchmark | ratio ≤ 1.1 | [ ] | [ ] |
| E.2 | RTF ≤1.1× (base) | Benchmark | ratio ≤ 1.1 | [ ] | [ ] |
| E.3 | RTF ≤1.1× (small) | Benchmark | ratio ≤ 1.1 | [ ] | [ ] |
| E.4 | Memory ≤1.1× (tiny) | Peak memory | ratio ≤ 1.1 | [ ] | [ ] |
| E.5 | Memory ≤1.1× (base) | Peak memory | ratio ≤ 1.1 | [ ] | [ ] |
| E.6 | Startup <550ms | Cold start | <550ms | [ ] | [ ] |
| E.7 | First token <110ms | Time to first | <110ms | [ ] | [ ] |
| E.8 | GPU acceleration works | --gpu | Faster than CPU | [ ] | [ ] |
| E.9 | Multi-thread scaling | -t 1 vs -t 4 | Near 4× | [ ] | [ ] |
| E.10 | Batch efficiency | Multiple files | Linear or better | [ ] | [ ] |
| E.11 | Memory stable | Long-running | No leaks | [ ] | [ ] |
| E.12 | No regression vs v1 | Benchmark history | ≤5% slower | [ ] | [ ] |
| E.13 | SIMD utilized | CPU features | AVX2/NEON used | [ ] | [ ] |
| E.14 | Flash attention works | --flash-attn | Memory reduced | [ ] | [ ] |
| E.15 | Quantized model speed | int8 model | Faster inference | [ ] | [ ] |

---

### Section F: Error Handling & Security (15 points)

| # | Check | Command/Action | Expected | Pass | Fail |
|---|-------|----------------|----------|------|------|
| F.1 | Corrupted audio handled | Invalid WAV | Graceful error | [ ] | [ ] |
| F.2 | Corrupted model handled | Invalid .bin | Graceful error | [ ] | [ ] |
| F.3 | OOM handled | Huge file, low mem | Graceful error | [ ] | [ ] |
| F.4 | Ctrl+C handled | Interrupt | Clean exit | [ ] | [ ] |
| F.5 | Disk full handled | No space | Graceful error | [ ] | [ ] |
| F.6 | Permission denied | No read access | Graceful error | [ ] | [ ] |
| F.7 | Exit codes correct | Various errors | Distinct codes | [ ] | [ ] |
| F.8 | Error messages helpful | Any error | Actionable message | [ ] | [ ] |
| F.9 | No panics in library | Fuzz testing | No panics | [ ] | [ ] |
| F.10 | No unwrap() calls | Code review | Zero unwrap | [ ] | [ ] |
| F.11 | Path traversal protection | `-f ../../../etc/passwd` | Access denied/Safe | [ ] | [ ] |
| F.12 | Large input resilience | 10GB dummy wav | No crash/DoS | [ ] | [ ] |
| F.13 | Recursive symlinks | Cyclic symlink | No infinite loop | [ ] | [ ] |
| F.14 | Argument fuzzing | Random bytes args | Safe parsing | [ ] | [ ] |
| F.15 | Memory limit enforcement | Cgroup constrained | OOM / Fail safe | [ ] | [ ] |

---

### Section G: Advanced Features (5 points)

| # | Check | Command/Action | Expected | Pass | Fail |
|---|-------|----------------|----------|------|------|
| G.1 | Server mode works | whisper-apr serve | HTTP 200 on /health | [ ] | [ ] |
| G.2 | Stream mode works | whisper-apr stream | Real-time output | [ ] | [ ] |
| G.3 | TUI mode works | whisper-apr tui | Interactive UI | [ ] | [ ] |
| G.4 | Batch mode works | whisper-apr batch | Parallel processing | [ ] | [ ] |
| G.5 | Voice commands work | whisper-apr command | Grammar constraint | [ ] | [ ] |

---

### Section H: Model Optimization Verification (10 points)

| # | Check | Command/Action | Expected | Pass | Fail |
|---|-------|----------------|----------|------|------|
| H.1 | Int8 Quantization Load | Load q8_0 model | Successful load | [ ] | [ ] |
| H.2 | Int8 Accuracy | `whisper-apr parity --quantized` | WER ≤ FP16 + 1% | [ ] | [ ] |
| H.3 | Int8 Speedup | `whisper-apr bench --quantized` | RTF ≤ 0.8× FP16 | [ ] | [ ] |
| H.4 | Int8 Memory | Check peak RSS | RAM ≤ 0.6× FP16 | [ ] | [ ] |
| H.5 | Distilled Load | Load distilled model | Successful load | [ ] | [ ] |
| H.6 | Distilled Semantic | Embedding comparison | Cosine Sim ≥ 0.95 | [ ] | [ ] |
| H.7 | Pruned Model Load | Load sparse model | Successful load | [ ] | [ ] |
| H.8 | Sparsity Verification | Inspect tensors | Zeros match target | [ ] | [ ] |
| H.9 | Mixed Precision | FP16/Int8 mix | Correct dispatch | [ ] | [ ] |
| H.10 | Export Formats | Export to GGUF/APR | Valid output | [ ] | [ ] |

---

### Scoring Summary

| Section | Points Available | Points Earned | Percentage |
|---------|-----------------|---------------|------------|
| A: Argument Parsing | 15 | __ | __% |
| B: Core Transcription | 20 | __ | __% |
| C: Output Formats | 10 | __ | __% |
| D: whisper.cpp Parity | 20 | __ | __% |
| E: Performance | 15 | __ | __% |
| F: Error Handling & Security | 15 | __ | __% |
| G: Advanced Features | 5 | __ | __% |
| H: Model Optimization | 10 | __ | __% |
| **TOTAL** | **110** | **__** | **__%** |

**Grade**: ___

---

## §12. Quality Gates

### §12.1 Tiered Enforcement (Certeza Methodology)

| Tier | Trigger | Checks | Max Time |
|------|---------|--------|----------|
| **Tier 1** | On save | `cargo check`, `cargo fmt --check`, `bashrs lint scripts/` | <1s |
| **Tier 2** | Pre-commit | + `cargo clippy`, `cargo test --lib`, `bashrs purify scripts/` | <5s |
| **Tier 3** | Pre-push | + Full tests, coverage ≥95%, PMAT, `bashrs analyze scripts/` | 1-5 min |
| **Tier 4** | CI/CD | + Mutation ≥85%, parity tests, benchmarks, bashrs determinism | 5-60 min |

### §12.2 PMAT Configuration

```toml
# .pmat-metrics.toml
[quality_gates]
min_coverage_pct = 95.0
min_mutation_score_pct = 85.0
max_cyclomatic_complexity = 10
max_unwrap_calls = 0
max_satd_comments = 0

[parity]
max_wer = 0.01              # 1% word error rate
max_timestamp_drift_ms = 50
max_rtf_ratio = 1.1         # 10% performance tolerance

[bashrs]
enabled = true
strict_mode = true          # Enforce set -euo pipefail
quote_variables = true      # All variables must be quoted
determinism_check = true    # Scripts must be idempotent
max_complexity = 15         # Max cyclomatic complexity per function
```

### §12.3 Bashrs Shell Script Quality

**All shell scripts MUST pass bashrs validation** (NOT shellcheck).

#### §12.3.1 Bashrs Requirements

| Requirement | Enforcement | Rationale |
|-------------|-------------|-----------|
| **Strict mode** | `set -euo pipefail` | Fail-fast on errors |
| **Quoted variables** | `"$VAR"` not `$VAR` | Prevent word splitting |
| **Explicit error handling** | `|| handle_error` | No silent failures |
| **Determinism** | Idempotent execution | Reproducible results |
| **No deprecated syntax** | Modern bash 4.0+ | Maintainability |

#### §12.3.2 Bashrs Configuration

**File**: `.bashrsignore`

```bash
# Documented exceptions for bashrs
# Format: FILE:LINE:RULE - Justification

# Legacy compatibility scripts (to be migrated)
# scripts/legacy/*.sh:*:* - Pending migration to bashrs compliance

# Third-party vendored scripts
# vendor/*.sh:*:* - External code, not under our control
```

#### §12.3.3 Script Quality Commands

```bash
# Lint all scripts (Tier 1)
bashrs lint scripts/

# Purify: auto-fix common issues (Tier 2)
bashrs purify scripts/

# Deep analysis with complexity metrics (Tier 3)
bashrs analyze scripts/ --max-complexity 15

# Determinism check: run twice, compare output (Tier 4)
bashrs determinism scripts/bench-ground-truth.sh --runs 2

# Makefile integration
bashrs make lint
bashrs make purify
```

#### §12.3.4 Required Script Header

All bash scripts MUST include this header:

```bash
#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# bashrs: compliant
set -euo pipefail

# Description: [Brief description of script purpose]
# Usage: [How to run the script]
# Dependencies: [External tools required]
```

### §12.4 CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CLI Parity CI

on: [push, pull_request]

jobs:
  tier2:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Format check
        run: cargo fmt --check

      - name: Clippy
        run: cargo clippy -- -D warnings

      - name: Bashrs lint (shell scripts)
        run: |
          cargo install bashrs --locked
          bashrs lint scripts/
          bashrs purify scripts/ --check  # Verify already purified

      - name: Unit tests
        run: cargo test --lib

  tier3:
    needs: tier2
    runs-on: ubuntu-latest
    steps:
      - name: Full tests
        run: cargo test --all

      - name: Coverage
        run: |
          cargo llvm-cov --all-features --lcov --output-path lcov.info
          # Fail if coverage < 95%
          cargo llvm-cov report --fail-under-lines 95

      - name: Bashrs deep analysis
        run: |
          bashrs analyze scripts/ --max-complexity 15
          bashrs analyze Makefile  # Makefile bash snippets

  tier4:
    needs: tier3
    runs-on: ubuntu-latest
    steps:
      - name: Mutation testing
        run: cargo mutants --no-times --timeout 300

      - name: Parity tests
        run: cargo test --test parity -- --ignored

      - name: Benchmarks
        run: cargo bench --bench parity_bench

      - name: Bashrs determinism check
        run: |
          # Verify benchmark scripts produce deterministic output structure
          bashrs determinism scripts/bench-ground-truth.sh \
            --args "test_data/jfk.wav" \
            --runs 2 \
            --compare-structure  # Compare JSON structure, not values
```

---

## §13. Peer-Reviewed Citations

The following peer-reviewed works inform this specification:

### [1] Martin, R.C. (2017)
**"Clean Architecture: A Craftsman's Guide to Software Structure and Design"**
Prentice Hall. ISBN: 978-0134494166.
*Justification*: Informs the testable logic separation principle (§4), requiring all CLI logic in library modules rather than binary shells.

### [2] Popper, K.R. (1959)
**"The Logic of Scientific Discovery"**
Routledge. ISBN: 978-0415278447.
*Justification*: Provides the falsificationism methodology (§11) for the 100-point checklist. Each check is designed to disprove functionality, and surviving checks represent verified claims.

### [3] Stroustrup, B. (1994)
**"The Design and Evolution of C++"**
Addison-Wesley. ISBN: 978-0201543308.
*Justification*: Zero-overhead abstraction principle (§2.1) ensures CLI wrapper adds minimal latency.

### [4] Shore, J. (2004)
**"Fail Fast"**
IEEE Software, 21(5), pp. 21-25. DOI: 10.1109/MS.2004.1331296.
*Justification*: Error handling strategy (§2.1, §F) requires immediate failure on invalid input rather than silent degradation.

### [5] Sculley, D. et al. (2015)
**"Hidden Technical Debt in Machine Learning Systems"**
NIPS 2015. Google Research.
*Justification*: Reproducibility requirement (§2.1) addresses ML system technical debt through deterministic outputs.

### [6] Liker, J.K. (2004)
**"The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer"**
McGraw-Hill. ISBN: 978-0071392310.
*Justification*: Toyota Production System principles (§3) guide quality infrastructure design, including Jidoka quality gates and Genchi Genbutsu parity testing.

### [7] Radford, A. et al. (2023)
**"Robust Speech Recognition via Large-Scale Weak Supervision"**
ICML 2023. OpenAI.
*Justification*: Whisper model architecture and capabilities define the transcription parity targets (§D).

### [8] Hannun, A. et al. (2014)
**"Deep Speech: Scaling up end-to-end speech recognition"**
arXiv:1412.5567.
*Justification*: Word Error Rate (WER) metric definition used in parity testing (§10).

### [9] Povey, D. et al. (2011)
**"The Kaldi Speech Recognition Toolkit"**
IEEE ASRU Workshop.
*Justification*: Audio preprocessing standards (mel filterbank, resampling) inform §B verification.

### [10] Vaswani, A. et al. (2017)
**"Attention Is All You Need"**
NeurIPS 2017. Google Brain.
*Justification*: Transformer architecture fundamentals underpin encoder/decoder parity verification (§D).

### [11] Beck, K. (2002)
**"Test Driven Development: By Example"**
Addison-Wesley. ISBN: 978-0321146533.
*Justification*: Establishes the RED-GREEN-REFACTOR cycle (§9.4) and the extreme testing methodology used to ensure functional parity.

### [12] Raymond, E.S. (2003)
**"The Art of Unix Programming"**
Addison-Wesley. ISBN: 978-0131429017.
*Justification*: Supports the "Rule of Silence" and "Rule of Repair" evident in the CLI's output design (§6.4) and error handling (§F).

### [13] Gamma, E., Helm, R., Johnson, R., Vlissides, J. (1994)
**"Design Patterns: Elements of Reusable Object-Oriented Software"**
Addison-Wesley. ISBN: 978-0201633610.
*Justification*: Validates the "Command" pattern usage in the CLI architecture (§4) for dispatching varied functionality through a unified interface.

### [14] Saltzer, J.H. & Schroeder, M.D. (1975)
**"The Protection of Information in Computer Systems"**
Proceedings of the IEEE, 63(9).
*Justification*: "Fail-safe defaults" and "Least privilege" principles underpin the security checks in §F, protecting against path traversal and resource exhaustion.

### [15] Hinton, G., Vinyals, O., & Dean, J. (2015)
**"Distilling the Knowledge in a Neural Network"**
NIPS Deep Learning Workshop.
*Justification*: Defines the "Student-Teacher" verification protocol used in Section H to validate that distilled models maintain semantic parity with larger baselines.

### [16] Jacob, B. et al. (2018)
**"Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"**
CVPR 2018. Google.
*Justification*: Establishes the accuracy/efficiency trade-off metrics (WER vs. Memory/IPS) verified in the Optimization Checklist (§H).

### [17] Frankle, J. & Carbin, M. (2019)
**"The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"**
ICLR 2019.
*Justification*: Supports the verification of pruned/sparse model performance (§H.7), ensuring that structural optimization does not destroy model capabilities.

---

### [14] Hoefler, T. & Belli, R. (2015)
**"Scientific Benchmarking of Parallel Computing Systems: Twelve Ways to Tell the Masses When Reporting Performance Results"**
SC'15: Proceedings of the International Conference for High Performance Computing. DOI: 10.1145/2807591.2807644.
*Justification*: Establishes the CV-based stopping criterion (§8.5.2), warmup methodology, and statistical rigor requirements for the ground truth benchmark infrastructure.

### [15] Georges, A., Buytaert, D., & Eeckhout, L. (2007)
**"Statistically Rigorous Java Performance Evaluation"**
OOPSLA'07: Proceedings of the 22nd Annual ACM SIGPLAN Conference. DOI: 10.1145/1297027.1297033.
*Justification*: Informs the statistical testing methodology (§8.5.3) including Welch's t-test, confidence intervals, and effect size reporting for benchmark comparisons.

### [16] Dean, J. & Barroso, L.A. (2013)
**"The Tail at Scale"**
Communications of the ACM, 56(2), pp. 74-80. DOI: 10.1145/2408776.2408794.
*Justification*: Supports the percentile-based latency reporting (p50/p95/p99) in §8.7 and the focus on tail latency for production parity verification.

---

## §14. References

### Standards
- IEEE 730-2014: Software Quality Assurance Processes
- ISO/IEC 25010:2011: Systems and software quality models
- WCAG 2.1: Web Content Accessibility Guidelines (for VTT output)
- RFC 6381: MIME types for subtitles
- RFC 8216: HTTP Live Streaming (M3U8 integration)

### Technical Documentation
- [whisper.cpp Repository](https://github.com/ggerganov/whisper.cpp)
- [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text)
- [SRT Specification](https://www.3playmedia.com/blog/create-srt-file/)
- [WebVTT Specification](https://www.w3.org/TR/webvtt1/)
- [LRC Format](https://en.wikipedia.org/wiki/LRC_(file_format))

### Aprender Ecosystem
- [aprender Specification](../aprender-spec-v1.md)
- [realizar Parity Testing](https://github.com/paiml/realizar)
- [PMAT Quality Gates](https://github.com/paiml/pmat)
- [Certeza TDD Methodology](../certeza-methodology.md)

---

## Appendix A: Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Model load failed |
| 5 | Audio decode failed |
| 6 | Inference failed |
| 7 | Output write failed |
| 8 | Permission denied |
| 9 | Out of memory |
| 10 | Interrupted (SIGINT) |

---

## Appendix B: Makefile Targets

```makefile
# Tier 1: On-save (<1s)
tier1:
	cargo check
	cargo fmt --check
	bashrs lint scripts/

# Tier 2: Pre-commit (<5s)
tier2: tier1
	cargo clippy -- -D warnings
	cargo test --lib
	bashrs purify scripts/ --check

# Tier 3: Pre-push (1-5 min)
tier3: tier2
	cargo test --all
	cargo llvm-cov --fail-under-lines 95
	bashrs analyze scripts/ --max-complexity 15
	bashrs make lint  # Validate Makefile bash snippets

# Tier 4: CI/CD (5-60 min)
tier4: tier3
	cargo mutants --no-times
	cargo test --test parity -- --ignored
	cargo bench --bench parity_bench
	bashrs determinism scripts/bench-ground-truth.sh --runs 2

# Parity testing
parity:
	./scripts/run_parity_tests.sh

# Falsification checklist
falsify:
	./scripts/run_falsification_checklist.sh

# Bashrs: Shell script quality (NOT shellcheck)
bashrs-lint:
	bashrs lint scripts/
	bashrs lint Makefile

bashrs-purify:
	bashrs purify scripts/

bashrs-analyze:
	bashrs analyze scripts/ --max-complexity 15 --report json > bashrs-report.json

bashrs-determinism:
	bashrs determinism scripts/bench-ground-truth.sh --runs 3 --compare-structure
```

---

## Appendix C: Response File Format

whisper-apr supports argument response files (like whisper.cpp):

```
# args.txt - Comments start with #
--model models/ggml-base.bin
--language auto
--format srt
--output transcriptions/

# Input files
--file audio/interview1.wav
--file audio/interview2.wav
```

Usage: `whisper-apr @args.txt`

---

## §15. Ecosystem Dependencies

This section documents the required features from the **aprender/realizar ecosystem** that whisper-apr CLI depends on for full functionality.

### §15.1 Existing Ecosystem Modules

The following modules **already exist** in the ecosystem and can be leveraged:

| Module | Location | Purpose | Status |
|--------|----------|---------|--------|
| `realizar::serve` | `realizar/src/serve.rs` | HTTP serving for `.apr` ML models | Available |
| `realizar::api` | `realizar/src/api.rs` | HTTP API (tokenize, generate, batch, streaming) | Available |
| `realizar::inference` | `realizar/src/inference.rs` | SIMD-accelerated transformer inference | Available |
| `realizar::quantize` | `realizar/src/quantize.rs` | Quantization (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K) | Available |
| `aprender::native` | `aprender/src/native/` | SIMD-native model formats | Available |
| `aprender::compute` | `aprender/src/compute/` | Tensor operations | Available |

### §15.2 Dependency Matrix

| whisper-apr Command | Required Module | Status | Notes |
|---------------------|-----------------|--------|-------|
| `transcribe` | `aprender::audio::mel` | **Planned** | Move from whisper.apr |
| `stream` | `aprender::audio::capture` | **Planned** | Audio capture |
| `stream` | `aprender::speech::vad` | **Planned** | Voice activity detection |
| `serve` | `realizar::api` | Available | Needs OpenAI Whisper API endpoints |
| `quantize` | `realizar::quantize` | Available | Full quantization support exists |
| `command` | `aprender::audio::capture` | **Planned** | Audio capture |
| `record` | `aprender::audio::capture` | **Planned** | Audio capture |

### §15.3 aprender Audio/Voice/Speech Architecture

The following modules will be added to **aprender** to support comprehensive audio/voice processing capabilities (ElevenLabs-style functionality in pure Rust).

#### §15.3.1 Module Overview

```
aprender/src/
├── audio/                    # Core audio I/O and processing
│   ├── mod.rs
│   ├── capture.rs           # Microphone input (ALSA/CoreAudio/WASAPI)
│   ├── playback.rs          # Speaker output
│   ├── stream.rs            # Chunked streaming primitives
│   ├── codec.rs             # Encode/decode (opus, mp3, aac, flac)
│   ├── resample.rs          # Sample rate conversion
│   ├── format.rs            # Container parsing (wav, mp4, webm, mkv)
│   └── mel.rs               # Mel spectrogram (MOVED from whisper.apr)
│
├── voice/                    # Voice-specific ML processing
│   ├── mod.rs
│   ├── embedding.rs         # Speaker embeddings (d-vector, x-vector)
│   ├── style.rs             # Voice style transfer
│   ├── clone.rs             # Voice cloning
│   ├── conversion.rs        # Voice-to-voice conversion
│   └── isolation.rs         # Voice isolation / noise removal
│
├── speech/                   # Speech recognition and synthesis
│   ├── mod.rs
│   ├── asr.rs               # ASR inference primitives
│   ├── tts.rs               # TTS inference (VITS, Tacotron-style)
│   ├── vad.rs               # Voice activity detection
│   └── diarization.rs       # Speaker diarization
```

#### §15.3.2 `aprender::audio` Module

**Purpose**: Core audio I/O and signal processing in pure Rust.

```rust
pub mod audio {
    // ===== Capture (Platform-specific FFI) =====
    pub mod capture {
        /// Audio input device
        pub struct AudioDevice {
            pub id: String,
            pub name: String,
            pub sample_rates: Vec<u32>,
            pub channels: u8,
        }

        /// List available audio input devices
        pub fn list_devices() -> Result<Vec<AudioDevice>, AudioError>;

        /// Open audio capture stream
        pub fn open(device: Option<&str>, config: CaptureConfig) -> Result<AudioCapture, AudioError>;

        /// Audio capture handle
        pub struct AudioCapture { /* ... */ }
        impl AudioCapture {
            pub fn read(&mut self, buffer: &mut [f32]) -> Result<usize, AudioError>;
            pub fn close(self) -> Result<(), AudioError>;
        }

        pub struct CaptureConfig {
            pub sample_rate: u32,      // Default: 16000 (Whisper)
            pub channels: u8,          // Default: 1 (mono)
            pub buffer_size_ms: u32,   // Default: 100
        }
    }

    // ===== Playback =====
    pub mod playback {
        pub fn list_devices() -> Result<Vec<AudioDevice>, AudioError>;
        pub fn open(device: Option<&str>, config: PlaybackConfig) -> Result<AudioPlayback, AudioError>;

        pub struct AudioPlayback { /* ... */ }
        impl AudioPlayback {
            pub fn write(&mut self, samples: &[f32]) -> Result<usize, AudioError>;
            pub fn close(self) -> Result<(), AudioError>;
        }
    }

    // ===== Streaming =====
    pub mod stream {
        /// Chunked audio stream for real-time processing
        pub struct AudioStream<R: Read> {
            reader: R,
            chunk_size: usize,
            overlap: usize,
        }

        impl<R: Read> Iterator for AudioStream<R> {
            type Item = Result<AudioChunk, AudioError>;
        }

        pub struct AudioChunk {
            pub samples: Vec<f32>,
            pub timestamp_ms: u64,
            pub is_final: bool,
        }
    }

    // ===== Codec (Pure Rust decode, symphonia-based) =====
    pub mod codec {
        pub enum AudioFormat {
            Wav, Mp3, Aac, Flac, Opus, Ogg, M4a,
        }

        /// Decode audio file to f32 samples
        pub fn decode(data: &[u8], format: AudioFormat) -> Result<DecodedAudio, CodecError>;

        /// Decode from container (mp4, webm, mkv) - extract audio track
        pub fn decode_container(data: &[u8]) -> Result<DecodedAudio, CodecError>;

        pub struct DecodedAudio {
            pub samples: Vec<f32>,
            pub sample_rate: u32,
            pub channels: u8,
            pub duration_ms: u64,
        }
    }

    // ===== Resample =====
    pub mod resample {
        /// Resample audio to target sample rate (high-quality sinc interpolation)
        pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32>;
    }

    // ===== Mel Spectrogram (MOVED from whisper.apr) =====
    pub mod mel {
        /// Mel filterbank configuration
        pub struct MelConfig {
            pub sample_rate: u32,      // Default: 16000
            pub n_fft: usize,          // Default: 400
            pub hop_length: usize,     // Default: 160
            pub n_mels: usize,         // Default: 80
            pub fmin: f32,             // Default: 0.0
            pub fmax: f32,             // Default: 8000.0
        }

        /// Compute mel spectrogram from audio samples
        pub fn mel_spectrogram(samples: &[f32], config: &MelConfig) -> Vec<Vec<f32>>;

        /// Precomputed mel filterbank for efficiency
        pub struct MelFilterbank { /* ... */ }
        impl MelFilterbank {
            pub fn new(config: &MelConfig) -> Self;
            pub fn apply(&self, fft_magnitudes: &[f32]) -> Vec<f32>;
        }
    }
}
```

**Platform Support**:
| Platform | Capture Backend | Playback Backend |
|----------|-----------------|------------------|
| Linux | ALSA (`libasound`) | ALSA |
| macOS | CoreAudio | CoreAudio |
| Windows | WASAPI | WASAPI |
| WASM | Web Audio API | Web Audio API |

**Codec Support (Pure Rust via symphonia)**:
| Format | Decode | Encode | Notes |
|--------|--------|--------|-------|
| WAV | ✅ | ✅ | Native implementation |
| MP3 | ✅ | ❌ | symphonia decoder |
| AAC | ✅ | ❌ | symphonia decoder |
| FLAC | ✅ | ✅ | symphonia |
| Opus | ✅ | ✅ | Pure Rust `opus` crate |
| OGG | ✅ | ❌ | `lewton` (vorbis) |
| MP4/M4A | ✅ | ❌ | Container extraction |
| WebM | ✅ | ❌ | Container extraction |
| MKV | ✅ | ❌ | Container extraction |

#### §15.3.3 `aprender::voice` Module

**Purpose**: Voice-specific ML processing for style transfer, cloning, and conversion (ElevenLabs-style).

```rust
pub mod voice {
    // ===== Speaker Embeddings =====
    pub mod embedding {
        /// Extract speaker embedding from audio (d-vector style)
        pub fn extract_embedding(samples: &[f32], model: &SpeakerEncoder) -> Result<SpeakerEmbedding, VoiceError>;

        /// Speaker embedding vector (typically 256-dim)
        pub struct SpeakerEmbedding {
            pub vector: Vec<f32>,
            pub model_type: EmbeddingModel,
        }

        pub enum EmbeddingModel {
            DVector,    // Google's d-vector
            XVector,    // Kaldi x-vector
            ECAPA,      // ECAPA-TDNN
            Resemblyzer, // Resemblyzer-style
        }

        /// Compare speaker similarity (cosine similarity)
        pub fn speaker_similarity(a: &SpeakerEmbedding, b: &SpeakerEmbedding) -> f32;
    }

    // ===== Voice Style Transfer =====
    pub mod style {
        /// Transfer voice style from source to target
        pub fn transfer_style(
            content_audio: &[f32],
            style_embedding: &SpeakerEmbedding,
            model: &StyleTransferModel,
        ) -> Result<Vec<f32>, VoiceError>;

        /// Style transfer model (OpenVoice-style)
        pub struct StyleTransferModel { /* ... */ }
    }

    // ===== Voice Cloning =====
    pub mod clone {
        /// Create voice clone from reference samples
        pub fn create_clone(
            reference_samples: &[&[f32]],
            model: &VoiceCloningModel,
        ) -> Result<VoiceClone, VoiceError>;

        /// Cloned voice that can be used for TTS
        pub struct VoiceClone {
            pub embedding: SpeakerEmbedding,
            pub style_params: StyleParams,
        }
    }

    // ===== Voice Conversion =====
    pub mod conversion {
        /// Convert voice A to sound like voice B
        pub fn convert_voice(
            source_audio: &[f32],
            target_embedding: &SpeakerEmbedding,
            model: &VoiceConversionModel,
        ) -> Result<Vec<f32>, VoiceError>;
    }

    // ===== Voice Isolation =====
    pub mod isolation {
        /// Isolate voice from background noise
        pub fn isolate_voice(audio: &[f32], model: &IsolationModel) -> Result<Vec<f32>, VoiceError>;

        /// Remove background music, keep voice
        pub fn remove_music(audio: &[f32], model: &IsolationModel) -> Result<Vec<f32>, VoiceError>;
    }
}
```

#### §15.3.4 `aprender::speech` Module

**Purpose**: Speech recognition (ASR) and synthesis (TTS) primitives.

```rust
pub mod speech {
    // ===== Automatic Speech Recognition =====
    pub mod asr {
        /// ASR inference session
        pub struct AsrSession<M: AsrModel> {
            model: M,
            config: AsrConfig,
        }

        impl<M: AsrModel> AsrSession<M> {
            /// Transcribe audio to text
            pub fn transcribe(&self, audio: &[f32]) -> Result<Transcription, AsrError>;

            /// Streaming transcription
            pub fn transcribe_streaming(&mut self) -> StreamingTranscription;
        }

        pub struct Transcription {
            pub text: String,
            pub segments: Vec<Segment>,
            pub language: Option<String>,
        }

        pub trait AsrModel {
            fn forward(&self, mel: &[Vec<f32>]) -> Result<Vec<Token>, AsrError>;
        }
    }

    // ===== Text-to-Speech =====
    pub mod tts {
        /// TTS inference session
        pub struct TtsSession<M: TtsModel> {
            model: M,
            config: TtsConfig,
        }

        impl<M: TtsModel> TtsSession<M> {
            /// Synthesize speech from text
            pub fn synthesize(&self, text: &str) -> Result<Vec<f32>, TtsError>;

            /// Synthesize with specific voice
            pub fn synthesize_with_voice(
                &self,
                text: &str,
                voice: &VoiceClone,
            ) -> Result<Vec<f32>, TtsError>;

            /// Streaming synthesis
            pub fn synthesize_streaming(&self, text: &str) -> StreamingSynthesis;
        }

        pub trait TtsModel {
            fn forward(&self, tokens: &[i32], speaker: Option<&SpeakerEmbedding>) -> Result<Vec<f32>, TtsError>;
        }
    }

    // ===== Voice Activity Detection =====
    pub mod vad {
        /// Detect voice activity in audio
        pub fn detect_voice_activity(
            audio: &[f32],
            config: &VadConfig,
        ) -> Result<Vec<VoiceSegment>, VadError>;

        pub struct VadConfig {
            pub threshold: f32,           // Default: 0.5
            pub min_speech_ms: u32,       // Default: 250
            pub min_silence_ms: u32,      // Default: 100
            pub window_size_ms: u32,      // Default: 30
        }

        pub struct VoiceSegment {
            pub start_ms: u64,
            pub end_ms: u64,
            pub confidence: f32,
        }

        /// Real-time VAD for streaming
        pub struct StreamingVad { /* ... */ }
        impl StreamingVad {
            pub fn feed(&mut self, samples: &[f32]) -> Option<VadEvent>;
        }

        pub enum VadEvent {
            SpeechStart { timestamp_ms: u64 },
            SpeechEnd { timestamp_ms: u64 },
        }
    }

    // ===== Speaker Diarization =====
    pub mod diarization {
        /// Identify who spoke when
        pub fn diarize(
            audio: &[f32],
            config: &DiarizationConfig,
        ) -> Result<Vec<SpeakerTurn>, DiarizationError>;

        pub struct SpeakerTurn {
            pub speaker_id: u32,
            pub start_ms: u64,
            pub end_ms: u64,
            pub embedding: Option<SpeakerEmbedding>,
        }

        pub struct DiarizationConfig {
            pub max_speakers: Option<u32>,
            pub min_segment_ms: u32,
        }
    }
}
```

### §15.4 Migration: Move Mel from whisper.apr to aprender

The mel spectrogram implementation will be **moved** from `whisper.apr/src/audio/mel.rs` to `aprender/src/audio/mel.rs`.

**Current Location**: `whisper.apr/src/audio/mel.rs`
**New Location**: `aprender/src/audio/mel.rs`

**whisper.apr Changes**:
```rust
// Before (whisper.apr/src/audio/mod.rs)
pub mod mel;
pub mod wav;

// After (whisper.apr/src/audio/mod.rs)
pub mod wav;
pub use aprender::audio::mel;  // Re-export from aprender
```

**Rationale**:
- Mel spectrograms are used by ASR, TTS, voice cloning, etc.
- Generic audio processing belongs in the foundational crate
- Avoids code duplication across whisper.apr, voice projects

### §15.5 Integration Work (whisper-apr)

The following integration work is needed **within whisper-apr** to leverage the new aprender modules:

#### §15.5.1 Use `aprender::audio::mel`

```rust
// whisper.apr/src/audio/mod.rs
pub use aprender::audio::mel;  // Re-export

// whisper.apr/src/model/encoder.rs
use aprender::audio::mel::{mel_spectrogram, MelConfig};
```

#### §15.5.2 Use `aprender::audio::capture` for streaming

```rust
// whisper.apr/src/cli/commands.rs
use aprender::audio::capture::{open, CaptureConfig};
use aprender::speech::vad::StreamingVad;

pub fn run_stream(args: StreamArgs, global: &Args) -> CliResult<CommandResult> {
    let capture = open(args.device.as_deref(), CaptureConfig::default())?;
    let mut vad = StreamingVad::new(VadConfig::default());
    // ... streaming transcription loop
}
```

#### §15.5.3 Use `aprender::audio::codec` for format support

```rust
// whisper.apr/src/cli/commands.rs
use aprender::audio::codec::{decode_container, AudioFormat};

fn load_audio_samples(path: &Path, data: &[u8]) -> CliResult<Vec<f32>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let decoded = match ext {
        "wav" => decode(data, AudioFormat::Wav)?,
        "mp3" => decode(data, AudioFormat::Mp3)?,
        "mp4" | "m4a" => decode_container(data)?,
        // ...
    };
    Ok(resample(&decoded.samples, decoded.sample_rate, 16000))
}
```

### §15.6 Implementation Tracking

| Ticket | Repo | Module | Status | Priority |
|--------|------|--------|--------|----------|
| [#131](https://github.com/paiml/aprender/issues/131) | paiml/aprender | `audio::*` (all) | Open | High |
| [#132](https://github.com/paiml/aprender/issues/132) | paiml/aprender | `voice::*` (all) | Open | Medium |
| [#133](https://github.com/paiml/aprender/issues/133) | paiml/aprender | `speech::*` (all) | Open | High |

**Sub-module Priority**:
| Module | Priority | Blocks |
|--------|----------|--------|
| `audio::mel` | High | whisper.apr transcription |
| `audio::capture` | High | stream, record, command |
| `audio::codec` | Medium | mp3/mp4 input support |
| `speech::vad` | High | streaming transcription |
| `voice::embedding` | Medium | voice cloning/conversion |
| `speech::tts` | Low | text-to-speech |

**Note**: realizar modules (`serve`, `api`, `inference`, `quantize`) already exist and are available.

### §15.7 Verification Checklist

Before marking a dependency as "Available":

- [ ] Module compiles with `cargo check`
- [ ] Public API matches specification above
- [ ] Unit tests achieve ≥80% coverage
- [ ] Integration test with whisper-apr passes
- [ ] Documentation includes usage examples
- [ ] No external crate dependencies (pure ecosystem)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0-draft | 2025-12-18 | whisper-apr team | Initial specification |
| 1.0.1-draft | 2025-12-18 | whisper-apr team | Review enhancements: consolidated §C (15→10 pts), expanded §F with security checks (10→15 pts), added 3 citations [11-13] |
| 1.1.0-draft | 2025-12-18 | whisper-apr team | **Ground truth benchmarking**: Added §8.4-§8.8 with realizar-style infrastructure, statistical methodology, CI/Jidoka gates, JSON schema, baseline management. Added 3 citations [14-16] |
| 1.2.0-draft | 2025-12-18 | whisper-apr team | **Bashrs integration**: Added §12.3 for shell script quality enforcement (NOT shellcheck). Updated all tiers with bashrs checks. Added bashrs-compliant headers to scripts. Updated Makefile with bashrs targets |
| 1.3.0-draft | 2025-12-18 | whisper-apr team | **Ecosystem dependencies**: Added §15 documenting aprender/realizar integration. Clarified that realizar modules (serve, api, inference, quantize) already exist. Only `aprender::audio` needs implementation ([#130](https://github.com/paiml/aprender/issues/130)). Integration work is within whisper-apr |
| 1.4.0-draft | 2025-12-18 | whisper-apr team | **Comprehensive audio/voice/speech architecture**: Expanded §15 with full aprender module design for ElevenLabs-style capabilities. Added `audio` (capture, playback, codec, mel), `voice` (embedding, style, clone, conversion, isolation), `speech` (asr, tts, vad, diarization). Defined mel migration from whisper.apr to aprender |

---

**Status**: UNDER REVIEW

### Review Summary (v1.2.0)

**Bashrs Shell Script Quality Integration**

All shell scripts now enforced via **bashrs** (NOT shellcheck) following aprender ecosystem conventions:

| Component | Addition | Purpose |
|-----------|----------|---------|
| §12.1 | Tiered bashrs checks | lint → purify → analyze → determinism |
| §12.2 | `[bashrs]` PMAT config | strict_mode, quote_variables, determinism |
| §12.3 | **New section** | Complete bashrs specification |
| §12.4 | CI bashrs steps | GitHub Actions integration |
| §8.4.2 | Script headers | bashrs-compliant header format |
| Appendix B | Makefile targets | bashrs-lint, bashrs-purify, bashrs-analyze |

**Bashrs Enforcement Tiers**:
```
Tier 1: bashrs lint scripts/           # Syntax + safety
Tier 2: bashrs purify scripts/ --check # Auto-fixable issues
Tier 3: bashrs analyze --max-complexity 15  # Deep analysis
Tier 4: bashrs determinism --runs 2    # Idempotency check
```

**Required Script Header**:
```bash
#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# bashrs: compliant
set -euo pipefail
```

---

### Review Summary (v1.1.0)

**Major Addition: Ground Truth Benchmark Infrastructure (§8.4-§8.8)**

Following realizar's proven methodology for benchmarking against llama.cpp/Ollama:

| Section | Content | Key Features |
|---------|---------|--------------|
| §8.4 | Benchmark Infrastructure | Side-by-side scripts, Criterion.rs suite, comparison targets |
| §8.5 | Statistical Methodology | CV-based stopping, Welch's t-test, Cohen's d, Bootstrap CI |
| §8.6 | CI Integration | GitHub Actions, Jidoka gates (>10% = STOP THE LINE) |
| §8.7 | Result Schema | JSON format with full metrics, Rust types |
| §8.8 | Baseline Management | 90-day retention, time-series tracking, trend visualization |

**New Citations Added**:
- [14] Hoefler & Belli SC'15: Scientific benchmarking methodology
- [15] Georges et al. OOPSLA'07: Statistically rigorous evaluation
- [16] Dean & Barroso CACM'13: Tail latency (p50/p95/p99)

**Citation Count**: 10 → 16 peer-reviewed works

---

### Review Summary (v1.0.1)

**Enhancements Applied**:
- ✅ Section C consolidated: Combined format checks for efficiency (15→10 points)
- ✅ Section F expanded: Added 5 security/resilience checks (10→15 points)
  - F.11: Path traversal protection
  - F.12: Large input resilience (10GB attack vector)
  - F.13: Recursive symlink handling
  - F.14: Argument fuzzing safety
  - F.15: Memory limit enforcement (cgroup)
- ✅ Citations expanded: 10→13 peer-reviewed works
  - [11] Beck: TDD by Example (RED-GREEN-REFACTOR)
  - [12] Raymond: Art of Unix Programming (CLI design)
  - [13] Gamma et al.: Design Patterns (Command pattern)

**Point Distribution Verified**:
| Section | Points | Notes |
|---------|--------|-------|
| A | 15 | Argument parsing |
| B | 20 | Core transcription |
| C | 10 | Output formats (consolidated) |
| D | 20 | whisper.cpp parity |
| E | 15 | Performance |
| F | 15 | Error handling + security (expanded) |
| G | 5 | Advanced features |
| **Total** | **100** | ✓ Verified |

### Remaining Review Items

Please verify:
1. [ ] All whisper.cpp arguments are mapped correctly
2. [ ] Performance targets (RTF ≤1.1×) are achievable
3. [ ] Security checks (F.11-F.15) cover attack surface
4. [ ] Citations are correctly attributed with DOIs/ISBNs
5. [ ] Quality gates are enforceable in CI/CD

Submit feedback via GitHub Issues or PR comments.

---

## §16. 100-Point CLI Transcription Falsification Checklist

**Version**: 1.0.0
**Created**: 2025-12-19
**Methodology**: Popperian Falsification + Five-Whys Root Cause Analysis
**Scope**: CLI `transcribe` command ONLY
**Diagnostic Tooling**: renacer (tracing), pmat five-whys

### Purpose

This checklist focuses exclusively on the `whisper-apr transcribe` command pipeline. Each check is designed to **falsify** a claim about transcription correctness. Failures indicate bugs requiring root cause analysis via **Toyota Way Five-Whys** methodology and **renacer tracing**.

### Diagnostic Protocol

When a check **FAILS**:

1. **Document the failure** with exact command and output
2. **Run renacer trace**: `renacer -s -- cargo test <failing_test>`
3. **Apply Five-Whys**: `pmat five-whys "transcription failed for <check>"`
4. **Create JIRA ticket**: `WAPR-TRANS-XXX`
5. **Do NOT fix inline** - another team handles remediation

### Grading Scale

- 95-100 points: A+ (Transcription Production Ready)
- 90-94 points: A (Release Candidate)
- 85-89 points: B (Beta Quality)
- 80-84 points: C (Alpha Quality)
- <80 points: F (Critical Bugs)

---

### Section T1: Audio Input Pipeline (15 points)

| # | Claim to Falsify | Command | Expected Result | Pass | Fail |
|---|------------------|---------|-----------------|------|------|
| T1.1 | 16kHz mono WAV transcribes correctly | `transcribe -f test-16k-mono.wav` | Non-empty text output | [ ] | [ ] |
| T1.2 | 44.1kHz audio resamples without artifacts | `transcribe -f test-44k.wav` | Same text as 16kHz version | [ ] | [ ] |
| T1.3 | 48kHz audio resamples correctly | `transcribe -f test-48k.wav` | Same text as 16kHz version | [ ] | [ ] |
| T1.4 | 8kHz upsampling produces valid mel | `transcribe -f test-8k.wav` | Non-empty output | [ ] | [ ] |
| T1.5 | Stereo→mono mixdown preserves content | `transcribe -f test-stereo.wav` | Same text as mono | [ ] | [ ] |
| T1.6 | 24-bit audio depth handled | `transcribe -f test-24bit.wav` | Valid transcription | [ ] | [ ] |
| T1.7 | 32-bit float audio handled | `transcribe -f test-32f.wav` | Valid transcription | [ ] | [ ] |
| T1.8 | Very short audio (<0.5s) handled | `transcribe -f test-300ms.wav` | Output or "no speech" | [ ] | [ ] |
| T1.9 | 30-second audio chunk boundary correct | `transcribe -f test-30s.wav` | No truncation at 30s | [ ] | [ ] |
| T1.10 | 60-second audio multi-chunk works | `transcribe -f test-60s.wav` | Complete transcription | [ ] | [ ] |
| T1.11 | Silent audio detected as no-speech | `transcribe -f silence-5s.wav` | Empty or "[BLANK_AUDIO]" | [ ] | [ ] |
| T1.12 | Near-silent audio with speech detected | `transcribe -f whisper-5s.wav` | Detects faint speech | [ ] | [ ] |
| T1.13 | DC offset in audio handled | `transcribe -f test-dc-offset.wav` | Valid transcription | [ ] | [ ] |
| T1.14 | Clipped audio (saturation) handled | `transcribe -f test-clipped.wav` | Degrades gracefully | [ ] | [ ] |
| T1.15 | Audio with leading silence handled | `transcribe -f test-lead-silence.wav` | Correct timestamp offset | [ ] | [ ] |

---

### Section T2: Mel Spectrogram Computation (10 points)

| # | Claim to Falsify | Command | Expected Result | Pass | Fail |
|---|------------------|---------|-----------------|------|------|
| T2.1 | Mel filterbank produces 80 bins | `transcribe -v -f test.wav` | Log shows 80-mel | [ ] | [ ] |
| T2.2 | Mel values in valid range | Internal check | Values in [-1, 4] range | [ ] | [ ] |
| T2.3 | Mel matches whisper.cpp reference | `parity --mel test.wav` | MSE < 1e-4 | [ ] | [ ] |
| T2.4 | FFT window size correct (400 samples) | Trace check | n_fft=400 | [ ] | [ ] |
| T2.5 | Hop length correct (160 samples) | Trace check | hop=160 | [ ] | [ ] |
| T2.6 | Log-mel scaling applied correctly | Trace check | log10 applied | [ ] | [ ] |
| T2.7 | Padding to 30s handled | `transcribe -f 5s.wav` | Padded to 3000 frames | [ ] | [ ] |
| T2.8 | No NaN/Inf in mel output | Internal check | All finite values | [ ] | [ ] |
| T2.9 | Mel symmetric around DC removed | Trace check | No DC component | [ ] | [ ] |
| T2.10 | Mel normalization applied | Trace check | Normalized to model range | [ ] | [ ] |

---

### Section T3: Encoder Forward Pass (15 points)

| # | Claim to Falsify | Command | Expected Result | Pass | Fail |
|---|------------------|---------|-----------------|------|------|
| T3.1 | Conv1 output shape correct | Trace | [batch, 384, 1500] for tiny | [ ] | [ ] |
| T3.2 | Conv2 output shape correct | Trace | [batch, 384, 1500] for tiny | [ ] | [ ] |
| T3.3 | Positional embedding added | Trace | Non-zero pos embed | [ ] | [ ] |
| T3.4 | Encoder block 0 attention works | Trace | Non-uniform attention | [ ] | [ ] |
| T3.5 | Encoder block 0 FFN works | Trace | Output differs from input | [ ] | [ ] |
| T3.6 | All encoder blocks execute | Trace | 4 blocks for tiny | [ ] | [ ] |
| T3.7 | LayerNorm applied correctly | Trace | Mean≈0, Var≈1 | [ ] | [ ] |
| T3.8 | Encoder output shape correct | Trace | [batch, 1500, 384] | [ ] | [ ] |
| T3.9 | Encoder output not all zeros | Internal check | max(abs) > 0.01 | [ ] | [ ] |
| T3.10 | Encoder output not all same | Internal check | std > 0.01 | [ ] | [ ] |
| T3.11 | Encoder deterministic (same input) | Run twice | Identical output | [ ] | [ ] |
| T3.12 | Encoder matches HF reference | `parity --encoder test.wav` | Cosine sim > 0.99 | [ ] | [ ] |
| T3.13 | GELU activation applied | Trace | GELU pattern visible | [ ] | [ ] |
| T3.14 | Attention softmax sums to 1 | Trace | Row sums = 1.0 | [ ] | [ ] |
| T3.15 | No gradient explosion | Trace | max(abs) < 100 | [ ] | [ ] |

---

### Section T4: Decoder Forward Pass (15 points)

| # | Claim to Falsify | Command | Expected Result | Pass | Fail |
|---|------------------|---------|-----------------|------|------|
| T4.1 | SOT token (50258) starts decode | Trace | First token = 50258 | [ ] | [ ] |
| T4.2 | Language token correct | `transcribe -l en` | Token 50259 (English) | [ ] | [ ] |
| T4.3 | Transcribe token present | Trace | Token 50359 | [ ] | [ ] |
| T4.4 | No-timestamps token if disabled | `--no-timestamps` | Token 50363 | [ ] | [ ] |
| T4.5 | Token embedding lookup works | Trace | Non-zero embeddings | [ ] | [ ] |
| T4.6 | Positional embedding added | Trace | Pos embed applied | [ ] | [ ] |
| T4.7 | Causal mask applied | Trace | Upper triangle masked | [ ] | [ ] |
| T4.8 | Cross-attention uses encoder | Trace | KV from encoder output | [ ] | [ ] |
| T4.9 | Cross-attention non-uniform | Trace | Attention varies | [ ] | [ ] |
| T4.10 | Decoder blocks execute | Trace | 4 blocks for tiny | [ ] | [ ] |
| T4.11 | Final LayerNorm applied | Trace | Pre-logits normalized | [ ] | [ ] |
| T4.12 | Logits shape correct | Trace | [batch, seq, 51865] | [ ] | [ ] |
| T4.13 | Logits not all same | Internal check | std > 0.01 | [ ] | [ ] |
| T4.14 | EOT token (50257) terminates | Trace | Decoding stops at EOT | [ ] | [ ] |
| T4.15 | Max tokens limit respected | `--max-tokens 10` | ≤10 tokens output | [ ] | [ ] |

---

### Section T5: Token Decoding & Sampling (10 points)

| # | Claim to Falsify | Command | Expected Result | Pass | Fail |
|---|------------------|---------|-----------------|------|------|
| T5.1 | Greedy decoding selects argmax | `--temperature 0` | Argmax tokens | [ ] | [ ] |
| T5.2 | Temperature 0.5 adds variance | `--temperature 0.5` | Varied outputs | [ ] | [ ] |
| T5.3 | Temperature 1.0 more random | `--temperature 1.0` | High variance | [ ] | [ ] |
| T5.4 | Beam search improves quality | `--beam-size 5` | Lower perplexity | [ ] | [ ] |
| T5.5 | Best-of sampling works | `--best-of 3` | Best candidate chosen | [ ] | [ ] |
| T5.6 | Suppress blank tokens | Default | No "[BLANK]" in output | [ ] | [ ] |
| T5.7 | Suppress regex works | `--suppress-regex "\\[.*\\]"` | Brackets removed | [ ] | [ ] |
| T5.8 | Timestamp tokens decoded | Default | Timestamps present | [ ] | [ ] |
| T5.9 | Word-level timestamps | `--word-timestamps` | Per-word timing | [ ] | [ ] |
| T5.10 | Token IDs map to valid text | Internal | All tokens decodable | [ ] | [ ] |

---

### Section T6: Text Output Generation (10 points)

| # | Claim to Falsify | Command | Expected Result | Pass | Fail |
|---|------------------|---------|-----------------|------|------|
| T6.1 | Output is valid UTF-8 | `transcribe -f test.wav` | Valid UTF-8 string | [ ] | [ ] |
| T6.2 | Leading/trailing whitespace trimmed | Default | No extra spaces | [ ] | [ ] |
| T6.3 | Repeated words filtered | `--hallucination-filter` | No "the the the" | [ ] | [ ] |
| T6.4 | Punctuation present | Speech with pauses | Periods/commas | [ ] | [ ] |
| T6.5 | Capitalization correct | Sentence start | Capital letters | [ ] | [ ] |
| T6.6 | Numbers as spoken | "one two three" | "one two three" | [ ] | [ ] |
| T6.7 | Unicode preserved | Non-ASCII speech | Correct chars | [ ] | [ ] |
| T6.8 | Empty audio → empty output | Silence only | "" or minimal | [ ] | [ ] |
| T6.9 | Special tokens removed | Default | No <|...|> tokens | [ ] | [ ] |
| T6.10 | Multi-segment joined | Long audio | Coherent text | [ ] | [ ] |

---

### Section T7: Timestamp Accuracy (10 points)

| # | Claim to Falsify | Command | Expected Result | Pass | Fail |
|---|------------------|---------|-----------------|------|------|
| T7.1 | Start timestamp ≥ 0 | Any audio | start ≥ 0.0 | [ ] | [ ] |
| T7.2 | End timestamp > start | Any audio | end > start | [ ] | [ ] |
| T7.3 | Timestamps monotonic | Multi-segment | Each start ≥ prev end | [ ] | [ ] |
| T7.4 | Timestamp matches audio position | Known speech at 5s | Timestamp ≈ 5.0s | [ ] | [ ] |
| T7.5 | Timestamp resolution 20ms | Check values | Multiples of 0.02 | [ ] | [ ] |
| T7.6 | Timestamp tokens parsed | `--print-special` | <\|X.XX\|> format | [ ] | [ ] |
| T7.7 | Offset applied correctly | `--offset-t 5000` | Timestamps +5s | [ ] | [ ] |
| T7.8 | Duration limits output | `--duration 10000` | Max timestamp ≤10s | [ ] | [ ] |
| T7.9 | No timestamp mode works | `--no-timestamps` | No timing in output | [ ] | [ ] |
| T7.10 | SRT timestamps correct | `--format srt` | HH:MM:SS,mmm format | [ ] | [ ] |

---

### Section T8: whisper.cpp Ground Truth Parity (10 points)

| # | Claim to Falsify | Command | Expected Result | Pass | Fail |
|---|------------------|---------|-----------------|------|------|
| T8.1 | WER ≤ 5% vs whisper.cpp | jfk.wav tiny | WER ≤ 0.05 | [ ] | [ ] |
| T8.2 | CER ≤ 3% vs whisper.cpp | jfk.wav tiny | CER ≤ 0.03 | [ ] | [ ] |
| T8.3 | Semantic similarity ≥ 0.95 | Embedding comparison | Cosine ≥ 0.95 | [ ] | [ ] |
| T8.4 | Same language detected | Auto-detect | Identical lang code | [ ] | [ ] |
| T8.5 | Timestamp drift ≤ 100ms | Compare timings | max drift ≤ 0.1s | [ ] | [ ] |
| T8.6 | Same segment count ±1 | Compare segments | count diff ≤ 1 | [ ] | [ ] |
| T8.7 | Translate mode matches | `--translate` | Same English output | [ ] | [ ] |
| T8.8 | Prompt effect matches | `--prompt "..."` | Similar behavior | [ ] | [ ] |
| T8.9 | Beam search matches | `--beam-size 5` | Similar output | [ ] | [ ] |
| T8.10 | Temperature fallback | Complex audio | Same recovery | [ ] | [ ] |

---

### Section T9: Edge Cases & Robustness (5 points)

| # | Claim to Falsify | Command | Expected Result | Pass | Fail |
|---|------------------|---------|-----------------|------|------|
| T9.1 | Handles music/noise gracefully | Music file | No crash, minimal text | [ ] | [ ] |
| T9.2 | Handles overlapping speech | Two speakers | Some transcription | [ ] | [ ] |
| T9.3 | Handles accented speech | Non-native speaker | Reasonable accuracy | [ ] | [ ] |
| T9.4 | Handles fast speech | Rapid speaking | Complete transcription | [ ] | [ ] |
| T9.5 | Handles whispered speech | Quiet speech | Detects content | [ ] | [ ] |

---

### Scoring Summary: CLI Transcription

| Section | Points | Description |
|---------|--------|-------------|
| T1 | 15 | Audio Input Pipeline |
| T2 | 10 | Mel Spectrogram Computation |
| T3 | 15 | Encoder Forward Pass |
| T4 | 15 | Decoder Forward Pass |
| T5 | 10 | Token Decoding & Sampling |
| T6 | 10 | Text Output Generation |
| T7 | 10 | Timestamp Accuracy |
| T8 | 10 | whisper.cpp Ground Truth Parity |
| T9 | 5 | Edge Cases & Robustness |
| **TOTAL** | **100** | |

---

### Diagnostic Commands Reference

```bash
# Run all transcription tests
cargo test --features cli transcription_ -- --nocapture

# Trace specific failure with renacer
renacer -s -- cargo test test_t3_encoder_output

# Five-whys analysis
pmat five-whys "T4.8 cross-attention not using encoder output"

# Full pipeline trace
RUST_LOG=trace ./target/release/whisper-apr-cli -v transcribe -f test.wav 2>&1 | tee trace.log

# Compare with whisper.cpp
./scripts/ground_truth_compare.sh test.wav

# Mel spectrogram comparison
cargo run --example compare_mel -- test.wav

# Encoder output comparison
cargo run --example compare_encoder -- test.wav
```

---

### Failure Escalation Matrix

| Severity | Sections | Action |
|----------|----------|--------|
| **Critical** | T3, T4 (encoder/decoder) | Block release, immediate fix |
| **High** | T1, T2, T5 (pipeline) | Fix before RC |
| **Medium** | T6, T7, T8 (output quality) | Fix before GA |
| **Low** | T9 (edge cases) | Backlog for future |

---

### Related Tickets

| Ticket | Description | Status |
|--------|-------------|--------|
| WAPR-TRANS-001 | Decoder produces empty/whitespace | Open |
| WAPR-MEL-001 | Mel spectrogram layout investigation | Open |
| WAPR-ENC-001 | Encoder output validation | Open |

---

### Changelog

- **v1.0.0** (2025-12-19): Initial 100-point transcription checklist
