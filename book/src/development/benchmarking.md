# Benchmarking

Whisper.apr includes comprehensive benchmarking tools for measuring pipeline performance.

## Quick Start

```bash
# Run the interactive benchmark TUI
cargo run --example benchmark_tui --features benchmark-tui --release

# Run non-interactive pipeline benchmark
cargo run --example benchmark_pipeline --release

# Run criterion benchmarks
cargo bench --bench inference
```

## Benchmark TUI

The `benchmark_tui` example provides a real-time, step-by-step visualization of the transcription pipeline:

```bash
cargo run --example benchmark_tui --features benchmark-tui --release
```

### Pipeline Steps

| Step | Name     | Target   | Description |
|------|----------|----------|-------------|
| A    | Model    | 500ms    | Load whisper-tiny.apr model |
| B    | Load     | 50ms     | Load audio file |
| C    | Parse    | 10ms     | Parse WAV headers |
| D    | Resample | 100ms    | Resample to 16kHz |
| F    | Mel      | 50ms     | Compute mel spectrogram |
| G    | Encode   | 500ms    | Run encoder transformer |
| H    | Decode   | 2000ms   | Run decoder (autoregressive) |

### TUI Controls

- `[s]` Start benchmark
- `[r]` Reset
- `[q]` Quit

### Sample Output

```
┌ PIPELINE PROGRESS (REAL) ───────────────┐┌ LIVE METRICS ───────────────────┐
│████████[A] ✓ Model       693.1ms ███████││                                 │
│████████[B] ✓ Load          5.2ms ███████││   RTF: 3.92x (target: <2.0x)    │
│████████[C] ✓ Parse         0.8ms ███████││                                 │
│████████[D] ✓ Resample     12.4ms ███████││          Model: 145.0MB         │
│████████[F] ✓ Mel          18.6ms ███████││Elapsed: 3917ms / Audio: 1000ms  │
│████████[G] ✓ Encode       50.9ms ███████││                                 │
│████████[H] ✓ Decode     3136.0ms ███████││                                 │
└─────────────────────────────────────────┘└─────────────────────────────────┘

┌ STATUS ────────────────────────────────────────────────────────────────────┐
│Status: Complete! RTF: 3.92x                                                │
└────────────────────────────────────────────────────────────────────────────┘
```

## Amdahl's Law Analysis

The benchmark clearly shows where optimization effort should focus:

| Step   | % of Total | Optimization Potential |
|--------|------------|------------------------|
| Decode | 62.3%      | **HIGH** - Autoregressive bottleneck |
| Model  | 15.6%      | Medium - Streaming load possible |
| Encode | 15.6%      | Medium - SIMD optimization |
| Others | 6.5%       | Low - Already fast |

**Key Insight**: Decode (H) dominates at 62.3%, making it the primary optimization target per Amdahl's Law.

## Makefile Targets

```bash
make bench-tui          # Run benchmark TUI (interactive)
make bench-tui-test     # Run TUI state machine tests (34 tests)
make bench-tui-render   # Run TUI render tests (25 tests)
```

## Criterion Benchmarks

For statistical benchmarking with regression detection:

```bash
# Full benchmark suite
cargo bench --bench inference

# Single benchmark
cargo bench --bench inference -- "greedy_decode"

# Generate HTML report
cargo bench --bench inference -- --save-baseline main
```

### Benchmark Results Location

- JSON results: `target/criterion/`
- HTML reports: `target/criterion/report/index.html`

## TUI Testing with Probar

The benchmark TUI is tested using probar's TUI testing framework:

### State Machine Tests (34 tests)

```bash
cargo test -p whisper-apr-demo-tests benchmark_tui_tests
```

Tests include:
- Initial state validation
- State transitions (Idle → Running → Completed)
- Step ordering (can't skip, can't go backwards)
- RTF calculation accuracy
- Error handling and recovery

### Render Tests (25 tests)

```bash
cargo test -p whisper-apr-demo-tests tui_render_tests
```

Tests include:
- Frame content validation
- Visual element presence
- State diff verification
- Terminal dimension handling

### Diagnostic Tests

Run with `--nocapture` to see actual frame output:

```bash
cd demos && cargo test -p whisper-apr-demo-tests diagnostic -- --nocapture
```

This displays:
- Full frame dumps for idle and completed states
- Line-by-line state diffs
- Timing target breakdown with Amdahl's Law analysis

## Performance Targets

| Model | Target RTF | Memory Peak |
|-------|------------|-------------|
| tiny  | ≤2.0x      | ≤150MB      |
| base  | ≤2.5x      | ≤350MB      |
| small | ≤4.0x      | ≤800MB      |

RTF = Real-Time Factor (processing time / audio duration). Lower is better.

## Continuous Benchmarking

The project uses criterion for regression detection in CI:

```yaml
# .github/workflows/bench.yml
- name: Run benchmarks
  run: cargo bench --bench inference -- --save-baseline pr-${{ github.event.number }}

- name: Compare to main
  run: cargo bench --bench inference -- --baseline main --load-baseline pr-${{ github.event.number }}
```
