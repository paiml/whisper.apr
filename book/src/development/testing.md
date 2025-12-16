# Testing

Whisper.apr follows **EXTREME TDD** methodology with comprehensive testing at multiple levels.

## Test Distribution

The project targets the following test distribution:
- **60%** Unit tests
- **30%** Property-based tests
- **10%** Integration tests

## Running Tests

```bash
# All tests (unit + property + integration + doc)
cargo test

# Fast tests with nextest (recommended)
make test-fast

# Unit tests only
cargo test --lib

# Property-based tests only
cargo test property_ --lib

# Doc tests
cargo test --doc
```

## Coverage

The project maintains **≥95% line coverage** using `cargo-llvm-cov`:

```bash
# Generate coverage report
make coverage

# View HTML report
open target/coverage/html/index.html
```

### Current Coverage Stats

| Module | Line Coverage |
|--------|---------------|
| audio/streaming.rs | 99.45% |
| audio/ring_buffer.rs | 98.94% |
| format/checksum.rs | 100.00% |
| simd.rs | 95.19% |
| **TOTAL** | **95.19%** |

## Test Categories

### Unit Tests

Located alongside the code in `#[cfg(test)]` modules:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_filterbank_new() {
        let mel = MelFilterbank::new(80, 400, 16000);
        assert_eq!(mel.n_mels(), 80);
    }
}
```

### Property-Based Tests

Using `proptest` for invariant validation:

```rust
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn property_softmax_sums_to_one(len in 4usize..128) {
            let logits: Vec<f32> = (0..len)
                .map(|i| (i as f32 * 0.1).sin())
                .collect();
            let probs = simd::softmax(&logits);
            let sum: f32 = probs.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-5);
        }
    }
}
```

### Integration Tests

End-to-end tests in `tests/` directory testing the full transcription pipeline.

## Makefile Targets

```bash
make test-fast    # Fast tests with nextest
make coverage     # Coverage with HTML report
make tier1        # Quick validation (<1s)
make tier2        # Pre-commit (<5s)
make tier3        # Pre-push (1-5min)
```

## Demo Coverage (Unified Pattern)

The demo applications in `demos/` use a unified coverage system following the probar/bashrs pattern:

### Running Demo Coverage

```bash
cd demos/

# Full coverage report
make coverage

# Quick summary
make coverage-summary

# Open HTML report
make coverage-open
```

### Coverage Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                 UNIFIED COVERAGE SYSTEM                      │
├──────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐    │
│  │ Rust Coverage │  │ GUI Coverage  │  │Pixel Coverage │    │
│  │  (llvm-cov)   │  │(UxTracker)    │  │(SSIM/PSNR)    │    │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘    │
│          └──────────┬───────┴──────────┬───────┘            │
│                     ▼                  ▼                    │
│          ┌────────────────────────────────────┐             │
│          │    cargo llvm-cov nextest          │             │
│          │   (unified instrumentation)        │             │
│          └────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────┘
```

### Key Pattern Components

1. **Nextest with llvm-cov**: `cargo llvm-cov --no-report nextest`
2. **Mold Linker Workaround**: Temporarily moves `~/.cargo/config.toml`
3. **Two-Phase Reporting**: Run tests first, then generate reports
4. **GUI Coverage via Probar**: `UxCoverageTracker` tests are instrumented

### Example GUI Coverage Test

```rust
use probar::gui_coverage;
use probar::ux_coverage::UxCoverageTracker;

fn demo_coverage() -> UxCoverageTracker {
    gui_coverage! {
        buttons: ["start_recording", "stop_recording", "clear"],
        inputs: ["audio_file_input"],
        screens: ["main", "recording"]
    }
}

#[test]
fn test_full_gui_coverage() {
    let mut gui = demo_coverage();
    gui.click("start_recording");
    gui.click("stop_recording");
    gui.click("clear");
    gui.input("audio_file_input");
    gui.visit("main");
    gui.visit("recording");

    assert!(gui.is_complete());
    assert!(gui.meets(95.0));
}
```

### Coverage Example

Run the coverage pattern example:

```bash
cargo run --example coverage_pattern -p whisper-apr-demo-tests
```

### Current Demo Coverage

| Demo | Line Coverage |
|------|---------------|
| realtime-transcription | 59% |
| realtime-translation | 76% |
| upload-transcription | 76% |
| upload-translation | 79% |

> **Note**: Remaining untested code (~20-40%) is browser-specific `web_sys` code
> that requires headless browser testing via probar's `BrowserController`.

## TUI Testing with Probar

The project uses probar's TUI testing framework for terminal UI validation.

### Running TUI Tests

```bash
# State machine tests (34 tests)
make bench-tui-test

# Render tests (25 tests)
make bench-tui-render

# Diagnostic tests with output
cd demos && cargo test -p whisper-apr-demo-tests diagnostic -- --nocapture
```

### Frame Capture Pattern

```rust
use probar::tui::{TuiFrame, expect_frame};
use ratatui::{backend::TestBackend, Terminal};

fn capture_app_frame(app: &TestApp, width: u16, height: u16) -> TuiFrame {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal.draw(|f| render_ui(f, app)).expect("draw");
    TuiFrame::from_buffer(terminal.backend().buffer(), 0)
}

#[test]
fn test_frame_contains_title() {
    let app = TestApp::new();
    let frame = capture_app_frame(&app, 80, 24);
    expect_frame(&frame).to_contain("PIPELINE PROGRESS");
}
```

### Frame Assertions

```rust
// Content assertions
expect_frame(&frame).to_contain("Status:");
expect_frame(&frame).not_to_contain("Error");
expect_frame(&frame).to_match_regex(r"RTF: \d+\.\d+x");

// State diff
let idle_frame = capture_app_frame(&idle_app, 80, 24);
let done_frame = capture_app_frame(&done_app, 80, 24);
let diff = idle_frame.diff(&done_frame);
assert!(diff.changed_lines() > 0);
```

### Diagnostic Output

The diagnostic tests dump actual frame content for debugging:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        IDLE STATE FRAME DUMP                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  0 │                                                                         ║
║  1 │  ┌ PIPELINE PROGRESS (REAL) ───────────────┐┌ LIVE METRICS ─────────┐   ║
║  2 │  │        [A]   Model         0.0ms        ││                       │   ║
...
```

## Mutation Testing

Validate test quality with `cargo-mutants`:

```bash
# Run mutation tests
make mutants

# Quick mutation test on changed files
make mutants-quick

# List mutants for a specific file
cargo mutants --list --file src/simd.rs
```
