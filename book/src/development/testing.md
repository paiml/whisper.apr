# Testing

Whisper.apr follows **EXTREME TDD** methodology with comprehensive testing at multiple levels.

## Test Distribution

The project targets the following test distribution:
- **60%** Unit tests
- **30%** Property-based tests
- **10%** Integration tests

## Running Tests

```bash
# Unit tests only (fast, no large models)
cargo test --lib

# Fast tests
make test-fast

# Property-based tests only
cargo test property_ --lib

# Doc tests
cargo test --doc

# Integration tests (feature-gated, requires large models)
cargo test --features integration-tests

# Run ignored heavy tests explicitly
cargo test --lib -- --ignored
```

### Test Isolation

Tests are organized into three tiers of isolation:

1. **Unit tests** (`cargo test --lib`): Fast, no large model allocation. Heavy tests that allocate large decoders are marked `#[ignore]` and skipped by default.
2. **Integration tests** (`cargo test --features integration-tests`): Gated behind the `integration-tests` feature flag. These load large model files and run end-to-end pipelines.
3. **WASM tests** (`wasm-pack test`): Browser-based tests requiring wasm-pack.

## Coverage

The project maintains **≥95% line coverage** using `cargo-llvm-cov`. Coverage uses `cargo llvm-cov test --lib` (not nextest) to avoid profraw file explosion:

```bash
# Generate coverage report with threshold check
make coverage

# CI coverage (LCOV output)
make coverage-ci

# View HTML report
make coverage-html
open target/coverage/html/index.html
```

The coverage pattern follows the paiml-mcp-agent-toolkit convention:
- `RUSTC_WRAPPER=` clears the mold linker (incompatible with instrumentation)
- `PROPTEST_CASES=2 QUICKCHECK_TESTS=2` keeps property tests fast
- `|| true` tolerates individual test failures
- Separate report step with `tee` for threshold checking via `bc`

### Current Coverage Stats

| Metric | Value |
|--------|-------|
| **Tests** | 2,920 |
| **Line coverage** | 97.92% |
| **Threshold** | 95% |

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

End-to-end tests in `tests/` directory testing the full transcription pipeline. These are gated behind the `integration-tests` feature flag to prevent accidental execution during normal development:

```bash
# Run integration tests explicitly
cargo test --features integration-tests
```

The feature-gated files include:
- `tests/ground_truth_tests.rs` — Ground truth validation against reference implementations
- `tests/integration_transcribe.rs` — Full transcription pipeline tests
- `tests/cli_parity_tests.rs` — CLI output parity tests
- `tests/pipeline_fuzz.rs` — Pipeline fuzz testing
- `tests/publish_integration.rs` — HuggingFace publishing workflow tests

## Makefile Targets

```bash
make test-fast    # Fast unit tests
make coverage     # Coverage with threshold check
make coverage-html # Coverage with HTML report
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
│          │    cargo llvm-cov test --lib       │             │
│          │   (unified instrumentation)        │             │
│          └────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────┘
```

### Key Pattern Components

1. **cargo llvm-cov test**: Uses `cargo llvm-cov test --lib` (not nextest) to avoid profraw explosion
2. **Mold Linker Workaround**: `RUSTC_WRAPPER=` clears the mold linker
3. **Two-Phase Reporting**: Run tests first, then generate reports with threshold check
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
