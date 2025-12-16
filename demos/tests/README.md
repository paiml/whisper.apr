# Whisper.apr Demo Test Suite

## Overview

This directory contains comprehensive tests for the Whisper.apr demo applications, implementing the probar testing framework for WASM-first applications.

## Test Structure

```
tests/
├── src/
│   ├── lib.rs              # Test harness and shared utilities
│   ├── browser_tests.rs    # Cross-browser compatibility tests
│   ├── performance_tests.rs # RTF, latency, memory benchmarks
│   ├── pixel_tests.rs      # Visual regression tests
│   └── quality_gates.rs    # CI quality gate assertions
├── examples/
│   ├── coverage_pattern.rs # Coverage pattern examples
│   └── uat_browser.rs      # User acceptance test patterns
└── Cargo.toml
```

## Test Categories

### 1. Unit Tests (`cargo test --lib`)
- Audio processing pipeline validation
- Tokenizer correctness
- Model configuration parsing
- State machine transitions

### 2. Integration Tests
- End-to-end transcription flows
- WebWorker communication
- Model loading and caching
- Audio stream handling

### 3. Browser Tests (`wasm-pack test --headless`)
- Chrome, Firefox, Safari, Edge compatibility
- WASM SIMD feature detection
- SharedArrayBuffer support
- Web Audio API integration

### 4. Performance Tests
- Real-Time Factor (RTF) validation: < 2.0x for tiny model
- Memory budget compliance: < 150MB peak
- First-token latency: < 500ms
- Streaming chunk latency: < 100ms p95

### 5. Visual Regression Tests
- Baseline snapshots in `../snapshots/`
- Responsive viewport testing
- Dark mode variants
- State-based screenshot comparison

### 6. Accessibility Tests
- WCAG 2.1 AA compliance
- ARIA attribute validation
- Keyboard navigation
- Screen reader announcements

## Running Tests

```bash
# All unit tests
cargo test

# Browser tests (requires Chrome)
wasm-pack test --headless --chrome

# Performance benchmarks
cargo bench --bench inference

# Probar playbook validation
probar test --playbook ../playbooks/realtime-transcription.yaml

# Full test suite with coverage
cargo llvm-cov --html
```

## Test Rationale

### Why State Machine Testing?
The demos implement complex UI states (idle, loading, recording, processing, error). State machine testing ensures all transitions are valid and no illegal states are reachable.

### Why Visual Regression?
WASM rendering can vary across browsers. Pixel-perfect snapshots catch subtle rendering differences that unit tests miss.

### Why Performance Gates?
Real-time transcription requires strict latency budgets. Performance tests ensure we don't regress below usable RTF thresholds.

### Why Accessibility?
Speech-to-text is critical for accessibility users. Our demos must themselves be accessible to demonstrate inclusive design.

## CI Integration

Tests run automatically on:
- Pull requests (unit + integration)
- Merge to main (full suite + performance)
- Nightly (browser matrix + chaos testing)

See `.github/workflows/test.yml` for CI configuration.

## Adding New Tests

1. Add test file to `src/`
2. Export from `lib.rs`
3. Add to appropriate playbook if state-based
4. Update snapshots if visual: `probar snapshot --update`
5. Document rationale in this README
