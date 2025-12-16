---
title: apr-cli trace and probar commands for layer-by-layer debugging
issue: WAPR-CLI-003
status: Completed
created: 2025-12-16T02:03:18.868386026+00:00
updated: 2025-12-16T02:05:00.000000000+00:00
---

# WAPR-CLI-003: apr-cli trace and probar commands

**Ticket ID**: WAPR-CLI-003
**Status**: Completed

## Summary

Added two new commands to the apr-cli tool for layer-by-layer model debugging and visual regression testing integration. The `trace` command provides tensor statistics and anomaly detection, while the `probar` command exports layer snapshots for visual comparison testing.

## Requirements

### Functional Requirements
- [x] `apr trace` command for layer-by-layer analysis with tensor statistics
- [x] Anomaly detection for NaN, Inf, near-zero variance, large values
- [x] JSON output support for `trace` command
- [x] `apr probar` command for visual testing export
- [x] Histogram generation for layer activations
- [x] Golden reference comparison and diff report generation

### Non-Functional Requirements
- [x] Zero clippy warnings with `-D warnings`
- [x] Test coverage for all new structures (7 unit tests)
- [x] Integration with existing apr-cli test suite (26 tests pass)

## Architecture

### Design Overview

The trace and probar commands follow the existing apr-cli patterns using the commands module structure. Both commands parse APR model format metadata to extract layer information and generate statistics or visual artifacts.

### API Design

```rust
// trace.rs - Layer-by-layer analysis
pub(crate) struct LayerTrace {
    pub name: String,
    pub index: Option<usize>,
    pub input_stats: Option<TensorStats>,
    pub output_stats: Option<TensorStats>,
    pub weight_stats: Option<TensorStats>,
    pub anomalies: Vec<String>,
}

pub(crate) struct TensorStats {
    pub count: usize,
    pub mean: f32,
    pub std: f32,
    pub l2_norm: f32,
    pub min: f32,
    pub max: f32,
    pub max_abs: f32,
    pub nan_count: usize,
    pub inf_count: usize,
}

// probar.rs - Visual testing export
pub(crate) struct LayerSnapshot {
    pub name: String,
    pub index: usize,
    pub histogram: Vec<u32>,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub heatmap: Option<Vec<f32>>,
    pub heatmap_width: Option<usize>,
    pub heatmap_height: Option<usize>,
}
```

## Implementation Plan

### Phase 1: Foundation ✅
- [x] Create `commands/trace.rs` module
- [x] Create `commands/probar.rs` module
- [x] Add to mod.rs and main.rs

### Phase 2: Core Implementation ✅
- [x] Implement TensorStats with anomaly detection
- [x] Implement LayerTrace for layer-by-layer output
- [x] Implement LayerSnapshot for histogram export
- [x] Add JSON and text output formatting

### Phase 3: Integration ✅
- [x] Add `--verbose` flag support
- [x] Add `--layer` filter option
- [x] Add `--reference` comparison for trace
- [x] Add `--golden` comparison for probar

## Testing Strategy

### Unit Tests ✅
- [x] `test_tensor_stats_empty` - Empty slice handling
- [x] `test_tensor_stats_basic` - Basic statistics computation
- [x] `test_tensor_stats_with_nan` - NaN value handling
- [x] `test_anomaly_detection` - Large mean detection
- [x] `test_anomaly_detection_nan` - NaN count anomaly
- [x] `test_export_format_parse` - Format string parsing
- [x] `test_layer_snapshot_serialize` - JSON serialization

### Integration Tests ✅
- [x] All 26 existing integration tests pass
- [x] Commands work with APR1 (whisper.apr) format

## Success Criteria

- ✅ All acceptance criteria met
- ✅ Test coverage via 7 new unit tests + 26 integration tests
- ✅ Zero clippy warnings (`cargo clippy -- -D warnings`)
- ✅ Documentation in command `--help` output

## Usage Examples

```bash
# Layer trace with verbose stats
apr trace model.apr --verbose

# Layer trace with JSON output
apr trace model.apr --json

# Compare with reference model
apr trace model.apr --reference other.apr

# Filter specific layers
apr trace model.apr --layer "block_0"

# Export for probar visual testing
apr probar model.apr -o ./probar-export --format both

# Export with golden comparison
apr probar model.apr -o ./test --golden ./golden-ref
```

## References

- Toyota Way: Visualization - Make hidden problems visible
- Toyota Way: Standardization - Repeatable debugging between models
- probar visual regression testing framework
