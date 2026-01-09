---
title: Streaming UX Bricks: WAPR-STREAMING-UX-001
issue: WAPR-STREAMING-UX-001
status: Complete
created: 2026-01-09T00:47:09.572443278+00:00
updated: 2026-01-09T01:30:00.000000000+00:00
---

# Streaming UX Bricks: WAPR-STREAMING-UX-001 Specification

**Ticket ID**: WAPR-STREAMING-UX-001
**Status**: Complete

## Summary

Implemented three new Brick components for real-time streaming UX in the whisper.apr demo:
- `ChunkProgressBrick`: Displays "Processing chunk 3/N..." with live buffer visualization
- `TimingStatsBrick`: Shows RTF, latency per chunk, and sparkline history graph
- `PerformanceStatsBrick`: Displays memory usage, model status, and system health

## Requirements

### Functional Requirements
- [x] ChunkProgressBrick displays chunk processing state (Waiting/Buffering/Processing/Paused/Complete)
- [x] ChunkProgressBrick shows buffer fill level indicator
- [x] TimingStatsBrick calculates and displays RTF per chunk
- [x] TimingStatsBrick shows average latency and RTF over time
- [x] TimingStatsBrick generates SVG sparkline for RTF history
- [x] PerformanceStatsBrick displays WASM heap memory usage
- [x] PerformanceStatsBrick shows model loading state
- [x] PerformanceStatsBrick displays capability badges (GPU/SIMD/Worker)

### Non-Functional Requirements
- [x] Performance: 16ms render budget for 60fps updates
- [x] Test coverage: 200 tests passing (all brick tests)

## Architecture

### Design Overview

Each brick follows the PROBAR-SPEC-009 Brick Architecture pattern:
1. State struct holds all data
2. Brick struct wraps state and implements `Brick + Widget` traits
3. `to_html()` generates semantic HTML with data-testid attributes
4. `to_css()` generates component-scoped CSS
5. Comprehensive unit tests verify all functionality

### API Design

```rust
// ChunkProgressBrick - Real-time chunk processing display
pub struct ChunkProgressBrick {
    state: ChunkProgressState,
}

pub struct ChunkProgressState {
    pub state: ChunkState,           // Waiting/Buffering/Processing/Paused/Complete
    pub current_chunk: u32,           // 1-based for display
    pub total_chunks: Option<u32>,    // Known for file uploads
    pub buffer_fill: f32,             // 0.0-1.0
    pub recent_chunks: Vec<ChunkStats>, // RTF history
}

// TimingStatsBrick - RTF and latency statistics
pub struct TimingStatsBrick {
    state: TimingStatsState,
}

pub struct TimingStatsState {
    measurements: VecDeque<TimingMeasurement>,
    pub total_audio_secs: f32,
    pub total_processing_ms: f32,
    pub best_rtf: Option<f32>,
    pub worst_rtf: Option<f32>,
    pub target_rtf: f32,
}

// PerformanceStatsBrick - System performance metrics
pub struct PerformanceStatsBrick {
    state: PerformanceStatsState,
}

pub struct PerformanceStatsState {
    pub memory: MemoryStats,         // Heap usage tracking
    pub model_state: ModelLoadState, // NotLoaded/Loading/Ready/Failed
    pub current_rtf: Option<f32>,
    pub gpu_available: bool,
    pub simd_available: bool,
    pub worker_active: bool,
}
```

## Implementation

### Files Created
- `demos/www-demo/src/bricks/chunk_progress.rs` - ChunkProgressBrick with buffer visualization
- `demos/www-demo/src/bricks/timing_stats.rs` - TimingStatsBrick with sparkline graph
- `demos/www-demo/src/bricks/performance_stats.rs` - PerformanceStatsBrick with health status

### Module Exports
Updated `demos/www-demo/src/bricks/mod.rs` to export:
- `ChunkProgressBrick`, `ChunkProgressState`, `ChunkState`, `ChunkStats`
- `TimingStatsBrick`, `TimingStatsState`, `TimingMeasurement`
- `PerformanceStatsBrick`, `PerformanceStatsState`, `MemoryStats`, `MemoryUnit`, `ModelLoadState`, `HealthStatus`

## Testing Strategy

### Unit Tests (All Passing)
- `test_chunk_state_display_name` - ChunkState enum display names
- `test_chunk_state_is_active` - Active state detection
- `test_chunk_stats_rtf` - RTF calculation
- `test_chunk_progress_state_transitions` - State machine transitions
- `test_timing_measurement_rtf` - Timing measurement RTF calculation
- `test_timing_stats_state_average_rtf` - Average RTF across chunks
- `test_timing_stats_state_best_worst_rtf` - Best/worst RTF tracking
- `test_memory_stats_usage_percent` - Memory percentage calculation
- `test_memory_stats_peak_tracking` - Peak memory tracking
- `test_performance_stats_state_health_status` - Health status determination
- `test_brick_verification` - Brick verification passes
- `test_brick_to_html_*` - HTML output for all states
- `test_brick_budget` - 16ms render budget

## Success Criteria

- ✅ All acceptance criteria met
- ✅ Test coverage: 200 tests passing
- ✅ Zero clippy warnings
- ✅ Documentation complete (module docs, struct docs, function docs)
- ✅ Spec 1.0-whisper-apr.md updated to v1.3.2 with P1 items marked complete

## References

- [PROBAR-SPEC-009: Brick Architecture](../../../../docs/specifications/1.0-whisper-apr.md#15-brick-architecture)
- [Section 9.3: Recording Mode Visual Requirements](../../../../docs/specifications/1.0-whisper-apr.md#93-recording-mode-visual-requirements)
- [Section 9.4: Upload Mode Visual Requirements](../../../../docs/specifications/1.0-whisper-apr.md#94-upload-mode-visual-requirements)
