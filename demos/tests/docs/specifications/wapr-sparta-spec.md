---
title: "THIS IS SPARTA" - Strict Streaming Validation Mode
issue: WAPR-SPARTA
status: In Progress
created: 2026-01-05T22:05:47.532602942+00:00
updated: 2026-01-05T22:10:00.000000000+00:00
---

# THIS IS SPARTA: Strict Streaming Validation Mode

**Ticket ID**: WAPR-SPARTA
**Status**: In Progress

## Summary

Implement "SPARTA mode" - an aggressive, no-mercy validation mode for streaming UX tests.
Like Leonidas kicking the Persian messenger into the pit, SPARTA mode rejects any test
that doesn't meet strict requirements with dramatic, unmistakable failure messages.

## Requirements

### Functional Requirements
- [x] `sparta!` macro for dramatic assertions
- [x] `SpartaValidator` with strict thresholds
- [x] Fail-fast behavior - no partial passes
- [x] Clear, dramatic error messages

### Non-Functional Requirements
- [x] Zero tolerance for flaky tests
- [x] Sub-100ms assertion overhead
- [x] Test coverage: 100% for SPARTA code

## Architecture

### Design Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    THIS IS SPARTA!                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Test Assertion ──► SpartaValidator ──► PASS or THE PIT   │
│                                                             │
│   Thresholds:                                               │
│   • SharedArrayBuffer: REQUIRED (no fallback)              │
│   • COOP/COEP: REQUIRED (no exceptions)                    │
│   • Latency: <100ms (no excuses)                           │
│   • State transitions: MUST occur (no waiting)             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### API Design

```rust
/// SPARTA mode - strict validation with no mercy
pub struct SpartaValidator {
    /// Maximum allowed latency (default: 100ms - THIS IS SPARTA)
    pub max_latency_ms: u64,
    /// Minimum required FPS (default: 30 - no stuttering in Sparta)
    pub min_fps: f64,
    /// Required state transitions (must see these or INTO THE PIT)
    pub required_states: Vec<&'static str>,
}

/// Dramatic assertion macro
macro_rules! sparta {
    ($cond:expr, $msg:expr) => {
        if !$cond {
            panic!("\n\n🔥 THIS IS SPARTA! 🔥\n\n{}\n\n⚔️  INTO THE PIT! ⚔️\n", $msg);
        }
    };
}
```

## Implementation Plan

### Phase 1: Core SPARTA Infrastructure
- [x] Create `sparta!` assertion macro
- [x] Implement `SpartaValidator` struct
- [x] Add strict threshold defaults

### Phase 2: Integration
- [x] Add SPARTA mode to streaming tests
- [x] Implement dramatic failure messages
- [x] Add SPARTA test runner

## Testing Strategy

### Unit Tests
- [x] `sparta!` macro passes on true
- [x] `sparta!` macro panics dramatically on false
- [x] `SpartaValidator` enforces all thresholds

### Integration Tests
- [x] Full streaming flow with SPARTA validation

## Success Criteria

- ✅ Tests fail LOUD and CLEAR when requirements not met
- ✅ No silent failures - every failure goes INTO THE PIT
- ✅ Zero tolerance mode works correctly
- ✅ Dramatic messages visible in test output

## References

- Film: "300" (2006) - "This is Sparta!" scene
- Philosophy: Fail fast, fail loud, no mercy for bugs
