---
title: "WAPR-WORKER: Web Worker Architecture for Non-Blocking Inference"
issue: WAPR-WORKER
status: In Progress
created: 2026-01-05T22:30:00.000000000+00:00
updated: 2026-01-05T22:30:00.000000000+00:00
---

# WAPR-WORKER: Web Worker Architecture for Non-Blocking Inference

**Ticket ID**: WAPR-WORKER
**Status**: In Progress
**Priority**: Critical (P0)
**Complexity**: High

## Executive Summary

The current whisper.apr demo architecture executes Whisper inference on the main
thread, causing UI freezes and degraded user experience. This specification
defines a rigorous Web Worker architecture with SharedArrayBuffer communication,
AudioWorklet audio capture, and comprehensive validation via probador playbooks
and renacer performance tracing.

## Problem Statement

### Current Architecture (Anti-Pattern)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAIN THREAD (BLOCKED)                        │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ ScriptProc-  │───►│   Whisper    │───►│     UI       │      │
│  │ essorNode    │    │  Inference   │    │   Updates    │      │
│  │ (deprecated) │    │  (blocking)  │    │  (starved)   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         └───────────────────┴───────────────────┘               │
│                    ALL COMPETE FOR SAME THREAD                  │
└─────────────────────────────────────────────────────────────────┘
```

**Observed Symptoms:**
1. UI freezes during `transcribe_partial()` calls
2. Audio buffer underruns causing dropped samples
3. VU meter stuttering
4. State transitions delayed by inference latency
5. ScriptProcessorNode deprecation warnings

### Root Cause Analysis (Toyota Way: Five Whys)

1. **Why does the UI freeze?**
   - Whisper inference blocks the main thread

2. **Why does inference block the main thread?**
   - `spawn_local` uses JavaScript microtask queue, not true parallelism

3. **Why isn't inference on a separate thread?**
   - Web Worker architecture defined but not implemented

4. **Why wasn't it implemented?**
   - Complexity of SharedArrayBuffer + COOP/COEP requirements

5. **Why is SharedArrayBuffer complex?**
   - Spectre mitigations require cross-origin isolation headers

## Proposed Architecture

### Target Architecture (Engineering Solution)

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN THREAD                             │
│  ┌──────────────┐              ┌──────────────┐                │
│  │  UI Updates  │◄────────────►│   Message    │                │
│  │  (60 FPS)    │              │   Handler    │                │
│  └──────────────┘              └──────┬───────┘                │
│                                       │ postMessage            │
└───────────────────────────────────────┼─────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │     SharedArrayBuffer Ring Buffer     │
                    │  ┌─────────────────────────────────┐  │
                    │  │ [Atomics.wait/notify signaling] │  │
                    │  └─────────────────────────────────┘  │
                    └───────────────────┼───────────────────┘
                                        │
┌───────────────────────────────────────┼─────────────────────────┐
│                    AUDIO WORKLET THREAD                         │
│  ┌──────────────┐              ┌──────────────┐                │
│  │  Audio       │─────────────►│   Ring       │                │
│  │  Capture     │  128 samples │   Buffer     │                │
│  │  (real-time) │              │   Writer     │                │
│  └──────────────┘              └──────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      WORKER THREAD                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Ring       │───►│   Whisper    │───►│   Result     │      │
│  │   Buffer     │    │   Inference  │    │   Encoder    │      │
│  │   Reader     │    │   (trueno)   │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                             │                                   │
│                      ┌──────┴──────┐                           │
│                      │   renacer   │                           │
│                      │   tracing   │                           │
│                      └─────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### Component Specifications

#### 1. AudioWorklet Processor

**File**: `demos/www-demo/src/audio_worklet.rs`

```rust
/// AudioWorklet processor for real-time audio capture
///
/// Replaces deprecated ScriptProcessorNode with low-latency
/// AudioWorklet running on dedicated audio thread.
///
/// ## Performance Constraints
/// - Process callback: <3ms (128 samples @ 44.1kHz = 2.9ms)
/// - Zero allocations in process() hot path
/// - Lock-free ring buffer writes
#[wasm_bindgen]
pub struct WhisperAudioProcessor {
    ring_buffer: SharedRingBuffer,
    sample_rate: u32,
    channels: u8,
}
```

**Latency Budget** (per audio quantum):
| Operation | Budget | Measured |
|-----------|--------|----------|
| Copy samples | 0.1ms | TBD |
| Ring buffer write | 0.2ms | TBD |
| Atomics.notify | 0.1ms | TBD |
| **Total** | <0.5ms | TBD |

#### 2. SharedArrayBuffer Ring Buffer

**File**: `demos/www-demo/src/ring_buffer.rs`

```rust
/// Lock-free SPSC ring buffer over SharedArrayBuffer
///
/// Single-Producer (AudioWorklet) Single-Consumer (Worker) design
/// eliminates mutex contention and enables wait-free audio writes.
///
/// ## Memory Layout
/// ```text
/// ┌────────────────────────────────────────────────────────────┐
/// │ Header (64 bytes, cache-line aligned)                      │
/// │ ┌──────────────┬──────────────┬──────────────┬───────────┐ │
/// │ │ write_idx    │ read_idx     │ capacity     │ flags     │ │
/// │ │ (Atomic u32) │ (Atomic u32) │ (u32)        │ (u32)     │ │
/// │ └──────────────┴──────────────┴──────────────┴───────────┘ │
/// ├────────────────────────────────────────────────────────────┤
/// │ Data (N * sizeof(f32) bytes)                               │
/// │ ┌──────────────────────────────────────────────────────┐   │
/// │ │ f32 samples...                                       │   │
/// │ └──────────────────────────────────────────────────────┘   │
/// └────────────────────────────────────────────────────────────┘
/// ```
///
/// ## References
/// - Lamport, L. (1977). "Proving the Correctness of Multiprocess
///   Programs". IEEE TSE. [Lock-free queue foundations]
/// - Herlihy & Shavit (2008). "The Art of Multiprocessor Programming".
///   Ch. 10: Concurrent Queues. [SPSC queue design]
pub struct SharedRingBuffer {
    buffer: js_sys::SharedArrayBuffer,
    header: js_sys::Int32Array,
    data: js_sys::Float32Array,
    capacity: usize,
}
```

**Capacity Sizing**:
- Audio: 16kHz * 3s = 48,000 samples = 192KB
- With 2x headroom: 384KB ring buffer
- Enables 3-second chunks without blocking

#### 3. Worker Thread Manager

**File**: `demos/www-demo/src/worker_manager.rs`

```rust
/// Manages Web Worker lifecycle and message passing
///
/// ## State Machine
/// ```text
///                    ┌─────────────────┐
///                    │  Uninitialized  │
///                    └────────┬────────┘
///                             │ spawn()
///                             ▼
///                    ┌─────────────────┐
///              ┌────►│    Loading      │◄────┐
///              │     └────────┬────────┘     │
///              │              │ ready        │ error
///              │              ▼              │
///              │     ┌─────────────────┐     │
///              │     │     Ready       │─────┤
///              │     └────────┬────────┘     │
///              │              │ transcribe   │
///              │              ▼              │
///              │     ┌─────────────────┐     │
///              └─────│  Transcribing   │─────┘
///                    └─────────────────┘
/// ```
#[wasm_bindgen]
pub struct WorkerManager {
    worker: web_sys::Worker,
    state: WorkerState,
    ring_buffer: SharedRingBuffer,
    pending_requests: HashMap<u32, oneshot::Sender<TranscriptionResult>>,
}
```

#### 4. Inference Worker Entry Point

**File**: `demos/www-demo/src/worker_entry.rs`

```rust
/// Worker thread entry point
///
/// Compiled to separate WASM module loaded by Web Worker.
/// Handles message protocol and runs Whisper inference.
///
/// ## Message Protocol (defined in worker.rs)
///
/// Request: { type: "transcribe", requestId: u32, options: {...} }
/// Response: { type: "result", requestId: u32, text: "...", segments: [...] }
///
/// ## renacer Integration
///
/// All inference operations are instrumented with tracing spans:
/// - `whisper.inference` - Top-level inference span
/// - `whisper.mel` - Mel spectrogram computation
/// - `whisper.encoder` - Encoder forward pass
/// - `whisper.decoder` - Decoder forward pass (per token)
#[wasm_bindgen(start)]
pub fn worker_main() {
    // Initialize tracing for renacer
    tracing_wasm::set_as_global_default();

    // Set up message handler
    let global = js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>();
    // ...
}
```

## Performance Requirements

### Latency Targets (SPARTA-Enforced)

| Metric | Target | Critical | Measurement Method |
|--------|--------|----------|-------------------|
| Audio-to-UI latency | <100ms | <200ms | renacer span timing |
| First partial result | <500ms | <1000ms | probador playbook |
| Chunk transcription | <1500ms | <3000ms | renacer span timing |
| UI frame budget | <16ms | <33ms | requestAnimationFrame timing |
| Ring buffer write | <0.5ms | <1ms | AudioWorklet timing |

### Throughput Targets

| Metric | Target | Critical |
|--------|--------|----------|
| Real-Time Factor (RTF) | <1.5 | <2.0 |
| Dropped audio chunks | 0 | 0 |
| Missed UI frames | <1% | <5% |

## Peer-Reviewed Citations

### Web Workers and SharedArrayBuffer

1. **Spectre Mitigations**
   - Kocher, P., et al. (2019). "Spectre Attacks: Exploiting Speculative
     Execution". IEEE S&P. DOI: 10.1109/SP.2019.00002
   - *Justification*: Explains why SharedArrayBuffer requires COOP/COEP
     headers for cross-origin isolation.

2. **SharedArrayBuffer Memory Model**
   - ECMA-262 (2024). "ECMAScript Language Specification". Section 25.4:
     SharedArrayBuffer Objects.
   - *Justification*: Defines memory ordering semantics for Atomics.

3. **Lock-Free Data Structures**
   - Herlihy, M. & Shavit, N. (2008). "The Art of Multiprocessor
     Programming". Morgan Kaufmann. ISBN: 978-0123705914.
   - *Justification*: Foundation for SPSC ring buffer design.

4. **Lamport's SPSC Queue**
   - Lamport, L. (1977). "Proving the Correctness of Multiprocess Programs".
     IEEE Trans. Software Engineering, SE-3(2), pp. 125-143.
   - *Justification*: Original lock-free queue proof.

### Audio Processing

5. **AudioWorklet Specification**
   - W3C (2021). "Web Audio API". Section 5.4: AudioWorklet Interface.
     https://www.w3.org/TR/webaudio/#audioworklet
   - *Justification*: Defines real-time audio processing constraints.

6. **Real-Time Audio Constraints**
   - Brandt, E. & Dannenberg, R.B. (1998). "Low-Latency Music Software
     Using Off-The-Shelf Operating Systems". ICMC Proceedings.
   - *Justification*: <10ms latency requirements for interactive audio.

7. **ScriptProcessorNode Deprecation**
   - Adenot, P. & Wilson, C. (2018). "Enter AudioWorklet". Google
     Developers. https://developer.chrome.com/blog/audio-worklet/
   - *Justification*: Main-thread audio processing causes glitches.

### WASM Threading

8. **WASM Threads Proposal**
   - WebAssembly Community Group (2023). "Threading Proposal for
     WebAssembly". https://github.com/WebAssembly/threads
   - *Justification*: Defines shared memory and atomic operations.

9. **wasm-bindgen-futures**
   - Rust and WebAssembly Working Group (2024). "wasm-bindgen Guide".
     https://rustwasm.github.io/wasm-bindgen/
   - *Justification*: Async/await integration patterns.

### ASR Performance

10. **Whisper Model Architecture**
    - Radford, A., et al. (2023). "Robust Speech Recognition via
      Large-Scale Weak Supervision". ICML.
    - *Justification*: Baseline performance expectations.

## Validation Strategy

### Probador Playbook: Worker Integration

**File**: `demos/playbooks/worker-integration.yaml`

```yaml
version: "1.0"
name: "Worker Integration Validation"
description: "WAPR-WORKER: Validates Web Worker architecture"

machine:
  id: "worker_integration"
  initial: "uninitialized"

  states:
    uninitialized:
      invariants:
        - description: "No worker spawned"
          condition: "worker_count() == 0"

    worker_spawning:
      invariants:
        - description: "Worker script loading"
          condition: "worker_state() == 'loading'"
        - description: "SharedArrayBuffer created"
          condition: "shared_buffer_exists()"

    worker_ready:
      invariants:
        - description: "Worker reports ready"
          condition: "worker_state() == 'ready'"
        - description: "Ring buffer initialized"
          condition: "ring_buffer_capacity() > 0"
        - description: "Main thread not blocked"
          condition: "main_thread_fps() >= 30"

    audio_streaming:
      invariants:
        - description: "AudioWorklet active"
          condition: "audio_worklet_running()"
        - description: "Ring buffer receiving samples"
          condition: "ring_buffer_write_count() > 0"
        - description: "No buffer underruns"
          condition: "buffer_underrun_count() == 0"

    transcribing:
      invariants:
        - description: "Worker processing"
          condition: "worker_state() == 'transcribing'"
        - description: "UI remains responsive"
          condition: "main_thread_fps() >= 30"
        - description: "Partial results flowing"
          condition: "partial_result_count() > 0 || elapsed_ms() < 1000"

  transitions:
    - from: "uninitialized"
      to: "worker_spawning"
      event: "init_called"

    - from: "worker_spawning"
      to: "worker_ready"
      event: "worker_ready_message"
      timeout_ms: 5000

    - from: "worker_ready"
      to: "audio_streaming"
      event: "start_recording"

    - from: "audio_streaming"
      to: "transcribing"
      event: "chunk_ready"

    - from: "transcribing"
      to: "audio_streaming"
      event: "result_received"

  forbidden:
    - from: "transcribing"
      to: "uninitialized"
      reason: "Cannot reset during transcription"

performance:
  # UI responsiveness during inference
  main_thread_fps_min: 30
  main_thread_fps_critical: 15

  # Worker latency
  first_partial_ms: 500
  first_partial_critical_ms: 1000

  # Ring buffer health
  max_buffer_fullness_percent: 80
  underrun_tolerance: 0
```

### Renacer Tracing Integration

**Span Hierarchy**:

```
whisper.session [10.5s]
├── whisper.worker.spawn [150ms]
│   ├── worker.script.load [80ms]
│   └── worker.wasm.compile [70ms]
├── whisper.audio.capture [10.2s]
│   └── audio.worklet.process [×48000, 0.02ms avg]
├── whisper.inference [×3 chunks]
│   ├── whisper.chunk.0 [1.2s]
│   │   ├── whisper.mel [45ms]
│   │   ├── whisper.encoder [380ms]
│   │   │   ├── encoder.conv1 [12ms]
│   │   │   ├── encoder.conv2 [15ms]
│   │   │   └── encoder.transformer [×4 layers, 88ms each]
│   │   └── whisper.decoder [775ms]
│   │       └── decoder.step [×25 tokens, 31ms avg]
│   ├── whisper.chunk.1 [1.1s]
│   └── whisper.chunk.2 [1.0s]
└── whisper.results.aggregate [5ms]
```

**Performance Assertions** (renacer hooks):

```rust
#[renacer::assert(duration_ms < 100)]
fn ring_buffer_write(samples: &[f32]) { ... }

#[renacer::assert(duration_ms < 1500)]
fn transcribe_chunk(audio: &[f32]) -> TranscriptionResult { ... }

#[renacer::assert(allocations == 0)]
fn audio_worklet_process(input: &[f32], output: &mut [f32]) { ... }
```

### SPARTA Validation

```rust
/// SPARTA enforces zero-tolerance performance requirements
fn validate_worker_performance(metrics: &WorkerMetrics) {
    let sparta = SpartaValidator::new();

    // UI must never freeze
    sparta_fmt!(
        metrics.main_thread_fps >= 30.0,
        "Main thread FPS {:.1} < 30! UI is freezing!",
        metrics.main_thread_fps
    );

    // No dropped audio
    sparta!(
        metrics.buffer_underruns == 0,
        "Audio buffer underruns detected! Ring buffer too small or worker too slow!"
    );

    // Latency within bounds
    sparta_fmt!(
        metrics.first_partial_ms <= 500,
        "First partial took {}ms > 500ms! Worker initialization too slow!",
        metrics.first_partial_ms
    );
}
```

## Implementation Plan

### Phase 1: AudioWorklet Migration (2 files)
- [ ] Create `audio_worklet.rs` processor
- [ ] Replace ScriptProcessorNode in demo
- [ ] Validate with probador: audio capture works

### Phase 2: Ring Buffer Implementation (1 file)
- [ ] Implement `SharedRingBuffer` with Atomics
- [ ] Unit tests for SPSC correctness
- [ ] Validate: zero allocations in hot path

### Phase 3: Worker Integration (3 files)
- [ ] Create `worker_entry.rs` WASM entry point
- [ ] Implement `WorkerManager` message passing
- [ ] Wire up in demo `lib.rs`

### Phase 4: Validation & Tracing
- [ ] Add renacer spans to all components
- [ ] Run probador playbook validation
- [ ] SPARTA performance enforcement

### Phase 5: Documentation
- [ ] Update architecture diagrams
- [ ] Performance benchmark results
- [ ] Deployment guide for COOP/COEP headers

## Success Criteria

1. **UI Responsiveness**: Main thread maintains 60 FPS during transcription
2. **Audio Quality**: Zero buffer underruns during 60-second recording
3. **Latency**: First partial result <500ms from speech onset
4. **Tracing**: All inference spans visible in renacer output
5. **Playbook**: Worker integration playbook passes 100%
6. **SPARTA**: All performance thresholds met

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Browser compatibility | Medium | High | Feature detection + fallback |
| SharedArrayBuffer security | Low | Critical | Strict COOP/COEP enforcement |
| Ring buffer sizing | Medium | Medium | Dynamic resizing with metrics |
| Worker compilation time | Medium | Low | Precompile + cache WASM |

## References

See [Peer-Reviewed Citations](#peer-reviewed-citations) section above.
