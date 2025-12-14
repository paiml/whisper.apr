# WAPR-SPEC-010: Async Worker-Based Real-Time Transcription

**Status:** DRAFT - Awaiting Review
**Version:** 0.2.1
**Authors:** Claude Code, Noah
**Created:** 2024-12-14
**Toyota Way Principle:** Genchi Genbutsu (Go and See) + Jidoka (Automation with Human Touch)

---

## Executive Summary

This specification defines a production-grade, real-time speech transcription system using Web Workers for non-blocking inference. The design eliminates UI freezing, enables continuous audio capture, and provides rich observability through renacer tracing integration.

**Problem Statement:** Current implementation blocks the main thread during transcription, causing browser timeout ("Script terminated by timeout") after ~10 seconds of inference on 3-second audio chunks.

**Root Cause:** `model.transcribe()` is synchronous and executes on the main thread, blocking all UI updates, audio callbacks, and event processing.

**Solution:** Dedicated Web Worker architecture with message-passing for audio/results, following the StreamYard/OBS pattern of separating capture from processing.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Current State Analysis](#2-current-state-analysis)
3. [Target State Design](#3-target-state-design)
4. [Implementation Phases](#4-implementation-phases)
5. [Testing Strategy](#5-testing-strategy)
6. [Tracing & Observability](#6-tracing--observability)
7. [Performance Targets](#7-performance-targets)
8. [Golden Rules Compliance](#8-golden-rules-compliance)
9. [References](#9-references)
10. [Appendices](#10-appendices)

---

## 1. Architecture

### 1.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MAIN THREAD                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │  Microphone │───>│   Audio     │───>│   Ring      │                  │
│  │  MediaStream│    │   Worklet   │    │   Buffer    │                  │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                  │
│                                               │                          │
│  ┌─────────────┐    ┌─────────────┐          │ postMessage              │
│  │     UI      │<───│   State     │<─────────┤ (Float32Array)           │
│  │   Updates   │    │   Machine   │          │                          │
│  └─────────────┘    └─────────────┘          │                          │
│         ▲                  ▲                  │                          │
│         │                  │                  ▼                          │
│         │           postMessage        ┌─────────────┐                   │
│         │           (result)           │   Worker    │                   │
│         └──────────────────────────────│   Bridge    │                   │
│                                        └─────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
                                               │
                          Transferable Object  │ (Float32Array)
                                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRANSCRIPTION WORKER                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │   Message   │───>│  WhisperApr │───>│   Result    │                  │
│  │   Handler   │    │    Model    │    │   Encoder   │                  │
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
│                            │                                             │
│                     ┌──────┴──────┐                                      │
│                     │   Rayon     │  (wasm-bindgen-rayon)                │
│                     │   Thread    │                                      │
│                     │    Pool     │                                      │
│                     └─────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Message Protocol

```rust
/// Worker-bound messages
enum WorkerCommand {
    LoadModel { data: Uint8Array },
    Transcribe {
        audio: Float32Array,
        chunk_id: u32,
        /// Context management for continuous transcription
        session_id: String,
        prompt_tokens: Vec<u32>,
        is_final: bool,
    },
    SetOptions { language: Option<String>, task: Task },
    Shutdown,
}

/// Main-thread-bound messages
enum WorkerResult {
    Ready,
    ModelLoaded { size_mb: f64, load_time_ms: f64 },
    Transcription { chunk_id: u32, text: String, rtf: f64 },
    Error { chunk_id: Option<u32>, message: String },
    Metrics { queue_depth: usize, avg_latency_ms: f64 },
}
```

### 1.3 Threading Model

| Component | Thread | Blocking Allowed | Tracing Level |
|-----------|--------|------------------|---------------|
| Audio Capture | Main | NO | Light |
| UI Updates | Main | NO | Light |
| State Machine | Main | NO | Medium |
| Worker Bridge | Main | NO | Medium |
| Model Loading | Worker | YES | Full |
| Transcription | Worker | YES | Full |
| Rayon Pool | Worker-spawned | YES | Full |

---

## 2. Current State Analysis

### 2.1 What Was Implemented

| Component | Status | Notes |
|-----------|--------|-------|
| `whisper-apr/parallel` feature | ✅ Complete | Cargo.toml updated |
| `src/wasm/threading.rs` | ✅ Complete | Thread pool, COOP/COEP detection |
| `probar --cross-origin-isolated` | ✅ Complete | COOP/COEP headers, Issue #11 |
| `worker-transcription` demo crate | ✅ Skeleton | Needs full implementation |
| Worker message protocol | ⚠️ Partial | `worker.rs` created, not wired |
| Main thread integration | ❌ Not started | Still uses blocking transcribe |

### 2.2 Root Cause of Timeout

```
Timeline of failure:
─────────────────────────────────────────────────────────────────────────
0ms     Audio callback fires, 3s chunk ready
        ↓
10ms    spawn_local schedules transcription (setTimeout(0))
        ↓
20ms    Event loop yields, callback scheduled
        ↓
30ms    transcribe() starts executing
        ↓
        ════════════════════════════════════════════════════
        │  MAIN THREAD BLOCKED - NO UI, NO AUDIO CALLBACKS │
        ════════════════════════════════════════════════════
        ↓
~10000ms Browser terminates script (slow script timeout)
```

### 2.3 Lessons Learned (Hansei - 反省)

1. **spawn_local is not parallelism** - It defers but doesn't offload
2. **Rayon doesn't fix main thread blocking** - Parallelizes internal ops only
3. **setTimeout(0) is insufficient** - Still blocks when callback runs
4. **Model loading must happen in worker** - Can't share across threads easily
5. **COOP/COEP required for SharedArrayBuffer** - Server config critical

---

## 3. Target State Design

### 3.1 Component Specification

#### 3.1.1 TranscriptionWorker (Rust)

```rust
// demos/realtime-transcription/src/worker.rs

pub struct TranscriptionWorker {
    model: Option<WhisperApr>,
    options: TranscribeOptions,
    metrics: WorkerMetrics,
}

impl TranscriptionWorker {
    /// Initialize worker, set up message handler
    pub fn init() -> Self;

    /// Load model from bytes (blocking in worker is OK)
    pub fn load_model(&mut self, bytes: &[u8]) -> Result<ModelInfo, WorkerError>;

    /// Process audio chunk (blocking in worker is OK)
    pub fn transcribe(&self, audio: &[f32], chunk_id: u32) -> Result<TranscriptionResult, WorkerError>;

    /// Get current metrics
    pub fn metrics(&self) -> WorkerMetrics;
}
```

#### 3.1.2 WorkerBridge (Rust, Main Thread)

```rust
// demos/realtime-transcription/src/bridge.rs

pub struct WorkerBridge {
    worker: web_sys::Worker,
    pending: HashMap<u32, PendingRequest>,
    on_result: Closure<dyn Fn(JsValue)>,
}

impl WorkerBridge {
    /// Create worker from same WASM module
    pub async fn new() -> Result<Self, BridgeError>;

    /// Send model bytes to worker
    pub async fn load_model(&self, bytes: &[u8]) -> Result<ModelInfo, BridgeError>;

    /// Queue audio for transcription (non-blocking)
    pub fn transcribe(&mut self, audio: &[f32]) -> u32; // Returns chunk_id

    /// Register callback for results
    pub fn on_transcription<F: Fn(TranscriptionResult) + 'static>(&mut self, f: F);
}
```

#### 3.1.3 State Machine

```rust
pub enum DemoState {
    Uninitialized,
    InitializingWorker,
    LoadingModel { progress: f32 },
    Ready,
    Recording { chunks_sent: u32, chunks_received: u32 },
    Paused,
    Error { message: String, recoverable: bool },
}

// All transitions must be tested
pub enum DemoEvent {
    WorkerReady,
    ModelLoaded(ModelInfo),
    StartRecording,
    StopRecording,
    AudioChunkReady(Vec<f32>),
    TranscriptionReceived(TranscriptionResult),
    ErrorOccurred(DemoError),
}
```

### 3.2 Streaming Best Practices (StreamYard/OBS Pattern)

Following industry-standard streaming architectures [1][2]:

1. **Capture-Process Separation**: Audio capture NEVER waits for processing
2. **Bounded Queues**: Drop oldest chunks if queue exceeds N (backpressure)
3. **Chunk Overlap**: **200ms overlap** (increased from 50ms)
    - **Rationale**: 50ms is insufficient for diphthongs and fricatives.
    - **Phonetic Constraints**:
    
      | Phoneme Class | Avg Duration | Risk at 50ms |
      |---------------|--------------|--------------|
      | Plosives      | 20-30ms      | Low          |
      | Fricatives    | 100-150ms    | High         |
      | Diphthongs    | 150-300ms    | Critical     |

4. **Adaptive Chunk Size**: Reduce chunk size if RTF > 0.8
5. **Graceful Degradation**: Show "Processing..." instead of freezing

```rust
pub struct AdaptiveChunker {
    base_duration: f32,      // 1.5s default
    min_duration: f32,       // 0.5s minimum
    max_queue_depth: usize,  // 3 chunks max
    overlap_samples: usize,  // 3200 samples (200ms @ 16kHz)
}
```

---

## 4. Implementation Phases

### Phase 1: Worker Foundation (Sprint 1)

| Task | Testable Assertion | Probar Coverage |
|------|-------------------|-----------------|
| Create `WorkerBridge::new()` | Worker initializes within 500ms | Unit + Browser |
| Implement message protocol | Round-trip latency < 5ms | Unit |
| Add worker error handling | All errors propagate to main | Unit + E2E |
| Export `worker_entry` | Function exists in WASM | Unit |

**Exit Criteria:** Worker can be created, receive ping, respond with pong.

### Phase 2: Model Loading (Sprint 2)

| Task | Testable Assertion | Probar Coverage |
|------|-------------------|-----------------|
| Transfer model bytes | 37MB transfers in < 2s | Browser |
| Load model in worker | Model ready event fires | Browser |
| Report loading progress | Progress updates 10+ times | Browser |
| Handle load failures | Error message displayed | E2E + Pixel |

**Exit Criteria:** Model loads in worker, main thread never blocks.

### Phase 3: Transcription Pipeline (Sprint 3)

| Task | Testable Assertion | Probar Coverage |
|------|-------------------|-----------------|
| Audio chunk transfer | Float32Array received intact | Unit |
| Transcription execution | Result returned for each chunk | Browser |
| Result display | Text appears in transcript div | E2E + Pixel |
| RTF calculation | RTF logged for each chunk | Browser |

**Exit Criteria:** Say "hello world", see "hello world" in transcript.

### Phase 4: Robustness (Sprint 4)

| Task | Testable Assertion | Probar Coverage |
|------|-------------------|-----------------|
| Queue management | Queue never exceeds 3 | Unit |
| Chunk dropping | Oldest dropped, logged | Unit |
| Error recovery | Can restart after error | E2E |
| Memory stability | No growth over 100 chunks | Browser |

**Exit Criteria:** 5-minute continuous transcription without degradation.

---

## 5. Testing Strategy

### 5.1 Test Pyramid

```
                    ┌───────────────┐
                    │  Pixel Tests  │  (5%)
                    │   via Probar  │
                    └───────┬───────┘
                            │
                ┌───────────┴───────────┐
                │     E2E Browser       │  (15%)
                │   via Probar + CDP    │
                └───────────┬───────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │          Integration Tests            │  (30%)
        │    Worker ↔ Main communication        │
        └───────────────────┬───────────────────┘
                            │
┌───────────────────────────┴───────────────────────────┐
│                    Unit Tests                          │  (50%)
│   State machine, message encoding, audio processing   │
└────────────────────────────────────────────────────────┘
```

### 5.2 Golden Rule Compliance Matrix

| Rule | Implementation | Verification |
|------|---------------|--------------|
| 100% testable | All public APIs have tests | `cargo llvm-cov` ≥ 95% |
| GUI testing | Probar `UxCoverageTracker` | Button coverage = 100% |
| Pixel testing | Probar SSIM/PSNR/CIEDE2000 | Δ < 0.01 for known states |
| No JavaScript | Zero `.js` files in src/ | `find src -name "*.js"` = 0 |
| HTML validated | Probar `HTMLValidator` | Zero errors/warnings |
| Rich tracing | renacer spans on all ops | Trace files parseable |

### 5.3 Probar Test Specifications

```rust
#[probar::test]
async fn test_transcription_does_not_block_ui() {
    let page = browser.new_page().await;
    page.goto("http://localhost:8080/realtime-transcription/").await;

    // Start recording
    page.click("#start_recording").await;
    page.wait_for_selector(".status-recording").await;

    // Wait for chunk to be processed
    sleep(Duration::from_secs(5)).await;

    // UI should remain responsive - click should work
    let click_time = Instant::now();
    page.click("#stop_recording").await;
    let response_time = click_time.elapsed();

    // UI response must be < 100ms even during transcription
    assert!(response_time < Duration::from_millis(100),
        "UI blocked for {}ms during transcription", response_time.as_millis());
}

#[probar::pixel_test]
async fn test_transcript_display_visual_regression() {
    let page = browser.new_page().await;
    page.goto("http://localhost:8080/realtime-transcription/").await;

    // Load known audio file
    page.evaluate("window.testAudio = new Float32Array([...])")  await;

    // Trigger transcription
    page.evaluate("transcribeTestAudio()").await;
    page.wait_for_text("#transcript", "hello world").await;

    // Pixel comparison
    let screenshot = page.screenshot("#transcript").await;
    probar::assert_visual_match(screenshot, "transcript_hello_world.png", 0.99);
}
```

---

## 6. Tracing & Observability

### 6.1 Tracing Levels (renacer integration)

```rust
pub enum TracingLevel {
    /// Minimal overhead (<1% CPU)
    /// - Error events only
    /// - Chunk completion summaries
    Light,

    /// Moderate overhead (<5% CPU)
    /// - All Light events
    /// - State transitions
    /// - Worker message timing
    /// - RTF per chunk
    Medium,

    /// Full instrumentation (~10% CPU)
    /// - All Medium events
    /// - Audio sample counts
    /// - Memory allocations
    /// - SIMD operation timing
    /// - Attention matrix stats
    Full,
}
```

### 6.2 Span Hierarchy

```
realtime_transcription_demo
├── worker_initialization
│   ├── worker_spawn
│   └── wasm_instantiate
├── model_loading
│   ├── fetch_model
│   ├── transfer_to_worker
│   └── model_parse
├── audio_capture
│   ├── microphone_access
│   ├── audio_context_create
│   └── worklet_connect
└── transcription_loop
    ├── chunk_capture [repeated]
    │   ├── resampling
    │   └── vad_detection
    ├── chunk_transfer [repeated]
    └── chunk_process [repeated]  # In worker
        ├── mel_spectrogram
        ├── encoder_forward
        ├── decoder_generate
        └── token_decode
```

### 6.3 Tracing Toggle API

```rust
/// Set tracing level at runtime
#[wasm_bindgen]
pub fn set_tracing_level(level: &str) {
    let level = match level {
        "light" => TracingLevel::Light,
        "medium" => TracingLevel::Medium,
        "full" => TracingLevel::Full,
        _ => TracingLevel::Medium,
    };
    TRACING_LEVEL.with(|l| *l.borrow_mut() = level);
}

/// Query current tracing level
#[wasm_bindgen]
pub fn get_tracing_level() -> String {
    TRACING_LEVEL.with(|l| l.borrow().to_string())
}
```

### 6.4 Metrics Export

```rust
#[derive(Serialize)]
pub struct TranscriptionMetrics {
    pub total_chunks: u64,
    pub total_audio_seconds: f64,
    pub total_processing_seconds: f64,
    pub average_rtf: f64,
    pub p50_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub queue_depth_max: usize,
    pub chunks_dropped: u64,
    pub errors: u64,
}
```

---

## 7. Performance Targets

### 7.1 Latency Budget

| Operation | Target | Max | Measurement |
|-----------|--------|-----|-------------|
| Audio callback | 2ms | 5ms | performance.now() |
| Chunk transfer | 5ms | 20ms | postMessage timing |
| UI update | 10ms | 50ms | requestAnimationFrame |
| **Total main-thread** | **17ms** | **75ms** | End-to-end |

### 7.2 Transcription Performance

| Model | Chunk Size | Target RTF | Max RTF | Memory |
|-------|------------|------------|---------|--------|
| tiny-int8 | 1.5s | 0.5x | 0.8x | 150MB |
| tiny-fp32 | 1.5s | 0.8x | 1.2x | 200MB |
| base-int8 | 2.0s | 1.0x | 1.5x | 350MB |

### 7.3 Stability Targets

| Metric | Target | Measurement Period |
|--------|--------|-------------------|
| Memory growth | < 1MB/hour | 1 hour continuous |
| Chunk drop rate | < 1% | 1000 chunks |
| Error rate | < 0.1% | 1000 chunks |
| Uptime | 99.9% | 24 hours |

---

## 8. Golden Rules Compliance

### 8.1 Rule: 100% Testable

Every public function has a corresponding test:

```rust
// For every pub fn foo(...) -> T
#[test]
fn test_foo_success() { ... }
#[test]
fn test_foo_error_case_1() { ... }
// etc.
```

### 8.2 Rule: GUI and Pixel Testing Required

```rust
// tests/gui_coverage.rs
#[probar::test]
async fn verify_all_buttons_covered() {
    let tracker = UxCoverageTracker::new();
    let page = browser.new_page().await;
    page.goto(URL).await;

    // Must click every button
    for button in ["start_recording", "stop_recording", "clear_transcript"] {
        page.click(&format!("#{button}")).await;
        tracker.record_interaction(button);
    }

    assert_eq!(tracker.button_coverage(), 1.0);
}

// tests/pixel_regression.rs
#[probar::pixel_test]
async fn verify_all_states_match_golden() {
    for state in ["idle", "recording", "processing", "error"] {
        let screenshot = capture_state(state).await;
        probar::assert_visual_match(screenshot, format!("golden/{state}.png"), 0.99);
    }
}
```

### 8.3 Rule: No JavaScript

Strict adherence to the Zero JavaScript policy is enforced with the following distinctions:

| Artifact Type | Allowed? | Rationale |
|---------------|----------|-----------|
| `src/**/*.js` | ❌ NO | Core logic must be Rust. |
| Business Logic in Strings | ❌ NO | `eval()` or string-embedded JS is prohibited. |
| `pkg/*.js` | ✅ YES | Auto-generated by `wasm-bindgen` (infrastructure only). |
| Bootstrap Shim | ✅ YES | Minimal `format!()` string to import/init WASM module. |

```bash
# CI check
find demos/realtime-transcription/src -name "*.js" | wc -l
# Must equal 0

# Only generated JS in pkg/ is allowed
```

### 8.4 Rule: HTML Validated by Probar

```rust
#[test]
fn validate_html() {
    let html = include_str!("../www/realtime-transcription/index.html");
    let result = probar::HTMLValidator::new()
        .allow_custom_elements(false)
        .require_lang_attribute(true)
        .require_viewport_meta(true)
        .validate(html);

    assert!(result.errors.is_empty(), "HTML errors: {:?}", result.errors);
    assert!(result.warnings.is_empty(), "HTML warnings: {:?}", result.warnings);
}
```

### 8.5 Rule: Tracing Toggle

```rust
// Runtime configuration
#[wasm_bindgen]
pub fn configure_tracing(level: &str) -> Result<(), JsValue> {
    match level {
        "off" => tracing::subscriber::set_global_default(NoopSubscriber),
        "light" => enable_light_tracing(),
        "medium" => enable_medium_tracing(),
        "full" => enable_full_tracing(),
        _ => return Err("Invalid level".into()),
    }
    Ok(())
}
```

---

## 9. References

### Peer-Reviewed Citations

1. **Radford, A., et al.** (2022). "Robust Speech Recognition via Large-Scale Weak Supervision." *OpenAI Technical Report*. https://arxiv.org/abs/2212.04356

2. **WebAssembly Community Group** (2023). "WebAssembly Threads Specification." *W3C Working Draft*. https://webassembly.github.io/threads/

3. **Nickolls, J., & Buck, I.** (2008). "Scalable Parallel Programming with CUDA." *ACM Queue*, 6(2), 40-53. https://doi.org/10.1145/1365490.1365500

4. **Dean, J., & Barroso, L. A.** (2013). "The Tail at Scale." *Communications of the ACM*, 56(2), 74-80. https://doi.org/10.1145/2408776.2408794

5. **Lamport, L.** (1978). "Time, Clocks, and the Ordering of Events in a Distributed System." *Communications of the ACM*, 21(7), 558-565. https://doi.org/10.1145/359545.359563

6. **Haas, A., et al.** (2017). "Bringing the Web up to Speed with WebAssembly." *PLDI '17*, 185-200. https://doi.org/10.1145/3062341.3062363

7. **Jouppi, N. P., et al.** (2017). "In-Datacenter Performance Analysis of a Tensor Processing Unit." *ISCA '17*, 1-12. https://doi.org/10.1145/3079856.3080246

8. **Graves, A., et al.** (2013). "Speech Recognition with Deep Recurrent Neural Networks." *ICASSP 2013*, 6645-6649. https://doi.org/10.1109/ICASSP.2013.6638947

9. **Vaswani, A., et al.** (2017). "Attention Is All You Need." *NeurIPS 2017*. https://arxiv.org/abs/1706.03762

10. **Liker, J. K.** (2004). *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill. ISBN: 978-0071392310

### Technical Standards

- W3C Web Audio API Specification
- W3C Web Workers Specification
- WHATWG HTML Living Standard
- Rust API Guidelines (RFC 1105)

---

## 10. Appendices

### A. COOP/COEP Header Configuration

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Implemented in probar via `--cross-origin-isolated` flag (Issue #11).

### B. Worker Bootstrap Sequence

```
1. Main thread: Create Worker with blob URL
2. Worker: Import WASM module
3. Worker: Call init() to instantiate WASM
4. Worker: Call worker_entry() to set up handlers
5. Worker: postMessage({ type: "ready" })
6. Main thread: Receive ready, send model bytes
7. Worker: Load model, postMessage({ type: "model_loaded" })
8. Main thread: Enable recording UI
```

### C. Error Recovery Matrix

| Error | Recovery Action | User Feedback |
|-------|-----------------|---------------|
| Worker crash | Recreate worker | "Restarting..." |
| Model load fail | Retry with backoff | "Retrying..." |
| Transcription error | Log, continue | Show partial |
| Memory exhaustion | Drop queue, GC | "Catching up..." |
| Network loss | Queue locally | "Offline mode" |

### D. File Structure

```
demos/realtime-transcription/
├── Cargo.toml
├── src/
│   ├── lib.rs          # Main entry, state machine
│   ├── worker.rs       # Worker entry point
│   ├── bridge.rs       # Main-thread worker communication
│   ├── audio.rs        # Audio capture pipeline
│   ├── state.rs        # State machine definition
│   └── metrics.rs      # Performance tracking
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── browser/
│   └── pixel/
└── www/
    └── index.html      # Validated, no JS
```

---

## Review Checklist

Before implementation proceeds, confirm:

- [ ] Architecture approved by team
- [ ] Performance targets are realistic
- [ ] Testing strategy is complete
- [ ] Tracing levels defined
- [ ] Golden rules enforceable
- [ ] No ambiguity in specifications
- [ ] Dependencies (probar, renacer) ready

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.2.1 | 2024-12-14 | Claude Code | Addressed falsification review: Increased overlap, added context management, clarified JS policy and Shared Memory. |
| 0.1.0 | 2024-12-14 | Claude Code | Initial draft |

---

*"Stop and fix problems when they occur. Build quality in from the start."* — Toyota Production System Principle #5 (Jidoka)