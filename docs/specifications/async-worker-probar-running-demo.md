# WAPR-SPEC-010: Async Worker-Based Real-Time Transcription

**Status:** ACTIVE - World-Class Streaming Implementation
**Version:** 0.7.0 (probar 0.4.1 features integrated)
**Authors:** Claude Code, Noah
**Created:** 2024-12-14
**Updated:** 2026-01-05
**Toyota Way Principle:** Genchi Genbutsu (Go and See) + Jidoka (Automation with Human Touch)

---

## Executive Summary

This specification defines a production-grade, real-time speech transcription system using Web Workers for non-blocking inference. The design eliminates UI freezing, enables continuous audio capture, and provides rich observability through renacer tracing integration.

**Problem Statement:** Current implementation blocks the main thread during transcription, causing browser timeout ("Script terminated by timeout") after ~10 seconds of inference on 3-second audio chunks. Even with `spawn_local`, the synchronous `transcribe` call halts the event loop.

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

### 2.1 Status Update (2026-01-05)

The project is resuming active development after a stability phase. The "File Upload" feature demonstrated that the WASM model works correctly (accuracy issue resolved), but confirmed that main-thread transcription is non-viable for real-time use due to UI blocking.

**Current Capabilities:**
- ✅ File Upload & Transcription (working but blocks UI)
- ✅ Microphone Capture (implemented)
- ✅ Audio Resampling (linear interpolation)
- ❌ Worker Offloading (not implemented)

### 2.2 What Was Implemented

| Component | Status | Notes |
|-----------|--------|-------|
| `whisper-apr/parallel` feature | ✅ Complete | Cargo.toml updated |
| `src/wasm/threading.rs` | ✅ Complete | Thread pool, COOP/COEP detection |
| `probar --cross-origin-isolated` | ✅ Complete | COOP/COEP headers, Issue #11 |
| `worker-transcription` demo crate | ✅ Skeleton | Needs full implementation |
| Worker message protocol | ⚠️ Partial | `worker.rs` created, not wired |
| Main thread integration | ❌ Not started | Still uses blocking transcribe |

### 2.3 Root Cause of Timeout

```
Timeline of failure:
─────────────────────────────────────────────────────────────────────────
0ms     Audio callback fires, 3s chunk ready
        ↓
10ms    spawn_local schedules transcription (setTimeout(0))
        ↓
20ms    Event loop yields, callback scheduled
        ↓
30ms    transcribe() starts executing (SYNCHRONOUS WASM)
        ↓
        ════════════════════════════════════════════════════
        │  MAIN THREAD BLOCKED - NO UI, NO AUDIO CALLBACKS │
        ════════════════════════════════════════════════════
        ↓
~10000ms Browser terminates script (slow script timeout)
```

**Crucial Insight:** `wasm_bindgen_futures::spawn_local` does not create a new thread. It merely schedules the future on the Javascript microtask queue. Since `model.transcribe()` is a CPU-bound synchronous function, once it starts executing, it monopolizes the single main thread until completion, freezing the browser.

### 2.4 File Upload Feature (Completed 2025-01-05)

As a workaround for real-time streaming issues, a file upload feature was implemented:

| Component | Status | Notes |
|-----------|--------|-------|
| Upload button UI | ✅ Complete | `📁 Upload Audio` button next to Record |
| File input element | ✅ Complete | Hidden input accepting audio/*,video/* |
| Audio decoding | ✅ Complete | Uses `AudioContext.decodeAudioData()` |
| Resampling to 16kHz | ✅ Complete | Linear interpolation resampler |
| Transcription | ✅ Complete | Synchronous (acceptable for file upload) |
| Probar tests | ✅ Complete | 5 tests in `browser_tests::zero_js_demo` |

**Implementation Location:** `demos/www-demo/src/lib.rs`
- `handle_file_upload()` - Event handler for file input change
- `process_audio_file()` - Async decode/resample/transcribe pipeline
- `resample_audio()` - Linear interpolation resampler

**Test Coverage:**
- `test_upload_button_exists` - UI element verification
- `test_file_upload_transcription` - Full E2E pipeline test

### 2.5 Model Accuracy Analysis (WAPR-ACCURACY)

#### 2.5.1 Pareto Frontier Discovery (2025-01-05)

Systematic testing of all model variants revealed critical findings:

| Model | Size | Native CLI | WASM | Status |
|-------|------|------------|------|--------|
| tiny-int4-sparse | 9MB | whitespace | whitespace | ❌ INT4 too aggressive |
| tiny-int4-fb | 24MB | whitespace | whitespace | ❌ INT4 broken |
| tiny-int8 (no fb) | 37MB | whitespace | whitespace | ❌ Missing filterbank |
| **tiny-int8-fb** | **37MB** | **✅ Correct** | **✅ Correct** | **Pareto optimal** |
| tiny-fp32 (no fb) | 145MB | " I" (wrong) | N/A | ❌ Missing filterbank |
| tiny-fp32-fb | 146MB | ✅ Correct | ✅ Correct | ✅ Works |

**Key Findings:**
1. **Filterbank embedding is REQUIRED** - Models without embedded filterbank (`-fb`) produce incorrect output
2. **INT8 is the minimum viable quantization** - INT4 quantization breaks accuracy regardless of filterbank
3. **Pareto optimal: whisper-tiny-int8-fb at 37MB** - Smallest working model

#### 2.5.2 Root Cause: Filterbank

Models without embedded filterbank (`-fb` suffix) fail because:
- Runtime mel filterbank computation differs from OpenAI's reference implementation
- The `-fb` models embed OpenAI's mel_filters.npz coefficients directly

#### 2.5.3 WASM Bug Resolution (WAPR-WASM-NUL) ✅ FIXED

**Issue:** WASM was producing NUL bytes (0x00) instead of text.

**Root Cause:** The www-demo was using a non-`-fb` model which lacks embedded vocabulary and filterbank.

**Fix:** Updated `MODEL_URL` in `demos/www-demo/src/lib.rs` to use `whisper-tiny-int8-fb.apr`:
```rust
const MODEL_URL: &str = "/models/whisper-tiny-int8-fb.apr";
```

**Verification:** Browser test passes with correct output:
- Expected: `" The birds can use"`
- WASM output: `" The birds can use"` ✅
- Hex: `20 54 68 65 20 62 69 72 64 73 20 63 61 6e 20 75 73 65`

#### 2.5.4 Execution Flow Analysis (2025-01-05)

##### PAGE LOAD SEQUENCE

| Step | Function | Type | What Happens |
|------|----------|------|--------------|
| 1 | `start()` | **SYNC** | WASM module entry point called by browser |
| 2 | `console_error_panic_hook::set_once()` | **SYNC** | Set up panic handler |
| 3 | `tracing_wasm::set_as_global_default()` | **SYNC** | Initialize tracing |
| 4 | DOM creation (lib.rs:69-197) | **SYNC** | Create all HTML elements, buttons disabled |
| 5 | `onclick` closure setup | **SYNC** | Attach click handlers to buttons |
| 6 | `spawn_model_load(document)` | **ASYNC SPAWN** | Spawns async task, returns immediately |
| 7 | `start()` returns `Ok(())` | **SYNC** | Page is now interactive |
| 8 | `fetch_model(MODEL_URL).await` | **ASYNC** | Download 37MB model |
| 9 | `WhisperAprWasm::from_apr_bytes(&bytes)` | **SYNC** | Parse model weights |
| 10 | `app.model = Some(Rc::new(model))` | **SYNC** | Store model, enable buttons |

##### RECORD CLICK SEQUENCE

| Step | Function | Type | What Happens |
|------|----------|------|--------------|
| 1 | `handle_record_click()` | **SYNC** | Click handler fires |
| 2 | `APP.with()` check model | **SYNC** | Check if model exists |
| 3 | `StreamingConfigWasm::new()` | **SYNC** | Create config |
| 4 | `StreamingSessionWasm::new(model, config)` | **SYNC** | **SUSPECT #1** - Creates session |
| 5 | `update_ui(document)` | **SYNC** | Update DOM |
| 6 | `spawn_local(start_recording)` | **ASYNC SPAWN** | Spawns mic request |
| 7 | `getUserMedia()` | **ASYNC** | Request mic permission |
| 8 | `AudioContext::new()` | **SYNC** | Create audio context |
| 9 | `create_script_processor()` | **SYNC** | Create audio processor |
| 10 | Audio callback registered | **SYNC** | Set `onaudioprocess` handler |

##### AUDIO CALLBACK (fires ~10x/sec with 4096 samples at 48kHz)

| Step | Function | Type | What Happens |
|------|----------|------|--------------|
| 1 | `onaudioprocess` event | **SYNC CALLBACK** | Browser calls with 4096 samples |
| 2 | `APP.with().try_borrow_mut()` | **SYNC** | Try to get app state |
| 3 | `session.push_audio(&samples)` | **SYNC** | **SUSPECT #2** - Process audio |
| 4 | `processor.process()` | **SYNC** | Check for chunks |
| 5 | `has_partial()` check | **SYNC** | Check if transcription needed |
| 6 | `transcribe_partial()` | **SYNC BLOCKING** | **SUSPECT #3** - RUNS INFERENCE |

##### FIVE WHYS ANALYSIS

1. **Why does recording hang instantly?** → Because a SYNC operation blocks the main thread
2. **Why does the main thread block?** → Because one of the SYNC operations takes too long
3. **Which SYNC operations are suspects?**
   - **SUSPECT #1:** `StreamingSessionWasm::new()` - Record click step 4
   - **SUSPECT #2:** `session.push_audio()` - Audio callback step 3
   - **SUSPECT #3:** `transcribe_partial()` - Audio callback step 6
4. **Why don't we know which one?** → No timing instrumentation exists
5. **What do we need?** → Timing measurements at each suspect point

##### INVESTIGATION STATUS

- [x] Add `performance.now()` timing to `StreamingSessionWasm::new()`
- [x] Add timing to `push_audio()` in audio callback
- [x] Add timing to `transcribe_partial()` calls
- [x] Identify actual bottleneck from console output
- [x] Root cause identified: `has_partial()` never fires (partial_threshold=3s = chunk_size)
- [x] Fix applied: `push_audio()` now checks `has_chunk()` before `has_partial()`
- [ ] Implement world-class endpoint-driven streaming (Section 3.2)

##### TIMING DATA (2026-01-05)

```
Test: Chrome, click Record, wait 4+ seconds, click Stop
```

| Metric | Measured Value | Threshold | Status |
|--------|----------------|-----------|--------|
| Session creation | 0.10ms | <100ms | ✅ OK |
| push_audio() | 5-8ms | <50ms | ✅ OK |
| Blocking calls (>50ms) | 0 | 0 | ✅ OK |
| Callbacks in 4.3s | 46 | ~46 expected | ✅ OK |
| has_update=true | **0** | >0 after 3s | ❌ **FAIL** |

**Key Finding:** Recording does NOT hang. 46 callbacks executed smoothly over 4.3 seconds. But `has_update` is ALWAYS false - meaning `transcribe_partial()` is NEVER called.

**Root Cause Identified:** The StreamingProcessor's `has_partial()` never returns true because:
1. `partial_threshold_samples` = 3.0s (same as chunk size!)
2. When chunk_progress reaches 100%, state becomes `ChunkReady`
3. `has_partial()` only checks for `AccumulatingSpeech` state
4. **Result:** Full chunks are never transcribed, partials are impossible

**Fix Applied (2026-01-05):**
```rust
// In StreamingSessionWasm::push_audio()
// Now checks has_chunk() FIRST, before has_partial()
if self.processor.has_chunk() {
    if let Some(chunk_audio) = self.processor.get_chunk() {
        // Full chunk transcription (is_final = true)
        if let Ok(result) = self.whisper.transcribe(&chunk_audio, ...) {
            return Some(PartialTranscriptionResultWasm { is_final: true, ... });
        }
    }
}
```

**Remaining Issue:** The `transcribe()` call is SYNC on main thread - will block UI. This confirms the need for Worker-based async transcription (Section 3).

**Next Step:** Implement world-class endpoint-driven streaming (Section 3.2) with Web Worker offloading.

---

## 3. Target State Design

### 3.1 Component Specification

#### 3.1.1 TranscriptionWorker (Rust)

```rust
// demos/www-demo/src/worker.rs

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
// demos/www-demo/src/bridge.rs

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

### 3.2 World-Class Partial Transcription Strategy

**Reference implementations:** Apple Dictation, Google Voice Typing, Otter.ai

#### 3.2.1 Industry Benchmark Analysis

| System | First Text | Update Freq | Trigger | Model Type |
|--------|-----------|-------------|---------|------------|
| Apple Dictation | 200-500ms | Per-word | Endpoint + streaming | RNN-T/CTC |
| Google Voice | 300ms | Per-word | Continuous | Streaming Conformer |
| Otter.ai | 500ms | Per-phrase | Endpoint | Hybrid |
| **whisper.apr (current)** | **3000ms** | **Per-chunk** | **Fixed threshold** | **Attention** |
| **whisper.apr (target)** | **300ms** | **Per-phrase** | **Endpoint-driven** | **Attention** |

#### 3.2.2 The Whisper Challenge

Whisper is an **attention-based encoder-decoder** model designed for full utterances, not true streaming. Each `transcribe()` call processes the entire audio buffer. This creates a fundamental tradeoff:

| Partial Interval | Latency | CPU Load | Transcriptions/Chunk | UX Quality |
|-----------------|---------|----------|----------------------|------------|
| 0.3s | 300ms | 10x baseline | 10 | Responsive, high CPU |
| 0.5s | 500ms | 6x baseline | 6 | Good balance |
| 1.0s | 1000ms | 3x baseline | 3 | Noticeable delay |
| 1.5s | 1500ms | 2x baseline | 2 | Sluggish |
| 3.0s (current) | 3000ms | 1x baseline | 1 | **Unacceptable** |

#### 3.2.3 Endpoint-Driven Architecture (World-Class)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     WORLD-CLASS STREAMING PIPELINE                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Audio ──► VAD ──► Endpoint ──► Transcribe ──► Diff ──► Speculative UI    │
│            │       Detector      (Worker)       Engine      Display        │
│            │           │                           │            │          │
│            │           │                           │            ▼          │
│            │           │                           │    ┌──────────────┐   │
│            │           │                           └───►│ Word Locking │   │
│            │           │                                │  (conf>0.85) │   │
│            │           │                                └──────────────┘   │
│            │           │                                                    │
│            │           └──► Silence ≥300ms ──► TRANSCRIBE NOW              │
│            │                                                                │
│            └──► Energy Level ──► Visual Feedback (waveform/mic indicator)  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

**Key insight:** Transcribe on **silence/pause detection** (natural phrase boundaries), not fixed intervals.

#### 3.2.4 Optimal Configuration

```rust
/// World-class streaming configuration
pub struct StreamingConfig {
    // === ENDPOINT DETECTION ===
    /// Minimum audio before ANY transcription (prevents micro-utterances)
    pub min_utterance_ms: u32,           // 250ms (Apple-like)

    /// Silence duration to trigger transcription (phrase boundary)
    pub endpoint_silence_ms: u32,        // 300ms (natural pause)

    /// Maximum pending audio before forced partial (even if speaking)
    pub max_pending_ms: u32,             // 1500ms (prevent infinite wait)

    // === VAD PARAMETERS ===
    /// Energy threshold for speech detection (dB)
    pub vad_energy_threshold_db: f32,    // -35 dB

    /// Zero-crossing rate threshold
    pub vad_zcr_threshold: f32,          // 0.1

    // === SPECULATIVE DISPLAY ===
    /// Number of trailing words that can be revised
    pub speculative_window_words: usize, // 3 words

    /// Confidence threshold to lock words (prevent jarring rewrites)
    pub confidence_lock_threshold: f32,  // 0.85

    // === CHUNK MANAGEMENT ===
    /// Full chunk size for final transcription
    pub chunk_duration_ms: u32,          // 3000ms

    /// Overlap for context continuity
    pub overlap_ms: u32,                 // 200ms
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            min_utterance_ms: 250,
            endpoint_silence_ms: 300,
            max_pending_ms: 1500,
            vad_energy_threshold_db: -35.0,
            vad_zcr_threshold: 0.1,
            speculative_window_words: 3,
            confidence_lock_threshold: 0.85,
            chunk_duration_ms: 3000,
            overlap_ms: 200,
        }
    }
}
```

#### 3.2.5 Speculative Display UX Pattern

```
User speaks: "The quick brown fox jumps"

t=0.0s  │                              │ (listening - mic active)
t=0.3s  │ The                          │ ← speculative (gray italic)
t=0.5s  │ The quick                    │ ← "The" locks (black)
t=0.8s  │ The quick brown              │ ← "quick" locks
t=1.1s  │ The quick brown fox          │ ← "brown" locks
t=1.4s  │ The quick brown fox jumps    │
t=1.7s  │ The quick brown fox jumps    │ ← silence detected, all locked
        │ ▌                            │ ← cursor ready for next phrase
        └──────────────────────────────┘

Visual states:
  • Gray italic  = speculative (may change)
  • Black        = locked (confidence > 0.85)
  • Underline    = currently being revised
```

#### 3.2.6 State Machine for Endpoint Detection

```rust
pub enum StreamingState {
    /// No speech detected, waiting for audio above threshold
    Silence,

    /// Speech detected, accumulating audio
    Speech {
        start_ms: u64,
        samples: Vec<f32>,
    },

    /// Speech ended, short silence detected (potential endpoint)
    PotentialEndpoint {
        speech_samples: Vec<f32>,
        silence_start_ms: u64,
    },

    /// Endpoint confirmed, transcription in progress
    Transcribing {
        chunk_id: u32,
    },
}

impl StreamingState {
    pub fn process_frame(&mut self, frame: &AudioFrame, config: &StreamingConfig) -> Option<TranscribeCommand> {
        match self {
            Self::Silence => {
                if frame.is_speech(config.vad_energy_threshold_db) {
                    *self = Self::Speech {
                        start_ms: frame.timestamp_ms,
                        samples: frame.samples.clone(),
                    };
                }
                None
            }
            Self::Speech { start_ms, samples } => {
                samples.extend(&frame.samples);

                if !frame.is_speech(config.vad_energy_threshold_db) {
                    // Potential endpoint - start silence timer
                    *self = Self::PotentialEndpoint {
                        speech_samples: std::mem::take(samples),
                        silence_start_ms: frame.timestamp_ms,
                    };
                } else if frame.timestamp_ms - *start_ms >= config.max_pending_ms as u64 {
                    // Force transcription after max_pending_ms
                    let audio = std::mem::take(samples);
                    *self = Self::Transcribing { chunk_id: next_chunk_id() };
                    return Some(TranscribeCommand { audio, is_final: false });
                }
                None
            }
            Self::PotentialEndpoint { speech_samples, silence_start_ms } => {
                if frame.is_speech(config.vad_energy_threshold_db) {
                    // False endpoint - resume speech
                    speech_samples.extend(&frame.samples);
                    *self = Self::Speech {
                        start_ms: *silence_start_ms, // preserve original start
                        samples: std::mem::take(speech_samples),
                    };
                    None
                } else if frame.timestamp_ms - *silence_start_ms >= config.endpoint_silence_ms as u64 {
                    // Confirmed endpoint - transcribe now!
                    let audio = std::mem::take(speech_samples);
                    if audio.len() >= (config.min_utterance_ms as usize * 16) {
                        *self = Self::Transcribing { chunk_id: next_chunk_id() };
                        return Some(TranscribeCommand { audio, is_final: true });
                    } else {
                        // Too short, discard
                        *self = Self::Silence;
                    }
                    None
                } else {
                    None
                }
            }
            Self::Transcribing { .. } => None, // Wait for result
        }
    }
}
```

### 3.2.7 World-Class UX Implementation (v0.5.0)

**Status:** ✅ IMPLEMENTED (2026-01-05)

The following world-class UX elements have been implemented in `demos/www-demo/src/lib.rs`:

#### Audio Level VU Meter

```rust
// Calculate RMS audio level for VU meter (world-class UX)
let rms: f32 = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
// Convert to 0-1 range with some headroom (typical speech is -20dB to -6dB)
let audio_level = (rms * 5.0).min(1.0);
```

Visual element: `#vu_meter` - Green-to-orange gradient bar that responds to speech volume in real-time.

#### State Indicator

Displays current streaming state in human-readable form:

| ProcessorState | Display Label | Color |
|---------------|---------------|-------|
| WaitingForSpeech | "Listening..." | #8b949e (gray) |
| AccumulatingSpeech | "Recording..." | #f85149 (red) |
| ChunkReady | "Transcribing..." | #58a6ff (blue) |
| PartialResultReady | "Partial..." | #58a6ff |
| Processing | "Processing..." | #58a6ff |

Visual element: `#state_label` - Updates in real-time as streaming state changes.

#### Chunk Progress Bar

Displays progress toward next chunk transcription:

```rust
let progress = session.chunk_progress();  // 0.0 - 1.0
```

Visual element: `#chunk_progress` - Blue progress bar showing accumulation toward 3s chunk.

#### Probar Compliance

Recording indicator has `.recording-indicator` class for probar playbook validation:

```yaml
# demos/playbooks/realtime-transcription.yaml
recording:
  invariants:
    - description: "Recording indicator visible"
      condition: "has_element('.recording-indicator')"
    - description: "VU meter visible during recording"
      condition: "has_element('#vu_meter')"
    - description: "State label visible during recording"
      condition: "has_element('#state_label')"
    - description: "Chunk progress bar visible"
      condition: "has_element('#chunk_progress')"
```

### 3.3 Streaming Best Practices (StreamYard/OBS Pattern)

Following industry-standard streaming architectures [1][2]:

1. **Capture-Process Separation**: Audio capture NEVER waits for processing
2. **Bounded Queues**: Drop oldest chunks if queue exceeds N (backpressure)
3. **Chunk Overlap**: **200ms overlap** for context continuity
4. **Adaptive Chunk Size**: Reduce chunk size if RTF > 0.8
5. **Graceful Degradation**: Show "Processing..." instead of freezing

```rust
pub struct AdaptiveChunker {
    base_duration: f32,      // 3.0s default (full chunk)
    min_duration: f32,       // 0.25s minimum (min_utterance)
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

### 5.4 Probar Compliance Checks (v0.4.1+)

Run WASM compliance validation before deployment:

```bash
probador comply --detailed .
```

| Check | Description | Status |
|-------|-------------|--------|
| C001 | Code execution verified | ✅ |
| C002 | Console errors fail tests | ✅ |
| C003 | Custom elements tested | ✅ |
| C004 | Threading modes tested | ✅ |
| C005 | Low memory tested | ✅ |
| C006 | COOP/COEP headers | ✅ via `probar serve --cross-origin-isolated` |
| C007 | Replay hash matches | ✅ |
| C008 | Cache handling | ✅ |
| C009 | WASM size limit | ✅ (< 5MB) |
| C010 | No panic paths | ✅ |

### 5.5 Probar Playbook State Machine Testing

Validate streaming state machine via playbook:

```bash
# Validate playbook
probador playbook playbooks/realtime-transcription.yaml --validate

# Export state diagram
probador playbook playbooks/realtime-transcription.yaml --export svg --export-output docs/state-machine.svg

# Run mutation testing (M1-M5)
probador playbook playbooks/realtime-transcription.yaml --mutate
```

### 5.6 Advanced Streaming UX Testing (probar 0.4.1+) ✅ IMPLEMENTED

The following probar features enable comprehensive streaming UX testing:

```rust
use jugar_probar::emulation::AudioEmulator;
use jugar_probar::capabilities::{WasmThreadCapabilities, WorkerEmulator};
use jugar_probar::validators::StreamingUxValidator;

#[probar::test]
async fn test_streaming_transcription_ux(page: &Page) -> ProbarResult<()> {
    // AudioEmulator - Inject controlled audio for VAD testing
    let audio = AudioEmulator::new(16000, 1);
    audio.inject_cdp(&page, AudioSource::SpeechPattern {
        pattern: vec![
            (0.0, 500),   // 500ms silence
            (0.8, 2000),  // 2s speech
            (0.0, 500),   // 500ms silence (endpoint trigger)
        ],
    }).await?;

    // WasmThreadCapabilities - Verify SharedArrayBuffer support
    let caps = WasmThreadCapabilities::detect_cdp(&page).await?;
    caps.assert_streaming_ready()?;

    // WorkerEmulator - Test Web Worker lifecycle
    let worker = WorkerEmulator::new();
    worker.attach_cdp(&page).await?;
    let workers = worker.get_workers_cdp(&page).await?;
    assert!(!workers.is_empty(), "Worker should be running");

    // StreamingUxValidator - Assert real-time UX elements
    let validator = StreamingUxValidator::new();
    validator.track_state_cdp(&page, "#state_label").await?;
    validator.assert_state_sequence(&["Listening", "Recording", "Transcribing"])?;
    validator.assert_vu_meter_active(&page, "#vu_meter", 0.1, 3000).await?;
    validator.assert_progress_advancing(&page, "#chunk_progress", 5000).await?;

    Ok(())
}
```

### 5.7 Probar Dual Compliance System (v0.4.1+)

```bash
# Check compliance
probador comply check .

# Generate multi-format reports
probador comply report --format markdown --output compliance.md
probador comply report --format html --output compliance.html

# Install pre-commit hooks
probador comply enforce --pre-commit

# Show migration path between versions
probador comply migrate --from 0.3.0 --to 0.4.1

# Diff between versions
probador comply diff --from 0.3.0 --to 0.4.1
```

### 5.8 WasmStrictMode Presets

```rust
use jugar_probar::strict::WasmStrictMode;

// Production: All checks enabled, fail on any issue
let strict = WasmStrictMode::production();

// Development: Relaxed, allow some warnings
let dev = WasmStrictMode::development();

// Minimal: Essential checks only
let minimal = WasmStrictMode::minimal();
```

**Implemented Features (probar 0.4.1):**
- ✅ AudioEmulator with CDP injection
- ✅ WasmThreadCapabilities with COOP/COEP detection
- ✅ WorkerEmulator with Lamport clock ordering
- ✅ StreamingUxValidator with VU meter tracking
- ✅ Dual comply system (check/migrate/diff/enforce/report)
- ✅ WasmStrictMode presets
- ✅ E2ETestChecklist with mandatory checks
- ✅ 103 falsification tests (Popperian methodology)

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
| DSL-Generated Worker/Worklet | ✅ YES | Generated via `probar-js-gen` DSL with validation. |

**DSL-Generated JavaScript (WAPR-JS-001):**

Worker and AudioWorklet JavaScript is generated from Rust using `probar-js-gen` DSL:

| File | Purpose | Validation |
|------|---------|------------|
| `audioworklet_js.rs` | AudioWorklet processor | 20 tests, validator |
| `worker_js.rs` | Transcription worker bootstrap | 11 tests, validator |

See: [`javascript-generation.md`](./javascript-generation.md) for full specification.

```bash
# CI check
find demos/www-demo/src -name "*.js" | wc -l
# Must equal 0

# DSL-generated JS verified by tests
cargo test -p whisper-apr-demo --lib -- validator
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
demos/www-demo/
├── Cargo.toml
├── src/
│   ├── lib.rs          # Main entry, state machine, DOM creation
│   ├── worker.rs       # [PENDING] Worker entry point
│   ├── bridge.rs       # [PENDING] Main-thread worker communication
│   ├── audio.rs        # [PENDING] Audio capture pipeline (currently in lib.rs)
│   ├── state.rs        # [PENDING] State machine definition (currently in lib.rs)
│   └── metrics.rs      # [PENDING] Performance tracking
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
- [x] File upload workaround implemented (2025-01-05)
- [x] Model accuracy issue resolved - using whisper-tiny-int8-fb.apr (2026-01-05)
- [x] JavaScript generation via probar-js-gen DSL (WAPR-JS-001) (2026-01-06)

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.7.0 | 2026-01-06 | Claude Code | Added WAPR-JS-001 reference (Section 8.3): DSL-generated Worker/Worklet JavaScript with probar-js-gen, 31 validation tests. |
| 0.5.0 | 2026-01-05 | Claude Code | Added world-class streaming strategy (Section 3.2): endpoint-driven VAD, speculative display UX, streaming state machine, industry benchmarks (Apple/Google/Otter.ai). |
| 0.4.0 | 2026-01-05 | Claude Code | Updated Status to ACTIVE, updated paths to `demos/www-demo/`, refined analysis of synchronous blocking. |
| 0.3.0 | 2025-01-05 | Claude Code | Added file upload feature (Section 2.4), documented model accuracy issue (Section 2.5), updated status to IN PROGRESS. |
| 0.2.1 | 2024-12-14 | Claude Code | Addressed falsification review: Increased overlap, added context management, clarified JS policy and Shared Memory. |
| 0.1.0 | 2024-12-14 | Claude Code | Initial draft |

---

*"Stop and fix problems when they occur. Build quality in from the start."* — Toyota Production System Principle #5 (Jidoka)