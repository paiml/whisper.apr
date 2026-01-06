# WAPR-WORKER Five Whys Root Cause Analysis #2

## Incident Summary

**Date**: 2026-01-06
**Previous Incident**: Fixed ES module loading (importScripts → dynamic import)
**New Issue**: Worker WASM init fails with `No window`

**Console Output**:
```
[Worker] Bootstrap received, loading WASM from: http://localhost:8081
[Worker] Init failed: No window
[Manager] Error: Worker init failed: No window
```

**Impact**: Same pattern repeated - code passed probar tests but failed in browser

---

## Five Whys Analysis

### Why #1: Why did the Worker fail with "No window"?

**Answer**: When WASM is loaded in the Worker via `wasmModule.default()`, it executes
initialization code in `lib.rs` that calls `web_sys::window()`. Workers don't have
access to `window` - they have `self` (DedicatedWorkerGlobalScope).

**Evidence**:
```rust
// lib.rs - runs when WASM is initialized
pub fn start() {
    // ...
    let window = web_sys::window().ok_or("No window")?;  // FAILS IN WORKER
}
```

### Why #2: Why does Worker WASM initialization call window()?

**Answer**: The WASM module has a single entry point (`start()`) designed for main
thread usage. When loaded in a Worker context, the same initialization runs but
fails because Workers lack the Window API.

**Evidence**:
- `whisper_apr_demo.js` exports `start()` as main entry
- Worker imports same module: `await import('/pkg/whisper_apr_demo.js')`
- No separate Worker entry point exists

### Why #3: Why wasn't a separate Worker entry point created?

**Answer**: The architecture assumed WASM would only run in main thread context.
The `initWorker()` function was added but the module initialization still runs
`start()` which depends on Window APIs.

**Evidence**:
```javascript
// Worker bootstrap - module init runs start() before initWorker() is called
const wasmModule = await import(baseUrl + '/pkg/whisper_apr_demo.js');
wasm = await wasmModule.default();  // <-- This triggers start() with window access
worker = wasmModule.initWorker();
```

### Why #4: Why didn't probar tests catch this?

**Answer**: probar tests verify main thread behavior. They do NOT:
1. Actually spawn a `new Worker()` with the blob URL
2. Verify Worker WASM initialization succeeds
3. Test Worker → Main thread message passing

**Evidence**:
```bash
$ probar test -v
# 167 passed, 0 failed
# BUT: No test spawns actual Web Worker
```

### Why #5: Why don't probar tests spawn actual Workers?

**Answer**: The spec's "Jidoka Gates" required `probar test -v` to pass, but
did NOT require tests that specifically verify Worker initialization. The
gate was necessary but insufficient.

**Root Cause Identified**:
> Gate 2 (probar test) was treated as sufficient, but it only tests main thread.
> Gate 3 (Manual browser verification) was marked "in_progress" but not completed.
> The process allowed declaring "tests passed" without completing all gates.

---

## Why The First Fix Was Insufficient

| Aspect | First Fix | What Was Missing |
|--------|-----------|------------------|
| ES module loading | ✅ Fixed | - |
| Worker type: module | ✅ Fixed | - |
| AudioWorklet MIME | ✅ Fixed | - |
| probar tests | ✅ Passed | Tests don't spawn Workers |
| **Manual verification** | ❌ Not done | Would have caught window() error |
| **Worker-specific test** | ❌ Not added | No test verifies Worker init |

---

## Toyota Way Countermeasures (Updated)

### 1. Jidoka Enhancement: Gate 3 Cannot Be Skipped

**Problem**: Gate 3 (Manual browser verification) was "in_progress" but declared complete.

**Countermeasure**: Gate 3 is BLOCKING. No commit until console screenshot provided.

```yaml
completion_gates:
  - gate: 1
    name: "Compilation"
    command: "cargo check && make build"
    blocking: true

  - gate: 2
    name: "Automated Tests"
    command: "probar test -v"
    blocking: true

  - gate: 3
    name: "Manual Browser Verification"
    type: manual
    blocking: true  # CANNOT BE SKIPPED
    evidence_required:
      - "Browser console screenshot showing no errors"
      - "Worker initialized message: '[Worker] WASM initialized successfully'"
      - "Recording works: VU meter responds to audio"
```

### 2. Poka-Yoke: Separate Worker Entry Point

**Problem**: Single WASM entry point assumes Window context.

**Countermeasure**: Conditional initialization based on execution context.

```rust
// lib.rs
#[wasm_bindgen(start)]
pub fn start() {
    // Check if we're in a Worker context (no window)
    if web_sys::window().is_none() {
        // Worker context - minimal init only
        console_error_panic_hook::set_once();
        return;
    }

    // Main thread context - full UI init
    // ...existing code...
}
```

### 3. Andon: Add Worker-Specific probar Test

**Problem**: probar tests don't verify Worker initialization.

**Countermeasure**: Add test that spawns Worker and verifies init message.

```rust
#[wasm_bindgen_test]
async fn test_worker_initialization() {
    // Spawn actual worker
    let worker = Worker::new_with_options(&worker_url, &options)?;

    // Wait for ready message
    let ready = wait_for_message(&worker, "ready", 5000).await;
    assert!(ready, "Worker must send 'ready' message within 5s");
}
```

### 4. Genchi Genbutsu: Evidence-Based Completion

**Problem**: "Manual browser verification" marked in_progress but not done.

**Countermeasure**: Commit message MUST include evidence.

```markdown
## Commit Template for Worker Changes

fix(worker): [description]

## Evidence (REQUIRED - commit will be rejected without these)

### Gate 1: Compilation
```
$ cargo check && make build
[output showing success]
```

### Gate 2: probar tests
```
$ probar test -v
167 passed, 0 failed
```

### Gate 3: Manual Browser Verification
- [ ] Console screenshot attached (no red errors)
- [ ] Worker init message visible: "[Worker] WASM initialized successfully"
- [ ] Recording started message: "[Manager] Recording started"
- [ ] VU meter responds to audio input
```

---

## Immediate Fixes Required

### Fix 1: Conditional WASM Initialization

```rust
// lib.rs - check context before using Window APIs
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();

    // Worker context has no window - skip UI init
    if web_sys::window().is_none() {
        web_sys::console::log_1(&"[WASM] Worker context detected, skipping UI init".into());
        return;
    }

    // Main thread - proceed with UI initialization
    // ...
}
```

### Fix 2: Add Worker Initialization Test

```rust
// tests/src/worker_tests.rs
#[wasm_bindgen_test]
async fn test_worker_spawns_and_initializes() {
    // This test MUST spawn an actual Worker and verify it initializes
}
```

### Fix 3: Complete Gate 3 Before Committing

**NO COMMIT** until:
1. Browser opened to http://localhost:8081
2. Record button clicked
3. Console shows "[Worker] WASM initialized successfully" (NOT "No window")
4. Screenshot taken as evidence

---

## Updated Verification Checklist

After implementing fixes, ALL must be verified IN ORDER:

### Gate 1: Compilation
- [ ] `cargo check` passes
- [ ] `cargo build --release` passes
- [ ] `make build` completes

### Gate 2: Automated Tests
- [ ] `probar test -v` passes
- [ ] New Worker init test exists and passes

### Gate 3: Manual Browser Verification (BLOCKING - DO NOT SKIP)
- [ ] Start server: `probar serve`
- [ ] Open http://localhost:8081 in browser
- [ ] Open DevTools console (F12)
- [ ] Click Record button
- [ ] **VERIFY**: Console shows `[Worker] WASM initialized successfully`
- [ ] **VERIFY**: NO red errors in console
- [ ] **VERIFY**: `[Manager] Recording started` appears
- [ ] **VERIFY**: VU meter responds to audio
- [ ] **TAKE SCREENSHOT** of console as evidence

### Gate 4: Commit with Evidence
- [ ] Commit message includes Gate 1-3 evidence
- [ ] Screenshot attached or described in commit

---

## Meta-Analysis: Why Did The Same Pattern Repeat?

| First Failure | Second Failure | Common Pattern |
|---------------|----------------|----------------|
| importScripts vs ES module | window() in Worker | Embedded JS not runtime-tested |
| Fixed via code change | Needs code change | probar tests insufficient |
| probar passed | probar passed | Tests don't spawn Workers |
| **Gate 3 not done** | **Gate 3 not done** | Manual verification skipped |

**The Root Pattern**:
Gate 3 (Manual browser verification) was treated as optional when it should be BLOCKING.

**Solution**:
No code is considered "fixed" until Gate 3 evidence is provided. Period.

---

## References

- Ohno, T. (1988). Toyota Production System
- Liker, J. (2004). The Toyota Way
- "The system is not at fault. The process allowed circumvention of the gates."
