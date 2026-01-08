# QA Falsification Checklist: WAPR-DEMO-REBUILD-TDD

**Objective:** rigorously attempt to falsify the claims made in the "TDD Demo Rebuild Summary" (WAPR-DEMO-REBUILD-TDD).
**Project Root:** `/home/noah/src/whisper.apr`
**Demo Directory:** `demos/`

## 1. Build Artifact Verification
*Claim: "WASM builds successfully (735KB)"*

- [ ] **Falsify Build Success:** Run `cd demos && make clean && make build`. Does it fail?
  - *Pass Condition:* Build completes with "✅ Demo built successfully".
- [ ] **Falsify Size Claim:** Run `ls -lh demos/www/pkg/whisper_apr_demo_bg.wasm`.
  - *Target:* Size should be ~718KiB (approx 735KB).
  - *Failure Condition:* Size > 750KB or < 700KB (indicates optimization failure or missing features).
- [ ] **Falsify Artifact Generation:** Verify these files exist after build:
  - `demos/www/pkg/whisper_apr_demo.js`
  - `demos/www/pkg/whisper_apr_demo_bg.wasm`
  - `demos/www/pkg/package.json`

## 2. Test Suite Integrity
*Claim: "235 tests pass (0 failures, 10 skipped browser tests awaiting server)"*

- [ ] **Falsify Test Count:** Run `cd demos && cargo test`.
  - *Pass Condition:* Output must show exactly `235 passed` and `10 ignored`.
- [ ] **Falsify Skipped Tests:** Start the server with `make serve-test` in one terminal, then run `cargo test --package whisper-apr-demo-tests -- --ignored` in another.
  - *Pass Condition:* The 10 previously ignored tests must now PASS.
  - *Critical Check:* `streaming_ux_tests::test_coop_coep_headers` must pass (verifies SharedArrayBuffer support).

## 3. "Critical Fix" Logic Verification
*Claim: "Stop Detection (whisper.cpp Pattern) ... processAudioTick checks isDone()"*

- [ ] **Code Inspection (Source of Truth):**
  - Verify `demos/www-demo/src/worker_js.rs` contains `if (ringBuffer && ringBuffer.isDone())`.
  - Verify `demos/www-demo/src/ring_buffer.rs` implements `mark_done()` using `Atomics`.
- [ ] **Runtime Falsification (Stop Button):**
  1. Open the demo (`make serve-parallel`).
  2. Click "Record". Speak for 5 seconds.
  3. Click "Stop".
  4. **Falsification Check:** Open DevTools Console. Look for `[Worker] Buffer done, getting final transcription`.
  5. *Failure Condition:* If this log is missing, the worker did not detect the stop signal via the ring buffer.

## 4. UI/UX & Accessibility Compliance
*Claim: "HTML demo with all required elements... ARIA labels"*

- [ ] **Element Presence Check:** Open `demos/www/index.html` (or viewed via server). Confirm existence of:
  - `#status` (Status Bar)
  - `#record` (Button)
  - `#transcript` (Output)
  - `#partial` (Output)
  - `#vu_meter` (Visualizer)
  - `#clear` (Button)
- [ ] **ARIA Falsification:** Inspect elements in DevTools.
  - `#status` must have `aria-live="polite"`.
  - `#vu_meter` must have `role="meter"`.
  - `#transcript` must have `role="log"`.
- [ ] **Accessibility Audit:** Run `cd demos && make a11y` (if available) or check console for accessibility warnings during runtime.

## 5. SharedArrayBuffer & Headers
*Claim: "Run with probar server (sets COOP/COEP headers)"*

- [ ] **Header Verification:**
  1. Run `make serve-parallel`.
  2. Open DevTools -> Network -> localhost check.
  3. Verify Response Headers:
     - `Cross-Origin-Opener-Policy: same-origin`
     - `Cross-Origin-Embedder-Policy: require-corp`
  4. *Failure Condition:* `SharedArrayBuffer` is undefined in the console (indicates headers are missing).

## 6. Zero-JS Policy Adherence
*Claim: "Worker JS... Key Files Created"*

- [ ] **Falsify "Zero-JS" (Main Thread):** Verify `demos/www/index.html` contains minimal logic.
  - Ideally, it should only import WASM and wire UI events.
  - *Warning:* If `index.html` contains complex processing logic (FFT, SAD, etc.), the "Zero-JS" architecture is compromised.
- [ ] **Worker Generation:** Verify `demos/www-demo/src/worker_js.rs` generates the worker code dynamically, ensuring Rust remains the single source of truth.

---

**Execution Instructions:**
Run this checklist from the project root. Report any failures immediately as a "Rejection of Release Candidate".
