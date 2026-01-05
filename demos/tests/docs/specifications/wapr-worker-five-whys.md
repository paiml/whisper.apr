# WAPR-WORKER Five Whys Root Cause Analysis

## Incident Summary

**Date**: 2026-01-06
**Issue**: Worker architecture code compiled but failed at runtime with:
1. `SyntaxError: Cannot use import statement outside a module`
2. `AbortError: Unable to load a worklet's module`

**Impact**: Feature marked "complete" but completely non-functional

---

## Five Whys Analysis

### Why #1: Why did the Worker fail at runtime?

**Answer**: The worker bootstrap code used `importScripts()` to load wasm-bindgen
generated JavaScript, but wasm-bindgen outputs ES modules which cannot be loaded
via `importScripts()`.

**Evidence**:
```javascript
// WRONG: importScripts only works with classic scripts
importScripts(baseUrl + '/pkg/whisper_apr_demo.js');

// wasm-bindgen generates ES module syntax:
export function initWorker() { ... }
```

### Why #2: Why was incompatible code written?

**Answer**: The embedded JavaScript (`WORKER_JS` constant) was never executed during
development. It was treated as "just a string" that Rust compiled, with no validation
of its runtime behavior.

**Evidence**:
- `cargo check` passed (Rust syntax valid)
- `cargo build` passed (WASM compiled)
- No browser execution occurred before marking complete

### Why #3: Why was the JavaScript never executed during development?

**Answer**: The development workflow relied solely on Rust compilation tools
(`cargo check`, `cargo build`) and never invoked browser-based testing tools
(`probar test`, manual browser testing).

**Evidence**:
```bash
# What was run:
cargo check  # Only checks Rust syntax
make build   # Only compiles WASM

# What was NOT run:
probar test  # Would have caught runtime errors
# Manual browser test - would have shown console errors
```

### Why #4: Why weren't browser tests run?

**Answer**: There was no **mandatory gate** requiring browser tests to pass before
a task could be marked complete. The development process allowed completion based
on compilation success alone.

**Evidence**:
- Todo items marked "completed" after `cargo check` passed
- No checkpoint requiring `probar test` before completion
- CLAUDE.md guidance was ignored/bypassed

### Why #5: Why was there no mandatory browser testing gate?

**Answer**: The specification (`wapr-worker-spec.md`) did not include **Jidoka**
(autonomation/stop-the-line) principles. It lacked:
1. Explicit test commands that MUST pass
2. Definition of "done" requiring runtime verification
3. Automated enforcement of testing requirements

**Root Cause Identified**:
> The specification allowed task completion without runtime verification.
> No Jidoka mechanism existed to "stop the line" when tests weren't run.

---

## Toyota Way Countermeasures

### 1. Jidoka (Autonomation) - Build Quality In

**Countermeasure**: Add mandatory testing gates to the specification that
MUST pass before any task can be marked complete.

```yaml
# Required in spec
completion_criteria:
  - command: "probar test -v"
    must_pass: true
    description: "Browser runtime tests"
  - command: "make test-browser"
    must_pass: true
    description: "E2E browser validation"
```

### 2. Genchi Genbutsu (Go and See)

**Countermeasure**: Require actual browser verification, not just compilation.

```markdown
## Definition of Done

1. [ ] `cargo check` passes
2. [ ] `cargo build --release` passes
3. [ ] `probar test -v` passes (MANDATORY - runtime verification)
4. [ ] Manual browser test: Open demo, click Record, verify console has no errors
5. [ ] Screenshot of working feature attached
```

### 3. Poka-Yoke (Error-Proofing)

**Countermeasure**: Add pre-completion hooks that automatically run browser tests.

```bash
# In pmat work complete hook:
probar test -v || {
    echo "ERROR: Browser tests failed. Cannot complete task."
    exit 1
}
```

### 4. Andon (Visual Management)

**Countermeasure**: Clear status indicators for test coverage.

```markdown
## Test Status Board

| Component | Unit Tests | Browser Tests | Manual Verified |
|-----------|------------|---------------|-----------------|
| ring_buffer.rs | [ ] | [ ] | [ ] |
| audio_worklet.rs | [ ] | [ ] | [ ] |
| worker.rs | [ ] | [ ] | [ ] |
| worker_manager.rs | [ ] | [ ] | [ ] |
```

---

## Updated Specification Requirements

### MANDATORY: No Task Completion Without Browser Tests

The following commands MUST pass before ANY worker-related task can be
marked complete:

```bash
# Tier 1: Compilation (necessary but NOT sufficient)
cargo check
cargo build --target wasm32-unknown-unknown

# Tier 2: Browser Runtime (MANDATORY)
probar test -v

# Tier 3: Manual Verification (MANDATORY for new features)
# Open http://localhost:8081
# Click Record button
# Speak for 3 seconds
# Verify transcript appears
# Screenshot console (no errors)
```

### Definition of "Complete"

A task is ONLY complete when:

1. **Compilation passes** - Rust and WASM build without errors
2. **Browser tests pass** - `probar test -v` exits 0
3. **Runtime verified** - Feature works in actual browser
4. **No console errors** - Browser console shows no errors during use
5. **Evidence provided** - Screenshot or test output attached

### Forbidden Practices

1. **NEVER** mark a task complete based solely on `cargo check`
2. **NEVER** claim "build successful" without browser verification
3. **NEVER** skip `probar test` for browser-related code
4. **NEVER** treat embedded JavaScript as "just strings"

---

## Immediate Fixes Required

### Fix 1: Worker Must Be ES Module

```javascript
// Create worker as ES module, not classic worker
const worker = new Worker(workerUrl, { type: 'module' });
```

### Fix 2: Use Dynamic Import Instead of importScripts

```javascript
// WRONG
importScripts('./whisper_apr_demo.js');

// CORRECT (ES module)
const wasm = await import('./pkg/whisper_apr_demo.js');
await wasm.default();
```

### Fix 3: AudioWorklet Must Load as Module

```javascript
// AudioWorklet modules are already ES modules
// But the blob URL approach may not work
// Need to serve worklet from actual URL or inline
```

---

## Verification Checklist

After implementing fixes, ALL must be checked:

- [ ] `cargo check` passes
- [ ] `cargo build --release` passes
- [ ] `make build` completes
- [ ] `probar test -v` passes
- [ ] Browser console shows no errors
- [ ] Recording captures audio (VU meter moves)
- [ ] Worker loads model (console shows "[Worker] Model loaded")
- [ ] Transcription appears when speaking
- [ ] Stop button works correctly

---

## References

- Ohno, T. (1988). Toyota Production System: Beyond Large-Scale Production
- Liker, J. (2004). The Toyota Way: 14 Management Principles
- Shingo, S. (1986). Zero Quality Control: Source Inspection and the Poka-Yoke System
