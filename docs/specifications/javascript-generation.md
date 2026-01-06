# JavaScript Generation Specification

## WAPR-JS-001: Zero Raw JavaScript Policy

### Status: IMPLEMENTED

### Overview

All JavaScript code in the whisper.apr demo is generated from Rust using the
`probar-js-gen` DSL. **Zero raw JavaScript strings are permitted.**

### Rationale

Raw JavaScript strings in Rust codebases cause:
1. **Silent failures**: Typos, syntax errors, reserved words pass the Rust compiler
2. **Untraceable modifications**: Generated files can be edited manually
3. **Testing gaps**: JS logic isn't covered by Rust tests

### Solution: probar-js-gen DSL

| Feature | Guarantee | Verification |
|---------|-----------|--------------|
| Type-safe HIR | Invalid JS unrepresentable | Compile-time |
| Identifier validation | Reserved words rejected | Runtime + tests |
| Blake3 manifests | Tampering detected | `verify()` function |
| Determinism | Same input = same output | Property tests |
| Forbidden patterns | window, eval blocked | Static analysis |

### Files Using DSL

| File | Purpose | Tests |
|------|---------|-------|
| `audioworklet_js.rs` | AudioWorklet processor | 20 tests |
| `worker_js.rs` | Transcription worker | 11 tests |

**Location:** `demos/www-demo/src/`

### Forbidden Patterns

These patterns **MUST NOT** appear in generated JavaScript:

| Pattern | Reason | Context |
|---------|--------|---------|
| `window.` | Workers have no window | Worker/Worklet |
| `document.` | Workers have no document | Worker/Worklet |
| `importScripts` | Use dynamic `import()` | Worker |
| `eval(` | Security risk | All |
| `Function(` | Security risk | All |
| `with(` | Deprecated | All |
| `__proto__` | Prototype pollution | All |

### Required Patterns

#### Web Worker

| Pattern | Purpose |
|---------|---------|
| `self.` | Worker global object |
| `import(` | Dynamic ES module import |
| `self.onmessage` | Message handler |
| `self.postMessage` | Send messages to main thread |

#### AudioWorklet

| Pattern | Purpose |
|---------|---------|
| `extends AudioWorkletProcessor` | Base class |
| `process(inputs, outputs, params)` | Audio callback |
| `return true` | Keep processor alive |
| `registerProcessor` | Register with audio thread |
| `Atomics.load/store/notify` | Lock-free ring buffer |

### Validation

Each generated JS file is validated:

```rust
// Worker validation
let js = generate_worker_js();
let errors = probar_js_gen::validator::validate_worker_js(&js);
assert!(errors.is_empty(), "Validation errors: {:?}", errors);

// Worklet validation
let js = generate_audioworklet_js();
let errors = probar_js_gen::validator::validate_worklet_js(&js);
assert!(errors.is_empty(), "Validation errors: {:?}", errors);
```

### Immutability Enforcement

Generated JavaScript MUST include manifests for tampering detection:

```rust
use probar_js_gen::manifest::{write_with_manifest, verify, GenerationMetadata};

// Write with manifest
write_with_manifest(
    Path::new("./worker.js"),
    &js,
    GenerationMetadata {
        tool: "whisper-apr-demo".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        input_hash: blake3::hash(SOURCE).to_hex().to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        regenerate_cmd: "cargo build --target wasm32-unknown-unknown".to_string(),
    },
)?;

// Verify before use
verify(Path::new("./worker.js"))?;  // Fails if file was manually edited
```

### Quality Gates

| Metric | Minimum | Target |
|--------|---------|--------|
| Test coverage | 90% | 95% |
| Mutation score | 85% | 90% |
| Property tests | 100 cases | 1000 cases |
| Clippy warnings | 0 | 0 |

### References

1. **probar-js-gen**: `../../probar/crates/probar-js-gen/`
2. **FALSIFICATION.md**: 100-point Popperian checklist
3. **DO-178C** (2011): Software Considerations in Airborne Systems
4. **Leveson** (2012): Engineering a Safer World

### Compliance

This specification implements:
- DO-178C DAL C: Appropriate for software with major failure consequences
- Toyota Way Jidoka: Stop the line on any validation failure
- Toyota Way Poka-Yoke: Type system prevents invalid constructs

---

_Version: 1.0.1_
_Last updated: 2026-01-06_
