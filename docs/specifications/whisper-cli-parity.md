# whisper-apr CLI Parity Specification

**Version**: 2.1.0
**Status**: UNIFIED CHECKLIST (210 points) + P0 ZERO MOCK TOLERANCE
**Created**: 2025-12-18
**Methodology**: EXTREME TDD + Toyota Way + Popperian Falsification
**Target Coverage**: ≥95% line coverage
**Parity Reference**: whisper.cpp (ggerganov/whisper.cpp)

---

## Table of Contents

- [§1. Executive Summary](#1-executive-summary)
- [§2. Design Principles](#2-design-principles)
- [§3. Toyota Way Alignment](#3-toyota-way-alignment)
- [§4. CLI Architecture](#4-cli-architecture)
- [§5. Command Parity Matrix](#5-command-parity-matrix)
- [§6. Argument Parity Specification](#6-argument-parity-specification)
- [§7. Output Format Parity](#7-output-format-parity)
- [§8. Performance Parity Requirements](#8-performance-parity-requirements)
- [§9. EXTREME TDD Methodology](#9-extreme-tdd-methodology)
- [§10. Parity Testing Framework](#10-parity-testing-framework)
- [§11. UNIFIED 210-POINT MASTER FALSIFICATION CHECKLIST](#11-unified-210-point-master-falsification-checklist)
- [§12. Quality Gates](#12-quality-gates)
- [§13. Peer-Reviewed Citations](#13-peer-reviewed-citations)
- [§14. References](#14-references)
- [§15. Ecosystem Dependencies](#15-ecosystem-dependencies)

---

## §1. Executive Summary

This specification defines **complete CLI parity** between `whisper-apr` and `whisper.cpp`, ensuring:

1. **Argument-level compatibility** - All whisper.cpp CLI flags have whisper-apr equivalents
2. **Output format compatibility** - Identical TXT/SRT/VTT/JSON/CSV/LRC output
3. **Performance parity** - Real-time factor (RTF) within 10% of whisper.cpp on equivalent hardware
4. **Behavioral equivalence** - Same audio input produces semantically identical transcriptions

The specification follows the **aprender ecosystem** conventions with apr-cli patterns, **realizar-style parity testing**, and **Popperian falsification methodology** to scientifically verify claims.

---

## §2. Design Principles

### §2.1 Core Tenets

| Principle | Description | Citation |
|-----------|-------------|----------|
| **Testable Logic Separation** | All CLI logic resides in library (`src/cli/`), binary is thin shell | [1] Martin, Clean Architecture |
| **Popperian Falsifiability** | Every claim must be testable and disprovable | [2] Popper, Logic of Scientific Discovery |
| **Zero-Overhead Abstraction** | CLI wrapper adds <1% latency vs direct library call | [3] Stroustrup, C++ Design |
| **Fail-Fast Error Handling** | No silent failures; explicit error messages with exit codes | [4] Shore, Fail Fast |
| **Deterministic Reproducibility** | Same input + seed → identical output across runs | [5] Sculley, ML Systems |
| **Fail-Safe Defaults** | Security configuration defaults to highest safety (e.g., path restrictions) | [14] Saltzer & Schroeder, Protection |
| **Single Inference Pathway** | WASM demos and CLI use identical code paths | [18] Parnas, Software Aging |

### §2.2 Unified Inference Pathway (MANDATORY)

**MANDATORY**: The WASM demo and CLI MUST share a **single inference code path**.

### §2.3 Zero Mock Tolerance (P0 BLOCKER)

**P0 CRITICAL**: ANY mock test or mock code in the codebase results in **automatic 0/100 score**.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    ZERO MOCK TOLERANCE POLICY (P0)                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  IF grep -rn "mock\|Mock\|MOCK\|stub\|Stub\|fake\|Fake" tests/ src/       │
│     RETURNS ANY MATCH                                                      │
│  THEN                                                                      │
│     SCORE = 0/100                                                          │
│     STATUS = BLOCKED                                                       │
│     GRADE = F (Automatic Failure)                                          │
│                                                                            │
│  RATIONALE:                                                                │
│  - Mocks hide real integration bugs [23] Fowler, "Mocks Aren't Stubs"     │
│  - Mocks create false confidence in broken code                            │
│  - Real integration tests catch real bugs                                  │
│  - Toyota Way: Genchi Genbutsu (go and see the actual process)            │
│                                                                            │
│  PROHIBITED PATTERNS:                                                      │
│  - `let simulated = ...` (fake data instead of real inference)            │
│  - `// TODO: Load model` (placeholder instead of real test)               │
│  - `#[ignore]` without real integration alternative                        │
│  - Hardcoded expected values without actual model execution                │
│                                                                            │
│  REQUIRED:                                                                 │
│  - All tests MUST load real models                                         │
│  - All tests MUST process real audio                                       │
│  - All tests MUST call actual inference code                               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**QA Enforcement**: Before scoring ANY section, QA MUST first run:
```bash
# P0 BLOCKER CHECK - Run FIRST before any other validation
grep -rn "mock\|Mock\|MOCK\|stub\|Stub\|fake\|Fake\|Simulated\|simulated" \
    tests/ src/ --include="*.rs" | grep -v "// OK:" || echo "CLEAN"

# If ANY matches found: SCORE = 0/100, STOP VALIDATION
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED INFERENCE PATHWAY                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────────────────┐    │
│   │  WASM Demo  │──┐   │   CLI       │──┐   │ Integration Tests       │──┐ │
│   │  (browser)  │  │   │  (native)   │  │   │ (cargo test)            │  │ │
│   └─────────────┘  │   └─────────────┘  │   └─────────────────────────┘  │ │
│                    │                    │                                │ │
│                    ▼                    ▼                                ▼ │
│              ┌──────────────────────────────────────────────────────────┐  │
│              │           whisper_apr::WhisperApr::transcribe()          │  │
│              │                (SINGLE ENTRY POINT)                      │  │
│              └──────────────────────────────────────────────────────────┘  │
│                                        │                                   │
│                    ┌───────────────────┼───────────────────┐               │
│                    ▼                   ▼                   ▼               │
│              ┌──────────┐       ┌──────────┐        ┌──────────┐          │
│              │  mel.rs  │       │encoder.rs│        │decoder.rs│          │
│              └──────────┘       └──────────┘        └──────────┘          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**Parity Invariant**: If WASM demo produces correct output for audio X, then CLI MUST produce identical output for audio X. 

**Requirements**:
1. **Single Library Path**: No separate mel/encoder/decoder implementations per target.
2. **Deterministic Output**: Same audio → identical text output across all platforms.
3. **Trace Parity**: Performance spans (renacer) must be identical in sequence and name.

---

## §3. Toyota Way Alignment

This specification embodies the **14 Principles of the Toyota Production System** [6]:

| TPS Principle | CLI Implementation | Verification |
|---------------|-------------------|--------------|
| **Genchi Genbutsu** | Compare actual whisper.cpp output byte-for-byte | Parity tests §10 |
| **Jidoka** | Compilation fails if parity tests fail | CI gates §12 |
| **Kaizen** | Performance regression alerts on every PR | Benchmarks §8 |
| **Poka-Yoke** | Type-safe argument parsing prevents invalid states | clap derive macros |

---

## §11. UNIFIED 210-POINT MASTER FALSIFICATION CHECKLIST

**Version**: 2.0.0
**Supersedes**: Former §11 (110 points) + Former §16 (100 points)
**Methodology**: Popperian Falsification + Toyota Way Five-Whys
**Unified Pathway**: WASM Demo and CLI share identical inference code path (§2.2)

### Part I: CLI Infrastructure (110 points - Sections A-H)

#### Section A: Argument Parsing (15 points)

| # | Check | Command/Action | Expected | Pass | Fail |
|---|-------|----------------|----------|------|------|
| A.1 | `--help` displays usage | `whisper-apr --help` | Usage: info | [x] | [ ] |
| A.2 | `-h` short form works | `whisper-apr -h` | Same as --help | [x] | [ ] |
| A.3 | `--version` shows version | `whisper-apr --version` | Semver format | [x] | [ ] |
| A.4 | Unknown flag errors | `whisper-apr --invalid` | Exit code ≠ 0 | [x] | [ ] |
| A.5 | Missing required arg | `whisper-apr transcribe` | Error: -f required | [x] | [ ] |
| A.6 | Invalid type rejected | `--threads abc` | Type error | [x] | [ ] |
| A.7 | Negative threads rejected | `--threads -1` | Range error | [x] | [ ] |
| A.8 | Temperature range valid | `--temperature 2.0` | Range error | [x] | [ ] |
| A.9 | Model file not found | `-m nonexistent.bin` | File error | [x] | [ ] |
| A.10 | Audio file not found | `-f nonexistent.wav` | File error | [x] | [ ] |
| A.11 | Response file works | `@args.txt` | Args from file | [ ] | [x] |
| A.12 | Conflicting flags error | `--quiet --verbose` | Conflict error | [x] | [ ] |

---

### Part II: Transcription Pipeline (100 points - Sections T0-T9)

#### Section T0: Unified Pathway Verification (5 points)

**Invariant**: CLI and WASM Demo MUST use identical library code paths (§2.2).

| # | Claim to Falsify | Verification | Expected | Pass | Fail |
|---|------------------|--------------|----------|------|------|
| T0.1 | Single library entry point | Code inspection | CLI calls `WhisperApr::transcribe` | [ ] | [ ] |
| T0.2 | No platform-specific mel | Grep for mel imports | No duplicate mel code | [ ] | [ ] |
| T0.3 | Identical encoder dispatch | Grep for encoder imports | No duplicate encoder | [ ] | [ ] |
| T0.4 | Identical token suppression | Grep for decoder imports | No duplicate decoder | [ ] | [ ] |
| T0.5 | Identical output for test audio | Run both, compare text | Text matches exactly | [ ] | [ ] |

#### Section T1: Audio Input Pipeline (15 points)

| # | Claim to Falsify | Expected Result | Pass | Fail |
|---|------------------|-----------------|------|------|
| T1.1 | 16kHz mono WAV transcribes | Non-empty text output | [ ] | [x] |
| T1.6 | 24-bit audio depth handled | Valid transcription | [x] | [ ] |
| T1.7 | 32-bit float audio handled | Valid transcription | [x] | [ ] |
| T1.8 | Very short audio (<0.5s) | Output or "no speech" | [x] | [ ] |

---

### §11.1 UNIFIED MASTER SCORECARD

**Status**: Baseline Validation (2025-12-19)

| Component | Points | Available | Earned |
|-----------|--------|-----------|--------|
| **Part I** | CLI Infrastructure | 110 | 85 (est) |
| **Part II** | Transcription Pipeline | 100 | 68 |
| **T0** | Integration Verification | 5 | 0 (pend) |
| **TOTAL** | | **215** | **153** |

**UNIFIED GRADE**: **C** (Alpha Quality - Baseline established)

---

### §11.2 Diagnostic Findings (Latest: 2025-12-19)

1. **Hallucination Loop (CRITICAL)**: Model stuck in repetitive `. . . .` generation. Verified via `validate` command showing LayerNorm mean ~11.0 (Critical Failure). Ref: [19] Ba et al.
2. **Audio Fixes**: ✅ 24-bit PCM and 32-bit float (WAVE_FORMAT_EXTENSIBLE) now parse correctly (T1.6, T1.7).
3. **Model Loading**: ✅ Auto-download from HuggingFace functional; CLI now supports `--model-path` or cached default.

---

## §12. Quality Gates

All shell scripts MUST pass **bashrs** validation (NOT shellcheck) following aprender ecosystem conventions.

---

## §13. Peer-Reviewed Citations

### [1] Martin, R.C. (2017)
**"Clean Architecture"** - Informs testable logic separation (§4).

### [18] Parnas, D.L. (1994)
**"Software Aging"** - Mandates single inference pathway (§2.2).

### [19] Ba, J.L., et al. (2016)
**"Layer Normalization"** - Theoretical basis for weight saturation diagnostics (T4.11).

### [20] Gulati, A., et al. (2020)
**"Conformer"** - Validates encoder convolutional-transformer hybrid (T3).

### [21] Chen, S., et al. (2022)
**"WavLM"** - Industry standards for mel spectrogram processing (T2).

### [22] Graves, A., et al. (2006)
**"CTC"** - Underlying theory for timestamp alignment verification (T7).

### [23] Fowler, M. (2007)
**"Mocks Aren't Stubs"** - Defines mock/stub distinction; justifies Zero Mock Tolerance policy (§2.3). Real integration tests catch real bugs; mocks create false confidence.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 2.1.0 | 2025-12-19 | whisper-apr team | **P0 ZERO MOCK TOLERANCE**: Added §2.3 - ANY mock/stub code = automatic 0/100. Citation [23] Fowler. |
| 2.0.0 | 2025-12-19 | whisper-apr team | **UNIFIED MASTER CHECKLIST**: Merged CLI Infrastructure and Transcription Pipeline. Mandated §2.2 Unified Inference Pathway. Added citations [18-22]. |
| 1.0.0 | 2025-12-18 | whisper-apr team | Initial CLI parity specification. |