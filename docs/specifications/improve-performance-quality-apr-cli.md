# APR CLI Performance & Quality Improvement Specification

**WAPR-APR-CLI-001: Systematic Quality, Performance & Feature Completeness for `apr` Subcommand Suite**

| Field | Value |
|-------|-------|
| Status | PLANNED |
| Author | Claude Code |
| Created | 2026-02-10 |
| Toyota Way Phase | Kaizen (改善) — Continuous Improvement |
| Upstream Sync | aprender 0.25.2 (local), 0.25.1 (crates.io) |
| Spec Predecessor | `update-improve-performance.md` (WAPR-PERF-002) |
| PMAT Comply | Full — work tickets, coverage gates, complexity guards |

---

## 1. Executive Summary

The `apr` subcommand suite (15 commands, 1488 LOC across `apr_commands.rs` + `apr_args.rs`) was integrated into `whisper-apr-cli` to expose aprender's format library for model inspection, conversion, and diagnostics. This specification defines a systematic plan to:

1. **Sync to latest aprender 0.25.x API** — expose 12 unused modules (golden traces, signing, encryption, layout contracts, validated tensors, model families, sharded import, quantization, export, compare, f16 safety, homomorphic encryption)
2. **Profile and optimize hot paths** — fingerprint takes O(n) tensor loads; diff loads two full models; import has noisy debug output
3. **Achieve world-class quality** — 95%+ coverage, zero clippy warnings, Popperian falsification suite, mutation testing
4. **Add missing CLI affordances** — `--json` everywhere, `--quiet` suppression, `--timing`, `--parallel` for batch ops

### Current State (Baseline)

| Metric | Current | Target | PMAT Ticket |
|--------|---------|--------|-------------|
| Commands implemented | 15 | 25+ | WAPR-APR-CLI-002 |
| Unit tests | 7 | 40+ | WAPR-APR-CLI-003 |
| Line coverage (apr_commands.rs) | ~12% (unit only) | 85%+ | WAPR-APR-CLI-004 |
| aprender API surface used | 8/20 modules | 20/20 | WAPR-APR-CLI-005 |
| Clippy status | Clean (our files) | Clean | — |
| Cyclomatic complexity max | ~10 | ≤30 (hook) | — |
| Cognitive complexity max | ~8 | ≤25 (hook) | — |

---

## 2. Peer-Reviewed Citations

The design decisions in this specification are grounded in established research:

| # | Citation | Relevance |
|---|----------|-----------|
| C1 | Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge. | Falsification methodology: every claim requires a testable prediction that could fail. §5 applies this to each performance claim. |
| C2 | Gregg, B. (2020). *Systems Performance: Enterprise and the Cloud*, 2nd ed. Addison-Wesley. Ch. 2 "Methodology", Ch. 6 "CPU". | USE method for profiling (Utilization, Saturation, Errors). Applied in §4 for tensor load bottleneck analysis. |
| C3 | Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*. Productivity Press. | Jidoka (stop-on-error), Poka-Yoke (mistake-proofing), Genchi Genbutsu (go and see). Applied in §3 via validated tensors and layout contracts. |
| C4 | Bernstein, D. J. et al. (2012). "High-speed high-security signatures." *Journal of Cryptographic Engineering*, 2(2), 77–89. | Ed25519 design rationale. Applied in §3.7 for model provenance signing. |
| C5 | Just, R. et al. (2014). "Are mutants a valid substitute for real faults in software testing?" *Proc. SIGSOFT FSE 2014*, pp. 654–665. ACM. | Mutation testing validity. Applied in §5.4 for mutation score targets. |
| C6 | Nagappan, N. & Ball, T. (2005). "Use of relative code churn measures to predict system defect density." *Proc. ICSE 2005*, pp. 284–292. ACM. | Churn-defect correlation. Applied in §5.3 via `pmat query --churn`. |
| C7 | Sajnani, H. et al. (2016). "SourcererCC: Scaling code clone detection." *Proc. ICSE 2016*, pp. 1157–1168. ACM. | Code clone detection. Applied in §5.3 via `pmat query --duplicates`. |
| C8 | McCabe, T. J. (1976). "A Complexity Measure." *IEEE Transactions on Software Engineering*, SE-2(4), 308–320. | Cyclomatic complexity. Applied in §6 for pre-commit hook thresholds. |
| C9 | Dequantization fused with matmul: Frantar, E. et al. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023*. | Quantization correctness. Applied in §3.11 for Q4_0/Q8_0 round-trip verification. |
| C10 | Gentry, C. (2009). "A Fully Homomorphic Encryption Scheme." Stanford PhD Thesis. | HE theory. Applied in §3.12 for privacy-preserving model inspection feasibility. |

---

## 3. Feature Gap Analysis — Unused aprender 0.25.x Modules

Each subsection identifies an aprender module NOT currently exposed via `whisper-apr-cli apr`, proposes a subcommand, and defines acceptance criteria.

### 3.1 Golden Trace Verification (`aprender::format::golden`)

**Module**: `golden.rs` — Proves model authenticity by comparing logit outputs against known-good traces.

**Proposed command**: `apr golden verify <MODEL> --trace <GOLDEN_FILE> [--tolerance F]`

**API**: `verify_logits(model_logits, golden: &GoldenTrace) -> TraceVerifyResult`

**PMAT Ticket**: WAPR-APR-CLI-010

**Acceptance Criteria**:
- [ ] Loads golden trace file (JSON with `GoldenTraceSet`)
- [ ] Runs model forward pass on trace input tokens
- [ ] Compares logit outputs within tolerance
- [ ] Reports pass/fail per layer, overall pass rate
- [ ] `--json` output includes per-token divergence

**Falsification (C1)**:
> H₀: "Golden verification catches all weight corruption."
> Test: Flip 1 bit in a random weight tensor, re-run verification. If it still passes, H₀ is falsified.

### 3.2 Model Signing (`aprender::format::signing`, feature: `format-signing`)

**Module**: `signing.rs` — Ed25519 signatures for model provenance (C4).

**Proposed commands**:
- `apr sign <MODEL> --key <PRIVATE_KEY>` — Sign a model file
- `apr verify-sig <MODEL> --pubkey <PUBLIC_KEY>` — Verify signature

**API**: `sign_model(path, key) -> Result<()>`, `verify_signature(path, pubkey) -> Result<bool>`

**PMAT Ticket**: WAPR-APR-CLI-011

**Acceptance Criteria**:
- [ ] Generate Ed25519 keypair (`apr keygen -o key.pem`)
- [ ] Sign produces valid 64-byte signature appended to file
- [ ] Verify returns exit code 0 on valid, 1 on invalid
- [ ] Tampered file fails verification
- [ ] Feature-gated: `--features format-signing`

**Falsification (C1)**:
> H₀: "Signed models are tamper-evident."
> Test: Sign model, flip 1 byte in payload, verify. Must fail. Flip 1 byte in signature, verify. Must fail.

### 3.3 Model Encryption (`aprender::format::encryption`, feature: `format-encryption`)

**Module**: `encryption.rs` — AES-256-GCM encryption with Argon2id key derivation.

**Proposed commands**:
- `apr encrypt <MODEL> -o <OUTPUT> --password` — Encrypt model
- `apr decrypt <MODEL> -o <OUTPUT> --password` — Decrypt model

**PMAT Ticket**: WAPR-APR-CLI-012

**Acceptance Criteria**:
- [ ] Password-based encryption with Argon2id KDF
- [ ] X25519 public-key encryption alternative (`--recipient <PUBKEY>`)
- [ ] Round-trip: encrypt → decrypt → diff shows IDENTICAL
- [ ] Wrong password returns clear error (not corruption)
- [ ] Feature-gated: `--features format-encryption`

**Falsification (C1)**:
> H₀: "Encrypted models are confidential."
> Test: Encrypt, attempt to load without decryption. Must fail with clear error, not partial data.

### 3.4 Layout Contract Verification (`aprender::format::layout_contract`)

**Module**: `layout_contract.rs` — Row-major mandate enforcement (LAYOUT-002).

**Proposed command**: `apr contract verify <MODEL> [--strict]`

**API**: `contract::validate(tensor_info) -> Result<(), ContractError>`

**PMAT Ticket**: WAPR-APR-CLI-013

**Acceptance Criteria**:
- [ ] Checks all tensors against LAYOUT-002 row-major mandate
- [ ] Reports contract violations (wrong layout, bad alignment, invalid block sizes)
- [ ] `--strict` fails on warnings, not just errors
- [ ] `--json` output with per-tensor contract status

**Falsification (C1)**:
> H₀: "All APR models satisfy the row-major contract."
> Test: Create a model with a manually transposed tensor. Contract verify must flag it.

### 3.5 Validated Tensors (`aprender::format::validated_tensors`)

**Module**: `validated_tensors.rs` — Compile-time Poka-Yoke contract enforcement (C3).

**Proposed command**: `apr validate <MODEL> [--strict]`

**API**: `ValidatedWeight::try_from(tensor) -> Result<ValidatedWeight<RowMajor>, ContractValidationError>`

**PMAT Ticket**: WAPR-APR-CLI-014

**Acceptance Criteria**:
- [ ] Validates each tensor type (embedding, weight, vector) against compile-time contracts
- [ ] Reports: valid count, invalid count, per-tensor errors
- [ ] Catches: NaN/Inf values, wrong dimensionality, shape mismatches
- [ ] `--json` output with detailed validation results

### 3.6 Model Family Contracts (`aprender::format::model_family`)

**Module**: `model_family.rs` + `model_family_loader.rs` — YAML-driven model family specs.

**Proposed command**: `apr family identify <MODEL>` / `apr family check <MODEL> --family <NAME>`

**API**: `ModelFamily::identify(inspection_report) -> Option<FamilyMatch>`

**PMAT Ticket**: WAPR-APR-CLI-015

**Acceptance Criteria**:
- [ ] Auto-identifies model family from tensor name patterns
- [ ] Validates tensor names/shapes against family contract YAML
- [ ] Reports missing tensors, extra tensors, shape mismatches
- [ ] Supports: llama, whisper, bert, qwen2, phi, mistral

### 3.7 Sharded Import (`aprender::format::sharded`)

**Module**: `sharded/mod.rs` — Multi-file model import with streaming.

**Proposed command**: `apr import-sharded <DIR_OR_REPO> -o <OUTPUT> [--cache-dir DIR]`

**API**: `ShardedImporter::new(config).import(source, output) -> ImportReport`

**PMAT Ticket**: WAPR-APR-CLI-016

**Acceptance Criteria**:
- [ ] Detects sharded repos (`is_sharded_model()`)
- [ ] Streams shards with progress reporting (`ImportProgress`)
- [ ] Memory-bounded: `estimate_shard_memory()` pre-check
- [ ] Caching: `ShardCache` for resumed downloads
- [ ] `--json` output with per-shard status

### 3.8 Weight Comparison (`aprender::format::compare`)

**Module**: `compare.rs` — Statistical comparison of model weights.

**Proposed command**: `apr compare <MODEL1> <MODEL2> [--threshold F] [--top N]`

**PMAT Ticket**: WAPR-APR-CLI-017

**Acceptance Criteria**:
- [ ] Per-tensor cosine similarity, L2 distance, max absolute diff
- [ ] Top-N most divergent tensors highlighted
- [ ] Threshold-based pass/fail for CI pipelines
- [ ] `--json` output with full comparison matrix

### 3.9 Quantization (`aprender::format::quantize`, feature: `format-quantize`)

**Module**: `quantize.rs` — Q4_0, Q8_0 quantizers.

**Proposed command**: `apr quantize <MODEL> -o <OUTPUT> --type <q4_0|q8_0>`

**API**: `Q8_0Quantizer::quantize(data) -> QuantizedTensor`, `dequantize(qt) -> Vec<f32>`

**PMAT Ticket**: WAPR-APR-CLI-018

**Acceptance Criteria**:
- [ ] Quantize F32/F16 model to Q4_0 or Q8_0
- [ ] Report compression ratio, max quantization error
- [ ] `--verify` flag: dequantize and compare against original
- [ ] Feature-gated: `--features format-quantize`

**Falsification (C1, C9)**:
> H₀: "Q8_0 quantization preserves model quality within 1% perplexity."
> Test: Quantize, dequantize, compute max/mean/std of error per tensor. If max error exceeds `2^-7` (Q8 precision), flag.

### 3.10 Export (`aprender::format::converter::export`)

**Proposed command**: `apr export <APR_MODEL> -o <OUTPUT> --format <gguf|safetensors>`

**API**: `apr_export(source, output, ExportOptions) -> ExportReport`

**PMAT Ticket**: WAPR-APR-CLI-019

**Acceptance Criteria**:
- [ ] Export APR → GGUF with correct column-major transpose (LAYOUT-002)
- [ ] Export APR → SafeTensors (native row-major, no transpose)
- [ ] Report: tensor count, format, any dropped metadata
- [ ] `--verify`: round-trip check after export

### 3.11 F16 Safety (`aprender::format::f16_safety`)

**Module**: `f16_safety.rs` — NaN/Inf prevention for F16 models.

**Proposed command**: `apr f16-audit <MODEL>`

**PMAT Ticket**: WAPR-APR-CLI-020

**Acceptance Criteria**:
- [ ] Scans all F16 tensors for NaN, Inf, subnormal values
- [ ] Reports: affected tensors, count of anomalous values, percentage
- [ ] Suggests: clamp ranges, safe conversion parameters
- [ ] `--json` output

### 3.12 Homomorphic Encryption (`aprender::format::homomorphic`, feature: `format-homomorphic`)

**Proposed command**: `apr he-inspect <ENCRYPTED_MODEL>` — Inspect HE-encrypted model metadata without decryption.

**PMAT Ticket**: WAPR-APR-CLI-021

**Acceptance Criteria**:
- [ ] Read HE parameters (scheme, security level, poly modulus degree)
- [ ] Report: tensor count (from metadata), estimated compute overhead
- [ ] Feature-gated: `--features format-homomorphic`
- [ ] Does NOT decrypt — metadata-only inspection

---

## 4. Performance Profiling & Optimization Plan

### 4.1 Profiling Methodology (C2 — USE Method)

For each command, measure:
- **Utilization**: CPU time vs wall time (parallelism opportunities)
- **Saturation**: Memory peak, allocation pressure
- **Errors**: Failure modes under stress (large models, corrupt files)

```bash
# Profile fingerprint command (known hot path)
pmat query "run_rosetta_fingerprint" --include-source --churn

# Benchmark with hyperfine
hyperfine --warmup 2 \
  'whisper-apr-cli apr fingerprint model.safetensors' \
  'whisper-apr-cli apr inspect model.safetensors'

# Memory profiling with heaptrack
heaptrack whisper-apr-cli apr fingerprint model.safetensors
```

### 4.2 Known Bottlenecks

| Command | Bottleneck | Root Cause | Fix | PMAT Ticket |
|---------|-----------|------------|-----|-------------|
| `fingerprint` | O(n) sequential tensor loads | `load_tensor_f32` called per-tensor in loop | Batch load via `RosettaStone::load_all_tensors_f32` or Rayon parallel iter | WAPR-APR-CLI-030 |
| `diff` (large models) | 2× full model load | `diff_models` loads both models entirely | Use streaming comparison (load tensor-by-tensor, compare, discard) | WAPR-APR-CLI-031 |
| `import` | Noisy debug output | `aprender` prints `[DEBUG-TOK-PATH]`, `[PMAT-224]`, `[CONTRACT]` | Suppress via log level, or redirect stderr when `--quiet` | WAPR-APR-CLI-032 |
| `canary` | O(n) sequential tensor loads | Same as fingerprint — per-tensor `load_tensor_f32` | Same parallel fix | WAPR-APR-CLI-033 |
| `rosetta convert` | Single-threaded tensor conversion | Sequential tensor-by-tensor conversion | Rayon `par_iter` over tensor chunks | WAPR-APR-CLI-034 |
| `tree` (1000+ tensors) | String allocation churn | `insert_tensor_path` creates many small `String`s | Pre-allocate tree with capacity, use `&str` slices | WAPR-APR-CLI-035 |

### 4.3 Performance Targets

| Command | Current (whisper-tiny, 167 tensors) | Target | Method |
|---------|-------------------------------------|--------|--------|
| `inspect` | ~50ms | ~50ms (I/O bound) | No change needed |
| `tensors --stats` | ~200ms | ~100ms | Parallel stat computation |
| `fingerprint` | ~2.5s | ~500ms | Parallel tensor loads |
| `diff` (tiny vs base) | ~400ms | ~400ms (acceptable) | Monitor only |
| `import` (safetensors→apr) | ~7.2s | ~3s | Parallel tensor conversion |
| `canary` | ~3s | ~600ms | Parallel tensor loads |
| `tree` | ~10ms | ~10ms (fast enough) | No change needed |
| `flow` | ~50ms | ~50ms (fast enough) | No change needed |

### 4.4 Parallelization Strategy

```rust
// Before (sequential):
for name in &tensor_names {
    match rosetta.load_tensor_f32(&args.file, name) {
        Ok(data) => { /* compute stats */ }
        Err(_) => { /* skip */ }
    }
}

// After (parallel with Rayon):
use rayon::prelude::*;

let stats_list: Vec<TensorStatistics> = tensor_names
    .par_iter()
    .filter_map(|name| {
        let rosetta = RosettaStone::new(); // thread-local
        rosetta.load_tensor_f32(&args.file, name).ok().map(|data| {
            let shape = /* lookup */;
            TensorStatistics::from_f32(name, shape, &data)
        })
    })
    .collect();
```

**Dependency**: Rayon is already a transitive dependency via aprender/trueno. Verify with:
```bash
pmat query --literal "rayon" --files-with-matches
cargo tree -i rayon
```

---

## 5. Quality Plan — Popperian Falsification Suite

### 5.1 Falsification Hypotheses (C1)

Each hypothesis is a testable claim that we attempt to **disprove**. Surviving falsification attempts strengthens confidence.

| ID | Hypothesis | Falsification Test | PMAT Ticket |
|----|-----------|-------------------|-------------|
| F1 | "inspect produces correct tensor counts for all formats" | Compare inspect output against `safetensors` crate's header parse, GGUF reader, APR v2 reader. Any mismatch falsifies. | WAPR-APR-CLI-040 |
| F2 | "diff reports IDENTICAL for a file compared to itself" | `apr diff model.apr model.apr` must return 0 differences. Any non-zero falsifies. | WAPR-APR-CLI-041 |
| F3 | "rosetta convert is lossless for F32 SafeTensors→APR→SafeTensors" | Round-trip, then bitwise compare. Any bit difference falsifies. | WAPR-APR-CLI-042 |
| F4 | "canary detects single-weight perturbation" | Create canary, flip 1 float in model, re-create canary, diff. Checksums must differ. If identical, falsified. | WAPR-APR-CLI-043 |
| F5 | "lint catches all known anti-patterns" | Feed models with known issues (missing embedding, duplicate tensor names, zero-weight layers). Each must produce a finding. | WAPR-APR-CLI-044 |
| F6 | "hex dump is byte-accurate" | Dump first 16 bytes of a known file. Compare against `xxd` output. Any mismatch falsifies. | WAPR-APR-CLI-045 |
| F7 | "tree view includes all tensors" | Count leaves in tree output. Must equal `inspect` tensor count. | WAPR-APR-CLI-046 |
| F8 | "flow aggregates parameters correctly" | Sum params across all flow layers. Must equal `inspect` total_params. Any discrepancy falsifies. | WAPR-APR-CLI-047 |
| F9 | "fingerprint detects NaN injection" | Inject NaN into one tensor, run fingerprint. `has_anomalies` must be true. | WAPR-APR-CLI-048 |
| F10 | "import score improves with --quantize" | Import same model with and without quantize. Output size must be smaller. Score must not decrease. | WAPR-APR-CLI-049 |

### 5.2 Coverage Plan

```bash
# Discovery: find coverage gaps in apr_commands.rs
pmat query --coverage-gaps --limit 30 --exclude-tests

# Targeted: check specific function coverage
pmat query "run_inspect" --coverage --include-source --limit 1
pmat query "run_fingerprint" --coverage --include-source --limit 1
pmat query "format_model_error" --coverage --include-source --limit 1

# After writing tests, verify improvement
pmat query --coverage-gaps --limit 30 --exclude-tests
```

**Coverage Targets**:

| Scope | Current | Target | Strategy |
|-------|---------|--------|----------|
| `apr_commands.rs` helpers | ~100% | 100% | Already tested (7 tests) |
| `apr_commands.rs` handlers | ~0% | 85%+ | Integration tests with fixture models |
| `apr_args.rs` | ~0% (derive) | N/A | Clap derive — no custom logic to test |
| Error paths | ~30% | 90%+ | Feed corrupt/missing/wrong-format files |
| JSON output paths | ~0% | 80%+ | Assert JSON structure with `serde_json::from_str` |

### 5.3 PMAT Quality Enrichment

```bash
# Audit apr_commands.rs for code smells
pmat query "apr_commands" --churn --duplicates --entropy --faults

# Find fault patterns (unwrap, panic, unsafe)
pmat query "apr" --faults --exclude-tests --include-source

# Find duplicated patterns (DRY violations)
pmat query "run_rosetta" --duplicates --limit 10

# Find volatile code (frequent changes = defect risk, C6)
pmat query "apr_commands" --churn

# Find repetitive boilerplate (C7)
pmat query "apr_commands" --entropy
```

### 5.4 Mutation Testing (C5)

```bash
# Target apr_commands.rs specifically
cargo mutants --file src/cli/apr_commands.rs --timeout 60

# Expected surviving mutants (acceptable):
# - println! mutations (output format, not logic)
# - Timing code (Instant::now, elapsed)
#
# Unacceptable surviving mutants:
# - Error path mutations (format_model_error)
# - Business logic (extract_layers_from_tensors, infer_layer_type)
# - Data transformation (insert_tensor_path)
```

**Target**: >80% mutation score for helper functions, >60% for handler functions.

---

## 6. Implementation Phases

### Phase 1: Foundation (WAPR-APR-CLI-050)

**Effort**: 1 session | **Priority**: P0

- [ ] Sync aprender path dep to latest local (0.25.2) — verify `cargo build --features cli`
- [ ] Add integration test framework: fixture models (tiny SafeTensors, minimal APR v2, vocab-only GGUF)
- [ ] Write falsification tests F1–F4 (self-diff, round-trip, canary perturbation, tensor count)
- [ ] Add `--timing` global flag to all commands (print elapsed time to stderr)
- [ ] Suppress aprender debug output when `--quiet` is set (redirect stderr or set log level)
- [ ] Run `pmat query --coverage-gaps` and document baseline

**pmat comply check**:
```bash
pmat analyze complexity --max-cyclomatic 30 --max-cognitive 25 src/cli/apr_commands.rs
pmat query --coverage-gaps --limit 10 --exclude-tests
```

### Phase 2: New Commands — Tier A (WAPR-APR-CLI-051)

**Effort**: 1 session | **Priority**: P1

- [ ] `apr golden verify` (§3.1) — golden trace verification
- [ ] `apr validate` (§3.5) — validated tensor contracts
- [ ] `apr contract verify` (§3.4) — layout contract checking
- [ ] `apr family identify` / `apr family check` (§3.6) — model family detection
- [ ] `apr compare` (§3.8) — statistical weight comparison
- [ ] `apr export` (§3.10) — APR → GGUF/SafeTensors
- [ ] `apr f16-audit` (§3.11) — F16 NaN/Inf scanning

Write falsification tests F5–F10 for each new command.

### Phase 3: New Commands — Tier B, Feature-Gated (WAPR-APR-CLI-052)

**Effort**: 1 session | **Priority**: P2

- [ ] `apr sign` / `apr verify-sig` (§3.2) — Ed25519 signing (feature: `format-signing`)
- [ ] `apr encrypt` / `apr decrypt` (§3.3) — AES-256-GCM (feature: `format-encryption`)
- [ ] `apr quantize` (§3.9) — Q4_0/Q8_0 quantization (feature: `format-quantize`)
- [ ] `apr import-sharded` (§3.7) — multi-shard streaming import
- [ ] `apr he-inspect` (§3.12) — HE metadata inspection (feature: `format-homomorphic`)

### Phase 4: Performance (WAPR-APR-CLI-053)

**Effort**: 1 session | **Priority**: P1

- [ ] Profile `fingerprint` and `canary` with `hyperfine` + `heaptrack`
- [ ] Implement Rayon parallel tensor loading (§4.4)
- [ ] Profile `import` and `rosetta convert` — identify tensor conversion bottleneck
- [ ] Benchmark before/after with `hyperfine --warmup 2`
- [ ] Verify performance targets (§4.3) met

**pmat comply check**:
```bash
pmat query "fingerprint" --churn --include-source
hyperfine 'whisper-apr-cli apr fingerprint model.safetensors'
```

### Phase 5: Quality Gate (WAPR-APR-CLI-054)

**Effort**: 1 session | **Priority**: P0

- [ ] All falsification tests (F1–F10) passing
- [ ] Coverage: `pmat query --coverage-gaps` shows <20 uncovered lines in apr_commands.rs
- [ ] Mutation testing: `cargo mutants --file src/cli/apr_commands.rs` — >60% killed
- [ ] Complexity: `pmat analyze complexity` — all functions below thresholds
- [ ] Zero clippy warnings: `cargo clippy --features cli -- -D warnings` (our files)
- [ ] Pre-commit hook passes on all modified files

---

## 7. PMAT Work Tickets

### Roadmap Items (for `docs/roadmaps/roadmap.yaml`)

```yaml
# Feature completeness
- id: WAPR-APR-CLI-002
  title: 'Expose all aprender 0.25.x format modules via apr CLI'
  status: planned
  priority: high
  labels: [enhancement, apr-cli]
  estimated_effort: 3 sessions

# Coverage
- id: WAPR-APR-CLI-003
  title: 'Add 40+ integration tests for apr subcommands'
  status: planned
  priority: high
  labels: [testing, apr-cli]
  estimated_effort: 2 sessions

- id: WAPR-APR-CLI-004
  title: 'Achieve 85%+ line coverage for apr_commands.rs'
  status: planned
  priority: high
  labels: [testing, coverage]

# New commands (Tier A)
- id: WAPR-APR-CLI-010
  title: 'apr golden verify — golden trace verification'
  status: planned
  priority: medium
  labels: [enhancement, apr-cli]

- id: WAPR-APR-CLI-013
  title: 'apr contract verify — layout contract checking'
  status: planned
  priority: medium
  labels: [enhancement, apr-cli]

- id: WAPR-APR-CLI-014
  title: 'apr validate — validated tensor contracts'
  status: planned
  priority: medium
  labels: [enhancement, apr-cli]

- id: WAPR-APR-CLI-015
  title: 'apr family identify/check — model family detection'
  status: planned
  priority: medium
  labels: [enhancement, apr-cli]

- id: WAPR-APR-CLI-017
  title: 'apr compare — statistical weight comparison'
  status: planned
  priority: medium
  labels: [enhancement, apr-cli]

- id: WAPR-APR-CLI-019
  title: 'apr export — APR to GGUF/SafeTensors'
  status: planned
  priority: high
  labels: [enhancement, apr-cli]

- id: WAPR-APR-CLI-020
  title: 'apr f16-audit — F16 NaN/Inf scanning'
  status: planned
  priority: medium
  labels: [enhancement, apr-cli]

# New commands (Tier B — feature-gated)
- id: WAPR-APR-CLI-011
  title: 'apr sign/verify-sig — Ed25519 model signing'
  status: planned
  priority: low
  labels: [enhancement, security, apr-cli]

- id: WAPR-APR-CLI-012
  title: 'apr encrypt/decrypt — AES-256-GCM model encryption'
  status: planned
  priority: low
  labels: [enhancement, security, apr-cli]

- id: WAPR-APR-CLI-016
  title: 'apr import-sharded — multi-shard streaming import'
  status: planned
  priority: medium
  labels: [enhancement, apr-cli]

- id: WAPR-APR-CLI-018
  title: 'apr quantize — Q4_0/Q8_0 quantization'
  status: planned
  priority: medium
  labels: [enhancement, apr-cli]

- id: WAPR-APR-CLI-021
  title: 'apr he-inspect — homomorphic encryption metadata'
  status: planned
  priority: low
  labels: [enhancement, security, apr-cli]

# Performance
- id: WAPR-APR-CLI-030
  title: 'Parallelize fingerprint tensor loading with Rayon'
  status: planned
  priority: high
  labels: [performance, apr-cli]

- id: WAPR-APR-CLI-031
  title: 'Streaming diff for large models'
  status: planned
  priority: medium
  labels: [performance, apr-cli]

- id: WAPR-APR-CLI-032
  title: 'Suppress aprender debug output in --quiet mode'
  status: planned
  priority: high
  labels: [quality, apr-cli]

- id: WAPR-APR-CLI-033
  title: 'Parallelize canary tensor loading with Rayon'
  status: planned
  priority: high
  labels: [performance, apr-cli]

- id: WAPR-APR-CLI-034
  title: 'Parallelize rosetta convert tensor processing'
  status: planned
  priority: medium
  labels: [performance, apr-cli]

# Falsification
- id: WAPR-APR-CLI-040
  title: 'Falsification F1: inspect tensor count cross-validation'
  status: planned
  priority: high
  labels: [testing, falsification]

- id: WAPR-APR-CLI-041
  title: 'Falsification F2: self-diff identity'
  status: planned
  priority: high
  labels: [testing, falsification]

- id: WAPR-APR-CLI-042
  title: 'Falsification F3: lossless F32 round-trip'
  status: planned
  priority: high
  labels: [testing, falsification]

- id: WAPR-APR-CLI-043
  title: 'Falsification F4: canary perturbation detection'
  status: planned
  priority: high
  labels: [testing, falsification]

# Quality gate
- id: WAPR-APR-CLI-054
  title: 'Quality gate: 85% coverage, 60% mutation, zero clippy'
  status: planned
  priority: high
  labels: [quality, apr-cli]
```

---

## 8. PMAT Comply Checklist

Each phase must pass this checklist before merging:

```bash
# 1. Complexity gate (pre-commit hook enforces this)
pmat analyze complexity --max-cyclomatic 30 --max-cognitive 25 \
  src/cli/apr_commands.rs src/cli/apr_args.rs

# 2. Coverage gap analysis
pmat query --coverage-gaps --limit 30 --exclude-tests

# 3. Fault pattern scan (no new unwrap/panic in non-test code)
pmat query "apr" --faults --exclude-tests

# 4. Code clone detection (no copy-paste violations)
pmat query "run_" --duplicates --exclude-tests

# 5. Churn analysis (flag volatile files)
pmat query "apr_commands" --churn

# 6. Build + clippy
cargo build --features cli
cargo clippy --features cli -- -D warnings 2>&1 | grep "apr_commands\|apr_args"

# 7. Tests
cargo test --features cli --lib apr_commands

# 8. SATD check (no TODO/FIXME/HACK in committed code)
pmat query --literal "TODO\|FIXME\|HACK" --files-with-matches

# 9. Work item reference in commit message
# Format: "Refs WAPR-APR-CLI-0XX"
```

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| aprender API breaks between 0.25.x and 0.26.x | Medium | High | Pin to path dep, test before sync |
| Feature-gated modules not available in default build | High | Low | Document required features, test both feature sets |
| Rayon parallelism causes non-deterministic test failures | Low | Medium | Use `par_bridge` with deterministic seed for tests |
| Large model files (>10GB) OOM on fingerprint/canary | Medium | High | Add `--max-tensors N` limit, stream instead of load-all |
| Pre-commit hook rejects functions in new commands | Low | Low | Design functions within complexity bounds from the start |
| HE module has heavy dependencies | Medium | Low | Feature-gate, don't include in default builds |

---

## 10. Success Criteria

This specification is **complete** when:

1. All 25+ subcommands build and pass smoke tests
2. Falsification hypotheses F1–F10 all survive testing (no false positives)
3. `pmat query --coverage-gaps` shows ≤20 uncovered lines in apr_commands.rs
4. `cargo mutants` mutation score ≥60% for apr_commands.rs
5. `pmat analyze complexity` shows all functions below thresholds
6. Performance targets in §4.3 met (verified by `hyperfine`)
7. All PMAT work tickets either completed or have clear blockers documented
8. Zero clippy warnings in `apr_commands.rs` and `apr_args.rs`

---

## Appendix A: Command Summary (Current + Planned)

| # | Command | Status | Tier | PMAT Ticket |
|---|---------|--------|------|-------------|
| 1 | `apr inspect` | Implemented | 1 | — |
| 2 | `apr tensors` | Implemented | 1 | — |
| 3 | `apr hex` | Implemented | 1 | — |
| 4 | `apr tree` | Implemented | 1 | — |
| 5 | `apr flow` | Implemented | 1 | — |
| 6 | `apr lint` | Implemented | 1 | — |
| 7 | `apr diff` | Implemented | 1 | — |
| 8 | `apr import` | Implemented | 2 | — |
| 9 | `apr merge` | Implemented | 2 | — |
| 10 | `apr rosetta inspect` | Implemented | 3 | — |
| 11 | `apr rosetta convert` | Implemented | 3 | — |
| 12 | `apr rosetta verify` | Implemented | 3 | — |
| 13 | `apr rosetta diff` | Implemented | 3 | — |
| 14 | `apr rosetta fingerprint` | Implemented | 3 | — |
| 15 | `apr canary` | Implemented | 4 | — |
| 16 | `apr golden verify` | Planned | A | WAPR-APR-CLI-010 |
| 17 | `apr sign` | Planned | B | WAPR-APR-CLI-011 |
| 18 | `apr verify-sig` | Planned | B | WAPR-APR-CLI-011 |
| 19 | `apr encrypt` | Planned | B | WAPR-APR-CLI-012 |
| 20 | `apr decrypt` | Planned | B | WAPR-APR-CLI-012 |
| 21 | `apr contract verify` | Planned | A | WAPR-APR-CLI-013 |
| 22 | `apr validate` | Planned | A | WAPR-APR-CLI-014 |
| 23 | `apr family identify` | Planned | A | WAPR-APR-CLI-015 |
| 24 | `apr family check` | Planned | A | WAPR-APR-CLI-015 |
| 25 | `apr import-sharded` | Planned | B | WAPR-APR-CLI-016 |
| 26 | `apr compare` | Planned | A | WAPR-APR-CLI-017 |
| 27 | `apr quantize` | Planned | B | WAPR-APR-CLI-018 |
| 28 | `apr export` | Planned | A | WAPR-APR-CLI-019 |
| 29 | `apr f16-audit` | Planned | A | WAPR-APR-CLI-020 |
| 30 | `apr he-inspect` | Planned | B | WAPR-APR-CLI-021 |

## Appendix B: pmat query Recipes for Ongoing Maintenance

```bash
# Weekly quality audit
pmat query "apr" --churn --duplicates --entropy --faults -G --exclude-tests

# Pre-release coverage check
pmat query --coverage-gaps --limit 50 --exclude-tests

# Find functions that need refactoring (high complexity + high churn)
pmat query "apr_commands" --churn --max-complexity 20

# Find code added by commit intent
pmat query "apr cli integration" -G --limit 10

# Verify no regressions after aprender upgrade
pmat query "format_model_error" --include-source --coverage --limit 1
```
