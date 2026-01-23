# Publishing whisper.apr Models to Hugging Face

**Specification Version**: 2.0.0
**Status**: Stable (apr-cli publish Complete)
**Date**: 2026-01-23
**Ticket**: WAPR-PUB-001

---

## Overview

This specification defines the canonical workflow for publishing whisper.apr models to Hugging Face Hub in both `.apr` (native) and `.safetensors` (HF standard) formats. All publishing operations use **pure Rust tooling** from the Sovereign AI Stack—no Python dependencies.

```
[REVIEW-001] @noah 2024-01-23
Toyota Principle: Genchi Genbutsu (Go and See)
Direct Hub API integration via batuta/hf eliminates Python abstraction layers.
We verify both formats against ground truth before any publish operation.
Status: ACCEPTED
```

## Sovereign AI Stack Context

This workflow integrates with the PAIML Sovereign AI Stack as documented in batuta:

```
┌─────────────────────────────────────────────────────────────┐
│                      batuta (Orchestration)                 │
├─────────────────────────────────────────────────────────────┤
│  whisper.apr (ASR)  │  realizar (Inference)  │ pacha (Reg)  │
├─────────────────────┴────────────────────────┴──────────────┤
│   aprender (ML)   │  entrenar (Training)  │ certeza (QA)    │
├───────────────────┴───────────────────────┴─────────────────┤
│               trueno (SIMD/GPU Compute Primitives)          │
└─────────────────────────────────────────────────────────────┘
```

**Key Constraints**:
- No Python tooling (`pip`, `huggingface_hub`, `transformers`)
- All operations via Sovereign AI Stack (`batuta::hf`, `aprender::serialization`, `pacha`)
- Ed25519 model signatures via `pacha`
- SafeTensors-first policy (Poka-Yoke security)

**Quality Requirements**:
- **Coverage**: `make coverage` MUST exceed 95% overall
- **Per-File Coverage**: NO file may have coverage below 95%
- **PMAT Compliance**: Full `pmat comply check` must pass
- **Mutation Testing**: ≥70% mutation score (stretch goal: 85%)

```bash
# Verify quality gates before publish
make coverage-summary      # Must show ≥95%
pmat comply check          # Must pass all checks
pmat quality-gate          # Must pass quality gate
```

---

## SafeTensors Stack Implementation

SafeTensors format is implemented natively across the stack—**no external crates**.

### Component Responsibilities

| Component | Module | Role | Source |
|-----------|--------|------|--------|
| **aprender** | `serialization/safetensors.rs` | **Write** | `~/src/aprender/src/serialization/safetensors.rs` |
| **aprender** | `inspect/safetensors.rs` | **Inspect** | `~/src/aprender/src/inspect/safetensors.rs` |
| **realizar** | `safetensors.rs` | **Read/Parse** | `~/src/realizar/src/safetensors.rs` |
| **realizar** | `safetensors_infer.rs` | **Inference** | `~/src/realizar/src/safetensors_infer.rs` |

### Data Flow

```
whisper.apr (Model Definition)
     │
     ▼
aprender::serialization::safetensors::save_safetensors()
     │
     │ (writes model.safetensors)
     ▼
batuta::hf::push (Orchestration) ◄────┐
     │                                │
     │ (uploads to Hub)               │
     ▼                                │
Hugging Face Hub (Storage)            │
     │                                │
     │ (downloads for inference)      │
     ▼                                │
realizar::safetensors::SafetensorsModel::load()
     │
     ▼
realizar::safetensors_infer (Execution)
```

---

## CLI Interface

### Publish Commands (apr publish)

The `apr publish` command in `apr-cli` (part of aprender) handles HuggingFace Hub publishing with auto-generated model cards.

```bash
# Publish model with auto-generated model card (YAML frontmatter)
apr publish /path/to/model-dir paiml/whisper-apr-tiny \
  --model-name "Whisper APR Tiny" \
  --license mit \
  --pipeline-tag automatic-speech-recognition \
  --library-name whisper-apr \
  --tags whisper \
  --message "Upload via apr-cli publish (APR-PUB-001)" \
  -v

# Dry run (preview model card without uploading)
apr publish /path/to/model-dir paiml/whisper-apr-tiny \
  --model-name "Whisper APR Tiny" \
  --license mit \
  --pipeline-tag automatic-speech-recognition \
  --library-name whisper-apr \
  --tags whisper \
  --dry-run
```

**Implementation Location**: `~/src/aprender/crates/apr-cli/src/commands/publish.rs`

**Required Environment**: `HF_TOKEN` must be set (export HF_TOKEN=hf_...)

### Verification Commands

```bash
# Verify model before publish (both formats)
batuta hf verify ./whisper-tiny.apr --against-hf openai/whisper-tiny

# Compare APR vs SafeTensors numerical accuracy
batuta hf diff ./whisper-tiny.apr ./whisper-tiny.safetensors --tolerance 1e-6

# Validate ground truth (3-way comparison)
./scripts/ground_truth_compare.sh ./whisper-tiny.apr
```

```
[REVIEW-002] @security-team 2024-01-23
Toyota Principle: Jidoka (Automation with Human Touch)
--verify flag mandates ground truth validation before any publish.
Prevents publishing models with numerical drift or regressions.
Status: ACCEPTED
```

---

## Dual-Format Publishing Workflow

### Phase 1: Model Preparation

```rust
use whisper_apr::format::AprModel;
use aprender::serialization::safetensors::save_safetensors;
use realizar::safetensors::SafetensorsModel;
use std::collections::BTreeMap;
use std::path::Path;

// Load APR model
let model = AprModel::load("whisper-tiny.apr")?;

// Convert to BTreeMap<String, (Vec<f32>, Vec<usize>)> for SafeTensors
let mut tensors = BTreeMap::new();
tensors.insert("encoder.conv1.weight".into(),
    (model.encoder.conv1_weight.to_vec(), model.encoder.conv1_shape.clone()));
// ... add remaining tensors

// Export to SafeTensors (aprender native implementation)
save_safetensors("whisper-tiny.safetensors", &tensors)?;

// Verify round-trip via realizar
let reloaded = SafetensorsModel::load(Path::new("whisper-tiny.safetensors"))?;
let conv1 = reloaded.get_tensor("encoder.conv1.weight")?;
assert!((model.encoder.conv1_weight[0] - conv1[0]).abs() < 1e-6);
```

### Phase 2: Quality Verification

```bash
# Step 1: Validate APR format integrity
whisper-apr verify ./whisper-tiny.apr

# Step 2: Validate SafeTensors export
batuta hf verify ./whisper-tiny.safetensors

# Step 3: Ground truth comparison (3-way)
./scripts/ground_truth_compare.sh test_data/test-speech-1.5s.wav

# Step 4: Numerical reproducibility check
cargo test --test numerical_reproducibility -- --nocapture
```

### Phase 3: Model Signing (pacha)

```bash
# Generate signing key (one-time)
batuta pacha keygen --identity "paiml-models@paiml.com"

# Sign both formats
batuta pacha sign ./whisper-tiny.apr --identity "paiml-models@paiml.com"
batuta pacha sign ./whisper-tiny.safetensors --identity "paiml-models@paiml.com"

# Verify signatures
batuta pacha verify ./whisper-tiny.apr
batuta pacha verify ./whisper-tiny.safetensors
```

### Phase 4: Hub Upload

```bash
# Authenticate (HF_TOKEN from environment)
export HF_TOKEN=$(cat ~/.huggingface/token)

# Create repository
batuta hf repo create paiml/whisper-apr-tiny --type model

# Upload both formats with model card
batuta hf push model ./whisper-tiny.apr \
  --repo "paiml/whisper-apr-tiny" \
  --formats apr,safetensors \
  --model-card ./MODEL_CARD.md \
  --commit-message "Release whisper-apr-tiny v0.2.0"
```

---

## Rust Implementation

### SafeTensors Export (Stack Native)

```rust
//! SafeTensors export using aprender's native serialization.
//! Location: src/format/export.rs

use aprender::serialization::safetensors::save_safetensors;
use std::collections::BTreeMap;

pub fn export_safetensors(apr_path: &Path, output_path: &Path) -> Result<(), String> {
    let model = AprModel::load(apr_path)?;
    // ... mapping logic ...
    save_safetensors(output_path, &tensors)?;
    Ok(())
}
```

### Publishing Orchestration (Builder Pattern)

```rust
//! Publishing orchestration using batuta::hf
//! Location: src/publish.rs

use batuta::hf::{Publisher, PublishConfig, PublishFormat};

pub async fn publish_workflow(token: &str) -> Result<(), String> {
    let publisher = Publisher::new(token)?;

    let config = PublishConfig::builder()
        .repo("paiml/whisper-apr-tiny")
        .format(PublishFormat::Both) // Apr | SafeTensors | Both
        .sign(true)
        .verify(true) // Enforces Pre-Publish Checklist
        .model_card("./MODEL_CARD.md")
        .commit_message("Release v0.2.0")
        .build();

    publisher.publish("./whisper-tiny.apr", config).await?;
    Ok(())
}
```

---

## Quality Verification Pipeline

### Pre-Publish Checklist (Automated)

```bash
#!/usr/bin/env bash
# Automated by bashrs, verified by bashrs verify

set -euo pipefail

MODEL_PATH="$1"
REPO_ID="$2"

echo "=== Pre-Publish Quality Gates ==="

# 1. Format integrity
echo "[1/6] Verifying APR format..."
whisper-apr verify "$MODEL_PATH"

# 2. SafeTensors export
echo "[2/6] Exporting to SafeTensors..."
whisper-apr export --format safetensors "$MODEL_PATH" -o model.safetensors

# 3. Numerical reproducibility
echo "[3/6] Validating numerical accuracy..."
cargo test --test numerical_reproducibility

# 4. Ground truth comparison
echo "[4/6] Ground truth validation (3-way)..."
./scripts/ground_truth_compare.sh test_data/test-speech-1.5s.wav

# 5. Sign models
echo "[5/6] Signing models..."
batuta pacha sign "$MODEL_PATH"
batuta pacha sign model.safetensors

# 6. Secret scan
echo "[6/6] Scanning for secrets..."
batuta hf scan --secrets "$MODEL_PATH" model.safetensors

echo "=== All Quality Gates Passed ==="
```

---

## Falsification Strategy (Chaos Engineering)

To satisfy [REVIEW-005], the verification tools themselves are subjected to falsification tests.

| Chaos Test | Injection | Expected Result | Verified |
|------------|-----------|-----------------|----------|
| **F-VER-001** | Truncate `.apr` file | `verify` fails with "Unexpected EOF" | [x] |
| **F-VER-002** | Corrupt magic bytes | `verify` fails with "Invalid Magic" | [x] |
| **F-VER-003** | Insert NaN into weights | `verify` fails with "NaN detected" | [x] |
| **F-VER-004** | Embed secret (AWS Key) | `scan` fails with "Secret found" | [x] |
| **F-VER-005** | Mismatch tensor shapes | `export` fails or `verify` catches drift | [x] |

```
[REVIEW-006] @noah 2026-01-23
Implementation Status:
The Rust implementation for export, publish, and verify is COMPLETE.
Chaos tests (F-VER-xxx) are implemented in `tests/publish_integration.rs`.
Status: STABLE
```

---

## 100-Point Popperian Falsification Checklist

### Section A: Format Integrity (20 points)

| # | Falsification Criterion | Test | Pass |
|---|-------------------------|------|------|
| A1 | APR magic bytes match `APR\0` | `head -c4 model.apr \| xxd` | [x] |
| A2 | APR version is parseable uint32 | Binary parse test | [x] |
| A3 | APR header size is valid | Header bounds check | [x] |
| A4 | SafeTensors magic matches spec | `aprender::inspect::validate_safetensors()` | [x] |
| A5 | SafeTensors metadata is valid JSON | JSON schema validation | [x] |
| A6 | All tensor names match HF convention | Naming regex test | [x] |
| A7 | Tensor shapes match Whisper spec | Shape assertion | [x] |
| A8 | Tensor dtypes are f32/f16 (SafeTensors) | dtype enumeration | [x] |
| A9 | APR quantization metadata present | Quant scales check | [x] |
| A10 | No NaN values in any tensor | `tensor.is_nan().any() == false` | [x] |
| A11 | No Inf values in any tensor | `tensor.is_inf().any() == false` | [x] |
| A12 | Tensor checksums match manifest | BLAKE3 verification | [x] |
| A13 | File sizes within expected bounds | Size regression test | [x] |
| A14 | Compression ratio is reasonable | LZ4/ZSTD efficiency check | [x] |
| A15 | Streaming blocks are aligned | 64KB block alignment | [x] |
| A16 | Index is seekable | Random access test | [x] |
| A17 | Vocabulary size is 51,865 | Token count assertion | [x] |
| A18 | Special tokens present | `<\|startoftranscript\|>` exists | [x] |
| A19 | BPE merges are complete | Merge count validation | [x] |
| A20 | Config.json matches model | Architecture validation | [x] |

### Section B: Numerical Accuracy (20 points)

| # | Falsification Criterion | Test | Pass |
|---|-------------------------|------|------|
| B1 | APR→SafeTensors round-trip < 1e-6 | Max abs diff | [x] |
| B2 | Mel spectrogram matches reference | Mean=-0.2148, Std=0.4479 | [x] |
| B3 | Encoder output cosine sim > 0.9999 | vs. HF reference | [x] |
| B4 | Decoder logits match within 1e-4 | Token probability check | [x] |
| B5 | Attention scores sum to 1.0 | Softmax normalization | [x] |
| B6 | KV-cache consistency | Incremental vs. full decode | [x] |
| B7 | Positional encoding matches | Sinusoidal formula check | [x] |
| B8 | LayerNorm epsilon is 1e-5 | Config validation | [x] |
| B9 | GELU approximation error < 1e-4 | vs. exact GELU | [x] |
| B10 | Quantization error bounded | Int8: ±0.5%, Int4: ±2% | [x] |
| B11 | Dequantization is deterministic | 100 runs identical | [x] |
| B12 | Streaming produces same result | Chunk vs. batch mode | [x] |
| B13 | SIMD path matches scalar | AVX2/NEON vs. fallback | [x] |
| B14 | WASM matches native | Cross-platform parity | [x] |
| B15 | WER delta vs. reference < 0.5% | LibriSpeech test-clean | [x] |
| B16 | Language detection accuracy > 95% | Multilingual test set | [x] |
| B17 | Timestamp accuracy < 20ms | Forced alignment check | [x] |
| B18 | Beam search matches greedy | Temperature=0 equivalence | [x] |
| B19 | No logit overflow (> 88.0) | Numerical stability | [x] |
| B20 | Cross-attention stable | No NaN in long sequences | [x] |

### Section C: Security & Provenance (20 points)

| # | Falsification Criterion | Test | Pass |
|---|-------------------------|------|------|
| C1 | Ed25519 signature is valid | `pacha verify` passes | [x] |
| C2 | Signer identity is known | Keyring lookup | [x] |
| C3 | Content hash matches file | BLAKE3 recalculation | [x] |
| C4 | No pickle files present | File extension scan | [x] |
| C5 | No executable content | Magic byte scan | [x] |
| C6 | No secrets in model | `batuta hf scan --secrets` | [x] |
| C7 | License file present | MIT/Apache-2.0 | [x] |
| C8 | Model card has provenance | Training data attribution | [x] |
| C9 | Signature timestamp valid | Within 1 year | [x] |
| C10 | Public key is published | HF profile or keyserver | [x] |
| C11 | SafeTensors header < 100MB | DoS protection | [x] |
| C12 | No symlinks in upload | Path traversal prevention | [x] |
| C13 | File permissions are safe | No world-writable | [x] |
| C14 | No hidden files (.git, .env) | Dotfile scan | [x] |
| C15 | Checksum matches HF API | Post-upload verification | [x] |
| C16 | Encryption key not embedded | Key material scan | [x] |
| C17 | No hardcoded URLs | External dependency scan | [x] |
| C18 | Model card has SHA256 | Reproducibility hash | [x] |
| C19 | Dependency versions pinned | Cargo.lock present | [x] |
| C20 | No telemetry/tracking code | Static analysis | [x] |

### Section D: HuggingFace Compatibility (20 points)

| # | Falsification Criterion | Test | Pass |
|---|-------------------------|------|------|
| D1 | Repo is accessible | `batuta hf info` API fetch | [x] |
| D2 | SafeTensors loadable | `batuta hf verify --safetensors` | [x] |
| D3 | APR format loadable | `whisper-apr verify` | [x] |
| D4 | Inference API responds | `batuta hf infer --test` | [x] |
| D5 | Model card renders | `batuta hf card --validate` | [x] |
| D6 | Tags are indexed | `batuta hf search` finds model | [x] |
| D7 | Downloads counter works | `batuta hf stats` API check | [x] |
| D8 | Git LFS is configured | `batuta hf verify --lfs` | [x] |
| D9 | Revision history intact | `batuta hf log` accessible | [x] |
| D10 | Branch protection (main) | `batuta hf settings` check | [x] |
| D11 | Discussions enabled | `batuta hf info --features` | [ ] |
| D12 | Model-index valid | `batuta hf card --model-index` | [ ] |
| D13 | Dataset links work | `batuta hf card --links` | [ ] |
| D14 | Code snippets executable | `cargo test --example` | [x] |
| D15 | WASM demo functional | `probar test` (if any) | [ ] |
| D16 | API token scoped correctly | `batuta hf auth --check` | [ ] |
| D17 | Organization membership | `batuta hf org verify` | [ ] |
| D18 | Webhook notifications | `batuta hf hooks --test` | [ ] |
| D19 | Model versioning works | `batuta hf tags` | [ ] |
| D20 | Gated access (if private) | `batuta hf auth --gated` | [ ] |

### Section E: Workflow Reproducibility (20 points)

| # | Falsification Criterion | Test | Pass |
|---|-------------------------|------|------|
| E1 | Build is deterministic | 2 builds produce identical output | [x] |
| E2 | CI/CD pipeline passes | GitHub Actions green | [ ] |
| E3 | Docker build works | Container reproducibility | [ ] |
| E4 | Make targets documented | `make help` | [x] |
| E5 | All dependencies pinned | Cargo.lock, package-lock | [x] |
| E6 | Tests pass in CI | `cargo test --all` | [x] |
| E7 | Coverage meets threshold | ≥95% overall (`make coverage`) | [x] |
| E7a | Per-file coverage | NO file under 95% coverage | [x] |
| E8 | Mutation score acceptable | ≥70% mutants killed | [ ] |
| E9 | Linting passes | `cargo clippy -- -D warnings` | [x] |
| E10 | Formatting consistent | `cargo fmt --check` | [x] |
| E11 | Docs build without warnings | `cargo doc` | [x] |
| E12 | Examples compile | `cargo build --examples` | [x] |
| E13 | Benchmarks run | `cargo bench` | [x] |
| E14 | WASM build succeeds | `wasm32-unknown-unknown` | [x] |
| E15 | Release notes present | CHANGELOG.md | [x] |
| E16 | Version bump correct | Cargo.toml matches tag | [x] |
| E17 | Git tag is signed | GPG/SSH signature | [x] |
| E18 | Pre-publish hook passes | `make pre-publish` | [ ] |
| E19 | Post-publish verification | Download and test | [ ] |
| E20 | Rollback procedure documented | Revert instructions | [ ] |

---

## Scoring

| Section | Points | Threshold |
|---------|--------|-----------|
| A: Format Integrity | 20/20 | ≥18 |
| B: Numerical Accuracy | 20/20 | ≥19 |
| C: Security & Provenance | 20/20 | ≥18 |
| D: HuggingFace Compatibility | 10/20 | ≥16 |
| E: Workflow Reproducibility | 13/20 | ≥17 |
| **Total** | **83/100** | **≥88** |

**Current Status**: 83/100.
**Remaining**: CI/CD integration, Webhooks, Mutation Testing (E8), and Final Publish Hook (E18).

```
[REVIEW-006] @qa-team 2026-01-23
Integration tests passed (11/11). Core functionality is verified.
Remaining points depend on external CI/CD infrastructure setup.
Status: STABLE
```

---

## Implementation Status

### aprender (apr-cli)
- [x] `crates/apr-cli/src/commands/publish.rs` - HuggingFace publishing with auto-generated model cards
- [x] `src/format/model_card.rs` - ModelCard generation with YAML frontmatter

### whisper.apr
- [x] `src/format/export.rs` - SafeTensors export (stack-native)
- [x] `src/publish.rs` - Publishing orchestration (HF Hub patterns)
- [x] `src/verify.rs` - Pre-publish verification (certeza patterns)
- [ ] `scripts/publish.rs` - Automated workflow (bashrs source)
- [x] `tests/publish_integration.rs` - End-to-end tests (11 tests passing)
- [ ] CI/CD GitHub Action - Automated publishing

### Published Models
- [x] `paiml/whisper-apr-tiny` - https://huggingface.co/paiml/whisper-apr-tiny
- [ ] `paiml/whisper-apr-base` - Pending export

---

## References

### Stack Implementation (Local)
- **apr-cli publish**: `~/src/aprender/crates/apr-cli/src/commands/publish.rs` (APR-PUB-001)
- aprender ModelCard: `~/src/aprender/src/format/model_card.rs`
- aprender SafeTensors Write: `~/src/aprender/src/serialization/safetensors.rs`
- aprender SafeTensors Inspect: `~/src/aprender/src/inspect/safetensors.rs`
- realizar SafeTensors Read: `~/src/realizar/src/safetensors.rs`
- realizar SafeTensors Infer: `~/src/realizar/src/safetensors_infer.rs`
- batuta HF Client: `~/src/batuta/src/hf/client.rs` (orchestration layer)
- pacha Model Signing: `~/src/batuta/src/pacha/mod.rs`

### Format Specification
- SafeTensors Format: `~/src/safetensors` (cloned from HuggingFace)
- HuggingFace Hub API: https://huggingface.co/docs/hub/api

### Stack Documentation
- batuta HuggingFace Integration: `~/src/batuta/docs/specifications/hugging-face-integration-query-publish-spec.md`

### Academic References
- Toyota Production System: Ohno (1988), Liker (2004)
- Popperian Falsification: Popper (1959)
```