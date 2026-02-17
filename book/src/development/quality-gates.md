# Quality Gates

Whisper.apr implements tiered quality gates for fast feedback loops.

## Tier Overview

| Tier | Trigger | Duration | Purpose |
|------|---------|----------|---------|
| Tier 1 | On save | <1s | Immediate feedback |
| Tier 2 | Pre-commit | <5s | Validation before commit |
| Tier 3 | Pre-push | 1-5min | Full validation |
| Tier 4 | CI/CD | 5-60min | Comprehensive analysis |

## Tier 1: On-Save (<1s)

Fast feedback for immediate issues:

```bash
cargo fmt --check && cargo clippy -- -D warnings && cargo check
```

**Validates:**
- Code formatting
- Clippy lint compliance
- Compilation

## Tier 2: Pre-Commit (<5s)

Quick validation before committing:

```bash
cargo test --lib && cargo clippy -- -D warnings
```

**Validates:**
- All unit tests pass
- Zero clippy warnings
- Code compiles in test mode

## Tier 3: Pre-Push (1-5min)

Full validation before pushing:

```bash
make coverage
```

**Validates:**
- All tests (unit + property + integration)
- Coverage >= 95%
- Documentation builds

### Coverage Requirements

```bash
make coverage
```

Uses `cargo llvm-cov test --lib` for coverage measurement:

```bash
COV_THRESHOLD ?= 95

coverage:
    env RUSTC_WRAPPER= PROPTEST_CASES=2 QUICKCHECK_TESTS=2 \
        cargo llvm-cov test --lib \
        --ignore-filename-regex '(tests\.rs|test_.*\.rs|_generated\.rs|golden_traces|book|demos|snippets)' \
        -- --test-threads=$(nproc)
    cargo llvm-cov report --summary-only | tee target/coverage/summary.txt
```

## Tier 4: CI/CD (5-60min)

Comprehensive analysis in CI pipeline:

**Validates:**
- Everything from Tier 3
- Mutation testing (target: >= 85%)
- Security audit
- PMAT quality analysis

## PMAT Quality Gates

```bash
# Check compliance
pmat comply check

# Run quality gates
pmat quality-gate

# Continuous improvement scan
pmat kaizen
```

## Lint Configuration

Whisper.apr uses strict clippy configuration in `Cargo.toml`:

```toml
[lints.clippy]
correctness = { level = "deny", priority = -1 }
suspicious = { level = "warn", priority = -1 }
perf = { level = "warn", priority = -1 }
pedantic = { level = "warn", priority = -1 }
unwrap_used = "deny"      # Zero tolerance
expect_used = "warn"
panic = "warn"
```

## Quality Metrics (v0.2.4)

| Metric | Value |
|--------|-------|
| **TDG Score** | 99.5/100 (A+) |
| **Unit tests** | 2,885 (147 ignored by default) |
| **Line coverage** | 96%+ |
| **Property tests** | 19 |
| **Clippy warnings** | 0 (strict mode) |
| **pmat compliance** | COMPLIANT |
| **Quality gate** | PASSED (0 violations) |
| **GitHub issues** | 0 open |

### Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| trueno | 0.14.6 | SIMD-accelerated tensor operations |
| aprender | 0.25.9 | .apr model format + GGUF parsing |
| realizar | 0.6.13 | Inference primitives |

## Best Practices

1. **Run tier1 on every save** - Use editor integration
2. **Run tier2 before every commit** - Git hooks enforce this
3. **Run tier3 before every push** - Catches integration issues
4. **Never skip CI** - Full validation catches edge cases
5. **Use pmat tools** - `pmat comply check`, `pmat quality-gate`, `pmat kaizen`
