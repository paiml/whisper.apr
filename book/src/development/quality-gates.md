# Quality Gates

Whisper.apr implements tiered quality gates following the **bashrs** methodology for fast feedback loops.

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
make tier1
# Or manually:
cargo fmt --check && cargo clippy -- -W all && cargo check
```

**Validates:**
- Code formatting
- Common lint issues
- Compilation

## Tier 2: Pre-Commit (<5s)

Quick validation before committing:

```bash
make tier2
# Or manually:
cargo test --lib && cargo clippy -- -D warnings
```

**Validates:**
- All unit tests pass
- Zero clippy warnings
- Code compiles in test mode

## Tier 3: Pre-Push (1-5min)

Full validation before pushing:

```bash
make tier3
```

**Validates:**
- All tests (unit + property + integration)
- Coverage ≥95%
- Documentation builds

### Coverage Requirements

```bash
make coverage
```

Current targets:
- **Line coverage**: ≥95% (achieved: 95.19%)
- **Function coverage**: ≥95%
- **Branch coverage**: tracked

## Tier 4: CI/CD (5-60min)

Comprehensive analysis in CI pipeline:

```bash
make tier4
```

**Validates:**
- Everything from Tier 3
- Mutation testing (target: ≥85%)
- Security audit
- PMAT quality analysis

### Mutation Testing

```bash
make mutants
```

Mutation testing validates test quality by introducing bugs and checking if tests catch them.

## Makefile Targets

```makefile
# Tier 1: Fast feedback
tier1:
    cargo fmt --check
    cargo clippy -- -W all
    cargo check

# Tier 2: Pre-commit
tier2:
    cargo test --lib
    cargo clippy -- -D warnings

# Tier 3: Pre-push
tier3:
    cargo test --all
    make coverage

# Tier 4: CI/CD
tier4:
    make tier3
    make mutants
```

## Lint Configuration

Whisper.apr uses strict clippy configuration in `Cargo.toml`:

```toml
[lints.clippy]
all = { level = "warn", priority = -1 }
pedantic = { level = "warn", priority = -1 }
nursery = { level = "warn", priority = -1 }
unwrap_used = "deny"      # Prevent panics
expect_used = "warn"      # Discourage panics
panic = "warn"            # Prevent explicit panics

# DSP-specific allows
cast_precision_loss = "allow"
cast_possible_truncation = "allow"
```

## CI Integration

Quality gates are enforced in `.github/workflows/ci.yml`:

```yaml
jobs:
  tier2:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: make tier2

  tier3:
    runs-on: ubuntu-latest
    needs: tier2
    steps:
      - uses: actions/checkout@v4
      - run: make tier3

  coverage:
    runs-on: ubuntu-latest
    needs: tier2
    steps:
      - uses: actions/checkout@v4
      - run: make coverage
      - uses: codecov/codecov-action@v4
```

## Quality Metrics

Current project status:
- **Test count**: 841 tests
- **Line coverage**: 95.19%
- **Property tests**: 19 tests
- **Zero clippy warnings** (in strict mode)
- **Zero unsafe code**

## Best Practices

1. **Run tier1 on every save** - Use editor integration
2. **Run tier2 before every commit** - Git hooks recommended
3. **Run tier3 before every push** - Catches integration issues
4. **Never skip CI** - Full validation catches edge cases
