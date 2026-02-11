# PMAT Integration

Whisper.apr uses [PMAT](https://github.com/paiml/pmat) for quality analysis, compliance checking, and code search.

## Compliance

PMAT compliance validates adherence to Sovereign AI Stack quality standards:

```bash
# Check compliance status
pmat comply check

# Migrate to latest PMAT version
pmat comply migrate
```

### Compliance Gates

| Gate | Requirement |
|------|-------------|
| Version Currency | Within 5 versions of latest PMAT |
| Coverage Patterns (CB-125) | ≤10 coverage exclusion patterns |
| Coverage Patterns (CB-126) | PROPTEST_CASES/QUICKCHECK_TESTS env vars in test targets |
| Coverage Patterns (CB-127) | Use `cargo llvm-cov test` (not nextest) |
| File Health | No CRITICAL files (>6000 lines) |
| Agent Context (CB-130) | CLAUDE.md references `pmat query` |
| Ignore Reasons (CB-123) | All `#[ignore]` must have a reason string |

## Code Search with pmat query

Use `pmat query` instead of grep for code discovery. Returns quality-annotated, semantically ranked results with TDG grades, complexity, and fault patterns.

```bash
# Find functions by intent
pmat query "mel spectrogram" --limit 10

# Find high-quality implementations
pmat query "encoder forward" --min-grade A

# Find with fault annotations
pmat query "decoder" --faults

# Find coverage gaps ranked by uncovered lines
pmat query --coverage-gaps --limit 20 --exclude-tests

# Coverage-enriched search
pmat query "validation" --coverage --limit 10

# Include source code in results
pmat query "tokenizer" --include-source --limit 5
```

## Quality Analysis

```bash
# Complexity analysis
pmat analyze complexity src/

# File health check
pmat analyze file-health

# Full quality report
pmat analyze quality
```

## Configuration

PMAT configuration lives in two files:

### pmat.toml

Project-level analysis settings:

```toml
[analysis]
include_patterns = ["**/*.rs"]
exclude_patterns = [
    "**/target/**",
    "**/*_generated.rs",
    "**/golden_traces/**",
]

[quality]
max_complexity = 40
max_cognitive_complexity = 65
min_coverage = 89.0
```

### .pmat-gates.toml

Quality gate thresholds:

```toml
[file-health]
max_lines_critical = 6000
max_lines_warning = 3000
exclude = ["tests.rs", "benchmark.rs", "*_generated.rs"]

[quality-gates]
exclude = ["**/*_generated.rs", "**/tests.rs", "**/test_*.rs"]
```

## Pre-commit Hook

The pre-commit hook enforces complexity thresholds on all staged `.rs` files:

- Maximum cyclomatic complexity: 30
- Maximum cognitive complexity: 25
- No SATD comments (TODO/FIXME/HACK)
- Commit messages must reference a work item (e.g., `Refs WAPR-001`)
