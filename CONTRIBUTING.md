# Contributing to whisper.apr

Thank you for your interest in contributing to whisper.apr!

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Style](#code-style)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Commit Messages](#commit-messages)
- [Quality Gates](#quality-gates)
- [License](#license)

## Development Setup

### Prerequisites

- Rust 1.75+ with the WASM target
- (Optional) Nix for reproducible development environment

### Option 1: Using Nix (Recommended)

```bash
# Enter development shell with all dependencies
nix develop

# Or use direnv for automatic activation
echo "use flake" > .envrc && direnv allow
```

### Option 2: Manual Setup

1. Install Rust 1.75+ and the WASM target:
   ```bash
   rustup target add wasm32-unknown-unknown
   ```

2. Clone and build:
   ```bash
   git clone https://github.com/paiml/whisper.apr.git
   cd whisper.apr
   cargo build --release
   ```

3. Run tests:
   ```bash
   cargo test
   ```

## Project Structure

```
src/
├── lib.rs          # Public API
├── audio/          # Audio preprocessing (mel spectrogram)
├── model/          # Transformer architecture (encoder/decoder)
├── inference/      # Decoding strategies (greedy/beam)
└── wasm/           # WASM bindings

tests/              # Integration tests
benches/            # Performance benchmarks
demos/              # Demo applications and test audio
```

## Code Style

- Follow Rust standard formatting with `cargo fmt`
- All code must pass `cargo clippy -- -D warnings`
- Maximum cyclomatic complexity: 10 per function
- Zero SATD comments (no TODO/FIXME/HACK)
- 95% minimum test coverage
- Document all public APIs with `///` doc comments

## Testing Requirements

### Test Types Required

1. **Unit Tests**: Test individual functions in isolation
2. **Property Tests**: Use proptest for invariant checking
3. **Integration Tests**: End-to-end transcription tests
4. **Benchmark Tests**: Performance regression tests

### Running Tests

```bash
# All tests
cargo test

# Unit tests only
cargo test --lib

# Integration tests
cargo test --test '*'

# Coverage report
make coverage

# Benchmarks
cargo bench
```

## Pull Request Process

1. **Create an issue** describing the change (for non-trivial changes)
2. **Fork** the repository and create a branch from `main`
3. **Write tests first** (TDD approach preferred)
4. **Implement** the feature/fix
5. **Run quality gates**:
   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo test
   pmat quality-gate
   ```
6. **Update documentation** as needed
7. **Create pull request** with clear description

### PR Checklist

- [ ] Tests pass: `cargo test`
- [ ] Clippy passes: `cargo clippy -- -D warnings`
- [ ] Formatted: `cargo fmt --check`
- [ ] Quality gate passes: `pmat quality-gate`
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if user-facing)

## Commit Messages

Use conventional commits format:

| Prefix | Description |
|--------|-------------|
| `feat:` | New features |
| `fix:` | Bug fixes |
| `docs:` | Documentation changes |
| `refactor:` | Code refactoring |
| `test:` | Test additions/changes |
| `perf:` | Performance improvements |
| `chore:` | Build/tooling changes |

Example:
```
feat(decoder): add beam search with configurable width

- Implement BeamSearchDecoder with width parameter
- Add hypothesis pruning for memory efficiency
- Include tests for beam width 1-10

Refs WAPR-XXX
```

## Quality Gates

All contributions must pass:

| Gate | Command | Threshold |
|------|---------|-----------|
| Clippy | `cargo clippy -- -D warnings` | 0 warnings |
| Coverage | `make coverage` | ≥95% |
| TDG Grade | `pmat tdg .` | ≥A (90+) |
| Complexity | `pmat quality-gate` | ≤10 per fn |

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
