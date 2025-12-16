# Contributing to whisper.apr

Thank you for your interest in contributing to whisper.apr!

## Development Setup

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

## Code Style

- Follow Rust standard formatting with `cargo fmt`
- All code must pass `cargo clippy -- -D warnings`
- Maximum cyclomatic complexity: 10 per function
- Zero SATD comments (no TODO/FIXME/HACK)
- 95% minimum test coverage

## Pull Request Process

1. Ensure all tests pass: `cargo test`
2. Run quality checks: `pmat quality-gate`
3. Update documentation as needed
4. Create a pull request with a clear description

## Commit Messages

Use conventional commits format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `refactor:` for code refactoring
- `test:` for test additions/changes
- `perf:` for performance improvements

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
