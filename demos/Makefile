# Whisper.apr Demo Applications Makefile
# EXTREME TDD + Probar-First Development
#
# Quality Gates:
# - GUI Coverage: ≥95%
# - Button Coverage: 100%
# - State Coverage: 100%
# - Error Path Coverage: ≥95%

.PHONY: all build test coverage tier1 tier2 tier3 clean help \
	playbook-validate playbook-mutate playbook-diagrams playbook-test playbook-quick

# Default target
all: build test

# ============================================================================
# Build Targets
# ============================================================================

## Build demo for WASM
build:
	@echo "🔨 Building www-demo for WASM..."
	@cd www-demo && wasm-pack build --target web --release --out-dir ../www/pkg
	@echo "✅ Demo built successfully"

## Build demo in development mode (faster, with debug info)
build-dev:
	@echo "🔨 Building www-demo (dev mode)..."
	@cd www-demo && wasm-pack build --target web --dev --out-dir ../www/pkg
	@echo "✅ Dev build complete"

# ============================================================================
# Test Targets (EXTREME TDD)
# ============================================================================

## Run all unit tests
test:
	@echo "🧪 Running unit tests..."
	@cargo test --workspace

## Run probar GUI tests (requires browser)
test-probar:
	@echo "🎭 Running Probar GUI tests..."
	@cargo test --workspace --features probar -- --test-threads=1

## Run tests for specific demo
test-%:
	@echo "🧪 Testing $*..."
	@cargo test --package whisper-apr-demo-$*

# ============================================================================
# Coverage & Quality Gates
# ============================================================================

## Generate unified coverage report (Rust + GUI + Pixel via llvm-cov)
## Follows probar/bashrs pattern exactly
coverage:
	@echo "📊 Running unified coverage analysis..."
	@echo "🔍 Checking for cargo-llvm-cov and cargo-nextest..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@which cargo-nextest > /dev/null 2>&1 || (echo "📦 Installing cargo-nextest..." && cargo install cargo-nextest --locked)
	@echo "🧹 Cleaning old coverage data..."
	@cargo llvm-cov clean --workspace
	@mkdir -p target/coverage
	@echo "⚙️  Temporarily disabling global cargo config (mold breaks coverage)..."
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@echo "🧪 Phase 1: Running ALL tests with instrumentation..."
	@cargo llvm-cov --no-report nextest --no-tests=warn --workspace
	@echo "📊 Phase 2: Generating coverage reports..."
	@cargo llvm-cov report --html --output-dir target/coverage/html
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info
	@echo "⚙️  Restoring global cargo config..."
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                    UNIFIED COVERAGE REPORT                       ║"
	@echo "╠══════════════════════════════════════════════════════════════════╣"
	@cargo llvm-cov report --summary-only
	@echo "╠══════════════════════════════════════════════════════════════════╣"
	@echo "║  Rust Code Coverage:  via llvm-cov instrumentation               ║"
	@echo "║  GUI Coverage:        via probar UxCoverageTracker tests         ║"
	@echo "║  Pixel Coverage:      via probar SSIM/PSNR/CIEDE2000 tests       ║"
	@echo "╠══════════════════════════════════════════════════════════════════╣"
	@echo "║  HTML Report: target/coverage/html/index.html                    ║"
	@echo "║  LCOV Report: target/coverage/lcov.info                          ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"

## Show coverage summary only
coverage-summary:
	@cargo llvm-cov report --summary-only 2>/dev/null || echo "Run 'make coverage' first"

## Open HTML coverage report
coverage-open:
	@xdg-open target/coverage/html/index.html 2>/dev/null || open target/coverage/html/index.html 2>/dev/null || echo "Open: target/coverage/html/index.html"

## Clean coverage artifacts
coverage-clean:
	@cargo llvm-cov clean --workspace
	@rm -rf target/coverage target/llvm-cov
	@find . -name "*.profraw" -delete
	@echo "✓ Coverage artifacts cleaned"

## Run quality gates (CRITICAL - must pass before merge)
quality-gates:
	@echo "🚦 Running quality gates..."
	@cargo test --workspace quality_gates -- --nocapture
	@echo "✅ All quality gates passed"

## Check accessibility compliance
a11y:
	@echo "♿ Running accessibility audit..."
	@cargo test --workspace verify_accessibility -- --nocapture

# ============================================================================
# Tiered Quality Gates (bashrs methodology)
# ============================================================================

## Tier 1: On-save (<1s) - Fast feedback
tier1:
	@echo "⚡ Tier 1: Fast feedback..."
	@cargo check --workspace
	@cargo fmt --all -- --check
	@cargo clippy --workspace -- -D warnings
	@echo "✅ Tier 1 passed"

## Tier 2: Pre-commit (<5s) - Validation before commit
tier2: tier1
	@echo "📝 Tier 2: Pre-commit validation..."
	@cargo test --workspace --lib
	@make playbook-validate
	@echo "✅ Tier 2 passed"

## Tier 3: Pre-push (1-5 min) - Full validation
tier3: tier2
	@echo "🚀 Tier 3: Pre-push validation..."
	@cargo test --workspace
	@make coverage
	@cargo test -p whisper-apr-demo-tests full_coverage_simulation -- --nocapture
	@echo "✅ Tier 3 passed"

## Tier 4: CI/CD (5-60 min) - Comprehensive
tier4: tier3
	@echo "🏭 Tier 4: CI/CD validation..."
	@make quality-gates
	@make a11y
	@cargo mutants --workspace --no-times
	@echo "✅ Tier 4 passed"

# ============================================================================
# Playbook State Machine Testing (Prevents Whack-a-Mole Bugs)
# ============================================================================

## Validate all playbooks (state machine verification)
playbook-validate:
	@echo "🎭 Validating state machine playbooks..."
	@for pb in playbooks/*.yaml; do \
		echo "  Validating $$pb..."; \
		probar playbook $$pb --validate || exit 1; \
	done
	@echo "✅ All playbooks valid"

## Run mutation testing on playbooks (M1-M5 falsification)
playbook-mutate:
	@echo "🧬 Running playbook mutation testing..."
	@for pb in playbooks/*.yaml; do \
		echo "  Mutating $$pb..."; \
		probar playbook $$pb --mutate; \
	done

## Export state diagrams as SVG
playbook-diagrams:
	@echo "📊 Generating state diagrams..."
	@mkdir -p target/playbook-diagrams
	@for pb in playbooks/*.yaml; do \
		name=$$(basename $$pb .yaml); \
		echo "  Exporting $$name.svg..."; \
		probar playbook $$pb --export svg --export-output target/playbook-diagrams/$$name.svg; \
	done
	@echo "✅ Diagrams exported to target/playbook-diagrams/"

## Run full playbook test suite
playbook-test: playbook-validate playbook-mutate
	@echo "✅ Playbook testing complete"

## Quick playbook check (validation only, for CI)
playbook-quick:
	@probar playbook playbooks/*.yaml --validate --format json

# ============================================================================
# Browser E2E Tests (via probar BrowserController)
# ============================================================================

## Run browser E2E tests (requires Chrome)
test-browser:
	@echo "🌐 Running browser E2E tests..."
	@echo "Starting probar server on port 8090..."
	@probar serve www --port 8090 --cors &
	@sleep 2
	@echo "Running browser tests..."
	@cargo test --package whisper-apr-demo-tests browser_tests -- --test-threads=1 || (pkill -f "probar serve" 2>/dev/null; exit 1)
	@pkill -f "probar serve" 2>/dev/null || true
	@echo "✅ Browser tests complete"

## Run browser tests with coverage
test-browser-coverage:
	@echo "🌐 Running browser E2E tests with coverage..."
	@probar serve www --port 8090 --cors &
	@sleep 2
	@cargo llvm-cov --no-report nextest --package whisper-apr-demo-tests --test browser_tests || (pkill -f "probar serve" 2>/dev/null; exit 1)
	@pkill -f "probar serve" 2>/dev/null || true

# ============================================================================
# Development Server (via probar)
# ============================================================================

## Start development server with probar
serve:
	@echo "🌐 Starting probar development server..."
	@probar serve www --port 8080 --cors

## Start server with COOP/COEP headers for SharedArrayBuffer (parallel demos)
serve-parallel:
	@echo "🌐 Starting server with COOP/COEP headers for parallel WASM..."
	@probar serve www --port 8080 --cors --cross-origin-isolated

## Start server on port 8090 (for tests)
serve-test:
	@echo "🌐 Starting probar test server on port 8090..."
	@probar serve www --port 8090 --cors

## Watch for changes and rebuild (via probar)
watch:
	@echo "👀 Watching for changes with probar..."
	@probar watch --serve --port 8080

## Legacy serve (python fallback)
serve-legacy:
	@echo "🌐 Starting legacy python server..."
	@python3 -m http.server 8080 --directory www/

# ============================================================================
# Cleaning
# ============================================================================

## Clean build artifacts
clean:
	@echo "🧹 Cleaning..."
	@cargo clean
	@rm -rf */pkg */target target
	@echo "✅ Clean complete"

# ============================================================================
# Package for Deployment (WAPR-182)
# ============================================================================

DEPLOY_PATH := whisper-apr

## Package demo for deployment (output to dist/)
package: build
	@echo "📦 Packaging demo for deployment..."
	@mkdir -p dist/$(DEPLOY_PATH)
	@cp -r www/pkg dist/$(DEPLOY_PATH)/
	@cp www/*.html dist/$(DEPLOY_PATH)/
	@echo "✅ Package ready at: dist/$(DEPLOY_PATH)/"
	@echo ""
	@echo "To deploy, copy dist/$(DEPLOY_PATH) to your deployment target."

# ============================================================================
# Help
# ============================================================================

## Show this help
help:
	@echo "Whisper.apr Demo (Zero-JS)"
	@echo ""
	@echo "Build Commands:"
	@echo "  make build        - Build demo for WASM (release)"
	@echo "  make build-dev    - Build demo for WASM (dev mode)"
	@echo "  make package      - Build and package for deployment"
	@echo "  make clean        - Clean artifacts"
	@echo ""
	@echo "Test Commands:"
	@echo "  make test         - Run unit tests"
	@echo "  make test-probar  - Run Probar GUI tests"
	@echo "  make test-browser - Run browser E2E tests"
	@echo "  make coverage     - Generate unified coverage report"
	@echo ""
	@echo "Quality Gates:"
	@echo "  make tier1        - Fast feedback (<1s)"
	@echo "  make tier2        - Pre-commit (<5s)"
	@echo "  make tier3        - Pre-push (1-5 min)"
	@echo "  make tier4        - CI/CD (5-60 min)"
	@echo ""
	@echo "Development:"
	@echo "  make serve        - Start probar dev server (port 8080)"
	@echo "  make watch        - Watch and rebuild with probar"
