# Whisper.apr Demo Applications Makefile
# EXTREME TDD + Probar-First Development
#
# Quality Gates:
# - GUI Coverage: ≥95%
# - Button Coverage: 100%
# - State Coverage: 100%
# - Error Path Coverage: ≥95%

.PHONY: all build test coverage tier1 tier2 tier3 clean help

# Default target
all: build test

# ============================================================================
# Build Targets
# ============================================================================

## Build all demos for WASM
build:
	@echo "🔨 Building all demos for WASM..."
	@for demo in realtime-transcription upload-transcription realtime-translation upload-translation; do \
		echo "  Building $$demo..."; \
		cd $$demo && wasm-pack build --target web --release && cd ..; \
	done
	@echo "✅ All demos built successfully"

## Build demos in development mode (faster, with debug info)
build-dev:
	@echo "🔨 Building demos (dev mode)..."
	@for demo in realtime-transcription upload-transcription realtime-translation upload-translation; do \
		cd $$demo && wasm-pack build --target web --dev && cd ..; \
	done

## Build specific demo
build-%:
	@echo "🔨 Building $*..."
	@cd $* && wasm-pack build --target web --release

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

## Generate GUI coverage report
coverage:
	@echo "📊 Generating GUI coverage report..."
	@probar coverage --workspace --output target/demo-coverage.html
	@echo "📄 Report: target/demo-coverage.html"

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
	@echo "✅ Tier 2 passed"

## Tier 3: Pre-push (1-5 min) - Full validation
tier3: tier2
	@echo "🚀 Tier 3: Pre-push validation..."
	@make test-probar
	@make coverage
	@probar coverage --workspace --assert-min 95.0
	@echo "✅ Tier 3 passed"

## Tier 4: CI/CD (5-60 min) - Comprehensive
tier4: tier3
	@echo "🏭 Tier 4: CI/CD validation..."
	@make quality-gates
	@make a11y
	@cargo mutants --workspace --no-times
	@echo "✅ Tier 4 passed"

# ============================================================================
# Development Server
# ============================================================================

## Start development server
serve:
	@echo "🌐 Starting development server..."
	@python3 -m http.server 8080 --directory pkg/

## Watch for changes and rebuild
watch:
	@echo "👀 Watching for changes..."
	@cargo watch -s "make build-dev"

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
# Help
# ============================================================================

## Show this help
help:
	@echo "Whisper.apr Demo Applications"
	@echo ""
	@echo "EXTREME TDD Commands:"
	@echo "  make build        - Build all demos for WASM"
	@echo "  make test         - Run unit tests"
	@echo "  make test-probar  - Run Probar GUI tests"
	@echo "  make coverage     - Generate GUI coverage report"
	@echo ""
	@echo "Quality Gates:"
	@echo "  make tier1        - Fast feedback (<1s)"
	@echo "  make tier2        - Pre-commit (<5s)"
	@echo "  make tier3        - Pre-push (1-5 min)"
	@echo "  make tier4        - CI/CD (5-60 min)"
	@echo ""
	@echo "Development:"
	@echo "  make serve        - Start dev server"
	@echo "  make watch        - Watch and rebuild"
	@echo "  make clean        - Clean artifacts"
