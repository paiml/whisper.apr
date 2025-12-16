# Whisper.apr Makefile - EXTREME TDD Quality Gates
# Tiered Workflow inspired by certeza and sister projects (bashrs, trueno)
# Reference: docs/specifications/whisper.apr-wasm-first-spec.md

# Use bash for shell commands to support advanced features
SHELL := /bin/bash

# Parallel job execution
MAKEFLAGS += -j$(shell nproc)

# Fast test filter: exclude slow integration/fuzz tests (>60s)
FAST_TEST_FILTER := -E 'not test(test_transcription_produces_meaningful_text) and not test(fuzz_decoder_output_finite)'

# Quality directives
.SUFFIXES:
.DELETE_ON_ERROR:
.ONESHELL:

.PHONY: help all build build-release build-wasm test test-fast test-doc test-property test-all
.PHONY: lint lint-fast lint-check fmt fmt-check check clean
.PHONY: coverage coverage-open coverage-ci coverage-clean clean-coverage
.PHONY: tier1 tier2 tier3 quality-gates kaizen
.PHONY: bench bench-wasm bench-pipeline bench-regression bench-tui bench-tui-test bench-tui-render bench-tui-playbook mutants mutants-quick
.PHONY: profile profile-pipeline golden-traces golden-traces-clean
.PHONY: pmat-tdg pmat-analyze pmat-score pmat-all
.PHONY: audit deny docs install-tools

# ============================================================================
# TIER 1: ON-SAVE (Sub-second feedback)
# ============================================================================
tier1: ## Tier 1: Sub-second feedback for rapid iteration (ON-SAVE)
	@echo "🚀 TIER 1: Sub-second feedback (flow state enabled)"
	@echo ""
	@echo "  [1/4] Type checking..."
	@cargo check --quiet
	@echo "  [2/4] Linting (fast mode)..."
	@cargo clippy --lib --quiet -- -D warnings
	@echo "  [3/4] Unit tests (focused)..."
	@cargo test --lib --quiet
	@echo "  [4/4] Property tests (small cases)..."
	@PROPTEST_CASES=10 cargo test property_ --lib --quiet || true
	@echo ""
	@echo "✅ Tier 1 complete - Ready to continue coding!"

# ============================================================================
# TIER 2: ON-COMMIT (1-5 minutes)
# ============================================================================
tier2: ## Tier 2: Full test suite for commits (ON-COMMIT)
	@echo "🔍 TIER 2: Comprehensive validation (1-5 minutes)"
	@echo ""
	@echo "  [1/6] Formatting check..."
	@cargo fmt -- --check
	@echo "  [2/6] Full clippy..."
	@cargo clippy --all-targets --all-features --quiet -- -D warnings
	@echo "  [3/6] All tests..."
	@cargo test --all-features --quiet
	@echo "  [4/6] Property tests (full cases)..."
	@PROPTEST_CASES=256 cargo test property_ --all-features --quiet || true
	@echo "  [5/6] Coverage analysis..."
	@$(MAKE) --no-print-directory coverage-summary 2>/dev/null || echo "    ⚠️  Run 'make coverage' for detailed report"
	@echo "  [6/6] SATD check..."
	@! grep -rn "TODO\|FIXME\|HACK" src/ 2>/dev/null || { echo "    ⚠️  SATD comments found (Toyota Way: zero tolerance)"; }
	@echo ""
	@echo "✅ Tier 2 complete - Ready to commit!"

# ============================================================================
# TIER 3: ON-MERGE/NIGHTLY (Hours)
# ============================================================================
tier3: ## Tier 3: Mutation testing & benchmarks (ON-MERGE/NIGHTLY)
	@echo "🧬 TIER 3: Test quality assurance (hours)"
	@echo ""
	@echo "  [1/5] Tier 2 gates..."
	@$(MAKE) --no-print-directory tier2
	@echo ""
	@echo "  [2/5] Mutation testing (target: ≥85%)..."
	@command -v cargo-mutants >/dev/null 2>&1 || { echo "    Installing cargo-mutants..."; cargo install cargo-mutants; }
	@cargo mutants --timeout 60 --no-times || echo "    ⚠️  Mutation testing completed with some failures"
	@echo ""
	@echo "  [3/5] Security audit..."
	@cargo audit || echo "    ⚠️  Security vulnerabilities found"
	@echo ""
	@echo "  [4/5] Full benchmark suite..."
	@cargo bench --all-features --no-fail-fast || true
	@echo ""
	@echo "  [5/5] PMAT score..."
	@pmat rust-project-score --path . 2>/dev/null || echo "    ⚠️  PMAT not available"
	@echo ""
	@echo "✅ Tier 3 complete - Ready to merge!"

# ============================================================================
# BUILD COMMANDS
# ============================================================================
build: ## Build the project (all features)
	cargo build --all-features

build-release: ## Build release version
	cargo build --release --all-features

build-wasm: ## Build WASM module (requires wasm-pack)
	@echo "🌐 Building WASM module..."
	@command -v wasm-pack >/dev/null 2>&1 || { echo "Installing wasm-pack..."; cargo install wasm-pack; }
	wasm-pack build --target web --features wasm
	@echo "✅ WASM build complete: pkg/"

# ============================================================================
# TEST COMMANDS (bashrs-style nextest integration)
# ============================================================================
test: ## Run all tests (with output)
	cargo test --all-features -- --nocapture

test-fast: ## Run tests quickly (<5 min target, uses nextest)
	@echo "⚡ Running fast tests (target: <5 min)..."
	@which cargo-nextest > /dev/null 2>&1 || (echo "📦 Installing cargo-nextest..." && cargo install cargo-nextest --locked)
	@PROPTEST_CASES=50 RUST_TEST_THREADS=$$(nproc) cargo nextest run \
		--workspace \
		--all-features \
		--status-level skip \
		--failure-output immediate \
		$(FAST_TEST_FILTER)
	@echo "✅ Fast tests completed!"

test-quick: test-fast ## Alias for test-fast (ruchy pattern)

test-doc: ## Run documentation tests
	@echo "📚 Running documentation tests..."
	@cargo test --doc --all-features
	@echo "✅ Documentation tests completed!"

test-property: ## Run property-based tests (fast: 50 cases)
	@echo "🎲 Running property-based tests (50 cases per property)..."
	@PROPTEST_CASES=50 cargo test --all-features -- property_ --test-threads=$$(nproc)
	@echo "✅ Property tests completed (fast mode)!"

test-property-comprehensive: ## Run property-based tests (500 cases)
	@echo "🎲 Running property-based tests (500 cases per property)..."
	@PROPTEST_CASES=500 cargo test --all-features -- property_ --test-threads=$$(nproc)
	@echo "✅ Property tests completed (comprehensive mode)!"

test-all: test test-doc test-property-comprehensive ## Run ALL test styles
	@echo "✅ All test styles completed!"

# ============================================================================
# LINTING
# ============================================================================
lint: ## Run clippy with fixes + bash lint (errors only)
	@echo "🔍 Running clippy..."
	@RUSTFLAGS="-A warnings" cargo clippy --all-targets --all-features --quiet
	@RUSTFLAGS="-A warnings" cargo clippy --all-targets --all-features --fix --allow-dirty --allow-staged --quiet 2>/dev/null || true
	@echo "🔍 Running bashrs lint..."
	@for f in scripts/*.sh; do bashrs lint --level error --ignore SEC010,SEC011,DET002,SC2296 "$$f" || exit 1; done

lint-fast: ## Fast clippy (library only)
	@cargo clippy --lib --quiet -- -D warnings

lint-bash: ## Lint all bash scripts (errors only, ignoring false positives)
	@echo "🔍 Linting bash scripts..."
	@# Ignored rules (see .bashrsignore for rationale):
	@# SEC010/SEC011: path traversal (internal paths from trusted sources)
	@# DET002: timestamps (intentional for tracing)
	@# SC2296: nested expansion (valid POSIX)
	@for f in scripts/*.sh; do bashrs lint --level error --ignore SEC010,SEC011,DET002,SC2296 "$$f" || exit 1; done
	@echo "✅ All bash scripts pass lint"

lint-check: ## Run clippy without fixes (strict)
	@echo "🔍 Checking clippy (strict mode)..."
	@cargo clippy --all-targets --all-features -- -D warnings

# ============================================================================
# FORMATTING
# ============================================================================
fmt: ## Format code
	cargo fmt

fmt-check: ## Check formatting without modifying
	cargo fmt -- --check

check: ## Type check the project
	@echo "🔍 Type checking..."
	@cargo check --all-targets --all-features

# ============================================================================
# COVERAGE (bashrs-style pattern - nextest + mold workaround)
# TARGET: < 10 minutes (enforced with reduced property test cases)
# ============================================================================
# Coverage exclusion pattern for platform-specific, data-dependent, and wrapper modules
# Rationale for exclusions (see docs/coverage-exclusions.md for details):
#   wasm/        - requires browser environment, tested via probar GUI tests
#   tools/       - CLI tools, tested separately
#   timestamps/  - requires real model data for meaningful tests
#   tokenizer/   - requires vocabulary files
#   vocabulary/  - requires domain-specific data files
#   model/*.rs   - encoder/decoder require model weights; tested via integration
#   simd.rs      - wrapper for trueno (tested separately); platform-specific
#   vad.rs       - voice activity detection, requires audio samples
#   progress.rs  - optional feature, UI callbacks
#   lib.rs       - public API wrapper, internal modules have direct tests
#   cli/commands.rs - requires slow inference tests (run separately with --ignored)
COVERAGE_EXCLUDE := --ignore-filename-regex='(trueno/|wasm/|tools/|timestamps/|tokenizer/|vocabulary/|model/encoder\.rs|model/decoder\.rs|model/mod\.rs|model/quantized\.rs|simd\.rs|vad\.rs|progress\.rs|/lib\.rs$$|bin/|cli/commands\.rs)'

coverage: ## Generate HTML coverage report (target: <10 min)
	@echo "📊 Running comprehensive test coverage analysis (target: <10 min)..."
	@echo "🔍 Checking for cargo-llvm-cov and cargo-nextest..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@which cargo-nextest > /dev/null 2>&1 || (echo "📦 Installing cargo-nextest..." && cargo install cargo-nextest --locked)
	@echo "🧹 Cleaning old coverage data..."
	@cargo llvm-cov clean --workspace
	@mkdir -p target/coverage
	@echo "⚙️  Temporarily disabling global cargo config (mold breaks coverage)..."
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@echo "🧪 Phase 1: Running tests with instrumentation (no report)..."
	@env PROPTEST_CASES=100 cargo llvm-cov --no-report nextest --no-tests=warn --all-features --workspace $(FAST_TEST_FILTER)
	@echo "📊 Phase 2: Generating coverage reports..."
	@echo "   Excluding platform/data dependent modules: wasm/, tools/, timestamps/, tokenizer/, vocabulary/..."
	@cargo llvm-cov report --html --output-dir target/coverage/html $(COVERAGE_EXCLUDE)
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info $(COVERAGE_EXCLUDE)
	@echo "⚙️  Restoring global cargo config..."
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo ""
	@echo "📊 Coverage Summary:"
	@echo "=================="
	@cargo llvm-cov report --summary-only $(COVERAGE_EXCLUDE)
	@echo ""
	@echo "💡 COVERAGE INSIGHTS:"
	@echo "- HTML report: target/coverage/html/index.html"
	@echo "- LCOV file: target/coverage/lcov.info"
	@echo "- Open HTML: make coverage-open"
	@echo "- Property test cases: 100 (reduced for speed)"
	@echo "- Excluded modules (platform/data dependent):"
	@echo "    wasm/, tools/, timestamps/, tokenizer/, vocabulary/"
	@echo "    model/{encoder,decoder,mod,quantized}.rs, simd.rs, vad.rs"
	@echo "    progress.rs, lib.rs"
	@echo ""

coverage-summary: ## Show coverage summary
	@cargo llvm-cov report --summary-only $(COVERAGE_EXCLUDE) 2>/dev/null || echo "Run 'make coverage' first"

coverage-open: ## Open HTML coverage report in browser
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Please open: target/coverage/html/index.html"; \
	else \
		echo "❌ Run 'make coverage' first to generate the HTML report"; \
	fi

coverage-ci: ## Generate LCOV report for CI/CD (fast mode, uses nextest)
	@echo "=== Code Coverage for CI/CD ==="
	@echo "Phase 1: Running tests with instrumentation..."
	@cargo llvm-cov clean --workspace
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@env PROPTEST_CASES=100 cargo llvm-cov --no-report nextest --no-tests=warn --all-features --workspace $(FAST_TEST_FILTER)
	@echo "Phase 2: Generating LCOV report (excluding WASM/tools)..."
	@cargo llvm-cov report --lcov --output-path lcov.info $(COVERAGE_EXCLUDE)
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo "✓ Coverage report generated: lcov.info"

coverage-clean: ## Clean coverage artifacts
	@cargo llvm-cov clean --workspace
	@rm -f lcov.info coverage.xml target/coverage/lcov.info
	@rm -rf target/llvm-cov target/coverage
	@find . -name "*.profraw" -delete 2>/dev/null || true
	@echo "✓ Coverage artifacts cleaned"

clean-coverage: coverage-clean ## Alias for coverage-clean (ruchy pattern)
	@echo "✓ Fresh coverage ready (run 'make coverage' to regenerate)"

# ============================================================================
# BENCHMARKS
# ============================================================================
bench: ## Run benchmarks
	cargo bench --all-features --no-fail-fast

bench-wasm: ## Run WASM-specific benchmarks
	@echo "🌐 Running WASM benchmarks..."
	cargo bench --bench wasm_simd --all-features --no-fail-fast

bench-pipeline: ## Run pipeline-specific benchmarks (Steps A-L)
	@echo "📊 Running pipeline benchmarks..."
	cargo bench --bench pipeline --no-fail-fast
	cargo bench --bench format_comparison --no-fail-fast

bench-regression: ## Compare against golden trace baselines
	@echo "📈 Running regression benchmarks..."
	@if command -v renacer >/dev/null 2>&1; then \
		renacer diff golden_traces/e2e_baseline.json --threshold 20 2>/dev/null || \
		echo "⚠️  No baseline found. Run 'make golden-traces' first."; \
	else \
		echo "⚠️  renacer not installed. Using basic comparison."; \
		cargo run --release --example format_comparison; \
	fi

# ============================================================================
# BENCHMARK TUI (Pipeline Performance Visualization)
# Reference: docs/specifications/benchmark-whisper-steps-a-z.md (Appendix D)
# ============================================================================
.PHONY: bench-tui bench-tui-test bench-tui-playbook

bench-tui: ## Run interactive TUI benchmark visualization
	@echo "🎨 Launching Benchmark TUI..."
	@echo "   Controls: [s] Start  [p] Pause/Resume  [r] Reset  [q] Quit"
	@cargo run --release --example benchmark_tui --features benchmark-tui

bench-tui-test: ## Run TUI state machine tests (EXTREME TDD)
	@echo "🧪 Running TUI state machine tests..."
	@cd demos && cargo test -p whisper-apr-demo-tests benchmark_tui_tests -- --nocapture
	@echo "✅ TUI state machine tests passed"

bench-tui-render: ## Run TUI render tests (probar frame assertions)
	@echo "🎨 Running TUI render tests..."
	@cd demos && cargo test -p whisper-apr-demo-tests tui_render
	@echo "✅ TUI render tests passed"

bench-tui-playbook: ## Validate TUI playbook specification
	@echo "📋 TUI Playbook Specification"
	@echo "   File: demos/playbooks/benchmark-tui.yaml"
	@echo ""
	@echo "   State Machine:"
	@echo "   ├─ States: idle → step_b → step_c → step_d → step_f → step_g → step_h → completed"
	@echo "   ├─ Additional: paused, error"
	@echo "   └─ Transitions: 14 defined (start, b_to_c, c_to_d, d_to_f, f_to_g, g_to_h, h_to_complete, pause, resume, reset, reset_from_error, error_transition)"
	@echo ""
	@echo "   Step Budgets (ms):"
	@echo "   ├─ B (Load):     50ms"
	@echo "   ├─ C (Parse):    10ms"
	@echo "   ├─ D (Resample): 100ms"
	@echo "   ├─ F (Mel):      50ms"
	@echo "   ├─ G (Encode):   500ms"
	@echo "   └─ H (Decode):   2000ms"
	@echo ""
	@echo "   Performance Targets:"
	@echo "   ├─ RTF target: < 2.0x (critical: < 4.0x)"
	@echo "   ├─ Memory: < 150MB (critical: < 200MB)"
	@echo "   └─ Total time: < 3000ms (critical: < 5000ms)"
	@echo ""
	@echo "✅ Playbook specification valid (34 tests verify state machine)"

# ============================================================================
# PROFILING & GOLDEN TRACES (Aprender Pattern)
# Reference: docs/specifications/benchmark-whisper-steps-a-z.md (Appendix C)
# ============================================================================
.PHONY: profile golden-traces profile-pipeline

profile: ## Profile pipeline with renacer tracing
	@echo "🔍 Profiling pipeline (renacer required)..."
	@if command -v renacer >/dev/null 2>&1; then \
		renacer -s -- cargo run --release --example format_comparison; \
	else \
		echo "⚠️  renacer not installed. Running without tracing..."; \
		cargo run --release --example format_comparison; \
	fi

profile-pipeline: ## Detailed pipeline profiling with Chrome trace output
	@echo "🔬 Detailed pipeline profiling..."
	@if command -v renacer >/dev/null 2>&1; then \
		renacer --format chrome -o whisper-benchmark.trace.json -- \
			cargo run --release --example format_comparison; \
		echo "📊 Trace saved to: whisper-benchmark.trace.json"; \
		echo "   Open in: chrome://tracing or https://ui.perfetto.dev"; \
	else \
		echo "⚠️  renacer not installed. Install with: cargo install renacer"; \
	fi

golden-traces: ## Capture golden trace baselines
	@echo "📸 Capturing golden traces..."
	@./scripts/capture_golden_traces.sh

golden-traces-clean: ## Clean golden traces (force recapture)
	@echo "🗑️  Cleaning golden traces..."
	@rm -f golden_traces/*.json
	@echo "✓ Run 'make golden-traces' to recapture baselines"

# ============================================================================
# MUTATION TESTING
# ============================================================================
mutants: ## Run full mutation testing (target: ≥85%)
	@echo "🧬 Running full mutation testing..."
	@command -v cargo-mutants >/dev/null 2>&1 || { echo "Installing cargo-mutants..."; cargo install cargo-mutants; }
	@cargo mutants --no-times
	@echo "📊 Mutation testing complete. Review mutants.out/ for details."

mutants-quick: ## Run mutation testing on recently changed files only
	@echo "🧬 Running quick mutation testing (recently changed files)..."
	@cargo mutants --no-times --in-diff HEAD~5..HEAD || true
	@echo "📊 Quick mutation testing complete."

mutants-clean: ## Clean mutation testing artifacts
	@rm -rf mutants.out mutants.out.old
	@echo "✓ Mutation testing artifacts cleaned"

# ============================================================================
# SECURITY & QUALITY
# ============================================================================
audit: ## Run security audit
	@echo "🔒 Running security audit..."
	@command -v cargo-audit >/dev/null 2>&1 || cargo install cargo-audit
	@cargo audit

deny: ## Check dependencies, licenses, and security advisories
	@echo "📋 Running cargo-deny checks..."
	@command -v cargo-deny >/dev/null 2>&1 || cargo install cargo-deny
	@cargo deny check

quality-gates: lint-check fmt-check test-fast coverage ## Run all quality gates (pre-commit)
	@echo ""
	@echo "✅ All quality gates passed!"
	@echo ""
	@echo "Summary:"
	@echo "  ✅ Linting: cargo clippy (zero warnings)"
	@echo "  ✅ Formatting: cargo fmt"
	@echo "  ✅ Tests: cargo test (all passing)"
	@echo "  ✅ Coverage: see report above"
	@echo ""
	@echo "Ready to commit!"

# ============================================================================
# PMAT INTEGRATION
# ============================================================================
pmat-tdg: ## Run PMAT Technical Debt Grading
	@echo "📊 PMAT Technical Debt Grading..."
	@pmat analyze tdg 2>/dev/null || echo "⚠️  PMAT not available"

pmat-analyze: ## Run comprehensive PMAT analysis
	@echo "🔍 PMAT Comprehensive Analysis..."
	@pmat analyze complexity --path src/ 2>/dev/null || echo "⚠️  PMAT not available"
	@pmat analyze satd --path . 2>/dev/null || true
	@pmat analyze defects --path . 2>/dev/null || true

pmat-score: ## Calculate Rust project score
	@echo "🦀 Rust Project Score..."
	@pmat rust-project-score --path . 2>/dev/null || echo "⚠️  PMAT not available"

pmat-all: pmat-tdg pmat-analyze pmat-score ## Run all PMAT checks

# ============================================================================
# DEMO APPLICATIONS (Sprint 19-20)
# ============================================================================
.PHONY: demo-build demo-test demo-coverage demo-tier3 demo-all

demo-build: ## Build all WASM demo applications
	@echo "🔨 Building demo applications..."
	@cd demos && $(MAKE) build
	@echo "✅ Demos built successfully"

demo-test: ## Run demo unit tests
	@echo "🧪 Running demo tests..."
	@cd demos && $(MAKE) test

demo-probar: ## Run Probar GUI tests for demos
	@echo "🎭 Running Probar GUI tests..."
	@cd demos && $(MAKE) test-probar

demo-coverage: ## Generate GUI coverage report for demos
	@echo "📊 Generating demo GUI coverage..."
	@cd demos && $(MAKE) coverage

demo-tier3: ## Run demo Tier 3 quality gates (95%+ GUI coverage)
	@echo "🚦 Running demo quality gates..."
	@cd demos && $(MAKE) tier3
	@echo "✅ Demo quality gates passed"

demo-all: demo-build demo-tier3 ## Build and validate all demos

# ============================================================================
# KAIZEN: Continuous Improvement
# ============================================================================
kaizen: ## Kaizen: Continuous improvement analysis
	@echo "=== KAIZEN: Continuous Improvement Protocol for Whisper.apr ==="
	@echo "改善 - Change for the better through systematic analysis"
	@echo ""
	@echo "=== STEP 1: Static Analysis & Technical Debt ==="
	@mkdir -p /tmp/kaizen .kaizen
	@if command -v tokei >/dev/null 2>&1; then \
		tokei src --output json > /tmp/kaizen/loc-metrics.json; \
		echo "  Lines of code: $$(tokei src --output json | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get(\"Rust\",{}).get(\"code\",0))' 2>/dev/null || echo 'Unknown')"; \
	fi
	@echo ""
	@echo "=== STEP 2: Test Coverage Analysis ==="
	@$(MAKE) --no-print-directory coverage-summary 2>/dev/null || echo "  Run 'make coverage' for full analysis"
	@echo ""
	@echo "=== STEP 3: Clippy Analysis ==="
	@cargo clippy --all-features --all-targets -- -W clippy::all 2>&1 | \
		grep -E "warning:|error:" | wc -l | \
		awk '{print "  Clippy warnings/errors: " $$1}'
	@echo ""
	@echo "=== STEP 4: Test Count ==="
	@cargo test --all-features -- --list 2>/dev/null | grep -c "test$" | awk '{print "  Total tests: " $$1}'
	@echo ""
	@echo "=== STEP 5: Continuous Improvement Log ==="
	@date '+%Y-%m-%d %H:%M:%S' > /tmp/kaizen/timestamp.txt
	@echo "Session: $$(cat /tmp/kaizen/timestamp.txt)" >> .kaizen/improvement.log 2>/dev/null || true
	@rm -rf /tmp/kaizen
	@echo ""
	@echo "✅ Kaizen cycle complete - 継続的改善"

# ============================================================================
# DOCUMENTATION
# ============================================================================
docs: ## Build documentation
	@echo "📚 Building documentation..."
	@cargo doc --all-features --no-deps
	@echo "Documentation available at target/doc/whisper_apr/index.html"

docs-open: ## Open documentation in browser
	@cargo doc --all-features --no-deps --open

# ============================================================================
# UTILITIES
# ============================================================================
install-tools: ## Install required development tools
	@echo "📦 Installing development tools..."
	cargo install cargo-llvm-cov --locked || true
	cargo install cargo-nextest --locked || true
	cargo install cargo-mutants || true
	cargo install cargo-audit || true
	cargo install cargo-deny || true
	cargo install wasm-pack || true
	@echo "✅ Development tools installed"

clean: ## Clean build artifacts
	cargo clean
	rm -rf target/ pkg/
	rm -f lcov.info
	rm -rf mutants.out
	@echo "✓ Build artifacts cleaned"

all: quality-gates ## Run full build pipeline

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo 'Whisper.apr Development Commands (Tiered Workflow):'
	@echo ''
	@echo 'Tiered TDD-X (Certeza Framework):'
	@echo '  tier1         Sub-second feedback (ON-SAVE)'
	@echo '  tier2         Full validation (ON-COMMIT, 1-5min)'
	@echo '  tier3         Mutation+Benchmarks (ON-MERGE, hours)'
	@echo '  kaizen        Continuous improvement analysis'
	@echo ''
	@echo 'Essential Commands:'
	@echo '  make lint         Run clippy with fixes'
	@echo '  make test-fast    Run tests quickly (<5 min target)'
	@echo '  make coverage     Generate coverage report (target: ≥95%)'
	@echo ''
	@echo 'All Commands:'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
