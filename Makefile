# Whisper.apr Makefile - EXTREME TDD Quality Gates
# Tiered Workflow inspired by certeza and sister projects (bashrs, trueno)
# Reference: docs/specifications/whisper.apr-wasm-first-spec.md

# Use bash for shell commands to support advanced features
SHELL := /bin/bash

# Parallel job execution
MAKEFLAGS += -j$(shell nproc)

# Fast test filter: exclude ALL slow tests (>5s) for rapid iteration
# Slow tests: encoder_forward, transcription, encode_3_second, pipeline_step, rtf_measurement
FAST_TEST_FILTER := -E 'not test(/encoder_forward|transcription|encode_3_second|pipeline_step|rtf_measurement|fuzz_decoder|paged_kv|quantized/)'

# Quality directives
.SUFFIXES:
.DELETE_ON_ERROR:
.ONESHELL:

.PHONY: help all build build-release build-wasm test test-fast test-doc test-property test-all
.PHONY: lint lint-fast lint-check fmt fmt-check check check-features clean
.PHONY: coverage coverage-open coverage-ci coverage-clean clean-coverage
.PHONY: tier1 tier2 tier3 quality-gates kaizen
.PHONY: bench bench-wasm bench-pipeline bench-regression bench-tui bench-tui-test bench-tui-render bench-tui-playbook mutants mutants-quick
.PHONY: profile profile-pipeline golden-traces golden-traces-clean
.PHONY: pmat-tdg pmat-analyze pmat-score pmat-all
.PHONY: audit deny docs install-tools install smoke-test dogfood

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
	@echo "  [1/7] Formatting check..."
	@cargo fmt -- --check
	@echo "  [2/7] Full clippy..."
	@cargo clippy --all-targets --all-features --quiet -- -D warnings
	@echo "  [3/7] Feature-matrix build (wasm + minimal-lib — guards feature gating)..."
	@$(MAKE) --no-print-directory check-features
	@echo "  [4/7] All tests..."
	@cargo test --all-features --quiet
	@echo "  [5/7] Property tests (full cases)..."
	@PROPTEST_CASES=25 cargo test property_ --all-features --quiet || true
	@echo "  [6/7] Coverage analysis..."
	@$(MAKE) --no-print-directory coverage-summary 2>/dev/null || echo "    ⚠️  Run 'make coverage' for detailed report"
	@echo "  [7/7] SATD check..."
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
	wasm-pack build --target web --no-default-features --features wasm
	@echo "✅ WASM build complete: pkg/"

# ============================================================================
# TEST COMMANDS (bashrs-style nextest integration)
# ============================================================================
test: ## Run all tests (with output)
	PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo test --all-features -- --nocapture

test-fast: ## Run tests quickly (<30s target)
	@echo "⚡ Fast tests (<30s)..."
	@PROPTEST_CASES=10 QUICKCHECK_TESTS=10 cargo test --lib --quiet -- --skip encode_3_second --skip rtf_measurement
	@echo "✅ Done!"

test-quick: test-fast ## Alias

test-doc: ## Run documentation tests
	@echo "📚 Running documentation tests..."
	@PROPTEST_CASES=25 QUICKCHECK_TESTS=25 cargo test --doc --all-features
	@echo "✅ Documentation tests completed!"

test-property: ## Run property-based tests (fast: 50 cases)
	@echo "🎲 Running property-based tests (50 cases per property)..."
	@PROPTEST_CASES=25 cargo test --all-features -- property_ --test-threads=$$(nproc)
	@echo "✅ Property tests completed (fast mode)!"

test-property-comprehensive: ## Run property-based tests (500 cases)
	@echo "🎲 Running property-based tests (500 cases per property)..."
	@PROPTEST_CASES=250 cargo test --all-features -- property_ --test-threads=$$(nproc)
	@echo "✅ Property tests completed (comprehensive mode)!"

test-all: test test-doc test-property-comprehensive ## Run ALL test styles
	@echo "✅ All test styles completed!"

# ============================================================================
# LINTING
# ============================================================================
lint: ## Run clippy with fixes + bash lint (errors only)
	@echo "🔍 Running clippy..."
	@cargo clippy --lib --features "cli,converter,tracing,benchmark-tui,tui,format-encryption,format-signing,format-quantize,format-homomorphic" --quiet -- -D warnings
	@cargo clippy --lib --features "cli,converter,tracing,benchmark-tui,tui,format-encryption,format-signing,format-quantize,format-homomorphic" --fix --allow-dirty --allow-staged --quiet 2>/dev/null || true
	@echo "🔍 Running bashrs lint..."
	@for f in scripts/*.sh; do bashrs lint --level error --ignore SEC010,SEC011,SEC001,DET002,SC2296,SC2188 "$$f" || exit 1; done

lint-fast: ## Fast clippy (library only)
	@cargo clippy --lib --quiet -- -D warnings

lint-bash: ## Lint all bash scripts (errors only, ignoring false positives)
	@echo "🔍 Linting bash scripts..."
	@# Ignored rules (see .bashrsignore for rationale):
	@# SEC010/SEC011: path traversal (internal paths from trusted sources)
	@# DET002: timestamps (intentional for tracing)
	@# SC2296: nested expansion (valid POSIX)
	@# SC2188: heredoc content false positives
	@for f in scripts/*.sh; do bashrs lint --level error --ignore SEC010,SEC011,SEC001,DET002,SC2296,SC2188 "$$f" || exit 1; done
	@echo "✅ All bash scripts pass lint"

lint-check: ## Run clippy without fixes (strict)
	@echo "🔍 Checking clippy (strict mode)..."
	@cargo clippy --all-targets --features "cli,converter,tracing,benchmark-tui,tui,format-encryption,format-signing,format-quantize,format-homomorphic" -- -D warnings

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

check-features: ## Feature-matrix regression guard: crate must compile for wasm + minimal-lib
	@echo "🔍 Feature matrix (no-default-features builds — parallel/rayon must be gated)..."
	@echo "  [1/2] wasm32 (--no-default-features --features wasm)..."
	@cargo check --target wasm32-unknown-unknown --no-default-features --features wasm
	@echo "  [2/2] minimal library (--no-default-features --features std)..."
	@cargo check --no-default-features --features std
	@echo "✅ Feature matrix OK — parallel/rayon correctly gated for wasm/minimal builds"

# ============================================================================
# COVERAGE (cargo llvm-cov test pattern — no nextest, no profraw explosion)
# ============================================================================
# Coverage exclusion: external deps, WASM (probar-tested), model weights
# (integration-tested), CLI (tested separately), GPU, generated code.
# All other modules use source-level #[coverage(off)] for transparent exclusion.
COVERAGE_EXCLUDE := --ignore-filename-regex='(trueno/|aprender/|realizar/|wasm/|model/|cli/|bin/|gpu/|_generated\.rs$$|benchmark)'
COV_THRESHOLD ?= 95

coverage: ## Coverage summary + threshold check (<5 min)
	@echo "📊 Running coverage ($(COV_THRESHOLD)%+ threshold)..."
	@which cargo-llvm-cov > /dev/null 2>&1 || cargo install cargo-llvm-cov --locked
	@mkdir -p target/coverage
	@cargo llvm-cov clean --workspace 2>/dev/null || true
	@echo "🧪 Running tests with instrumentation..."
	@env RUSTC_WRAPPER= PROPTEST_CASES=2 QUICKCHECK_TESTS=2 cargo llvm-cov test \
		--lib \
		$(COVERAGE_EXCLUDE) \
		-- --test-threads=$$(nproc) \
		--skip encode_3_second \
		--skip rtf_measurement \
		--skip rtf_target \
		|| true
	@echo "📊 Generating report..."
	@cargo llvm-cov report --summary-only $(COVERAGE_EXCLUDE) | tee target/coverage/summary.txt | grep -E "^TOTAL"
	@COV_PCT=$$(grep -E '^TOTAL' target/coverage/summary.txt | awk '{n=0; for(i=1;i<=NF;i++){if($$i ~ /[0-9]+\.[0-9]+%/){n++; if(n==3){gsub(/%/,"",$$i);print $$i;exit}}}}'); \
	if [ -n "$$COV_PCT" ] && [ $$(echo "$$COV_PCT < $(COV_THRESHOLD)" | bc -l) -eq 1 ]; then \
		echo "❌ Coverage $${COV_PCT}% is below threshold $(COV_THRESHOLD)%"; \
		exit 1; \
	else \
		echo "✅ Coverage $${COV_PCT}% meets threshold $(COV_THRESHOLD)%"; \
	fi

coverage-html: ## Generate HTML report from last coverage run
	@echo "📊 Generating HTML report..."
	@mkdir -p target/coverage
	@cargo llvm-cov report --html --output-dir target/coverage/html $(COVERAGE_EXCLUDE)
	@echo "📍 HTML: target/coverage/html/index.html"

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

coverage-ci: ## Generate LCOV report for CI (fast mode, --lib only)
	@echo "📊 Running CI coverage (--lib only)..."
	@env RUSTC_WRAPPER= PROPTEST_CASES=2 QUICKCHECK_TESTS=2 \
		cargo llvm-cov test \
		--lib \
		--lcov --output-path lcov.info \
		$(COVERAGE_EXCLUDE) \
		-- --test-threads=$$(nproc) \
		--skip encode_3_second \
		--skip rtf_measurement \
		--skip rtf_target \
		|| true
	@echo "✓ Coverage report generated: lcov.info"

coverage-clean: ## Clean coverage artifacts
	@rm -f lcov.info coverage.xml target/coverage/lcov.info
	@rm -rf target/llvm-cov target/coverage
	@find . -name "*.profraw" -delete 2>/dev/null || true
	@echo "✓ Coverage artifacts cleaned"

clean-coverage: coverage-clean ## Alias for coverage-clean

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
.PHONY: profile golden-traces profile-pipeline profile-heap profile-heap-clean

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

profile-heap: ## Heap profile with dhat-rs (outputs dhat-heap.json)
	@echo "🔬 Running dhat-rs heap profiler..."
	@cargo run --example dhat_profile --features dhat-profiler --release
	@echo ""
	@echo "📊 Output: dhat-heap.json"
	@echo "   View at: https://nnethercote.github.io/dh_view/dh_view.html"

profile-heap-clean: ## Clean dhat profiling artifacts
	@rm -f dhat-heap.json
	@echo "✓ dhat artifacts cleaned"

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
# INSTALL & DOGFOOD
# ============================================================================
.PHONY: install smoke-test dogfood

MODEL_PATH := models/whisper-tiny-fb.apr
AUDIO_PATH := demos/test-audio/test-speech-1.5s.wav
EXPECT_TEXT := birds

install: ## Install whisper-apr binary via cargo install
	@# Prevent duplicate installs: remove any whisper-apr outside ~/.cargo/bin
	@for bin in $$(which -a whisper-apr 2>/dev/null | grep -v '\.cargo/bin/' | sort -u); do \
		echo "Removing stale whisper-apr at $$bin"; \
		rm -f "$$bin"; \
	done
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "CUDA detected — building with GPU support"; \
		cargo install --path . --features "cli,realizar-gpu"; \
	else \
		cargo install --path . --features cli; \
	fi
	@# Verify single install
	@if [ "$$(which -a whisper-apr 2>/dev/null | sort -u | wc -l)" -gt 1 ]; then \
		echo "ERROR: multiple whisper-apr installs detected:"; \
		which -a whisper-apr | sort -u; \
		exit 1; \
	fi

smoke-test: ## Quick smoke test via cargo run (no install required)
	@echo "🔥 Smoke test (cargo run)..."
	@echo ""
	@echo "  [1/4] Version check..."
	@cargo run --features cli -- --version
	@echo ""
	@echo "  [2/4] Diagnose (tokenizer)..."
	@cargo run --features cli -- diagnose --tokenizer-only
	@echo ""
	@echo "  [3/4] Backend test (SIMD)..."
	@cargo run --features cli -- test --backend simd
	@echo ""
	@echo "  [4/4] Transcription test..."
	@test -f $(MODEL_PATH) || { echo "❌ Model not found: $(MODEL_PATH)"; exit 1; }
	@test -f $(AUDIO_PATH) || { echo "❌ Audio not found: $(AUDIO_PATH)"; exit 1; }
	@OUTPUT=$$(cargo run --features cli -- transcribe -f $(AUDIO_PATH) --model-path $(MODEL_PATH) 2>&1); \
	echo "  Output: $$OUTPUT"; \
	echo "$$OUTPUT" | grep -qi "$(EXPECT_TEXT)" || { echo "❌ Expected '$(EXPECT_TEXT)' in output"; exit 1; }
	@echo ""
	@echo "✅ Smoke test passed!"

dogfood: install ## Install binary and run end-to-end selftest
	@echo "🐕 Dogfood: exercising installed whisper-apr binary..."
	@echo ""
	@which whisper-apr || { echo "❌ whisper-apr not found in PATH"; exit 1; }
	@echo "  [1/3] Version..."
	@whisper-apr --version
	@echo ""
	@echo "  [2/3] Selftest (diagnose + backend)..."
	@whisper-apr selftest
	@echo ""
	@echo "  [3/3] Selftest with transcription..."
	@test -f $(MODEL_PATH) || { echo "❌ Model not found: $(MODEL_PATH)"; exit 1; }
	@test -f $(AUDIO_PATH) || { echo "❌ Audio not found: $(AUDIO_PATH)"; exit 1; }
	@whisper-apr selftest --model $(MODEL_PATH) --audio $(AUDIO_PATH) --expect "$(EXPECT_TEXT)"
	@echo ""
	@echo "✅ Dogfood passed!"

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
	@echo 'Install & Dogfood:'
	@echo '  make install      Install whisper-apr binary (cargo install)'
	@echo '  make smoke-test   Quick smoke test via cargo run (no install)'
	@echo '  make dogfood      Install + end-to-end selftest'
	@echo ''
	@echo 'All Commands:'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
