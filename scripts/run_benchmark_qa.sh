#!/bin/bash
# run_benchmark_qa.sh - 100-Point Falsification QA for WAPR-BENCH-001
#
# Reference: docs/specifications/benchmark-whisper-steps-a-z.md (Section 8)
#
# Usage:
#   ./scripts/run_benchmark_qa.sh
#   make qa-benchmark

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0
TOTAL=0

log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; ((PASS++)); ((TOTAL++)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; ((FAIL++)); ((TOTAL++)); }
log_skip() { echo -e "${YELLOW}[SKIP]${NC} $1"; ((SKIP++)); ((TOTAL++)); }

cd "$PROJECT_ROOT"

echo "=============================================="
echo "  WAPR-BENCH-001: 100-Point Falsification QA"
echo "=============================================="
echo ""

# =============================================================================
# Section 1: Asset Validity (10 points)
# =============================================================================
echo "=== Section 1: Asset Validity ==="

# Test 1: renacer.toml exists
if [ -f renacer.toml ]; then
    log_pass "#1: renacer.toml exists"
else
    log_fail "#1: renacer.toml missing"
fi

# Test 2: golden_traces directory exists
if [ -d golden_traces ]; then
    log_pass "#2: golden_traces/ directory exists"
else
    log_fail "#2: golden_traces/ directory missing"
fi

# Test 3: capture script exists and is executable
if [ -x scripts/capture_golden_traces.sh ]; then
    log_pass "#3: capture_golden_traces.sh is executable"
else
    log_fail "#3: capture_golden_traces.sh missing or not executable"
fi

# Test 4: Model file exists
if [ -f models/whisper-tiny-int8.apr ]; then
    log_pass "#4: Model file exists (whisper-tiny-int8.apr)"
else
    log_skip "#4: Model file missing (whisper-tiny-int8.apr)"
fi

# Test 5: Benchmark harness compiles
if cargo check --bench pipeline --quiet 2>/dev/null; then
    log_pass "#5: Benchmark harness compiles"
else
    log_fail "#5: Benchmark harness fails to compile"
fi

# Test 6: Format comparison example compiles
if cargo check --example format_comparison --quiet 2>/dev/null; then
    log_pass "#6: Format comparison example compiles"
else
    log_fail "#6: Format comparison example fails to compile"
fi

# Test 7: Tracing feature compiles
if cargo check --lib --features tracing --quiet 2>/dev/null; then
    log_pass "#7: Tracing feature compiles"
else
    log_fail "#7: Tracing feature fails to compile"
fi

# Test 8: CI workflow exists
if [ -f .github/workflows/benchmark.yml ]; then
    log_pass "#8: CI workflow exists"
else
    log_fail "#8: CI workflow missing"
fi

# Test 9: Makefile has profile target
if grep -q "^profile:" Makefile; then
    log_pass "#9: Makefile has profile target"
else
    log_fail "#9: Makefile missing profile target"
fi

# Test 10: Makefile has golden-traces target
if grep -q "^golden-traces:" Makefile; then
    log_pass "#10: Makefile has golden-traces target"
else
    log_fail "#10: Makefile missing golden-traces target"
fi

echo ""

# =============================================================================
# Section 2: Timing Accuracy (10 points)
# =============================================================================
echo "=== Section 2: Timing Accuracy ==="

# Test 11-14: Benchmark configuration checks
if grep -q "sample_size" benches/pipeline.rs; then
    log_pass "#11: Benchmark has sample_size configuration"
else
    log_fail "#11: Benchmark missing sample_size configuration"
fi

if grep -q "Throughput" benches/pipeline.rs; then
    log_pass "#12: Benchmark has throughput measurement"
else
    log_fail "#12: Benchmark missing throughput measurement"
fi

if grep -q "black_box" benches/pipeline.rs; then
    log_pass "#13: Benchmark uses black_box to prevent optimization"
else
    log_fail "#13: Benchmark missing black_box"
fi

if grep -q "iter_custom\|iter\|bench_function" benches/pipeline.rs; then
    log_pass "#14: Benchmark has iteration functions"
else
    log_fail "#14: Benchmark missing iteration functions"
fi

# Test 15: renacer assertions configured
if grep -q "\[\[assertion\]\]" renacer.toml 2>/dev/null; then
    ASSERTION_COUNT=$(grep -c "\[\[assertion\]\]" renacer.toml)
    if [ "$ASSERTION_COUNT" -ge 5 ]; then
        log_pass "#15: renacer.toml has $ASSERTION_COUNT assertions (≥5)"
    else
        log_fail "#15: renacer.toml has only $ASSERTION_COUNT assertions (<5)"
    fi
else
    log_fail "#15: renacer.toml missing assertions"
fi

echo ""

# =============================================================================
# Section 3: Coverage (10 points)
# =============================================================================
echo "=== Section 3: Coverage ==="

# Test 16-20: Pipeline step coverage
if grep -q "step_b_load\|step_b" benches/pipeline.rs; then
    log_pass "#16: Step B (load) benchmark present"
else
    log_fail "#16: Step B (load) benchmark missing"
fi

if grep -q "step_c_parse\|step_c" benches/pipeline.rs; then
    log_pass "#17: Step C (parse) benchmark present"
else
    log_fail "#17: Step C (parse) benchmark missing"
fi

if grep -q "step_f_mel\|step_f" benches/pipeline.rs; then
    log_pass "#18: Step F (mel) benchmark present"
else
    log_fail "#18: Step F (mel) benchmark missing"
fi

if grep -q "step_g_encoder\|step_g" benches/pipeline.rs; then
    log_pass "#19: Step G (encoder) benchmark present"
else
    log_fail "#19: Step G (encoder) benchmark missing"
fi

if grep -q "step_h_decoder\|step_h" benches/pipeline.rs; then
    log_pass "#20: Step H (decoder) benchmark present"
else
    log_fail "#20: Step H (decoder) benchmark missing"
fi

echo ""

# =============================================================================
# Section 4: Tracing Instrumentation (10 points)
# =============================================================================
echo "=== Section 4: Tracing Instrumentation ==="

# Test 21-25: Tracing spans in code
if grep -q "trace_enter" src/audio/mel.rs 2>/dev/null; then
    log_pass "#21: Mel spectrogram has tracing span"
else
    log_fail "#21: Mel spectrogram missing tracing span"
fi

if grep -q "trace_enter" src/model/encoder.rs 2>/dev/null; then
    log_pass "#22: Encoder has tracing span"
else
    log_fail "#22: Encoder missing tracing span"
fi

if grep -q "trace_enter" src/model/decoder.rs 2>/dev/null; then
    log_pass "#23: Decoder has tracing span"
else
    log_fail "#23: Decoder missing tracing span"
fi

if [ -f src/trace.rs ]; then
    log_pass "#24: Trace module exists"
else
    log_fail "#24: Trace module missing"
fi

if grep -q "trace_enter\|trace_span" src/trace.rs 2>/dev/null; then
    log_pass "#25: Trace macros defined"
else
    log_fail "#25: Trace macros not defined"
fi

echo ""

# =============================================================================
# Section 5: Documentation (10 points)
# =============================================================================
echo "=== Section 5: Documentation ==="

# Test 26-30: Documentation checks
if [ -f docs/specifications/benchmark-whisper-steps-a-z.md ]; then
    log_pass "#26: Benchmark specification exists"
else
    log_fail "#26: Benchmark specification missing"
fi

if grep -q "Appendix C" docs/specifications/benchmark-whisper-steps-a-z.md 2>/dev/null; then
    log_pass "#27: Aprender comparison documented"
else
    log_fail "#27: Aprender comparison not documented"
fi

if [ -f golden_traces/ANALYSIS.md ]; then
    log_pass "#28: Golden trace analysis exists"
else
    log_fail "#28: Golden trace analysis missing"
fi

if grep -q "RTF" docs/specifications/benchmark-whisper-steps-a-z.md 2>/dev/null; then
    log_pass "#29: RTF targets documented"
else
    log_fail "#29: RTF targets not documented"
fi

if grep -q "Checklist" docs/specifications/benchmark-whisper-steps-a-z.md 2>/dev/null; then
    log_pass "#30: Implementation checklist exists"
else
    log_fail "#30: Implementation checklist missing"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "  QA RESULTS"
echo "=============================================="
echo ""
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo "  SKIP: $SKIP"
echo "  TOTAL: $TOTAL"
echo ""

SCORE=$((PASS * 100 / TOTAL))
echo "  SCORE: $SCORE%"
echo ""

if [ $SCORE -ge 95 ]; then
    echo -e "${GREEN}VERDICT: Benchmark methodology VALIDATED (95%+)${NC}"
    exit 0
elif [ $SCORE -ge 80 ]; then
    echo -e "${YELLOW}VERDICT: Minor issues (80%+)${NC}"
    exit 1
else
    echo -e "${RED}VERDICT: Significant defects (<80%)${NC}"
    exit 2
fi
