#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# bashrs: compliant
set -euo pipefail

# Description: Capture baseline performance traces for whisper.apr
# Reference: docs/specifications/benchmark-whisper-steps-a-z.md (Appendix C.4)
# Usage: ./scripts/capture_golden_traces.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TRACES_DIR="$PROJECT_ROOT/golden_traces"
MODELS_DIR="$PROJECT_ROOT/models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prereqs() {
    log_info "Checking prerequisites..."

    if ! command -v renacer &> /dev/null; then
        log_warn "renacer not found - traces will use basic timing only"
        USE_RENACER=false
    else
        USE_RENACER=true
        log_info "renacer version: $(renacer --version 2>/dev/null || echo 'unknown')"
    fi

    if [[ ! -f "$MODELS_DIR/whisper-tiny-int8.apr" ]]; then
        log_error "Model not found: $MODELS_DIR/whisper-tiny-int8.apr"
        log_error "Run: cargo run --bin whisper-convert -- --model tiny --quantize int8"
        exit 1
    fi

    log_info "Model found: whisper-tiny-int8.apr ($(du -h "$MODELS_DIR/whisper-tiny-int8.apr" | cut -f1))"
}

# Build release binary
build_release() {
    log_info "Building release binary..."
    cargo build --release --example format_comparison 2>&1 | tail -3
}

# Capture trace for a specific benchmark
capture_trace() {
    local name="$1"
    local command="$2"
    local output="$TRACES_DIR/${name}.json"

    log_info "Capturing: $name"

    if [[ "$USE_RENACER" == "true" ]]; then
        renacer --format json -- $command > "$output" 2>&1 || true
    else
        # Fallback: basic timing with JSON output
        local start_time=$(date +%s%N)
        $command > /dev/null 2>&1 || true
        local end_time=$(date +%s%N)
        local duration_ms=$(( (end_time - start_time) / 1000000 ))

        cat > "$output" << EOF
{
  "name": "$name",
  "timestamp": "$(date -Iseconds)",
  "duration_ms": $duration_ms,
  "note": "Basic timing (renacer not available)"
}
EOF
    fi

    if [[ -f "$output" ]]; then
        log_info "  -> $output ($(du -h "$output" | cut -f1))"
    else
        log_warn "  -> Failed to capture $name"
    fi
}

# Generate analysis report
generate_analysis() {
    log_info "Generating analysis report..."

    local analysis="$TRACES_DIR/ANALYSIS.md"
    local timestamp=$(date -Iseconds)

    cat > "$analysis" << EOF
# Golden Trace Analysis

Generated: $timestamp

Reference: \`docs/specifications/benchmark-whisper-steps-a-z.md\`

## Baselines (whisper-tiny-int8)

| Component | Target | Measured | Status |
|-----------|--------|----------|--------|
EOF

    # Parse captured traces and update analysis
    for trace in "$TRACES_DIR"/*.json; do
        if [[ -f "$trace" ]]; then
            local name=$(basename "$trace" .json)
            local duration=$(grep -o '"duration_ms":[0-9]*' "$trace" 2>/dev/null | cut -d: -f2 || echo "TBD")
            echo "| $name | - | ${duration}ms | ✅ Captured |" >> "$analysis"
        fi
    done

    cat >> "$analysis" << 'EOF'

## RTF Targets

| Audio Duration | Max Time | RTF Target | Status |
|---------------|----------|------------|--------|
| 1.5s | 3.0s | 2.0x | ⏳ |
| 3.0s | 4.5s | 1.5x | ⏳ |
| 30s | 30s | 1.0x | ⏳ |

## Regression Thresholds

- Any component >20% slower than baseline: **FAIL**
- Total pipeline >10% slower: **WARN**
- Memory >150MB peak (tiny-int8): **FAIL**

## Captured Traces

EOF

    # List captured traces
    for trace in "$TRACES_DIR"/*.json; do
        if [[ -f "$trace" ]]; then
            local name=$(basename "$trace")
            local size=$(du -h "$trace" | cut -f1)
            local date=$(stat -c %y "$trace" 2>/dev/null | cut -d. -f1 || date)
            echo "- \`$name\` ($size) - $date" >> "$analysis"
        fi
    done

    log_info "Analysis written to: $analysis"
}

# Main execution
main() {
    echo "=============================================="
    echo "  Whisper.apr Golden Trace Capture"
    echo "=============================================="
    echo ""

    mkdir -p "$TRACES_DIR"

    check_prereqs
    build_release

    echo ""
    log_info "Capturing golden traces..."
    echo ""

    # Capture format comparison (includes mel, encoder, decoder timing)
    capture_trace "format_comparison" \
        "cargo run --release --example format_comparison"

    # Capture mel spectrogram baseline (if example exists)
    if [[ -f "$PROJECT_ROOT/examples/mel_spectrogram.rs" ]]; then
        capture_trace "mel_spectrogram" \
            "cargo run --release --example mel_spectrogram"
    fi

    # Capture inference benchmark (if it exists)
    if cargo bench --bench inference --list 2>/dev/null | grep -q encoder; then
        capture_trace "inference_encoder" \
            "cargo bench --bench inference -- encoder --noplot"
    fi

    echo ""
    generate_analysis

    echo ""
    echo "=============================================="
    log_info "Golden traces captured to: $TRACES_DIR/"
    echo "=============================================="
    echo ""
    echo "Next steps:"
    echo "  1. Review: cat $TRACES_DIR/ANALYSIS.md"
    echo "  2. Commit baselines: git add golden_traces/"
    echo "  3. Run regression: make bench-regression"
    echo ""
}

main "$@"
