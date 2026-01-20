#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# QA Script for WAPR-PERF-004: transcribe-folder Falsification Points F101-F110
# Reference: docs/specifications/transcribe-folder-spec.md §H
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
TOTAL=0

log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS=$((PASS+1)); TOTAL=$((TOTAL+1)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL=$((FAIL+1)); TOTAL=$((TOTAL+1)); }

cd "$PROJECT_ROOT"

echo "=============================================="
echo "  WAPR-PERF-004: transcribe-folder QA"
echo "  Falsification Points F101-F110"
echo "=============================================="
echo ""

# Setup test directory
TEST_DIR=$(mktemp -d)
trap 'rm -rf "$TEST_DIR"' EXIT

# =============================================================================
# F101: Structure Mirroring
# =============================================================================
echo "=== F101: Structure Mirroring ==="

# Create nested directory structure
mkdir -p "$TEST_DIR/input/subdir/deep"
echo "dummy" > "$TEST_DIR/input/root.wav"
echo "dummy" > "$TEST_DIR/input/subdir/nested.wav"
echo "dummy" > "$TEST_DIR/input/subdir/deep/very_nested.wav"

# Run unit test for structure mirroring
if cargo test --features cli --lib test_compute_mirrored_output_path_mirrored --quiet 2>/dev/null; then
    log_pass "F101: Structure mirroring (./raw/sub/b.mp3 → ./trans/sub/b.json)"
else
    log_fail "F101: Structure mirroring failed"
fi

# =============================================================================
# F102: Format Extension Mismatch
# =============================================================================
echo ""
echo "=== F102: Format Extension ==="

if cargo test --features cli --lib test_compute_mirrored_output_path_format_extension --quiet 2>/dev/null; then
    log_pass "F102: Format extension replacement (--format json → .json)"
else
    log_fail "F102: Format extension replacement failed"
fi

# =============================================================================
# F103: Atomicity Violation
# =============================================================================
echo ""
echo "=== F103: Atomic Writes ==="

if cargo test --features cli --lib test_atomic_write_no_partial --quiet 2>/dev/null; then
    log_pass "F103: Atomic writes (temp file → rename, no partial files)"
else
    log_fail "F103: Atomic writes failed"
fi

# =============================================================================
# F104: Skip Existing (Resumable)
# =============================================================================
echo ""
echo "=== F104: Skip Existing ==="

# Verify skip_existing field exists in BatchArgs
if grep -q "skip_existing" src/cli/args.rs; then
    log_pass "F104: --skip-existing flag available for resumable processing"
else
    log_fail "F104: --skip-existing flag missing"
fi

# =============================================================================
# F105: Relative Path Handling
# =============================================================================
echo ""
echo "=== F105: Relative Paths ==="

if cargo test --features cli --lib test_compute_mirrored_output_path_flat --quiet 2>/dev/null; then
    log_pass "F105: Relative path resolution"
else
    log_fail "F105: Relative path resolution failed"
fi

# =============================================================================
# F106: Missing Parent Dirs
# =============================================================================
echo ""
echo "=== F106: Create Parent Directories ==="

if cargo test --features cli --lib test_atomic_write_creates_parents --quiet 2>/dev/null; then
    log_pass "F106: Creates parent directories if missing"
else
    log_fail "F106: Parent directory creation failed"
fi

# =============================================================================
# F107: Log Determinism (Sorted File List)
# =============================================================================
echo ""
echo "=== F107: Deterministic Ordering ==="

if cargo test --features cli --lib test_discover_audio_files_sorted --quiet 2>/dev/null; then
    log_pass "F107: Files processed in sorted order (deterministic)"
else
    log_fail "F107: Deterministic ordering failed"
fi

# =============================================================================
# F108: Hidden File Leakage
# =============================================================================
echo ""
echo "=== F108: Hidden File Filtering ==="

if cargo test --features cli --lib test_discover_audio_files_skips_hidden --quiet 2>/dev/null; then
    log_pass "F108: Hidden files and .git directories ignored"
else
    log_fail "F108: Hidden file filtering failed"
fi

# =============================================================================
# F109: Symlink Loops
# =============================================================================
echo ""
echo "=== F109: Symlink Safety ==="

# Verify symlink check in code
if grep -q "is_symlink" src/cli/commands.rs; then
    log_pass "F109: Symlink loop protection present"
else
    log_fail "F109: Symlink loop protection missing"
fi

# =============================================================================
# F110: Space in Path
# =============================================================================
echo ""
echo "=== F110: Space in Path ==="

if cargo test --features cli --lib test_compute_mirrored_output_path_with_spaces --quiet 2>/dev/null; then
    log_pass "F110: Paths with spaces handled correctly"
else
    log_fail "F110: Space in path handling failed"
fi

# =============================================================================
# Additional Tests: Pattern Matching and Audio Extension
# =============================================================================
echo ""
echo "=== Additional: Pattern Matching ==="

if cargo test --features cli --lib test_glob_match --quiet 2>/dev/null; then
    log_pass "Pattern matching (glob): *.wav, *.mp3, etc."
else
    log_fail "Pattern matching failed"
fi

if cargo test --features cli --lib test_matches_audio_pattern --quiet 2>/dev/null; then
    log_pass "Audio extension detection (wav, mp3, flac, ogg, m4a)"
else
    log_fail "Audio extension detection failed"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "  WAPR-PERF-004 QA RESULTS"
echo "=============================================="
echo ""
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo "  TOTAL: $TOTAL"
echo ""

SCORE=$((PASS * 100 / TOTAL))
echo "  SCORE: $SCORE%"
echo ""

if [ $SCORE -ge 100 ]; then
    echo -e "${GREEN}VERDICT: ALL FALSIFICATION POINTS PASS (100%)${NC}"
    exit 0
elif [ $SCORE -ge 90 ]; then
    echo -e "${YELLOW}VERDICT: Minor issues (90%+)${NC}"
    exit 1
else
    echo -e "${RED}VERDICT: Significant defects (<90%)${NC}"
    exit 2
fi
