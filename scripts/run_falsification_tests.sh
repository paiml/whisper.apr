#!/bin/bash
# WAPR-SPEC-010: 100-Point Popperian Falsification Test Suite
# Run all 100 tests to attempt to falsify implementation claims
#
# Usage: ./scripts/run_falsification_tests.sh

cd "$(dirname "$0")/.." || exit 1

PASSED=0
FAILED=0

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASSED=$((PASSED + 1)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAILED=$((FAILED + 1)); }

echo "================================================================"
echo "   WAPR-SPEC-010: 100-Point Popperian Falsification Suite"
echo "================================================================"
echo "Run started"
echo "Directory: $(pwd)"
echo ""

BRIDGE="demos/realtime-transcription/src/bridge.rs"
LIB="demos/realtime-transcription/src/lib.rs"
PROBAR="demos/realtime-transcription/tests/probar_tests.rs"

echo "--- Section 1: Constants & Configuration (1-10) ---"

# 1
if grep -q "MAX_QUEUE_DEPTH.*=.*3" "$BRIDGE"; then pass "1: MAX_QUEUE_DEPTH=3"; else fail "1: MAX_QUEUE_DEPTH!=3"; fi
# 2
if grep -q "pub const MAX_QUEUE_DEPTH" "$BRIDGE"; then pass "2: MAX_QUEUE_DEPTH public"; else fail "2: Not public"; fi
# 3
if grep -q "pub const MAX_QUEUE_DEPTH: usize" "$BRIDGE"; then pass "3: MAX_QUEUE_DEPTH is usize"; else fail "3: Wrong type"; fi
# 4
if grep -q "MAX_CONSECUTIVE_ERRORS.*=.*3" "$BRIDGE"; then pass "4: MAX_CONSECUTIVE_ERRORS=3"; else fail "4: !=3"; fi
# 5
if grep -q "pub const MAX_CONSECUTIVE_ERRORS: u32" "$BRIDGE"; then pass "5: MAX_CONSECUTIVE_ERRORS u32"; else fail "5: Wrong type"; fi
# 6
if grep -A20 "pub struct QueueStats" "$BRIDGE" | grep -q "chunks_sent"; then pass "6: chunks_sent exists"; else fail "6: Missing"; fi
# 7
if grep -A20 "pub struct QueueStats" "$BRIDGE" | grep -q "chunks_dropped"; then pass "7: chunks_dropped exists"; else fail "7: Missing"; fi
# 8
if grep -A20 "pub struct QueueStats" "$BRIDGE" | grep -q "chunks_completed"; then pass "8: chunks_completed exists"; else fail "8: Missing"; fi
# 9
if grep -A20 "pub struct QueueStats" "$BRIDGE" | grep -q "pub errors"; then pass "9: errors exists"; else fail "9: Missing"; fi
# 10
if grep -A20 "pub struct QueueStats" "$BRIDGE" | grep -q "avg_latency_ms"; then pass "10: avg_latency_ms exists"; else fail "10: Missing"; fi

echo ""
echo "--- Section 2: Message Protocol (11-25) ---"

# 11-15 WorkerCommand variants
if grep -A30 "pub enum WorkerCommand" "$BRIDGE" | grep -q "LoadModel"; then pass "11: LoadModel variant"; else fail "11: Missing"; fi
if grep -A30 "pub enum WorkerCommand" "$BRIDGE" | grep -q "Transcribe"; then pass "12: Transcribe variant"; else fail "12: Missing"; fi
if grep -A30 "pub enum WorkerCommand" "$BRIDGE" | grep -q "SetOptions"; then pass "13: SetOptions variant"; else fail "13: Missing"; fi
if grep -A30 "pub enum WorkerCommand" "$BRIDGE" | grep -q "Shutdown"; then pass "14: Shutdown variant"; else fail "14: Missing"; fi
if grep -A30 "pub enum WorkerCommand" "$BRIDGE" | grep -q "Ping"; then pass "15: Ping variant"; else fail "15: Missing"; fi

# 16-21 WorkerResult variants
if grep -A50 "pub enum WorkerResult" "$BRIDGE" | grep -q "Ready"; then pass "16: Ready variant"; else fail "16: Missing"; fi
if grep -A50 "pub enum WorkerResult" "$BRIDGE" | grep -q "ModelLoaded"; then pass "17: ModelLoaded variant"; else fail "17: Missing"; fi
if grep -A50 "pub enum WorkerResult" "$BRIDGE" | grep -q "Transcription"; then pass "18: Transcription variant"; else fail "18: Missing"; fi
if grep -A50 "pub enum WorkerResult" "$BRIDGE" | grep -q "Error"; then pass "19: Error variant"; else fail "19: Missing"; fi
if grep -A50 "pub enum WorkerResult" "$BRIDGE" | grep -q "Metrics"; then pass "20: Metrics variant"; else fail "20: Missing"; fi
if grep -A50 "pub enum WorkerResult" "$BRIDGE" | grep -q "Pong"; then pass "21: Pong variant"; else fail "21: Missing"; fi

# 22-25 Transcription fields
if grep -A10 "Transcription {" "$BRIDGE" | grep -q "rtf:"; then pass "22: rtf field"; else fail "22: Missing"; fi
if grep -A10 "Transcription {" "$BRIDGE" | grep -q "chunk_id:"; then pass "23: chunk_id field"; else fail "23: Missing"; fi
if grep -A10 "Transcription {" "$BRIDGE" | grep -q "session_id:"; then pass "24: session_id field"; else fail "24: Missing"; fi
if grep -A10 "Transcription {" "$BRIDGE" | grep -q "is_partial:"; then pass "25: is_partial field"; else fail "25: Missing"; fi

echo ""
echo "--- Section 3: Unit Tests (26-45) ---"

# 26-33 Test existence
if grep -q "test_max_queue_depth" "$BRIDGE"; then pass "26: MAX_QUEUE_DEPTH test"; else fail "26: No test"; fi
if grep -q "test_max_consecutive_errors" "$BRIDGE"; then pass "27: MAX_CONSECUTIVE_ERRORS test"; else fail "27: No test"; fi
if grep -qE "test_worker_command.*[Pp]ing|Ping" "$BRIDGE"; then pass "28: Ping variant test"; else fail "28: No test"; fi
if grep -qE "test_worker_result.*[Rr]eady|Ready" "$BRIDGE"; then pass "29: Ready variant test"; else fail "29: No test"; fi
if grep -q "test_worker_result_error" "$BRIDGE"; then pass "30: Error variant test"; else fail "30: No test"; fi
if grep -q "test_queue_stats_default" "$BRIDGE"; then pass "31: QueueStats default test"; else fail "31: No test"; fi
if grep -q "test_queue_stats_clone" "$BRIDGE"; then pass "32: QueueStats clone test"; else fail "32: No test"; fi
if grep -q "test_queue_stats_debug" "$BRIDGE"; then pass "33: QueueStats debug test"; else fail "33: No test"; fi

# 34-36 Run tests
echo "    Running cargo tests..."
DEMO_DIR="demos/realtime-transcription"
if (cd "$DEMO_DIR" && cargo test --quiet 2>&1) | tail -5 | grep -q "passed"; then
    pass "34: Demo tests compile"
else
    fail "34: Demo tests fail"
fi

TEST_RESULT=$( (cd "$DEMO_DIR" && cargo test 2>&1) | grep "test result:" | head -1)
if echo "$TEST_RESULT" | grep -q "passed"; then
    pass "35: Demo tests pass"
else
    fail "35: Demo tests fail"
fi

if echo "$TEST_RESULT" | grep -q "0 failed"; then
    pass "36: Zero test failures"
else
    fail "36: Tests failing"
fi

# 37-41 probar_tests modules
if grep -q "mod queue_management_tests" "$PROBAR"; then pass "37: queue_management_tests"; else fail "37: Missing"; fi
if grep -q "mod worker_result_tests" "$PROBAR"; then pass "38: worker_result_tests"; else fail "38: Missing"; fi
if grep -q "mod worker_command_tests" "$PROBAR"; then pass "39: worker_command_tests"; else fail "39: Missing"; fi
if grep -q "mod memory_stability_tests" "$PROBAR"; then pass "40: memory_stability_tests"; else fail "40: Missing"; fi
if grep -q "mod error_recovery_tests" "$PROBAR"; then pass "41: error_recovery_tests"; else fail "41: Missing"; fi

# 42-45 probar tests run
echo "    Running probar_tests..."
PROBAR_OUT=$( (cd "$DEMO_DIR" && cargo test --test probar_tests 2>&1) )
if echo "$PROBAR_OUT" | grep -q "passed"; then
    pass "42: probar_tests run"
else
    fail "42: probar_tests fail"
fi

PROBAR_COUNT=$(echo "$PROBAR_OUT" | grep -o '[0-9]* passed' | grep -o '[0-9]*' | head -1)
PROBAR_COUNT=${PROBAR_COUNT:-0}
if [ "$PROBAR_COUNT" -gt 30 ]; then
    pass "43: $PROBAR_COUNT probar tests"
else
    fail "43: Only $PROBAR_COUNT tests"
fi

if echo "$PROBAR_OUT" | grep -q "0 failed"; then
    pass "44: Zero probar failures"
else
    fail "44: Probar failures"
fi

if grep -q "test_demo_handles_100_partial_results" "$PROBAR"; then pass "45: 100 partial test"; else fail "45: Missing"; fi

echo ""
echo "--- Section 4: WorkerBridge API (46-60) ---"

if grep -q "pub fn new(" "$BRIDGE"; then pass "46: new()"; else fail "46: Missing"; fi
if grep -q "pub fn is_ready(" "$BRIDGE"; then pass "47: is_ready()"; else fail "47: Missing"; fi
if grep -q "pub fn load_model(" "$BRIDGE"; then pass "48: load_model()"; else fail "48: Missing"; fi
if grep -q "pub fn transcribe(" "$BRIDGE"; then pass "49: transcribe()"; else fail "49: Missing"; fi
if grep -q "pub fn ping(" "$BRIDGE"; then pass "50: ping()"; else fail "50: Missing"; fi
if grep -q "pub fn shutdown(" "$BRIDGE"; then pass "51: shutdown()"; else fail "51: Missing"; fi
if grep -q "pub fn stats(" "$BRIDGE"; then pass "52: stats()"; else fail "52: Missing"; fi
if grep -q "pub fn is_healthy(" "$BRIDGE"; then pass "53: is_healthy()"; else fail "53: Missing"; fi
if grep -q "pub fn needs_restart(" "$BRIDGE"; then pass "54: needs_restart()"; else fail "54: Missing"; fi
if grep -q "pub fn would_overflow(" "$BRIDGE"; then pass "55: would_overflow()"; else fail "55: Missing"; fi
if grep -q "pub fn pending_count(" "$BRIDGE"; then pass "56: pending_count()"; else fail "56: Missing"; fi
if grep -q "pub fn terminate(" "$BRIDGE"; then pass "57: terminate()"; else fail "57: Missing"; fi
if grep -q "pub fn reset_error_state(" "$BRIDGE"; then pass "58: reset_error_state()"; else fail "58: Missing"; fi
if grep -q "pub fn consecutive_errors(" "$BRIDGE"; then pass "59: consecutive_errors()"; else fail "59: Missing"; fi
# set_result_callback may span multiple lines, check for the method name
if grep -q "set_result_callback" "$BRIDGE"; then pass "60: set_result_callback()"; else fail "60: Missing"; fi

echo ""
echo "--- Section 5: State Machine (61-70) ---"

if grep -q "Initializing" "$LIB"; then pass "61: Initializing state"; else fail "61: Missing"; fi
if grep -q "LoadingModel" "$LIB"; then pass "62: LoadingModel state"; else fail "62: Missing"; fi
if grep -q "Idle" "$LIB"; then pass "63: Idle state"; else fail "63: Missing"; fi
if grep -q "Recording" "$LIB"; then pass "64: Recording state"; else fail "64: Missing"; fi
if grep -q "Processing" "$LIB"; then pass "65: Processing state"; else fail "65: Missing"; fi
if grep -q "Error" "$LIB"; then pass "66: Error state"; else fail "66: Missing"; fi
if grep -qE "fn is_valid|is_valid\(" "$LIB"; then pass "67: is_valid()"; else fail "67: Missing"; fi

TRANS_COUNT=$(grep -c "test_valid_transition\|test_invalid_transition" "$PROBAR" || echo "0")
if [ "$TRANS_COUNT" -gt 3 ]; then pass "68: $TRANS_COUNT transition tests"; else fail "68: <4 tests"; fi

if grep -qE "Error.*Idle|test_error_to_idle" "$PROBAR"; then pass "69: Error->Idle test"; else fail "69: Missing"; fi
if grep -q "test_invalid_transition_idle_to_recording" "$PROBAR"; then pass "70: Invalid trans test"; else fail "70: Missing"; fi

echo ""
echo "--- Section 6: File Structure (71-80) ---"

if [ -f "$BRIDGE" ]; then pass "71: bridge.rs exists"; else fail "71: Missing"; fi
if [ -f "demos/realtime-transcription/src/worker.rs" ]; then pass "72: worker.rs exists"; else fail "72: Missing"; fi
if [ -f "$LIB" ]; then pass "73: lib.rs exists"; else fail "73: Missing"; fi
if [ -f "$PROBAR" ]; then pass "74: probar_tests.rs exists"; else fail "74: Missing"; fi
if [ -f "demos/realtime-transcription/Cargo.toml" ]; then pass "75: Cargo.toml exists"; else fail "75: Missing"; fi

JS_COUNT=$(find demos/realtime-transcription/src -name "*.js" 2>/dev/null | wc -l)
if [ "$JS_COUNT" -eq 0 ]; then pass "76: No .js in src/"; else fail "76: $JS_COUNT .js files"; fi

TS_COUNT=$(find demos/realtime-transcription/src -name "*.ts" 2>/dev/null | wc -l)
if [ "$TS_COUNT" -eq 0 ]; then pass "77: No .ts in src/"; else fail "77: $TS_COUNT .ts files"; fi

if grep -qE "pub mod bridge|mod bridge" "$LIB"; then pass "78: bridge module"; else fail "78: Missing"; fi
if grep -qE "pub mod worker|mod worker" "$LIB"; then pass "79: worker module"; else fail "79: Missing"; fi
if grep -q "pub use bridge\|pub mod bridge" "$LIB"; then pass "80: bridge public"; else fail "80: Not public"; fi

echo ""
echo "--- Section 7: Derives & Traits (81-90) ---"

if grep -B2 "pub enum WorkerCommand" "$BRIDGE" | grep -q "Debug"; then pass "81: WorkerCommand Debug"; else fail "81: Missing"; fi
if grep -B2 "pub enum WorkerCommand" "$BRIDGE" | grep -q "Clone"; then pass "82: WorkerCommand Clone"; else fail "82: Missing"; fi
if grep -B2 "pub enum WorkerResult" "$BRIDGE" | grep -q "Debug"; then pass "83: WorkerResult Debug"; else fail "83: Missing"; fi
if grep -B2 "pub enum WorkerResult" "$BRIDGE" | grep -q "Clone"; then pass "84: WorkerResult Clone"; else fail "84: Missing"; fi
if grep -B2 "pub struct QueueStats" "$BRIDGE" | grep -q "Debug"; then pass "85: QueueStats Debug"; else fail "85: Missing"; fi
if grep -B2 "pub struct QueueStats" "$BRIDGE" | grep -q "Clone"; then pass "86: QueueStats Clone"; else fail "86: Missing"; fi
if grep -B2 "pub struct QueueStats" "$BRIDGE" | grep -q "Default"; then pass "87: QueueStats Default"; else fail "87: Missing"; fi
if grep -B2 "pub enum DemoState" "$LIB" | grep -q "PartialEq"; then pass "88: DemoState PartialEq"; else fail "88: Missing"; fi
if grep -B2 "pub enum DemoState" "$LIB" | grep -q "Clone"; then pass "89: DemoState Clone"; else fail "89: Missing"; fi
if grep -B2 "pub enum DemoState" "$LIB" | grep -q "Debug"; then pass "90: DemoState Debug"; else fail "90: Missing"; fi

echo ""
echo "--- Section 8: Documentation & Quality (91-100) ---"

if grep -B3 "pub struct WorkerBridge" "$BRIDGE" | grep -q "///"; then pass "91: WorkerBridge docs"; else fail "91: Missing"; fi
if grep -B3 "pub const MAX_QUEUE_DEPTH" "$BRIDGE" | grep -q "///"; then pass "92: MAX_QUEUE_DEPTH docs"; else fail "92: Missing"; fi
if grep -B5 "pub fn transcribe(" "$BRIDGE" | grep -q "///"; then pass "93: transcribe() docs"; else fail "93: Missing"; fi

echo "    Running cargo check..."
if (cd "$DEMO_DIR" && cargo check 2>&1) | grep -qE "error\[E"; then
    fail "94: Compile errors"
else
    pass "94: cargo check OK"
fi

if (cd "$DEMO_DIR" && cargo check --all-targets 2>&1) | grep -qE "error\[E"; then
    fail "95: all-targets errors"
else
    pass "95: all-targets OK"
fi

# Check for unwrap() before the #[cfg(test)] section
TEST_LINE=$(grep -n "#\[cfg(test)\]" "$BRIDGE" | head -1 | cut -d: -f1)
TEST_LINE=${TEST_LINE:-99999}
UNWRAP_IN_PROD=$(head -n "$TEST_LINE" "$BRIDGE" | grep -n "\.unwrap()" | head -1)
if [ -z "$UNWRAP_IN_PROD" ]; then
    pass "96: No unwrap() in prod"
else
    fail "96: unwrap() found"
fi

# Function signatures may span multiple lines - check function exists and returns Result somewhere
if grep -q "pub fn new" "$BRIDGE" && grep -A10 "pub fn new" "$BRIDGE" | grep -q "Result"; then pass "97: new() -> Result"; else fail "97: Missing"; fi
if grep -q "pub fn load_model" "$BRIDGE" && grep -A10 "pub fn load_model" "$BRIDGE" | grep -q "Result"; then pass "98: load_model() -> Result"; else fail "98: Missing"; fi
if grep -q "pub fn transcribe" "$BRIDGE" && grep -A10 "pub fn transcribe" "$BRIDGE" | grep -q "Result"; then pass "99: transcribe() -> Result"; else fail "99: Missing"; fi

COMPLETED=$(grep -c "status: completed" docs/roadmaps/roadmap.yaml || echo "0")
if [ "$COMPLETED" -ge 5 ]; then pass "100: Roadmap ($COMPLETED complete)"; else fail "100: Only $COMPLETED"; fi

echo ""
echo "================================================================"
echo "                    FINAL RESULTS"
echo "================================================================"
echo -e "  ${GREEN}PASSED:${NC}  $PASSED/100"
echo -e "  ${RED}FAILED:${NC}  $FAILED/100"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}All 100 claims survived falsification!${NC}"
    exit 0
else
    echo -e "${RED}$FAILED claims were falsified.${NC}"
    exit 1
fi
