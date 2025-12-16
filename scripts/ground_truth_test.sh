#!/bin/bash
# Ground Truth 3-Column Comparison with Swappable Models
# Usage: ./scripts/ground_truth_test.sh [tiny|base|small]
#
# bashrs lint: SC2154 warnings expected (vars from sourced config)
# bashrs lint: SC2140 warnings expected (heredoc var expansion is intentional)

set -euo pipefail

MODEL_SIZE="${1:-tiny}"

# Load config (sets WAPR_MODEL, WCPP_MODEL, HF_MODEL, TEST_AUDIO)
# shellcheck source=ground_truth_config.sh
source "$(dirname "$0")/ground_truth_config.sh" "$MODEL_SIZE"

echo ""
echo "Running transcription tests..."
echo ""

# Check if model file exists
check_model() {
  [[ -f "$1" ]]
}

echo "[1/3] whisper.apr ($MODEL_SIZE):"
if check_model "$WAPR_MODEL"; then
  cargo run --release --bin whisper-apr-cli --features cli -- transcribe \
    --model-path "$WAPR_MODEL" -q "$TEST_AUDIO" 2>/dev/null || echo "FAILED"
else
  echo "Model not found: $WAPR_MODEL"
  echo "Convert with: cargo run --bin whisper-convert --features converter -- $MODEL_SIZE"
fi
echo ""

echo "[2/3] whisper.cpp ($MODEL_SIZE):"
if check_model "$WCPP_MODEL"; then
  /home/noah/.local/bin/main -m "$WCPP_MODEL" -f "$TEST_AUDIO" 2>/dev/null || echo "FAILED"
else
  echo "Model not found: $WCPP_MODEL"
fi
echo ""

echo "[3/3] HuggingFace ($MODEL_SIZE):"
uv run --with openai-whisper --with torch - <<PYTHON
import whisper, json, time
model = whisper.load_model("${MODEL_SIZE}")
start = time.perf_counter()
result = model.transcribe("${TEST_AUDIO}", language="en", fp16=False)
elapsed = (time.perf_counter() - start) * 1000
print(json.dumps({"text": result["text"].strip(), "time_ms": round(elapsed, 2), "model": "${MODEL_SIZE}"}))
PYTHON
echo ""

echo "======================================================================"
echo "Compare outputs above. If whisper.apr differs, check Step 20 (cross-attn)"
echo "======================================================================"
