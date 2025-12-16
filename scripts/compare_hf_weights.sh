#!/bin/bash
# Compare APR weights against HuggingFace source
# Usage: ./scripts/compare_hf_weights.sh [tiny|base|small] [--tensor PATTERN]
#
# This script uses apr-cli to compare whisper.apr model weights against
# the HuggingFace reference implementation, proving H6 (weight values).

set -euo pipefail

MODEL_SIZE="${1:-tiny}"
TENSOR_FILTER="${2:-}"

# Model paths
APR_MODEL="models/whisper-${MODEL_SIZE}.apr"
HF_REPO="openai/whisper-${MODEL_SIZE}"

# Check model exists
if [[ ! -f "$APR_MODEL" ]]; then
    echo "Error: Model not found: $APR_MODEL"
    echo "Convert with: cargo run --bin whisper-convert --features converter -- $MODEL_SIZE"
    exit 1
fi

echo "=== HuggingFace Weight Comparison ==="
echo "APR Model:  $APR_MODEL"
echo "HF Source:  $HF_REPO"
echo ""

# Build apr-cli if needed
APR_CLI="../aprender/target/release/apr"
if [[ ! -f "$APR_CLI" ]]; then
    echo "Building apr-cli..."
    cargo build --release -p apr-cli --manifest-path ../aprender/Cargo.toml
fi

# Run comparison
if [[ -n "$TENSOR_FILTER" ]]; then
    "$APR_CLI" compare-hf "$APR_MODEL" --hf "$HF_REPO" --tensor "$TENSOR_FILTER"
else
    "$APR_CLI" compare-hf "$APR_MODEL" --hf "$HF_REPO"
fi
