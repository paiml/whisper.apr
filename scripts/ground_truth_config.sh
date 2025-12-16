#!/bin/bash
# Ground Truth Testing - Swappable Configuration
# Usage: source scripts/ground_truth_config.sh [tiny|base|small]
# shellcheck disable=SC2032,SC2311  # Intended to be sourced

set -euo pipefail

MODEL_SIZE="${1:-tiny}"

case "$MODEL_SIZE" in
  tiny)
    WAPR_MODEL='models/whisper-tiny.apr'
    WCPP_MODEL='/home/noah/src/whisper.cpp/models/ggml-tiny.bin'
    HF_MODEL='openai/whisper-tiny'
    EXPECTED_PARAMS='39M'
    ;;
  base)
    WAPR_MODEL='models/whisper-base.apr'
    WCPP_MODEL='/home/noah/src/whisper.cpp/models/ggml-base.en.bin'
    HF_MODEL='openai/whisper-base.en'
    EXPECTED_PARAMS='74M'
    ;;
  small)
    WAPR_MODEL='models/whisper-small.apr'
    WCPP_MODEL='/home/noah/src/whisper.cpp/models/ggml-small.bin'
    HF_MODEL='openai/whisper-small'
    EXPECTED_PARAMS='244M'
    ;;
  *)
    echo "Unknown model size: $MODEL_SIZE" >&2
    echo "Usage: source ground_truth_config.sh [tiny|base|small]" >&2
    false
    ;;
esac

TEST_AUDIO='demos/test-audio/test-speech-1.5s.wav'

export WAPR_MODEL WCPP_MODEL HF_MODEL EXPECTED_PARAMS TEST_AUDIO MODEL_SIZE

echo "=== Ground Truth Config ==="
echo "Model Size:    $MODEL_SIZE ($EXPECTED_PARAMS)"
echo "whisper.apr:   $WAPR_MODEL"
echo "whisper.cpp:   $WCPP_MODEL"
echo "HuggingFace:   $HF_MODEL"
echo "Test Audio:    $TEST_AUDIO"
echo "==========================="
