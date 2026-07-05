#!/bin/bash
set -e

echo "Starting Benchmark: whisper.apr vs whisper.cpp vs whisper (python)"
echo "--------------------------------------------------------------"

# Use existing test audio
if [ ! -f "test_jfk.wav" ]; then
    echo "Using local sample audio..."
    cp demos/www/test-audio.wav test_jfk.wav
fi

# 1. Benchmark whisper.cpp
echo "1. Benchmarking whisper.cpp (tiny.en)"
cd /home/noah/src/whisper.cpp
if [ ! -f "main" ]; then
    make
fi
if [ ! -f "models/ggml-tiny.en.bin" ]; then
    bash models/download-ggml-model.sh tiny.en
fi
time ./main -m models/ggml-tiny.en.bin -f /home/noah/src/whisper.apr/test_jfk.wav -t 8 > /dev/null 2>&1
cd /home/noah/src/whisper.apr

# 2. Benchmark whisper (python CLI)
echo "2. Benchmarking OpenAI whisper (tiny.en)"
time whisper test_jfk.wav --model tiny.en --output_dir . > /dev/null 2>&1

# 3. Benchmark whisper.apr
echo "3. Benchmarking whisper.apr (tiny.en)"
if [ ! -d "models" ]; then
    mkdir models
fi
# Ensure we have the converted APR model
if [ ! -f "models/tiny.apr" ]; then
    echo "Converting tiny to APR format (f16)..."
    cargo run --release --features="converter" --bin whisper-convert -- tiny -q f16 -o models/tiny.apr
fi
# Run inference
cargo build --release --bin whisper-apr
time ./target/release/whisper-apr transcribe --model-path models/tiny.apr -f test_jfk.wav -t 8 > /dev/null 2>&1

echo "--------------------------------------------------------------"
echo "Benchmark completed."
