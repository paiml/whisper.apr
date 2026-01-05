# Model Files Documentation

## Overview

This directory contains Whisper model files in the .apr (Aprender) format.
Models are converted from OpenAI's original weights using the converter tool.

## Model Versioning

| Model | Version | Source | SHA-256 |
|-------|---------|--------|---------|
| whisper-tiny.apr | 1.0.0 | OpenAI Whisper tiny | TBD on release |
| whisper-tiny-int8-fb.apr | 1.0.0 | Quantized from tiny | TBD on release |
| whisper-base.apr | 1.0.0 | OpenAI Whisper base | TBD on release |

## Reproducibility

### Source Weights

Models are converted from official OpenAI Whisper weights:
- Repository: https://github.com/openai/whisper
- HuggingFace: https://huggingface.co/openai/whisper-tiny

### Conversion Process

```bash
# Download and convert official weights
cargo run --release --bin whisper-apr-cli -- convert \
  --model tiny \
  --output models/whisper-tiny.apr

# Verify conversion
cargo run --release --bin whisper-apr-cli -- verify \
  --model models/whisper-tiny.apr \
  --reference-output test_data/whisper_cpp_output.txt
```

### Deterministic Inference

Whisper.apr uses greedy decoding by default, which is fully deterministic:
- No random sampling (temperature=0)
- Fixed mel filterbank from model file
- Deterministic SIMD operations

For beam search, the search order is deterministic given the same input.

## Model Specifications

### whisper-tiny

- Parameters: 39M
- Encoder: 4 layers, 384 dim, 6 heads
- Decoder: 4 layers, 384 dim, 6 heads
- Vocabulary: 51,865 tokens
- Languages: 99 (multilingual)
- Expected RTF: < 2.0x (CPU)

### whisper-base

- Parameters: 74M
- Encoder: 6 layers, 512 dim, 8 heads
- Decoder: 6 layers, 512 dim, 8 heads
- Vocabulary: 51,865 tokens
- Languages: 99 (multilingual)
- Expected RTF: < 2.5x (CPU)

## Quality Assurance

Models are validated against ground truth:

```bash
# Run ground truth tests
cargo test ground_truth

# Expected output for test-speech-1.5s.wav:
# "The birds can use"
```

## License

Model weights are subject to OpenAI's Whisper license (MIT).
The .apr format and conversion tools are MIT licensed.
