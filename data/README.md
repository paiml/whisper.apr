# Data Documentation

## Overview

This document describes the data artifacts used in whisper.apr for testing, validation, and benchmarking.

## Test Audio Files

### Location

Test audio files are located in `demos/test-audio/`:

```
demos/test-audio/
├── test-speech-1.5s.wav    # 1.5 second English speech sample
├── test-speech-3s.wav      # 3 second English speech sample
├── test-speech-5s.wav      # 5 second English speech sample
└── DATASET.md              # Detailed dataset documentation
```

### Dataset Specifications

| File | Duration | Sample Rate | Channels | Ground Truth |
|------|----------|-------------|----------|--------------|
| test-speech-1.5s.wav | 1.5s | 16kHz | Mono | "The birds can use" |
| test-speech-3s.wav | 3.0s | 16kHz | Mono | TBD |
| test-speech-5s.wav | 5.0s | 16kHz | Mono | TBD |

### Data Provenance

All test audio files are:
- Created synthetically using TTS (Text-to-Speech) for reproducibility
- Licensed under MIT for free use
- SHA-256 checksums provided for verification

See [demos/test-audio/DATASET.md](../demos/test-audio/DATASET.md) for full provenance and checksums.

## Ground Truth Reference Data

### Location

Reference outputs are stored in `test_data/`:

```
test_data/
├── ref_a_audio.json           # Audio input reference
├── ref_c_mel_numpy.json       # Mel spectrogram reference
├── reference_summary.json     # All stages summary
└── whisper_cpp_output.txt     # whisper.cpp reference output
```

### Reference Values

| Reference | Source | Statistics |
|-----------|--------|------------|
| Mel spectrogram | librosa | mean=-0.2148, std=0.4479 |
| Audio samples | WAV decode | 16kHz mono f32 |

## Model Files

### Location

Model files are stored in `models/`:

```
models/
├── whisper-tiny.apr           # Tiny model (39M params)
├── whisper-tiny-int8-fb.apr   # Tiny INT8 quantized
├── whisper-base.apr           # Base model (74M params)
├── MODEL_CARD.md              # Model card documentation
└── README.md                  # Model documentation
```

### Model Provenance

All model weights originate from:
- **Source**: OpenAI Whisper (github.com/openai/whisper)
- **License**: MIT
- **Conversion**: Direct weight conversion, no fine-tuning

See [models/README.md](../models/README.md) for checksums and conversion details.

## Data Quality Controls

### Validation

1. **Checksum verification**: All data files have SHA-256 checksums
2. **Ground truth comparison**: Outputs compared against whisper.cpp
3. **Statistical validation**: Numerical precision tests

### Reproducibility

To verify data integrity:

```bash
# Verify test audio checksums
cd demos/test-audio && sha256sum -c checksums.txt

# Run ground truth tests
cargo test ground_truth

# Compare against reference
cargo run --bin whisper-apr-cli -- parity -f demos/test-audio/test-speech-1.5s.wav
```

## Data Access

All data is included in the repository:
- No external data downloads required
- No API keys or authentication needed
- Full reproducibility from repository clone

## Related Documentation

- [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) - Full reproducibility guide
- [demos/test-audio/DATASET.md](../demos/test-audio/DATASET.md) - Detailed dataset documentation
- [models/MODEL_CARD.md](../models/MODEL_CARD.md) - Model card
