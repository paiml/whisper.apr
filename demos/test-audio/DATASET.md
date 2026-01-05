# Test Audio Dataset Documentation

## Overview

This directory contains audio files used for testing and benchmarking whisper.apr.
All audio files are in a format compatible with Whisper's preprocessing pipeline.

## Dataset Provenance

| File | Source | License | Ground Truth |
|------|--------|---------|--------------|
| `test-speech-1.5s.wav` | LibriSpeech Dev Clean | CC BY 4.0 | "The birds can use" |
| `test-speech-3s.wav` | LibriSpeech Dev Clean | CC BY 4.0 | See ground truth file |
| `silence-5s.wav` | Synthetically generated | Public Domain | "" (empty) |

## Audio Specifications

All audio files are normalized to Whisper's input format:

- **Sample Rate**: 16,000 Hz (16kHz)
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit signed integer (WAV) or 32-bit float
- **Format**: RIFF WAVE (PCM)

## Checksums (SHA-256)

```
# Verify dataset integrity
sha256sum demos/test-audio/*.wav

# Expected checksums:
# test-speech-1.5s.wav: [computed on first use]
# test-speech-3s.wav:   [computed on first use]
# silence-5s.wav:       [computed on first use]
```

## Ground Truth Transcriptions

Ground truth transcriptions are validated against:
1. **whisper.cpp** (C++ reference implementation)
2. **HuggingFace Transformers** (Python reference)

See `test_data/whisper_cpp_output.txt` for detailed reference outputs.

### Primary Test File: test-speech-1.5s.wav

- **Duration**: ~1.5 seconds
- **Ground Truth**: "The birds can use"
- **Speaker**: Female, American English
- **Quality**: Clean recording, no background noise
- **Use Case**: Primary unit test, RTF benchmark

### Silence Test File: silence-5s.wav

- **Duration**: 5.0 seconds
- **Ground Truth**: "" (empty string) or "[BLANK_AUDIO]"
- **Use Case**: VAD testing, no-speech detection

## Reproducibility

To regenerate or verify the dataset:

```bash
# Extract from LibriSpeech
# File: 1995-1826-0003.flac from dev-clean

# Convert to Whisper format
ffmpeg -i input.flac -ar 16000 -ac 1 -f wav output.wav

# Verify sample rate
soxi output.wav
```

## Usage in Tests

```rust
// Load test audio for unit tests
const TEST_AUDIO: &str = "demos/test-audio/test-speech-1.5s.wav";
const EXPECTED_OUTPUT: &str = "The birds can use";

#[test]
fn test_transcription() {
    let audio = load_wav(TEST_AUDIO);
    let result = model.transcribe(&audio);
    assert!(result.text.contains(EXPECTED_OUTPUT));
}
```

## Quality Control

Audio files are validated by:
- Correct sample rate (16kHz)
- Mono channel
- No clipping (peak < 0.99)
- Correct duration
- Matching ground truth
