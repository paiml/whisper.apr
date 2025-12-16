# WAPR-GROUND-TRUTH-001: Pipeline Ground Truth Verification

## Overview

Stop-the-line approach to debug transcription failure. Instead of ad-hoc debugging,
create systematic ground truth tests comparing each pipeline step against known-good
implementations (whisper.cpp, HuggingFace transformers).

## Inspiration

Similar to how trueno and bashrs achieved success with visual simulation of pipeline
state vs ground truth alternatives.

## Pipeline Steps to Verify

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          WHISPER PIPELINE                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │  A      │    │  B      │    │  C      │    │  D      │    │  E      │       │
│  │ Audio   │───▶│ STFT    │───▶│ Mel     │───▶│ Conv1   │───▶│ Conv2   │       │
│  │ Input   │    │ FFT     │    │ Filter  │    │ Frontend│    │ Stride  │       │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘       │
│       │              │              │              │              │             │
│       ▼              ▼              ▼              ▼              ▼             │
│    [16kHz]      [spectrogram]   [80 mels]    [384 dims]     [384 dims]         │
│    [480000]      [201×3000]     [3000×80]    [3000×384]     [1500×384]         │
│                                                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │  F      │    │  G      │    │  H      │    │  I      │    │  J      │       │
│  │ Pos     │───▶│ Encoder │───▶│ Cross   │───▶│ Decoder │───▶│ Logits  │       │
│  │ Embed   │    │ Blocks  │    │ Attn K  │    │ Blocks  │    │ Softmax │       │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘       │
│       │              │              │              │              │             │
│       ▼              ▼              ▼              ▼              ▼             │
│  [1500×384]     [1500×384]     [1500×384]    [seq×384]      [seq×vocab]        │
│  (sinusoidal)   (transformer)  (cached K)   (self+cross)   (probability)      │
│                                                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                                     │
│  │  K      │    │  L      │    │  M      │                                     │
│  │ Token   │───▶│ Detok   │───▶│ Text    │                                     │
│  │ Sample  │    │ enize   │    │ Output  │                                     │
│  └─────────┘    └─────────┘    └─────────┘                                     │
│       │              │              │                                           │
│       ▼              ▼              ▼                                           │
│   [token_id]    ["Hello"]    ["Hello world"]                                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Ground Truth Sources

### Primary: whisper.cpp
- C++ reference implementation
- Produces correct transcriptions
- Can dump intermediate values at each step

### Secondary: HuggingFace Transformers
- Python reference with `openai/whisper-tiny`
- Easy to extract intermediate activations
- Use for validation when whisper.cpp values unavailable

## Visual Simulation Status

```
Step │ Name          │ Status │ Our Value      │ Ground Truth   │ Delta
─────┼───────────────┼────────┼────────────────┼────────────────┼─────────
  A  │ Audio Input   │   ✓    │ [480000]       │ [480000]       │ 0.00%
  B  │ STFT          │   ?    │ std=???        │ std=???        │ ???
  C  │ Mel Filter    │   ?    │ [3000×80]      │ [3000×80]      │ ???
  D  │ Conv1 Out     │   ?    │ std=???        │ std=???        │ ???
  E  │ Conv2 Out     │   ?    │ std=???        │ std=???        │ ???
  F  │ +Pos Embed    │   ?    │ std=???        │ std=???        │ ???
  G  │ Encoder Out   │   ✓    │ std=1.3        │ std=???        │ ???
  H  │ Cross-K       │   ✓    │ std=0.88       │ std=???        │ ???
  I  │ Decoder State │   ✓    │ std=1.6        │ std=???        │ ???
  J  │ Logits        │   ?    │ ???            │ ???            │ ???
  K  │ Token Sample  │   ✗    │ padding pos    │ audio pos      │ WRONG
  L  │ Detokenize    │   ?    │ ???            │ ???            │ ???
  M  │ Text Output   │   ✗    │ gibberish      │ "Hello world"  │ FAIL
```

## Test Implementation Plan

### Phase 1: Extract Ground Truth Data
1. Run whisper.cpp with test audio, dump intermediate values
2. Run HuggingFace Python with same audio, extract activations
3. Save as JSON/binary reference files in `test_data/ground_truth/`

### Phase 2: Create Probar Step Tests
Each step gets a probar test that:
1. Loads test audio
2. Runs pipeline up to that step
3. Compares output against ground truth
4. Reports delta as percentage difference

### Phase 3: Visual Pipeline TUI
Create a TUI showing:
- Pipeline flow diagram
- Current step status (✓/✗/?)
- Value comparisons vs ground truth
- Delta highlighting (green=match, red=mismatch)

## File Structure

```
test_data/
├── ground_truth/
│   ├── step_a_audio.bin          # Raw audio samples
│   ├── step_b_stft.json          # STFT output stats
│   ├── step_c_mel.bin            # Mel spectrogram [3000×80]
│   ├── step_c_mel_stats.json     # Mean, std, range
│   ├── step_d_conv1.json         # Conv1 output stats
│   ├── step_e_conv2.json         # Conv2 output stats
│   ├── step_f_posembed.json      # After positional embedding
│   ├── step_g_encoder.bin        # Encoder output [1500×384]
│   ├── step_g_encoder_stats.json
│   ├── step_h_crossk.json        # Cross-attention K stats
│   ├── step_i_decoder.json       # Decoder state stats
│   ├── step_j_logits.json        # Logits stats
│   ├── step_k_tokens.json        # Token sequence
│   └── step_m_text.txt           # Final transcription
└── test_audio/
    └── test-speech-1.5s.wav      # Standard test audio
```

## Probar Test Commands

```bash
# Run all ground truth tests
cargo test --test ground_truth_tests

# Run specific step test
cargo test --test ground_truth_tests step_c_mel

# Run with visual output
cargo test --test ground_truth_tests -- --nocapture

# Generate ground truth (requires Python + whisper.cpp)
python tools/extract_ground_truth.py
```

## Success Criteria

1. Each pipeline step produces output within 1% of ground truth
2. Attention focuses on audio positions (0-75) not padding (1312+)
3. Final transcription matches ground truth text

## Current Status

**BLOCKED AT**: Step K (Token Sample) - attention prefers padding positions

**Root Cause Hypothesis Chain**:
- H6 FALSIFIED: Weights match HuggingFace exactly
- H7 FALSIFIED: Encoder output healthy (std=1.256)
- H8 FALSIFIED: K projection working (L1=1.073)
- H10 FALSIFIED: Scale factor correct (0.125)
- H16 CONFIRMED: Attention IS peaked (39.6% entropy)
- H17 CONFIRMED: Model attends to PADDING not AUDIO
- H19 CONFIRMED: Q·K alignment wrong (padding cos=+0.79, audio cos=-0.55)
- H23 INVESTIGATED: Mel transpose attempted but no effect

**Next**: Create ground truth tests to identify EXACTLY which step first diverges
from reference implementation.
