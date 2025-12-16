# Test Audio Files

Test audio samples for performance benchmarking and debugging whisper.apr.

## Files

| File | Duration | Description |
|------|----------|-------------|
| `test-speech-1.5s.wav` | 1.5s | Quick test - real speech |
| `test-speech-3s.wav` | 3s | Standard test - real speech |
| `test-speech-full.wav` | 33.6s | Full sample - real speech |
| `test-tone-2s.wav` | 2s | Synthetic 440Hz tone |

## Format

All files are:
- **Sample rate**: 16000 Hz (native for Whisper)
- **Channels**: Mono
- **Bit depth**: 16-bit PCM
- **Format**: WAV (RIFF)

## Usage

### Browser Demo Testing

Copy to www directory for browser testing:
```bash
cp test-speech-3s.wav ../www/test-audio.wav
```

### Rust Benchmark

```rust
let audio = std::fs::read("demos/test-audio/test-speech-3s.wav")?;
let samples = parse_wav(&audio)?;
let result = model.transcribe(&samples, Default::default())?;
```

### Performance Baseline

Expected performance on whisper-tiny-int8:

| Audio | Encode | Decode | Total | RTF |
|-------|--------|--------|-------|-----|
| 1.5s | ~200ms | ~2-4s | ~2-4s | ~1.5-2.5x |
| 3s | ~400ms | ~3-6s | ~3-6s | ~1-2x |

RTF = Real-Time Factor (processing time / audio duration)
Target: RTF < 2.0x for real-time transcription.

## Source

Speech samples from public domain sources (OSR Open Speech Repository).
