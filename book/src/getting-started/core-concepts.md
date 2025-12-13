# Core Concepts

Understanding the fundamentals of Whisper.apr.

## The Whisper Model

Whisper is a transformer-based encoder-decoder model for automatic speech recognition (ASR). It was trained on 680,000 hours of multilingual audio data.

### Architecture Summary

```
Audio → Mel Spectrogram → Encoder → Cross-Attention → Decoder → Text
```

1. **Audio Input**: Raw audio waveform (any sample rate)
2. **Preprocessing**: Resample to 16kHz, compute 80-bin mel spectrogram
3. **Encoder**: Process audio features through transformer layers
4. **Decoder**: Generate text tokens autoregressively
5. **Output**: Transcribed text with optional timestamps

## Model Sizes

| Model | Parameters | Multilingual | English-Only |
|-------|------------|--------------|--------------|
| tiny  | 39M        | ✓            | ✓            |
| base  | 74M        | ✓            | ✓            |
| small | 244M       | ✓            | ✓            |
| medium| 769M       | ✓            | ✓            |
| large | 1.5B       | ✓            | -            |

**Whisper.apr v1.0 supports**: tiny, base, small

## Tasks

### Transcription

Convert speech to text in the original language:

```rust
let options = TranscribeOptions {
    task: Task::Transcribe,
    language: Some("es".into()),  // Spanish audio → Spanish text
    ..Default::default()
};
```

### Translation

Convert speech in any language to English:

```rust
let options = TranscribeOptions {
    task: Task::Translate,
    language: None,  // Auto-detect source language
    ..Default::default()
};
```

## Decoding Strategies

### Greedy Decoding

Select the highest probability token at each step:

```rust
let options = TranscribeOptions {
    strategy: DecodingStrategy::Greedy,
    ..Default::default()
};
```

**Pros**: Fast, memory-efficient
**Cons**: May miss better sequences

### Beam Search

Explore multiple hypotheses in parallel:

```rust
let options = TranscribeOptions {
    strategy: DecodingStrategy::BeamSearch {
        beam_size: 5,      // Number of parallel hypotheses
        temperature: 0.0,  // Deterministic (no sampling)
        patience: 1.0,     // Early stopping patience
    },
    ..Default::default()
};
```

**Pros**: Higher quality transcriptions
**Cons**: Slower, more memory

### Sampling

Sample from the probability distribution:

```rust
let options = TranscribeOptions {
    strategy: DecodingStrategy::Sampling {
        temperature: 0.8,  // Higher = more random
        top_k: Some(40),   // Consider top-k tokens
        top_p: Some(0.9),  // Nucleus sampling
    },
    ..Default::default()
};
```

**Use case**: Creative applications, multiple hypotheses

## The .apr Format

Whisper.apr uses a custom binary format optimized for web delivery:

### Structure

```
┌────────────────────────────┐
│ Magic: "APR1" (4 bytes)    │
├────────────────────────────┤
│ Version (2 bytes)          │
├────────────────────────────┤
│ Model Config               │
├────────────────────────────┤
│ Mel Filterbank             │
├────────────────────────────┤
│ Vocabulary (BPE)           │
├────────────────────────────┤
│ Encoder Weights (LZ4)      │
├────────────────────────────┤
│ Decoder Weights (LZ4)      │
├────────────────────────────┤
│ Checksum (CRC32)           │
└────────────────────────────┘
```

### Features

- **LZ4 Compression**: 2-3x size reduction
- **Streaming**: 64KB blocks for progressive loading
- **Quantization**: fp32, fp16, int8 support
- **Integrity**: CRC32 checksum

## Language Support

Whisper supports 99 languages. Specify the language or use auto-detection:

```rust
// Explicit language
let options = TranscribeOptions {
    language: Some("ja".into()),  // Japanese
    ..Default::default()
};

// Auto-detect
let options = TranscribeOptions {
    language: None,  // or Some("auto".into())
    ..Default::default()
};
```

### Language Codes

Common language codes (ISO 639-1):
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean
- `pt` - Portuguese
- `ru` - Russian
- `ar` - Arabic

## Word Timestamps

Get word-level timing information:

```rust
let options = TranscribeOptions {
    word_timestamps: true,
    ..Default::default()
};

let result = whisper.transcribe(&audio, options)?;

for segment in result.segments {
    println!("[{:.2}s - {:.2}s] {}",
        segment.start, segment.end, segment.text);
}
```

## Error Handling

All operations return `WhisperResult<T>`:

```rust
use whisper_apr::{WhisperError, WhisperResult};

fn transcribe_file(path: &str) -> WhisperResult<String> {
    let model_data = std::fs::read("model.apr")?;  // Io error
    let whisper = WhisperApr::load(&model_data)?;  // Format/Model error
    let audio = load_audio(path)?;                  // Audio error
    let result = whisper.transcribe(&audio, Default::default())?;
    Ok(result.text)
}
```

Error variants:
- `WhisperError::Audio` - Invalid audio format
- `WhisperError::Model` - Model loading/inference error
- `WhisperError::Format` - Invalid .apr format
- `WhisperError::Tokenizer` - Tokenization error
- `WhisperError::Inference` - Decoding error
