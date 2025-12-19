# Translation

This example demonstrates speech-to-text translation (any language to English) using whisper.apr.

## Running the Example

```bash
# CLI translation
whisper-apr translate -f german_speech.wav --output-file english.txt

# Rust example
cargo run --example basic_transcription
```

## Code Overview

```rust
use whisper_apr::{Task, TranscribeOptions, WhisperApr};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    let model_bytes = std::fs::read("models/whisper-tiny.apr")?;
    let whisper = WhisperApr::load_from_apr(&model_bytes)?;

    // Load audio (German speech)
    let audio = whisper_apr::audio::load_audio("german_speech.wav")?;

    // Translate to English
    let options = TranscribeOptions {
        language: Some("de".to_string()), // Source language
        task: Task::Translate,            // Translate to English
        ..Default::default()
    };

    let result = whisper.transcribe(&audio, options)?;
    println!("English translation: {}", result.text);

    Ok(())
}
```

## Task Types

| Task | Description |
|------|-------------|
| `Task::Transcribe` | Output text in source language |
| `Task::Translate` | Translate to English |

## Supported Source Languages

Whisper can translate from 99 languages to English:

- European: Spanish, French, German, Italian, Portuguese, Russian, Polish, Dutch, etc.
- Asian: Chinese, Japanese, Korean, Vietnamese, Thai, Indonesian, etc.
- Middle Eastern: Arabic, Hebrew, Turkish, Persian, etc.
- Indian: Hindi, Tamil, Telugu, Bengali, etc.

## CLI Usage

```bash
# Basic translation
whisper-apr translate -f audio.wav

# With explicit source language
whisper-apr translate -f audio.wav --language de

# Save to file
whisper-apr translate -f audio.wav --output-file translation.txt

# With timestamps
whisper-apr translate -f audio.wav --timestamps --format srt
```

## Auto-Detection with Translation

When source language is not specified, Whisper auto-detects:

```rust
let options = TranscribeOptions {
    language: None,        // Auto-detect source
    task: Task::Translate, // Still translate to English
    ..Default::default()
};
```

## Quality Considerations

Translation quality varies by:

1. **Source language**: Higher-resource languages (Spanish, French, German) translate better
2. **Audio quality**: Clear audio improves accuracy
3. **Domain**: General content works better than specialized terminology
4. **Model size**: Larger models (base, small) improve translation quality

## See Also

- [CLI translate command](../getting-started/cli.md#translate)
- [Basic Transcription](./basic-transcription.md)
- [Language Detection](./language-detection.md)
