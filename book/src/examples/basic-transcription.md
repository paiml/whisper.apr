# Basic Transcription

This example demonstrates the core whisper.apr API for transcribing audio files.

## Running the Example

```bash
cargo run --example basic_transcription
```

## Code Overview

```rust
use whisper_apr::{
    audio::MelFilterbank, model::ModelConfig, DecodingStrategy, Task,
    TranscribeOptions, WhisperApr,
};

fn main() {
    // Create a tiny model (fastest, smallest)
    let config = ModelConfig::tiny();
    let whisper = WhisperApr::tiny();

    // Create transcription options
    let options = TranscribeOptions {
        language: Some("en".to_string()),
        task: Task::Transcribe,
        strategy: DecodingStrategy::Greedy,
        word_timestamps: false,
    };

    // Load and process audio
    let audio: Vec<f32> = load_audio("audio.wav");

    // Compute mel spectrogram
    let mel = MelFilterbank::new(80, 400, 16000);
    let mel_spec = mel.compute(&audio, 160)?;

    // Note: Full transcription requires loading a .apr model file
}
```

## Model Configuration

The example displays the model configuration:

| Parameter | tiny | base | small |
|-----------|------|------|-------|
| Encoder layers | 4 | 6 | 12 |
| Decoder layers | 4 | 6 | 12 |
| Audio state dim | 384 | 512 | 768 |
| Attention heads | 6 | 8 | 12 |
| Vocabulary size | 51,865 | 51,865 | 51,865 |

## Transcription Options

- **language**: Source language (ISO 639-1 code) or `None` for auto-detection
- **task**: `Task::Transcribe` or `Task::Translate`
- **strategy**: `DecodingStrategy::Greedy` or `DecodingStrategy::BeamSearch { beam_size }`
- **word_timestamps**: Enable word-level timing

## Full Example with Model Loading

```rust
use whisper_apr::WhisperApr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model from .apr file
    let model_bytes = std::fs::read("models/whisper-tiny.apr")?;
    let whisper = WhisperApr::load_from_apr(&model_bytes)?;

    // Load audio (must be 16kHz mono)
    let audio = whisper_apr::audio::load_audio("audio.wav")?;

    // Transcribe
    let result = whisper.transcribe(&audio, TranscribeOptions::default())?;

    println!("Transcription: {}", result.text);
    Ok(())
}
```

## See Also

- [CLI transcribe command](../getting-started/cli.md#transcribe)
- [TranscribeOptions API](../api-reference/transcribe-options.md)
- [Audio Pipeline](../architecture/audio-pipeline.md)
