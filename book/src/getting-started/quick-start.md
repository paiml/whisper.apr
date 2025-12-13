# Quick Start

This guide walks through transcribing audio with Whisper.apr.

## Browser (JavaScript)

### Basic Setup

```html
<!DOCTYPE html>
<html>
<head>
  <title>Whisper.apr Demo</title>
</head>
<body>
  <input type="file" id="audioFile" accept="audio/*">
  <button id="transcribe">Transcribe</button>
  <pre id="result"></pre>

  <script type="module">
    import init, { WhisperApr } from './whisper_apr.js';

    let whisper = null;

    async function setup() {
      // Initialize WASM
      await init();

      // Load model with progress callback
      whisper = await WhisperApr.load('/models/base.apr', {
        onProgress: (loaded, total) => {
          console.log(`Loading: ${(loaded/total*100).toFixed(1)}%`);
        }
      });

      console.log('Model ready!');
    }

    document.getElementById('transcribe').onclick = async () => {
      const file = document.getElementById('audioFile').files[0];
      if (!file || !whisper) return;

      // Decode audio to Float32Array
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const arrayBuffer = await file.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      const samples = audioBuffer.getChannelData(0);

      // Transcribe
      const result = await whisper.transcribe(samples, {
        language: 'auto',
        task: 'transcribe',
      });

      document.getElementById('result').textContent = result.text;
    };

    setup();
  </script>
</body>
</html>
```

### With Timestamps

```javascript
const result = await whisper.transcribe(samples, {
  language: 'en',
  task: 'transcribe',
  wordTimestamps: true,
});

for (const segment of result.segments) {
  console.log(`[${segment.start.toFixed(2)}s - ${segment.end.toFixed(2)}s] ${segment.text}`);
}
```

## Rust

### Basic Transcription

```rust
use whisper_apr::{WhisperApr, TranscribeOptions, Task};

fn main() -> whisper_apr::WhisperResult<()> {
    // Load model
    let model_data = std::fs::read("base.apr")?;
    let whisper = WhisperApr::load(&model_data)?;

    // Load audio (must be 16kHz mono f32)
    let audio: Vec<f32> = load_audio("speech.wav")?;

    // Transcribe
    let result = whisper.transcribe(&audio, TranscribeOptions::default())?;

    println!("Transcription: {}", result.text);
    println!("Language: {}", result.language);

    Ok(())
}
```

### Translation

```rust
let result = whisper.transcribe(&audio, TranscribeOptions {
    task: Task::Translate,  // Translate to English
    ..Default::default()
})?;
```

### Beam Search

```rust
use whisper_apr::DecodingStrategy;

let result = whisper.transcribe(&audio, TranscribeOptions {
    strategy: DecodingStrategy::BeamSearch {
        beam_size: 5,
        temperature: 0.0,
        patience: 1.0,
    },
    ..Default::default()
})?;
```

## Audio Requirements

Whisper.apr expects:
- **Sample rate**: 16,000 Hz
- **Channels**: Mono (single channel)
- **Format**: 32-bit float (-1.0 to 1.0)
- **Duration**: Up to 30 seconds per chunk

### Converting Audio

Using ffmpeg:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f f32le output.raw
```

In JavaScript:

```javascript
const audioContext = new AudioContext({ sampleRate: 16000 });
const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
const mono = audioBuffer.getChannelData(0);  // First channel as Float32Array
```

## Next Steps

- [Browser Integration](./browser-integration.md) - Web Worker setup, React hooks
- [Core Concepts](./core-concepts.md) - Understanding the transcription pipeline
- [Performance](../performance/benchmarks.md) - Optimizing for your use case
