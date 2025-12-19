# Real-Time Microphone

This example demonstrates real-time audio streaming and transcription from a microphone input.

## Running the Example

```bash
cargo run --example streaming_audio
```

## Code Overview

```rust
use whisper_apr::audio::{RingBuffer, StreamingConfig, StreamingProcessor};

fn main() {
    // Configure streaming processor
    let config = StreamingConfig::with_sample_rate(16000)
        .without_vad(); // Disable VAD for demo

    println!("Chunk samples: {}", config.chunk_samples());
    println!("Overlap samples: {}", config.overlap_samples());

    // Create streaming processor
    let mut processor = StreamingProcessor::new(config);

    // Simulate streaming audio input
    let chunk_duration_ms = 100;
    let chunk_samples = (16000 * chunk_duration_ms / 1000) as usize;

    for chunk_idx in 0..5 {
        let audio_chunk = capture_audio(chunk_samples);
        processor.push_audio(&audio_chunk);

        let stats = processor.stats();
        println!("Buffer fill: {:.1}%", stats.buffer_fill() * 100.0);

        // Check if a chunk is ready for processing
        if let Some(ready_chunk) = processor.get_chunk() {
            process_chunk(&ready_chunk);
        }
    }
}
```

## StreamingConfig Options

| Option | Default | Description |
|--------|---------|-------------|
| sample_rate | 16000 | Input sample rate in Hz |
| chunk_ms | 3000 | Chunk duration in milliseconds |
| overlap_ms | 200 | Overlap between chunks |
| vad_enabled | true | Enable voice activity detection |
| vad_threshold | 0.5 | VAD sensitivity (0.0-1.0) |

## Ring Buffer

The `RingBuffer` provides lock-free audio buffering:

```rust
use whisper_apr::audio::RingBuffer;

let mut ring = RingBuffer::new(1024);

// Write audio data
ring.write(&audio_data);
println!("Available: {}", ring.available_read());

// Read without consuming (peek)
let mut peek_buf = vec![0.0; 10];
ring.peek(&mut peek_buf);

// Read and consume
let mut output = vec![0.0; 50];
let read = ring.read(&mut output);
```

## CLI Usage

The CLI provides the `stream` command for real-time transcription:

```bash
# Basic streaming
whisper-apr stream

# With custom settings
whisper-apr stream --step 3000 --length 10000 --keep 200

# With VAD
whisper-apr stream --vad-thold 0.6

# Save audio while streaming
whisper-apr stream --save-audio

# Translate while streaming
whisper-apr stream --translate
```

## Browser Integration

For WASM, use the Web Audio API to capture microphone input:

```javascript
import init, { StreamingProcessor } from 'whisper-apr';

await init();

const processor = new StreamingProcessor({
    sampleRate: 16000,
    chunkMs: 3000,
});

const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const audioContext = new AudioContext({ sampleRate: 16000 });
const source = audioContext.createMediaStreamSource(stream);

const scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
scriptProcessor.onaudioprocess = (e) => {
    const audioData = e.inputBuffer.getChannelData(0);
    processor.pushAudio(audioData);

    const chunk = processor.getChunk();
    if (chunk) {
        transcribe(chunk);
    }
};

source.connect(scriptProcessor);
scriptProcessor.connect(audioContext.destination);
```

## See Also

- [CLI stream command](../getting-started/cli.md#stream)
- [Streaming Inference](../advanced/streaming.md)
- [Voice Activity Detection](../advanced/vad.md)
