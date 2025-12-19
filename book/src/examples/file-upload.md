# File Upload

This example demonstrates file upload and transcription in a browser environment using whisper.apr WASM.

## HTML Setup

```html
<!DOCTYPE html>
<html>
<head>
    <title>Whisper.apr File Upload</title>
</head>
<body>
    <input type="file" id="audioFile" accept="audio/*,video/*" />
    <button id="transcribe" disabled>Transcribe</button>
    <pre id="output"></pre>

    <script type="module" src="app.js"></script>
</body>
</html>
```

## JavaScript Implementation

```javascript
import init, { WhisperApr, TranscribeOptions } from 'whisper-apr';

let whisper = null;

async function initialize() {
    await init();

    // Fetch and load the model
    const response = await fetch('/models/whisper-tiny.apr');
    const modelBytes = new Uint8Array(await response.arrayBuffer());
    whisper = WhisperApr.loadFromApr(modelBytes);

    document.getElementById('transcribe').disabled = false;
}

async function transcribeFile(file) {
    const output = document.getElementById('output');
    output.textContent = 'Processing...';

    // Decode audio file
    const arrayBuffer = await file.arrayBuffer();
    const audioContext = new AudioContext({ sampleRate: 16000 });
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    // Get mono audio data
    const audioData = audioBuffer.getChannelData(0);

    // Transcribe
    const options = TranscribeOptions.default();
    const result = await whisper.transcribe(audioData, options);

    output.textContent = result.text;
}

document.getElementById('audioFile').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        transcribeFile(file);
    }
});

initialize();
```

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV | .wav | Native support, preferred |
| MP3 | .mp3 | Decoded via Web Audio API |
| FLAC | .flac | Decoded via Web Audio API |
| OGG | .ogg | Vorbis/Opus codecs |
| MP4 | .mp4, .m4a | Audio extracted from video |
| WebM | .webm | Audio extracted from video |

## Progress Reporting

For larger files, show progress:

```javascript
async function transcribeWithProgress(audioData) {
    const output = document.getElementById('output');
    const progressBar = document.getElementById('progress');

    // Split into chunks for progress updates
    const chunkSize = 16000 * 30; // 30 seconds
    const chunks = [];

    for (let i = 0; i < audioData.length; i += chunkSize) {
        chunks.push(audioData.slice(i, i + chunkSize));
    }

    let fullText = '';
    for (let i = 0; i < chunks.length; i++) {
        const result = await whisper.transcribe(chunks[i], options);
        fullText += result.text + ' ';

        progressBar.value = ((i + 1) / chunks.length) * 100;
        output.textContent = fullText;
    }
}
```

## Drag and Drop

```javascript
const dropZone = document.getElementById('dropZone');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
        await transcribeFile(file);
    }
});
```

## Error Handling

```javascript
async function transcribeFile(file) {
    try {
        // Validate file size (e.g., max 100MB)
        if (file.size > 100 * 1024 * 1024) {
            throw new Error('File too large. Maximum size is 100MB.');
        }

        // Validate file type
        if (!file.type.startsWith('audio/') && !file.type.startsWith('video/')) {
            throw new Error('Please select an audio or video file.');
        }

        const result = await processFile(file);
        showResult(result);
    } catch (error) {
        showError(error.message);
    }
}
```

## See Also

- [Browser Integration](../getting-started/browser-integration.md)
- [Web Worker Integration](./web-workers.md)
- [JavaScript Bindings](../api-reference/javascript-bindings.md)
