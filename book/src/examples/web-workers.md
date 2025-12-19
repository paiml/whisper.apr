# Web Worker Integration

Run whisper.apr in a Web Worker to keep the main thread responsive during transcription.

## Why Web Workers?

- Transcription is CPU-intensive and can block the UI
- Web Workers run in a separate thread
- WASM SIMD works fully in Web Workers
- Progress updates can be sent to the main thread

## Worker Setup

### worker.js

```javascript
import init, { WhisperApr, TranscribeOptions } from 'whisper-apr';

let whisper = null;

self.onmessage = async (e) => {
    const { type, data } = e.data;

    switch (type) {
        case 'init':
            await initialize(data.modelUrl);
            break;
        case 'transcribe':
            await transcribe(data.audio, data.options);
            break;
    }
};

async function initialize(modelUrl) {
    try {
        await init();

        const response = await fetch(modelUrl);
        const modelBytes = new Uint8Array(await response.arrayBuffer());
        whisper = WhisperApr.loadFromApr(modelBytes);

        self.postMessage({ type: 'ready' });
    } catch (error) {
        self.postMessage({ type: 'error', error: error.message });
    }
}

async function transcribe(audioData, options) {
    try {
        self.postMessage({ type: 'progress', progress: 0 });

        const opts = TranscribeOptions.fromJson(options);
        const result = await whisper.transcribe(audioData, opts);

        self.postMessage({ type: 'progress', progress: 100 });
        self.postMessage({ type: 'result', result: result.toJson() });
    } catch (error) {
        self.postMessage({ type: 'error', error: error.message });
    }
}
```

### main.js

```javascript
class WhisperWorker {
    constructor() {
        this.worker = new Worker(new URL('./worker.js', import.meta.url), {
            type: 'module'
        });
        this.pending = new Map();
        this.messageId = 0;

        this.worker.onmessage = (e) => this.handleMessage(e.data);
    }

    handleMessage(data) {
        switch (data.type) {
            case 'ready':
                this.onReady?.();
                break;
            case 'progress':
                this.onProgress?.(data.progress);
                break;
            case 'result':
                this.onResult?.(data.result);
                break;
            case 'error':
                this.onError?.(data.error);
                break;
        }
    }

    async init(modelUrl) {
        return new Promise((resolve, reject) => {
            this.onReady = resolve;
            this.onError = reject;
            this.worker.postMessage({ type: 'init', data: { modelUrl } });
        });
    }

    async transcribe(audio, options = {}) {
        return new Promise((resolve, reject) => {
            this.onResult = resolve;
            this.onError = reject;
            this.worker.postMessage({
                type: 'transcribe',
                data: { audio, options }
            });
        });
    }

    terminate() {
        this.worker.terminate();
    }
}

// Usage
const whisperWorker = new WhisperWorker();

whisperWorker.onProgress = (progress) => {
    console.log(`Progress: ${progress}%`);
};

await whisperWorker.init('/models/whisper-tiny.apr');
const result = await whisperWorker.transcribe(audioData);
console.log(result.text);
```

## Transferable Objects

For better performance, transfer audio data instead of copying:

```javascript
// Main thread
const audioBuffer = audioData.buffer;
worker.postMessage(
    { type: 'transcribe', data: { audio: audioData } },
    [audioBuffer] // Transfer ownership
);

// Worker
self.onmessage = (e) => {
    // audioData is now owned by the worker
    const audioData = new Float32Array(e.data.data.audio);
};
```

## SharedArrayBuffer (Advanced)

For real-time streaming with shared memory:

```javascript
// Requires cross-origin isolation headers:
// Cross-Origin-Opener-Policy: same-origin
// Cross-Origin-Embedder-Policy: require-corp

const sharedBuffer = new SharedArrayBuffer(16000 * 30 * 4); // 30s of audio
const audioView = new Float32Array(sharedBuffer);

// Share with worker
worker.postMessage({ type: 'setBuffer', buffer: sharedBuffer });

// Write audio directly (both threads see updates)
audioView.set(newSamples, writeOffset);
```

## Multiple Workers

For parallel processing of long audio:

```javascript
class WhisperPool {
    constructor(workerCount = navigator.hardwareConcurrency) {
        this.workers = [];
        for (let i = 0; i < workerCount; i++) {
            this.workers.push(new WhisperWorker());
        }
    }

    async initAll(modelUrl) {
        await Promise.all(this.workers.map(w => w.init(modelUrl)));
    }

    async transcribeParallel(audioChunks) {
        const results = await Promise.all(
            audioChunks.map((chunk, i) =>
                this.workers[i % this.workers.length].transcribe(chunk)
            )
        );
        return results.map(r => r.text).join(' ');
    }
}
```

## Webpack/Vite Configuration

### Vite

```javascript
// vite.config.js
export default {
    worker: {
        format: 'es'
    },
    optimizeDeps: {
        exclude: ['whisper-apr']
    }
};
```

### Webpack

```javascript
// webpack.config.js
module.exports = {
    module: {
        rules: [
            {
                test: /\.wasm$/,
                type: 'webassembly/async'
            }
        ]
    },
    experiments: {
        asyncWebAssembly: true
    }
};
```

## See Also

- [Browser Integration](../getting-started/browser-integration.md)
- [File Upload](./file-upload.md)
- [Memory Optimization](../performance/memory.md)
