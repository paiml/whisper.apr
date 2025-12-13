# Browser Integration

Integrating Whisper.apr into web applications.

## Web Workers

For smooth UI, run inference in a Web Worker:

```javascript
// whisper-worker.js
import init, { WhisperApr } from 'whisper-apr';

let whisper = null;

self.onmessage = async (e) => {
  const { type, data } = e.data;

  switch (type) {
    case 'init':
      await init();
      whisper = await WhisperApr.load(data.modelUrl, {
        onProgress: (loaded, total) => {
          self.postMessage({ type: 'progress', loaded, total });
        }
      });
      self.postMessage({ type: 'ready' });
      break;

    case 'transcribe':
      const result = await whisper.transcribe(data.audio, data.options);
      self.postMessage({ type: 'result', result });
      break;
  }
};
```

```javascript
// main.js
const worker = new Worker('./whisper-worker.js', { type: 'module' });

worker.postMessage({ type: 'init', data: { modelUrl: '/models/base.apr' } });

worker.onmessage = (e) => {
  const { type, ...data } = e.data;

  switch (type) {
    case 'progress':
      console.log(`Loading: ${(data.loaded/data.total*100).toFixed(1)}%`);
      break;
    case 'ready':
      console.log('Model ready!');
      break;
    case 'result':
      console.log('Transcription:', data.result.text);
      break;
  }
};
```

## React Integration

```jsx
import { useState, useEffect, useCallback } from 'react';
import init, { WhisperApr } from 'whisper-apr';

function useWhisper(modelUrl) {
  const [whisper, setWhisper] = useState(null);
  const [loading, setLoading] = useState(true);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    async function load() {
      await init();
      const model = await WhisperApr.load(modelUrl, {
        onProgress: (loaded, total) => setProgress(loaded / total),
      });
      setWhisper(model);
      setLoading(false);
    }
    load();
  }, [modelUrl]);

  const transcribe = useCallback(async (audio, options = {}) => {
    if (!whisper) throw new Error('Model not loaded');
    return whisper.transcribe(audio, options);
  }, [whisper]);

  return { whisper, loading, progress, transcribe };
}

function TranscriptionApp() {
  const { loading, progress, transcribe } = useWhisper('/models/base.apr');
  const [result, setResult] = useState('');

  const handleFile = async (e) => {
    const file = e.target.files[0];
    const audioContext = new AudioContext({ sampleRate: 16000 });
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const samples = audioBuffer.getChannelData(0);

    const transcription = await transcribe(samples);
    setResult(transcription.text);
  };

  if (loading) {
    return <div>Loading model: {(progress * 100).toFixed(1)}%</div>;
  }

  return (
    <div>
      <input type="file" accept="audio/*" onChange={handleFile} />
      <pre>{result}</pre>
    </div>
  );
}
```

## Vue Integration

```vue
<template>
  <div>
    <div v-if="loading">Loading: {{ (progress * 100).toFixed(1) }}%</div>
    <div v-else>
      <input type="file" accept="audio/*" @change="handleFile" />
      <pre>{{ result }}</pre>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import init, { WhisperApr } from 'whisper-apr';

const whisper = ref(null);
const loading = ref(true);
const progress = ref(0);
const result = ref('');

onMounted(async () => {
  await init();
  whisper.value = await WhisperApr.load('/models/base.apr', {
    onProgress: (loaded, total) => { progress.value = loaded / total; }
  });
  loading.value = false;
});

async function handleFile(e) {
  const file = e.target.files[0];
  const audioContext = new AudioContext({ sampleRate: 16000 });
  const arrayBuffer = await file.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  const samples = audioBuffer.getChannelData(0);

  const transcription = await whisper.value.transcribe(samples);
  result.value = transcription.text;
}
</script>
```

## Microphone Recording

```javascript
async function recordAndTranscribe(whisper, durationMs = 5000) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioContext = new AudioContext({ sampleRate: 16000 });
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);

  const chunks = [];

  processor.onaudioprocess = (e) => {
    chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };

  source.connect(processor);
  processor.connect(audioContext.destination);

  // Record for specified duration
  await new Promise(resolve => setTimeout(resolve, durationMs));

  // Stop recording
  processor.disconnect();
  source.disconnect();
  stream.getTracks().forEach(track => track.stop());

  // Combine chunks
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const audio = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    audio.set(chunk, offset);
    offset += chunk.length;
  }

  // Transcribe
  return whisper.transcribe(audio);
}
```

## Progressive Web App (PWA)

For offline support, cache the model in a Service Worker:

```javascript
// service-worker.js
const CACHE_NAME = 'whisper-apr-v1';
const MODEL_URLS = [
  '/models/base.apr',
  '/whisper_apr.js',
  '/whisper_apr_bg.wasm',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(MODEL_URLS))
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});
```
