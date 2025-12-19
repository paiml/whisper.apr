# Progressive Web App

Build an offline-capable speech transcription PWA with whisper.apr.

## Service Worker Setup

### sw.js

```javascript
const CACHE_NAME = 'whisper-pwa-v1';
const MODEL_CACHE = 'whisper-models-v1';

const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/app.js',
    '/style.css',
    '/whisper_apr_bg.wasm',
    '/whisper_apr.js'
];

// Install: cache static assets
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(STATIC_ASSETS);
        })
    );
});

// Activate: clean old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys
                    .filter((key) => key !== CACHE_NAME && key !== MODEL_CACHE)
                    .map((key) => caches.delete(key))
            );
        })
    );
});

// Fetch: serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // Model files: cache with separate strategy
    if (url.pathname.endsWith('.apr')) {
        event.respondWith(cacheFirst(event.request, MODEL_CACHE));
        return;
    }

    // Static assets: cache first
    event.respondWith(cacheFirst(event.request, CACHE_NAME));
});

async function cacheFirst(request, cacheName) {
    const cached = await caches.match(request);
    if (cached) return cached;

    const response = await fetch(request);
    if (response.ok) {
        const cache = await caches.open(cacheName);
        cache.put(request, response.clone());
    }
    return response;
}
```

## Web App Manifest

### manifest.json

```json
{
    "name": "Whisper Transcriber",
    "short_name": "Whisper",
    "description": "Offline speech-to-text transcription",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#1a1a2e",
    "theme_color": "#4a90d9",
    "icons": [
        {
            "src": "/icons/icon-192.png",
            "sizes": "192x192",
            "type": "image/png"
        },
        {
            "src": "/icons/icon-512.png",
            "sizes": "512x512",
            "type": "image/png"
        }
    ]
}
```

## HTML Entry Point

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#4a90d9">
    <link rel="manifest" href="/manifest.json">
    <link rel="apple-touch-icon" href="/icons/icon-192.png">
    <title>Whisper Transcriber</title>
</head>
<body>
    <div id="app">
        <div id="status">Loading...</div>
        <div id="offline-indicator" hidden>Offline Mode</div>

        <button id="record" disabled>Record</button>
        <input type="file" id="file" accept="audio/*" disabled />

        <div id="output"></div>
    </div>

    <script type="module" src="/app.js"></script>
    <script>
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js');
        }
    </script>
</body>
</html>
```

## Application Logic

### app.js

```javascript
import init, { WhisperApr } from './whisper_apr.js';

let whisper = null;
let modelCached = false;

async function initialize() {
    updateStatus('Initializing WASM...');
    await init();

    updateStatus('Loading model...');
    await loadModel();

    updateStatus('Ready');
    enableControls();

    // Monitor online/offline status
    window.addEventListener('online', () => hideOfflineIndicator());
    window.addEventListener('offline', () => showOfflineIndicator());
}

async function loadModel() {
    const modelUrl = '/models/whisper-tiny.apr';

    // Check if model is cached
    const cache = await caches.open('whisper-models-v1');
    const cached = await cache.match(modelUrl);

    let modelBytes;
    if (cached) {
        modelCached = true;
        modelBytes = new Uint8Array(await cached.arrayBuffer());
    } else {
        // Download and cache
        const response = await fetch(modelUrl);
        modelBytes = new Uint8Array(await response.arrayBuffer());
        await cache.put(modelUrl, new Response(modelBytes));
        modelCached = true;
    }

    whisper = WhisperApr.loadFromApr(modelBytes);
}

function updateStatus(message) {
    document.getElementById('status').textContent = message;
}

function enableControls() {
    document.getElementById('record').disabled = false;
    document.getElementById('file').disabled = false;
}

// Check storage quota
async function checkStorage() {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
        const { usage, quota } = await navigator.storage.estimate();
        const percentUsed = ((usage / quota) * 100).toFixed(2);
        console.log(`Storage: ${percentUsed}% used (${formatBytes(usage)} of ${formatBytes(quota)})`);
    }
}

function formatBytes(bytes) {
    const units = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    while (bytes >= 1024 && i < units.length - 1) {
        bytes /= 1024;
        i++;
    }
    return `${bytes.toFixed(1)} ${units[i]}`;
}

initialize();
```

## Caching Strategy

### Model Pre-caching

```javascript
async function precacheModels(sizes = ['tiny', 'base']) {
    const cache = await caches.open('whisper-models-v1');

    for (const size of sizes) {
        const url = `/models/whisper-${size}.apr`;
        const cached = await cache.match(url);

        if (!cached) {
            updateStatus(`Downloading ${size} model...`);
            const response = await fetch(url);
            await cache.put(url, response);
        }
    }
}
```

### Storage Persistence

```javascript
async function requestPersistence() {
    if ('storage' in navigator && 'persist' in navigator.storage) {
        const granted = await navigator.storage.persist();
        if (granted) {
            console.log('Persistent storage granted');
        } else {
            console.warn('Persistent storage denied - data may be evicted');
        }
    }
}
```

## Background Sync

For queuing transcriptions when offline:

```javascript
// In app.js
async function transcribeWithSync(audioData) {
    if (navigator.onLine) {
        // Transcribe immediately
        return await whisper.transcribe(audioData);
    } else {
        // Queue for later
        await saveToIndexedDB('pending-transcriptions', audioData);
        await registerBackgroundSync('transcribe-pending');
        return { text: '[Queued for transcription]' };
    }
}

// In sw.js
self.addEventListener('sync', (event) => {
    if (event.tag === 'transcribe-pending') {
        event.waitUntil(processPendingTranscriptions());
    }
});
```

## IndexedDB Storage

```javascript
const DB_NAME = 'whisper-pwa';
const DB_VERSION = 1;

function openDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);

        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            db.createObjectStore('transcriptions', { keyPath: 'id', autoIncrement: true });
            db.createObjectStore('audio-queue', { keyPath: 'id', autoIncrement: true });
        };
    });
}

async function saveTranscription(text, timestamp) {
    const db = await openDB();
    const tx = db.transaction('transcriptions', 'readwrite');
    tx.objectStore('transcriptions').add({ text, timestamp, synced: false });
    return tx.complete;
}
```

## See Also

- [Browser Integration](../getting-started/browser-integration.md)
- [Web Worker Integration](./web-workers.md)
- [Memory Optimization](../performance/memory.md)
