//! Worker JavaScript Generator
//!
//! Generates the Web Worker JavaScript.
//! Key feature: processAudioTick checks `isDone()` flag (like whisper.cpp's `g_running`)

/// Generate the complete worker JavaScript
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn generate_worker_js() -> String {
    r"
// Worker state variables
let baseUrl = '';
let wasmModule = null;
let wasm = null;
let worker = null;
let ringBuffer = null;
let processingInterval = null;
let initialized = false;

// Message handler
self.onmessage = async (e) => {
    const msg = e.data;

    // Bootstrap case - load WASM
    if (msg.type === 'bootstrap') {
        baseUrl = msg.baseUrl || '';
        try {
            console.log('[Worker] Bootstrap received, loading WASM from:', baseUrl);
            wasmModule = await import(baseUrl + '/pkg/whisper_apr_demo.js');
            wasm = await wasmModule['default'](baseUrl + '/pkg/whisper_apr_demo_bg.wasm');
            worker = wasmModule.initWorker();
            initialized = true;
            console.log('[Worker] WASM initialized successfully');
            console.log('[Worker] TranscriptionWorker created, ready for commands');
            self.postMessage({ type: 'Ready' });
        } catch (e) {
            console.error('[Worker] Init failed:', e);
            self.postMessage({ type: 'Error', message: 'Worker init failed: ' + e.toString() });
        }
        return;
    }

    // Check initialized before processing other messages
    if (!initialized) {
        console.warn('[Worker] Not initialized, ignoring message:', msg.type);
        return;
    }

    console.log('[Worker] Received message type:', msg.type);

    switch (msg.type) {
        case 'init':
            console.log('[Worker] Processing init message');
            if (msg.buffer) {
                ringBuffer = wasmModule.SharedRingBuffer.fromBuffer(msg.buffer);
                worker.setRingBuffer(ringBuffer);
                console.log('[Worker] Ring buffer attached');
            }
            const modelUrl = msg.modelUrl.startsWith('/')
                ? baseUrl + msg.modelUrl
                : msg.modelUrl;
            try {
                console.log('[Worker] Loading model from:', modelUrl);
                const result = await worker.loadModel(modelUrl);
                self.postMessage(result);
            } catch (e) {
                console.error('[Worker] Model load failed:', e);
                self.postMessage({ type: 'Error', message: e.toString() });
            }
            break;

        case 'start':
            console.log('[Worker] Starting processing at sample rate:', msg.sampleRate);
            worker.startProcessing(msg.sampleRate);
            // Start processing interval - checks isDone flag
            processingInterval = setInterval(processAudioTick, 50);
            break;

        case 'stop':
            console.log('[Worker] Stopping processing (via message)');
            if (processingInterval) {
                clearInterval(processingInterval);
                processingInterval = null;
            }
            const result = worker.stopProcessing();
            if (result) {
                self.postMessage(result);
            }
            break;

        case 'metrics':
            self.postMessage(worker.getMetrics());
            break;

        case 'shutdown':
            console.log('[Worker] Shutting down');
            if (processingInterval) {
                clearInterval(processingInterval);
            }
            self.close();
            break;

        default:
            console.warn('[Worker] Unknown message type:', msg.type);
    }
};

// Process audio helper function - checks isDone flag like whisper.cpp
const processAudioTick = () => {
    // Check if buffer is done (producer stopped) - CRITICAL for stop detection
    if (ringBuffer && ringBuffer.isDone()) {
        // Stop the interval
        if (processingInterval) {
            clearInterval(processingInterval);
            processingInterval = null;
        }
        // Get final transcription
        console.log('[Worker] Buffer done, getting final transcription');
        const finalResult = worker.stopProcessing();
        if (finalResult) {
            self.postMessage(finalResult);
        }
        return;
    }

    // Normal processing
    const result = worker.processAudio();
    if (result) {
        self.postMessage(result);
    }
};

console.log('[Worker] Module loaded, waiting for bootstrap message');
".to_string()
}
