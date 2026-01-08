//! `AudioWorklet` Processor for Real-Time Audio Capture
//!
//! Replaces deprecated `ScriptProcessorNode` with low-latency `AudioWorklet`
//! running on dedicated audio thread.
//!
//! # References
//! - W3C Web Audio API (2021), Section 5.4: `AudioWorklet` Interface
//! - Adenot & Wilson (2018), "Enter `AudioWorklet`", Google Developers
//!
//! # Performance Constraints
//! - Process callback: <3ms (128 samples @ 44.1kHz = 2.9ms)
//! - Zero allocations in `process()` hot path
//! - Lock-free ring buffer writes via `SharedArrayBuffer`

use crate::ring_buffer::SharedRingBuffer;
use wasm_bindgen::prelude::*;

/// `AudioWorklet` processor state
#[wasm_bindgen]
pub struct AudioWorkletBridge {
    ring_buffer: Option<SharedRingBuffer>,
    sample_rate: u32,
    channels: u8,
    samples_written: u64,
    underruns: u32,
}

#[wasm_bindgen]
impl AudioWorkletBridge {
    /// Create a new `AudioWorklet` bridge
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            ring_buffer: None,
            sample_rate: 48000,
            channels: 1,
            samples_written: 0,
            underruns: 0,
        }
    }

    /// Initialize with a `SharedRingBuffer`
    #[wasm_bindgen]
    pub fn init(&mut self, buffer: &SharedRingBuffer, sample_rate: u32, channels: u8) {
        self.ring_buffer = Some(buffer.clone());
        self.sample_rate = sample_rate;
        self.channels = channels;
        self.samples_written = 0;
        self.underruns = 0;
    }

    /// Process audio samples from `AudioWorklet`
    ///
    /// Called from JavaScript `AudioWorkletProcessor.process()`
    /// Must complete within audio quantum budget (~3ms for 128 samples)
    #[wasm_bindgen(js_name = processSamples)]
    pub fn process_samples(&mut self, samples: &[f32]) -> bool {
        let Some(ref buffer) = self.ring_buffer else {
            return false;
        };

        match buffer.write(samples) {
            Ok(written) => {
                self.samples_written += written as u64;
                if written < samples.len() {
                    self.underruns += 1;
                    web_sys::console::warn_1(
                        &format!(
                            "[AudioWorklet] Buffer full, dropped {} samples",
                            samples.len() - written
                        )
                        .into(),
                    );
                }
                true
            }
            Err(e) => {
                web_sys::console::error_1(&format!("[AudioWorklet] Write error: {e:?}").into());
                false
            }
        }
    }

    /// Get total samples written
    #[wasm_bindgen(getter, js_name = samplesWritten)]
    #[must_use]
    pub fn samples_written(&self) -> u64 {
        self.samples_written
    }

    /// Get underrun count
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn underruns(&self) -> u32 {
        self.underruns
    }

    /// Get sample rate
    #[wasm_bindgen(getter, js_name = sampleRate)]
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl Default for AudioWorkletBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// JavaScript code for `AudioWorkletProcessor`
///
/// This must be loaded as a separate module via `audioContext.audioWorklet.addModule()`
pub const AUDIO_WORKLET_JS: &str = r"
// WhisperAudioProcessor - AudioWorklet for real-time audio capture
// Writes samples to SharedArrayBuffer ring buffer for worker consumption

class WhisperAudioProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();

        // Ring buffer will be passed via port message
        this.ringBuffer = null;
        this.headerView = null;
        this.dataView = null;
        this.capacity = 0;
        this.samplesWritten = 0;
        this.underruns = 0;

        // Constants matching Rust ring_buffer.rs
        this.HEADER_SIZE = 64;
        this.WRITE_IDX_OFFSET = 0;
        this.READ_IDX_OFFSET = 1;
        this.CAPACITY_OFFSET = 2;

        // Handle messages from main thread
        this.port.onmessage = (event) => {
            if (event.data.type === 'init') {
                this.initBuffer(event.data.buffer);
            } else if (event.data.type === 'reset') {
                this.samplesWritten = 0;
                this.underruns = 0;
            }
        };
    }

    initBuffer(sharedBuffer) {
        this.ringBuffer = sharedBuffer;
        this.headerView = new Int32Array(sharedBuffer, 0, this.HEADER_SIZE / 4);

        // Read capacity from header
        this.capacity = Atomics.load(this.headerView, this.CAPACITY_OFFSET);
        this.dataView = new Float32Array(sharedBuffer, this.HEADER_SIZE, this.capacity);

        console.log(`[AudioWorklet] Initialized with capacity: ${this.capacity} samples`);
    }

    process(inputs, outputs, parameters) {
        // Get mono input (first channel of first input)
        const input = inputs[0];
        if (!input || input.length === 0) {
            return true;
        }

        const samples = input[0]; // Mono channel
        if (!samples || samples.length === 0) {
            return true;
        }

        // If no buffer yet, just continue
        if (!this.ringBuffer) {
            return true;
        }

        // Write to ring buffer (lock-free)
        const written = this.writeToBuffer(samples);
        this.samplesWritten += written;

        if (written < samples.length) {
            this.underruns++;
        }

        // Return true to keep processor alive
        return true;
    }

    writeToBuffer(samples) {
        // Read current indices atomically
        const writeIdx = Atomics.load(this.headerView, this.WRITE_IDX_OFFSET);
        const readIdx = Atomics.load(this.headerView, this.READ_IDX_OFFSET);

        // Calculate available space (leave 1 slot to distinguish full from empty)
        let available;
        if (writeIdx >= readIdx) {
            available = this.capacity - 1 - (writeIdx - readIdx);
        } else {
            available = readIdx - writeIdx - 1;
        }

        const toWrite = Math.min(samples.length, available);
        if (toWrite === 0) {
            return 0;
        }

        // Write samples, handling wrap-around
        let idx = writeIdx;
        for (let i = 0; i < toWrite; i++) {
            this.dataView[idx] = samples[i];
            idx = (idx + 1) % this.capacity;
        }

        // Update write index with release semantics
        Atomics.store(this.headerView, this.WRITE_IDX_OFFSET, idx);

        // Notify waiting consumers
        Atomics.notify(this.headerView, this.WRITE_IDX_OFFSET, 1);

        return toWrite;
    }
}

registerProcessor('whisper-audio-processor', WhisperAudioProcessor);
";

/// Create a Blob URL for the `AudioWorklet` processor code
#[wasm_bindgen(js_name = createAudioWorkletBlobUrl)]
pub fn create_audio_worklet_blob_url() -> Result<String, JsValue> {
    let blob_parts = js_sys::Array::new();
    blob_parts.push(&JsValue::from_str(AUDIO_WORKLET_JS));

    // AudioWorklet requires JavaScript MIME type
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("application/javascript");

    let blob = web_sys::Blob::new_with_str_sequence_and_options(&blob_parts, &options)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)?;

    Ok(url)
}

/// Set up `AudioWorklet` with ring buffer
///
/// # Arguments
/// * `context` - `AudioContext`
/// * `ring_buffer` - `SharedRingBuffer` to write samples to
/// * `source` - `MediaStreamAudioSourceNode`
#[wasm_bindgen(js_name = setupAudioWorklet)]
pub async fn setup_audio_worklet(
    context: &web_sys::AudioContext,
    ring_buffer: &SharedRingBuffer,
    source: &web_sys::MediaStreamAudioSourceNode,
) -> Result<web_sys::AudioWorkletNode, JsValue> {
    // Create blob URL for worklet code
    let worklet_url = create_audio_worklet_blob_url()?;

    // Add module to audio worklet
    let worklet = context.audio_worklet()?;
    let promise = worklet.add_module(&worklet_url)?;
    wasm_bindgen_futures::JsFuture::from(promise).await?;

    // Clean up blob URL
    web_sys::Url::revoke_object_url(&worklet_url)?;

    // Create AudioWorkletNode
    let options = web_sys::AudioWorkletNodeOptions::new();
    options.set_number_of_inputs(1);
    options.set_number_of_outputs(0); // We don't need output, just capturing

    let node =
        web_sys::AudioWorkletNode::new_with_options(context, "whisper-audio-processor", &options)?;

    // Send ring buffer to worklet via port
    let port = node.port()?;
    let init_msg = js_sys::Object::new();
    js_sys::Reflect::set(&init_msg, &"type".into(), &"init".into())?;
    js_sys::Reflect::set(&init_msg, &"buffer".into(), &ring_buffer.buffer())?;
    port.post_message(&init_msg)?;

    // Connect source -> worklet
    source.connect_with_audio_node(&node)?;

    web_sys::console::log_1(&"[AudioWorklet] Setup complete, connected to source".into());

    Ok(node)
}
