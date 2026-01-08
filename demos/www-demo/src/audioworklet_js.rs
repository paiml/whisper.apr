//! `AudioWorklet` JavaScript Generator
//!
//! Generates the `AudioWorklet` processor that captures microphone audio
//! and writes to the `SharedRingBuffer`.

/// Generate the `AudioWorklet` processor JavaScript
///
/// This is a simple class that captures audio from the microphone
/// and writes it to the `SharedRingBuffer`.
#[must_use] 
pub fn generate_audioworklet_js() -> String {
    r"
class WhisperProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.ringBuffer = null;

        // Handle messages from main thread
        this.port.onmessage = (e) => {
            if (e.data.type === 'init') {
                this.ringBuffer = e.data.ringBuffer;
                console.log('[AudioWorklet] Ring buffer attached');
            }
        };
    }

    process(inputs, _outputs, _parameters) {
        const input = inputs[0];
        if (input && input.length > 0) {
            const samples = input[0];
            // Write to ring buffer if available
            if (this.ringBuffer && samples) {
                this.ringBuffer.write(samples);
            }
        }
        // Return true to keep processor alive
        return true;
    }
}

registerProcessor('whisper-processor', WhisperProcessor);
".to_string()
}
