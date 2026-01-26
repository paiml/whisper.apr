//! WASM diarizer

use wasm_bindgen::prelude::*;

use super::config::DiarizationConfigWasm;
use super::result::DiarizationResultWasm;
use crate::diarization::{DiarizationConfig, Diarizer};

/// WASM bindings for speaker diarization
///
/// This provides a JavaScript-friendly API for identifying
/// who spoke when in an audio stream.
#[wasm_bindgen]
pub struct DiarizerWasm {
    inner: Diarizer,
}

#[wasm_bindgen]
impl DiarizerWasm {
    /// Create a new diarizer with default configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Diarizer::new(DiarizationConfig::default()),
        }
    }

    /// Create a diarizer with custom configuration
    #[wasm_bindgen(js_name = withConfig)]
    pub fn with_config(config: DiarizationConfigWasm) -> Self {
        Self {
            inner: Diarizer::new(config.into()),
        }
    }

    /// Create a diarizer optimized for real-time processing
    #[wasm_bindgen(js_name = forRealtime)]
    pub fn for_realtime() -> Self {
        Self::with_config(DiarizationConfigWasm::for_realtime())
    }

    /// Create a diarizer optimized for accuracy
    #[wasm_bindgen(js_name = forAccuracy)]
    pub fn for_accuracy() -> Self {
        Self::with_config(DiarizationConfigWasm::for_accuracy())
    }

    /// Process audio and return diarization result
    ///
    /// # Arguments
    /// * `audio` - Audio samples as Float32Array (mono, normalized to [-1, 1])
    /// * `sample_rate` - Audio sample rate (e.g., 16000)
    ///
    /// # Returns
    /// Diarization result with speaker-labeled segments
    #[wasm_bindgen]
    pub fn process(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<DiarizationResultWasm, JsValue> {
        self.inner
            .process(audio, sample_rate)
            .map(|r| r.into())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for DiarizerWasm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diarizer_wasm_new() {
        let diarizer = DiarizerWasm::new();
        assert!(std::mem::size_of_val(&diarizer) > 0);
    }

    #[test]
    fn test_diarizer_wasm_for_realtime() {
        let diarizer = DiarizerWasm::for_realtime();
        assert!(std::mem::size_of_val(&diarizer) > 0);
    }

    #[test]
    fn test_diarizer_wasm_for_accuracy() {
        let diarizer = DiarizerWasm::for_accuracy();
        assert!(std::mem::size_of_val(&diarizer) > 0);
    }

    #[test]
    fn test_diarizer_wasm_default() {
        let diarizer = DiarizerWasm::default();
        assert!(std::mem::size_of_val(&diarizer) > 0);
    }
}
