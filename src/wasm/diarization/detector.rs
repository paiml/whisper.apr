//! WASM turn detector

use wasm_bindgen::prelude::*;

use super::segment::SpeakerSegmentWasm;
use crate::diarization::segmentation::{SegmentationConfig, TurnDetector};

/// WASM-friendly turn detector for voice activity and speaker changes
#[wasm_bindgen]
pub struct TurnDetectorWasm {
    inner: TurnDetector,
}

#[wasm_bindgen]
impl TurnDetectorWasm {
    /// Create a new turn detector
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: TurnDetector::new(SegmentationConfig::default()),
        }
    }

    /// Detect speech segments in audio
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono, normalized)
    /// * `sample_rate` - Audio sample rate
    ///
    /// # Returns
    /// Array of detected speech segments
    #[wasm_bindgen(js_name = detectSegments)]
    pub fn detect_segments(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<SpeakerSegmentWasm>, JsValue> {
        self.inner
            .detect_segments(audio, sample_rate)
            .map(|segs| segs.into_iter().map(|s| s.into()).collect())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Detect potential speaker change points
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono, normalized)
    /// * `sample_rate` - Audio sample rate
    ///
    /// # Returns
    /// Array of timestamps (in seconds) where speaker changes may occur
    #[wasm_bindgen(js_name = detectChangePoints)]
    pub fn detect_change_points(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<f32>, JsValue> {
        self.inner
            .detect_change_points(audio, sample_rate)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for TurnDetectorWasm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turn_detector_wasm_default_trait() {
        let detector = TurnDetectorWasm::default();
        assert!(std::mem::size_of_val(&detector) > 0);
    }
}
