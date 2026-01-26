//! WASM diarization configuration

use wasm_bindgen::prelude::*;

use crate::diarization::DiarizationConfig;

/// WASM-friendly diarization configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DiarizationConfigWasm {
    pub(crate) max_speakers: Option<usize>,
    pub(crate) min_speakers: usize,
    pub(crate) min_segment_duration: f32,
    pub(crate) embedding_dim: usize,
    pub(crate) clustering_threshold: f32,
}

#[wasm_bindgen]
impl DiarizationConfigWasm {
    /// Create default diarization config
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            max_speakers: None,
            min_speakers: 1,
            min_segment_duration: 0.3,
            embedding_dim: 256,
            clustering_threshold: 0.5,
        }
    }

    /// Create config optimized for real-time processing
    #[wasm_bindgen(js_name = forRealtime)]
    pub fn for_realtime() -> Self {
        Self {
            max_speakers: Some(4),
            min_speakers: 1,
            min_segment_duration: 0.5,
            embedding_dim: 128,
            clustering_threshold: 0.6,
        }
    }

    /// Create config optimized for accuracy
    #[wasm_bindgen(js_name = forAccuracy)]
    pub fn for_accuracy() -> Self {
        Self {
            max_speakers: None,
            min_speakers: 1,
            min_segment_duration: 0.2,
            embedding_dim: 256,
            clustering_threshold: 0.4,
        }
    }

    /// Set maximum number of speakers
    #[wasm_bindgen(js_name = setMaxSpeakers)]
    pub fn set_max_speakers(&mut self, max: Option<usize>) {
        self.max_speakers = max;
    }

    /// Set minimum number of speakers
    #[wasm_bindgen(js_name = setMinSpeakers)]
    pub fn set_min_speakers(&mut self, min: usize) {
        self.min_speakers = min;
    }

    /// Set minimum segment duration in seconds
    #[wasm_bindgen(js_name = setMinSegmentDuration)]
    pub fn set_min_segment_duration(&mut self, duration: f32) {
        self.min_segment_duration = duration;
    }

    /// Set embedding dimension (128 or 256)
    #[wasm_bindgen(js_name = setEmbeddingDim)]
    pub fn set_embedding_dim(&mut self, dim: usize) {
        self.embedding_dim = dim;
    }

    /// Set clustering distance threshold (0.0 - 1.0)
    #[wasm_bindgen(js_name = setClusteringThreshold)]
    pub fn set_clustering_threshold(&mut self, threshold: f32) {
        self.clustering_threshold = threshold;
    }

    /// Get max speakers setting
    #[wasm_bindgen(getter, js_name = maxSpeakers)]
    pub fn max_speakers(&self) -> Option<usize> {
        self.max_speakers
    }

    /// Get min speakers setting
    #[wasm_bindgen(getter, js_name = minSpeakers)]
    pub fn min_speakers(&self) -> usize {
        self.min_speakers
    }

    /// Get min segment duration
    #[wasm_bindgen(getter, js_name = minSegmentDuration)]
    pub fn min_segment_duration(&self) -> f32 {
        self.min_segment_duration
    }
}

impl Default for DiarizationConfigWasm {
    fn default() -> Self {
        Self::new()
    }
}

impl From<DiarizationConfigWasm> for DiarizationConfig {
    fn from(wasm: DiarizationConfigWasm) -> Self {
        let mut config = Self::default().with_min_segment_duration(wasm.min_segment_duration);
        if let Some(max) = wasm.max_speakers {
            config = config.with_max_speakers(max);
        }
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diarization_config_wasm_default() {
        let config = DiarizationConfigWasm::new();
        assert!(config.max_speakers.is_none());
        assert_eq!(config.min_speakers, 1);
        assert!((config.min_segment_duration - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diarization_config_wasm_for_realtime() {
        let config = DiarizationConfigWasm::for_realtime();
        assert_eq!(config.max_speakers, Some(4));
        assert_eq!(config.embedding_dim, 128);
    }

    #[test]
    fn test_diarization_config_wasm_for_accuracy() {
        let config = DiarizationConfigWasm::for_accuracy();
        assert!(config.max_speakers.is_none());
        assert_eq!(config.embedding_dim, 256);
    }

    #[test]
    fn test_diarization_config_wasm_setters() {
        let mut config = DiarizationConfigWasm::new();

        config.set_max_speakers(Some(6));
        assert_eq!(config.max_speakers(), Some(6));

        config.set_min_speakers(2);
        assert_eq!(config.min_speakers(), 2);

        config.set_min_segment_duration(0.5);
        assert!((config.min_segment_duration() - 0.5).abs() < f32::EPSILON);
    }
}
