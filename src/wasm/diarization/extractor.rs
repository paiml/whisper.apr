//! WASM embedding extractor

use wasm_bindgen::prelude::*;

use super::embedding::SpeakerEmbeddingWasm;
use crate::diarization::embedding::{EmbeddingConfig, EmbeddingExtractor};

/// WASM-friendly embedding extractor for advanced use cases
#[wasm_bindgen]
pub struct EmbeddingExtractorWasm {
    inner: EmbeddingExtractor,
}

#[wasm_bindgen]
impl EmbeddingExtractorWasm {
    /// Create a new embedding extractor
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: EmbeddingExtractor::new(EmbeddingConfig::default()),
        }
    }

    /// Extract speaker embedding from audio segment
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono, normalized)
    /// * `sample_rate` - Audio sample rate
    ///
    /// # Returns
    /// Speaker embedding vector
    #[wasm_bindgen]
    pub fn extract(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<SpeakerEmbeddingWasm, JsValue> {
        self.inner
            .extract(audio, sample_rate)
            .map(|e| e.into())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Compare two audio segments for speaker similarity
    ///
    /// # Returns
    /// Similarity score (0.0 - 1.0), higher means more similar
    #[wasm_bindgen(js_name = compareSpeakers)]
    pub fn compare_speakers(
        &self,
        audio1: &[f32],
        audio2: &[f32],
        sample_rate: u32,
    ) -> Result<f32, JsValue> {
        let emb1 = self
            .inner
            .extract(audio1, sample_rate)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let emb2 = self
            .inner
            .extract(audio2, sample_rate)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(emb1.cosine_similarity(&emb2))
    }
}

impl Default for EmbeddingExtractorWasm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_extractor_wasm_default_trait() {
        let extractor = EmbeddingExtractorWasm::default();
        assert!(std::mem::size_of_val(&extractor) > 0);
    }
}
