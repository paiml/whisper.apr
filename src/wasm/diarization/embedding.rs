//! WASM speaker embedding

use wasm_bindgen::prelude::*;

use crate::diarization::embedding::SpeakerEmbedding;

/// WASM-friendly speaker embedding
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct SpeakerEmbeddingWasm {
    pub(crate) vector: Vec<f32>,
    pub(crate) speaker_id: usize,
    pub(crate) confidence: f32,
}

#[wasm_bindgen]
impl SpeakerEmbeddingWasm {
    /// Get embedding dimension
    #[wasm_bindgen(getter)]
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Get speaker ID
    #[wasm_bindgen(getter, js_name = speakerId)]
    pub fn speaker_id(&self) -> usize {
        self.speaker_id
    }

    /// Get confidence score
    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Get embedding vector
    #[wasm_bindgen(getter)]
    pub fn vector(&self) -> Vec<f32> {
        self.vector.clone()
    }

    /// Compute cosine similarity with another embedding
    #[wasm_bindgen(js_name = cosineSimilarity)]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        if self.vector.len() != other.vector.len() {
            return 0.0;
        }

        let dot: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }
}

impl From<SpeakerEmbedding> for SpeakerEmbeddingWasm {
    fn from(emb: SpeakerEmbedding) -> Self {
        Self {
            vector: emb.vector().to_vec(),
            speaker_id: emb.speaker_id(),
            confidence: emb.confidence(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speaker_embedding_wasm_cosine_similarity() {
        let emb1 = SpeakerEmbeddingWasm {
            vector: vec![1.0, 0.0, 0.0],
            speaker_id: 0,
            confidence: 1.0,
        };
        let emb2 = SpeakerEmbeddingWasm {
            vector: vec![1.0, 0.0, 0.0],
            speaker_id: 1,
            confidence: 1.0,
        };

        // Same vectors should have similarity 1.0
        let sim = emb1.cosine_similarity(&emb2);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_speaker_embedding_wasm_cosine_orthogonal() {
        let emb1 = SpeakerEmbeddingWasm {
            vector: vec![1.0, 0.0, 0.0],
            speaker_id: 0,
            confidence: 1.0,
        };
        let emb2 = SpeakerEmbeddingWasm {
            vector: vec![0.0, 1.0, 0.0],
            speaker_id: 1,
            confidence: 1.0,
        };

        // Orthogonal vectors should have similarity 0.0
        let sim = emb1.cosine_similarity(&emb2);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_speaker_embedding_wasm_dim_mismatch() {
        let emb1 = SpeakerEmbeddingWasm {
            vector: vec![1.0, 0.0],
            speaker_id: 0,
            confidence: 1.0,
        };
        let emb2 = SpeakerEmbeddingWasm {
            vector: vec![1.0, 0.0, 0.0],
            speaker_id: 1,
            confidence: 1.0,
        };

        // Different dimensions should return 0.0
        let sim = emb1.cosine_similarity(&emb2);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_speaker_embedding_wasm_zero_vector() {
        let emb1 = SpeakerEmbeddingWasm {
            vector: vec![0.0, 0.0, 0.0],
            speaker_id: 0,
            confidence: 1.0,
        };
        let emb2 = SpeakerEmbeddingWasm {
            vector: vec![1.0, 0.0, 0.0],
            speaker_id: 1,
            confidence: 1.0,
        };

        // Zero vector should return 0.0
        let sim = emb1.cosine_similarity(&emb2);
        assert!(sim.abs() < 1e-6);
    }
}
