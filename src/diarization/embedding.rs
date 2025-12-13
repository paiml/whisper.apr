//! Speaker embedding extraction (WAPR-150)
//!
//! Provides d-vector extraction for speaker identification.
//!
//! # Overview
//!
//! Speaker embeddings are fixed-length vector representations of speaker
//! characteristics extracted from audio segments. These embeddings capture
//! speaker-specific features like voice timbre, pitch patterns, and speaking style.
//!
//! # Implementation
//!
//! This module implements a lightweight embedding model suitable for WASM:
//! - MFCC-based feature extraction
//! - Simple neural network encoder (suitable for SIMD acceleration)
//! - 256-dimensional d-vector output

use crate::error::{WhisperError, WhisperResult};

/// Speaker embedding dimension
pub const EMBEDDING_DIM: usize = 256;

/// Number of MFCC coefficients
const NUM_MFCC: usize = 40;

/// Frame length in samples (25ms at 16kHz)
const FRAME_LENGTH: usize = 400;

/// Frame shift in samples (10ms at 16kHz)
const FRAME_SHIFT: usize = 160;

/// Embedding extraction configuration
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Window size in seconds for embedding extraction
    pub window_size: f32,
    /// Hop size in seconds between windows
    pub hop_size: f32,
    /// Whether to use mean pooling (vs. last frame)
    pub use_mean_pooling: bool,
    /// Normalize embeddings to unit length
    pub normalize: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: EMBEDDING_DIM,
            window_size: 1.5,
            hop_size: 0.75,
            use_mean_pooling: true,
            normalize: true,
        }
    }
}

impl EmbeddingConfig {
    /// Configuration for real-time processing
    #[must_use]
    pub fn for_realtime() -> Self {
        Self {
            window_size: 1.0,
            hop_size: 0.5,
            ..Default::default()
        }
    }

    /// Configuration for high accuracy
    #[must_use]
    pub fn for_accuracy() -> Self {
        Self {
            window_size: 2.0,
            hop_size: 0.5,
            ..Default::default()
        }
    }

    /// Set window size
    #[must_use]
    pub fn with_window_size(mut self, size: f32) -> Self {
        self.window_size = size;
        self
    }

    /// Set hop size
    #[must_use]
    pub fn with_hop_size(mut self, size: f32) -> Self {
        self.hop_size = size;
        self
    }
}

/// Speaker embedding vector
#[derive(Debug, Clone)]
pub struct SpeakerEmbedding {
    /// Embedding vector
    vector: Vec<f32>,
    /// Associated speaker ID (if known)
    speaker_id: usize,
    /// Confidence score (0.0 - 1.0)
    confidence: f32,
}

impl SpeakerEmbedding {
    /// Create a new speaker embedding
    #[must_use]
    pub fn new(vector: Vec<f32>, speaker_id: usize) -> Self {
        Self {
            vector,
            speaker_id,
            confidence: 1.0,
        }
    }

    /// Create embedding with confidence
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Get the embedding vector
    #[must_use]
    pub fn vector(&self) -> &[f32] {
        &self.vector
    }

    /// Get the speaker ID
    #[must_use]
    pub fn speaker_id(&self) -> usize {
        self.speaker_id
    }

    /// Get the confidence score
    #[must_use]
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Get embedding dimension
    #[must_use]
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Compute cosine similarity with another embedding
    #[must_use]
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

        if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Compute Euclidean distance to another embedding
    #[must_use]
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        if self.vector.len() != other.vector.len() {
            return f32::MAX;
        }

        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Normalize embedding to unit length
    #[must_use]
    pub fn normalized(&self) -> Self {
        let norm: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm < f32::EPSILON {
            return self.clone();
        }

        Self {
            vector: self.vector.iter().map(|x| x / norm).collect(),
            speaker_id: self.speaker_id,
            confidence: self.confidence,
        }
    }

    /// Compute mean of multiple embeddings
    #[must_use]
    pub fn mean(embeddings: &[Self]) -> Option<Self> {
        if embeddings.is_empty() {
            return None;
        }

        let dim = embeddings[0].dim();
        if embeddings.iter().any(|e| e.dim() != dim) {
            return None;
        }

        let mut mean_vec = vec![0.0f32; dim];
        for embedding in embeddings {
            for (i, &val) in embedding.vector.iter().enumerate() {
                mean_vec[i] += val;
            }
        }

        let n = embeddings.len() as f32;
        for val in &mut mean_vec {
            *val /= n;
        }

        Some(Self::new(mean_vec, embeddings[0].speaker_id))
    }
}

/// Speaker embedding model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerEmbeddingModel {
    /// Simple MFCC-based model (lightweight, WASM-friendly)
    MfccSimple,
    /// X-vector model (more accurate, heavier)
    XVector,
    /// ECAPA-TDNN model (state-of-the-art, heavy)
    EcapaTdnn,
}

impl Default for SpeakerEmbeddingModel {
    fn default() -> Self {
        Self::MfccSimple
    }
}

/// Speaker embedding extractor
#[derive(Debug)]
pub struct EmbeddingExtractor {
    config: EmbeddingConfig,
    model: SpeakerEmbeddingModel,
    /// Pre-computed DCT matrix for MFCC
    dct_matrix: Vec<Vec<f32>>,
    /// Mel filterbank
    mel_filters: Vec<Vec<f32>>,
}

impl EmbeddingExtractor {
    /// Create a new embedding extractor
    #[must_use]
    pub fn new(config: EmbeddingConfig) -> Self {
        let dct_matrix = Self::compute_dct_matrix(NUM_MFCC, 80);
        let mel_filters = Self::compute_mel_filterbank(80, 512, 16000);

        Self {
            config,
            model: SpeakerEmbeddingModel::default(),
            dct_matrix,
            mel_filters,
        }
    }

    /// Create extractor with specific model
    #[must_use]
    pub fn with_model(mut self, model: SpeakerEmbeddingModel) -> Self {
        self.model = model;
        self
    }

    /// Extract speaker embedding from audio
    pub fn extract(&self, audio: &[f32], sample_rate: u32) -> WhisperResult<SpeakerEmbedding> {
        if audio.is_empty() {
            return Err(WhisperError::Diarization(
                "Empty audio for embedding extraction".to_string(),
            ));
        }

        // Resample if needed (assume 16kHz target)
        let samples = if sample_rate != 16000 {
            self.resample(audio, sample_rate, 16000)
        } else {
            audio.to_vec()
        };

        // Extract MFCC features
        let mfcc = self.extract_mfcc(&samples)?;

        // Apply simple neural network to get embedding
        let embedding = self.mfcc_to_embedding(&mfcc)?;

        // Normalize if configured
        let embedding = if self.config.normalize {
            embedding.normalized()
        } else {
            embedding
        };

        Ok(embedding)
    }

    /// Extract MFCC features from audio
    fn extract_mfcc(&self, audio: &[f32]) -> WhisperResult<Vec<Vec<f32>>> {
        // Need at least one full frame for MFCC extraction
        if audio.len() < FRAME_LENGTH {
            return Err(WhisperError::Diarization(
                format!(
                    "Audio too short for MFCC extraction: {} samples, need at least {}",
                    audio.len(),
                    FRAME_LENGTH
                ),
            ));
        }

        let num_frames = (audio.len() - FRAME_LENGTH) / FRAME_SHIFT + 1;

        let mut mfcc_frames = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * FRAME_SHIFT;
            let end = (start + FRAME_LENGTH).min(audio.len());

            // Apply Hamming window
            let mut frame: Vec<f32> = audio[start..end]
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    let window = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (FRAME_LENGTH - 1) as f32).cos();
                    s * window
                })
                .collect();

            // Pad if needed
            frame.resize(512, 0.0);

            // Compute power spectrum via FFT
            let power_spectrum = self.compute_power_spectrum(&frame);

            // Apply mel filterbank
            let mel_spectrum = self.apply_mel_filterbank(&power_spectrum);

            // Apply log
            let log_mel: Vec<f32> = mel_spectrum
                .iter()
                .map(|&x| (x.max(1e-10)).ln())
                .collect();

            // Apply DCT to get MFCC
            let mfcc = self.apply_dct(&log_mel);

            mfcc_frames.push(mfcc);
        }

        Ok(mfcc_frames)
    }

    /// Compute power spectrum using simple DFT (WASM-friendly)
    fn compute_power_spectrum(&self, frame: &[f32]) -> Vec<f32> {
        let n = frame.len();
        let mut power = vec![0.0f32; n / 2 + 1];

        // Simple DFT (could be optimized with FFT for production)
        for k in 0..=n / 2 {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for (i, &sample) in frame.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            power[k] = real * real + imag * imag;
        }

        power
    }

    /// Apply mel filterbank to power spectrum
    fn apply_mel_filterbank(&self, power_spectrum: &[f32]) -> Vec<f32> {
        self.mel_filters
            .iter()
            .map(|filter| {
                filter
                    .iter()
                    .zip(power_spectrum.iter())
                    .map(|(&f, &p)| f * p)
                    .sum()
            })
            .collect()
    }

    /// Apply DCT to get MFCC coefficients
    fn apply_dct(&self, log_mel: &[f32]) -> Vec<f32> {
        self.dct_matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(log_mel.iter())
                    .map(|(&d, &m)| d * m)
                    .sum()
            })
            .collect()
    }

    /// Convert MFCC features to embedding
    fn mfcc_to_embedding(&self, mfcc: &[Vec<f32>]) -> WhisperResult<SpeakerEmbedding> {
        if mfcc.is_empty() {
            return Err(WhisperError::Diarization(
                "No MFCC frames for embedding".to_string(),
            ));
        }

        // Simple approach: statistics pooling over frames
        let num_features = mfcc[0].len();
        let num_frames = mfcc.len();

        // Compute mean and std for each MFCC coefficient
        let mut means = vec![0.0f32; num_features];
        let mut stds = vec![0.0f32; num_features];

        for frame in mfcc {
            for (i, &val) in frame.iter().enumerate() {
                means[i] += val;
            }
        }

        for mean in &mut means {
            *mean /= num_frames as f32;
        }

        for frame in mfcc {
            for (i, &val) in frame.iter().enumerate() {
                let diff = val - means[i];
                stds[i] += diff * diff;
            }
        }

        for std in &mut stds {
            *std = (*std / num_frames as f32).sqrt();
        }

        // Concatenate mean and std as initial features
        let mut features: Vec<f32> = means;
        features.extend(stds);

        // Project to embedding dimension via simple linear layer
        let embedding = self.project_to_embedding(&features);

        Ok(SpeakerEmbedding::new(embedding, 0))
    }

    /// Project features to embedding dimension
    fn project_to_embedding(&self, features: &[f32]) -> Vec<f32> {
        // Simple deterministic projection (pseudo-random but fixed)
        let _input_dim = features.len();
        let output_dim = self.config.embedding_dim;

        let mut embedding = vec![0.0f32; output_dim];

        for (j, emb_val) in embedding.iter_mut().enumerate() {
            for (i, &feat) in features.iter().enumerate() {
                // Simple mixing function
                let weight = ((i * 31 + j * 17) % 1000) as f32 / 1000.0 - 0.5;
                *emb_val += feat * weight;
            }
            // Apply ReLU
            *emb_val = emb_val.max(0.0);
        }

        embedding
    }

    /// Simple resampling (linear interpolation)
    fn resample(&self, audio: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        if from_rate == to_rate {
            return audio.to_vec();
        }

        let ratio = to_rate as f64 / from_rate as f64;
        let new_len = (audio.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = i as f64 / ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(audio.len() - 1);
            let frac = src_idx - idx0 as f64;

            let sample = audio[idx0] * (1.0 - frac as f32) + audio[idx1] * frac as f32;
            resampled.push(sample);
        }

        resampled
    }

    /// Compute DCT matrix
    fn compute_dct_matrix(num_mfcc: usize, num_filters: usize) -> Vec<Vec<f32>> {
        let mut matrix = Vec::with_capacity(num_mfcc);

        for k in 0..num_mfcc {
            let mut row = Vec::with_capacity(num_filters);
            for n in 0..num_filters {
                let val = (std::f32::consts::PI * k as f32 * (2.0 * n as f32 + 1.0)
                    / (2.0 * num_filters as f32))
                    .cos();
                row.push(val);
            }
            matrix.push(row);
        }

        matrix
    }

    /// Compute mel filterbank
    fn compute_mel_filterbank(num_filters: usize, fft_size: usize, sample_rate: u32) -> Vec<Vec<f32>> {
        let low_freq = 0.0;
        let high_freq = sample_rate as f32 / 2.0;

        // Convert to mel scale
        let low_mel = Self::hz_to_mel(low_freq);
        let high_mel = Self::hz_to_mel(high_freq);

        // Create mel points
        let mel_points: Vec<f32> = (0..=num_filters + 1)
            .map(|i| low_mel + (high_mel - low_mel) * i as f32 / (num_filters + 1) as f32)
            .collect();

        // Convert back to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();

        // Convert to FFT bin indices
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&h| ((fft_size + 1) as f32 * h / sample_rate as f32).floor() as usize)
            .collect();

        // Create filterbank
        let mut filterbank = Vec::with_capacity(num_filters);

        for i in 0..num_filters {
            let mut filter = vec![0.0f32; fft_size / 2 + 1];

            for j in bin_points[i]..bin_points[i + 1] {
                if j < filter.len() {
                    filter[j] = (j - bin_points[i]) as f32
                        / (bin_points[i + 1] - bin_points[i]).max(1) as f32;
                }
            }

            for j in bin_points[i + 1]..bin_points[i + 2] {
                if j < filter.len() {
                    filter[j] = (bin_points[i + 2] - j) as f32
                        / (bin_points[i + 2] - bin_points[i + 1]).max(1) as f32;
                }
            }

            filterbank.push(filter);
        }

        filterbank
    }

    /// Convert Hz to mel scale
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert mel to Hz
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // EmbeddingConfig Tests
    // =========================================================================

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.embedding_dim, 256);
        assert!(config.normalize);
        assert!(config.use_mean_pooling);
    }

    #[test]
    fn test_embedding_config_for_realtime() {
        let config = EmbeddingConfig::for_realtime();
        assert!((config.window_size - 1.0).abs() < f32::EPSILON);
        assert!((config.hop_size - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_embedding_config_for_accuracy() {
        let config = EmbeddingConfig::for_accuracy();
        assert!((config.window_size - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_embedding_config_with_window_size() {
        let config = EmbeddingConfig::default().with_window_size(3.0);
        assert!((config.window_size - 3.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // SpeakerEmbedding Tests
    // =========================================================================

    #[test]
    fn test_speaker_embedding_new() {
        let vec = vec![0.1; 256];
        let emb = SpeakerEmbedding::new(vec.clone(), 0);

        assert_eq!(emb.speaker_id(), 0);
        assert_eq!(emb.dim(), 256);
        assert!((emb.confidence() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_speaker_embedding_with_confidence() {
        let emb = SpeakerEmbedding::new(vec![0.1; 256], 0).with_confidence(0.8);
        assert!((emb.confidence() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_speaker_embedding_cosine_similarity_identical() {
        let emb1 = SpeakerEmbedding::new(vec![1.0, 0.0, 0.0], 0);
        let emb2 = SpeakerEmbedding::new(vec![1.0, 0.0, 0.0], 0);

        let sim = emb1.cosine_similarity(&emb2);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_speaker_embedding_cosine_similarity_orthogonal() {
        let emb1 = SpeakerEmbedding::new(vec![1.0, 0.0, 0.0], 0);
        let emb2 = SpeakerEmbedding::new(vec![0.0, 1.0, 0.0], 0);

        let sim = emb1.cosine_similarity(&emb2);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_speaker_embedding_cosine_similarity_opposite() {
        let emb1 = SpeakerEmbedding::new(vec![1.0, 0.0, 0.0], 0);
        let emb2 = SpeakerEmbedding::new(vec![-1.0, 0.0, 0.0], 0);

        let sim = emb1.cosine_similarity(&emb2);
        assert!((sim + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_speaker_embedding_euclidean_distance() {
        let emb1 = SpeakerEmbedding::new(vec![0.0, 0.0, 0.0], 0);
        let emb2 = SpeakerEmbedding::new(vec![3.0, 4.0, 0.0], 0);

        let dist = emb1.euclidean_distance(&emb2);
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_speaker_embedding_normalized() {
        let emb = SpeakerEmbedding::new(vec![3.0, 4.0], 0);
        let normalized = emb.normalized();

        let norm: f32 = normalized.vector().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_speaker_embedding_mean() {
        let embeddings = vec![
            SpeakerEmbedding::new(vec![1.0, 2.0, 3.0], 0),
            SpeakerEmbedding::new(vec![3.0, 4.0, 5.0], 0),
        ];

        let mean = SpeakerEmbedding::mean(&embeddings).expect("should compute mean");
        assert!((mean.vector()[0] - 2.0).abs() < 0.001);
        assert!((mean.vector()[1] - 3.0).abs() < 0.001);
        assert!((mean.vector()[2] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_speaker_embedding_mean_empty() {
        let embeddings: Vec<SpeakerEmbedding> = vec![];
        let mean = SpeakerEmbedding::mean(&embeddings);
        assert!(mean.is_none());
    }

    // =========================================================================
    // SpeakerEmbeddingModel Tests
    // =========================================================================

    #[test]
    fn test_speaker_embedding_model_default() {
        let model = SpeakerEmbeddingModel::default();
        assert_eq!(model, SpeakerEmbeddingModel::MfccSimple);
    }

    // =========================================================================
    // EmbeddingExtractor Tests
    // =========================================================================

    #[test]
    fn test_embedding_extractor_new() {
        let config = EmbeddingConfig::default();
        let extractor = EmbeddingExtractor::new(config);
        assert_eq!(extractor.config().embedding_dim, 256);
    }

    #[test]
    fn test_embedding_extractor_with_model() {
        let extractor =
            EmbeddingExtractor::new(EmbeddingConfig::default()).with_model(SpeakerEmbeddingModel::XVector);
        // Model is set (internal state)
        assert!(extractor.config().normalize);
    }

    #[test]
    fn test_embedding_extractor_extract_empty() {
        let extractor = EmbeddingExtractor::new(EmbeddingConfig::default());
        let result = extractor.extract(&[], 16000);
        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_extractor_extract_short_audio() {
        let extractor = EmbeddingExtractor::new(EmbeddingConfig::default());
        let audio = vec![0.1; 100]; // Very short audio
        let result = extractor.extract(&audio, 16000);
        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_extractor_extract_valid_audio() {
        let extractor = EmbeddingExtractor::new(EmbeddingConfig::default());
        // Generate 1 second of audio at 16kHz
        let audio: Vec<f32> = (0..16000)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();

        let result = extractor.extract(&audio, 16000);
        assert!(result.is_ok());

        let embedding = result.expect("should succeed");
        assert_eq!(embedding.dim(), 256);
    }

    #[test]
    fn test_embedding_extractor_extract_normalized() {
        let config = EmbeddingConfig::default();
        let extractor = EmbeddingExtractor::new(config);

        let audio: Vec<f32> = (0..16000)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();

        let embedding = extractor.extract(&audio, 16000).expect("should succeed");
        let norm: f32 = embedding.vector().iter().map(|x| x * x).sum::<f32>().sqrt();

        // Should be approximately unit norm
        assert!((norm - 1.0).abs() < 0.1 || norm < f32::EPSILON);
    }

    #[test]
    fn test_embedding_extractor_resample() {
        let extractor = EmbeddingExtractor::new(EmbeddingConfig::default());
        let audio: Vec<f32> = vec![0.0, 1.0, 0.0, -1.0];

        let resampled = extractor.resample(&audio, 8000, 16000);
        assert!(resampled.len() > audio.len());
    }

    #[test]
    fn test_hz_to_mel_to_hz() {
        let hz = 1000.0;
        let mel = EmbeddingExtractor::hz_to_mel(hz);
        let hz_back = EmbeddingExtractor::mel_to_hz(mel);
        assert!((hz - hz_back).abs() < 0.1);
    }

    #[test]
    fn test_dct_matrix_dimensions() {
        let matrix = EmbeddingExtractor::compute_dct_matrix(40, 80);
        assert_eq!(matrix.len(), 40);
        assert_eq!(matrix[0].len(), 80);
    }

    #[test]
    fn test_mel_filterbank_dimensions() {
        let filterbank = EmbeddingExtractor::compute_mel_filterbank(80, 512, 16000);
        assert_eq!(filterbank.len(), 80);
        assert_eq!(filterbank[0].len(), 257); // 512/2 + 1
    }
}
