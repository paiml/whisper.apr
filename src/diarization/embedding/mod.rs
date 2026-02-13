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

#[cfg(test)]
mod tests;

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
        let denom = norm_a * norm_b;

        if denom < f32::EPSILON {
            0.0
        } else {
            dot / denom
        }
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
        let scale = if norm < f32::EPSILON { 1.0 } else { 1.0 / norm };

        Self {
            vector: self.vector.iter().map(|&x| x * scale).collect(),
            speaker_id: self.speaker_id,
            confidence: self.confidence,
        }
    }

    /// Compute mean of multiple embeddings
    #[must_use]
    pub fn mean(embeddings: &[Self]) -> Option<Self> {
        let dim = embeddings.first()?.dim();
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpeakerEmbeddingModel {
    /// Simple MFCC-based model (lightweight, WASM-friendly)
    #[default]
    MfccSimple,
    /// X-vector model (more accurate, heavier)
    XVector,
    /// ECAPA-TDNN model (state-of-the-art, heavy)
    EcapaTdnn,
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
        let samples = if sample_rate == 16000 {
            audio.to_vec()
        } else {
            self.resample(audio, sample_rate, 16000)
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
            return Err(WhisperError::Diarization(format!(
                "Audio too short for MFCC extraction: {} samples, need at least {}",
                audio.len(),
                FRAME_LENGTH
            )));
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
                    let window = 0.46f32.mul_add(
                        -(2.0 * std::f32::consts::PI * i as f32 / (FRAME_LENGTH - 1) as f32).cos(),
                        0.54,
                    );
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
            let log_mel: Vec<f32> = mel_spectrum.iter().map(|&x| (x.max(1e-10)).ln()).collect();

            // Apply DCT to get MFCC
            let mfcc = self.apply_dct(&log_mel);

            mfcc_frames.push(mfcc);
        }

        Ok(mfcc_frames)
    }

    /// Compute power spectrum using simple DFT (WASM-friendly)
    #[allow(clippy::needless_range_loop)]
    fn compute_power_spectrum(&self, frame: &[f32]) -> Vec<f32> {
        let _ = self; // Method for consistency
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

            power[k] = real.mul_add(real, imag * imag);
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
            .map(|row| row.iter().zip(log_mel.iter()).map(|(&d, &m)| d * m).sum())
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
        let _ = self; // Method for consistency
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

            let sample = audio[idx0].mul_add(1.0 - frac as f32, audio[idx1] * frac as f32);
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
                let val = (std::f32::consts::PI * k as f32 * 2.0f32.mul_add(n as f32, 1.0)
                    / (2.0 * num_filters as f32))
                    .cos();
                row.push(val);
            }
            matrix.push(row);
        }

        matrix
    }

    /// Compute mel filterbank
    fn compute_mel_filterbank(
        num_filters: usize,
        fft_size: usize,
        sample_rate: u32,
    ) -> Vec<Vec<f32>> {
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

            let rising_end = bin_points[i + 1].min(filter.len());
            let rising_base = bin_points[i];
            let rising_span = (bin_points[i + 1] - rising_base).max(1) as f32;
            for (j, val) in filter
                .iter_mut()
                .enumerate()
                .take(rising_end)
                .skip(rising_base)
            {
                *val = (j - rising_base) as f32 / rising_span;
            }

            let falling_end = bin_points[i + 2].min(filter.len());
            let falling_peak = bin_points[i + 2];
            let falling_span = (falling_peak - bin_points[i + 1]).max(1) as f32;
            for (j, val) in filter
                .iter_mut()
                .enumerate()
                .take(falling_end)
                .skip(bin_points[i + 1])
            {
                *val = (falling_peak - j) as f32 / falling_span;
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
