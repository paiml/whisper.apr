//! Batch audio preprocessing (WAPR-080)
//!
//! Efficient batch processing of multiple audio segments for parallel inference.

use super::{AudioConfig, MelFilterbank};
use crate::error::WhisperResult;

/// Batch of audio samples for parallel processing
#[derive(Debug, Clone)]
pub struct AudioBatch {
    /// Individual audio segments (each is a Vec<f32> of samples)
    segments: Vec<Vec<f32>>,
    /// Audio configuration
    config: AudioConfig,
}

impl AudioBatch {
    /// Create a new empty batch
    #[must_use]
    pub fn new(config: AudioConfig) -> Self {
        Self {
            segments: Vec::new(),
            config,
        }
    }

    /// Create batch with default config
    #[must_use]
    pub fn with_default_config() -> Self {
        Self::new(AudioConfig::default())
    }

    /// Add an audio segment to the batch
    pub fn add_segment(&mut self, samples: Vec<f32>) {
        self.segments.push(samples);
    }

    /// Get the number of segments in the batch
    #[must_use]
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Check if the batch is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Get a reference to the segments
    #[must_use]
    pub fn segments(&self) -> &[Vec<f32>] {
        &self.segments
    }

    /// Get mutable reference to segments
    pub fn segments_mut(&mut self) -> &mut Vec<Vec<f32>> {
        &mut self.segments
    }

    /// Clear all segments
    pub fn clear(&mut self) {
        self.segments.clear();
    }

    /// Get the audio configuration
    #[must_use]
    pub const fn config(&self) -> &AudioConfig {
        &self.config
    }
}

/// Result of batch mel spectrogram computation
#[derive(Debug, Clone)]
pub struct BatchMelResult {
    /// Mel spectrograms for each segment (batch_size × n_mels × n_frames)
    pub mels: Vec<Vec<f32>>,
    /// Frame counts for each segment
    pub frame_counts: Vec<usize>,
    /// Maximum number of frames (for padding)
    pub max_frames: usize,
}

impl BatchMelResult {
    /// Get the batch size
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.mels.len()
    }

    /// Check if result is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mels.is_empty()
    }

    /// Get mel spectrogram for a specific segment
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Vec<f32>> {
        self.mels.get(index)
    }

    /// Get padded tensor for batch inference
    ///
    /// Returns a flat tensor of shape (batch_size, n_mels, max_frames) with zero-padding.
    #[must_use]
    pub fn to_padded_tensor(&self, n_mels: usize) -> Vec<f32> {
        let batch_size = self.batch_size();
        let total_size = batch_size * n_mels * self.max_frames;
        let mut tensor = vec![0.0_f32; total_size];

        for (batch_idx, mel) in self.mels.iter().enumerate() {
            let frames = self.frame_counts[batch_idx];
            for frame in 0..frames {
                for mel_idx in 0..n_mels {
                    let src_idx = frame * n_mels + mel_idx;
                    let dst_idx =
                        batch_idx * n_mels * self.max_frames + mel_idx * self.max_frames + frame;
                    if src_idx < mel.len() {
                        tensor[dst_idx] = mel[src_idx];
                    }
                }
            }
        }

        tensor
    }
}

/// Batch audio preprocessor
#[derive(Debug, Clone)]
pub struct BatchPreprocessor {
    /// Audio configuration
    config: AudioConfig,
    /// Mel filterbank
    filterbank: MelFilterbank,
}

impl BatchPreprocessor {
    /// Create a new batch preprocessor
    #[must_use]
    pub fn new(config: AudioConfig) -> Self {
        let filterbank = MelFilterbank::new(config.n_mels, config.n_fft, config.sample_rate);
        Self { config, filterbank }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_default_config() -> Self {
        Self::new(AudioConfig::default())
    }

    /// Process a batch of audio samples into mel spectrograms
    ///
    /// # Errors
    ///
    /// Returns error if any mel spectrogram computation fails.
    pub fn process_batch(&self, batch: &AudioBatch) -> WhisperResult<BatchMelResult> {
        let mut mels = Vec::with_capacity(batch.len());
        let mut frame_counts = Vec::with_capacity(batch.len());
        let mut max_frames = 0_usize;

        for samples in batch.segments() {
            let mel = self.filterbank.compute(samples, self.config.hop_length)?;
            let frames = mel.len() / self.config.n_mels;
            frame_counts.push(frames);
            max_frames = max_frames.max(frames);
            mels.push(mel);
        }

        Ok(BatchMelResult {
            mels,
            frame_counts,
            max_frames,
        })
    }

    /// Normalize a batch of audio samples
    #[must_use]
    pub fn normalize_batch(&self, batch: &AudioBatch) -> AudioBatch {
        let mut normalized = AudioBatch::new(self.config.clone());

        for samples in batch.segments() {
            let normalized_samples = normalize_audio(samples);
            normalized.add_segment(normalized_samples);
        }

        normalized
    }

    /// Get the number of mel channels
    #[must_use]
    pub fn n_mels(&self) -> usize {
        self.config.n_mels
    }
}

/// Normalize audio samples to [-1, 1] range
#[must_use]
fn normalize_audio(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let max_abs = samples
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f32, |a, b| a.max(b));

    if max_abs < f32::EPSILON {
        return samples.to_vec();
    }

    samples.iter().map(|x| x / max_abs).collect()
}

/// Split audio into fixed-size chunks for batch processing
#[must_use]
pub fn split_into_chunks(samples: &[f32], chunk_size: usize, overlap: usize) -> Vec<Vec<f32>> {
    if samples.is_empty() || chunk_size == 0 {
        return Vec::new();
    }

    let step = chunk_size.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < samples.len() {
        let end = (start + chunk_size).min(samples.len());
        chunks.push(samples[start..end].to_vec());
        start += step;

        // Stop if we've processed all samples
        if end >= samples.len() {
            break;
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // AudioBatch Tests
    // =========================================================================

    #[test]
    fn test_audio_batch_new() {
        let batch = AudioBatch::with_default_config();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_audio_batch_add_segment() {
        let mut batch = AudioBatch::with_default_config();
        batch.add_segment(vec![0.1, 0.2, 0.3]);
        batch.add_segment(vec![0.4, 0.5]);

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_audio_batch_clear() {
        let mut batch = AudioBatch::with_default_config();
        batch.add_segment(vec![0.1, 0.2, 0.3]);
        batch.clear();

        assert!(batch.is_empty());
    }

    #[test]
    fn test_audio_batch_segments() {
        let mut batch = AudioBatch::with_default_config();
        batch.add_segment(vec![1.0, 2.0]);
        batch.add_segment(vec![3.0, 4.0, 5.0]);

        let segments = batch.segments();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0], vec![1.0, 2.0]);
        assert_eq!(segments[1], vec![3.0, 4.0, 5.0]);
    }

    // =========================================================================
    // BatchMelResult Tests
    // =========================================================================

    #[test]
    fn test_batch_mel_result_batch_size() {
        let result = BatchMelResult {
            mels: vec![vec![0.0; 80], vec![0.0; 80]],
            frame_counts: vec![1, 1],
            max_frames: 1,
        };

        assert_eq!(result.batch_size(), 2);
    }

    #[test]
    fn test_batch_mel_result_get() {
        let result = BatchMelResult {
            mels: vec![vec![1.0; 80], vec![2.0; 80]],
            frame_counts: vec![1, 1],
            max_frames: 1,
        };

        assert!(result.get(0).is_some());
        assert!(result.get(1).is_some());
        assert!(result.get(2).is_none());
    }

    #[test]
    fn test_batch_mel_result_is_empty() {
        let empty = BatchMelResult {
            mels: Vec::new(),
            frame_counts: Vec::new(),
            max_frames: 0,
        };
        assert!(empty.is_empty());

        let non_empty = BatchMelResult {
            mels: vec![vec![0.0]],
            frame_counts: vec![1],
            max_frames: 1,
        };
        assert!(!non_empty.is_empty());
    }

    // =========================================================================
    // BatchPreprocessor Tests
    // =========================================================================

    #[test]
    fn test_batch_preprocessor_new() {
        let preprocessor = BatchPreprocessor::with_default_config();
        assert_eq!(preprocessor.n_mels(), 80);
    }

    #[test]
    fn test_batch_preprocessor_process_empty() {
        let preprocessor = BatchPreprocessor::with_default_config();
        let batch = AudioBatch::with_default_config();

        let result = preprocessor
            .process_batch(&batch)
            .expect("process empty batch");
        assert!(result.is_empty());
        assert_eq!(result.max_frames, 0);
    }

    #[test]
    fn test_batch_preprocessor_process_single() {
        let preprocessor = BatchPreprocessor::with_default_config();
        let mut batch = AudioBatch::with_default_config();

        // Add 16000 samples (1 second at 16kHz)
        let samples: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.001).sin()).collect();
        batch.add_segment(samples);

        let result = preprocessor.process_batch(&batch).expect("process single");
        assert_eq!(result.batch_size(), 1);
        assert!(result.max_frames > 0);
    }

    #[test]
    fn test_batch_preprocessor_process_multiple() {
        let preprocessor = BatchPreprocessor::with_default_config();
        let mut batch = AudioBatch::with_default_config();

        // Add multiple segments of different lengths
        batch.add_segment((0..8000).map(|i| (i as f32 * 0.001).sin()).collect());
        batch.add_segment((0..16000).map(|i| (i as f32 * 0.001).sin()).collect());
        batch.add_segment((0..4000).map(|i| (i as f32 * 0.001).sin()).collect());

        let result = preprocessor
            .process_batch(&batch)
            .expect("process multiple");
        assert_eq!(result.batch_size(), 3);
        assert_eq!(result.frame_counts.len(), 3);

        // Second segment should have most frames
        assert!(result.frame_counts[1] >= result.frame_counts[0]);
        assert!(result.frame_counts[1] >= result.frame_counts[2]);
    }

    #[test]
    fn test_batch_preprocessor_normalize() {
        let preprocessor = BatchPreprocessor::with_default_config();
        let mut batch = AudioBatch::with_default_config();

        batch.add_segment(vec![-2.0, 0.0, 2.0]);
        batch.add_segment(vec![-1.0, 0.5, 1.0]);

        let normalized = preprocessor.normalize_batch(&batch);
        assert_eq!(normalized.len(), 2);

        // First segment should be normalized to [-1, 0, 1]
        let first = &normalized.segments()[0];
        assert!((first[0] - (-1.0)).abs() < f32::EPSILON);
        assert!((first[2] - 1.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Normalization Tests
    // =========================================================================

    #[test]
    fn test_normalize_audio_empty() {
        let result = normalize_audio(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_normalize_audio_zeros() {
        let result = normalize_audio(&[0.0, 0.0, 0.0]);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_audio_positive() {
        let result = normalize_audio(&[0.0, 0.5, 1.0]);
        assert!((result[2] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalize_audio_negative() {
        let result = normalize_audio(&[-0.5, 0.0, -1.0]);
        assert!((result[2] - (-1.0)).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Chunk Splitting Tests
    // =========================================================================

    #[test]
    fn test_split_into_chunks_empty() {
        let result = split_into_chunks(&[], 100, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_split_into_chunks_zero_size() {
        let result = split_into_chunks(&[1.0, 2.0, 3.0], 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_split_into_chunks_no_overlap() {
        let samples: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let chunks = split_into_chunks(&samples, 3, 0);

        assert_eq!(chunks.len(), 4); // 10 / 3 = 3.33, so 4 chunks
        assert_eq!(chunks[0], vec![0.0, 1.0, 2.0]);
        assert_eq!(chunks[1], vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_split_into_chunks_with_overlap() {
        let samples: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let chunks = split_into_chunks(&samples, 4, 2);

        // Step = 4 - 2 = 2, so we get chunks starting at 0, 2, 4, 6, 8
        assert!(chunks.len() >= 3);
        assert_eq!(chunks[0], vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(chunks[1], vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_split_into_chunks_exact_fit() {
        let samples: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let chunks = split_into_chunks(&samples, 3, 0);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[2], vec![6.0, 7.0, 8.0]);
    }

    // =========================================================================
    // Padded Tensor Tests
    // =========================================================================

    #[test]
    fn test_batch_mel_to_padded_tensor() {
        let result = BatchMelResult {
            mels: vec![vec![1.0; 160], vec![2.0; 80]], // 2 frames, 1 frame
            frame_counts: vec![2, 1],
            max_frames: 2,
        };

        let tensor = result.to_padded_tensor(80);
        // Should have shape (2, 80, 2) = 320 elements
        assert_eq!(tensor.len(), 2 * 80 * 2);
    }
}
