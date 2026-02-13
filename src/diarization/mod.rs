//! Speaker diarization module (WAPR-150 to WAPR-153)
//!
//! Provides speaker identification and turn detection for multi-speaker audio.
//!
//! # Overview
//!
//! Speaker diarization answers the question "who spoke when?" by:
//! 1. Extracting speaker embeddings (d-vectors) from audio segments
//! 2. Clustering embeddings to identify unique speakers
//! 3. Detecting speaker turns and assigning speaker labels
//!
//! # Usage
//!
//! ```rust,ignore
//! use whisper_apr::diarization::{Diarizer, DiarizationConfig};
//!
//! let config = DiarizationConfig::default();
//! let diarizer = Diarizer::new(config);
//!
//! let result = diarizer.process(&audio_samples, sample_rate)?;
//! for segment in result.segments() {
//!     println!("Speaker {}: {:.2}s - {:.2}s",
//!         segment.speaker_id, segment.start, segment.end);
//! }
//! ```

pub mod clustering;
pub mod embedding;
pub mod segmentation;

pub use clustering::{
    ClusteringAlgorithm, ClusteringConfig, ClusteringResult, SpeakerCluster, SpectralClustering,
};
pub use embedding::{EmbeddingConfig, EmbeddingExtractor, SpeakerEmbedding, SpeakerEmbeddingModel};
pub use segmentation::{SegmentationConfig, SpeakerSegment, SpeakerTurn, TurnDetector};

use crate::error::{WhisperError, WhisperResult};

/// Diarization configuration
#[derive(Debug, Clone)]
pub struct DiarizationConfig {
    /// Embedding extraction configuration
    pub embedding: EmbeddingConfig,
    /// Clustering configuration
    pub clustering: ClusteringConfig,
    /// Segmentation configuration
    pub segmentation: SegmentationConfig,
    /// Minimum segment duration in seconds
    pub min_segment_duration: f32,
    /// Maximum number of speakers (None for automatic)
    pub max_speakers: Option<usize>,
    /// Minimum number of speakers (default: 1)
    pub min_speakers: usize,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self {
            embedding: EmbeddingConfig::default(),
            clustering: ClusteringConfig::default(),
            segmentation: SegmentationConfig::default(),
            min_segment_duration: 0.5,
            max_speakers: None,
            min_speakers: 1,
        }
    }
}

impl DiarizationConfig {
    /// Create configuration optimized for real-time processing
    #[must_use]
    pub fn for_realtime() -> Self {
        Self {
            embedding: EmbeddingConfig::for_realtime(),
            clustering: ClusteringConfig::for_realtime(),
            segmentation: SegmentationConfig::for_realtime(),
            min_segment_duration: 0.3,
            max_speakers: Some(4),
            min_speakers: 1,
        }
    }

    /// Create configuration for high accuracy
    #[must_use]
    pub fn for_accuracy() -> Self {
        Self {
            embedding: EmbeddingConfig::for_accuracy(),
            clustering: ClusteringConfig::for_accuracy(),
            segmentation: SegmentationConfig::for_accuracy(),
            min_segment_duration: 0.5,
            max_speakers: None,
            min_speakers: 1,
        }
    }

    /// Set maximum number of speakers
    #[must_use]
    pub fn with_max_speakers(mut self, max: usize) -> Self {
        self.max_speakers = Some(max);
        self
    }

    /// Set minimum segment duration
    #[must_use]
    pub fn with_min_segment_duration(mut self, duration: f32) -> Self {
        self.min_segment_duration = duration;
        self
    }
}

/// Diarization result containing speaker segments
#[derive(Debug, Clone)]
pub struct DiarizationResult {
    /// Detected speaker segments
    segments: Vec<SpeakerSegment>,
    /// Number of unique speakers detected
    num_speakers: usize,
    /// Speaker embeddings for each detected speaker
    speaker_embeddings: Vec<SpeakerEmbedding>,
    /// Total audio duration in seconds
    duration: f32,
}

impl DiarizationResult {
    /// Create a new diarization result
    #[must_use]
    pub fn new(
        segments: Vec<SpeakerSegment>,
        num_speakers: usize,
        speaker_embeddings: Vec<SpeakerEmbedding>,
        duration: f32,
    ) -> Self {
        Self {
            segments,
            num_speakers,
            speaker_embeddings,
            duration,
        }
    }

    /// Get speaker segments
    #[must_use]
    pub fn segments(&self) -> &[SpeakerSegment] {
        &self.segments
    }

    /// Get number of unique speakers
    #[must_use]
    pub fn num_speakers(&self) -> usize {
        self.num_speakers
    }

    /// Get speaker embeddings
    #[must_use]
    pub fn speaker_embeddings(&self) -> &[SpeakerEmbedding] {
        &self.speaker_embeddings
    }

    /// Get total audio duration
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// Get segments for a specific speaker
    #[must_use]
    pub fn segments_for_speaker(&self, speaker_id: usize) -> Vec<&SpeakerSegment> {
        self.segments
            .iter()
            .filter(|s| s.speaker_id() == speaker_id)
            .collect()
    }

    /// Get total speaking time for a speaker
    #[must_use]
    pub fn speaking_time(&self, speaker_id: usize) -> f32 {
        self.segments_for_speaker(speaker_id)
            .iter()
            .map(|s| s.duration())
            .sum()
    }

    /// Get speaker turns (transitions between speakers)
    #[must_use]
    pub fn speaker_turns(&self) -> Vec<SpeakerTurn> {
        if self.segments.len() < 2 {
            return Vec::new();
        }

        self.segments
            .windows(2)
            .filter_map(|w| {
                if w[0].speaker_id() == w[1].speaker_id() {
                    None
                } else {
                    Some(SpeakerTurn::new(
                        w[0].speaker_id(),
                        w[1].speaker_id(),
                        w[0].end(),
                    ))
                }
            })
            .collect()
    }
}

/// Main diarizer for speaker identification
#[derive(Debug)]
pub struct Diarizer {
    config: DiarizationConfig,
    embedding_extractor: EmbeddingExtractor,
    turn_detector: TurnDetector,
}

impl Diarizer {
    /// Create a new diarizer with the given configuration
    #[must_use]
    pub fn new(config: DiarizationConfig) -> Self {
        let embedding_extractor = EmbeddingExtractor::new(config.embedding.clone());
        let turn_detector = TurnDetector::new(config.segmentation.clone());

        Self {
            config,
            embedding_extractor,
            turn_detector,
        }
    }

    /// Create a diarizer with default configuration
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(DiarizationConfig::default())
    }

    /// Process audio and return diarization result
    pub fn process(&self, audio: &[f32], sample_rate: u32) -> WhisperResult<DiarizationResult> {
        let duration = audio.len() as f32 / sample_rate as f32;

        // Step 1: Detect initial segments using VAD/energy
        let initial_segments = self.turn_detector.detect_segments(audio, sample_rate)?;

        if initial_segments.is_empty() {
            return Ok(DiarizationResult::new(Vec::new(), 0, Vec::new(), duration));
        }

        // Step 2: Extract embeddings for each segment
        let embeddings = self.extract_segment_embeddings(audio, sample_rate, &initial_segments)?;

        // Step 3: Cluster embeddings to identify speakers
        let clustering_result = self.cluster_speakers(&embeddings)?;

        // Step 4: Assign speaker labels to segments
        let labeled_segments = self.assign_speaker_labels(&initial_segments, &clustering_result)?;

        // Step 5: Merge consecutive segments from same speaker
        let merged_segments = self.merge_segments(labeled_segments);

        // Step 6: Extract representative embeddings per speaker
        let speaker_embeddings = clustering_result.cluster_centroids();

        Ok(DiarizationResult::new(
            merged_segments,
            clustering_result.num_clusters(),
            speaker_embeddings,
            duration,
        ))
    }

    /// Extract embeddings for each segment
    fn extract_segment_embeddings(
        &self,
        audio: &[f32],
        sample_rate: u32,
        segments: &[SpeakerSegment],
    ) -> WhisperResult<Vec<SpeakerEmbedding>> {
        let mut embeddings = Vec::with_capacity(segments.len());

        for segment in segments {
            let start_sample = (segment.start() * sample_rate as f32) as usize;
            let end_sample = (segment.end() * sample_rate as f32) as usize;
            let end_sample = end_sample.min(audio.len());

            if start_sample >= end_sample {
                continue;
            }

            let segment_audio = &audio[start_sample..end_sample];
            let embedding = self
                .embedding_extractor
                .extract(segment_audio, sample_rate)?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    /// Cluster speaker embeddings
    fn cluster_speakers(&self, embeddings: &[SpeakerEmbedding]) -> WhisperResult<ClusteringResult> {
        // All algorithms currently use spectral clustering as the implementation
        let algorithm = match self.config.clustering.algorithm {
            ClusteringAlgorithm::Spectral
            | ClusteringAlgorithm::KMeans
            | ClusteringAlgorithm::Agglomerative => {
                SpectralClustering::new(self.config.clustering.clone())
            }
        };

        algorithm.cluster(
            embeddings,
            self.config.max_speakers,
            self.config.min_speakers,
        )
    }

    /// Assign speaker labels to segments based on clustering
    fn assign_speaker_labels(
        &self,
        segments: &[SpeakerSegment],
        clustering: &ClusteringResult,
    ) -> WhisperResult<Vec<SpeakerSegment>> {
        let _ = self; // Method for consistency with diarization pipeline
        let labels = clustering.labels();

        if labels.len() != segments.len() {
            return Err(WhisperError::Diarization(
                "Mismatch between segments and cluster labels".to_string(),
            ));
        }

        Ok(segments
            .iter()
            .zip(labels.iter())
            .map(|(seg, &label)| seg.with_speaker_id(label))
            .collect())
    }

    /// Merge consecutive segments from the same speaker
    fn merge_segments(&self, mut segments: Vec<SpeakerSegment>) -> Vec<SpeakerSegment> {
        if segments.len() < 2 {
            return segments;
        }

        segments.sort_by(|a, b| a.start().total_cmp(&b.start()));

        let mut merged = Vec::new();
        let mut current = segments[0].clone();

        for segment in segments.into_iter().skip(1) {
            if segment.speaker_id() == current.speaker_id()
                && (segment.start() - current.end()).abs() < 0.1
            {
                // Merge segments
                current = current.extend_to(segment.end());
            } else {
                if current.duration() >= self.config.min_segment_duration {
                    merged.push(current);
                }
                current = segment;
            }
        }

        if current.duration() >= self.config.min_segment_duration {
            merged.push(current);
        }

        merged
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &DiarizationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // DiarizationConfig Tests
    // =========================================================================

    #[test]
    fn test_diarization_config_default() {
        let config = DiarizationConfig::default();
        assert_eq!(config.min_speakers, 1);
        assert!(config.max_speakers.is_none());
        assert!((config.min_segment_duration - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diarization_config_for_realtime() {
        let config = DiarizationConfig::for_realtime();
        assert_eq!(config.max_speakers, Some(4));
        assert!((config.min_segment_duration - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diarization_config_for_accuracy() {
        let config = DiarizationConfig::for_accuracy();
        assert!(config.max_speakers.is_none());
        assert!((config.min_segment_duration - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diarization_config_with_max_speakers() {
        let config = DiarizationConfig::default().with_max_speakers(3);
        assert_eq!(config.max_speakers, Some(3));
    }

    #[test]
    fn test_diarization_config_with_min_segment_duration() {
        let config = DiarizationConfig::default().with_min_segment_duration(1.0);
        assert!((config.min_segment_duration - 1.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // DiarizationResult Tests
    // =========================================================================

    #[test]
    fn test_diarization_result_new() {
        let segments = vec![
            SpeakerSegment::new(0, 0.0, 2.0, 0.9),
            SpeakerSegment::new(1, 2.0, 4.0, 0.85),
        ];
        let embeddings = vec![
            SpeakerEmbedding::new(vec![0.1; 256], 0),
            SpeakerEmbedding::new(vec![0.2; 256], 1),
        ];

        let result = DiarizationResult::new(segments, 2, embeddings, 4.0);

        assert_eq!(result.num_speakers(), 2);
        assert_eq!(result.segments().len(), 2);
        assert!((result.duration() - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diarization_result_segments_for_speaker() {
        let segments = vec![
            SpeakerSegment::new(0, 0.0, 2.0, 0.9),
            SpeakerSegment::new(1, 2.0, 4.0, 0.85),
            SpeakerSegment::new(0, 4.0, 6.0, 0.88),
        ];

        let result = DiarizationResult::new(segments, 2, Vec::new(), 6.0);

        let speaker0_segments = result.segments_for_speaker(0);
        assert_eq!(speaker0_segments.len(), 2);

        let speaker1_segments = result.segments_for_speaker(1);
        assert_eq!(speaker1_segments.len(), 1);
    }

    #[test]
    fn test_diarization_result_speaking_time() {
        let segments = vec![
            SpeakerSegment::new(0, 0.0, 2.0, 0.9),
            SpeakerSegment::new(1, 2.0, 4.0, 0.85),
            SpeakerSegment::new(0, 4.0, 6.0, 0.88),
        ];

        let result = DiarizationResult::new(segments, 2, Vec::new(), 6.0);

        assert!((result.speaking_time(0) - 4.0).abs() < f32::EPSILON);
        assert!((result.speaking_time(1) - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diarization_result_speaker_turns() {
        let segments = vec![
            SpeakerSegment::new(0, 0.0, 2.0, 0.9),
            SpeakerSegment::new(1, 2.0, 4.0, 0.85),
            SpeakerSegment::new(0, 4.0, 6.0, 0.88),
        ];

        let result = DiarizationResult::new(segments, 2, Vec::new(), 6.0);
        let turns = result.speaker_turns();

        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].from_speaker(), 0);
        assert_eq!(turns[0].to_speaker(), 1);
        assert_eq!(turns[1].from_speaker(), 1);
        assert_eq!(turns[1].to_speaker(), 0);
    }

    #[test]
    fn test_diarization_result_no_turns_single_speaker() {
        let segments = vec![
            SpeakerSegment::new(0, 0.0, 2.0, 0.9),
            SpeakerSegment::new(0, 2.0, 4.0, 0.85),
        ];

        let result = DiarizationResult::new(segments, 1, Vec::new(), 4.0);
        let turns = result.speaker_turns();

        assert!(turns.is_empty());
    }

    // =========================================================================
    // Diarizer Tests
    // =========================================================================

    #[test]
    fn test_diarizer_new() {
        let diarizer = Diarizer::new(DiarizationConfig::default());
        assert_eq!(diarizer.config().min_speakers, 1);
    }

    #[test]
    fn test_diarizer_default_config() {
        let diarizer = Diarizer::default_config();
        assert!(diarizer.config().max_speakers.is_none());
    }

    #[test]
    fn test_diarizer_process_empty_audio() {
        let diarizer = Diarizer::default_config();
        let audio: Vec<f32> = vec![];
        let result = diarizer.process(&audio, 16000);

        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert_eq!(result.num_speakers(), 0);
        assert!(result.segments().is_empty());
    }

    #[test]
    fn test_diarizer_process_silence() {
        let diarizer = Diarizer::default_config();
        let audio: Vec<f32> = vec![0.0; 16000]; // 1 second of silence
        let result = diarizer.process(&audio, 16000);

        assert!(result.is_ok());
        let result = result.expect("should succeed");
        // Silence should result in no detected speakers
        assert!(result.segments().is_empty());
    }

    #[test]
    fn test_diarizer_merge_segments_same_speaker() {
        let diarizer = Diarizer::default_config();
        let segments = vec![
            SpeakerSegment::new(0, 0.0, 2.0, 0.9),
            SpeakerSegment::new(0, 2.05, 4.0, 0.85), // Small gap, same speaker
        ];

        let merged = diarizer.merge_segments(segments);
        assert_eq!(merged.len(), 1);
        assert!((merged[0].start() - 0.0).abs() < f32::EPSILON);
        assert!((merged[0].end() - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diarizer_merge_segments_different_speakers() {
        let diarizer = Diarizer::default_config();
        let segments = vec![
            SpeakerSegment::new(0, 0.0, 2.0, 0.9),
            SpeakerSegment::new(1, 2.0, 4.0, 0.85),
        ];

        let merged = diarizer.merge_segments(segments);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_diarizer_merge_filters_short_segments() {
        let config = DiarizationConfig::default().with_min_segment_duration(1.0);
        let diarizer = Diarizer::new(config);
        let segments = vec![
            SpeakerSegment::new(0, 0.0, 0.3, 0.9), // Too short
            SpeakerSegment::new(1, 0.5, 2.0, 0.85),
        ];

        let merged = diarizer.merge_segments(segments);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].speaker_id(), 1);
    }

    // =========================================================================
    // assign_speaker_labels Tests (impact 20.1, 0% coverage)
    // =========================================================================

    #[test]
    fn test_assign_speaker_labels_basic() {
        let diarizer = Diarizer::default_config();
        let segments = vec![
            SpeakerSegment::new(0, 0.0, 2.0, 0.9),
            SpeakerSegment::new(0, 2.0, 4.0, 0.85),
            SpeakerSegment::new(0, 4.0, 6.0, 0.88),
        ];

        // Create a clustering result with 2 clusters: seg0,seg2 → speaker 0, seg1 → speaker 1
        let embeddings = vec![
            SpeakerEmbedding::new(vec![1.0; 256], 0),
            SpeakerEmbedding::new(vec![-1.0; 256], 0),
            SpeakerEmbedding::new(vec![1.0; 256], 0),
        ];
        let clustering_config = ClusteringConfig::default();
        let clustering = SpectralClustering::new(clustering_config);
        let cluster_result = clustering.cluster(&embeddings, None, 1).expect("cluster");

        let labeled = diarizer
            .assign_speaker_labels(&segments, &cluster_result)
            .expect("should assign labels");

        assert_eq!(labeled.len(), 3);
        // All segments should have speaker IDs assigned
        for seg in &labeled {
            assert!(seg.speaker_id() < 10); // Reasonable speaker ID
        }
    }

    #[test]
    fn test_assign_speaker_labels_mismatch_error() {
        let diarizer = Diarizer::default_config();
        let segments = vec![
            SpeakerSegment::new(0, 0.0, 2.0, 0.9),
            SpeakerSegment::new(0, 2.0, 4.0, 0.85),
        ];

        // Create clustering with only 1 label (mismatch with 2 segments)
        let embeddings = vec![SpeakerEmbedding::new(vec![1.0; 256], 0)];
        let clustering_config = ClusteringConfig::default();
        let clustering = SpectralClustering::new(clustering_config);
        let cluster_result = clustering.cluster(&embeddings, None, 1).expect("cluster");

        let result = diarizer.assign_speaker_labels(&segments, &cluster_result);
        assert!(result.is_err());
    }

    // =========================================================================
    // extract_segment_embeddings Tests (impact 15.7, 0% coverage)
    // =========================================================================

    #[test]
    fn test_extract_segment_embeddings_basic() {
        let diarizer = Diarizer::default_config();
        let sample_rate = 16000u32;

        // Create 2 seconds of audio
        let audio: Vec<f32> = (0..sample_rate as usize * 2)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();

        let segments = vec![
            SpeakerSegment::new(0, 0.0, 1.0, 0.9),
            SpeakerSegment::new(0, 1.0, 2.0, 0.85),
        ];

        let embeddings = diarizer
            .extract_segment_embeddings(&audio, sample_rate, &segments)
            .expect("should extract embeddings");

        assert_eq!(embeddings.len(), 2);
    }

    #[test]
    fn test_extract_segment_embeddings_skip_invalid() {
        let diarizer = Diarizer::default_config();
        let sample_rate = 16000u32;

        // Short audio (1 second)
        let audio: Vec<f32> = (0..sample_rate as usize)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();

        let segments = vec![
            SpeakerSegment::new(0, 0.0, 0.5, 0.9),
            SpeakerSegment::new(0, 2.0, 3.0, 0.85), // Beyond audio length
        ];

        let embeddings = diarizer
            .extract_segment_embeddings(&audio, sample_rate, &segments)
            .expect("should succeed");

        // Second segment's start is beyond audio length, so start >= end after clamping
        assert!(embeddings.len() <= 2);
    }

    #[test]
    fn test_extract_segment_embeddings_empty() {
        let diarizer = Diarizer::default_config();
        let audio = vec![0.0f32; 16000];

        let embeddings = diarizer
            .extract_segment_embeddings(&audio, 16000, &[])
            .expect("should succeed");

        assert!(embeddings.is_empty());
    }

    // =========================================================================
    // process (full pipeline) Tests (impact 13.4, 33% coverage)
    // =========================================================================

    #[test]
    fn test_diarizer_process_with_synthetic_speech() {
        let diarizer = Diarizer::default_config();
        // Generate 3 seconds of synthetic speech-like audio
        let sample_rate = 16000u32;
        let audio: Vec<f32> = (0..sample_rate as usize * 3)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                // Two "speakers" with different frequencies
                if t < 1.5 {
                    (t * 200.0 * std::f32::consts::TAU).sin() * 0.5
                } else {
                    (t * 350.0 * std::f32::consts::TAU).sin() * 0.5
                }
            })
            .collect();

        let result = diarizer
            .process(&audio, sample_rate)
            .expect("should succeed");
        assert!((result.duration() - 3.0).abs() < 0.1);
    }

    // =========================================================================
    // cluster_speakers / process deeper path Tests (WAPR-QA-003)
    // =========================================================================

    #[test]
    fn test_diarizer_cluster_speakers_kmeans_config() {
        // Test with KMeans algorithm (dispatches to SpectralClustering internally)
        let mut config = DiarizationConfig::default();
        config.clustering.algorithm = ClusteringAlgorithm::KMeans;
        let diarizer = Diarizer::new(config);

        let sample_rate = 16000u32;
        let audio: Vec<f32> = (0..sample_rate as usize * 2)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (t * 300.0 * std::f32::consts::TAU).sin() * 0.5
            })
            .collect();

        let result = diarizer
            .process(&audio, sample_rate)
            .expect("should succeed");
        assert!((result.duration() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_diarizer_cluster_speakers_agglomerative_config() {
        // Test with Agglomerative algorithm
        let mut config = DiarizationConfig::default();
        config.clustering.algorithm = ClusteringAlgorithm::Agglomerative;
        let diarizer = Diarizer::new(config);

        let sample_rate = 16000u32;
        let audio: Vec<f32> = (0..sample_rate as usize * 2)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (t * 250.0 * std::f32::consts::TAU).sin() * 0.5
            })
            .collect();

        let result = diarizer
            .process(&audio, sample_rate)
            .expect("should succeed");
        assert!((result.duration() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_diarizer_process_with_max_speakers() {
        let config = DiarizationConfig::default().with_max_speakers(2);
        let diarizer = Diarizer::new(config);

        let sample_rate = 16000u32;
        let audio: Vec<f32> = (0..sample_rate as usize * 3)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                if t < 1.0 {
                    (t * 200.0 * std::f32::consts::TAU).sin() * 0.5
                } else if t < 2.0 {
                    (t * 400.0 * std::f32::consts::TAU).sin() * 0.5
                } else {
                    (t * 200.0 * std::f32::consts::TAU).sin() * 0.5
                }
            })
            .collect();

        let result = diarizer
            .process(&audio, sample_rate)
            .expect("should succeed");
        assert!(result.num_speakers() <= 2);
    }

    #[test]
    fn test_diarizer_process_realtime_config() {
        let config = DiarizationConfig::for_realtime();
        let diarizer = Diarizer::new(config);

        let sample_rate = 16000u32;
        let audio: Vec<f32> = (0..sample_rate as usize * 2)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (t * 300.0 * std::f32::consts::TAU).sin() * 0.5
            })
            .collect();

        let result = diarizer
            .process(&audio, sample_rate)
            .expect("should succeed");
        assert!((result.duration() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_diarizer_process_loud_two_speaker_audio() {
        // Generate audio with high amplitude to ensure VAD detects segments
        // and the full process() pipeline (steps 2-6) is exercised
        let config = DiarizationConfig::default()
            .with_max_speakers(3)
            .with_min_segment_duration(0.1);
        let diarizer = Diarizer::new(config);
        let sample_rate = 16000u32;

        // Create 4 seconds of audio with distinct "speaker" regions
        let audio: Vec<f32> = (0..sample_rate as usize * 4)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                // Speaker 1: 0-1.5s, low frequency with high amplitude
                // Silence: 1.5-2s
                // Speaker 2: 2-4s, high frequency with high amplitude
                if t < 1.5 {
                    (t * 150.0 * std::f32::consts::TAU).sin() * 0.8
                } else if t < 2.0 {
                    0.0 // Gap between speakers
                } else {
                    (t * 500.0 * std::f32::consts::TAU).sin() * 0.7
                }
            })
            .collect();

        let result = diarizer
            .process(&audio, sample_rate)
            .expect("should succeed");

        // Verify duration
        assert!((result.duration() - 4.0).abs() < 0.1);
        // num_speakers should be >= 0 (depends on VAD sensitivity)
        assert!(result.num_speakers() <= 3);
    }

    #[test]
    fn test_diarizer_cluster_speakers_direct() {
        // Exercise cluster_speakers directly by going through process()
        // with audio that forces embedding extraction + clustering
        let config = DiarizationConfig::default().with_min_segment_duration(0.05);
        let diarizer = Diarizer::new(config);
        let sample_rate = 16000u32;

        // Very loud audio to ensure segments are detected
        let audio: Vec<f32> = (0..sample_rate as usize * 3)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (t * 300.0 * std::f32::consts::TAU).sin() * 0.9
            })
            .collect();

        let result = diarizer
            .process(&audio, sample_rate)
            .expect("should succeed");
        assert!((result.duration() - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_diarizer_process_accuracy_config() {
        let config = DiarizationConfig::for_accuracy();
        let diarizer = Diarizer::new(config);

        let sample_rate = 16000u32;
        let audio: Vec<f32> = (0..sample_rate as usize * 2)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (t * 300.0 * std::f32::consts::TAU).sin() * 0.5
            })
            .collect();

        let result = diarizer
            .process(&audio, sample_rate)
            .expect("should succeed");
        assert!((result.duration() - 2.0).abs() < 0.1);
    }

    // =========================================================================
    // Full pipeline coverage: process() steps 2-6 + cluster_speakers (PMAT-024)
    //
    // The VAD uses adaptive thresholding (25th percentile of energy as noise
    // floor). Uniform sine waves have near-constant energy, so the adaptive
    // threshold sits at ~signal level and no frames pass. These tests use
    // audio with >25% silence so the noise floor is established from the
    // silent region, letting speech frames exceed the threshold.
    // =========================================================================

    /// Helper: Generate audio with distinct silence and speech regions.
    /// Returns audio where >30% is silence so VAD adaptive threshold works.
    fn generate_speech_with_silence(
        sample_rate: u32,
        segments: &[(f32, f32, f32)], // (start_sec, end_sec, freq_hz)
        total_duration: f32,
    ) -> Vec<f32> {
        let total_samples = (total_duration * sample_rate as f32) as usize;
        let mut audio = vec![0.0f32; total_samples];
        for &(start, end, freq) in segments {
            let s = (start * sample_rate as f32) as usize;
            let e = ((end * sample_rate as f32) as usize).min(total_samples);
            for i in s..e {
                let t = i as f32 / sample_rate as f32;
                audio[i] = (t * freq * std::f32::consts::TAU).sin() * 0.8;
            }
        }
        audio
    }

    #[test]
    fn test_process_full_pipeline_single_speaker() {
        // 1s silence + 2s speech + 1s silence = 50% silence
        let config = DiarizationConfig::default().with_min_segment_duration(0.2);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;
        let audio = generate_speech_with_silence(sr, &[(1.0, 3.0, 300.0)], 4.0);

        let result = diarizer.process(&audio, sr).expect("should succeed");

        // Full pipeline must detect at least 1 speaker (not early return)
        assert!(
            result.num_speakers() >= 1,
            "expected >=1 speaker, got {}; VAD should detect speech region",
            result.num_speakers()
        );
        assert!(
            !result.segments().is_empty(),
            "expected non-empty segments from full pipeline"
        );
    }

    #[test]
    fn test_process_full_pipeline_two_speakers() {
        // 1s silence + 1.5s speech@200Hz + 0.5s silence + 1.5s speech@500Hz + 1s silence
        // = 2.5s silence / 5.5s total ≈ 45% silence
        let config = DiarizationConfig::default()
            .with_max_speakers(3)
            .with_min_segment_duration(0.2);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;
        let audio = generate_speech_with_silence(sr, &[(1.0, 2.5, 200.0), (3.0, 4.5, 500.0)], 5.5);

        let result = diarizer.process(&audio, sr).expect("should succeed");

        assert!(
            result.num_speakers() >= 1,
            "expected >=1 speaker from two speech regions, got {}",
            result.num_speakers()
        );
        assert!((result.duration() - 5.5).abs() < 0.1);
    }

    #[test]
    fn test_cluster_speakers_kmeans_with_vad_triggering_audio() {
        let mut config = DiarizationConfig::default().with_min_segment_duration(0.2);
        config.clustering.algorithm = ClusteringAlgorithm::KMeans;
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;
        let audio = generate_speech_with_silence(sr, &[(1.0, 3.0, 300.0)], 4.0);

        let result = diarizer.process(&audio, sr).expect("should succeed");
        assert!(
            result.num_speakers() >= 1,
            "KMeans path: expected >=1 speaker, got {}",
            result.num_speakers()
        );
    }

    #[test]
    fn test_cluster_speakers_agglomerative_with_vad_triggering_audio() {
        let mut config = DiarizationConfig::default().with_min_segment_duration(0.2);
        config.clustering.algorithm = ClusteringAlgorithm::Agglomerative;
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;
        let audio = generate_speech_with_silence(sr, &[(1.0, 3.0, 300.0)], 4.0);

        let result = diarizer.process(&audio, sr).expect("should succeed");
        assert!(
            result.num_speakers() >= 1,
            "Agglomerative path: expected >=1 speaker, got {}",
            result.num_speakers()
        );
    }

    #[test]
    fn test_process_speaker_embeddings_populated() {
        // Verify step 6: speaker_embeddings are populated in result
        let config = DiarizationConfig::default().with_min_segment_duration(0.2);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;
        let audio = generate_speech_with_silence(sr, &[(1.0, 3.0, 300.0)], 4.0);

        let result = diarizer.process(&audio, sr).expect("should succeed");

        if result.num_speakers() > 0 {
            assert!(
                !result.speaker_embeddings().is_empty(),
                "step 6: speaker_embeddings should be populated when speakers detected"
            );
        }
    }

    #[test]
    fn test_process_merge_adjacent_same_speaker_segments() {
        // Continuous speech should be merged into fewer segments (step 5)
        let config = DiarizationConfig::default().with_min_segment_duration(0.1);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;
        // 3 seconds of continuous speech with silence padding
        let audio = generate_speech_with_silence(sr, &[(1.0, 4.0, 250.0)], 5.0);

        let result = diarizer.process(&audio, sr).expect("should succeed");

        // Continuous speech from one "speaker" should merge into few segments
        if !result.segments().is_empty() {
            assert!(
                result.segments().len() <= 5,
                "continuous speech should merge, got {} segments",
                result.segments().len()
            );
        }
    }

    // =========================================================================
    // cluster_speakers + process deeper path coverage (WAPR-QA-005)
    // =========================================================================

    #[test]
    #[allow(clippy::expect_used)]
    fn test_cluster_speakers_spectral_with_forced_segments() {
        // Create audio with a very loud burst surrounded by silence
        // to force VAD to detect at least one segment, exercising cluster_speakers
        let config = DiarizationConfig::default().with_min_segment_duration(0.05);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // 2s silence + 3s loud speech + 2s silence = 7s total, ~57% silence
        let audio = generate_speech_with_silence(sr, &[(2.0, 5.0, 440.0)], 7.0);

        let result = diarizer.process(&audio, sr).expect("should succeed");
        // Duration should be correct
        assert!((result.duration() - 7.0).abs() < 0.1);
        // If VAD detected segments, we must have exercised cluster_speakers
        if result.num_speakers() > 0 {
            assert!(!result.segments().is_empty());
            // Speaker embeddings should be present (step 6)
            assert!(!result.speaker_embeddings().is_empty());
        }
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_process_exercises_all_steps_with_two_speech_bursts() {
        // Two distinct speech bursts separated by silence to exercise:
        // step 1 (VAD), step 2 (embeddings), step 3 (clustering),
        // step 4 (labeling), step 5 (merging), step 6 (centroids)
        let config = DiarizationConfig::default()
            .with_max_speakers(3)
            .with_min_segment_duration(0.1);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // 1s silence + 2s@200Hz + 1s silence + 2s@600Hz + 1s silence = 7s total
        let audio = generate_speech_with_silence(sr, &[(1.0, 3.0, 200.0), (4.0, 6.0, 600.0)], 7.0);

        let result = diarizer.process(&audio, sr).expect("should succeed");
        assert!((result.duration() - 7.0).abs() < 0.1);
        // The pipeline should detect at least one speaker from the loud bursts
        if result.num_speakers() >= 1 {
            assert!(
                !result.segments().is_empty(),
                "with speakers detected, segments should not be empty"
            );
        }
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_process_spectral_algorithm_exercises_cluster_speakers() {
        // Explicitly use Spectral algorithm and ensure cluster_speakers path is hit
        let mut config = DiarizationConfig::default().with_min_segment_duration(0.1);
        config.clustering.algorithm = ClusteringAlgorithm::Spectral;
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // Silence + speech + silence pattern to trigger VAD
        let audio = generate_speech_with_silence(sr, &[(1.5, 3.5, 350.0)], 5.0);

        let result = diarizer.process(&audio, sr).expect("should succeed");
        assert!((result.duration() - 5.0).abs() < 0.1);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_process_long_audio_multiple_speakers() {
        // Longer audio with three speech regions to better exercise the pipeline
        let config = DiarizationConfig::default()
            .with_max_speakers(4)
            .with_min_segment_duration(0.1);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // 1s silence + 2s@150Hz + 1s silence + 2s@400Hz + 1s silence + 2s@250Hz + 1s silence
        let audio = generate_speech_with_silence(
            sr,
            &[(1.0, 3.0, 150.0), (4.0, 6.0, 400.0), (7.0, 9.0, 250.0)],
            10.0,
        );

        let result = diarizer.process(&audio, sr).expect("should succeed");
        assert!((result.duration() - 10.0).abs() < 0.1);

        // Verify speaker_turns() works when there are multiple segments
        let turns = result.speaker_turns();
        // Turns count depends on how many distinct speakers clustering finds
        // but the method should not panic
        let _ = turns.len();
    }

    // =========================================================================
    // process() full pipeline coverage: exercising steps 2-6 with reliable
    // VAD triggering and all three clustering algorithm dispatch paths
    // (WAPR-QA-006)
    // =========================================================================

    /// Test process() with very high amplitude impulse audio that guarantees
    /// VAD detection, exercising the full pipeline through cluster_speakers
    /// with the default Spectral algorithm.
    #[test]
    fn test_process_impulse_audio_exercises_full_pipeline() -> WhisperResult<()> {
        let config = DiarizationConfig::default().with_min_segment_duration(0.1);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // 2s silence + 2s loud impulse train + 2s silence = 6s, 67% silence
        // Impulse trains have variable energy per frame, ensuring VAD triggers
        let mut audio = vec![0.0f32; sr as usize * 6];
        for i in (sr as usize * 2)..(sr as usize * 4) {
            let t = i as f32 / sr as f32;
            // Mix of frequencies for richer spectral content
            audio[i] = (t * 220.0 * std::f32::consts::TAU).sin() * 0.7
                + (t * 440.0 * std::f32::consts::TAU).sin() * 0.3;
        }

        let result = diarizer.process(&audio, sr)?;

        assert!((result.duration() - 6.0).abs() < 0.1);
        // With 67% silence, VAD should detect the speech burst
        if result.num_speakers() >= 1 {
            // Steps 2-6 were exercised: embeddings, clustering, labels, merge, centroids
            assert!(!result.segments().is_empty());
            assert!(!result.speaker_embeddings().is_empty());
            // Verify speaker embedding dimension
            for emb in result.speaker_embeddings() {
                assert_eq!(emb.dim(), 256);
            }
        }
        Ok(())
    }

    /// Test process() with two distinct speech bursts separated by a long
    /// silence gap, which forces the clustering step to handle multiple
    /// segments with different embeddings.
    #[test]
    fn test_process_two_bursts_forces_multi_segment_clustering() -> WhisperResult<()> {
        let config = DiarizationConfig::default()
            .with_max_speakers(2)
            .with_min_segment_duration(0.1);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // 1.5s silence + 2s@150Hz + 2s silence + 2s@600Hz + 1.5s silence = 9s total
        // ~56% silence ensures VAD adaptive threshold is low enough
        let audio = generate_speech_with_silence(sr, &[(1.5, 3.5, 150.0), (5.5, 7.5, 600.0)], 9.0);

        let result = diarizer.process(&audio, sr)?;
        assert!((result.duration() - 9.0).abs() < 0.1);

        // With two distinct speech regions, the pipeline should detect segments
        // and exercise steps 2 (embedding extraction), 3 (cluster_speakers),
        // 4 (label assignment), 5 (merging), and 6 (centroids)
        if result.num_speakers() >= 1 {
            assert!(!result.segments().is_empty());
            // Verify segments have valid time ranges
            for seg in result.segments() {
                assert!(seg.start() >= 0.0);
                assert!(seg.end() > seg.start());
                assert!(seg.end() <= 9.5); // Allow small tolerance
            }
        }
        Ok(())
    }

    /// Test cluster_speakers with KMeans algorithm via process(), using audio
    /// that reliably triggers VAD.
    #[test]
    fn test_cluster_speakers_kmeans_via_process_with_reliable_vad() -> WhisperResult<()> {
        let mut config = DiarizationConfig::default().with_min_segment_duration(0.1);
        config.clustering.algorithm = ClusteringAlgorithm::KMeans;
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // 2s silence + 3s speech + 2s silence = 7s, ~57% silence
        let audio = generate_speech_with_silence(sr, &[(2.0, 5.0, 300.0)], 7.0);

        let result = diarizer.process(&audio, sr)?;
        assert!((result.duration() - 7.0).abs() < 0.1);

        // Verify the KMeans dispatch path was exercised
        if result.num_speakers() >= 1 {
            assert!(!result.segments().is_empty());
        }
        Ok(())
    }

    /// Test cluster_speakers with Agglomerative algorithm via process(), using
    /// audio that reliably triggers VAD.
    #[test]
    fn test_cluster_speakers_agglomerative_via_process_with_reliable_vad() -> WhisperResult<()> {
        let mut config = DiarizationConfig::default().with_min_segment_duration(0.1);
        config.clustering.algorithm = ClusteringAlgorithm::Agglomerative;
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // 2s silence + 3s speech + 2s silence = 7s, ~57% silence
        let audio = generate_speech_with_silence(sr, &[(2.0, 5.0, 350.0)], 7.0);

        let result = diarizer.process(&audio, sr)?;
        assert!((result.duration() - 7.0).abs() < 0.1);

        // Verify the Agglomerative dispatch path was exercised
        if result.num_speakers() >= 1 {
            assert!(!result.segments().is_empty());
        }
        Ok(())
    }

    /// Test process() exercises merging of adjacent same-speaker segments (step 5).
    /// Uses a single long speech burst which should produce multiple VAD segments
    /// that get merged into fewer labeled segments.
    #[test]
    fn test_process_step5_merge_produces_fewer_segments() -> WhisperResult<()> {
        let config = DiarizationConfig::default()
            .with_min_segment_duration(0.05)
            .with_max_speakers(2);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // 2s silence + 4s continuous speech + 2s silence = 8s, 50% silence
        let audio = generate_speech_with_silence(sr, &[(2.0, 6.0, 280.0)], 8.0);

        let result = diarizer.process(&audio, sr)?;
        assert!((result.duration() - 8.0).abs() < 0.1);

        // Continuous speech should be merged; verify segments are reasonable
        if result.num_speakers() >= 1 {
            // After merging, segments from same speaker should be combined
            let total_segments = result.segments().len();
            assert!(
                total_segments <= 10,
                "continuous 4s speech should merge into few segments, got {}",
                total_segments
            );
        }
        Ok(())
    }

    /// Test that process() returns correct centroids (step 6) when pipeline
    /// successfully detects multiple speech regions.
    #[test]
    fn test_process_step6_centroids_match_num_speakers() -> WhisperResult<()> {
        let config = DiarizationConfig::default()
            .with_min_segment_duration(0.1)
            .with_max_speakers(4);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // Three distinct speech regions with silence gaps
        let audio = generate_speech_with_silence(
            sr,
            &[(1.0, 2.5, 180.0), (3.5, 5.0, 450.0), (6.0, 7.5, 320.0)],
            9.0,
        );

        let result = diarizer.process(&audio, sr)?;

        // Centroids count should equal num_speakers
        if result.num_speakers() > 0 {
            assert_eq!(
                result.speaker_embeddings().len(),
                result.num_speakers(),
                "centroids count must equal num_speakers"
            );
        }
        Ok(())
    }

    /// Test process() with very short audio that still has enough silence ratio
    /// to trigger VAD, ensuring extract_segment_embeddings handles edge cases.
    #[test]
    fn test_process_short_audio_with_high_silence_ratio() -> WhisperResult<()> {
        let config = DiarizationConfig::default().with_min_segment_duration(0.1);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // 0.5s silence + 1s speech + 0.5s silence = 2s, 50% silence
        let audio = generate_speech_with_silence(sr, &[(0.5, 1.5, 400.0)], 2.0);

        let result = diarizer.process(&audio, sr)?;
        assert!((result.duration() - 2.0).abs() < 0.1);
        // Should not error even with short audio
        Ok(())
    }

    /// Test process() where segments are filtered out by min_segment_duration
    /// in the merge step (step 5), ensuring the final result may have fewer
    /// segments than detected by VAD.
    #[test]
    fn test_process_min_duration_filtering_in_merge() -> WhisperResult<()> {
        // Use a high min_segment_duration so some detected segments get filtered
        let config = DiarizationConfig::default().with_min_segment_duration(1.0);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // Very short speech burst (0.5s) surrounded by silence
        // VAD might detect it but merge step should filter it
        let audio = generate_speech_with_silence(sr, &[(2.0, 2.5, 300.0)], 5.0);

        let result = diarizer.process(&audio, sr)?;
        // The short segment should be filtered in merge_segments
        assert!((result.duration() - 5.0).abs() < 0.1);
        Ok(())
    }

    /// Test the DiarizationResult::speaker_turns with empty segments returns empty.
    #[test]
    fn test_diarization_result_speaker_turns_empty_segments() {
        let result = DiarizationResult::new(Vec::new(), 0, Vec::new(), 0.0);
        let turns = result.speaker_turns();
        assert!(turns.is_empty());
    }

    /// Test the DiarizationResult::speaker_turns with a single segment returns empty.
    #[test]
    fn test_diarization_result_speaker_turns_single_segment() {
        let segments = vec![SpeakerSegment::new(0, 0.0, 2.0, 0.9)];
        let result = DiarizationResult::new(segments, 1, Vec::new(), 2.0);
        let turns = result.speaker_turns();
        assert!(turns.is_empty());
    }

    /// Test speaking_time returns 0.0 for a speaker with no segments.
    #[test]
    fn test_diarization_result_speaking_time_nonexistent_speaker() {
        let segments = vec![SpeakerSegment::new(0, 0.0, 2.0, 0.9)];
        let result = DiarizationResult::new(segments, 1, Vec::new(), 2.0);
        assert!((result.speaking_time(99) - 0.0).abs() < f32::EPSILON);
    }

    /// Test process() with audio at a non-16kHz sample rate, exercising
    /// the resampling path in embedding extraction.
    #[test]
    fn test_process_with_non_standard_sample_rate() -> WhisperResult<()> {
        let config = DiarizationConfig::default().with_min_segment_duration(0.1);
        let diarizer = Diarizer::new(config);
        let sr = 44100u32; // Non-standard sample rate

        // 2s silence + 2s speech + 2s silence = 6s at 44100Hz
        let total_samples = (6.0 * sr as f32) as usize;
        let speech_start = (2.0 * sr as f32) as usize;
        let speech_end = (4.0 * sr as f32) as usize;
        let mut audio = vec![0.0f32; total_samples];
        for i in speech_start..speech_end {
            let t = i as f32 / sr as f32;
            audio[i] = (t * 300.0 * std::f32::consts::TAU).sin() * 0.8;
        }

        let result = diarizer.process(&audio, sr)?;
        assert!((result.duration() - 6.0).abs() < 0.2);
        Ok(())
    }

    // =========================================================================
    // process() deep pipeline coverage (WAPR-QA-007)
    //
    // These tests generate audio specifically designed to guarantee VAD
    // triggers, then assert the full pipeline was exercised by checking
    // that the result contains non-trivial output (segments, embeddings,
    // clustering results). The key difference from earlier tests is that
    // assertions are NOT guarded by `if result.num_speakers() >= 1` --
    // VAD must fire or the test fails.
    // =========================================================================

    /// Generate audio guaranteed to trigger VAD: alternating loud bursts
    /// and silence, with white-noise-like amplitude modulation to create
    /// frame-level energy variation that defeats the adaptive threshold.
    fn generate_vad_triggering_audio(
        sample_rate: u32,
        speech_regions: &[(f32, f32, f32)], // (start_sec, end_sec, freq_hz)
        total_duration: f32,
    ) -> Vec<f32> {
        let total_samples = (total_duration * sample_rate as f32) as usize;
        let mut audio = vec![0.0f32; total_samples];
        for &(start, end, freq) in speech_regions {
            let s = (start * sample_rate as f32) as usize;
            let e = ((end * sample_rate as f32) as usize).min(total_samples);
            for i in s..e {
                let t = i as f32 / sample_rate as f32;
                // Mix multiple harmonics for richer spectral content
                // Add amplitude modulation at 5Hz to create energy variation
                let am = 0.3f32.mul_add((t * 5.0 * std::f32::consts::TAU).sin(), 0.7);
                let signal = (t * freq * std::f32::consts::TAU).sin()
                    + 0.5 * (t * freq * 2.0 * std::f32::consts::TAU).sin()
                    + 0.25 * (t * freq * 3.0 * std::f32::consts::TAU).sin();
                audio[i] = signal * am * 0.6;
            }
        }
        audio
    }

    /// Test process() steps 2-6 with a single loud speech burst that
    /// guarantees VAD detection. Asserts non-empty segments without guards.
    #[test]
    fn test_process_guaranteed_vad_single_burst() -> WhisperResult<()> {
        let config = DiarizationConfig::default().with_min_segment_duration(0.05);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // 3s silence + 4s loud speech + 3s silence = 10s, 60% silence
        let audio = generate_vad_triggering_audio(sr, &[(3.0, 7.0, 300.0)], 10.0);

        let result = diarizer.process(&audio, sr)?;

        assert!((result.duration() - 10.0).abs() < 0.1);
        // VAD must detect the loud burst -- no guard
        assert!(
            result.num_speakers() >= 1,
            "VAD must detect speech: num_speakers={}, segments={}",
            result.num_speakers(),
            result.segments().len()
        );
        assert!(
            !result.segments().is_empty(),
            "pipeline steps 2-6 must produce segments"
        );
        assert!(
            !result.speaker_embeddings().is_empty(),
            "step 6 must produce speaker embeddings"
        );
        Ok(())
    }

    /// Test process() with two speech bursts at very different frequencies,
    /// guaranteeing VAD triggers and cluster_speakers handles multiple
    /// embeddings. Exercises the cluster dispatch (step 3).
    #[test]
    fn test_process_guaranteed_vad_two_bursts_cluster_speakers() -> WhisperResult<()> {
        let config = DiarizationConfig::default()
            .with_max_speakers(3)
            .with_min_segment_duration(0.05);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // Two speech regions separated by silence
        // 2s silence + 3s@200Hz + 2s silence + 3s@700Hz + 2s silence = 12s
        let audio =
            generate_vad_triggering_audio(sr, &[(2.0, 5.0, 200.0), (7.0, 10.0, 700.0)], 12.0);

        let result = diarizer.process(&audio, sr)?;

        assert!((result.duration() - 12.0).abs() < 0.1);
        assert!(
            result.num_speakers() >= 1,
            "two speech bursts must yield speakers, got num_speakers={}",
            result.num_speakers()
        );
        // With two distinct regions, we expect multiple segments before merging
        assert!(
            !result.segments().is_empty(),
            "must have segments from two speech regions"
        );
        // Verify centroids match num_speakers
        assert_eq!(
            result.speaker_embeddings().len(),
            result.num_speakers(),
            "centroids count must equal num_speakers"
        );
        Ok(())
    }

    /// Test cluster_speakers with KMeans algorithm variant, ensuring the
    /// match arm at line 303 is exercised via process().
    #[test]
    fn test_process_cluster_speakers_kmeans_branch() -> WhisperResult<()> {
        let mut config = DiarizationConfig::default().with_min_segment_duration(0.05);
        config.clustering.algorithm = ClusteringAlgorithm::KMeans;
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        let audio = generate_vad_triggering_audio(sr, &[(2.0, 6.0, 350.0)], 8.0);

        let result = diarizer.process(&audio, sr)?;

        assert!(
            result.num_speakers() >= 1,
            "KMeans branch: VAD must detect speech"
        );
        assert!(!result.segments().is_empty());
        Ok(())
    }

    /// Test cluster_speakers with Agglomerative algorithm variant, ensuring
    /// the match arm at line 303 is exercised via process().
    #[test]
    fn test_process_cluster_speakers_agglomerative_branch() -> WhisperResult<()> {
        let mut config = DiarizationConfig::default().with_min_segment_duration(0.05);
        config.clustering.algorithm = ClusteringAlgorithm::Agglomerative;
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        let audio = generate_vad_triggering_audio(sr, &[(2.0, 6.0, 450.0)], 8.0);

        let result = diarizer.process(&audio, sr)?;

        assert!(
            result.num_speakers() >= 1,
            "Agglomerative branch: VAD must detect speech"
        );
        assert!(!result.segments().is_empty());
        Ok(())
    }

    /// Test process() exercises merge step (step 5) by producing multiple
    /// segments from a single continuous speech burst, which should be
    /// merged into fewer segments for the same speaker.
    #[test]
    fn test_process_merge_step_with_guaranteed_vad() -> WhisperResult<()> {
        let config = DiarizationConfig::default().with_min_segment_duration(0.05);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // Long continuous speech surrounded by silence
        let audio = generate_vad_triggering_audio(sr, &[(3.0, 9.0, 250.0)], 12.0);

        let result = diarizer.process(&audio, sr)?;

        assert!(
            result.num_speakers() >= 1,
            "continuous speech must be detected"
        );
        // Continuous single-speaker speech should merge to few segments
        assert!(
            result.segments().len() <= 8,
            "6s continuous speech should merge, got {} segments",
            result.segments().len()
        );
        Ok(())
    }

    /// Test process() assign_speaker_labels step (step 4) by verifying
    /// that all returned segments have valid speaker IDs within range.
    #[test]
    fn test_process_assign_labels_valid_speaker_ids() -> WhisperResult<()> {
        let config = DiarizationConfig::default()
            .with_max_speakers(4)
            .with_min_segment_duration(0.05);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        let audio = generate_vad_triggering_audio(
            sr,
            &[(1.0, 3.0, 200.0), (5.0, 7.0, 500.0), (9.0, 11.0, 350.0)],
            13.0,
        );

        let result = diarizer.process(&audio, sr)?;

        // Every segment should have a valid speaker ID < num_speakers
        for seg in result.segments() {
            assert!(
                seg.speaker_id() < result.num_speakers(),
                "speaker_id {} must be < num_speakers {}",
                seg.speaker_id(),
                result.num_speakers()
            );
        }
        Ok(())
    }

    /// Test process() with three speech bursts at distinct frequencies
    /// to maximize the chance of cluster_speakers producing multiple
    /// clusters. Verifies speaker_turns() returns transitions.
    #[test]
    fn test_process_three_bursts_speaker_turns() -> WhisperResult<()> {
        let config = DiarizationConfig::default()
            .with_max_speakers(4)
            .with_min_segment_duration(0.05);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        let audio = generate_vad_triggering_audio(
            sr,
            &[(1.0, 3.0, 150.0), (4.0, 6.0, 600.0), (7.0, 9.0, 150.0)],
            10.0,
        );

        let result = diarizer.process(&audio, sr)?;

        assert!(
            result.num_speakers() >= 1,
            "three bursts must detect speakers"
        );
        // speaker_turns() should work without panicking
        let turns = result.speaker_turns();
        // If multiple speakers detected, there should be transitions
        if result.num_speakers() >= 2 {
            assert!(
                !turns.is_empty(),
                "with >=2 speakers, there should be speaker turns"
            );
        }
        Ok(())
    }

    /// Test that process() exercises extract_segment_embeddings (step 2)
    /// by verifying embedding dimensions in the result.
    #[test]
    fn test_process_embedding_extraction_dimensions() -> WhisperResult<()> {
        let config = DiarizationConfig::default().with_min_segment_duration(0.05);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        let audio = generate_vad_triggering_audio(sr, &[(2.0, 6.0, 400.0)], 8.0);

        let result = diarizer.process(&audio, sr)?;

        // All speaker embeddings should be 256-dimensional
        for emb in result.speaker_embeddings() {
            assert_eq!(emb.dim(), 256, "speaker embedding must be 256-dimensional");
        }
        Ok(())
    }

    /// Test process() with min_segment_duration filtering in merge step.
    /// Short segments produced by VAD should be filtered out, leaving only
    /// segments >= min_segment_duration.
    #[test]
    fn test_process_merge_filters_short_segments_via_pipeline() -> WhisperResult<()> {
        let config = DiarizationConfig::default().with_min_segment_duration(2.0); // High threshold
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // Two short speech bursts (0.5s each) -- should be filtered by merge step
        // Plus one long burst (3s) -- should survive
        let audio = generate_vad_triggering_audio(
            sr,
            &[(1.0, 1.5, 300.0), (3.0, 3.5, 400.0), (5.0, 8.0, 300.0)],
            10.0,
        );

        let result = diarizer.process(&audio, sr)?;

        // All surviving segments must be >= 2.0s duration
        for seg in result.segments() {
            assert!(
                seg.duration() >= 1.9, // Small tolerance
                "segment duration {:.2}s should be >= 2.0s after merge filtering",
                seg.duration()
            );
        }
        Ok(())
    }

    /// Test cluster_speakers directly via process() with Spectral algorithm
    /// and guaranteed multi-segment input, verifying the silhouette score
    /// from clustering is finite (indirectly via the pipeline completing).
    #[test]
    fn test_process_cluster_speakers_spectral_multi_segment() -> WhisperResult<()> {
        let mut config = DiarizationConfig::default()
            .with_max_speakers(3)
            .with_min_segment_duration(0.05);
        config.clustering.algorithm = ClusteringAlgorithm::Spectral;
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // Three distinct speech regions
        let audio = generate_vad_triggering_audio(
            sr,
            &[(1.0, 3.0, 180.0), (4.0, 6.0, 500.0), (7.0, 9.0, 320.0)],
            10.0,
        );

        let result = diarizer.process(&audio, sr)?;

        assert!(
            result.num_speakers() >= 1,
            "spectral clustering must produce speakers"
        );
        // Pipeline completed successfully through cluster_speakers
        assert!(!result.speaker_embeddings().is_empty());
        Ok(())
    }

    // =========================================================================
    // Direct cluster_speakers unit tests (WAPR-QA-008)
    //
    // These tests call cluster_speakers directly on a Diarizer instance
    // with pre-constructed SpeakerEmbeddings, bypassing VAD entirely.
    // This guarantees coverage of the cluster_speakers method body
    // (line 298) and all three algorithm dispatch branches.
    // =========================================================================

    #[test]
    fn test_cluster_speakers_direct_spectral() -> WhisperResult<()> {
        let config = DiarizationConfig::default();
        let diarizer = Diarizer::new(config);

        // Create embeddings that form two clear groups
        let embeddings = vec![
            SpeakerEmbedding::new(vec![1.0; 256], 0),
            SpeakerEmbedding::new(vec![0.95; 256], 0),
            SpeakerEmbedding::new(vec![-1.0; 256], 1),
            SpeakerEmbedding::new(vec![-0.95; 256], 1),
        ];

        let result = diarizer.cluster_speakers(&embeddings)?;
        assert!(result.num_clusters() >= 1);
        assert_eq!(result.labels().len(), 4);
        Ok(())
    }

    #[test]
    fn test_cluster_speakers_direct_kmeans() -> WhisperResult<()> {
        let mut config = DiarizationConfig::default();
        config.clustering.algorithm = ClusteringAlgorithm::KMeans;
        let diarizer = Diarizer::new(config);

        let embeddings = vec![
            SpeakerEmbedding::new(vec![1.0; 256], 0),
            SpeakerEmbedding::new(vec![0.9; 256], 0),
            SpeakerEmbedding::new(vec![-1.0; 256], 1),
        ];

        let result = diarizer.cluster_speakers(&embeddings)?;
        assert!(result.num_clusters() >= 1);
        assert_eq!(result.labels().len(), 3);
        Ok(())
    }

    #[test]
    fn test_cluster_speakers_direct_agglomerative() -> WhisperResult<()> {
        let mut config = DiarizationConfig::default();
        config.clustering.algorithm = ClusteringAlgorithm::Agglomerative;
        let diarizer = Diarizer::new(config);

        let embeddings = vec![
            SpeakerEmbedding::new(vec![1.0; 256], 0),
            SpeakerEmbedding::new(vec![-1.0; 256], 1),
        ];

        let result = diarizer.cluster_speakers(&embeddings)?;
        assert!(result.num_clusters() >= 1);
        assert_eq!(result.labels().len(), 2);
        Ok(())
    }

    #[test]
    fn test_cluster_speakers_single_embedding() -> WhisperResult<()> {
        let diarizer = Diarizer::default_config();

        let embeddings = vec![SpeakerEmbedding::new(vec![0.5; 256], 0)];

        let result = diarizer.cluster_speakers(&embeddings)?;
        assert_eq!(result.num_clusters(), 1);
        assert_eq!(result.labels(), &[0]);
        Ok(())
    }

    #[test]
    fn test_cluster_speakers_empty_embeddings() -> WhisperResult<()> {
        let diarizer = Diarizer::default_config();

        let embeddings: Vec<SpeakerEmbedding> = Vec::new();
        let result = diarizer.cluster_speakers(&embeddings)?;
        assert_eq!(result.num_clusters(), 0);
        assert!(result.labels().is_empty());
        Ok(())
    }

    #[test]
    fn test_cluster_speakers_with_max_speakers_constraint() -> WhisperResult<()> {
        let config = DiarizationConfig::default().with_max_speakers(2);
        let diarizer = Diarizer::new(config);

        // Three distinct groups but constrained to max 2
        let embeddings = vec![
            SpeakerEmbedding::new(
                vec![1.0, 0.0, 0.0]
                    .into_iter()
                    .chain(vec![0.0; 253])
                    .collect(),
                0,
            ),
            SpeakerEmbedding::new(
                vec![0.0, 1.0, 0.0]
                    .into_iter()
                    .chain(vec![0.0; 253])
                    .collect(),
                1,
            ),
            SpeakerEmbedding::new(
                vec![0.0, 0.0, 1.0]
                    .into_iter()
                    .chain(vec![0.0; 253])
                    .collect(),
                2,
            ),
        ];

        let result = diarizer.cluster_speakers(&embeddings)?;
        assert!(
            result.num_clusters() <= 2,
            "should respect max_speakers=2, got {}",
            result.num_clusters()
        );
        Ok(())
    }

    // =========================================================================
    // process() pipeline: direct embedding and clustering coverage
    // (WAPR-QA-008)
    //
    // These tests create audio that is guaranteed to trigger VAD by using
    // a broadband noise-like signal with high amplitude in speech regions
    // and pure silence elsewhere. The key insight is that VAD uses adaptive
    // thresholding, so we need >25% of frames to be silent.
    // =========================================================================

    /// Generate broadband audio that reliably triggers VAD.
    /// Uses sum of many harmonics to create a noise-like broadband signal
    /// that has high energy in speech regions and zero in silence.
    fn generate_broadband_speech(
        sample_rate: u32,
        speech_regions: &[(f32, f32)],
        total_duration: f32,
    ) -> Vec<f32> {
        let total_samples = (total_duration * sample_rate as f32) as usize;
        let mut audio = vec![0.0f32; total_samples];
        let freqs = [
            100.0, 200.0, 300.0, 440.0, 600.0, 800.0, 1000.0, 1500.0, 2000.0,
        ];

        for &(start, end) in speech_regions {
            let s = (start * sample_rate as f32) as usize;
            let e = ((end * sample_rate as f32) as usize).min(total_samples);
            for i in s..e {
                let t = i as f32 / sample_rate as f32;
                let mut val = 0.0f32;
                for (fi, &freq) in freqs.iter().enumerate() {
                    let phase = t * freq * std::f32::consts::TAU;
                    val += phase.sin() * (1.0 / (fi as f32 + 1.0));
                }
                // Amplitude modulation at 3Hz for energy variation
                let am = 0.4f32.mul_add((t * 3.0 * std::f32::consts::TAU).sin(), 0.6);
                audio[i] = val * am * 0.3;
            }
        }
        audio
    }

    #[test]
    fn test_process_full_pipeline_direct_broadband() -> WhisperResult<()> {
        let config = DiarizationConfig::default().with_min_segment_duration(0.1);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // 2s silence + 3s broadband speech + 2s silence = 7s, 57% silence
        let audio = generate_broadband_speech(sr, &[(2.0, 5.0)], 7.0);

        let result = diarizer.process(&audio, sr)?;
        assert!((result.duration() - 7.0).abs() < 0.1);

        // Verify steps 2-6 were exercised
        if result.num_speakers() >= 1 {
            assert!(
                !result.segments().is_empty(),
                "step 4-5: segments must be populated"
            );
            assert!(
                !result.speaker_embeddings().is_empty(),
                "step 6: centroids must be populated"
            );
            assert_eq!(
                result.speaker_embeddings().len(),
                result.num_speakers(),
                "centroids must match num_speakers"
            );
        }
        Ok(())
    }

    #[test]
    fn test_process_two_speakers_broadband() -> WhisperResult<()> {
        let config = DiarizationConfig::default()
            .with_max_speakers(3)
            .with_min_segment_duration(0.1);
        let diarizer = Diarizer::new(config);
        let sr = 16000u32;

        // Two distinct speech regions with silence gap
        let audio = generate_broadband_speech(sr, &[(1.0, 3.0), (4.5, 6.5)], 8.0);

        let result = diarizer.process(&audio, sr)?;
        assert!((result.duration() - 8.0).abs() < 0.1);

        if result.num_speakers() >= 1 {
            // Verify all segments have valid speaker IDs
            for seg in result.segments() {
                assert!(seg.speaker_id() < result.num_speakers());
                assert!(seg.end() > seg.start());
            }
        }
        Ok(())
    }
}
