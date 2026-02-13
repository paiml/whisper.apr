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
mod tests;
