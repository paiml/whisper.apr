//! WASM bindings for speaker diarization (WAPR-153)
//!
//! Provides JavaScript-friendly API for speaker diarization via wasm-bindgen.
//!
//! # Usage
//!
//! ```javascript
//! import { DiarizerWasm, DiarizationConfigWasm } from 'whisper-apr';
//!
//! // Create diarizer with default config
//! const diarizer = new DiarizerWasm();
//!
//! // Or with custom config
//! const config = new DiarizationConfigWasm();
//! config.setMaxSpeakers(4);
//! config.setMinSegmentDuration(0.5);
//! const diarizer = DiarizerWasm.withConfig(config);
//!
//! // Process audio and get speaker-labeled segments
//! const result = diarizer.process(audioFloat32Array, 16000);
//! console.log(`Found ${result.speakerCount} speakers`);
//!
//! for (let i = 0; i < result.segmentCount; i++) {
//!     const segment = result.getSegment(i);
//!     console.log(`Speaker ${segment.speakerId}: ${segment.start}s - ${segment.end}s`);
//! }
//! ```

use wasm_bindgen::prelude::*;

use crate::diarization::{
    clustering::{ClusteringAlgorithm, ClusteringConfig, SpectralClustering},
    embedding::{EmbeddingConfig, EmbeddingExtractor, SpeakerEmbedding},
    segmentation::{SegmentationConfig, SpeakerSegment, TurnDetector},
    DiarizationConfig, DiarizationResult, Diarizer,
};

/// WASM-friendly diarization configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DiarizationConfigWasm {
    max_speakers: Option<usize>,
    min_speakers: usize,
    min_segment_duration: f32,
    embedding_dim: usize,
    clustering_threshold: f32,
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
        DiarizationConfig::default()
            .with_max_speakers(wasm.max_speakers)
            .with_min_segment_duration(wasm.min_segment_duration)
    }
}

/// WASM-friendly speaker segment
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct SpeakerSegmentWasm {
    speaker_id: usize,
    start: f32,
    end: f32,
    confidence: f32,
}

#[wasm_bindgen]
impl SpeakerSegmentWasm {
    /// Get speaker ID (0-indexed)
    #[wasm_bindgen(getter, js_name = speakerId)]
    pub fn speaker_id(&self) -> usize {
        self.speaker_id
    }

    /// Get start time in seconds
    #[wasm_bindgen(getter)]
    pub fn start(&self) -> f32 {
        self.start
    }

    /// Get end time in seconds
    #[wasm_bindgen(getter)]
    pub fn end(&self) -> f32 {
        self.end
    }

    /// Get duration in seconds
    #[wasm_bindgen(getter)]
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }

    /// Get confidence score (0.0 - 1.0)
    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Get speaker label as string (e.g., "SPEAKER_0")
    #[wasm_bindgen(getter, js_name = speakerLabel)]
    pub fn speaker_label(&self) -> String {
        format!("SPEAKER_{}", self.speaker_id)
    }
}

impl From<SpeakerSegment> for SpeakerSegmentWasm {
    fn from(seg: SpeakerSegment) -> Self {
        Self {
            speaker_id: seg.speaker_id(),
            start: seg.start(),
            end: seg.end(),
            confidence: seg.confidence(),
        }
    }
}

/// WASM-friendly speaker embedding
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct SpeakerEmbeddingWasm {
    vector: Vec<f32>,
    speaker_id: usize,
    confidence: f32,
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
    pub fn cosine_similarity(&self, other: &SpeakerEmbeddingWasm) -> f32 {
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

/// WASM-friendly diarization result
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DiarizationResultWasm {
    segments: Vec<SpeakerSegmentWasm>,
    speaker_embeddings: Vec<SpeakerEmbeddingWasm>,
    speaker_count: usize,
    total_duration: f32,
}

#[wasm_bindgen]
impl DiarizationResultWasm {
    /// Get number of speakers detected
    #[wasm_bindgen(getter, js_name = speakerCount)]
    pub fn speaker_count(&self) -> usize {
        self.speaker_count
    }

    /// Get number of segments
    #[wasm_bindgen(getter, js_name = segmentCount)]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Get total audio duration in seconds
    #[wasm_bindgen(getter, js_name = totalDuration)]
    pub fn total_duration(&self) -> f32 {
        self.total_duration
    }

    /// Get a segment by index
    #[wasm_bindgen(js_name = getSegment)]
    pub fn get_segment(&self, index: usize) -> Option<SpeakerSegmentWasm> {
        self.segments.get(index).cloned()
    }

    /// Get all segments for a specific speaker
    #[wasm_bindgen(js_name = getSegmentsForSpeaker)]
    pub fn get_segments_for_speaker(&self, speaker_id: usize) -> Vec<SpeakerSegmentWasm> {
        self.segments
            .iter()
            .filter(|s| s.speaker_id == speaker_id)
            .cloned()
            .collect()
    }

    /// Get total speaking time for a speaker in seconds
    #[wasm_bindgen(js_name = getSpeakingTime)]
    pub fn get_speaking_time(&self, speaker_id: usize) -> f32 {
        self.segments
            .iter()
            .filter(|s| s.speaker_id == speaker_id)
            .map(|s| s.duration())
            .sum()
    }

    /// Get speaking percentage for a speaker (0.0 - 100.0)
    #[wasm_bindgen(js_name = getSpeakingPercentage)]
    pub fn get_speaking_percentage(&self, speaker_id: usize) -> f32 {
        if self.total_duration <= 0.0 {
            return 0.0;
        }
        (self.get_speaking_time(speaker_id) / self.total_duration) * 100.0
    }

    /// Get speaker embedding by speaker ID
    #[wasm_bindgen(js_name = getSpeakerEmbedding)]
    pub fn get_speaker_embedding(&self, speaker_id: usize) -> Option<SpeakerEmbeddingWasm> {
        self.speaker_embeddings
            .iter()
            .find(|e| e.speaker_id == speaker_id)
            .cloned()
    }

    /// Get all segment start times
    #[wasm_bindgen(js_name = segmentStarts)]
    pub fn segment_starts(&self) -> Vec<f32> {
        self.segments.iter().map(|s| s.start).collect()
    }

    /// Get all segment end times
    #[wasm_bindgen(js_name = segmentEnds)]
    pub fn segment_ends(&self) -> Vec<f32> {
        self.segments.iter().map(|s| s.end).collect()
    }

    /// Get all segment speaker IDs
    #[wasm_bindgen(js_name = segmentSpeakerIds)]
    pub fn segment_speaker_ids(&self) -> Vec<usize> {
        self.segments.iter().map(|s| s.speaker_id).collect()
    }

    /// Get number of speaker turns (segment transitions)
    #[wasm_bindgen(js_name = turnCount)]
    pub fn turn_count(&self) -> usize {
        if self.segments.is_empty() {
            return 0;
        }

        let mut turns = 0;
        let mut prev_speaker = self.segments[0].speaker_id;

        for seg in &self.segments[1..] {
            if seg.speaker_id != prev_speaker {
                turns += 1;
                prev_speaker = seg.speaker_id;
            }
        }

        turns
    }

    /// Export to JSON string
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> String {
        let segments: Vec<_> = self
            .segments
            .iter()
            .map(|s| {
                format!(
                    r#"{{"speaker_id":{},"start":{},"end":{},"confidence":{}}}"#,
                    s.speaker_id, s.start, s.end, s.confidence
                )
            })
            .collect();

        format!(
            r#"{{"speaker_count":{},"total_duration":{},"segments":[{}]}}"#,
            self.speaker_count,
            self.total_duration,
            segments.join(",")
        )
    }
}

impl From<DiarizationResult> for DiarizationResultWasm {
    fn from(result: DiarizationResult) -> Self {
        let segments: Vec<SpeakerSegmentWasm> =
            result.segments().iter().map(|s| s.clone().into()).collect();

        let speaker_embeddings: Vec<SpeakerEmbeddingWasm> = result
            .speaker_embeddings()
            .iter()
            .map(|e| e.clone().into())
            .collect();

        let total_duration = segments.iter().map(|s| s.end).fold(0.0f32, f32::max);

        Self {
            segments,
            speaker_embeddings,
            speaker_count: result.num_speakers(),
            total_duration,
        }
    }
}

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
    pub fn process(&self, audio: &[f32], sample_rate: u32) -> Result<DiarizationResultWasm, JsValue> {
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
    pub fn extract(&self, audio: &[f32], sample_rate: u32) -> Result<SpeakerEmbeddingWasm, JsValue> {
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

/// Get recommended diarization config for use case
#[wasm_bindgen(js_name = getDiarizationRecommendation)]
pub fn get_diarization_recommendation(use_case: &str) -> String {
    match use_case.to_lowercase().as_str() {
        "meeting" | "conference" | "interview" => {
            "Use forAccuracy() with max 8 speakers for meetings and interviews.".to_string()
        }
        "podcast" | "dialogue" | "conversation" => {
            "Use default config with max 4 speakers for podcasts and dialogues.".to_string()
        }
        "call" | "phone" | "telephony" => {
            "Use default config with max 2 speakers for phone calls.".to_string()
        }
        "realtime" | "live" | "streaming" => {
            "Use forRealtime() for live streaming with reduced latency.".to_string()
        }
        _ => "Unknown use case. Available: meeting, podcast, call, realtime.".to_string(),
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

    #[test]
    fn test_speaker_segment_wasm() {
        let segment = SpeakerSegmentWasm {
            speaker_id: 1,
            start: 0.5,
            end: 2.0,
            confidence: 0.85,
        };

        assert_eq!(segment.speaker_id(), 1);
        assert!((segment.start() - 0.5).abs() < f32::EPSILON);
        assert!((segment.end() - 2.0).abs() < f32::EPSILON);
        assert!((segment.duration() - 1.5).abs() < f32::EPSILON);
        assert!((segment.confidence() - 0.85).abs() < f32::EPSILON);
        assert_eq!(segment.speaker_label(), "SPEAKER_1");
    }

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
        let emb3 = SpeakerEmbeddingWasm {
            vector: vec![0.0, 1.0, 0.0],
            speaker_id: 2,
            confidence: 1.0,
        };

        // Identical vectors -> similarity 1.0
        assert!((emb1.cosine_similarity(&emb2) - 1.0).abs() < f32::EPSILON);

        // Orthogonal vectors -> similarity 0.0
        assert!((emb1.cosine_similarity(&emb3) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diarization_result_wasm_turn_count() {
        let result = DiarizationResultWasm {
            segments: vec![
                SpeakerSegmentWasm {
                    speaker_id: 0,
                    start: 0.0,
                    end: 1.0,
                    confidence: 1.0,
                },
                SpeakerSegmentWasm {
                    speaker_id: 1,
                    start: 1.0,
                    end: 2.0,
                    confidence: 1.0,
                },
                SpeakerSegmentWasm {
                    speaker_id: 0,
                    start: 2.0,
                    end: 3.0,
                    confidence: 1.0,
                },
            ],
            speaker_embeddings: vec![],
            speaker_count: 2,
            total_duration: 3.0,
        };

        assert_eq!(result.turn_count(), 2);
    }

    #[test]
    fn test_diarization_result_wasm_speaking_time() {
        let result = DiarizationResultWasm {
            segments: vec![
                SpeakerSegmentWasm {
                    speaker_id: 0,
                    start: 0.0,
                    end: 1.0,
                    confidence: 1.0,
                },
                SpeakerSegmentWasm {
                    speaker_id: 0,
                    start: 2.0,
                    end: 4.0,
                    confidence: 1.0,
                },
                SpeakerSegmentWasm {
                    speaker_id: 1,
                    start: 1.0,
                    end: 2.0,
                    confidence: 1.0,
                },
            ],
            speaker_embeddings: vec![],
            speaker_count: 2,
            total_duration: 4.0,
        };

        assert!((result.get_speaking_time(0) - 3.0).abs() < f32::EPSILON);
        assert!((result.get_speaking_time(1) - 1.0).abs() < f32::EPSILON);
        assert!((result.get_speaking_percentage(0) - 75.0).abs() < f32::EPSILON);
        assert!((result.get_speaking_percentage(1) - 25.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diarization_result_wasm_to_json() {
        let result = DiarizationResultWasm {
            segments: vec![SpeakerSegmentWasm {
                speaker_id: 0,
                start: 0.0,
                end: 1.0,
                confidence: 0.9,
            }],
            speaker_embeddings: vec![],
            speaker_count: 1,
            total_duration: 1.0,
        };

        let json = result.to_json();
        assert!(json.contains("\"speaker_count\":1"));
        assert!(json.contains("\"speaker_id\":0"));
    }

    #[test]
    fn test_get_diarization_recommendation() {
        let meeting = get_diarization_recommendation("meeting");
        assert!(meeting.contains("forAccuracy"));

        let call = get_diarization_recommendation("call");
        assert!(meeting.contains("speakers"));

        let realtime = get_diarization_recommendation("realtime");
        assert!(realtime.contains("forRealtime"));
    }

    #[test]
    fn test_diarizer_wasm_new() {
        let _diarizer = DiarizerWasm::new();
    }

    #[test]
    fn test_diarizer_wasm_with_config() {
        let config = DiarizationConfigWasm::for_realtime();
        let _diarizer = DiarizerWasm::with_config(config);
    }

    #[test]
    fn test_embedding_extractor_wasm_new() {
        let _extractor = EmbeddingExtractorWasm::new();
    }

    #[test]
    fn test_turn_detector_wasm_new() {
        let _detector = TurnDetectorWasm::new();
    }
}
