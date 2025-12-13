//! Word boundary detection with confidence (WAPR-161)
//!
//! Detects word boundaries from token alignments with confidence scoring.
//!
//! # Overview
//!
//! This module refines word boundaries by:
//! 1. Analyzing attention weight patterns
//! 2. Detecting silence gaps between words
//! 3. Computing confidence scores for boundaries
//! 4. Adjusting boundaries based on audio features

use crate::error::WhisperResult;

use super::alignment::TokenAlignment;

/// Boundary detection configuration
#[derive(Debug, Clone)]
pub struct BoundaryConfig {
    /// Minimum silence duration to split words (seconds)
    pub min_silence_duration: f32,
    /// Energy threshold for silence detection
    pub silence_threshold: f32,
    /// Minimum word duration (seconds)
    pub min_word_duration: f32,
    /// Maximum word duration (seconds)
    pub max_word_duration: f32,
    /// Use audio energy for refinement
    pub use_audio_energy: bool,
}

impl Default for BoundaryConfig {
    fn default() -> Self {
        Self {
            min_silence_duration: 0.05,
            silence_threshold: 0.01,
            min_word_duration: 0.05,
            max_word_duration: 5.0,
            use_audio_energy: true,
        }
    }
}

impl BoundaryConfig {
    /// Create config for precise boundary detection
    #[must_use]
    pub fn precise() -> Self {
        Self {
            min_silence_duration: 0.03,
            silence_threshold: 0.005,
            min_word_duration: 0.03,
            max_word_duration: 3.0,
            use_audio_energy: true,
        }
    }

    /// Create config for fast processing
    #[must_use]
    pub fn fast() -> Self {
        Self {
            min_silence_duration: 0.1,
            silence_threshold: 0.02,
            min_word_duration: 0.1,
            max_word_duration: 10.0,
            use_audio_energy: false,
        }
    }

    /// Set minimum silence duration
    #[must_use]
    pub fn with_min_silence(mut self, duration: f32) -> Self {
        self.min_silence_duration = duration;
        self
    }

    /// Set minimum word duration
    #[must_use]
    pub fn with_min_word_duration(mut self, duration: f32) -> Self {
        self.min_word_duration = duration;
        self
    }
}

/// Word boundary information
#[derive(Debug, Clone)]
pub struct WordBoundary {
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Confidence in start boundary (0.0 - 1.0)
    pub start_confidence: f32,
    /// Confidence in end boundary (0.0 - 1.0)
    pub end_confidence: f32,
    /// Whether boundary was refined using audio
    pub audio_refined: bool,
    /// Token indices covered
    pub token_indices: Vec<usize>,
}

impl WordBoundary {
    /// Create new word boundary
    #[must_use]
    pub fn new(start: f32, end: f32) -> Self {
        Self {
            start,
            end,
            start_confidence: 0.5,
            end_confidence: 0.5,
            audio_refined: false,
            token_indices: Vec::new(),
        }
    }

    /// Get boundary duration
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }

    /// Get overall confidence
    #[must_use]
    pub fn confidence(&self) -> f32 {
        (self.start_confidence + self.end_confidence) / 2.0
    }

    /// Check if boundary is high confidence
    #[must_use]
    pub fn is_high_confidence(&self) -> bool {
        self.confidence() >= 0.7
    }

    /// Set confidence scores
    #[must_use]
    pub fn with_confidence(mut self, start: f32, end: f32) -> Self {
        self.start_confidence = start;
        self.end_confidence = end;
        self
    }

    /// Set token indices
    #[must_use]
    pub fn with_tokens(mut self, indices: Vec<usize>) -> Self {
        self.token_indices = indices;
        self
    }

    /// Mark as audio-refined
    #[must_use]
    pub fn with_audio_refined(mut self, refined: bool) -> Self {
        self.audio_refined = refined;
        self
    }
}

/// Word boundary detector
#[derive(Debug, Clone)]
pub struct BoundaryDetector {
    /// Configuration
    config: BoundaryConfig,
}

impl BoundaryDetector {
    /// Create new boundary detector
    #[must_use]
    pub fn new(config: BoundaryConfig) -> Self {
        Self { config }
    }

    /// Detect word boundaries from token alignments
    ///
    /// # Arguments
    /// * `alignments` - Token alignments from cross-attention
    /// * `word_starts` - Indices where new words start
    pub fn detect_boundaries(
        &self,
        alignments: &[TokenAlignment],
        word_starts: &[usize],
    ) -> WhisperResult<Vec<WordBoundary>> {
        if alignments.is_empty() || word_starts.is_empty() {
            return Ok(Vec::new());
        }

        let mut boundaries = Vec::with_capacity(word_starts.len());

        for (i, &start_idx) in word_starts.iter().enumerate() {
            // Determine end index
            let end_idx = if i + 1 < word_starts.len() {
                word_starts[i + 1] - 1
            } else {
                alignments.len() - 1
            };

            if start_idx >= alignments.len() {
                continue;
            }

            let start_alignment = &alignments[start_idx];
            let end_alignment = &alignments[end_idx.min(alignments.len() - 1)];

            // Create boundary
            let mut boundary = WordBoundary::new(start_alignment.start_time, end_alignment.end_time)
                .with_confidence(start_alignment.confidence, end_alignment.confidence)
                .with_tokens((start_idx..=end_idx.min(alignments.len() - 1)).collect());

            // Validate boundary
            boundary = self.validate_boundary(boundary);

            boundaries.push(boundary);
        }

        Ok(boundaries)
    }

    /// Refine boundaries using audio energy
    ///
    /// # Arguments
    /// * `boundaries` - Initial boundaries
    /// * `audio_energy` - Energy values per frame (frame rate = 50fps)
    /// * `frame_rate` - Frames per second
    pub fn refine_with_audio(
        &self,
        boundaries: &[WordBoundary],
        audio_energy: &[f32],
        frame_rate: f32,
    ) -> WhisperResult<Vec<WordBoundary>> {
        if !self.config.use_audio_energy || audio_energy.is_empty() {
            return Ok(boundaries.to_vec());
        }

        let mut refined = Vec::with_capacity(boundaries.len());

        for boundary in boundaries {
            let refined_boundary = self.refine_single_boundary(boundary, audio_energy, frame_rate);
            refined.push(refined_boundary);
        }

        Ok(refined)
    }

    /// Refine a single boundary using audio energy
    fn refine_single_boundary(
        &self,
        boundary: &WordBoundary,
        audio_energy: &[f32],
        frame_rate: f32,
    ) -> WordBoundary {
        let mut result = boundary.clone();

        let start_frame = (boundary.start * frame_rate) as usize;
        let end_frame = (boundary.end * frame_rate) as usize;

        // Find actual speech onset
        if let Some(refined_start) =
            self.find_speech_onset(audio_energy, start_frame, frame_rate)
        {
            result.start = refined_start;
            result.start_confidence = 0.9;
        }

        // Find actual speech offset
        if let Some(refined_end) = self.find_speech_offset(audio_energy, end_frame, frame_rate) {
            result.end = refined_end;
            result.end_confidence = 0.9;
        }

        result.audio_refined = true;
        result
    }

    /// Find speech onset near a frame
    #[allow(clippy::needless_range_loop)]
    fn find_speech_onset(
        &self,
        energy: &[f32],
        approx_frame: usize,
        frame_rate: f32,
    ) -> Option<f32> {
        let search_window = (self.config.min_word_duration * frame_rate) as usize;
        let start = approx_frame.saturating_sub(search_window);
        let end = (approx_frame + search_window).min(energy.len());

        if start >= end || start >= energy.len() {
            return None;
        }

        // Look for energy rise
        for i in start..end {
            if energy[i] > self.config.silence_threshold {
                return Some(i as f32 / frame_rate);
            }
        }

        None
    }

    /// Find speech offset near a frame
    fn find_speech_offset(
        &self,
        energy: &[f32],
        approx_frame: usize,
        frame_rate: f32,
    ) -> Option<f32> {
        let search_window = (self.config.min_word_duration * frame_rate) as usize;
        let start = approx_frame.saturating_sub(search_window);
        let end = (approx_frame + search_window).min(energy.len());

        if start >= end || end > energy.len() {
            return None;
        }

        // Look for energy drop (search backwards)
        for i in (start..end).rev() {
            if energy[i] > self.config.silence_threshold {
                return Some((i + 1) as f32 / frame_rate);
            }
        }

        None
    }

    /// Validate and adjust boundary
    fn validate_boundary(&self, mut boundary: WordBoundary) -> WordBoundary {
        // Ensure minimum duration
        if boundary.duration() < self.config.min_word_duration {
            boundary.end = boundary.start + self.config.min_word_duration;
        }

        // Cap maximum duration
        if boundary.duration() > self.config.max_word_duration {
            boundary.end = boundary.start + self.config.max_word_duration;
            boundary.end_confidence *= 0.5; // Lower confidence for capped boundary
        }

        // Ensure start < end
        if boundary.end <= boundary.start {
            boundary.end = boundary.start + self.config.min_word_duration;
        }

        boundary
    }

    /// Compute boundary confidence from attention weights
    pub fn compute_boundary_confidence(&self, alignments: &[TokenAlignment]) -> f32 {
        if alignments.is_empty() {
            return 0.0;
        }

        // Average token confidences
        let avg_confidence: f32 =
            alignments.iter().map(|a| a.confidence).sum::<f32>() / alignments.len() as f32;

        // Check for monotonic frame progression
        let mut monotonic_score = 1.0f32;
        for i in 1..alignments.len() {
            if alignments[i].frame_position < alignments[i - 1].frame_position {
                monotonic_score *= 0.9;
            }
        }

        avg_confidence * monotonic_score
    }

    /// Detect silence gaps between boundaries
    pub fn detect_silence_gaps(&self, boundaries: &[WordBoundary]) -> Vec<(f32, f32)> {
        let mut gaps = Vec::new();

        for i in 1..boundaries.len() {
            let gap_start = boundaries[i - 1].end;
            let gap_end = boundaries[i].start;
            let gap_duration = gap_end - gap_start;

            if gap_duration >= self.config.min_silence_duration {
                gaps.push((gap_start, gap_end));
            }
        }

        gaps
    }
}

impl Default for BoundaryDetector {
    fn default() -> Self {
        Self::new(BoundaryConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // BoundaryConfig Tests
    // =========================================================================

    #[test]
    fn test_boundary_config_default() {
        let config = BoundaryConfig::default();
        assert!((config.min_silence_duration - 0.05).abs() < f32::EPSILON);
        assert!(config.use_audio_energy);
    }

    #[test]
    fn test_boundary_config_precise() {
        let config = BoundaryConfig::precise();
        assert!((config.min_silence_duration - 0.03).abs() < f32::EPSILON);
        assert!((config.min_word_duration - 0.03).abs() < f32::EPSILON);
    }

    #[test]
    fn test_boundary_config_fast() {
        let config = BoundaryConfig::fast();
        assert!(!config.use_audio_energy);
        assert!((config.min_silence_duration - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_boundary_config_with_min_silence() {
        let config = BoundaryConfig::default().with_min_silence(0.1);
        assert!((config.min_silence_duration - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_boundary_config_with_min_word_duration() {
        let config = BoundaryConfig::default().with_min_word_duration(0.2);
        assert!((config.min_word_duration - 0.2).abs() < f32::EPSILON);
    }

    // =========================================================================
    // WordBoundary Tests
    // =========================================================================

    #[test]
    fn test_word_boundary_new() {
        let boundary = WordBoundary::new(1.0, 2.0);
        assert!((boundary.start - 1.0).abs() < f32::EPSILON);
        assert!((boundary.end - 2.0).abs() < f32::EPSILON);
        assert!((boundary.start_confidence - 0.5).abs() < f32::EPSILON);
        assert!(!boundary.audio_refined);
    }

    #[test]
    fn test_word_boundary_duration() {
        let boundary = WordBoundary::new(1.0, 2.5);
        assert!((boundary.duration() - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_word_boundary_confidence() {
        let boundary = WordBoundary::new(1.0, 2.0).with_confidence(0.8, 0.6);
        assert!((boundary.confidence() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_word_boundary_is_high_confidence() {
        let high = WordBoundary::new(1.0, 2.0).with_confidence(0.9, 0.9);
        let low = WordBoundary::new(1.0, 2.0).with_confidence(0.3, 0.3);
        assert!(high.is_high_confidence());
        assert!(!low.is_high_confidence());
    }

    #[test]
    fn test_word_boundary_with_tokens() {
        let boundary = WordBoundary::new(1.0, 2.0).with_tokens(vec![0, 1, 2]);
        assert_eq!(boundary.token_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_word_boundary_with_audio_refined() {
        let boundary = WordBoundary::new(1.0, 2.0).with_audio_refined(true);
        assert!(boundary.audio_refined);
    }

    // =========================================================================
    // BoundaryDetector Tests
    // =========================================================================

    #[test]
    fn test_boundary_detector_new() {
        let detector = BoundaryDetector::new(BoundaryConfig::default());
        assert!(detector.config.use_audio_energy);
    }

    #[test]
    fn test_boundary_detector_default() {
        let detector = BoundaryDetector::default();
        assert!((detector.config.min_silence_duration - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn test_detect_boundaries_empty() {
        let detector = BoundaryDetector::default();
        let result = detector.detect_boundaries(&[], &[]).expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_boundaries_single_word() {
        let detector = BoundaryDetector::default();

        let mut alignments = vec![
            TokenAlignment::new(0, 100, 50, 0.9),
            TokenAlignment::new(1, 101, 60, 0.8),
        ];
        alignments[0].set_end_time(60);
        alignments[1].set_end_time(80);

        let word_starts = vec![0];

        let result = detector
            .detect_boundaries(&alignments, &word_starts)
            .expect("should succeed");

        assert_eq!(result.len(), 1);
        assert!((result[0].start - 1.0).abs() < f32::EPSILON); // frame 50 / 50fps
    }

    #[test]
    fn test_detect_boundaries_multiple_words() {
        let detector = BoundaryDetector::default();

        let mut alignments = vec![
            TokenAlignment::new(0, 100, 0, 0.9),
            TokenAlignment::new(1, 101, 25, 0.8),
            TokenAlignment::new(2, 102, 50, 0.85),
            TokenAlignment::new(3, 103, 75, 0.9),
        ];
        for (i, a) in alignments.iter_mut().enumerate() {
            a.set_end_time((i + 1) * 25);
        }

        let word_starts = vec![0, 2];

        let result = detector
            .detect_boundaries(&alignments, &word_starts)
            .expect("should succeed");

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_validate_boundary_min_duration() {
        let detector = BoundaryDetector::new(BoundaryConfig::default().with_min_word_duration(0.1));

        let boundary = WordBoundary::new(1.0, 1.01); // Very short
        let validated = detector.validate_boundary(boundary);

        assert!(validated.duration() >= 0.1);
    }

    #[test]
    fn test_validate_boundary_max_duration() {
        let mut config = BoundaryConfig::default();
        config.max_word_duration = 2.0;
        let detector = BoundaryDetector::new(config);

        let boundary = WordBoundary::new(0.0, 10.0); // Too long
        let validated = detector.validate_boundary(boundary);

        assert!((validated.duration() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_boundary_confidence_empty() {
        let detector = BoundaryDetector::default();
        let confidence = detector.compute_boundary_confidence(&[]);
        assert!((confidence - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_boundary_confidence_monotonic() {
        let detector = BoundaryDetector::default();

        let alignments = vec![
            TokenAlignment::new(0, 100, 10, 0.8),
            TokenAlignment::new(1, 101, 20, 0.8),
            TokenAlignment::new(2, 102, 30, 0.8),
        ];

        let confidence = detector.compute_boundary_confidence(&alignments);
        assert!((confidence - 0.8).abs() < f32::EPSILON); // No penalty for monotonic
    }

    #[test]
    fn test_compute_boundary_confidence_non_monotonic() {
        let detector = BoundaryDetector::default();

        let alignments = vec![
            TokenAlignment::new(0, 100, 30, 0.8),
            TokenAlignment::new(1, 101, 20, 0.8), // Out of order
            TokenAlignment::new(2, 102, 40, 0.8),
        ];

        let confidence = detector.compute_boundary_confidence(&alignments);
        assert!(confidence < 0.8); // Penalty for non-monotonic
    }

    #[test]
    fn test_detect_silence_gaps() {
        let detector = BoundaryDetector::default();

        let boundaries = vec![
            WordBoundary::new(0.0, 1.0),
            WordBoundary::new(1.5, 2.5), // 0.5s gap
            WordBoundary::new(2.6, 3.5), // 0.1s gap
        ];

        let gaps = detector.detect_silence_gaps(&boundaries);

        assert_eq!(gaps.len(), 2);
        assert!((gaps[0].0 - 1.0).abs() < f32::EPSILON);
        assert!((gaps[0].1 - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_detect_silence_gaps_no_gaps() {
        let mut config = BoundaryConfig::default();
        config.min_silence_duration = 1.0; // High threshold
        let detector = BoundaryDetector::new(config);

        let boundaries = vec![
            WordBoundary::new(0.0, 1.0),
            WordBoundary::new(1.1, 2.0), // Only 0.1s gap
        ];

        let gaps = detector.detect_silence_gaps(&boundaries);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_refine_with_audio_disabled() {
        let config = BoundaryConfig::fast(); // Audio refinement disabled
        let detector = BoundaryDetector::new(config);

        let boundaries = vec![WordBoundary::new(0.0, 1.0)];
        let audio_energy = vec![0.5; 100];

        let refined = detector
            .refine_with_audio(&boundaries, &audio_energy, 50.0)
            .expect("should succeed");

        assert!(!refined[0].audio_refined);
    }

    #[test]
    fn test_refine_with_audio_empty_energy() {
        let detector = BoundaryDetector::default();

        let boundaries = vec![WordBoundary::new(0.0, 1.0)];

        let refined = detector
            .refine_with_audio(&boundaries, &[], 50.0)
            .expect("should succeed");

        assert!(!refined[0].audio_refined);
    }

    // =========================================================================
    // Additional Coverage Tests (WAPR-QA)
    // =========================================================================

    #[test]
    fn test_boundary_config_silence_builder() {
        let config = BoundaryConfig::default().with_min_silence(0.1);
        assert!((config.min_silence_duration - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_boundary_config_word_duration_builder() {
        let config = BoundaryConfig::default().with_min_word_duration(0.2);
        assert!((config.min_word_duration - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_word_boundary_with_both_confidences() {
        let boundary = WordBoundary::new(0.0, 1.0)
            .with_confidence(0.9, 0.8);

        assert!((boundary.start_confidence - 0.9).abs() < f32::EPSILON);
        assert!((boundary.end_confidence - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_word_boundary_tokens_builder() {
        let boundary = WordBoundary::new(0.0, 1.0).with_tokens(vec![1, 2, 3]);
        assert_eq!(boundary.token_indices, vec![1, 2, 3]);
    }

    #[test]
    fn test_word_boundary_duration_calculation() {
        let boundary = WordBoundary::new(1.5, 3.5);
        assert!((boundary.duration() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_word_boundary_avg_confidence() {
        let boundary = WordBoundary::new(0.0, 1.0).with_confidence(0.8, 0.6);
        // confidence() returns average
        let conf = boundary.confidence();
        assert!((conf - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_word_boundary_is_high_confidence_both() {
        let high = WordBoundary::new(0.0, 1.0).with_confidence(0.9, 0.85);
        let low = WordBoundary::new(0.0, 1.0).with_confidence(0.5, 0.6);

        assert!(high.is_high_confidence());
        assert!(!low.is_high_confidence());
    }

    #[test]
    fn test_word_boundary_audio_refined_flag() {
        let boundary = WordBoundary::new(0.0, 1.0).with_audio_refined(true);
        assert!(boundary.audio_refined);
    }

    #[test]
    fn test_detect_boundaries_empty_alignments() {
        let detector = BoundaryDetector::default();
        let boundaries = detector.detect_boundaries(&[], &[]).expect("should succeed");
        assert!(boundaries.is_empty());
    }

    #[test]
    fn test_detect_boundaries_single_alignment() {
        let detector = BoundaryDetector::default();
        let alignments = vec![TokenAlignment::new(0, 100, 30, 0.9)];
        let word_starts = vec![0];
        let boundaries = detector.detect_boundaries(&alignments, &word_starts).expect("should succeed");
        assert_eq!(boundaries.len(), 1);
    }

    #[test]
    fn test_refine_with_audio_boundaries() {
        let detector = BoundaryDetector::default();

        let boundaries = vec![
            WordBoundary::new(0.0, 0.5),
            WordBoundary::new(0.6, 1.0),
        ];

        // Create audio energy with some variation
        let audio_energy: Vec<f32> = (0..100)
            .map(|i| if i < 50 { 0.5 } else { 0.02 })
            .collect();

        let refined = detector
            .refine_with_audio(&boundaries, &audio_energy, 100.0)
            .expect("should succeed");

        assert_eq!(refined.len(), 2);
    }

    #[test]
    fn test_detect_silence_gaps_multiple() {
        let detector = BoundaryDetector::default();

        let boundaries = vec![
            WordBoundary::new(0.0, 0.5),
            WordBoundary::new(1.0, 1.5),
            WordBoundary::new(2.0, 2.5),
        ];

        let gaps = detector.detect_silence_gaps(&boundaries);
        assert_eq!(gaps.len(), 2);
    }

    #[test]
    fn test_compute_boundary_confidence_single() {
        let detector = BoundaryDetector::default();
        let alignments = vec![TokenAlignment::new(0, 100, 30, 0.8)];
        let conf = detector.compute_boundary_confidence(&alignments);
        assert!(conf > 0.0 && conf <= 1.0);
    }
}
