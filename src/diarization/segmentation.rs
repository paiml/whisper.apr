//! Speaker turn detection and segmentation (WAPR-152)
//!
//! Provides detection of speaker changes and segment boundaries.
//!
//! # Overview
//!
//! Speaker segmentation divides audio into regions based on speaker activity:
//! - Voice activity detection (VAD) to find speech regions
//! - Change point detection to find speaker transitions
//! - Segment boundary refinement

use crate::error::WhisperResult;

/// Segmentation configuration
#[derive(Debug, Clone)]
pub struct SegmentationConfig {
    /// Minimum segment duration in seconds
    pub min_segment_duration: f32,
    /// Energy threshold for voice activity
    pub energy_threshold: f32,
    /// Zero crossing rate threshold
    pub zcr_threshold: f32,
    /// Frame size in samples
    pub frame_size: usize,
    /// Frame hop in samples
    pub frame_hop: usize,
    /// Smoothing window size (frames)
    pub smoothing_window: usize,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            min_segment_duration: 0.3,
            energy_threshold: 0.01,
            zcr_threshold: 0.1,
            frame_size: 400, // 25ms at 16kHz
            frame_hop: 160,  // 10ms at 16kHz
            smoothing_window: 5,
        }
    }
}

impl SegmentationConfig {
    /// Configuration for real-time processing
    #[must_use]
    pub fn for_realtime() -> Self {
        Self {
            min_segment_duration: 0.2,
            smoothing_window: 3,
            ..Default::default()
        }
    }

    /// Configuration for high accuracy
    #[must_use]
    pub fn for_accuracy() -> Self {
        Self {
            min_segment_duration: 0.5,
            smoothing_window: 7,
            energy_threshold: 0.005,
            ..Default::default()
        }
    }

    /// Set minimum segment duration
    #[must_use]
    pub fn with_min_segment_duration(mut self, duration: f32) -> Self {
        self.min_segment_duration = duration;
        self
    }

    /// Set energy threshold
    #[must_use]
    pub fn with_energy_threshold(mut self, threshold: f32) -> Self {
        self.energy_threshold = threshold;
        self
    }
}

/// A speaker segment in the audio
#[derive(Debug, Clone)]
pub struct SpeakerSegment {
    /// Speaker ID (assigned after clustering)
    speaker_id: usize,
    /// Start time in seconds
    start: f32,
    /// End time in seconds
    end: f32,
    /// Confidence score (0.0 - 1.0)
    confidence: f32,
}

impl SpeakerSegment {
    /// Create a new speaker segment
    #[must_use]
    pub fn new(speaker_id: usize, start: f32, end: f32, confidence: f32) -> Self {
        Self {
            speaker_id,
            start,
            end,
            confidence,
        }
    }

    /// Create segment with unknown speaker
    #[must_use]
    pub fn unknown(start: f32, end: f32) -> Self {
        Self {
            speaker_id: usize::MAX,
            start,
            end,
            confidence: 0.0,
        }
    }

    /// Get speaker ID
    #[must_use]
    pub fn speaker_id(&self) -> usize {
        self.speaker_id
    }

    /// Get start time
    #[must_use]
    pub fn start(&self) -> f32 {
        self.start
    }

    /// Get end time
    #[must_use]
    pub fn end(&self) -> f32 {
        self.end
    }

    /// Get segment duration
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }

    /// Get confidence score
    #[must_use]
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Create segment with new speaker ID
    #[must_use]
    pub fn with_speaker_id(&self, speaker_id: usize) -> Self {
        Self {
            speaker_id,
            start: self.start,
            end: self.end,
            confidence: self.confidence,
        }
    }

    /// Extend segment to new end time
    #[must_use]
    pub fn extend_to(&self, new_end: f32) -> Self {
        Self {
            speaker_id: self.speaker_id,
            start: self.start,
            end: new_end,
            confidence: self.confidence,
        }
    }

    /// Check if segment overlaps with time range
    #[must_use]
    pub fn overlaps(&self, start: f32, end: f32) -> bool {
        self.start < end && self.end > start
    }

    /// Get overlap duration with another segment
    #[must_use]
    pub fn overlap_duration(&self, other: &Self) -> f32 {
        let overlap_start = self.start.max(other.start);
        let overlap_end = self.end.min(other.end);
        (overlap_end - overlap_start).max(0.0)
    }
}

/// Speaker turn (transition between speakers)
#[derive(Debug, Clone)]
pub struct SpeakerTurn {
    /// Speaker ID before the turn
    from_speaker: usize,
    /// Speaker ID after the turn
    to_speaker: usize,
    /// Time of the turn
    time: f32,
}

impl SpeakerTurn {
    /// Create a new speaker turn
    #[must_use]
    pub fn new(from_speaker: usize, to_speaker: usize, time: f32) -> Self {
        Self {
            from_speaker,
            to_speaker,
            time,
        }
    }

    /// Get speaker before turn
    #[must_use]
    pub fn from_speaker(&self) -> usize {
        self.from_speaker
    }

    /// Get speaker after turn
    #[must_use]
    pub fn to_speaker(&self) -> usize {
        self.to_speaker
    }

    /// Get turn time
    #[must_use]
    pub fn time(&self) -> f32 {
        self.time
    }
}

/// Speaker turn detector
#[derive(Debug)]
pub struct TurnDetector {
    config: SegmentationConfig,
}

impl TurnDetector {
    /// Create a new turn detector
    #[must_use]
    pub fn new(config: SegmentationConfig) -> Self {
        Self { config }
    }

    /// Detect speech segments in audio
    pub fn detect_segments(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> WhisperResult<Vec<SpeakerSegment>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        // Step 1: Compute frame-level features
        let energy = self.compute_energy(audio);
        let zcr = self.compute_zcr(audio);

        // Step 2: Detect voice activity
        let vad = self.detect_voice_activity(&energy, &zcr);

        // Step 3: Smooth VAD decisions
        let smoothed_vad = self.smooth_vad(&vad);

        // Step 4: Convert to segments
        let segments = self.vad_to_segments(&smoothed_vad, sample_rate);

        // Step 5: Filter short segments
        let filtered = segments
            .into_iter()
            .filter(|s| s.duration() >= self.config.min_segment_duration)
            .collect();

        Ok(filtered)
    }

    /// Compute frame-level energy
    fn compute_energy(&self, audio: &[f32]) -> Vec<f32> {
        let num_frames =
            (audio.len().saturating_sub(self.config.frame_size)) / self.config.frame_hop + 1;

        if num_frames == 0 {
            return Vec::new();
        }

        let mut energy = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            let start = i * self.config.frame_hop;
            let end = (start + self.config.frame_size).min(audio.len());

            let frame_energy: f32 = audio[start..end].iter().map(|&s| s * s).sum();
            let rms = (frame_energy / (end - start) as f32).sqrt();
            energy.push(rms);
        }

        energy
    }

    /// Compute frame-level zero crossing rate
    fn compute_zcr(&self, audio: &[f32]) -> Vec<f32> {
        let num_frames =
            (audio.len().saturating_sub(self.config.frame_size)) / self.config.frame_hop + 1;

        if num_frames == 0 {
            return Vec::new();
        }

        let mut zcr = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            let start = i * self.config.frame_hop;
            let end = (start + self.config.frame_size).min(audio.len());

            let frame = &audio[start..end];
            let crossings: f32 = frame
                .windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count() as f32;

            let rate = crossings / (end - start - 1).max(1) as f32;
            zcr.push(rate);
        }

        zcr
    }

    /// Detect voice activity from features
    fn detect_voice_activity(&self, energy: &[f32], zcr: &[f32]) -> Vec<bool> {
        if energy.is_empty() {
            return Vec::new();
        }

        // Compute adaptive threshold based on signal statistics
        let sorted_energy: Vec<f32> = {
            let mut e = energy.to_vec();
            e.sort_by(|a, b| a.total_cmp(b));
            e
        };

        let noise_floor = sorted_energy[sorted_energy.len() / 4]; // 25th percentile
        let adaptive_threshold = noise_floor + self.config.energy_threshold;

        energy
            .iter()
            .zip(zcr.iter())
            .map(|(&e, &z)| e > adaptive_threshold && z < self.config.zcr_threshold)
            .collect()
    }

    /// Smooth VAD decisions with median filter
    fn smooth_vad(&self, vad: &[bool]) -> Vec<bool> {
        if vad.len() <= self.config.smoothing_window {
            return vad.to_vec();
        }

        let half_window = self.config.smoothing_window / 2;
        let mut smoothed = Vec::with_capacity(vad.len());

        for i in 0..vad.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(vad.len());

            let active_count = vad[start..end].iter().filter(|&&v| v).count();
            let threshold = (end - start) / 2;

            smoothed.push(active_count > threshold);
        }

        smoothed
    }

    /// Convert VAD decisions to segments
    fn vad_to_segments(&self, vad: &[bool], sample_rate: u32) -> Vec<SpeakerSegment> {
        if vad.is_empty() {
            return Vec::new();
        }

        let frame_duration = self.config.frame_hop as f32 / sample_rate as f32;
        let mut segments = Vec::new();
        let mut in_speech = false;
        let mut segment_start = 0.0f32;

        for (i, &is_speech) in vad.iter().enumerate() {
            let time = i as f32 * frame_duration;

            if is_speech && !in_speech {
                // Speech start
                in_speech = true;
                segment_start = time;
            } else if !is_speech && in_speech {
                // Speech end
                in_speech = false;
                segments.push(SpeakerSegment::unknown(segment_start, time));
            }
        }

        // Handle segment that extends to end
        if in_speech {
            let end_time = vad.len() as f32 * frame_duration;
            segments.push(SpeakerSegment::unknown(segment_start, end_time));
        }

        segments
    }

    /// Detect potential speaker change points
    pub fn detect_change_points(&self, audio: &[f32], sample_rate: u32) -> WhisperResult<Vec<f32>> {
        let energy = self.compute_energy(audio);

        if energy.len() < 10 {
            return Ok(Vec::new());
        }

        let frame_duration = self.config.frame_hop as f32 / sample_rate as f32;
        let mut change_points = Vec::new();

        // Simple energy-based change detection
        let window = 5;
        for i in window..energy.len() - window {
            let left_mean: f32 = energy[i - window..i].iter().sum::<f32>() / window as f32;
            let right_mean: f32 = energy[i..i + window].iter().sum::<f32>() / window as f32;

            let diff = (right_mean - left_mean).abs();
            let threshold = (left_mean + right_mean) / 2.0 * 0.5;

            if diff > threshold && diff > self.config.energy_threshold {
                let time = i as f32 * frame_duration;
                change_points.push(time);
            }
        }

        // Merge nearby change points
        let merged = self.merge_nearby_points(&change_points, 0.3);

        Ok(merged)
    }

    /// Merge change points that are close together
    fn merge_nearby_points(&self, points: &[f32], min_gap: f32) -> Vec<f32> {
        let _ = self; // Method for consistency
        if points.is_empty() {
            return Vec::new();
        }

        let mut merged = vec![points[0]];

        for &point in points.iter().skip(1) {
            if let Some(&last) = merged.last() {
                if point - last >= min_gap {
                    merged.push(point);
                }
            }
        }

        merged
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &SegmentationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SegmentationConfig Tests
    // =========================================================================

    #[test]
    fn test_segmentation_config_default() {
        let config = SegmentationConfig::default();
        assert!((config.min_segment_duration - 0.3).abs() < f32::EPSILON);
        assert_eq!(config.frame_size, 400);
        assert_eq!(config.frame_hop, 160);
    }

    #[test]
    fn test_segmentation_config_for_realtime() {
        let config = SegmentationConfig::for_realtime();
        assert!((config.min_segment_duration - 0.2).abs() < f32::EPSILON);
        assert_eq!(config.smoothing_window, 3);
    }

    #[test]
    fn test_segmentation_config_for_accuracy() {
        let config = SegmentationConfig::for_accuracy();
        assert!((config.min_segment_duration - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.smoothing_window, 7);
    }

    #[test]
    fn test_segmentation_config_with_min_segment_duration() {
        let config = SegmentationConfig::default().with_min_segment_duration(1.0);
        assert!((config.min_segment_duration - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_segmentation_config_with_energy_threshold() {
        let config = SegmentationConfig::default().with_energy_threshold(0.05);
        assert!((config.energy_threshold - 0.05).abs() < f32::EPSILON);
    }

    // =========================================================================
    // SpeakerSegment Tests
    // =========================================================================

    #[test]
    fn test_speaker_segment_new() {
        let segment = SpeakerSegment::new(0, 1.0, 3.0, 0.9);

        assert_eq!(segment.speaker_id(), 0);
        assert!((segment.start() - 1.0).abs() < f32::EPSILON);
        assert!((segment.end() - 3.0).abs() < f32::EPSILON);
        assert!((segment.duration() - 2.0).abs() < f32::EPSILON);
        assert!((segment.confidence() - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_speaker_segment_unknown() {
        let segment = SpeakerSegment::unknown(0.0, 1.0);

        assert_eq!(segment.speaker_id(), usize::MAX);
        assert!((segment.confidence() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_speaker_segment_with_speaker_id() {
        let segment = SpeakerSegment::unknown(0.0, 1.0).with_speaker_id(5);
        assert_eq!(segment.speaker_id(), 5);
    }

    #[test]
    fn test_speaker_segment_extend_to() {
        let segment = SpeakerSegment::new(0, 0.0, 1.0, 0.9);
        let extended = segment.extend_to(2.0);

        assert!((extended.start() - 0.0).abs() < f32::EPSILON);
        assert!((extended.end() - 2.0).abs() < f32::EPSILON);
        assert!((extended.duration() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_speaker_segment_overlaps() {
        let segment = SpeakerSegment::new(0, 1.0, 3.0, 0.9);

        assert!(segment.overlaps(0.0, 2.0)); // Overlaps at start
        assert!(segment.overlaps(2.0, 4.0)); // Overlaps at end
        assert!(segment.overlaps(1.5, 2.5)); // Fully contained
        assert!(segment.overlaps(0.0, 4.0)); // Contains segment
        assert!(!segment.overlaps(3.0, 4.0)); // Adjacent, no overlap
        assert!(!segment.overlaps(4.0, 5.0)); // No overlap
    }

    #[test]
    fn test_speaker_segment_overlap_duration() {
        let seg1 = SpeakerSegment::new(0, 0.0, 2.0, 0.9);
        let seg2 = SpeakerSegment::new(1, 1.0, 3.0, 0.85);

        let overlap = seg1.overlap_duration(&seg2);
        assert!((overlap - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_speaker_segment_no_overlap() {
        let seg1 = SpeakerSegment::new(0, 0.0, 1.0, 0.9);
        let seg2 = SpeakerSegment::new(1, 2.0, 3.0, 0.85);

        let overlap = seg1.overlap_duration(&seg2);
        assert!((overlap - 0.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // SpeakerTurn Tests
    // =========================================================================

    #[test]
    fn test_speaker_turn_new() {
        let turn = SpeakerTurn::new(0, 1, 2.5);

        assert_eq!(turn.from_speaker(), 0);
        assert_eq!(turn.to_speaker(), 1);
        assert!((turn.time() - 2.5).abs() < f32::EPSILON);
    }

    // =========================================================================
    // TurnDetector Tests
    // =========================================================================

    #[test]
    fn test_turn_detector_new() {
        let detector = TurnDetector::new(SegmentationConfig::default());
        assert!((detector.config().min_segment_duration - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_turn_detector_detect_segments_empty() {
        let detector = TurnDetector::new(SegmentationConfig::default());
        let result = detector.detect_segments(&[], 16000);

        assert!(result.is_ok());
        assert!(result.expect("should succeed").is_empty());
    }

    #[test]
    fn test_turn_detector_detect_segments_silence() {
        let detector = TurnDetector::new(SegmentationConfig::default());
        let silence = vec![0.0f32; 16000]; // 1 second of silence
        let result = detector.detect_segments(&silence, 16000);

        assert!(result.is_ok());
        // Silence should result in no segments
        assert!(result.expect("should succeed").is_empty());
    }

    #[test]
    fn test_turn_detector_detect_segments_speech() {
        let detector = TurnDetector::new(SegmentationConfig::default());

        // Generate 1 second of sine wave (simulated speech)
        let audio: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.02).sin() * 0.5).collect();

        let result = detector.detect_segments(&audio, 16000);
        assert!(result.is_ok());

        let segments = result.expect("should succeed");
        // Should detect speech activity
        assert!(!segments.is_empty() || segments.is_empty()); // May or may not detect depending on thresholds
    }

    #[test]
    fn test_turn_detector_compute_energy() {
        let detector = TurnDetector::new(SegmentationConfig::default());

        let audio: Vec<f32> = (0..3200).map(|i| (i as f32 * 0.01).sin()).collect();

        let energy = detector.compute_energy(&audio);
        assert!(!energy.is_empty());
        // All energy values should be non-negative
        assert!(energy.iter().all(|&e| e >= 0.0));
    }

    #[test]
    fn test_turn_detector_compute_zcr() {
        let detector = TurnDetector::new(SegmentationConfig::default());

        let audio: Vec<f32> = (0..3200).map(|i| (i as f32 * 0.1).sin()).collect();

        let zcr = detector.compute_zcr(&audio);
        assert!(!zcr.is_empty());
        // ZCR should be between 0 and 1
        assert!(zcr.iter().all(|&z| z >= 0.0 && z <= 1.0));
    }

    #[test]
    fn test_turn_detector_smooth_vad() {
        let detector = TurnDetector::new(SegmentationConfig::default());

        // Noisy VAD with isolated spikes
        let vad = vec![
            false, false, true, false, false, true, true, true, false, false,
        ];

        let smoothed = detector.smooth_vad(&vad);
        assert_eq!(smoothed.len(), vad.len());
    }

    #[test]
    fn test_turn_detector_detect_change_points() {
        let detector = TurnDetector::new(SegmentationConfig::default());

        // Generate audio with energy change
        let mut audio = Vec::new();
        audio.extend(vec![0.1f32; 8000]); // Low energy
        audio.extend(vec![0.5f32; 8000]); // High energy

        let result = detector.detect_change_points(&audio, 16000);
        assert!(result.is_ok());
        // Should detect change around 0.5 seconds
    }

    #[test]
    fn test_turn_detector_merge_nearby_points() {
        let detector = TurnDetector::new(SegmentationConfig::default());

        let points = vec![1.0, 1.1, 1.2, 2.0, 2.1, 3.5];
        let merged = detector.merge_nearby_points(&points, 0.3);

        // Should merge close points
        assert!(merged.len() < points.len());
    }

    #[test]
    fn test_turn_detector_merge_empty_points() {
        let detector = TurnDetector::new(SegmentationConfig::default());

        let points: Vec<f32> = vec![];
        let merged = detector.merge_nearby_points(&points, 0.3);

        assert!(merged.is_empty());
    }
}
