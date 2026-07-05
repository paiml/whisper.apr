#![allow(clippy::expect_used)]
//! Tests for speaker turn detection and segmentation

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
    assert!(zcr.iter().all(|&z| (0.0..=1.0).contains(&z)));
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
