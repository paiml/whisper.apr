//! Tests for timestamp interpolation

use super::*;

// =========================================================================
// InterpolationConfig Tests
// =========================================================================

#[test]
fn test_interpolation_config_default() {
    let config = InterpolationConfig::default();
    assert_eq!(config.method, InterpolationMethod::Weighted);
    assert_eq!(config.smoothing_window, 3);
}

#[test]
fn test_interpolation_config_linear() {
    let config = InterpolationConfig::linear();
    assert_eq!(config.method, InterpolationMethod::Linear);
    assert!((config.uniform_weight - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_interpolation_config_character_proportional() {
    let config = InterpolationConfig::character_proportional();
    assert_eq!(config.method, InterpolationMethod::CharacterProportional);
    assert!((config.char_weight - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_interpolation_config_with_smoothing() {
    let config = InterpolationConfig::default().with_smoothing(5);
    assert_eq!(config.smoothing_window, 5);
}

#[test]
fn test_interpolation_config_with_method() {
    let config = InterpolationConfig::default().with_method(InterpolationMethod::Linear);
    assert_eq!(config.method, InterpolationMethod::Linear);
}

// =========================================================================
// TokenTimestamp Tests
// =========================================================================

#[test]
fn test_token_timestamp_new() {
    let ts = TokenTimestamp::new(0, "hello".to_string(), 0.0, 1.0);
    assert_eq!(ts.index, 0);
    assert_eq!(ts.text, "hello");
    assert!(!ts.interpolated);
    assert!((ts.confidence - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_token_timestamp_duration() {
    let ts = TokenTimestamp::new(0, "test".to_string(), 1.0, 2.5);
    assert!((ts.duration() - 1.5).abs() < f32::EPSILON);
}

#[test]
fn test_token_timestamp_mark_interpolated() {
    let mut ts = TokenTimestamp::new(0, "test".to_string(), 0.0, 1.0);
    ts.mark_interpolated(0.7);
    assert!(ts.interpolated);
    assert!((ts.confidence - 0.7).abs() < f32::EPSILON);
}

#[test]
fn test_token_timestamp_interpolated() {
    let ts = TokenTimestamp::interpolated(0, "test".to_string(), 0.0, 1.0, 0.6);
    assert!(ts.interpolated);
    assert!((ts.confidence - 0.6).abs() < f32::EPSILON);
}

// =========================================================================
// TimestampInterpolator Tests
// =========================================================================

#[test]
fn test_timestamp_interpolator_new() {
    let interpolator = TimestampInterpolator::new(InterpolationConfig::default());
    assert_eq!(interpolator.config.method, InterpolationMethod::Weighted);
}

#[test]
fn test_timestamp_interpolator_default() {
    let interpolator = TimestampInterpolator::default();
    assert_eq!(interpolator.config.smoothing_window, 3);
}

#[test]
fn test_interpolate_word_tokens_empty() {
    let interpolator = TimestampInterpolator::default();
    let result = interpolator
        .interpolate_word_tokens(0.0, 1.0, &[], 0)
        .expect("should succeed");
    assert!(result.is_empty());
}

#[test]
fn test_interpolate_word_tokens_single() {
    let interpolator = TimestampInterpolator::default();
    let tokens = vec!["hello".to_string()];
    let result = interpolator
        .interpolate_word_tokens(0.0, 1.0, &tokens, 0)
        .expect("should succeed");

    assert_eq!(result.len(), 1);
    assert!(!result[0].interpolated); // Single token, no interpolation needed
    assert!((result[0].start - 0.0).abs() < f32::EPSILON);
    assert!((result[0].end - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_interpolate_linear() {
    let interpolator = TimestampInterpolator::new(InterpolationConfig::linear());
    let tokens = vec!["hel".to_string(), "lo".to_string()];
    let result = interpolator
        .interpolate_word_tokens(0.0, 1.0, &tokens, 0)
        .expect("should succeed");

    assert_eq!(result.len(), 2);
    assert!(result[0].interpolated);
    assert!(result[1].interpolated);

    // Linear: each token gets 0.5s
    assert!((result[0].duration() - 0.5).abs() < f32::EPSILON);
    assert!((result[1].duration() - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_interpolate_character_proportional() {
    let interpolator =
        TimestampInterpolator::new(InterpolationConfig::character_proportional());
    let tokens = vec!["a".to_string(), "abc".to_string()]; // 1 char + 3 chars = 4 total
    let result = interpolator
        .interpolate_word_tokens(0.0, 1.0, &tokens, 0)
        .expect("should succeed");

    assert_eq!(result.len(), 2);

    // Character proportional: 1/4 and 3/4 of duration
    assert!((result[0].duration() - 0.25).abs() < f32::EPSILON);
    assert!((result[1].duration() - 0.75).abs() < f32::EPSILON);
}

#[test]
fn test_interpolate_weighted() {
    let interpolator = TimestampInterpolator::default(); // Uses Weighted
    let tokens = vec!["hel".to_string(), "lo".to_string()];
    let result = interpolator
        .interpolate_word_tokens(0.0, 1.0, &tokens, 0)
        .expect("should succeed");

    assert_eq!(result.len(), 2);
    assert!(result[0].interpolated);
    assert!(result[1].interpolated);

    // Weighted should end at word boundary
    assert!((result[1].end - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_interpolate_preserves_continuity() {
    let interpolator = TimestampInterpolator::default();
    let tokens = vec!["un".to_string(), "break".to_string(), "able".to_string()];
    let result = interpolator
        .interpolate_word_tokens(0.0, 1.5, &tokens, 0)
        .expect("should succeed");

    assert_eq!(result.len(), 3);

    // Check continuity: each token ends where next begins
    assert!((result[0].end - result[1].start).abs() < f32::EPSILON);
    assert!((result[1].end - result[2].start).abs() < f32::EPSILON);
}

#[test]
fn test_interpolate_correct_indices() {
    let interpolator = TimestampInterpolator::default();
    let tokens = vec!["a".to_string(), "b".to_string()];
    let result = interpolator
        .interpolate_word_tokens(0.0, 1.0, &tokens, 5) // Start at index 5
        .expect("should succeed");

    assert_eq!(result[0].index, 5);
    assert_eq!(result[1].index, 6);
}

#[test]
fn test_smooth_timestamps_no_smoothing() {
    let interpolator =
        TimestampInterpolator::new(InterpolationConfig::default().with_smoothing(0));

    let mut timestamps = vec![
        TokenTimestamp::interpolated(0, "a".to_string(), 0.0, 0.3, 0.5),
        TokenTimestamp::interpolated(1, "b".to_string(), 0.3, 0.7, 0.5),
    ];

    interpolator.smooth_timestamps(&mut timestamps);

    // No change with smoothing disabled
    assert!((timestamps[0].end - 0.3).abs() < f32::EPSILON);
}

#[test]
fn test_smooth_timestamps_fixes_overlap() {
    let interpolator =
        TimestampInterpolator::new(InterpolationConfig::default().with_smoothing(3));

    let mut timestamps = vec![
        TokenTimestamp::interpolated(0, "a".to_string(), 0.0, 0.6, 0.5),
        TokenTimestamp::interpolated(1, "b".to_string(), 0.4, 0.8, 0.5), // Overlaps with previous
        TokenTimestamp::interpolated(2, "c".to_string(), 0.8, 1.0, 0.5),
    ];

    interpolator.smooth_timestamps(&mut timestamps);

    // No overlap after smoothing
    assert!(timestamps[1].start >= timestamps[0].end);
    assert!(timestamps[2].start >= timestamps[1].end);
}

#[test]
fn test_interpolate_with_attention() {
    let interpolator = TimestampInterpolator::default();
    let tokens = vec!["hel".to_string(), "lo".to_string()];

    // Create attention weights with distinct peaks
    let attention1 = vec![0.1, 0.8, 0.1]; // Peak at frame 1
    let attention2 = vec![0.1, 0.1, 0.8]; // Peak at frame 2
    let attention_weights = vec![attention1, attention2];

    let result = interpolator
        .interpolate_with_attention(0.0, 1.0, &tokens, &attention_weights, 0, 50.0)
        .expect("should succeed");

    assert_eq!(result.len(), 2);
    assert!(result[0].interpolated);
    assert!(result[1].interpolated);
    assert!((result[0].confidence - 0.8).abs() < f32::EPSILON); // Higher confidence with attention
}

#[test]
fn test_interpolate_with_attention_mismatched_lengths() {
    let interpolator = TimestampInterpolator::default();
    let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let attention_weights = vec![vec![0.5; 10], vec![0.5; 10]]; // Only 2 attention vectors for 3 tokens

    let result = interpolator
        .interpolate_with_attention(0.0, 1.0, &tokens, &attention_weights, 0, 50.0)
        .expect("should succeed");

    // Should fall back to regular interpolation
    assert_eq!(result.len(), 3);
}
