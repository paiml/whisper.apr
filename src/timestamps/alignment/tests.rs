//! Tests for cross-attention alignment extraction

use super::*;

// =========================================================================
// AlignmentConfig Tests
// =========================================================================

#[test]
fn test_alignment_config_default() {
    let config = AlignmentConfig::default();
    assert_eq!(config.layers.len(), 6);
    assert!(config.heads.is_none());
    assert!((config.min_attention - 0.1).abs() < f32::EPSILON);
}

#[test]
fn test_alignment_config_for_accuracy() {
    let config = AlignmentConfig::for_accuracy();
    assert_eq!(config.layers, vec![2, 3, 4, 5]);
    assert!(config.use_median);
}

#[test]
fn test_alignment_config_for_speed() {
    let config = AlignmentConfig::for_speed();
    assert_eq!(config.layers, vec![3, 4]);
    assert!(config.heads.is_some());
}

#[test]
fn test_alignment_config_with_layers() {
    let config = AlignmentConfig::default().with_layers(vec![1, 2, 3]);
    assert_eq!(config.layers, vec![1, 2, 3]);
}

#[test]
fn test_alignment_config_with_min_attention() {
    let config = AlignmentConfig::default().with_min_attention(0.2);
    assert!((config.min_attention - 0.2).abs() < f32::EPSILON);
}

// =========================================================================
// TokenAlignment Tests
// =========================================================================

#[test]
fn test_token_alignment_new() {
    let alignment = TokenAlignment::new(0, 100, 50, 0.9);
    assert_eq!(alignment.token_index, 0);
    assert_eq!(alignment.token_id, 100);
    assert_eq!(alignment.frame_position, 50);
    assert!((alignment.confidence - 0.9).abs() < f32::EPSILON);
    assert!((alignment.start_time - 1.0).abs() < f32::EPSILON); // 50 / 50 fps
}

#[test]
fn test_token_alignment_set_end_time() {
    let mut alignment = TokenAlignment::new(0, 100, 50, 0.9);
    alignment.set_end_time(100);
    assert!((alignment.end_time - 2.0).abs() < f32::EPSILON); // 100 / 50 fps
}

#[test]
fn test_token_alignment_duration() {
    let mut alignment = TokenAlignment::new(0, 100, 50, 0.9);
    alignment.set_end_time(100);
    assert!((alignment.duration() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_token_alignment_is_confident() {
    let high = TokenAlignment::new(0, 100, 50, 0.6);
    let low = TokenAlignment::new(0, 100, 50, 0.4);
    assert!(high.is_confident());
    assert!(!low.is_confident());
}

#[test]
fn test_token_alignment_with_attention_weights() {
    let alignment =
        TokenAlignment::new(0, 100, 50, 0.9).with_attention_weights(vec![0.1, 0.2, 0.7]);
    assert_eq!(alignment.attention_weights, vec![0.1, 0.2, 0.7]);
}

// =========================================================================
// WordAlignment Tests
// =========================================================================

#[test]
fn test_word_alignment_new() {
    let tokens = vec![
        TokenAlignment::new(0, 100, 50, 0.9),
        TokenAlignment::new(1, 101, 60, 0.8),
    ];
    let word = WordAlignment::new("hello".to_string(), tokens);

    assert_eq!(word.word, "hello");
    assert!((word.start_time - 1.0).abs() < f32::EPSILON);
    assert!((word.confidence - 0.85).abs() < f32::EPSILON);
    assert_eq!(word.token_count(), 2);
}

#[test]
fn test_word_alignment_empty_tokens() {
    let word = WordAlignment::new("empty".to_string(), vec![]);
    assert!((word.start_time - 0.0).abs() < f32::EPSILON);
    assert!((word.end_time - 0.0).abs() < f32::EPSILON);
    assert!((word.confidence - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_word_alignment_duration() {
    let mut token1 = TokenAlignment::new(0, 100, 50, 0.9);
    token1.set_end_time(60);
    let mut token2 = TokenAlignment::new(1, 101, 60, 0.8);
    token2.set_end_time(80);

    let word = WordAlignment::new("test".to_string(), vec![token1, token2]);
    assert!((word.duration() - 0.6).abs() < 0.01); // (80-50) / 50 fps
}

// =========================================================================
// CrossAttentionAlignment Tests
// =========================================================================

#[test]
fn test_cross_attention_alignment_new() {
    let alignment = CrossAttentionAlignment::new(AlignmentConfig::default());
    assert!((alignment.config.min_attention - 0.1).abs() < f32::EPSILON);
}

#[test]
fn test_cross_attention_alignment_default() {
    let alignment = CrossAttentionAlignment::default();
    assert_eq!(alignment.config.layers.len(), 6);
}

#[test]
fn test_extract_token_alignments_empty() {
    let alignment = CrossAttentionAlignment::default();
    let result = alignment.extract_token_alignments(&[], &[], 100);
    assert!(result.is_err());
}

#[test]
fn test_extract_token_alignments_no_tokens() {
    let alignment = CrossAttentionAlignment::default();
    // Create minimal attention weights
    let weights = vec![vec![vec![vec![0.1f32; 10]; 1]; 4]; 6];
    let result = alignment.extract_token_alignments(&weights, &[], 10);
    assert!(result.is_ok());
    assert!(result.expect("should succeed").is_empty());
}

#[test]
fn test_extract_token_alignments_single_token() {
    let config = AlignmentConfig::default().with_layers(vec![0]);
    let alignment = CrossAttentionAlignment::new(config);

    // Create attention weights with peak at frame 5
    let mut token_attention = vec![0.1f32; 10];
    token_attention[5] = 0.9;

    let weights = vec![vec![vec![token_attention]; 1]];
    let token_ids = vec![100u32];

    let result = alignment
        .extract_token_alignments(&weights, &token_ids, 10)
        .expect("should succeed");

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].frame_position, 5);
    assert!(result[0].confidence > 0.0);
}

#[test]
fn test_find_peak() {
    let alignment = CrossAttentionAlignment::default();
    let attention = vec![0.1, 0.2, 0.8, 0.3, 0.1];
    let (peak_idx, peak_val) = alignment.find_peak(&attention);

    assert_eq!(peak_idx, 2);
    assert!((peak_val - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_find_peak_empty() {
    let alignment = CrossAttentionAlignment::default();
    let (peak_idx, peak_val) = alignment.find_peak(&[]);
    assert_eq!(peak_idx, 0);
    assert!((peak_val - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_compute_confidence() {
    let alignment = CrossAttentionAlignment::default();

    // High concentration attention
    let attention = vec![0.0, 0.0, 0.9, 0.1, 0.0];
    let confidence = alignment.compute_confidence(&attention, 2, 0.9);
    assert!(confidence > 0.5);

    // Flat attention
    let flat_attention = vec![0.2, 0.2, 0.2, 0.2, 0.2];
    let flat_confidence = alignment.compute_confidence(&flat_attention, 2, 0.2);
    assert!(flat_confidence < confidence);
}

// =========================================================================
// WordTimestampExtractor Tests
// =========================================================================

#[test]
fn test_word_timestamp_extractor_new() {
    let extractor = WordTimestampExtractor::new(AlignmentConfig::default());
    assert!((extractor.alignment.config.min_attention - 0.1).abs() < f32::EPSILON);
}

#[test]
fn test_word_timestamp_extractor_default() {
    let extractor = WordTimestampExtractor::default();
    assert_eq!(extractor.alignment.config.layers.len(), 6);
}

#[test]
fn test_group_tokens_into_words_simple() {
    let extractor = WordTimestampExtractor::default();

    let alignments = vec![
        TokenAlignment::new(0, 100, 0, 0.9),
        TokenAlignment::new(1, 101, 10, 0.8),
        TokenAlignment::new(2, 102, 20, 0.85),
    ];

    let texts = vec!["hello".to_string(), " world".to_string(), "!".to_string()];

    let words = extractor.group_tokens_into_words(&alignments, &texts);

    assert_eq!(words.len(), 2);
    assert_eq!(words[0].word, "hello");
    assert_eq!(words[1].word, "world!");
}

#[test]
fn test_group_tokens_into_words_sentencepiece() {
    let extractor = WordTimestampExtractor::default();

    let alignments = vec![
        TokenAlignment::new(0, 100, 0, 0.9),
        TokenAlignment::new(1, 101, 10, 0.8),
    ];

    let texts = vec!["▁hello".to_string(), "▁world".to_string()];

    let words = extractor.group_tokens_into_words(&alignments, &texts);

    assert_eq!(words.len(), 2);
    assert_eq!(words[0].word, "hello");
    assert_eq!(words[1].word, "world");
}

#[test]
fn test_group_tokens_into_words_empty() {
    let extractor = WordTimestampExtractor::default();
    let words = extractor.group_tokens_into_words(&[], &[]);
    assert!(words.is_empty());
}

// Additional coverage tests (unique)

#[test]
fn test_token_alignment_with_attention_weights_coverage() {
    let alignment =
        TokenAlignment::new(0, 100, 50, 0.9).with_attention_weights(vec![0.1, 0.2, 0.3]);
    assert_eq!(alignment.attention_weights.len(), 3);
}

#[test]
fn test_alignment_config_with_layers_coverage() {
    let config = AlignmentConfig::default().with_layers(vec![0, 1, 2]);
    assert_eq!(config.layers, vec![0, 1, 2]);
}

#[test]
fn test_alignment_config_with_min_attention_coverage() {
    let config = AlignmentConfig::default().with_min_attention(0.2);
    assert!((config.min_attention - 0.2).abs() < 0.001);
}

// =========================================================================
// extract_word_alignments Tests (impact 29.1)
// =========================================================================

#[test]
fn test_extract_word_alignments_basic() {
    let config = AlignmentConfig::default().with_layers(vec![0]);
    let extractor = WordTimestampExtractor::new(config);

    // Create attention weights: 1 layer, 1 head, 3 tokens, 10 frames
    // Token 0 peaks at frame 0, token 1 at frame 3, token 2 at frame 7
    let mut attn_0 = vec![0.1f32; 10];
    attn_0[0] = 0.9;
    let mut attn_1 = vec![0.1f32; 10];
    attn_1[3] = 0.9;
    let mut attn_2 = vec![0.1f32; 10];
    attn_2[7] = 0.9;

    let weights = vec![vec![vec![attn_0, attn_1, attn_2]]];
    let token_ids = vec![100u32, 101, 102];
    let token_texts = vec!["hello".to_string(), " world".to_string(), "!".to_string()];

    let words = extractor
        .extract_word_alignments(&weights, &token_ids, &token_texts, 10)
        .expect("should extract word alignments");

    assert_eq!(words.len(), 2);
    assert_eq!(words[0].word, "hello");
    assert_eq!(words[1].word, "world!");
    assert!(words[0].confidence > 0.0);
    assert!(words[1].confidence > 0.0);
}

#[test]
fn test_extract_word_alignments_single_word() {
    let config = AlignmentConfig::default().with_layers(vec![0]);
    let extractor = WordTimestampExtractor::new(config);

    let mut attn = vec![0.1f32; 10];
    attn[5] = 0.9;

    let weights = vec![vec![vec![attn]]];
    let token_ids = vec![100u32];
    let token_texts = vec!["hello".to_string()];

    let words = extractor
        .extract_word_alignments(&weights, &token_ids, &token_texts, 10)
        .expect("should succeed");

    assert_eq!(words.len(), 1);
    assert_eq!(words[0].word, "hello");
}

#[test]
fn test_extract_word_alignments_empty_tokens() {
    let config = AlignmentConfig::default().with_layers(vec![0]);
    let extractor = WordTimestampExtractor::new(config);

    let weights = vec![vec![vec![vec![0.1f32; 10]]]];
    let token_ids: Vec<u32> = vec![];
    let token_texts: Vec<String> = vec![];

    let words = extractor
        .extract_word_alignments(&weights, &token_ids, &token_texts, 10)
        .expect("should succeed");

    assert!(words.is_empty());
}

// =========================================================================
// average_attention Tests (impact 1.2, partial coverage)
// =========================================================================

#[test]
fn test_average_attention_layer_filtering() {
    // Config only uses layer 0
    let config = AlignmentConfig::default().with_layers(vec![0]);
    let alignment = CrossAttentionAlignment::new(config);

    // 2 layers, 1 head each, 2 tokens, 4 frames
    let layer0_attn = vec![vec![vec![1.0f32; 4]; 2]]; // all 1.0
    let layer1_attn = vec![vec![vec![0.0f32; 4]; 2]]; // all 0.0

    let weights = vec![layer0_attn, layer1_attn];
    let result = alignment
        .average_attention(&weights, 4, 2)
        .expect("should succeed");

    // Should only include layer 0 (1.0), layer 1 should be ignored
    assert_eq!(result.len(), 2);
    assert!((result[0][0] - 1.0).abs() < 0.01);
}

#[test]
fn test_average_attention_head_filtering() {
    let mut config = AlignmentConfig::default().with_layers(vec![0]);
    config.heads = Some(vec![0]); // Only head 0
    let alignment = CrossAttentionAlignment::new(config);

    // 1 layer, 2 heads, 1 token, 4 frames
    let head0_attn = vec![vec![1.0f32; 4]]; // all 1.0
    let head1_attn = vec![vec![0.0f32; 4]]; // all 0.0

    let weights = vec![vec![head0_attn, head1_attn]];
    let result = alignment
        .average_attention(&weights, 4, 1)
        .expect("should succeed");

    assert_eq!(result.len(), 1);
    assert!((result[0][0] - 1.0).abs() < 0.01);
}
