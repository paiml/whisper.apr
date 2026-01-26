//! Tests for hotword boosting

use super::*;

// ============================================================
// HotwordConfig Tests
// ============================================================

#[test]
fn test_hotword_config_new() {
    let config = HotwordConfig::new();
    assert!((config.default_bias - 1.0).abs() < f32::EPSILON);
    assert!((config.max_bias - 5.0).abs() < f32::EPSILON);
    assert_eq!(config.min_tokens, 1);
    assert!(!config.case_sensitive);
    assert!((config.partial_match_decay - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_hotword_config_default() {
    let config = HotwordConfig::default();
    assert!((config.default_bias - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_hotword_config_with_default_bias() {
    let config = HotwordConfig::new().with_default_bias(2.5);
    assert!((config.default_bias - 2.5).abs() < f32::EPSILON);
}

#[test]
fn test_hotword_config_with_max_bias() {
    let config = HotwordConfig::new().with_max_bias(10.0);
    assert!((config.max_bias - 10.0).abs() < f32::EPSILON);
}

#[test]
fn test_hotword_config_with_min_tokens() {
    let config = HotwordConfig::new().with_min_tokens(3);
    assert_eq!(config.min_tokens, 3);
}

#[test]
fn test_hotword_config_with_case_sensitive() {
    let config = HotwordConfig::new().with_case_sensitive(true);
    assert!(config.case_sensitive);
}

#[test]
fn test_hotword_config_with_partial_match_decay() {
    let config = HotwordConfig::new().with_partial_match_decay(0.5);
    assert!((config.partial_match_decay - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_hotword_config_builder_chain() {
    let config = HotwordConfig::new()
        .with_default_bias(2.0)
        .with_max_bias(8.0)
        .with_min_tokens(2)
        .with_case_sensitive(true)
        .with_partial_match_decay(0.7);

    assert!((config.default_bias - 2.0).abs() < f32::EPSILON);
    assert!((config.max_bias - 8.0).abs() < f32::EPSILON);
    assert_eq!(config.min_tokens, 2);
    assert!(config.case_sensitive);
    assert!((config.partial_match_decay - 0.7).abs() < f32::EPSILON);
}

// ============================================================
// Hotword Tests
// ============================================================

#[test]
fn test_hotword_new() {
    let hotword = Hotword::new("test".to_string(), vec![100, 200], 1.5);
    assert_eq!(hotword.text, "test");
    assert_eq!(hotword.tokens, vec![100, 200]);
    assert!((hotword.bias - 1.5).abs() < f32::EPSILON);
    assert_eq!(hotword.priority, 0);
}

#[test]
fn test_hotword_with_priority() {
    let hotword = Hotword::new("test".to_string(), vec![100], 1.0).with_priority(5);
    assert_eq!(hotword.priority, 5);
}

#[test]
fn test_hotword_len() {
    let hotword = Hotword::new("test".to_string(), vec![1, 2, 3], 1.0);
    assert_eq!(hotword.len(), 3);
}

#[test]
fn test_hotword_is_empty() {
    let empty = Hotword::new("".to_string(), vec![], 1.0);
    let non_empty = Hotword::new("test".to_string(), vec![1], 1.0);
    assert!(empty.is_empty());
    assert!(!non_empty.is_empty());
}

#[test]
fn test_hotword_prefix_match_len_no_match() {
    let hotword = Hotword::new("test".to_string(), vec![100, 200, 300], 1.0);
    let context = vec![1, 2, 3];
    assert_eq!(hotword.prefix_match_len(&context), 0);
}

#[test]
fn test_hotword_prefix_match_len_single_token() {
    let hotword = Hotword::new("test".to_string(), vec![100, 200, 300], 1.0);
    let context = vec![1, 2, 100];
    assert_eq!(hotword.prefix_match_len(&context), 1);
}

#[test]
fn test_hotword_prefix_match_len_multiple_tokens() {
    let hotword = Hotword::new("test".to_string(), vec![100, 200, 300], 1.0);
    let context = vec![1, 100, 200];
    assert_eq!(hotword.prefix_match_len(&context), 2);
}

#[test]
fn test_hotword_prefix_match_len_full_match() {
    let hotword = Hotword::new("test".to_string(), vec![100, 200], 1.0);
    let context = vec![100, 200];
    assert_eq!(hotword.prefix_match_len(&context), 2);
}

#[test]
fn test_hotword_prefix_match_len_empty_context() {
    let hotword = Hotword::new("test".to_string(), vec![100], 1.0);
    let context: Vec<u32> = vec![];
    assert_eq!(hotword.prefix_match_len(&context), 0);
}

#[test]
fn test_hotword_prefix_match_len_empty_tokens() {
    let hotword = Hotword::new("".to_string(), vec![], 1.0);
    let context = vec![1, 2, 3];
    assert_eq!(hotword.prefix_match_len(&context), 0);
}

#[test]
fn test_hotword_next_token() {
    let hotword = Hotword::new("test".to_string(), vec![100, 200, 300], 1.0);
    assert_eq!(hotword.next_token(0), Some(100));
    assert_eq!(hotword.next_token(1), Some(200));
    assert_eq!(hotword.next_token(2), Some(300));
    assert_eq!(hotword.next_token(3), None);
}

// ============================================================
// HotwordBooster Tests
// ============================================================

#[test]
fn test_hotword_booster_new() {
    let booster = HotwordBooster::new();
    assert!(booster.is_empty());
    assert_eq!(booster.len(), 0);
}

#[test]
fn test_hotword_booster_default() {
    let booster = HotwordBooster::default();
    assert!(booster.is_empty());
}

#[test]
fn test_hotword_booster_with_config() {
    let config = HotwordConfig::new().with_max_bias(10.0);
    let booster = HotwordBooster::with_config(config);
    assert!((booster.config().max_bias - 10.0).abs() < f32::EPSILON);
}

#[test]
fn test_hotword_booster_add_hotword() {
    let mut booster = HotwordBooster::new();
    booster.add_hotword_with_tokens("test", vec![100, 200], 2.0);

    assert_eq!(booster.len(), 1);
    assert!(!booster.is_empty());
    assert_eq!(booster.hotwords()[0].text, "test");
    assert!((booster.hotwords()[0].bias - 2.0).abs() < f32::EPSILON);
}

#[test]
fn test_hotword_booster_add_hotword_empty() {
    let mut booster = HotwordBooster::new();
    booster.add_hotword_with_tokens("empty", vec![], 2.0);

    // Should not add empty token sequences
    assert!(booster.is_empty());
}

#[test]
fn test_hotword_booster_add_hotword_clamps_bias() {
    let config = HotwordConfig::new().with_max_bias(3.0);
    let mut booster = HotwordBooster::with_config(config);
    booster.add_hotword_with_tokens("test", vec![100], 10.0);

    // Bias should be clamped to max_bias
    assert!((booster.hotwords()[0].bias - 3.0).abs() < f32::EPSILON);
}

#[test]
fn test_hotword_booster_add_hotword_default_bias() {
    let config = HotwordConfig::new().with_default_bias(1.5);
    let mut booster = HotwordBooster::with_config(config);
    booster.add_hotword_with_tokens_default("test", vec![100]);

    assert!((booster.hotwords()[0].bias - 1.5).abs() < f32::EPSILON);
}

#[test]
fn test_hotword_booster_clear() {
    let mut booster = HotwordBooster::new();
    booster.add_hotword_with_tokens("test1", vec![100], 1.0);
    booster.add_hotword_with_tokens("test2", vec![200], 1.0);

    assert_eq!(booster.len(), 2);
    booster.clear();
    assert!(booster.is_empty());
}

#[test]
fn test_hotword_booster_apply_bias_empty() {
    let booster = HotwordBooster::new();
    let mut logits = vec![0.0, 1.0, 2.0];
    let context: Vec<u32> = vec![];

    booster.apply_bias(&mut logits, &context);

    // No hotwords, logits unchanged
    assert!((logits[0] - 0.0).abs() < f32::EPSILON);
    assert!((logits[1] - 1.0).abs() < f32::EPSILON);
    assert!((logits[2] - 2.0).abs() < f32::EPSILON);
}

#[test]
fn test_hotword_booster_apply_bias_first_token() {
    let mut booster = HotwordBooster::new();
    booster.add_hotword_with_tokens("test", vec![1], 2.0);

    let mut logits = vec![0.0, 0.0, 0.0];
    let context: Vec<u32> = vec![];

    booster.apply_bias(&mut logits, &context);

    // First token (index 1) should be boosted
    assert!(logits[1] > 0.0);
}

#[test]
fn test_hotword_booster_apply_bias_continuation() {
    let mut booster = HotwordBooster::new();
    booster.add_hotword_with_tokens("test", vec![100, 200], 2.0);

    let mut logits = vec![0.0; 300];
    let context = vec![50, 100]; // Ends with first token of hotword

    booster.apply_bias(&mut logits, &context);

    // Second token (200) should be boosted
    assert!(logits[200] > 0.0);
}

#[test]
fn test_hotword_booster_apply_bias_multiple_hotwords() {
    let mut booster = HotwordBooster::new();
    booster.add_hotword_with_tokens("first", vec![10], 1.0);
    booster.add_hotword_with_tokens("second", vec![20], 1.5);

    let mut logits = vec![0.0; 30];
    let context: Vec<u32> = vec![];

    booster.apply_bias(&mut logits, &context);

    // Both first tokens should be boosted
    assert!(logits[10] > 0.0);
    assert!(logits[20] > 0.0);
}

#[test]
fn test_hotword_booster_apply_bias_out_of_bounds() {
    let mut booster = HotwordBooster::new();
    booster.add_hotword_with_tokens("test", vec![1000], 2.0);

    let mut logits = vec![0.0; 10]; // Only 10 logits
    let context: Vec<u32> = vec![];

    // Should not panic even if token index is out of bounds
    booster.apply_bias(&mut logits, &context);

    // Logits should remain unchanged
    for &logit in &logits {
        assert!((logit - 0.0).abs() < f32::EPSILON);
    }
}

#[test]
fn test_hotword_booster_get_completion_tokens() {
    let mut booster = HotwordBooster::new();
    booster.add_hotword_with_tokens("test", vec![100, 200], 2.0);

    let context = vec![50, 100];
    let completions = booster.get_completion_tokens(&context);

    assert!(!completions.is_empty());
    assert!(completions.iter().any(|(token, _)| *token == 200));
}

#[test]
fn test_hotword_booster_scaled_bias_by_progress() {
    let mut booster = HotwordBooster::new();
    booster.add_hotword_with_tokens("test", vec![100, 200, 300], 2.0);

    // Check bias at different match positions
    let context_1 = vec![100]; // 1/3 matched
    let context_2 = vec![100, 200]; // 2/3 matched

    let completions_1 = booster.get_completion_tokens(&context_1);
    let completions_2 = booster.get_completion_tokens(&context_2);

    let bias_1 = completions_1
        .iter()
        .find(|(t, _)| *t == 200)
        .map(|(_, b)| *b)
        .unwrap_or(0.0);
    let bias_2 = completions_2
        .iter()
        .find(|(t, _)| *t == 300)
        .map(|(_, b)| *b)
        .unwrap_or(0.0);

    // Bias should increase as we match more of the hotword
    assert!(bias_2 > bias_1);
}

#[test]
fn test_hotword_booster_negative_bias() {
    let mut booster = HotwordBooster::new();
    booster.add_hotword_with_tokens("suppress", vec![50], -2.0);

    let mut logits = vec![5.0; 100];
    let context: Vec<u32> = vec![];

    booster.apply_bias(&mut logits, &context);

    // Token 50 should be suppressed (lower logit)
    assert!(logits[50] < 5.0);
}

#[test]
fn test_hotword_booster_overlapping_hotwords() {
    let mut booster = HotwordBooster::new();
    // Two hotwords that share the same first token
    booster.add_hotword_with_tokens("hello", vec![100, 200], 1.0);
    booster.add_hotword_with_tokens("help", vec![100, 300], 1.0);

    let mut logits = vec![0.0; 400];
    let context = vec![100]; // First token matched for both

    booster.apply_bias(&mut logits, &context);

    // Both continuations should be boosted
    assert!(logits[200] > 0.0);
    assert!(logits[300] > 0.0);
}

#[test]
fn test_hotword_booster_multiple_add_same_first_token() {
    let mut booster = HotwordBooster::new();
    booster.add_hotword_with_tokens("word1", vec![10, 20], 1.0);
    booster.add_hotword_with_tokens("word2", vec![10, 30], 1.5);

    // Should handle multiple hotwords starting with same token
    assert_eq!(booster.len(), 2);
}
