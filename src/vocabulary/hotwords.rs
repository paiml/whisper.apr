//! Hotword boosting via logit biasing (WAPR-170)
//!
//! Boosts recognition of specific words/phrases by adding bias to token logits.
//!
//! # Overview
//!
//! Hotword boosting works by:
//! 1. Tokenizing hotwords into their BPE token sequences
//! 2. Tracking partial matches during decoding
//! 3. Adding bias to logits for tokens that continue a partial match
//!
//! # Algorithm
//!
//! During decoding:
//! - If context tokens match a hotword prefix, boost the next expected token
//! - Bias is scaled by match confidence and hotword weight
//! - Multiple overlapping hotwords are handled independently
//!
//! # Example
//!
//! ```rust,ignore
//! use whisper_apr::vocabulary::{HotwordBooster, Hotword};
//!
//! let mut booster = HotwordBooster::new();
//! booster.add_hotword_with_tokens("Anthropic", vec![1234, 5678], 2.0);
//! booster.add_hotword_with_tokens("Claude", vec![9012], 1.5);
//!
//! // During decoding
//! let biased_logits = booster.apply_bias(&mut logits, &context_tokens);
//! ```

use std::collections::HashMap;

/// Configuration for hotword boosting
#[derive(Debug, Clone)]
pub struct HotwordConfig {
    /// Default bias value for hotwords without explicit bias
    pub default_bias: f32,
    /// Maximum bias value (prevents extreme boosting)
    pub max_bias: f32,
    /// Minimum token sequence length for matching
    pub min_tokens: usize,
    /// Whether to match case-sensitively
    pub case_sensitive: bool,
    /// Decay factor for partial matches (longer matches get more boost)
    pub partial_match_decay: f32,
}

impl HotwordConfig {
    /// Create default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_bias: 1.0,
            max_bias: 5.0,
            min_tokens: 1,
            case_sensitive: false,
            partial_match_decay: 0.9,
        }
    }

    /// Set default bias
    #[must_use]
    pub fn with_default_bias(mut self, bias: f32) -> Self {
        self.default_bias = bias;
        self
    }

    /// Set maximum bias
    #[must_use]
    pub fn with_max_bias(mut self, max: f32) -> Self {
        self.max_bias = max;
        self
    }

    /// Set minimum tokens for matching
    #[must_use]
    pub fn with_min_tokens(mut self, min: usize) -> Self {
        self.min_tokens = min;
        self
    }

    /// Set case sensitivity
    #[must_use]
    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Set partial match decay
    #[must_use]
    pub fn with_partial_match_decay(mut self, decay: f32) -> Self {
        self.partial_match_decay = decay;
        self
    }
}

impl Default for HotwordConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A hotword with its token sequence and boost value
#[derive(Debug, Clone)]
pub struct Hotword {
    /// Original text of the hotword
    pub text: String,
    /// Token sequence for this hotword
    pub tokens: Vec<u32>,
    /// Bias value to add to logits
    pub bias: f32,
    /// Priority for conflict resolution (higher = more priority)
    pub priority: u32,
}

impl Hotword {
    /// Create a new hotword
    #[must_use]
    pub fn new(text: String, tokens: Vec<u32>, bias: f32) -> Self {
        Self {
            text,
            tokens,
            bias,
            priority: 0,
        }
    }

    /// Set priority
    #[must_use]
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Get token sequence length
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if hotword is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Check if context ends with hotword prefix
    ///
    /// Returns the number of matching tokens (0 if no match)
    #[must_use]
    pub fn prefix_match_len(&self, context: &[u32]) -> usize {
        if self.tokens.is_empty() || context.is_empty() {
            return 0;
        }

        // Check each possible prefix length
        for prefix_len in (1..=self.tokens.len().min(context.len())).rev() {
            let hotword_prefix = &self.tokens[..prefix_len];
            let context_suffix = &context[context.len() - prefix_len..];

            if hotword_prefix == context_suffix {
                return prefix_len;
            }
        }

        0
    }

    /// Get the next expected token after a prefix match
    #[must_use]
    pub fn next_token(&self, prefix_len: usize) -> Option<u32> {
        if prefix_len < self.tokens.len() {
            Some(self.tokens[prefix_len])
        } else {
            None
        }
    }
}

/// Hotword booster for logit biasing
#[derive(Debug, Clone)]
pub struct HotwordBooster {
    /// Configuration
    config: HotwordConfig,
    /// Registered hotwords
    hotwords: Vec<Hotword>,
    /// Token to hotword index map for quick lookup
    first_token_map: HashMap<u32, Vec<usize>>,
}

impl HotwordBooster {
    /// Create a new hotword booster with default config
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(HotwordConfig::default())
    }

    /// Create a new hotword booster with custom config
    #[must_use]
    pub fn with_config(config: HotwordConfig) -> Self {
        Self {
            config,
            hotwords: Vec::new(),
            first_token_map: HashMap::new(),
        }
    }

    /// Add a hotword with its token sequence
    pub fn add_hotword_with_tokens(&mut self, text: &str, tokens: Vec<u32>, bias: f32) {
        if tokens.is_empty() {
            return;
        }

        let clamped_bias = bias.clamp(-self.config.max_bias, self.config.max_bias);
        let first_token = tokens[0];
        let hotword_idx = self.hotwords.len();

        self.hotwords
            .push(Hotword::new(text.to_string(), tokens, clamped_bias));

        self.first_token_map
            .entry(first_token)
            .or_default()
            .push(hotword_idx);
    }

    /// Add a hotword with default bias
    pub fn add_hotword_with_tokens_default(&mut self, text: &str, tokens: Vec<u32>) {
        self.add_hotword_with_tokens(text, tokens, self.config.default_bias);
    }

    /// Remove all hotwords
    pub fn clear(&mut self) {
        self.hotwords.clear();
        self.first_token_map.clear();
    }

    /// Get number of registered hotwords
    #[must_use]
    pub fn len(&self) -> usize {
        self.hotwords.len()
    }

    /// Check if no hotwords registered
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hotwords.is_empty()
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &HotwordConfig {
        &self.config
    }

    /// Get all hotwords
    #[must_use]
    pub fn hotwords(&self) -> &[Hotword] {
        &self.hotwords
    }

    /// Apply bias to logits based on context
    ///
    /// # Arguments
    /// * `logits` - Mutable slice of logits to modify
    /// * `context` - Previous token sequence
    pub fn apply_bias(&self, logits: &mut [f32], context: &[u32]) {
        if self.hotwords.is_empty() {
            return;
        }

        // Collect all biases to apply
        let biases = self.compute_biases(context);

        // Apply biases to logits
        for (token_id, bias) in biases {
            if (token_id as usize) < logits.len() {
                logits[token_id as usize] += bias;
            }
        }
    }

    /// Compute biases for all matching hotwords
    fn compute_biases(&self, context: &[u32]) -> Vec<(u32, f32)> {
        let mut biases: HashMap<u32, f32> = HashMap::new();

        for hotword in &self.hotwords {
            // Check for prefix match
            let match_len = hotword.prefix_match_len(context);

            if match_len > 0 {
                // We have a partial match, boost the next token
                if let Some(next_token) = hotword.next_token(match_len) {
                    // Scale bias by how much of the hotword is matched
                    let progress = match_len as f32 / hotword.tokens.len() as f32;
                    let scaled_bias = hotword.bias * (1.0 + progress);

                    // Accumulate biases (multiple hotwords may boost same token)
                    *biases.entry(next_token).or_insert(0.0) += scaled_bias;
                }
            } else if context.is_empty() || !self.has_recent_hotword_match(context) {
                // No match yet, boost first token of all hotwords
                let first_token = hotword.tokens[0];
                let scaled_bias = hotword.bias * self.config.partial_match_decay;
                *biases.entry(first_token).or_insert(0.0) += scaled_bias;
            }
        }

        // Clamp final biases
        biases
            .into_iter()
            .map(|(token, bias)| {
                (
                    token,
                    bias.clamp(-self.config.max_bias, self.config.max_bias),
                )
            })
            .collect()
    }

    /// Check if context recently matched a hotword
    fn has_recent_hotword_match(&self, context: &[u32]) -> bool {
        if context.is_empty() {
            return false;
        }

        for hotword in &self.hotwords {
            if hotword.prefix_match_len(context) > 0 {
                return true;
            }
        }

        false
    }

    /// Get tokens that would complete any registered hotword
    #[must_use]
    pub fn get_completion_tokens(&self, context: &[u32]) -> Vec<(u32, f32)> {
        self.compute_biases(context)
    }
}

impl Default for HotwordBooster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
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
}
