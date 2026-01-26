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

#[cfg(test)]
mod tests;

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
