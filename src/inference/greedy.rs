//! Greedy decoding
//!
//! Fast, memory-efficient greedy token selection.
//!
//! # Algorithm
//!
//! At each step, selects the token with highest probability (argmax).
//! This is the fastest decoding method but may not find optimal sequences.
//!
//! # Example
//!
//! ```rust,ignore
//! use whisper_apr::inference::GreedyDecoder;
//!
//! let decoder = GreedyDecoder::new(448);
//! let tokens = decoder.decode(&logits, n_vocab, sot_token, eot_token);
//! ```

use crate::error::WhisperResult;
use crate::tokenizer::special_tokens;

/// Greedy decoder for token generation
///
/// Simple argmax selection at each step. Fast and memory-efficient.
#[derive(Debug, Clone)]
pub struct GreedyDecoder {
    /// Maximum tokens to generate
    max_tokens: usize,
    /// Temperature for logit scaling (0.0 = argmax)
    temperature: f32,
}

impl GreedyDecoder {
    /// Create a new greedy decoder
    #[must_use]
    pub const fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            temperature: 0.0,
        }
    }

    /// Set temperature for sampling
    ///
    /// At temperature 0.0 (default), uses pure argmax.
    /// Higher temperatures make the distribution softer.
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Get maximum tokens
    #[must_use]
    pub const fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Get temperature
    #[must_use]
    pub const fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Decode a single step from logits
    ///
    /// # Arguments
    /// * `logits` - Logits over vocabulary (n_vocab)
    ///
    /// # Returns
    /// Selected token ID
    pub fn decode_step(&self, logits: &[f32]) -> u32 {
        if self.temperature == 0.0 {
            // Pure argmax
            Self::argmax(logits)
        } else {
            // Temperature-scaled sampling
            self.sample_with_temperature(logits)
        }
    }

    /// Select argmax token
    fn argmax(logits: &[f32]) -> u32 {
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;

        for (idx, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }

        max_idx as u32
    }

    /// Sample with temperature scaling
    fn sample_with_temperature(&self, logits: &[f32]) -> u32 {
        // Scale logits by temperature
        let scaled: Vec<f32> = logits.iter().map(|&x| x / self.temperature).collect();

        // Convert to probabilities with softmax
        let probs = softmax(&scaled);

        // Sample from distribution (deterministic for reproducibility - use max)
        // In a real implementation, this would use random sampling
        Self::argmax(&probs)
    }

    /// Run greedy decoding to generate a sequence of tokens
    ///
    /// # Arguments
    /// * `logits_fn` - Function that takes current tokens and returns next logits
    /// * `n_vocab` - Vocabulary size
    /// * `initial_tokens` - Initial tokens to start with (e.g., [SOT, language, task])
    ///
    /// # Returns
    /// Generated token sequence (including initial tokens)
    pub fn decode<F>(&self, mut logits_fn: F, initial_tokens: &[u32]) -> WhisperResult<Vec<u32>>
    where
        F: FnMut(&[u32]) -> WhisperResult<Vec<f32>>,
    {
        let mut tokens = initial_tokens.to_vec();
        let eot = special_tokens::EOT;

        for _ in 0..self.max_tokens {
            // Get logits for next token
            let logits = logits_fn(&tokens)?;

            // Select next token
            let next_token = self.decode_step(&logits);

            // Stop if EOT
            if next_token == eot {
                tokens.push(next_token);
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Apply logit suppression for specific tokens
    ///
    /// # Arguments
    /// * `logits` - Logits to modify (mutated in place)
    /// * `suppress_tokens` - Tokens to suppress (set to -inf)
    pub fn suppress_tokens(logits: &mut [f32], suppress_tokens: &[u32]) {
        for &token in suppress_tokens {
            if (token as usize) < logits.len() {
                logits[token as usize] = f32::NEG_INFINITY;
            }
        }
    }

    /// Apply logit bias for specific tokens
    ///
    /// # Arguments
    /// * `logits` - Logits to modify (mutated in place)
    /// * `biases` - Vec of (token, bias) pairs
    pub fn apply_bias(logits: &mut [f32], biases: &[(u32, f32)]) {
        for &(token, bias) in biases {
            if (token as usize) < logits.len() {
                logits[token as usize] += bias;
            }
        }
    }
}

impl Default for GreedyDecoder {
    fn default() -> Self {
        Self::new(448)
    }
}

/// Compute softmax probabilities
fn softmax(logits: &[f32]) -> Vec<f32> {
    // Find max for numerical stability
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Compute exp and sum
    let exp_vals: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();

    // Normalize
    if sum > 1e-10 {
        exp_vals.iter().map(|&x| x / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Construction Tests
    // =========================================================================

    #[test]
    fn test_greedy_decoder_new() {
        let decoder = GreedyDecoder::new(448);
        assert_eq!(decoder.max_tokens(), 448);
        assert!((decoder.temperature() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_greedy_decoder_with_temperature() {
        let decoder = GreedyDecoder::new(448).with_temperature(0.5);
        assert!((decoder.temperature() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_greedy_decoder_default() {
        let decoder = GreedyDecoder::default();
        assert_eq!(decoder.max_tokens(), 448);
    }

    // =========================================================================
    // Argmax Tests
    // =========================================================================

    #[test]
    fn test_argmax_simple() {
        let logits = vec![0.1, 0.5, 0.2, 0.9, 0.3];
        let result = GreedyDecoder::argmax(&logits);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_argmax_first_element() {
        let logits = vec![1.0, 0.5, 0.2, 0.3];
        let result = GreedyDecoder::argmax(&logits);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_argmax_last_element() {
        let logits = vec![0.1, 0.5, 0.2, 1.0];
        let result = GreedyDecoder::argmax(&logits);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_argmax_negative_values() {
        let logits = vec![-1.0, -0.5, -2.0, -0.1];
        let result = GreedyDecoder::argmax(&logits);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_argmax_single_element() {
        let logits = vec![0.5];
        let result = GreedyDecoder::argmax(&logits);
        assert_eq!(result, 0);
    }

    // =========================================================================
    // Decode Step Tests
    // =========================================================================

    #[test]
    fn test_decode_step_argmax() {
        let decoder = GreedyDecoder::new(10);
        let logits = vec![0.1, 0.5, 0.9, 0.2];
        let result = decoder.decode_step(&logits);
        assert_eq!(result, 2);
    }

    #[test]
    fn test_decode_step_with_temperature() {
        let decoder = GreedyDecoder::new(10).with_temperature(1.0);
        let logits = vec![0.1, 0.5, 0.9, 0.2];
        let result = decoder.decode_step(&logits);
        // With temperature sampling (but using argmax of softmax), still picks max
        assert_eq!(result, 2);
    }

    // =========================================================================
    // Softmax Tests
    // =========================================================================

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1.0");
    }

    #[test]
    fn test_softmax_all_same() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let probs = softmax(&logits);
        for p in &probs {
            assert!(
                (p - 0.25).abs() < 1e-5,
                "Equal logits should give uniform probs"
            );
        }
    }

    #[test]
    fn test_softmax_preserves_order() {
        let logits = vec![1.0, 3.0, 2.0];
        let probs = softmax(&logits);
        assert!(probs[1] > probs[2]);
        assert!(probs[2] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without max subtraction
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // =========================================================================
    // Token Suppression Tests
    // =========================================================================

    #[test]
    fn test_suppress_tokens() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        GreedyDecoder::suppress_tokens(&mut logits, &[1, 3]);
        assert!((logits[0] - 1.0).abs() < f32::EPSILON);
        assert!(logits[1] == f32::NEG_INFINITY);
        assert!((logits[2] - 3.0).abs() < f32::EPSILON);
        assert!(logits[3] == f32::NEG_INFINITY);
    }

    #[test]
    fn test_suppress_tokens_out_of_bounds() {
        let mut logits = vec![1.0, 2.0, 3.0];
        // Should not panic for out of bounds
        GreedyDecoder::suppress_tokens(&mut logits, &[10, 100]);
        assert!((logits[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_suppress_empty() {
        let mut logits = vec![1.0, 2.0, 3.0];
        GreedyDecoder::suppress_tokens(&mut logits, &[]);
        assert!((logits[0] - 1.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Bias Tests
    // =========================================================================

    #[test]
    fn test_apply_bias() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        GreedyDecoder::apply_bias(&mut logits, &[(0, 5.0), (2, -1.0)]);
        assert!((logits[0] - 6.0).abs() < f32::EPSILON);
        assert!((logits[1] - 2.0).abs() < f32::EPSILON);
        assert!((logits[2] - 2.0).abs() < f32::EPSILON);
        assert!((logits[3] - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_apply_bias_out_of_bounds() {
        let mut logits = vec![1.0, 2.0];
        GreedyDecoder::apply_bias(&mut logits, &[(10, 5.0)]);
        // Should not panic, values unchanged
        assert!((logits[0] - 1.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Decode Tests
    // =========================================================================

    #[test]
    fn test_decode_stops_at_eot() {
        use std::cell::Cell;

        let decoder = GreedyDecoder::new(100);
        let eot = special_tokens::EOT;

        // Mock logits function that returns EOT after 3 tokens
        let step = Cell::new(0);
        let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
            step.set(step.get() + 1);
            let mut logits = vec![0.0_f32; 51865];
            if step.get() >= 3 {
                logits[eot as usize] = 10.0; // Force EOT
            } else {
                logits[100] = 10.0; // Force token 100
            }
            Ok(logits)
        };

        let result = decoder
            .decode(logits_fn, &[special_tokens::SOT])
            .expect("decode should succeed");

        // Should have: SOT + 2 tokens + EOT = 4 tokens
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], special_tokens::SOT);
        assert_eq!(result[3], eot);
    }

    #[test]
    fn test_decode_respects_max_tokens() {
        let decoder = GreedyDecoder::new(5);

        // Mock logits function that never returns EOT
        let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
            let mut logits = vec![0.0_f32; 51865];
            logits[100] = 10.0; // Always pick token 100
            Ok(logits)
        };

        let result = decoder
            .decode(logits_fn, &[special_tokens::SOT])
            .expect("decode should succeed");

        // Should have: SOT + 5 tokens = 6 tokens (respects max_tokens)
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_decode_preserves_initial_tokens() {
        let decoder = GreedyDecoder::new(2);
        let eot = special_tokens::EOT;

        let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
            let mut logits = vec![0.0_f32; 51865];
            logits[eot as usize] = 10.0;
            Ok(logits)
        };

        let initial = vec![special_tokens::SOT, special_tokens::LANG_BASE];
        let result = decoder
            .decode(logits_fn, &initial)
            .expect("decode should succeed");

        assert_eq!(result[0], special_tokens::SOT);
        assert_eq!(result[1], special_tokens::LANG_BASE);
        assert_eq!(result[2], eot);
    }
}
