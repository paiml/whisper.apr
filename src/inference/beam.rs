//! Beam search decoding
//!
//! Higher quality decoding with configurable beam width.
//!
//! # Algorithm
//!
//! Beam search maintains K (beam_size) candidates at each step,
//! expanding each with top-K next tokens and keeping the best K overall.
//! This explores more of the search space than greedy decoding.
//!
//! # Example
//!
//! ```rust,ignore
//! use whisper_apr::inference::BeamSearchDecoder;
//!
//! let decoder = BeamSearchDecoder::new(5, 448);
//! let tokens = decoder.decode(&logits_fn, &initial_tokens);
//! ```

use crate::error::WhisperResult;
use crate::tokenizer::special_tokens;

/// A single hypothesis in beam search
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// Token sequence
    pub tokens: Vec<u32>,
    /// Log probability score
    pub score: f32,
    /// Whether this hypothesis is complete (ended with EOT)
    pub is_complete: bool,
}

impl Hypothesis {
    /// Create new hypothesis
    fn new(tokens: Vec<u32>, score: f32) -> Self {
        Self {
            tokens,
            score,
            is_complete: false,
        }
    }

    /// Length-normalized score for comparison
    fn normalized_score(&self, length_penalty: f32) -> f32 {
        // Score normalized by sequence length to avoid bias toward shorter sequences
        let len = self.tokens.len() as f32;
        self.score / len.powf(length_penalty)
    }
}

/// Beam search decoder for token generation
///
/// Explores multiple hypotheses in parallel for better results.
#[derive(Debug, Clone)]
pub struct BeamSearchDecoder {
    /// Number of beams
    beam_size: usize,
    /// Maximum tokens to generate
    max_tokens: usize,
    /// Temperature for logit scaling
    temperature: f32,
    /// Patience factor for early stopping
    patience: f32,
    /// Length penalty (alpha in length normalization)
    length_penalty: f32,
}

impl BeamSearchDecoder {
    /// Create a new beam search decoder
    ///
    /// # Arguments
    /// * `beam_size` - Number of beams (default: 5)
    /// * `max_tokens` - Maximum tokens to generate
    #[must_use]
    pub const fn new(beam_size: usize, max_tokens: usize) -> Self {
        Self {
            beam_size,
            max_tokens,
            temperature: 0.0,
            patience: 1.0,
            length_penalty: 1.0,
        }
    }

    /// Set temperature
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set patience factor for early stopping
    #[must_use]
    pub const fn with_patience(mut self, patience: f32) -> Self {
        self.patience = patience;
        self
    }

    /// Set length penalty (higher values favor longer sequences)
    #[must_use]
    pub const fn with_length_penalty(mut self, length_penalty: f32) -> Self {
        self.length_penalty = length_penalty;
        self
    }

    /// Get beam size
    #[must_use]
    pub const fn beam_size(&self) -> usize {
        self.beam_size
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

    /// Get patience factor
    #[must_use]
    pub const fn patience(&self) -> f32 {
        self.patience
    }

    /// Get length penalty
    #[must_use]
    pub const fn length_penalty(&self) -> f32 {
        self.length_penalty
    }

    /// Run beam search decoding
    ///
    /// # Arguments
    /// * `logits_fn` - Function that takes tokens and returns logits
    /// * `initial_tokens` - Initial tokens (e.g., [SOT, language, task])
    ///
    /// # Returns
    /// Best token sequence found
    pub fn decode<F>(&self, mut logits_fn: F, initial_tokens: &[u32]) -> WhisperResult<Vec<u32>>
    where
        F: FnMut(&[u32]) -> WhisperResult<Vec<f32>>,
    {
        let eot = special_tokens::EOT;

        // Initialize with single hypothesis
        let mut hypotheses = vec![Hypothesis::new(initial_tokens.to_vec(), 0.0)];
        let mut completed: Vec<Hypothesis> = Vec::new();

        // Loop until all hypotheses reach max_tokens (total length, not new tokens)
        loop {
            // Check if shortest hypothesis has reached max_tokens
            let min_len = hypotheses
                .iter()
                .map(|h| h.tokens.len())
                .min()
                .unwrap_or(self.max_tokens);
            if min_len >= self.max_tokens {
                break;
            }

            let mut all_candidates: Vec<Hypothesis> = Vec::new();

            // Expand each hypothesis
            for hyp in &hypotheses {
                if hyp.is_complete || hyp.tokens.len() >= self.max_tokens {
                    // Don't expand hypotheses at max length
                    continue;
                }

                // Get logits for this hypothesis
                let logits = logits_fn(&hyp.tokens)?;
                let log_probs = self.log_softmax(&logits);

                // Get top-K candidates for this hypothesis
                let top_k = Self::top_k_indices(&log_probs, self.beam_size);

                for (token, log_prob) in top_k {
                    let mut new_tokens = hyp.tokens.clone();
                    new_tokens.push(token);

                    let mut new_hyp = Hypothesis::new(new_tokens, hyp.score + log_prob);

                    if token == eot {
                        new_hyp.is_complete = true;
                        completed.push(new_hyp);
                    } else {
                        all_candidates.push(new_hyp);
                    }
                }
            }

            // Keep top beam_size hypotheses
            all_candidates.sort_by(|a, b| {
                b.normalized_score(self.length_penalty)
                    .partial_cmp(&a.normalized_score(self.length_penalty))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            hypotheses = all_candidates.into_iter().take(self.beam_size).collect();

            // Early stopping with patience
            if self.should_stop_early(&completed, &hypotheses) {
                break;
            }

            // All hypotheses completed
            if hypotheses.is_empty() {
                break;
            }
        }

        // Add remaining incomplete hypotheses to completed (if any)
        for hyp in hypotheses {
            if !hyp.is_complete {
                completed.push(hyp);
            }
        }

        // Return best hypothesis
        completed.sort_by(|a, b| {
            b.normalized_score(self.length_penalty)
                .partial_cmp(&a.normalized_score(self.length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        completed
            .into_iter()
            .next()
            .map(|h| h.tokens)
            .ok_or_else(|| crate::error::WhisperError::Inference("no valid hypothesis".into()))
    }

    /// Compute log softmax of logits
    fn log_softmax(&self, logits: &[f32]) -> Vec<f32> {
        let scaled: Vec<f32> = if self.temperature > 0.0 {
            logits.iter().map(|&x| x / self.temperature).collect()
        } else {
            logits.to_vec()
        };

        // Find max for numerical stability
        let max_val = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute log-sum-exp
        let log_sum_exp = scaled
            .iter()
            .map(|&x| (x - max_val).exp())
            .sum::<f32>()
            .ln()
            + max_val;

        // Log softmax = logit - log_sum_exp
        scaled.iter().map(|&x| x - log_sum_exp).collect()
    }

    /// Get top K indices by value
    fn top_k_indices(values: &[f32], k: usize) -> Vec<(u32, f32)> {
        // Create index-value pairs
        let mut indexed: Vec<(usize, f32)> =
            values.iter().enumerate().map(|(i, &v)| (i, v)).collect();

        // Partial sort to get top K
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed
            .into_iter()
            .take(k)
            .map(|(i, v)| (i as u32, v))
            .collect()
    }

    /// Check if we should stop early based on patience
    fn should_stop_early(&self, completed: &[Hypothesis], candidates: &[Hypothesis]) -> bool {
        if completed.is_empty() || candidates.is_empty() {
            return false;
        }

        // Get best completed score
        let best_completed = completed
            .iter()
            .map(|h| h.normalized_score(self.length_penalty))
            .fold(f32::NEG_INFINITY, f32::max);

        // Get worst candidate score (upper bound on what we could achieve)
        let worst_candidate = candidates
            .iter()
            .map(|h| h.normalized_score(self.length_penalty))
            .fold(f32::INFINITY, f32::min);

        // Stop if best completed is better than worst candidate by patience factor
        best_completed > worst_candidate * self.patience
    }

    /// Get N-best hypotheses from beam search
    pub fn decode_nbest<F>(
        &self,
        mut logits_fn: F,
        initial_tokens: &[u32],
        n: usize,
    ) -> WhisperResult<Vec<Vec<u32>>>
    where
        F: FnMut(&[u32]) -> WhisperResult<Vec<f32>>,
    {
        let eot = special_tokens::EOT;
        let mut hypotheses = vec![Hypothesis::new(initial_tokens.to_vec(), 0.0)];
        let mut completed: Vec<Hypothesis> = Vec::new();

        // Loop until all hypotheses reach max_tokens (total length, not new tokens)
        loop {
            // Check if shortest hypothesis has reached max_tokens
            let min_len = hypotheses
                .iter()
                .map(|h| h.tokens.len())
                .min()
                .unwrap_or(self.max_tokens);
            if min_len >= self.max_tokens {
                break;
            }

            let mut all_candidates: Vec<Hypothesis> = Vec::new();

            for hyp in &hypotheses {
                if hyp.is_complete || hyp.tokens.len() >= self.max_tokens {
                    continue;
                }

                let logits = logits_fn(&hyp.tokens)?;
                let log_probs = self.log_softmax(&logits);
                let top_k = Self::top_k_indices(&log_probs, self.beam_size);

                for (token, log_prob) in top_k {
                    let mut new_tokens = hyp.tokens.clone();
                    new_tokens.push(token);

                    let mut new_hyp = Hypothesis::new(new_tokens, hyp.score + log_prob);

                    if token == eot {
                        new_hyp.is_complete = true;
                        completed.push(new_hyp);
                    } else {
                        all_candidates.push(new_hyp);
                    }
                }
            }

            all_candidates.sort_by(|a, b| {
                b.normalized_score(self.length_penalty)
                    .partial_cmp(&a.normalized_score(self.length_penalty))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            hypotheses = all_candidates.into_iter().take(self.beam_size).collect();

            if hypotheses.is_empty() {
                break;
            }
        }

        // Add remaining incomplete hypotheses
        for hyp in hypotheses {
            if !hyp.is_complete {
                completed.push(hyp);
            }
        }

        // Sort and return top n
        completed.sort_by(|a, b| {
            b.normalized_score(self.length_penalty)
                .partial_cmp(&a.normalized_score(self.length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(completed.into_iter().take(n).map(|h| h.tokens).collect())
    }
}

impl Default for BeamSearchDecoder {
    fn default() -> Self {
        Self::new(5, 448)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Construction Tests
    // =========================================================================

    #[test]
    fn test_beam_search_decoder_new() {
        let decoder = BeamSearchDecoder::new(5, 448);
        assert_eq!(decoder.beam_size(), 5);
        assert_eq!(decoder.max_tokens(), 448);
        assert!((decoder.temperature() - 0.0).abs() < f32::EPSILON);
        assert!((decoder.patience() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_beam_search_decoder_builder() {
        let decoder = BeamSearchDecoder::new(3, 256)
            .with_temperature(0.5)
            .with_patience(1.5)
            .with_length_penalty(0.8);
        assert_eq!(decoder.beam_size(), 3);
        assert!((decoder.temperature() - 0.5).abs() < f32::EPSILON);
        assert!((decoder.patience() - 1.5).abs() < f32::EPSILON);
        assert!((decoder.length_penalty() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_beam_search_decoder_default() {
        let decoder = BeamSearchDecoder::default();
        assert_eq!(decoder.beam_size(), 5);
        assert_eq!(decoder.max_tokens(), 448);
    }

    // =========================================================================
    // Hypothesis Tests
    // =========================================================================

    #[test]
    fn test_hypothesis_new() {
        let hyp = Hypothesis::new(vec![1, 2, 3], -1.5);
        assert_eq!(hyp.tokens, vec![1, 2, 3]);
        assert!((hyp.score - (-1.5)).abs() < f32::EPSILON);
        assert!(!hyp.is_complete);
    }

    #[test]
    fn test_hypothesis_normalized_score() {
        let hyp = Hypothesis::new(vec![1, 2, 3, 4], -4.0);

        // With length_penalty = 1.0, normalized = score / len = -4.0 / 4 = -1.0
        let norm = hyp.normalized_score(1.0);
        assert!((norm - (-1.0)).abs() < 1e-5);

        // With length_penalty = 0.5, normalized = score / len^0.5 = -4.0 / 2 = -2.0
        let norm = hyp.normalized_score(0.5);
        assert!((norm - (-2.0)).abs() < 1e-5);
    }

    // =========================================================================
    // Log Softmax Tests
    // =========================================================================

    #[test]
    fn test_log_softmax_sums_correctly() {
        let decoder = BeamSearchDecoder::new(3, 10);
        let logits = vec![1.0, 2.0, 3.0];
        let log_probs = decoder.log_softmax(&logits);

        // exp(log_probs) should sum to 1
        let sum: f32 = log_probs.iter().map(|&x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_preserves_order() {
        let decoder = BeamSearchDecoder::new(3, 10);
        let logits = vec![1.0, 3.0, 2.0];
        let log_probs = decoder.log_softmax(&logits);

        assert!(log_probs[1] > log_probs[2]);
        assert!(log_probs[2] > log_probs[0]);
    }

    #[test]
    fn test_log_softmax_with_temperature() {
        let decoder = BeamSearchDecoder::new(3, 10).with_temperature(0.5);
        let logits = vec![1.0, 2.0];
        let log_probs = decoder.log_softmax(&logits);

        // Higher temperature should soften distribution
        let decoder_hot = BeamSearchDecoder::new(3, 10).with_temperature(2.0);
        let log_probs_hot = decoder_hot.log_softmax(&logits);

        // With lower temp, difference should be larger
        let diff_cold = log_probs[1] - log_probs[0];
        let diff_hot = log_probs_hot[1] - log_probs_hot[0];
        assert!(diff_cold.abs() > diff_hot.abs());
    }

    // =========================================================================
    // Top-K Tests
    // =========================================================================

    #[test]
    fn test_top_k_indices() {
        let values = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let top_k = BeamSearchDecoder::top_k_indices(&values, 3);

        assert_eq!(top_k.len(), 3);
        assert_eq!(top_k[0].0, 3); // index of 0.9
        assert_eq!(top_k[1].0, 1); // index of 0.5
        assert_eq!(top_k[2].0, 2); // index of 0.3
    }

    #[test]
    fn test_top_k_indices_k_larger_than_len() {
        let values = vec![0.1, 0.5];
        let top_k = BeamSearchDecoder::top_k_indices(&values, 5);

        assert_eq!(top_k.len(), 2);
    }

    #[test]
    fn test_top_k_indices_single() {
        let values = vec![0.1, 0.5, 0.3];
        let top_k = BeamSearchDecoder::top_k_indices(&values, 1);

        assert_eq!(top_k.len(), 1);
        assert_eq!(top_k[0].0, 1); // index of 0.5
    }

    // =========================================================================
    // Early Stopping Tests
    // =========================================================================

    #[test]
    fn test_should_stop_early_no_completed() {
        let decoder = BeamSearchDecoder::new(3, 10);
        let completed: Vec<Hypothesis> = vec![];
        let candidates = vec![Hypothesis::new(vec![1], -1.0)];

        assert!(!decoder.should_stop_early(&completed, &candidates));
    }

    #[test]
    fn test_should_stop_early_no_candidates() {
        let decoder = BeamSearchDecoder::new(3, 10);
        let completed = vec![Hypothesis::new(vec![1], -1.0)];
        let candidates: Vec<Hypothesis> = vec![];

        assert!(!decoder.should_stop_early(&completed, &candidates));
    }

    // =========================================================================
    // Decode Tests
    // =========================================================================

    #[test]
    fn test_beam_decode_stops_at_eot() {
        let decoder = BeamSearchDecoder::new(3, 100);
        let eot = special_tokens::EOT;

        // Always return EOT as highest probability - should stop immediately
        let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
            let mut logits = vec![-10.0_f32; 51865];
            logits[eot as usize] = 10.0;
            Ok(logits)
        };

        let result = decoder
            .decode(logits_fn, &[special_tokens::SOT])
            .expect("beam decode should succeed");

        // Result should contain EOT (might have intermediate tokens too)
        assert!(result.len() >= 2, "result too short: {:?}", result);
        assert!(result.contains(&eot), "should contain EOT: {:?}", result);
    }

    #[test]
    fn test_beam_decode_respects_max_tokens() {
        let decoder = BeamSearchDecoder::new(2, 3);

        let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
            let mut logits = vec![-10.0_f32; 51865];
            logits[100] = 10.0;
            Ok(logits)
        };

        let result = decoder
            .decode(logits_fn, &[special_tokens::SOT])
            .expect("beam decode should succeed");

        // Should stop at max_tokens
        assert!(result.len() <= 5); // initial + max_tokens
    }

    #[test]
    fn test_beam_decode_explores_multiple_paths() {
        let decoder = BeamSearchDecoder::new(3, 5);

        // Give different tokens similar scores to force beam to explore
        let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
            let mut logits = vec![-10.0_f32; 51865];
            // Multiple tokens with similar scores
            logits[100] = 1.0;
            logits[101] = 0.9;
            logits[102] = 0.8;
            logits[special_tokens::EOT as usize] = 0.5;
            Ok(logits)
        };

        let result = decoder
            .decode(logits_fn, &[special_tokens::SOT])
            .expect("beam decode should succeed");

        // Should return a valid sequence
        assert!(!result.is_empty());
    }

    // =========================================================================
    // N-Best Tests
    // =========================================================================

    #[test]
    fn test_decode_nbest() {
        let decoder = BeamSearchDecoder::new(3, 5);
        let eot = special_tokens::EOT;

        let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
            let mut logits = vec![-10.0_f32; 51865];
            logits[100] = 1.0;
            logits[101] = 0.5;
            logits[eot as usize] = 0.9;
            Ok(logits)
        };

        let results = decoder
            .decode_nbest(logits_fn, &[special_tokens::SOT], 2)
            .expect("decode_nbest should succeed");

        // Should return up to 2 results
        assert!(results.len() <= 2);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_decode_nbest_empty_initial() {
        let decoder = BeamSearchDecoder::new(2, 3);
        let eot = special_tokens::EOT;

        let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
            let mut logits = vec![-10.0_f32; 51865];
            logits[eot as usize] = 10.0;
            Ok(logits)
        };

        let results = decoder
            .decode_nbest(logits_fn, &[], 3)
            .expect("decode_nbest should succeed");
        assert!(!results.is_empty());
    }

    // =========================================================================
    // EXTREME TDD: Token Limit Invariant Tests
    // =========================================================================

    #[test]
    fn test_beam_decode_total_tokens_never_exceeds_max() {
        // BUG: beam search loop runs `for _ in 0..max_tokens` generating
        // up to max_tokens NEW tokens, ignoring initial_tokens length.
        // This violates the invariant: output.len() <= max_tokens
        let decoder = BeamSearchDecoder::new(2, 10); // beam=2, max=10

        // Mock logits that never returns EOT
        let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
            let mut logits = vec![0.0_f32; 51865];
            logits[100] = 10.0; // Always pick token 100
            logits[101] = 9.0; // Second choice
            Ok(logits)
        };

        // Start with 5 initial tokens
        let initial = vec![1, 2, 3, 4, 5];
        let result = decoder
            .decode(logits_fn, &initial)
            .expect("decode should succeed");

        // INVARIANT: total tokens must never exceed max_tokens
        assert!(
            result.len() <= decoder.max_tokens(),
            "beam search: total tokens {} exceeds max_tokens {}",
            result.len(),
            decoder.max_tokens()
        );
    }

    // =========================================================================
    // EXTREME TDD: O(n) Complexity Assertions
    // =========================================================================

    #[test]
    fn test_beam_decode_is_on_not_on2() {
        // Performance test: decoding N tokens should take O(N) time, not O(N²)
        // We measure by counting logits_fn calls - should be O(N), not O(N²)
        use std::sync::atomic::{AtomicUsize, Ordering};

        let call_count = AtomicUsize::new(0);

        let decoder = BeamSearchDecoder::new(1, 50); // beam=1 for predictable calls
        let eot = special_tokens::EOT;

        let logits_fn = |tokens: &[u32]| -> WhisperResult<Vec<f32>> {
            call_count.fetch_add(1, Ordering::SeqCst);
            let mut logits = vec![-10.0_f32; 51865];
            // Return EOT after 20 tokens
            if tokens.len() >= 20 {
                logits[eot as usize] = 10.0;
            } else {
                logits[100] = 10.0;
            }
            Ok(logits)
        };

        let result = decoder
            .decode(logits_fn, &[special_tokens::SOT])
            .expect("decode should succeed");

        let calls = call_count.load(Ordering::SeqCst);
        let tokens_generated = result.len() - 1; // minus initial SOT

        // O(n) means calls should be roughly equal to tokens_generated
        // O(n²) would mean calls ≈ tokens_generated² / 2
        // Allow 2x overhead for beam search bookkeeping
        let max_allowed_calls = tokens_generated * 3;

        assert!(
            calls <= max_allowed_calls,
            "O(n²) detected: {} calls for {} tokens (max allowed: {}). \
             Expected O(n) complexity.",
            calls,
            tokens_generated,
            max_allowed_calls
        );
    }

    // =========================================================================
    // EXTREME TDD: Property-Based Tests for Beam Search
    // =========================================================================

    #[test]
    fn property_beam_output_length_bounded_by_max_tokens() {
        // Property: For any initial_tokens and max_tokens,
        // output.len() <= max_tokens
        for max_tokens in [5, 10, 20, 50] {
            for initial_len in [0, 1, 3, max_tokens / 2, max_tokens - 1, max_tokens] {
                let decoder = BeamSearchDecoder::new(2, max_tokens);

                let logits_fn = |_: &[u32]| -> WhisperResult<Vec<f32>> {
                    let mut logits = vec![0.0_f32; 51865];
                    logits[100] = 10.0;
                    logits[101] = 9.0;
                    Ok(logits)
                };

                let initial: Vec<u32> = (0..initial_len).map(|i| i as u32).collect();
                let result = decoder
                    .decode(logits_fn, &initial)
                    .expect("decode should succeed");

                assert!(
                    result.len() <= max_tokens,
                    "Property violated: output.len()={} > max_tokens={} (initial_len={})",
                    result.len(),
                    max_tokens,
                    initial_len
                );
            }
        }
    }

    #[test]
    fn property_beam_initial_tokens_preserved() {
        // Property: output[0..initial.len()] == initial (prefix preserved)
        let decoder = BeamSearchDecoder::new(2, 100);
        let eot = special_tokens::EOT;

        for initial_len in [1, 3, 5, 10] {
            let logits_fn = |_: &[u32]| -> WhisperResult<Vec<f32>> {
                let mut logits = vec![0.0_f32; 51865];
                logits[eot as usize] = 10.0;
                Ok(logits)
            };

            let initial: Vec<u32> = (100..100 + initial_len).collect();
            let result = decoder
                .decode(logits_fn, &initial)
                .expect("decode should succeed");

            assert_eq!(
                &result[..initial.len()],
                &initial[..],
                "Property violated: initial tokens not preserved (initial_len={})",
                initial_len
            );
        }
    }

    #[test]
    fn property_beam_nbest_all_bounded() {
        // Property: All N-best results respect max_tokens limit
        let decoder = BeamSearchDecoder::new(3, 15);

        let logits_fn = |_: &[u32]| -> WhisperResult<Vec<f32>> {
            let mut logits = vec![0.0_f32; 51865];
            logits[100] = 10.0;
            logits[101] = 9.5;
            logits[102] = 9.0;
            Ok(logits)
        };

        let initial = vec![1, 2, 3, 4, 5]; // 5 initial tokens
        let results = decoder
            .decode_nbest(logits_fn, &initial, 3)
            .expect("decode_nbest should succeed");

        for (i, result) in results.iter().enumerate() {
            assert!(
                result.len() <= decoder.max_tokens(),
                "N-best[{}]: output.len()={} > max_tokens={}",
                i,
                result.len(),
                decoder.max_tokens()
            );
        }
    }
}
