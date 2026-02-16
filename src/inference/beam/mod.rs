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

#[cfg(test)]
mod tests;

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
    /// * `eot_token` - End-of-transcription token ID (model-specific)
    ///
    /// # Returns
    /// Best token sequence found
    pub fn decode<F>(
        &self,
        mut logits_fn: F,
        initial_tokens: &[u32],
        eot_token: u32,
    ) -> WhisperResult<Vec<u32>>
    where
        F: FnMut(&[u32]) -> WhisperResult<Vec<f32>>,
    {
        let eot = eot_token;

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
    ///
    /// Numerically stable: handles degenerate cases where all logits are
    /// suppressed to `-inf` (e.g., after token suppression) by returning
    /// uniform log-probabilities.
    pub(crate) fn log_softmax(&self, logits: &[f32]) -> Vec<f32> {
        let scaled: Vec<f32> = if self.temperature > 0.0 {
            logits.iter().map(|&x| x / self.temperature).collect()
        } else {
            logits.to_vec()
        };

        // Find max for numerical stability
        let max_val = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Degenerate case: all logits are -inf or NaN (e.g., all tokens suppressed)
        // Return uniform distribution to avoid NaN propagation
        if !max_val.is_finite() {
            let uniform = -(logits.len() as f32).ln();
            return vec![uniform; logits.len()];
        }

        // Compute log-sum-exp
        let log_sum_exp = scaled
            .iter()
            .map(|&x| (x - max_val).exp())
            .sum::<f32>()
            .ln()
            + max_val;

        // Log softmax = logit - log_sum_exp
        let log_probs: Vec<f32> = scaled.iter().map(|&x| x - log_sum_exp).collect();

        debug_assert_eq!(
            log_probs.len(),
            logits.len(),
            "log_softmax output must match input length"
        );
        debug_assert!(
            log_probs.iter().all(|x| !x.is_nan() && *x != f32::INFINITY),
            "log probabilities must not be NaN or +inf"
        );

        log_probs
    }

    /// Get top K indices by value
    pub(crate) fn top_k_indices(values: &[f32], k: usize) -> Vec<(u32, f32)> {
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
    pub(crate) fn should_stop_early(
        &self,
        completed: &[Hypothesis],
        candidates: &[Hypothesis],
    ) -> bool {
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
        eot_token: u32,
        n: usize,
    ) -> WhisperResult<Vec<Vec<u32>>>
    where
        F: FnMut(&[u32]) -> WhisperResult<Vec<f32>>,
    {
        let eot = eot_token;
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
