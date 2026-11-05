//! OpenAI-style temperature fallback and no-speech skip (issue #107).
//!
//! Replaces the ad-hoc repetition penalties that used to run inside the
//! decode loop (a cumulative -2.0 per recent token, a -10 trigram block, a
//! forced EOT on repeated 4-grams and a unigram cap). Those rewrote every
//! window's logits, normal or not. OpenAI's `transcribe()` leaves the logits
//! alone and instead judges each finished decode:
//!
//! * a decode whose text compresses too well (`compression_ratio > 2.4`, a
//!   repetition loop) or whose tokens are too unlikely (`avg_logprob < -1.0`)
//!   is retried at the next temperature of `0.0, 0.2, ..., 1.0`;
//! * a window whose `<|nospeech|>` probability at the SOT position is above
//!   0.6 is never retried, and when its `avg_logprob` is also below -1.0 it
//!   emits no segments (silence or music, where the model invents a closing
//!   "Thank you so much for joining us today").
//!
//! Everything here is pure: the decode itself is passed in as a closure, so
//! the policy is tested without a model. Sampling is seeded from the window
//! index, so the same input gives the same bytes on every run.

use std::collections::HashMap;

use crate::error::WhisperResult;

/// Thresholds, temperatures and seed of the fallback (OpenAI defaults).
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeFallback {
    /// Temperatures tried in order; the first is the normal decode.
    pub temperatures: Vec<f32>,
    /// Retry when the text's gzip-style compression ratio is above this.
    pub compression_ratio_threshold: f32,
    /// Retry when the average log-probability is below this.
    pub logprob_threshold: f32,
    /// A window is silent when its no-speech probability is above this
    /// (and its average log-probability is below `logprob_threshold`).
    pub no_speech_threshold: f32,
    /// Base seed of the sampler; each window and temperature mixes it with
    /// its own index.
    pub seed: u64,
}

impl Default for DecodeFallback {
    fn default() -> Self {
        Self {
            temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold: 2.4,
            logprob_threshold: -1.0,
            no_speech_threshold: 0.6,
            seed: 0,
        }
    }
}

/// What the fallback policy reads from one finished decode.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DecodeStats {
    /// `len(text) / len(zlib(text))` of the decoded text, in bytes.
    pub compression_ratio: f32,
    /// Sum of the sampled tokens' log-probabilities over `n + 1`, where `n`
    /// is the number of sampled tokens before EOT (OpenAI's definition).
    pub avg_logprob: f32,
    /// Softmax probability of `<|nospeech|>` at the SOT position.
    pub no_speech_prob: f32,
}

/// One attempt's output and its stats.
#[derive(Debug, Clone, PartialEq)]
pub struct FallbackOutcome<T> {
    /// The kept attempt's output.
    pub output: T,
    /// The kept attempt's stats.
    pub stats: DecodeStats,
    /// The temperature the kept attempt ran at.
    pub temperature: f32,
    /// How many attempts ran (1 = no fallback).
    pub attempts: usize,
    /// True when the window is silence or music: it emits no segments.
    pub silent: bool,
}

impl<T: Default> FallbackOutcome<T> {
    /// The kept output, or the empty default when the window is silent.
    #[must_use]
    pub fn into_kept_output(self) -> T {
        if self.silent {
            T::default()
        } else {
            self.output
        }
    }
}

impl DecodeFallback {
    /// Whether the no-speech check marks the decode as silence. Such a
    /// decode is never retried (OpenAI: "silence").
    #[must_use]
    pub fn is_no_speech(&self, stats: &DecodeStats) -> bool {
        stats.no_speech_prob > self.no_speech_threshold
    }

    /// Whether a decode is retried at the next temperature.
    #[must_use]
    #[provable_contracts_macros::contract(
        "whisper-decode-fallback-v1",
        equation = "fallback_policy"
    )]
    pub fn needs_fallback(&self, stats: &DecodeStats) -> bool {
        if self.is_no_speech(stats) {
            return false;
        }
        stats.compression_ratio > self.compression_ratio_threshold
            || stats.avg_logprob < self.logprob_threshold
    }

    /// Whether the window emits no segments: no speech AND unlikely tokens.
    #[must_use]
    #[provable_contracts_macros::contract(
        "whisper-decode-fallback-v1",
        equation = "no_speech_skip"
    )]
    pub fn is_silent(&self, stats: &DecodeStats) -> bool {
        self.is_no_speech(stats) && stats.avg_logprob < self.logprob_threshold
    }

    /// The sampler seed of one attempt: the base seed mixed with the window
    /// index and the temperature index, so it depends on nothing else.
    #[must_use]
    #[provable_contracts_macros::contract("whisper-decode-fallback-v1", equation = "attempt_seed")]
    pub fn attempt_seed(&self, window_index: u64, temperature_index: usize) -> u64 {
        let mut rng = SplitMix64::new(self.seed ^ window_index.rotate_left(32));
        let mut seed = rng.next_u64();
        for _ in 0..temperature_index {
            seed = rng.next_u64();
        }
        seed
    }

    /// Run `attempt(temperature, seed)` at each temperature until a decode
    /// needs no fallback; keep the last attempt (OpenAI's
    /// `decode_with_fallback`).
    ///
    /// # Errors
    /// Returns the first error an attempt returns.
    pub fn run<T, F>(&self, window_index: u64, mut attempt: F) -> WhisperResult<FallbackOutcome<T>>
    where
        F: FnMut(f32, u64) -> WhisperResult<(T, DecodeStats)>,
    {
        let temperatures: &[f32] = if self.temperatures.is_empty() {
            &[0.0]
        } else {
            &self.temperatures
        };
        let mut kept = None;
        for (index, &temperature) in temperatures.iter().enumerate() {
            let (output, stats) = attempt(temperature, self.attempt_seed(window_index, index))?;
            let retry = self.needs_fallback(&stats);
            kept = Some(FallbackOutcome {
                output,
                stats,
                temperature,
                attempts: index + 1,
                silent: self.is_silent(&stats),
            });
            if !retry {
                break;
            }
        }
        kept.ok_or_else(|| crate::error::WhisperError::Inference("no decode attempt".into()))
    }
}

/// `len(text) / len(zlib.compress(text))` over the UTF-8 bytes, as OpenAI's
/// `compression_ratio` (zlib level 6, Python's default). Empty text is 0.
#[must_use]
pub fn compression_ratio(text: &str) -> f32 {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return 0.0;
    }
    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(bytes, 6);
    bytes.len() as f32 / compressed.len().max(1) as f32
}

/// Softmax probability of `no_speech_id` over raw `logits` (the decoder's
/// output at the SOT position, before any suppression).
#[must_use]
pub fn no_speech_prob(logits: &[f32], no_speech_id: u32) -> f32 {
    let Some(&target) = logits.get(no_speech_id as usize) else {
        return 0.0;
    };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() || !target.is_finite() {
        return 0.0;
    }
    let sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    (target - max).exp() / sum
}

/// SplitMix64: a tiny, portable, seedable generator for the sampler.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// A generator starting from `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform value in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Sample a token from `softmax(logits / temperature)` with `rng`.
/// Banned (`-inf`) tokens are never drawn.
pub fn sample_token(logits: &[f32], temperature: f32, rng: &mut SplitMix64) -> u32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return 0;
    }
    let t = f64::from(temperature.max(f32::MIN_POSITIVE));
    let weights: Vec<f64> = logits
        .iter()
        .map(|&x| (f64::from(x - max) / t).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let mut target = rng.next_f64() * total;
    let mut last_allowed = 0;
    for (index, &w) in weights.iter().enumerate() {
        if w > 0.0 {
            last_allowed = index;
            if target < w {
                return index as u32;
            }
            target -= w;
        }
    }
    last_allowed as u32
}

/// Log-probabilities of the tokens a decoder sampled, recorded from the
/// processed logits it was given.
///
/// Greedy and sampling decoders call the logits function once per token on
/// a context that grows by one, so each token's log-probability is read
/// exactly from the previous call's logits. Beam search calls it on several
/// contexts per step; it only ever keeps a token from its context's top
/// `top_k`, which the ledger records per context.
#[derive(Debug, Clone)]
pub struct LogprobLedger {
    top_k: usize,
    previous: Option<(Vec<u32>, Vec<f32>)>,
    exact: HashMap<u64, f32>,
    top: HashMap<u64, Vec<(u32, f32)>>,
}

impl LogprobLedger {
    /// A ledger keeping each context's `top_k` log-probabilities.
    #[must_use]
    pub fn new(top_k: usize) -> Self {
        Self {
            top_k: top_k.max(1),
            previous: None,
            exact: HashMap::new(),
            top: HashMap::new(),
        }
    }

    /// Record the processed `logits` the decoder sees for `context`.
    pub fn observe(&mut self, context: &[u32], logits: &[f32]) {
        if let Some((prev_context, prev_logprobs)) = self.previous.take() {
            if let Some((&last, head)) = context.split_last() {
                if head == prev_context.as_slice() {
                    if let Some(&lp) = prev_logprobs.get(last as usize) {
                        self.exact.insert(hash_tokens(context), lp);
                    }
                }
            }
        }
        let logprobs = log_softmax(logits);
        let mut ranked: Vec<(u32, f32)> = logprobs
            .iter()
            .enumerate()
            .map(|(i, &lp)| (i as u32, lp))
            .collect();
        let k = self.top_k.min(ranked.len());
        if k > 0 && k < ranked.len() {
            ranked.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            ranked.truncate(k);
        }
        self.top.insert(hash_tokens(context), ranked);
        self.previous = Some((context.to_vec(), logprobs));
    }

    /// The log-probability of `tokens[index]` given `tokens[..index]`, or
    /// `None` when the decoder never scored that context.
    fn logprob_at(&self, tokens: &[u32], index: usize) -> Option<f32> {
        let token = *tokens.get(index)?;
        if let Some(&lp) = self.exact.get(&hash_tokens(&tokens[..=index])) {
            return Some(lp);
        }
        if let Some((prev_context, prev_logprobs)) = &self.previous {
            if prev_context.as_slice() == &tokens[..index] {
                return prev_logprobs.get(token as usize).copied();
            }
        }
        self.top
            .get(&hash_tokens(&tokens[..index]))?
            .iter()
            .find(|(t, _)| *t == token)
            .map(|&(_, lp)| lp)
    }

    /// OpenAI's `avg_logprob` of a finished decode: the sum of the sampled
    /// tokens' log-probabilities (EOT's included) over `n + 1`, where `n`
    /// counts the sampled tokens before EOT. A token the ledger never saw
    /// scores `-inf`, so it can only push the decode towards a retry.
    #[must_use]
    pub fn avg_logprob(&self, tokens: &[u32], sample_begin: usize, eot: u32) -> f32 {
        let sampled = tokens.get(sample_begin..).unwrap_or(&[]);
        let n_text = sampled
            .iter()
            .position(|&t| t == eot)
            .unwrap_or(sampled.len());
        let n_scored = (n_text + 1).min(sampled.len());
        let sum: f32 = (sample_begin..sample_begin + n_scored)
            .map(|i| self.logprob_at(tokens, i).unwrap_or(f32::NEG_INFINITY))
            .sum();
        sum / (n_text + 1) as f32
    }
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![f32::NEG_INFINITY; logits.len()];
    }
    let lse = logits.iter().map(|&x| (x - max).exp()).sum::<f32>().ln() + max;
    logits.iter().map(|&x| x - lse).collect()
}

fn hash_tokens(tokens: &[u32]) -> u64 {
    // FNV-1a over the token ids: deterministic across runs and platforms.
    tokens.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &t| {
        t.to_le_bytes().iter().fold(h, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
        })
    }) ^ tokens.len() as u64
}

#[cfg(test)]
#[path = "fallback_tests.rs"]
mod tests;
