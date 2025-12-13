//! Transformer decoder
//!
//! Implements the Whisper text decoder with causal self-attention
//! and cross-attention to encoder outputs.
//!
//! # Architecture
//!
//! 1. Token embedding + positional embedding
//! 2. N decoder blocks:
//!    - Masked self-attention (causal)
//!    - Cross-attention to encoder output
//!    - Feed-forward network
//! 3. Final layer norm
//!
//! # KV Cache
//!
//! For efficient autoregressive generation, the decoder supports KV caching.
//! During incremental decoding, only the new token is processed and the
//! key/value tensors from previous positions are reused.
//!
//! # References
//!
//! - Radford et al. (2023): "Robust Speech Recognition via Large-Scale Weak Supervision"

use super::encoder::{FeedForward, LayerNorm};
use super::{ModelConfig, MultiHeadAttention};
use crate::error::{WhisperError, WhisperResult};

// ============================================================================
// KV Cache
// ============================================================================

/// Key-Value cache for a single attention layer
///
/// Stores the computed key and value tensors to avoid recomputation
/// during autoregressive generation.
#[derive(Debug, Clone)]
pub struct LayerKVCache {
    /// Cached key tensor (seq_len x d_model)
    pub key: Vec<f32>,
    /// Cached value tensor (seq_len x d_model)
    pub value: Vec<f32>,
    /// Current sequence length in cache
    pub seq_len: usize,
    /// Model dimension
    pub d_model: usize,
    /// Maximum cache capacity
    pub max_len: usize,
}

impl LayerKVCache {
    /// Create a new empty KV cache for a layer
    #[must_use]
    pub fn new(d_model: usize, max_len: usize) -> Self {
        Self {
            key: Vec::with_capacity(max_len * d_model),
            value: Vec::with_capacity(max_len * d_model),
            seq_len: 0,
            d_model,
            max_len,
        }
    }

    /// Create a new KV cache with pre-allocated memory (WASM optimization)
    ///
    /// Unlike `new()`, this allocates the full buffer upfront which is more
    /// efficient in WASM where memory growth is expensive. The buffers are
    /// zero-initialized and ready for use.
    #[must_use]
    pub fn new_preallocated(d_model: usize, max_len: usize) -> Self {
        let capacity = max_len * d_model;
        let mut key = vec![0.0_f32; capacity];
        let mut value = vec![0.0_f32; capacity];
        // Truncate to 0 length but keep capacity
        key.truncate(0);
        value.truncate(0);
        // Restore capacity
        key.reserve(capacity);
        value.reserve(capacity);

        Self {
            key,
            value,
            seq_len: 0,
            d_model,
            max_len,
        }
    }

    /// Get remaining capacity in tokens
    #[must_use]
    pub fn remaining_capacity(&self) -> usize {
        self.max_len.saturating_sub(self.seq_len)
    }

    /// Check if cache is at capacity
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.seq_len >= self.max_len
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Get current cache length
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Append new key/value to cache
    ///
    /// # Arguments
    /// * `new_key` - New key tensor (new_len x d_model)
    /// * `new_value` - New value tensor (new_len x d_model)
    pub fn append(&mut self, new_key: &[f32], new_value: &[f32]) -> WhisperResult<()> {
        let new_len = new_key.len() / self.d_model;

        if new_key.len() != new_value.len() {
            return Err(WhisperError::Model(
                "key and value must have same size".into(),
            ));
        }
        if new_key.len() % self.d_model != 0 {
            return Err(WhisperError::Model(
                "key size not divisible by d_model".into(),
            ));
        }
        if self.seq_len + new_len > self.max_len {
            return Err(WhisperError::Model(format!(
                "cache overflow: {} + {} > {}",
                self.seq_len, new_len, self.max_len
            )));
        }

        self.key.extend_from_slice(new_key);
        self.value.extend_from_slice(new_value);
        self.seq_len += new_len;

        Ok(())
    }

    /// Get full cached key tensor
    #[must_use]
    pub fn get_key(&self) -> &[f32] {
        &self.key
    }

    /// Get full cached value tensor
    #[must_use]
    pub fn get_value(&self) -> &[f32] {
        &self.value
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.key.clear();
        self.value.clear();
        self.seq_len = 0;
    }

    /// Clear cache but keep allocated memory (WASM optimization)
    ///
    /// This is more efficient than `clear()` in WASM environments where
    /// you plan to reuse the cache for another sequence. Uses `truncate(0)`
    /// which is safe and preserves capacity.
    pub fn reset(&mut self) {
        // truncate(0) is safe and keeps the allocated capacity
        self.key.truncate(0);
        self.value.truncate(0);
        self.seq_len = 0;
    }

    /// Append a batch of key/value pairs efficiently
    ///
    /// This is more efficient than multiple single appends for batch processing.
    ///
    /// # Arguments
    /// * `keys` - Batch of key tensors (batch_size x d_model)
    /// * `values` - Batch of value tensors (batch_size x d_model)
    /// * `batch_size` - Number of positions in the batch
    pub fn append_batch(
        &mut self,
        keys: &[f32],
        values: &[f32],
        batch_size: usize,
    ) -> WhisperResult<()> {
        let expected_len = batch_size * self.d_model;

        if keys.len() != expected_len || values.len() != expected_len {
            return Err(WhisperError::Model(format!(
                "batch size mismatch: expected {} elements, got keys={}, values={}",
                expected_len,
                keys.len(),
                values.len()
            )));
        }

        if self.seq_len + batch_size > self.max_len {
            return Err(WhisperError::Model(format!(
                "cache overflow: {} + {} > {}",
                self.seq_len, batch_size, self.max_len
            )));
        }

        // extend_from_slice is optimized by the compiler for SIMD when possible
        self.key.extend_from_slice(keys);
        self.value.extend_from_slice(values);

        self.seq_len += batch_size;
        Ok(())
    }

    /// Get key slice for a specific position range
    #[must_use]
    pub fn get_key_range(&self, start: usize, end: usize) -> Option<&[f32]> {
        if end > self.seq_len || start > end {
            return None;
        }
        let start_idx = start * self.d_model;
        let end_idx = end * self.d_model;
        Some(&self.key[start_idx..end_idx])
    }

    /// Get value slice for a specific position range
    #[must_use]
    pub fn get_value_range(&self, start: usize, end: usize) -> Option<&[f32]> {
        if end > self.seq_len || start > end {
            return None;
        }
        let start_idx = start * self.d_model;
        let end_idx = end * self.d_model;
        Some(&self.value[start_idx..end_idx])
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        (self.key.len() + self.value.len()) * core::mem::size_of::<f32>()
    }

    /// Get allocated capacity in bytes
    #[must_use]
    pub fn capacity_bytes(&self) -> usize {
        (self.key.capacity() + self.value.capacity()) * core::mem::size_of::<f32>()
    }
}

/// KV cache for the entire decoder
///
/// Contains caches for both self-attention and cross-attention
/// across all decoder layers.
#[derive(Debug, Clone)]
pub struct DecoderKVCache {
    /// Self-attention KV caches (one per layer)
    pub self_attn_cache: Vec<LayerKVCache>,
    /// Cross-attention KV caches (one per layer)
    /// Note: Cross-attention K/V only needs to be computed once per encoder output
    pub cross_attn_cache: Vec<LayerKVCache>,
    /// Number of layers
    pub n_layers: usize,
    /// Model dimension
    pub d_model: usize,
    /// Maximum sequence length
    pub max_len: usize,
    /// Whether cross-attention cache is populated
    pub cross_attn_cached: bool,
}

impl DecoderKVCache {
    /// Create a new decoder KV cache
    #[must_use]
    pub fn new(n_layers: usize, d_model: usize, max_len: usize) -> Self {
        let self_attn_cache = (0..n_layers)
            .map(|_| LayerKVCache::new(d_model, max_len))
            .collect();
        let cross_attn_cache = (0..n_layers)
            .map(|_| LayerKVCache::new(d_model, max_len * 4)) // Cross-attention can be longer
            .collect();

        Self {
            self_attn_cache,
            cross_attn_cache,
            n_layers,
            d_model,
            max_len,
            cross_attn_cached: false,
        }
    }

    /// Get current sequence length (from self-attention cache)
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.self_attn_cache.first().map_or(0, LayerKVCache::len)
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq_len() == 0
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        for cache in &mut self.self_attn_cache {
            cache.clear();
        }
        for cache in &mut self.cross_attn_cache {
            cache.clear();
        }
        self.cross_attn_cached = false;
    }

    /// Clear only self-attention cache (keep cross-attention for same audio)
    pub fn clear_self_attn(&mut self) {
        for cache in &mut self.self_attn_cache {
            cache.clear();
        }
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let self_attn: usize = self
            .self_attn_cache
            .iter()
            .map(|c| (c.key.len() + c.value.len()) * 4)
            .sum();
        let cross_attn: usize = self
            .cross_attn_cache
            .iter()
            .map(|c| (c.key.len() + c.value.len()) * 4)
            .sum();
        self_attn + cross_attn
    }
}

// ============================================================================
// Streaming KV Cache (WAPR-111)
// ============================================================================

/// Streaming KV cache optimized for low-latency inference
///
/// This cache variant supports:
/// - Sliding window operation for bounded memory
/// - Efficient warm-up from previous chunk context
/// - Quick reset without deallocation
/// - Memory-bounded operation for long streaming sessions
#[derive(Debug, Clone)]
pub struct StreamingKVCache {
    /// Inner decoder cache
    inner: DecoderKVCache,
    /// Maximum sliding window size (in tokens)
    window_size: usize,
    /// Context overlap (tokens to keep when sliding)
    context_overlap: usize,
    /// Total tokens processed (may exceed window_size)
    total_tokens: usize,
    /// Number of times the window has slid
    slide_count: usize,
}

impl StreamingKVCache {
    /// Create a new streaming KV cache
    ///
    /// # Arguments
    /// * `n_layers` - Number of transformer layers
    /// * `d_model` - Model dimension
    /// * `window_size` - Maximum tokens in cache before sliding
    /// * `context_overlap` - Tokens to keep when sliding (for context)
    #[must_use]
    pub fn new(n_layers: usize, d_model: usize, window_size: usize, context_overlap: usize) -> Self {
        Self {
            inner: DecoderKVCache::new(n_layers, d_model, window_size),
            window_size,
            context_overlap: context_overlap.min(window_size / 2), // Max 50% overlap
            total_tokens: 0,
            slide_count: 0,
        }
    }

    /// Create with low-latency settings (smaller window, less overlap)
    ///
    /// Optimized for 500ms chunk processing:
    /// - Window: 64 tokens (~2 seconds of output)
    /// - Overlap: 16 tokens (~500ms of context)
    #[must_use]
    pub fn low_latency(n_layers: usize, d_model: usize) -> Self {
        Self::new(n_layers, d_model, 64, 16)
    }

    /// Create with ultra-low latency settings
    ///
    /// Optimized for 250ms chunk processing:
    /// - Window: 32 tokens (~1 second of output)
    /// - Overlap: 8 tokens (~250ms of context)
    #[must_use]
    pub fn ultra_low_latency(n_layers: usize, d_model: usize) -> Self {
        Self::new(n_layers, d_model, 32, 8)
    }

    /// Create with standard settings (larger window for accuracy)
    ///
    /// Optimized for standard 30s chunk processing:
    /// - Window: 448 tokens (full context)
    /// - Overlap: 64 tokens
    #[must_use]
    pub fn standard(n_layers: usize, d_model: usize) -> Self {
        Self::new(n_layers, d_model, 448, 64)
    }

    /// Get the current sequence length in cache
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.inner.seq_len()
    }

    /// Get total tokens processed (including those that have slid out)
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Get number of times the window has slid
    #[must_use]
    pub fn slide_count(&self) -> usize {
        self.slide_count
    }

    /// Get remaining capacity before sliding is needed
    #[must_use]
    pub fn remaining_capacity(&self) -> usize {
        self.window_size.saturating_sub(self.seq_len())
    }

    /// Check if cache will need to slide on next append
    #[must_use]
    pub fn will_slide(&self) -> bool {
        self.seq_len() >= self.window_size
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the window size
    #[must_use]
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get the context overlap
    #[must_use]
    pub fn context_overlap(&self) -> usize {
        self.context_overlap
    }

    /// Get a reference to the inner cache for reading K/V
    #[must_use]
    pub fn inner(&self) -> &DecoderKVCache {
        &self.inner
    }

    /// Get a mutable reference to the inner cache
    pub fn inner_mut(&mut self) -> &mut DecoderKVCache {
        &mut self.inner
    }

    /// Append key/value to a specific layer with automatic sliding
    ///
    /// If the cache is full, this will slide the window by removing
    /// old entries and keeping `context_overlap` tokens for context.
    pub fn append_with_slide(
        &mut self,
        layer_idx: usize,
        key: &[f32],
        value: &[f32],
    ) -> WhisperResult<()> {
        let new_len = key.len() / self.inner.d_model;

        // Check if we need to slide
        if self.seq_len() + new_len > self.window_size {
            self.slide_window()?;
        }

        // Append to cache
        self.inner.self_attn_cache[layer_idx].append(key, value)?;
        self.total_tokens += new_len;

        Ok(())
    }

    /// Slide the window, keeping only the context overlap
    pub fn slide_window(&mut self) -> WhisperResult<()> {
        let keep_from = self.seq_len().saturating_sub(self.context_overlap);

        for cache in &mut self.inner.self_attn_cache {
            if let (Some(k_range), Some(v_range)) = (
                cache.get_key_range(keep_from, cache.len()),
                cache.get_value_range(keep_from, cache.len()),
            ) {
                let new_keys = k_range.to_vec();
                let new_values = v_range.to_vec();

                cache.reset();
                cache.key.extend_from_slice(&new_keys);
                cache.value.extend_from_slice(&new_values);
                cache.seq_len = self.context_overlap;
            }
        }

        self.slide_count += 1;
        Ok(())
    }

    /// Reset the cache for a new streaming segment
    ///
    /// This clears all data but preserves allocated memory for efficiency.
    pub fn reset(&mut self) {
        for cache in &mut self.inner.self_attn_cache {
            cache.reset();
        }
        for cache in &mut self.inner.cross_attn_cache {
            cache.reset();
        }
        self.inner.cross_attn_cached = false;
        // Keep total_tokens and slide_count for statistics
    }

    /// Full reset including statistics
    pub fn full_reset(&mut self) {
        self.reset();
        self.total_tokens = 0;
        self.slide_count = 0;
    }

    /// Warm up the cache with context from a previous chunk
    ///
    /// This pre-fills the cache with key/value tensors from the end
    /// of a previous transcription, providing context continuity.
    pub fn warm_up(&mut self, layer_idx: usize, keys: &[f32], values: &[f32]) -> WhisperResult<()> {
        if layer_idx >= self.inner.n_layers {
            return Err(WhisperError::Model(format!(
                "layer index {} out of bounds (max {})",
                layer_idx,
                self.inner.n_layers
            )));
        }

        let n_tokens = keys.len() / self.inner.d_model;
        let tokens_to_use = n_tokens.min(self.context_overlap);

        if tokens_to_use > 0 {
            let start_idx = (n_tokens - tokens_to_use) * self.inner.d_model;
            self.inner.self_attn_cache[layer_idx].append(
                &keys[start_idx..],
                &values[start_idx..],
            )?;
        }

        Ok(())
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Get statistics about the streaming cache
    #[must_use]
    pub fn stats(&self) -> StreamingCacheStats {
        StreamingCacheStats {
            seq_len: self.seq_len(),
            total_tokens: self.total_tokens,
            slide_count: self.slide_count,
            window_size: self.window_size,
            context_overlap: self.context_overlap,
            memory_bytes: self.memory_bytes(),
        }
    }
}

/// Statistics about a streaming KV cache
#[derive(Debug, Clone)]
pub struct StreamingCacheStats {
    /// Current sequence length in cache
    pub seq_len: usize,
    /// Total tokens processed
    pub total_tokens: usize,
    /// Number of window slides
    pub slide_count: usize,
    /// Window size
    pub window_size: usize,
    /// Context overlap
    pub context_overlap: usize,
    /// Memory usage in bytes
    pub memory_bytes: usize,
}

impl StreamingCacheStats {
    /// Get cache utilization (0.0 to 1.0)
    #[must_use]
    pub fn utilization(&self) -> f32 {
        if self.window_size == 0 {
            0.0
        } else {
            self.seq_len as f32 / self.window_size as f32
        }
    }

    /// Get average tokens per slide
    #[must_use]
    pub fn tokens_per_slide(&self) -> f32 {
        if self.slide_count == 0 {
            self.total_tokens as f32
        } else {
            self.total_tokens as f32 / self.slide_count as f32
        }
    }
}

// ============================================================================
// Batch KV Cache (WAPR-082)
// ============================================================================

/// Batch of KV caches for parallel decoding
///
/// Each batch item has its own independent KV cache for self-attention,
/// allowing parallel decoding of multiple sequences.
#[derive(Debug, Clone)]
pub struct BatchDecoderCache {
    /// Individual caches for each batch item
    caches: Vec<DecoderKVCache>,
    /// Number of layers
    pub n_layers: usize,
    /// Model dimension
    pub d_model: usize,
    /// Maximum sequence length
    pub max_len: usize,
}

impl BatchDecoderCache {
    /// Create a new batch of KV caches
    #[must_use]
    pub fn new(batch_size: usize, n_layers: usize, d_model: usize, max_len: usize) -> Self {
        let caches = (0..batch_size)
            .map(|_| DecoderKVCache::new(n_layers, d_model, max_len))
            .collect();

        Self {
            caches,
            n_layers,
            d_model,
            max_len,
        }
    }

    /// Get the batch size
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.caches.len()
    }

    /// Get a reference to a specific cache
    #[must_use]
    pub fn get_cache(&self, index: usize) -> Option<&DecoderKVCache> {
        self.caches.get(index)
    }

    /// Get a mutable reference to a specific cache
    pub fn get_cache_mut(&mut self, index: usize) -> Option<&mut DecoderKVCache> {
        self.caches.get_mut(index)
    }

    /// Check if all caches are empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.caches.iter().all(DecoderKVCache::is_empty)
    }

    /// Clear all caches
    pub fn clear_all(&mut self) {
        for cache in &mut self.caches {
            cache.clear();
        }
    }

    /// Get sequence lengths for all batch items
    #[must_use]
    pub fn seq_lengths(&self) -> Vec<usize> {
        self.caches.iter().map(DecoderKVCache::seq_len).collect()
    }

    /// Get maximum sequence length across all batch items
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.caches.iter().map(DecoderKVCache::seq_len).max().unwrap_or(0)
    }

    /// Get total memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.caches.iter().map(DecoderKVCache::memory_bytes).sum()
    }
}

/// Output from batch decoder forward pass
#[derive(Debug, Clone)]
pub struct BatchDecoderOutput {
    /// Logits for each batch item (batch_size × seq_len × n_vocab or batch_size × n_vocab)
    pub logits: Vec<Vec<f32>>,
    /// Sequence lengths for each batch item
    pub seq_lengths: Vec<usize>,
}

impl BatchDecoderOutput {
    /// Get batch size
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.logits.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.logits.is_empty()
    }

    /// Get logits for a specific batch item
    #[must_use]
    pub fn get_logits(&self, index: usize) -> Option<&Vec<f32>> {
        self.logits.get(index)
    }
}

/// Single transformer decoder block
///
/// Contains masked self-attention, cross-attention to encoder, and FFN.
#[derive(Debug, Clone)]
pub struct DecoderBlock {
    /// Masked self-attention layer
    pub self_attn: MultiHeadAttention,
    /// Layer norm before self-attention
    pub ln1: LayerNorm,
    /// Cross-attention layer (to encoder output)
    pub cross_attn: MultiHeadAttention,
    /// Layer norm before cross-attention
    pub ln2: LayerNorm,
    /// Feed-forward network
    pub ffn: FeedForward,
    /// Layer norm before FFN
    pub ln3: LayerNorm,
}

impl DecoderBlock {
    /// Create new decoder block
    #[must_use]
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(n_heads, d_model),
            ln1: LayerNorm::new(d_model),
            cross_attn: MultiHeadAttention::new(n_heads, d_model),
            ln2: LayerNorm::new(d_model),
            ffn: FeedForward::new(d_model, d_ff),
            ln3: LayerNorm::new(d_model),
        }
    }

    /// Forward pass through decoder block
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len x d_model)
    /// * `encoder_output` - Encoder hidden states (enc_len x d_model)
    /// * `causal_mask` - Causal attention mask for self-attention
    ///
    /// # Returns
    /// Output tensor (seq_len x d_model)
    pub fn forward(
        &self,
        x: &[f32],
        encoder_output: &[f32],
        causal_mask: Option<&[f32]>,
    ) -> WhisperResult<Vec<f32>> {
        // Pre-norm masked self-attention with residual
        let normed = self.ln1.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, causal_mask)?;
        let mut residual: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

        // Pre-norm cross-attention with residual
        let normed = self.ln2.forward(&residual)?;
        let cross_out = self
            .cross_attn
            .forward_cross(&normed, encoder_output, None)?;
        for (r, c) in residual.iter_mut().zip(cross_out.iter()) {
            *r += c;
        }

        // Pre-norm FFN with residual
        let normed = self.ln3.forward(&residual)?;
        let ffn_out = self.ffn.forward(&normed)?;

        for (r, f) in residual.iter_mut().zip(ffn_out.iter()) {
            *r += f;
        }

        Ok(residual)
    }
}

/// Transformer decoder for text generation
///
/// Implements autoregressive text generation from encoder features.
#[derive(Debug)]
pub struct Decoder {
    /// Number of layers
    n_layers: usize,
    /// Hidden state dimension
    d_model: usize,
    /// Number of attention heads
    n_heads: usize,
    /// Decoder blocks
    blocks: Vec<DecoderBlock>,
    /// Final layer norm
    ln_post: LayerNorm,
    /// Token embeddings (n_vocab x d_model)
    token_embedding: Vec<f32>,
    /// Positional embeddings (max_len x d_model)
    positional_embedding: Vec<f32>,
    /// Vocabulary size
    n_vocab: usize,
    /// Maximum sequence length
    max_len: usize,
}

impl Decoder {
    /// Create a new decoder from model configuration
    #[must_use]
    pub fn new(config: &ModelConfig) -> Self {
        let n_layers = config.n_text_layer as usize;
        let d_model = config.n_text_state as usize;
        let n_heads = config.n_text_head as usize;
        let d_ff = d_model * 4;
        let n_vocab = config.n_vocab as usize;
        let max_len = config.n_text_ctx as usize;

        // Create decoder blocks
        let blocks: Vec<DecoderBlock> = (0..n_layers)
            .map(|_| DecoderBlock::new(d_model, n_heads, d_ff))
            .collect();

        // Create learned positional embeddings (initialized to zeros, will be loaded)
        let positional_embedding = vec![0.0_f32; max_len * d_model];

        // Create token embeddings (initialized to zeros, will be loaded)
        let token_embedding = vec![0.0_f32; n_vocab * d_model];

        Self {
            n_layers,
            d_model,
            n_heads,
            blocks,
            ln_post: LayerNorm::new(d_model),
            token_embedding,
            positional_embedding,
            n_vocab,
            max_len,
        }
    }

    /// Forward pass through decoder
    ///
    /// # Arguments
    /// * `tokens` - Token IDs (seq_len)
    /// * `encoder_output` - Encoder hidden states (enc_len x d_model)
    ///
    /// # Returns
    /// Logits over vocabulary (seq_len x n_vocab)
    ///
    /// # Errors
    /// Returns error if sequence too long or invalid tokens
    pub fn forward(&self, tokens: &[u32], encoder_output: &[f32]) -> WhisperResult<Vec<f32>> {
        let seq_len = tokens.len();

        if seq_len == 0 {
            return Err(WhisperError::Model("empty token sequence".into()));
        }
        if seq_len > self.max_len {
            return Err(WhisperError::Model(format!(
                "sequence length {} exceeds max {}",
                seq_len, self.max_len
            )));
        }

        // Validate encoder output size
        if encoder_output.len() % self.d_model != 0 {
            return Err(WhisperError::Model("encoder output size mismatch".into()));
        }

        // Embed tokens and add positional embeddings
        let mut x = self.embed_tokens(tokens)?;

        // Add positional embeddings
        for pos in 0..seq_len {
            for d in 0..self.d_model {
                x[pos * self.d_model + d] += self.positional_embedding[pos * self.d_model + d];
            }
        }

        // Create causal mask
        let causal_mask = MultiHeadAttention::causal_mask(seq_len);

        // Pass through decoder blocks
        for block in &self.blocks {
            x = block.forward(&x, encoder_output, Some(&causal_mask))?;
        }

        // Final layer norm
        let x = self.ln_post.forward(&x)?;

        // Project to vocabulary (x @ embedding.T)
        Ok(self.project_to_vocab(&x, seq_len))
    }

    /// Embed token IDs to vectors
    fn embed_tokens(&self, tokens: &[u32]) -> WhisperResult<Vec<f32>> {
        let seq_len = tokens.len();
        let mut embeddings = vec![0.0_f32; seq_len * self.d_model];

        for (pos, &token) in tokens.iter().enumerate() {
            let token_idx = token as usize;
            if token_idx >= self.n_vocab {
                return Err(WhisperError::Model(format!(
                    "token {} out of vocabulary range {}",
                    token, self.n_vocab
                )));
            }

            let emb_start = token_idx * self.d_model;
            let out_start = pos * self.d_model;

            embeddings[out_start..out_start + self.d_model]
                .copy_from_slice(&self.token_embedding[emb_start..emb_start + self.d_model]);
        }

        Ok(embeddings)
    }

    /// Project hidden states to vocabulary logits
    ///
    /// Computes x @ W_embedding^T (weight tying with token embeddings)
    fn project_to_vocab(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let mut logits = vec![0.0_f32; seq_len * self.n_vocab];

        for s in 0..seq_len {
            for v in 0..self.n_vocab {
                let mut sum = 0.0_f32;
                for d in 0..self.d_model {
                    sum += x[s * self.d_model + d] * self.token_embedding[v * self.d_model + d];
                }
                logits[s * self.n_vocab + v] = sum;
            }
        }

        logits
    }

    /// Get number of layers
    #[must_use]
    pub const fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Get model dimension
    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.d_model
    }

    /// Get number of attention heads
    #[must_use]
    pub const fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Get vocabulary size
    #[must_use]
    pub const fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// Get maximum sequence length
    #[must_use]
    pub const fn max_len(&self) -> usize {
        self.max_len
    }

    /// Get decoder blocks reference
    #[must_use]
    pub fn blocks(&self) -> &[DecoderBlock] {
        &self.blocks
    }

    /// Get token embedding reference
    #[must_use]
    pub fn token_embedding(&self) -> &[f32] {
        &self.token_embedding
    }

    /// Get mutable token embedding reference (for loading weights)
    pub fn token_embedding_mut(&mut self) -> &mut [f32] {
        &mut self.token_embedding
    }

    /// Get positional embedding reference
    #[must_use]
    pub fn positional_embedding(&self) -> &[f32] {
        &self.positional_embedding
    }

    /// Get mutable positional embedding reference (for loading weights)
    pub fn positional_embedding_mut(&mut self) -> &mut [f32] {
        &mut self.positional_embedding
    }

    /// Get mutable decoder blocks reference (for loading weights)
    pub fn blocks_mut(&mut self) -> &mut [DecoderBlock] {
        &mut self.blocks
    }

    /// Get layer norm reference
    #[must_use]
    pub fn ln_post(&self) -> &super::encoder::LayerNorm {
        &self.ln_post
    }

    /// Get mutable layer norm reference (for loading weights)
    pub fn ln_post_mut(&mut self) -> &mut super::encoder::LayerNorm {
        &mut self.ln_post
    }

    // =========================================================================
    // KV Cache Methods
    // =========================================================================

    /// Create a new KV cache for this decoder
    #[must_use]
    pub fn create_kv_cache(&self) -> DecoderKVCache {
        DecoderKVCache::new(self.n_layers, self.d_model, self.max_len)
    }

    /// Forward pass for a single token with KV cache (incremental decoding)
    ///
    /// This is more efficient than `forward` for autoregressive generation
    /// as it only processes the new token and reuses cached key/value tensors.
    ///
    /// # Arguments
    /// * `token` - Single token ID to process
    /// * `encoder_output` - Encoder hidden states (enc_len x d_model)
    /// * `cache` - Mutable reference to KV cache
    ///
    /// # Returns
    /// Logits over vocabulary for the new token (n_vocab)
    pub fn forward_one(
        &self,
        token: u32,
        encoder_output: &[f32],
        cache: &mut DecoderKVCache,
    ) -> WhisperResult<Vec<f32>> {
        let pos = cache.seq_len();

        if pos >= self.max_len {
            return Err(WhisperError::Model(format!(
                "cache position {} exceeds max {}",
                pos, self.max_len
            )));
        }

        // Embed the new token
        if token as usize >= self.n_vocab {
            return Err(WhisperError::Model(format!(
                "token {} out of vocabulary range {}",
                token, self.n_vocab
            )));
        }

        let emb_start = (token as usize) * self.d_model;
        let mut x: Vec<f32> = self.token_embedding[emb_start..emb_start + self.d_model].to_vec();

        // Add positional embedding for current position
        let pos_start = pos * self.d_model;
        for d in 0..self.d_model {
            x[d] += self.positional_embedding[pos_start + d];
        }

        // Pass through decoder blocks with cache
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            x = self.forward_block_cached(block, &x, encoder_output, layer_idx, cache)?;
        }

        // Final layer norm
        let x = self.ln_post.forward(&x)?;

        // Project to vocabulary
        Ok(self.project_to_vocab(&x, 1))
    }

    /// Forward pass through a single decoder block with KV cache
    fn forward_block_cached(
        &self,
        block: &DecoderBlock,
        x: &[f32],
        encoder_output: &[f32],
        layer_idx: usize,
        cache: &mut DecoderKVCache,
    ) -> WhisperResult<Vec<f32>> {
        // Pre-norm self-attention with cache
        let normed = block.ln1.forward(x)?;

        // Compute Q, K, V for the new position
        let q = block.self_attn.w_q().forward(&normed, 1)?;
        let k_new = block.self_attn.w_k().forward(&normed, 1)?;
        let v_new = block.self_attn.w_v().forward(&normed, 1)?;

        // Append new K, V to cache
        cache.self_attn_cache[layer_idx].append(&k_new, &v_new)?;

        // Get full K, V from cache for attention computation
        let k_full = cache.self_attn_cache[layer_idx].get_key();
        let v_full = cache.self_attn_cache[layer_idx].get_value();

        // Compute attention with full K, V (no mask needed - causal is implicit)
        // For incremental decoding, current position attends to all cached positions
        let attn_out = self.compute_attention_cached(&block.self_attn, &q, k_full, v_full)?;

        // Apply output projection and residual
        let attn_out = block.self_attn.w_o().forward(&attn_out, 1)?;
        let mut residual: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

        // Pre-norm cross-attention
        let normed = block.ln2.forward(&residual)?;
        let cross_out = block
            .cross_attn
            .forward_cross(&normed, encoder_output, None)?;
        for (r, c) in residual.iter_mut().zip(cross_out.iter()) {
            *r += c;
        }

        // Pre-norm FFN
        let normed = block.ln3.forward(&residual)?;
        let ffn_out = block.ffn.forward(&normed)?;
        for (r, f) in residual.iter_mut().zip(ffn_out.iter()) {
            *r += f;
        }

        Ok(residual)
    }

    /// Compute attention using cached K, V
    fn compute_attention_cached(
        &self,
        attn: &MultiHeadAttention,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> WhisperResult<Vec<f32>> {
        let n_heads = attn.n_heads();
        let d_head = attn.d_head();
        let kv_len = k.len() / self.d_model;

        let mut head_outputs = Vec::with_capacity(n_heads);

        for head in 0..n_heads {
            // Extract Q for this head (query_len = 1)
            let q_head: Vec<f32> = (0..d_head).map(|d| q[head * d_head + d]).collect();

            // Extract K, V for this head (all cached positions)
            let k_head: Vec<f32> = (0..kv_len)
                .flat_map(|pos| (0..d_head).map(move |d| k[pos * self.d_model + head * d_head + d]))
                .collect();

            let v_head: Vec<f32> = (0..kv_len)
                .flat_map(|pos| (0..d_head).map(move |d| v[pos * self.d_model + head * d_head + d]))
                .collect();

            // Compute attention for this head
            let head_out = attn.scaled_dot_product_attention(&q_head, &k_head, &v_head, None)?;
            head_outputs.push(head_out);
        }

        // Concatenate heads
        let mut output = vec![0.0_f32; self.d_model];
        for (head, head_data) in head_outputs.iter().enumerate() {
            for d in 0..d_head {
                output[head * d_head + d] = head_data[d];
            }
        }

        Ok(output)
    }

    /// Generate tokens autoregressively with KV cache
    ///
    /// # Arguments
    /// * `encoder_output` - Encoder hidden states
    /// * `initial_tokens` - Initial token sequence (e.g., SOT token)
    /// * `max_tokens` - Maximum number of tokens to generate
    /// * `eos_token` - End-of-sequence token ID
    ///
    /// # Returns
    /// Generated token sequence (including initial tokens)
    pub fn generate(
        &self,
        encoder_output: &[f32],
        initial_tokens: &[u32],
        max_tokens: usize,
        eos_token: u32,
    ) -> WhisperResult<Vec<u32>> {
        let mut cache = self.create_kv_cache();
        let mut tokens = initial_tokens.to_vec();

        // Process initial tokens (prime the cache)
        for &token in initial_tokens {
            let _ = self.forward_one(token, encoder_output, &mut cache)?;
        }

        // Generate new tokens
        for _ in initial_tokens.len()..max_tokens {
            let last_token = *tokens
                .last()
                .ok_or_else(|| WhisperError::Model("empty token sequence".into()))?;

            let logits = self.forward_one(last_token, encoder_output, &mut cache)?;

            // Greedy selection (argmax)
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(eos_token);

            tokens.push(next_token);

            if next_token == eos_token {
                break;
            }
        }

        Ok(tokens)
    }

    // =========================================================================
    // Batch Decoding Methods (WAPR-082)
    // =========================================================================

    /// Create a batch of KV caches for parallel decoding
    #[must_use]
    pub fn create_batch_cache(&self, batch_size: usize) -> BatchDecoderCache {
        BatchDecoderCache::new(batch_size, self.n_layers, self.d_model, self.max_len)
    }

    /// Forward pass for a batch of token sequences
    ///
    /// Processes multiple independent sequences in parallel. Each sequence
    /// can have a different length and encoder output.
    ///
    /// # Arguments
    /// * `tokens_batch` - Batch of token sequences (batch_size × variable seq_len)
    /// * `encoder_outputs` - Encoder hidden states for each batch item
    ///
    /// # Returns
    /// Batch of logits over vocabulary
    ///
    /// # Errors
    /// Returns error if batch sizes don't match or any sequence is invalid
    pub fn forward_batch(
        &self,
        tokens_batch: &[Vec<u32>],
        encoder_outputs: &[Vec<f32>],
    ) -> WhisperResult<BatchDecoderOutput> {
        if tokens_batch.is_empty() {
            return Err(WhisperError::Model("empty batch".into()));
        }
        if tokens_batch.len() != encoder_outputs.len() {
            return Err(WhisperError::Model(format!(
                "batch size mismatch: {} tokens vs {} encoders",
                tokens_batch.len(),
                encoder_outputs.len()
            )));
        }

        let mut logits = Vec::with_capacity(tokens_batch.len());
        let mut seq_lengths = Vec::with_capacity(tokens_batch.len());

        for (tokens, encoder_out) in tokens_batch.iter().zip(encoder_outputs.iter()) {
            let item_logits = self.forward(tokens, encoder_out)?;
            seq_lengths.push(tokens.len());
            logits.push(item_logits);
        }

        Ok(BatchDecoderOutput { logits, seq_lengths })
    }

    /// Forward pass for a single position across all batch items with KV cache
    ///
    /// Processes one token per batch item, updating the KV cache for each.
    /// This is efficient for autoregressive generation where all sequences
    /// advance by one position at a time.
    ///
    /// # Arguments
    /// * `tokens` - One token per batch item (length = batch_size)
    /// * `encoder_outputs` - Encoder hidden states for each batch item
    /// * `cache` - Mutable batch KV cache
    ///
    /// # Returns
    /// Logits for each batch item (batch_size × n_vocab)
    pub fn forward_one_batch(
        &self,
        tokens: &[u32],
        encoder_outputs: &[Vec<f32>],
        cache: &mut BatchDecoderCache,
    ) -> WhisperResult<BatchDecoderOutput> {
        let batch_size = cache.batch_size();

        if tokens.len() != batch_size {
            return Err(WhisperError::Model(format!(
                "token count {} doesn't match batch size {}",
                tokens.len(),
                batch_size
            )));
        }
        if encoder_outputs.len() != batch_size {
            return Err(WhisperError::Model(format!(
                "encoder count {} doesn't match batch size {}",
                encoder_outputs.len(),
                batch_size
            )));
        }

        let mut logits = Vec::with_capacity(batch_size);

        for (idx, (&token, encoder_out)) in tokens.iter().zip(encoder_outputs.iter()).enumerate() {
            let item_cache = cache.get_cache_mut(idx).ok_or_else(|| {
                WhisperError::Model(format!("cache index {} out of bounds", idx))
            })?;

            let item_logits = self.forward_one(token, encoder_out, item_cache)?;
            logits.push(item_logits);
        }

        Ok(BatchDecoderOutput {
            logits,
            seq_lengths: vec![1; batch_size],
        })
    }

    /// Generate tokens autoregressively for a batch of sequences
    ///
    /// # Arguments
    /// * `encoder_outputs` - Encoder hidden states for each batch item
    /// * `initial_tokens` - Initial token sequences for each batch item
    /// * `max_tokens` - Maximum number of tokens to generate per sequence
    /// * `eos_token` - End-of-sequence token ID
    ///
    /// # Returns
    /// Generated token sequences (one per batch item)
    pub fn generate_batch(
        &self,
        encoder_outputs: &[Vec<f32>],
        initial_tokens: &[Vec<u32>],
        max_tokens: usize,
        eos_token: u32,
    ) -> WhisperResult<Vec<Vec<u32>>> {
        let batch_size = encoder_outputs.len();

        if initial_tokens.len() != batch_size {
            return Err(WhisperError::Model(format!(
                "initial tokens count {} doesn't match batch size {}",
                initial_tokens.len(),
                batch_size
            )));
        }

        let mut cache = self.create_batch_cache(batch_size);
        let mut sequences: Vec<Vec<u32>> = initial_tokens.to_vec();
        let mut finished = vec![false; batch_size];

        // Prime the caches with initial tokens
        for (idx, tokens) in initial_tokens.iter().enumerate() {
            let item_cache = cache.get_cache_mut(idx).ok_or_else(|| {
                WhisperError::Model(format!("cache index {} out of bounds", idx))
            })?;

            for &token in tokens {
                let _ = self.forward_one(token, &encoder_outputs[idx], item_cache)?;
            }
        }

        // Generate new tokens
        for _ in 0..max_tokens {
            // Check if all sequences are finished
            if finished.iter().all(|&f| f) {
                break;
            }

            // Get last token for each sequence
            let last_tokens: Vec<u32> = sequences
                .iter()
                .map(|seq| *seq.last().unwrap_or(&0))
                .collect();

            // Forward pass for all batch items
            let outputs = self.forward_one_batch(&last_tokens, encoder_outputs, &mut cache)?;

            // Greedy selection for each batch item
            for (idx, logits) in outputs.logits.iter().enumerate() {
                if finished[idx] {
                    continue;
                }

                let next_token = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(eos_token);

                sequences[idx].push(next_token);

                if next_token == eos_token {
                    finished[idx] = true;
                }
            }
        }

        Ok(sequences)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Decoder Block Tests
    // =========================================================================

    #[test]
    fn test_decoder_block_new() {
        let block = DecoderBlock::new(64, 4, 256);
        assert_eq!(block.self_attn.d_model(), 64);
        assert_eq!(block.cross_attn.d_model(), 64);
        assert_eq!(block.ffn.d_model, 64);
    }

    #[test]
    fn test_decoder_block_forward() {
        let block = DecoderBlock::new(8, 2, 32);

        let x = vec![0.1_f32; 16]; // seq_len=2, d_model=8
        let encoder_out = vec![0.1_f32; 24]; // enc_len=3, d_model=8

        let output = block
            .forward(&x, &encoder_out, None)
            .expect("forward should succeed");

        assert_eq!(output.len(), 16); // Same as input
    }

    #[test]
    fn test_decoder_block_with_causal_mask() {
        let block = DecoderBlock::new(8, 2, 32);

        let x = vec![0.1_f32; 16]; // seq_len=2
        let encoder_out = vec![0.1_f32; 8]; // enc_len=1
        let causal_mask = MultiHeadAttention::causal_mask(2);

        let output = block
            .forward(&x, &encoder_out, Some(&causal_mask))
            .expect("forward should succeed");

        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_decoder_block_residual() {
        let block = DecoderBlock::new(8, 2, 32);

        let x = vec![1.0_f32; 8]; // seq_len=1
        let encoder_out = vec![0.0_f32; 8]; // enc_len=1

        let output = block
            .forward(&x, &encoder_out, None)
            .expect("forward should succeed");

        // Output should be modified by residual connections
        assert_eq!(output.len(), 8);
    }

    // =========================================================================
    // Decoder Construction Tests
    // =========================================================================

    #[test]
    fn test_decoder_new() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);
        assert_eq!(decoder.n_layers(), 4);
        assert_eq!(decoder.d_model(), 384);
        assert_eq!(decoder.n_heads(), 6);
    }

    #[test]
    fn test_decoder_vocab_size() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);
        assert_eq!(decoder.n_vocab(), 51865);
    }

    #[test]
    fn test_decoder_max_len() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);
        assert_eq!(decoder.max_len(), 448);
    }

    #[test]
    fn test_decoder_embedding_shapes() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        assert_eq!(
            decoder.token_embedding().len(),
            decoder.n_vocab() * decoder.d_model()
        );
        assert_eq!(
            decoder.positional_embedding().len(),
            decoder.max_len() * decoder.d_model()
        );
    }

    #[test]
    fn test_decoder_blocks_count() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);
        assert_eq!(decoder.blocks().len(), 4);
    }

    // =========================================================================
    // Token Embedding Tests
    // =========================================================================

    #[test]
    fn test_embed_tokens_basic() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let tokens = vec![0, 1, 2];
        let embeddings = decoder.embed_tokens(&tokens).expect("should succeed");

        assert_eq!(embeddings.len(), 3 * decoder.d_model());
    }

    #[test]
    fn test_embed_tokens_single() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let tokens = vec![100];
        let embeddings = decoder.embed_tokens(&tokens).expect("should succeed");

        assert_eq!(embeddings.len(), decoder.d_model());
    }

    #[test]
    fn test_embed_tokens_invalid() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let tokens = vec![100000]; // Out of vocab range
        let result = decoder.embed_tokens(&tokens);
        assert!(result.is_err());
    }

    // =========================================================================
    // Forward Pass Tests
    // =========================================================================

    #[test]
    fn test_decoder_forward_basic() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let tokens = vec![0, 1, 2]; // seq_len=3
        let encoder_out = vec![0.0_f32; 10 * 384]; // enc_len=10, d_model=384

        let logits = decoder
            .forward(&tokens, &encoder_out)
            .expect("forward should succeed");

        assert_eq!(logits.len(), 3 * decoder.n_vocab()); // seq_len * n_vocab
    }

    #[test]
    fn test_decoder_forward_single_token() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let tokens = vec![50258]; // SOT token
        let encoder_out = vec![0.0_f32; 5 * 384]; // enc_len=5

        let logits = decoder
            .forward(&tokens, &encoder_out)
            .expect("forward should succeed");

        assert_eq!(logits.len(), decoder.n_vocab());
    }

    #[test]
    fn test_decoder_forward_empty_tokens() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let tokens: Vec<u32> = vec![];
        let encoder_out = vec![0.0_f32; 5 * 384];

        let result = decoder.forward(&tokens, &encoder_out);
        assert!(result.is_err());
    }

    #[test]
    fn test_decoder_forward_sequence_too_long() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let tokens: Vec<u32> = vec![0; 500]; // Exceeds max_len (448)
        let encoder_out = vec![0.0_f32; 5 * 384];

        let result = decoder.forward(&tokens, &encoder_out);
        assert!(result.is_err());
    }

    #[test]
    fn test_decoder_forward_encoder_size_mismatch() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let tokens = vec![0, 1];
        let encoder_out = vec![0.0_f32; 100]; // Not divisible by d_model

        let result = decoder.forward(&tokens, &encoder_out);
        assert!(result.is_err());
    }

    // =========================================================================
    // Projection Tests
    // =========================================================================

    #[test]
    fn test_project_to_vocab_shape() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let hidden = vec![0.0_f32; 2 * 384]; // seq_len=2
        let logits = decoder.project_to_vocab(&hidden, 2);

        assert_eq!(logits.len(), 2 * decoder.n_vocab());
    }

    #[test]
    fn test_project_to_vocab_single() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let hidden = vec![0.0_f32; 384]; // seq_len=1
        let logits = decoder.project_to_vocab(&hidden, 1);

        assert_eq!(logits.len(), decoder.n_vocab());
    }

    // =========================================================================
    // Mutable Accessor Tests
    // =========================================================================

    #[test]
    fn test_token_embedding_mut() {
        let config = ModelConfig::tiny();
        let mut decoder = Decoder::new(&config);

        // Modify token embedding
        decoder.token_embedding_mut()[0] = 1.0;
        assert!((decoder.token_embedding()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_positional_embedding_mut() {
        let config = ModelConfig::tiny();
        let mut decoder = Decoder::new(&config);

        // Modify positional embedding
        decoder.positional_embedding_mut()[0] = 2.0;
        assert!((decoder.positional_embedding()[0] - 2.0).abs() < 1e-6);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_decoder_output_different_for_different_tokens() {
        let config = ModelConfig::tiny();
        let mut decoder = Decoder::new(&config);
        let d_model = decoder.d_model();

        // Set up different embeddings for tokens 0 and 1
        for d in 0..d_model {
            decoder.token_embedding_mut()[d] = 0.1;
            decoder.token_embedding_mut()[d_model + d] = 0.2;
        }

        let encoder_out = vec![0.0_f32; 384];

        let logits0 = decoder.forward(&[0], &encoder_out).expect("should succeed");
        let logits1 = decoder.forward(&[1], &encoder_out).expect("should succeed");

        // Outputs should be different
        let diff: f32 = logits0
            .iter()
            .zip(logits1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.0,
            "Different tokens should produce different outputs"
        );
    }

    #[test]
    fn test_decoder_base_config() {
        let config = ModelConfig::base();
        let decoder = Decoder::new(&config);

        assert_eq!(decoder.n_layers(), 6);
        assert_eq!(decoder.d_model(), 512);
        assert_eq!(decoder.n_heads(), 8);
    }

    // =========================================================================
    // Decoder Accessor Tests
    // =========================================================================

    #[test]
    fn test_decoder_blocks_mut() {
        let config = ModelConfig::tiny();
        let mut decoder = Decoder::new(&config);

        let blocks = decoder.blocks_mut();
        assert_eq!(blocks.len(), 4);
    }

    #[test]
    fn test_decoder_ln_post() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let ln = decoder.ln_post();
        assert_eq!(ln.normalized_shape, decoder.d_model());
    }

    #[test]
    fn test_decoder_ln_post_mut() {
        let config = ModelConfig::tiny();
        let mut decoder = Decoder::new(&config);

        // Modify ln_post weight
        decoder.ln_post_mut().weight[0] = 3.0;
        assert!((decoder.ln_post().weight[0] - 3.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // LayerKVCache Tests
    // =========================================================================

    #[test]
    fn test_layer_kv_cache_new() {
        let cache = LayerKVCache::new(64, 100);
        assert_eq!(cache.d_model, 64);
        assert_eq!(cache.max_len, 100);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_layer_kv_cache_append() {
        let mut cache = LayerKVCache::new(8, 100);

        let key = vec![1.0_f32; 8]; // 1 position
        let value = vec![2.0_f32; 8];

        cache.append(&key, &value).expect("append should succeed");

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
        assert_eq!(cache.get_key().len(), 8);
        assert_eq!(cache.get_value().len(), 8);
    }

    #[test]
    fn test_layer_kv_cache_append_multiple() {
        let mut cache = LayerKVCache::new(8, 100);

        // Append 3 positions
        for _ in 0..3 {
            let key = vec![1.0_f32; 8];
            let value = vec![2.0_f32; 8];
            cache.append(&key, &value).expect("append should succeed");
        }

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get_key().len(), 24);
        assert_eq!(cache.get_value().len(), 24);
    }

    #[test]
    fn test_layer_kv_cache_overflow() {
        let mut cache = LayerKVCache::new(8, 2);

        // Fill to capacity
        cache.append(&[1.0; 8], &[2.0; 8]).expect("first append");
        cache.append(&[1.0; 8], &[2.0; 8]).expect("second append");

        // This should fail
        let result = cache.append(&[1.0; 8], &[2.0; 8]);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_kv_cache_size_mismatch() {
        let mut cache = LayerKVCache::new(8, 100);

        // Key and value different sizes
        let result = cache.append(&[1.0; 8], &[2.0; 16]);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_kv_cache_clear() {
        let mut cache = LayerKVCache::new(8, 100);
        cache.append(&[1.0; 8], &[2.0; 8]).expect("append");

        cache.clear();

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    // =========================================================================
    // LayerKVCache Optimization Tests (WAPR-PERF-003)
    // =========================================================================

    #[test]
    fn test_layer_kv_cache_preallocated() {
        let cache = LayerKVCache::new_preallocated(64, 100);
        assert_eq!(cache.d_model, 64);
        assert_eq!(cache.max_len, 100);
        assert!(cache.is_empty());
        // Capacity should be pre-allocated
        assert!(cache.key.capacity() >= 64 * 100);
        assert!(cache.value.capacity() >= 64 * 100);
    }

    #[test]
    fn test_layer_kv_cache_remaining_capacity() {
        let mut cache = LayerKVCache::new(8, 10);
        assert_eq!(cache.remaining_capacity(), 10);

        cache.append(&[1.0; 8], &[2.0; 8]).expect("append");
        assert_eq!(cache.remaining_capacity(), 9);

        cache
            .append(&[1.0; 16], &[2.0; 16])
            .expect("append 2 positions");
        assert_eq!(cache.remaining_capacity(), 7);
    }

    #[test]
    fn test_layer_kv_cache_is_full() {
        let mut cache = LayerKVCache::new(8, 2);
        assert!(!cache.is_full());

        cache.append(&[1.0; 8], &[2.0; 8]).expect("append 1");
        assert!(!cache.is_full());

        cache.append(&[1.0; 8], &[2.0; 8]).expect("append 2");
        assert!(cache.is_full());
    }

    #[test]
    fn test_layer_kv_cache_reset_preserves_capacity() {
        let mut cache = LayerKVCache::new(8, 100);

        // Add some data
        for _ in 0..10 {
            cache.append(&[1.0; 8], &[2.0; 8]).expect("append");
        }
        let cap_before = cache.key.capacity();
        assert!(cap_before >= 80);

        // Reset should preserve capacity
        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert!(cache.key.capacity() >= cap_before);
    }

    #[test]
    fn test_layer_kv_cache_append_batch() {
        let mut cache = LayerKVCache::new(8, 100);

        // Append 3 positions at once
        let keys = vec![1.0_f32; 24]; // 3 x 8
        let values = vec![2.0_f32; 24];

        cache.append_batch(&keys, &values, 3).expect("batch append");

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get_key().len(), 24);
        assert_eq!(cache.get_value().len(), 24);
    }

    #[test]
    fn test_layer_kv_cache_append_batch_mismatch() {
        let mut cache = LayerKVCache::new(8, 100);

        // Wrong number of keys
        let keys = vec![1.0_f32; 16]; // 2 x 8
        let values = vec![2.0_f32; 24]; // 3 x 8

        let result = cache.append_batch(&keys, &values, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_kv_cache_append_batch_overflow() {
        let mut cache = LayerKVCache::new(8, 2);

        // Try to append 3 positions to cache with max_len=2
        let keys = vec![1.0_f32; 24];
        let values = vec![2.0_f32; 24];

        let result = cache.append_batch(&keys, &values, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_kv_cache_get_key_range() {
        let mut cache = LayerKVCache::new(4, 100);

        // Append 5 positions
        for i in 0..5 {
            let keys: Vec<f32> = (0..4).map(|d| (i * 4 + d) as f32).collect();
            let values: Vec<f32> = (0..4).map(|d| (i * 4 + d + 100) as f32).collect();
            cache.append(&keys, &values).expect("append");
        }

        // Get positions 1-3
        let range = cache.get_key_range(1, 3).expect("should get range");
        assert_eq!(range.len(), 8); // 2 positions x 4 d_model
        assert!((range[0] - 4.0).abs() < f32::EPSILON); // First element of position 1
    }

    #[test]
    fn test_layer_kv_cache_get_value_range() {
        let mut cache = LayerKVCache::new(4, 100);

        for i in 0..5 {
            let keys: Vec<f32> = (0..4).map(|d| (i * 4 + d) as f32).collect();
            let values: Vec<f32> = (0..4).map(|d| (i * 4 + d + 100) as f32).collect();
            cache.append(&keys, &values).expect("append");
        }

        let range = cache.get_value_range(2, 4).expect("should get range");
        assert_eq!(range.len(), 8);
        assert!((range[0] - 108.0).abs() < f32::EPSILON); // First element of position 2's value
    }

    #[test]
    fn test_layer_kv_cache_get_range_out_of_bounds() {
        let mut cache = LayerKVCache::new(4, 100);
        cache.append(&[1.0; 4], &[2.0; 4]).expect("append");

        // Out of bounds
        assert!(cache.get_key_range(0, 5).is_none());
        assert!(cache.get_value_range(2, 3).is_none());

        // Invalid range (start > end)
        assert!(cache.get_key_range(3, 1).is_none());
    }

    #[test]
    fn test_layer_kv_cache_memory_bytes() {
        let mut cache = LayerKVCache::new(8, 100);
        assert_eq!(cache.memory_bytes(), 0);

        cache.append(&[1.0; 8], &[2.0; 8]).expect("append");
        // 8 keys + 8 values = 16 floats = 64 bytes
        assert_eq!(cache.memory_bytes(), 64);
    }

    #[test]
    fn test_layer_kv_cache_capacity_bytes() {
        let cache = LayerKVCache::new_preallocated(8, 10);
        // Should have capacity for at least 10 * 8 * 2 floats = 640 bytes
        assert!(cache.capacity_bytes() >= 640);
    }

    // =========================================================================
    // DecoderKVCache Tests
    // =========================================================================

    #[test]
    fn test_decoder_kv_cache_new() {
        let cache = DecoderKVCache::new(4, 64, 100);
        assert_eq!(cache.n_layers, 4);
        assert_eq!(cache.d_model, 64);
        assert_eq!(cache.max_len, 100);
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_decoder_kv_cache_layer_count() {
        let cache = DecoderKVCache::new(4, 64, 100);
        assert_eq!(cache.self_attn_cache.len(), 4);
        assert_eq!(cache.cross_attn_cache.len(), 4);
    }

    #[test]
    fn test_decoder_kv_cache_clear() {
        let mut cache = DecoderKVCache::new(4, 8, 100);

        // Add some data
        cache.self_attn_cache[0]
            .append(&[1.0; 8], &[2.0; 8])
            .expect("append");
        cache.cross_attn_cached = true;

        cache.clear();

        assert!(cache.is_empty());
        assert!(!cache.cross_attn_cached);
    }

    #[test]
    fn test_decoder_kv_cache_clear_self_attn() {
        let mut cache = DecoderKVCache::new(4, 8, 100);

        cache.self_attn_cache[0]
            .append(&[1.0; 8], &[2.0; 8])
            .expect("append");
        cache.cross_attn_cache[0]
            .append(&[1.0; 8], &[2.0; 8])
            .expect("append");
        cache.cross_attn_cached = true;

        cache.clear_self_attn();

        assert!(cache.self_attn_cache[0].is_empty());
        assert!(!cache.cross_attn_cache[0].is_empty()); // Cross-attention preserved
        assert!(cache.cross_attn_cached); // Flag preserved
    }

    #[test]
    fn test_decoder_kv_cache_memory_bytes() {
        let mut cache = DecoderKVCache::new(2, 8, 100);

        // Empty cache should use 0 bytes
        assert_eq!(cache.memory_bytes(), 0);

        // Add data to one layer
        cache.self_attn_cache[0]
            .append(&[1.0; 8], &[2.0; 8])
            .expect("append");

        // 8 floats for key + 8 floats for value = 16 * 4 bytes = 64 bytes
        assert_eq!(cache.memory_bytes(), 64);
    }

    // =========================================================================
    // Decoder KV Cache Integration Tests
    // =========================================================================

    #[test]
    fn test_decoder_create_kv_cache() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let cache = decoder.create_kv_cache();

        assert_eq!(cache.n_layers, decoder.n_layers());
        assert_eq!(cache.d_model, decoder.d_model());
        assert_eq!(cache.max_len, decoder.max_len());
    }

    #[test]
    fn test_decoder_forward_one_basic() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);
        let mut cache = decoder.create_kv_cache();

        let encoder_out = vec![0.0_f32; 5 * 384]; // enc_len=5

        let logits = decoder
            .forward_one(0, &encoder_out, &mut cache)
            .expect("forward_one should succeed");

        assert_eq!(logits.len(), decoder.n_vocab());
        assert_eq!(cache.seq_len(), 1);
    }

    #[test]
    fn test_decoder_forward_one_multiple() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);
        let mut cache = decoder.create_kv_cache();

        let encoder_out = vec![0.0_f32; 5 * 384];

        // Process 3 tokens
        for token in 0..3 {
            let _ = decoder
                .forward_one(token, &encoder_out, &mut cache)
                .expect("forward_one should succeed");
        }

        assert_eq!(cache.seq_len(), 3);
    }

    #[test]
    fn test_decoder_forward_one_invalid_token() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);
        let mut cache = decoder.create_kv_cache();

        let encoder_out = vec![0.0_f32; 5 * 384];

        let result = decoder.forward_one(100000, &encoder_out, &mut cache);
        assert!(result.is_err());
    }

    #[test]
    fn test_decoder_generate_basic() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let encoder_out = vec![0.0_f32; 5 * 384];
        let initial = vec![50258_u32]; // SOT token
        let eos = 50257_u32; // EOS token

        // Generate up to 5 tokens
        let tokens = decoder
            .generate(&encoder_out, &initial, 5, eos)
            .expect("generate should succeed");

        // Should have at least the initial tokens
        assert!(tokens.len() >= initial.len());
        assert!(tokens.len() <= 5);
        assert_eq!(tokens[0], 50258);
    }

    #[test]
    fn test_decoder_generate_stops_at_eos() {
        let config = ModelConfig::tiny();
        let mut decoder = Decoder::new(&config);

        // Set up embeddings so that token 1 strongly predicts EOS (token 50257)
        let d_model = decoder.d_model();

        // Make token 1 embedding strongly positive in dimension 0
        for d in 0..d_model {
            decoder.token_embedding_mut()[d_model + d] = if d == 0 { 10.0 } else { 0.0 };
        }
        // Make EOS embedding strongly positive in dimension 0
        let eos_start = 50257 * d_model;
        for d in 0..d_model {
            decoder.token_embedding_mut()[eos_start + d] = if d == 0 { 10.0 } else { 0.0 };
        }

        let encoder_out = vec![0.0_f32; 384];
        let initial = vec![1_u32];
        let eos = 50257_u32;

        let tokens = decoder
            .generate(&encoder_out, &initial, 100, eos)
            .expect("generate should succeed");

        // Should stop early due to EOS
        assert!(tokens.len() < 100);
    }

    #[test]
    fn test_kv_cache_reuse() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);
        let mut cache = decoder.create_kv_cache();

        let encoder_out = vec![0.0_f32; 5 * 384];

        // Process first token
        let logits1 = decoder
            .forward_one(0, &encoder_out, &mut cache)
            .expect("first forward");

        // Process second token (should use cached K,V from first)
        let logits2 = decoder
            .forward_one(1, &encoder_out, &mut cache)
            .expect("second forward");

        // Both should produce valid logits
        assert_eq!(logits1.len(), decoder.n_vocab());
        assert_eq!(logits2.len(), decoder.n_vocab());

        // Cache should have grown
        assert_eq!(cache.seq_len(), 2);
    }

    // =========================================================================
    // WAPR-082: Batched Decoder with Shared KV Cache Tests
    // =========================================================================

    #[test]
    fn test_batch_decoder_cache_new() {
        let cache = BatchDecoderCache::new(3, 4, 64, 100);
        assert_eq!(cache.batch_size(), 3);
        assert_eq!(cache.n_layers, 4);
        assert_eq!(cache.d_model, 64);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_batch_decoder_cache_get_cache() {
        let cache = BatchDecoderCache::new(3, 4, 64, 100);

        let item0 = cache.get_cache(0);
        assert!(item0.is_some());
        assert_eq!(item0.unwrap().n_layers, 4);

        let item3 = cache.get_cache(3);
        assert!(item3.is_none()); // Out of bounds
    }

    #[test]
    fn test_batch_decoder_cache_get_cache_mut() {
        let mut cache = BatchDecoderCache::new(2, 4, 8, 100);

        // Append to first cache
        {
            let item0 = cache.get_cache_mut(0).unwrap();
            item0.self_attn_cache[0].append(&[1.0; 8], &[2.0; 8]).unwrap();
        }

        // Verify
        assert_eq!(cache.get_cache(0).unwrap().seq_len(), 1);
        assert_eq!(cache.get_cache(1).unwrap().seq_len(), 0);
    }

    #[test]
    fn test_batch_decoder_cache_clear_all() {
        let mut cache = BatchDecoderCache::new(2, 4, 8, 100);

        // Add data
        cache.get_cache_mut(0).unwrap().self_attn_cache[0]
            .append(&[1.0; 8], &[2.0; 8]).unwrap();
        cache.get_cache_mut(1).unwrap().self_attn_cache[0]
            .append(&[1.0; 8], &[2.0; 8]).unwrap();

        cache.clear_all();

        assert!(cache.get_cache(0).unwrap().is_empty());
        assert!(cache.get_cache(1).unwrap().is_empty());
    }

    #[test]
    fn test_batch_decoder_cache_seq_lengths() {
        let mut cache = BatchDecoderCache::new(3, 4, 8, 100);

        // Add different lengths
        cache.get_cache_mut(0).unwrap().self_attn_cache[0]
            .append(&[1.0; 16], &[2.0; 16]).unwrap(); // 2 positions
        cache.get_cache_mut(1).unwrap().self_attn_cache[0]
            .append(&[1.0; 8], &[2.0; 8]).unwrap(); // 1 position
        // Item 2 stays empty

        let lengths = cache.seq_lengths();
        assert_eq!(lengths, vec![2, 1, 0]);
    }

    #[test]
    fn test_batch_decoder_cache_max_seq_len() {
        let mut cache = BatchDecoderCache::new(3, 4, 8, 100);

        cache.get_cache_mut(0).unwrap().self_attn_cache[0]
            .append(&[1.0; 8], &[2.0; 8]).unwrap();
        cache.get_cache_mut(1).unwrap().self_attn_cache[0]
            .append(&[1.0; 24], &[2.0; 24]).unwrap(); // 3 positions

        assert_eq!(cache.max_seq_len(), 3);
    }

    #[test]
    fn test_decoder_create_batch_cache() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let cache = decoder.create_batch_cache(4);

        assert_eq!(cache.batch_size(), 4);
        assert_eq!(cache.n_layers, decoder.n_layers());
        assert_eq!(cache.d_model, decoder.d_model());
    }

    #[test]
    fn test_decoder_forward_batch_basic() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        // Batch of 2 token sequences
        let tokens_batch = vec![
            vec![0_u32, 1, 2],  // seq_len=3
            vec![3_u32, 4],    // seq_len=2
        ];
        let encoder_outputs = vec![
            vec![0.0_f32; 5 * 384],  // enc_len=5
            vec![0.0_f32; 3 * 384],  // enc_len=3
        ];

        let result = decoder.forward_batch(&tokens_batch, &encoder_outputs)
            .expect("forward_batch should succeed");

        assert_eq!(result.batch_size(), 2);
        assert_eq!(result.logits.len(), 2);
        assert_eq!(result.logits[0].len(), 3 * decoder.n_vocab()); // seq_len * vocab
        assert_eq!(result.logits[1].len(), 2 * decoder.n_vocab());
    }

    #[test]
    fn test_decoder_forward_batch_empty() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let result = decoder.forward_batch(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decoder_forward_batch_mismatch() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let tokens = vec![vec![0_u32]];
        let encoders = vec![vec![0.0_f32; 384], vec![0.0_f32; 384]]; // Mismatch

        let result = decoder.forward_batch(&tokens, &encoders);
        assert!(result.is_err());
    }

    #[test]
    fn test_decoder_forward_one_batch_basic() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);
        let mut cache = decoder.create_batch_cache(2);

        let tokens = vec![0_u32, 1_u32]; // One token per batch item
        let encoder_outputs = vec![
            vec![0.0_f32; 5 * 384],
            vec![0.0_f32; 3 * 384],
        ];

        let result = decoder.forward_one_batch(&tokens, &encoder_outputs, &mut cache)
            .expect("forward_one_batch should succeed");

        assert_eq!(result.batch_size(), 2);
        assert_eq!(result.logits.len(), 2);
        assert_eq!(result.logits[0].len(), decoder.n_vocab());
        assert_eq!(result.logits[1].len(), decoder.n_vocab());

        // Cache should be updated
        assert_eq!(cache.get_cache(0).unwrap().seq_len(), 1);
        assert_eq!(cache.get_cache(1).unwrap().seq_len(), 1);
    }

    #[test]
    fn test_decoder_forward_one_batch_multiple_steps() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);
        let mut cache = decoder.create_batch_cache(2);

        let encoder_outputs = vec![
            vec![0.0_f32; 5 * 384],
            vec![0.0_f32; 5 * 384],
        ];

        // Step 1
        decoder.forward_one_batch(&[0, 1], &encoder_outputs, &mut cache).unwrap();
        // Step 2
        decoder.forward_one_batch(&[2, 3], &encoder_outputs, &mut cache).unwrap();
        // Step 3
        decoder.forward_one_batch(&[4, 5], &encoder_outputs, &mut cache).unwrap();

        assert_eq!(cache.get_cache(0).unwrap().seq_len(), 3);
        assert_eq!(cache.get_cache(1).unwrap().seq_len(), 3);
    }

    #[test]
    fn test_decoder_forward_one_batch_size_mismatch() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);
        let mut cache = decoder.create_batch_cache(2);

        let tokens = vec![0_u32, 1, 2]; // 3 tokens but batch size is 2
        let encoder_outputs = vec![vec![0.0_f32; 384], vec![0.0_f32; 384]];

        let result = decoder.forward_one_batch(&tokens, &encoder_outputs, &mut cache);
        assert!(result.is_err());
    }

    #[test]
    fn test_decoder_generate_batch_basic() {
        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        let encoder_outputs = vec![
            vec![0.0_f32; 5 * 384],
            vec![0.0_f32; 5 * 384],
        ];
        let initial_tokens = vec![
            vec![50258_u32], // SOT for first
            vec![50258_u32], // SOT for second
        ];
        let eos = 50257_u32;

        let result = decoder.generate_batch(&encoder_outputs, &initial_tokens, 5, eos)
            .expect("generate_batch should succeed");

        assert_eq!(result.len(), 2);
        assert!(result[0].len() >= 1);
        assert!(result[1].len() >= 1);
    }

    #[test]
    fn test_batch_decoder_output_batch_size() {
        let output = BatchDecoderOutput {
            logits: vec![vec![0.0; 100], vec![0.0; 100], vec![0.0; 100]],
            seq_lengths: vec![5, 3, 4],
        };
        assert_eq!(output.batch_size(), 3);
    }

    #[test]
    fn test_batch_decoder_output_get_logits() {
        let output = BatchDecoderOutput {
            logits: vec![vec![1.0; 10], vec![2.0; 10]],
            seq_lengths: vec![1, 1],
        };

        assert_eq!(output.get_logits(0).unwrap()[0], 1.0);
        assert_eq!(output.get_logits(1).unwrap()[0], 2.0);
        assert!(output.get_logits(2).is_none());
    }

    #[test]
    fn test_batch_decoder_output_is_empty() {
        let empty = BatchDecoderOutput {
            logits: vec![],
            seq_lengths: vec![],
        };
        assert!(empty.is_empty());

        let non_empty = BatchDecoderOutput {
            logits: vec![vec![0.0]],
            seq_lengths: vec![1],
        };
        assert!(!non_empty.is_empty());
    }

    // =========================================================================
    // WAPR-111: Streaming KV Cache Tests
    // =========================================================================

    #[test]
    fn test_streaming_kv_cache_new() {
        let cache = StreamingKVCache::new(4, 64, 100, 20);
        assert_eq!(cache.window_size(), 100);
        assert_eq!(cache.context_overlap(), 20);
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.total_tokens(), 0);
        assert_eq!(cache.slide_count(), 0);
    }

    #[test]
    fn test_streaming_kv_cache_low_latency() {
        let cache = StreamingKVCache::low_latency(4, 64);
        assert_eq!(cache.window_size(), 64);
        assert_eq!(cache.context_overlap(), 16);
    }

    #[test]
    fn test_streaming_kv_cache_ultra_low_latency() {
        let cache = StreamingKVCache::ultra_low_latency(4, 64);
        assert_eq!(cache.window_size(), 32);
        assert_eq!(cache.context_overlap(), 8);
    }

    #[test]
    fn test_streaming_kv_cache_standard() {
        let cache = StreamingKVCache::standard(4, 64);
        assert_eq!(cache.window_size(), 448);
        assert_eq!(cache.context_overlap(), 64);
    }

    #[test]
    fn test_streaming_kv_cache_overlap_clamped() {
        // Context overlap should be clamped to max 50% of window
        let cache = StreamingKVCache::new(4, 64, 100, 80);
        assert_eq!(cache.context_overlap(), 50); // Clamped to 50% of 100
    }

    #[test]
    fn test_streaming_kv_cache_remaining_capacity() {
        let mut cache = StreamingKVCache::new(4, 8, 10, 2);
        assert_eq!(cache.remaining_capacity(), 10);

        // Add some data
        cache.inner_mut().self_attn_cache[0]
            .append(&[1.0; 8], &[2.0; 8])
            .unwrap();
        assert_eq!(cache.remaining_capacity(), 9);
    }

    #[test]
    fn test_streaming_kv_cache_will_slide() {
        let mut cache = StreamingKVCache::new(4, 8, 3, 1);
        assert!(!cache.will_slide());

        // Fill to capacity
        cache.inner_mut().self_attn_cache[0].append(&[1.0; 8], &[2.0; 8]).unwrap();
        cache.inner_mut().self_attn_cache[0].append(&[1.0; 8], &[2.0; 8]).unwrap();
        cache.inner_mut().self_attn_cache[0].append(&[1.0; 8], &[2.0; 8]).unwrap();

        assert!(cache.will_slide());
    }

    #[test]
    fn test_streaming_kv_cache_append_with_slide() {
        let mut cache = StreamingKVCache::new(2, 8, 4, 2);

        // Append 3 tokens - should not trigger slide
        for _ in 0..3 {
            cache.append_with_slide(0, &[1.0; 8], &[2.0; 8]).unwrap();
        }
        assert_eq!(cache.seq_len(), 3);
        assert_eq!(cache.slide_count(), 0);

        // Append 2 more - should trigger slide (total would be 5 > window 4)
        cache.append_with_slide(0, &[1.0; 8], &[2.0; 8]).unwrap();
        cache.append_with_slide(0, &[1.0; 8], &[2.0; 8]).unwrap();

        assert!(cache.slide_count() > 0);
        assert!(cache.seq_len() <= cache.window_size());
    }

    #[test]
    fn test_streaming_kv_cache_slide_preserves_overlap() {
        let mut cache = StreamingKVCache::new(1, 4, 5, 2);

        // Fill with recognizable values
        for i in 0..5 {
            let keys: Vec<f32> = (0..4).map(|d| (i * 4 + d) as f32).collect();
            let values: Vec<f32> = (0..4).map(|d| (i * 4 + d + 100) as f32).collect();
            cache.inner_mut().self_attn_cache[0].append(&keys, &values).unwrap();
        }
        assert_eq!(cache.seq_len(), 5);

        // Slide window
        cache.slide_window().unwrap();

        // Should have kept last 2 tokens (context_overlap)
        assert_eq!(cache.seq_len(), 2);
        assert_eq!(cache.slide_count(), 1);

        // Check that the preserved values are from the end
        let keys = cache.inner().self_attn_cache[0].get_key();
        // Should have tokens 3 and 4 (indices from 0..5)
        // Token 3's first key value = 12.0
        assert!((keys[0] - 12.0).abs() < 0.01);
    }

    #[test]
    fn test_streaming_kv_cache_reset() {
        let mut cache = StreamingKVCache::new(2, 8, 10, 2);

        // Add data and slide
        for _ in 0..15 {
            cache.append_with_slide(0, &[1.0; 8], &[2.0; 8]).unwrap();
        }

        let prev_total = cache.total_tokens();
        let prev_slides = cache.slide_count();

        cache.reset();

        assert!(cache.is_empty());
        // Statistics should be preserved
        assert_eq!(cache.total_tokens(), prev_total);
        assert_eq!(cache.slide_count(), prev_slides);
    }

    #[test]
    fn test_streaming_kv_cache_full_reset() {
        let mut cache = StreamingKVCache::new(2, 8, 10, 2);

        for _ in 0..15 {
            cache.append_with_slide(0, &[1.0; 8], &[2.0; 8]).unwrap();
        }

        cache.full_reset();

        assert!(cache.is_empty());
        assert_eq!(cache.total_tokens(), 0);
        assert_eq!(cache.slide_count(), 0);
    }

    #[test]
    fn test_streaming_kv_cache_warm_up() {
        let mut cache = StreamingKVCache::new(2, 4, 10, 3);

        // Warm up with 5 tokens worth of data
        let keys: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let values: Vec<f32> = (0..20).map(|i| i as f32 + 100.0).collect();

        cache.warm_up(0, &keys, &values).unwrap();

        // Should have used last context_overlap (3) tokens
        assert_eq!(cache.seq_len(), 3);

        // Check that it's the last 3 tokens
        let cached_keys = cache.inner().self_attn_cache[0].get_key();
        // Token 2 (3rd from end of 5): value 8 (2*4)
        assert!((cached_keys[0] - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_streaming_kv_cache_warm_up_invalid_layer() {
        let mut cache = StreamingKVCache::new(2, 4, 10, 3);

        let result = cache.warm_up(5, &[1.0; 8], &[2.0; 8]); // Layer 5 doesn't exist
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_kv_cache_stats() {
        let mut cache = StreamingKVCache::new(2, 8, 10, 2);

        // Add some tokens
        for _ in 0..5 {
            cache.append_with_slide(0, &[1.0; 8], &[2.0; 8]).unwrap();
        }

        let stats = cache.stats();
        assert_eq!(stats.seq_len, 5);
        assert_eq!(stats.total_tokens, 5);
        assert_eq!(stats.window_size, 10);
        assert_eq!(stats.context_overlap, 2);
        assert!((stats.utilization() - 0.5).abs() < 0.01); // 5/10 = 0.5
    }

    #[test]
    fn test_streaming_cache_stats_utilization() {
        let stats = StreamingCacheStats {
            seq_len: 25,
            total_tokens: 100,
            slide_count: 3,
            window_size: 50,
            context_overlap: 10,
            memory_bytes: 1000,
        };

        assert!((stats.utilization() - 0.5).abs() < 0.01); // 25/50
    }

    #[test]
    fn test_streaming_cache_stats_tokens_per_slide() {
        let stats = StreamingCacheStats {
            seq_len: 25,
            total_tokens: 100,
            slide_count: 4,
            window_size: 50,
            context_overlap: 10,
            memory_bytes: 1000,
        };

        assert!((stats.tokens_per_slide() - 25.0).abs() < 0.01); // 100/4
    }

    #[test]
    fn test_streaming_cache_stats_tokens_per_slide_no_slides() {
        let stats = StreamingCacheStats {
            seq_len: 10,
            total_tokens: 10,
            slide_count: 0,
            window_size: 50,
            context_overlap: 10,
            memory_bytes: 1000,
        };

        assert!((stats.tokens_per_slide() - 10.0).abs() < 0.01); // total_tokens
    }

    #[test]
    fn test_streaming_cache_stats_zero_window() {
        let stats = StreamingCacheStats {
            seq_len: 0,
            total_tokens: 0,
            slide_count: 0,
            window_size: 0,
            context_overlap: 0,
            memory_bytes: 0,
        };

        assert!((stats.utilization() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_streaming_kv_cache_inner_accessors() {
        let mut cache = StreamingKVCache::new(2, 8, 10, 2);

        // Test inner() accessor
        let inner_ref = cache.inner();
        assert_eq!(inner_ref.n_layers, 2);

        // Test inner_mut() accessor
        let inner_mut = cache.inner_mut();
        inner_mut.self_attn_cache[0].append(&[1.0; 8], &[2.0; 8]).unwrap();

        assert_eq!(cache.seq_len(), 1);
    }

    #[test]
    fn test_streaming_kv_cache_memory_bytes() {
        let mut cache = StreamingKVCache::new(2, 8, 10, 2);
        assert_eq!(cache.memory_bytes(), 0);

        cache.inner_mut().self_attn_cache[0].append(&[1.0; 8], &[2.0; 8]).unwrap();
        assert!(cache.memory_bytes() > 0);
    }

    #[test]
    fn test_streaming_kv_cache_continuous_streaming() {
        // Simulate a long streaming session
        let mut cache = StreamingKVCache::new(2, 8, 20, 5);

        for _ in 0..100 {
            cache.append_with_slide(0, &[1.0; 8], &[2.0; 8]).unwrap();
        }

        // Cache should be bounded
        assert!(cache.seq_len() <= cache.window_size());

        // Should have slid multiple times
        assert!(cache.slide_count() > 0);

        // Total tokens should be tracked
        assert_eq!(cache.total_tokens(), 100);

        let stats = cache.stats();
        assert!(stats.tokens_per_slide() > 0.0);
    }
}
