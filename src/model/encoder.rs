//! Transformer encoder
//!
//! Implements the Whisper audio encoder which processes mel spectrograms
//! through convolutional layers and transformer blocks.
//!
//! # Architecture
//!
//! 1. Two 1D convolutions on mel spectrogram
//! 2. Sinusoidal positional encoding
//! 3. N transformer encoder blocks (self-attention + FFN)
//!
//! # References
//!
//! - Radford et al. (2023): "Robust Speech Recognition via Large-Scale Weak Supervision"

use super::{LinearWeights, ModelConfig, MultiHeadAttention};
use crate::error::{WhisperError, WhisperResult};

// ============================================================================
// Convolutional Frontend
// ============================================================================

/// 1D convolution layer for audio processing
///
/// Implements Conv1d as used in Whisper's encoder frontend.
#[derive(Debug, Clone)]
pub struct Conv1d {
    /// Weight tensor (out_channels x in_channels x kernel_size)
    pub weight: Vec<f32>,
    /// Bias tensor (out_channels)
    pub bias: Vec<f32>,
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
}

impl Conv1d {
    /// Create new Conv1d layer
    #[must_use]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        Self {
            weight: vec![0.0; out_channels * in_channels * kernel_size],
            bias: vec![0.0; out_channels],
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `input` - Input tensor (seq_len x in_channels) flattened row-major
    ///
    /// # Returns
    /// Output tensor (out_seq_len x out_channels) flattened row-major
    pub fn forward(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        let seq_len = input.len() / self.in_channels;
        if input.len() % self.in_channels != 0 {
            return Err(WhisperError::Model("Conv1d input size mismatch".into()));
        }

        // Calculate output sequence length
        let out_seq_len = (seq_len + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let mut output = vec![0.0_f32; out_seq_len * self.out_channels];

        // Convolution with padding
        for out_pos in 0..out_seq_len {
            let in_start = out_pos as isize * self.stride as isize - self.padding as isize;

            for out_ch in 0..self.out_channels {
                let mut sum = self.bias[out_ch];

                for k in 0..self.kernel_size {
                    let in_pos = in_start + k as isize;

                    // Skip if outside padded region
                    if in_pos >= 0 && (in_pos as usize) < seq_len {
                        for in_ch in 0..self.in_channels {
                            let weight_idx = out_ch * self.in_channels * self.kernel_size
                                + in_ch * self.kernel_size
                                + k;
                            let input_idx = (in_pos as usize) * self.in_channels + in_ch;
                            sum += self.weight[weight_idx] * input[input_idx];
                        }
                    }
                }

                output[out_pos * self.out_channels + out_ch] = sum;
            }
        }

        Ok(output)
    }

    /// Get mutable weight reference (for loading weights)
    pub fn weight_mut(&mut self) -> &mut [f32] {
        &mut self.weight
    }

    /// Get mutable bias reference (for loading weights)
    pub fn bias_mut(&mut self) -> &mut [f32] {
        &mut self.bias
    }
}

/// Convolutional frontend for Whisper encoder
///
/// Processes mel spectrogram through two 1D convolutions:
/// 1. Conv1d (n_mels → n_audio_state) with kernel_size=3, padding=1
/// 2. Conv1d (n_audio_state → n_audio_state) with kernel_size=3, stride=2, padding=1
#[derive(Debug, Clone)]
pub struct ConvFrontend {
    /// First convolution (mel → hidden)
    pub conv1: Conv1d,
    /// Second convolution (hidden → hidden, with stride 2)
    pub conv2: Conv1d,
    /// Number of mel channels
    pub n_mels: usize,
    /// Hidden dimension
    pub d_model: usize,
}

impl ConvFrontend {
    /// Create new convolutional frontend
    #[must_use]
    pub fn new(n_mels: usize, d_model: usize) -> Self {
        Self {
            // First conv: n_mels → d_model, kernel=3, stride=1, padding=1
            conv1: Conv1d::new(n_mels, d_model, 3, 1, 1),
            // Second conv: d_model → d_model, kernel=3, stride=2, padding=1
            conv2: Conv1d::new(d_model, d_model, 3, 2, 1),
            n_mels,
            d_model,
        }
    }

    /// Forward pass through convolutional frontend
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram (n_frames x n_mels) flattened row-major
    ///
    /// # Returns
    /// Convolved features (out_frames x d_model) ready for transformer
    pub fn forward(&self, mel: &[f32]) -> WhisperResult<Vec<f32>> {
        // First conv + GELU
        let mut x = self.conv1.forward(mel)?;
        for v in &mut x {
            *v = gelu(*v);
        }

        // Second conv (with stride 2) + GELU
        let mut x = self.conv2.forward(&x)?;
        for v in &mut x {
            *v = gelu(*v);
        }

        Ok(x)
    }

    /// Get expected output sequence length for given input length
    #[must_use]
    pub fn output_length(&self, input_len: usize) -> usize {
        // First conv: same length (stride 1, padding 1)
        let after_conv1 = input_len;
        // Second conv: halved (stride 2)
        (after_conv1 + 2 - 3) / 2 + 1
    }
}

/// Layer normalization weights
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Scale parameter (gamma)
    pub weight: Vec<f32>,
    /// Shift parameter (beta)
    pub bias: Vec<f32>,
    /// Normalized dimension
    pub normalized_shape: usize,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl LayerNorm {
    /// Create new layer normalization
    #[must_use]
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            weight: vec![1.0; normalized_shape],
            bias: vec![0.0; normalized_shape],
            normalized_shape,
            eps: 1e-5,
        }
    }

    /// Apply layer normalization
    ///
    /// # Arguments
    /// * `input` - Input tensor (seq_len x normalized_shape) flattened
    ///
    /// # Returns
    /// Normalized output with same shape
    pub fn forward(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        if input.len() % self.normalized_shape != 0 {
            return Err(WhisperError::Model(
                "input size mismatch for layer norm".into(),
            ));
        }

        let seq_len = input.len() / self.normalized_shape;
        let mut output = vec![0.0_f32; input.len()];

        for s in 0..seq_len {
            let start = s * self.normalized_shape;
            let end = start + self.normalized_shape;
            let slice = &input[start..end];

            // Compute mean
            let mean: f32 = slice.iter().sum::<f32>() / self.normalized_shape as f32;

            // Compute variance
            let variance: f32 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                / self.normalized_shape as f32;

            let inv_std = 1.0 / (variance + self.eps).sqrt();

            // Normalize and apply affine transform
            for i in 0..self.normalized_shape {
                let normalized = (slice[i] - mean) * inv_std;
                output[start + i] = normalized * self.weight[i] + self.bias[i];
            }
        }

        Ok(output)
    }
}

/// Feed-forward network (FFN) in transformer block
#[derive(Debug, Clone)]
pub struct FeedForward {
    /// First linear layer (expansion)
    pub fc1: LinearWeights,
    /// Second linear layer (projection)
    pub fc2: LinearWeights,
    /// Hidden dimension (typically 4 * d_model)
    pub d_ff: usize,
    /// Model dimension
    pub d_model: usize,
}

impl FeedForward {
    /// Create new feed-forward network
    #[must_use]
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            fc1: LinearWeights::new(d_model, d_ff),
            fc2: LinearWeights::new(d_ff, d_model),
            d_ff,
            d_model,
        }
    }

    /// Forward pass with GELU activation
    ///
    /// FFN(x) = fc2(GELU(fc1(x)))
    pub fn forward(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        let seq_len = input.len() / self.d_model;

        // First linear + GELU
        let mut hidden = self.fc1.forward(input, seq_len)?;

        // Apply GELU activation
        for x in &mut hidden {
            *x = gelu(*x);
        }

        // Second linear
        self.fc2.forward(&hidden, seq_len)
    }

    /// Finalize weights for optimized SIMD matmul
    pub fn finalize_weights(&mut self) {
        self.fc1.finalize_weights();
        self.fc2.finalize_weights();
    }

    /// Check if weights have been finalized
    #[must_use]
    pub fn is_finalized(&self) -> bool {
        self.fc1.is_finalized() && self.fc2.is_finalized()
    }
}

/// GELU activation function (approximate)
///
/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
#[inline]
fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.797_884_6; // sqrt(2/π)
    let coef = 0.044_715;
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + coef * x.powi(3))).tanh())
}

/// Single transformer encoder block
#[derive(Debug, Clone)]
pub struct EncoderBlock {
    /// Self-attention layer
    pub self_attn: MultiHeadAttention,
    /// Layer norm before attention
    pub ln1: LayerNorm,
    /// Feed-forward network
    pub ffn: FeedForward,
    /// Layer norm before FFN
    pub ln2: LayerNorm,
}

impl EncoderBlock {
    /// Create new encoder block
    #[must_use]
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(n_heads, d_model),
            ln1: LayerNorm::new(d_model),
            ffn: FeedForward::new(d_model, d_ff),
            ln2: LayerNorm::new(d_model),
        }
    }

    /// Forward pass: x + Attention(LN(x)) then x + FFN(LN(x))
    pub fn forward(&self, x: &[f32]) -> WhisperResult<Vec<f32>> {
        // Pre-norm self-attention with residual
        let normed = self.ln1.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, None)?;
        let mut residual: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

        // Pre-norm FFN with residual
        let normed = self.ln2.forward(&residual)?;
        let ffn_out = self.ffn.forward(&normed)?;

        for (r, f) in residual.iter_mut().zip(ffn_out.iter()) {
            *r += f;
        }

        Ok(residual)
    }

    /// Finalize weights by caching transposed/pre-computed data
    pub fn finalize_weights(&mut self) {
        self.self_attn.finalize_weights();
        self.ffn.finalize_weights();
    }
}

/// Transformer encoder for audio features
#[derive(Debug, Clone)]
pub struct Encoder {
    /// Number of layers
    n_layers: usize,
    /// Hidden state dimension
    d_model: usize,
    /// Number of attention heads
    n_heads: usize,
    /// Number of mel channels
    n_mels: usize,
    /// Convolutional frontend
    conv_frontend: ConvFrontend,
    /// Encoder blocks
    blocks: Vec<EncoderBlock>,
    /// Final layer norm
    ln_post: LayerNorm,
    /// Positional embeddings (max_len x d_model)
    positional_embedding: Vec<f32>,
    /// Maximum sequence length
    max_len: usize,
}

impl Encoder {
    /// Create a new encoder from model configuration
    #[must_use]
    pub fn new(config: &ModelConfig) -> Self {
        let n_layers = config.n_audio_layer as usize;
        let d_model = config.n_audio_state as usize;
        let n_heads = config.n_audio_head as usize;
        let d_ff = d_model * 4;
        let max_len = config.n_audio_ctx as usize;
        let n_mels = config.n_mels as usize;

        // Create convolutional frontend
        let conv_frontend = ConvFrontend::new(n_mels, d_model);

        // Create encoder blocks
        let blocks: Vec<EncoderBlock> = (0..n_layers)
            .map(|_| EncoderBlock::new(d_model, n_heads, d_ff))
            .collect();

        // Create sinusoidal positional embeddings
        let positional_embedding = Self::create_positional_embedding(max_len, d_model);

        Self {
            n_layers,
            d_model,
            n_heads,
            n_mels,
            conv_frontend,
            blocks,
            ln_post: LayerNorm::new(d_model),
            positional_embedding,
            max_len,
        }
    }

    /// Create sinusoidal positional embeddings
    fn create_positional_embedding(max_len: usize, d_model: usize) -> Vec<f32> {
        let mut pe = vec![0.0_f32; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..d_model / 2 {
                let angle = pos as f32 / 10000_f32.powf(2.0 * i as f32 / d_model as f32);
                pe[pos * d_model + 2 * i] = angle.sin();
                pe[pos * d_model + 2 * i + 1] = angle.cos();
            }
        }

        pe
    }

    /// Forward pass through encoder
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram features (seq_len x d_model) after conv layers
    ///
    /// # Returns
    /// Encoded audio features (seq_len x d_model)
    pub fn forward(&self, mel: &[f32]) -> WhisperResult<Vec<f32>> {
        let seq_len = mel.len() / self.d_model;

        if mel.len() % self.d_model != 0 {
            return Err(WhisperError::Model("input size mismatch".into()));
        }
        if seq_len > self.max_len {
            return Err(WhisperError::Model(format!(
                "sequence length {} exceeds max {}",
                seq_len, self.max_len
            )));
        }

        // Add positional embeddings
        let mut x = mel.to_vec();
        for pos in 0..seq_len {
            for d in 0..self.d_model {
                x[pos * self.d_model + d] += self.positional_embedding[pos * self.d_model + d];
            }
        }

        // Pass through encoder blocks
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Final layer norm
        self.ln_post.forward(&x)
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

    /// Get maximum sequence length
    #[must_use]
    pub const fn max_len(&self) -> usize {
        self.max_len
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

    /// Get encoder blocks reference
    #[must_use]
    pub fn blocks(&self) -> &[EncoderBlock] {
        &self.blocks
    }

    /// Get mutable encoder blocks reference (for loading weights)
    pub fn blocks_mut(&mut self) -> &mut [EncoderBlock] {
        &mut self.blocks
    }

    /// Get layer norm reference
    #[must_use]
    pub fn ln_post(&self) -> &LayerNorm {
        &self.ln_post
    }

    /// Get mutable layer norm reference (for loading weights)
    pub fn ln_post_mut(&mut self) -> &mut LayerNorm {
        &mut self.ln_post
    }

    /// Get number of mel channels
    #[must_use]
    pub const fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Get convolutional frontend reference
    #[must_use]
    pub fn conv_frontend(&self) -> &ConvFrontend {
        &self.conv_frontend
    }

    /// Get mutable convolutional frontend reference (for loading weights)
    pub fn conv_frontend_mut(&mut self) -> &mut ConvFrontend {
        &mut self.conv_frontend
    }

    /// Forward pass from raw mel spectrogram
    ///
    /// Processes mel through convolutional frontend then transformer blocks.
    ///
    /// # Arguments
    /// * `mel` - Raw mel spectrogram (n_frames x n_mels) flattened row-major
    ///
    /// # Returns
    /// Encoded audio features (out_frames x d_model)
    ///
    /// # Errors
    /// Returns error if mel size is invalid or sequence too long
    pub fn forward_mel(&self, mel: &[f32]) -> WhisperResult<Vec<f32>> {
        // Validate mel input
        if mel.len() % self.n_mels != 0 {
            return Err(WhisperError::Model(format!(
                "mel size {} not divisible by n_mels {}",
                mel.len(),
                self.n_mels
            )));
        }

        // Process through convolutional frontend
        let conv_output = self.conv_frontend.forward(mel)?;

        // Forward through transformer blocks
        self.forward(&conv_output)
    }

    // =========================================================================
    // Batch Processing (WAPR-081)
    // =========================================================================

    /// Forward pass for a batch of mel spectrograms
    ///
    /// Processes multiple mel spectrograms in sequence (not parallelized).
    ///
    /// # Arguments
    /// * `batch` - Vector of mel spectrograms (each n_frames x n_mels)
    ///
    /// # Returns
    /// Vector of encoded features (each out_frames x d_model)
    ///
    /// # Errors
    /// Returns error if any mel processing fails
    pub fn forward_batch(&self, batch: &[Vec<f32>]) -> WhisperResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(batch.len());

        for mel in batch {
            let encoded = self.forward_mel(mel)?;
            results.push(encoded);
        }

        Ok(results)
    }

    /// Forward pass for batch with padding
    ///
    /// Processes batch and returns padded tensor suitable for batched attention.
    ///
    /// # Arguments
    /// * `batch` - Vector of mel spectrograms
    ///
    /// # Returns
    /// Tuple of (padded_features, sequence_lengths)
    /// - padded_features: (batch_size × max_seq_len × d_model) flattened
    /// - sequence_lengths: actual sequence length for each item
    ///
    /// # Errors
    /// Returns error if processing fails
    pub fn forward_batch_padded(&self, batch: &[Vec<f32>]) -> WhisperResult<BatchEncoderOutput> {
        let encoded = self.forward_batch(batch)?;

        // Find max sequence length
        let max_seq_len = encoded
            .iter()
            .map(|e| e.len() / self.d_model)
            .max()
            .unwrap_or(0);

        // Collect sequence lengths
        let seq_lengths: Vec<usize> = encoded.iter().map(|e| e.len() / self.d_model).collect();

        // Create padded tensor
        let batch_size = encoded.len();
        let total_size = batch_size * max_seq_len * self.d_model;
        let mut padded = vec![0.0_f32; total_size];

        for (batch_idx, features) in encoded.iter().enumerate() {
            let seq_len = features.len() / self.d_model;
            for t in 0..seq_len {
                for d in 0..self.d_model {
                    let src_idx = t * self.d_model + d;
                    let dst_idx = batch_idx * max_seq_len * self.d_model + t * self.d_model + d;
                    padded[dst_idx] = features[src_idx];
                }
            }
        }

        Ok(BatchEncoderOutput {
            features: padded,
            seq_lengths,
            max_seq_len,
            batch_size,
            d_model: self.d_model,
        })
    }

    /// Finalize all weights by caching transposed/pre-computed data
    ///
    /// Call this after loading weights to trade memory for speed.
    /// This pre-computes transposed weight matrices and caches them.
    pub fn finalize_weights(&mut self) {
        for block in &mut self.blocks {
            block.finalize_weights();
        }
    }
}

/// Batched encoder output with padding information
#[derive(Debug, Clone)]
pub struct BatchEncoderOutput {
    /// Padded features (batch_size × max_seq_len × d_model) flattened
    pub features: Vec<f32>,
    /// Actual sequence length for each item
    pub seq_lengths: Vec<usize>,
    /// Maximum sequence length (for padding)
    pub max_seq_len: usize,
    /// Batch size
    pub batch_size: usize,
    /// Model dimension
    pub d_model: usize,
}

impl BatchEncoderOutput {
    /// Get features for a specific batch item (unpadded)
    #[must_use]
    pub fn get(&self, batch_idx: usize) -> Option<Vec<f32>> {
        if batch_idx >= self.batch_size {
            return None;
        }

        let seq_len = self.seq_lengths[batch_idx];
        let mut features = Vec::with_capacity(seq_len * self.d_model);

        for t in 0..seq_len {
            for d in 0..self.d_model {
                let idx = batch_idx * self.max_seq_len * self.d_model + t * self.d_model + d;
                features.push(self.features[idx]);
            }
        }

        Some(features)
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.batch_size == 0
    }

    /// Get the total number of tokens (sum of sequence lengths)
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.seq_lengths.iter().sum()
    }
}

// ============================================================================
// Fused FFN (for realizar-inference feature)
// ============================================================================

/// Fused Feed-Forward Network combining LayerNorm + Linear
///
/// This optimization eliminates the intermediate tensor between LayerNorm
/// and the first linear layer of FFN.
#[cfg(feature = "realizar-inference")]
#[derive(Debug, Clone)]
pub struct FusedFFN {
    /// Combined LayerNorm + FC1 weights
    pub fused_weight: Vec<f32>,
    /// FC1 bias
    pub fc1_bias: Vec<f32>,
    /// FC2 weights
    pub fc2_weight: Vec<f32>,
    /// FC2 bias
    pub fc2_bias: Vec<f32>,
    /// Model dimension
    pub d_model: usize,
    /// Hidden dimension
    pub d_ff: usize,
}

#[cfg(feature = "realizar-inference")]
impl FusedFFN {
    /// Create a new FusedFFN
    pub fn new(d_model: usize, d_ff: usize) -> WhisperResult<Self> {
        Ok(Self {
            fused_weight: vec![0.0; d_model * d_ff],
            fc1_bias: vec![0.0; d_ff],
            fc2_weight: vec![0.0; d_ff * d_model],
            fc2_bias: vec![0.0; d_model],
            d_model,
            d_ff,
        })
    }

    /// Forward pass
    pub fn forward(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        // Simplified implementation - just returns zeros for now
        // Full implementation would fuse LayerNorm + FC1
        let seq_len = input.len() / self.d_model;
        Ok(vec![0.0; seq_len * self.d_model])
    }

    /// Set fused weights from LayerNorm and FC1
    pub fn set_fused_weights(&mut self, ln_weight: &[f32], ln_bias: &[f32], fc1_weight: &[f32]) {
        // Fuse LayerNorm scale into FC1 weights
        // w_fused[i,j] = w_fc1[i,j] * ln_weight[j]
        for i in 0..self.d_ff {
            for j in 0..self.d_model {
                self.fused_weight[i * self.d_model + j] =
                    fc1_weight[i * self.d_model + j] * ln_weight[j];
            }
        }
        // Note: ln_bias handling is more complex and omitted for simplicity
        let _ = ln_bias; // Suppress unused warning
    }

    /// Set FC1 weights and bias
    pub fn set_fc1_weights(&mut self, weight: &[f32], bias: &[f32]) {
        let len = weight.len().min(self.fused_weight.len());
        self.fused_weight[..len].copy_from_slice(&weight[..len]);
        let bias_len = bias.len().min(self.fc1_bias.len());
        self.fc1_bias[..bias_len].copy_from_slice(&bias[..bias_len]);
    }

    /// Set FC2 weights and bias
    pub fn set_fc2_weights(&mut self, weight: &[f32], bias: &[f32]) {
        let len = weight.len().min(self.fc2_weight.len());
        self.fc2_weight[..len].copy_from_slice(&weight[..len]);
        let bias_len = bias.len().min(self.fc2_bias.len());
        self.fc2_bias[..bias_len].copy_from_slice(&bias[..bias_len]);
    }

    /// Set LayerNorm weights (stored for potential fusion)
    pub fn set_norm_weights(&mut self, _weight: &[f32], _bias: &[f32]) {
        // LayerNorm weights are used during fusion with FC1
        // For now, we just store them implicitly in the fused weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Layer Norm Tests
    // =========================================================================

    #[test]
    fn test_layer_norm_new() {
        let ln = LayerNorm::new(64);
        assert_eq!(ln.normalized_shape, 64);
        assert_eq!(ln.weight.len(), 64);
        assert_eq!(ln.bias.len(), 64);
    }

    #[test]
    fn test_layer_norm_forward() {
        let ln = LayerNorm::new(4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = ln.forward(&input).expect("forward should succeed");

        assert_eq!(output.len(), 4);

        // After normalization, mean should be ~0 and std ~1
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
    }

    #[test]
    fn test_layer_norm_batch() {
        let ln = LayerNorm::new(4);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 sequences
        let output = ln.forward(&input).expect("forward should succeed");

        assert_eq!(output.len(), 8);
    }

    // =========================================================================
    // Feed-Forward Tests
    // =========================================================================

    #[test]
    fn test_feed_forward_new() {
        let ffn = FeedForward::new(64, 256);
        assert_eq!(ffn.d_model, 64);
        assert_eq!(ffn.d_ff, 256);
    }

    #[test]
    fn test_feed_forward_forward() {
        let ffn = FeedForward::new(8, 32);
        let input = vec![0.0_f32; 16]; // seq_len=2, d_model=8
        let output = ffn.forward(&input).expect("forward should succeed");

        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_gelu_at_zero() {
        let result = gelu(0.0);
        assert!(result.abs() < 1e-6, "GELU(0) should be ~0");
    }

    #[test]
    fn test_gelu_positive() {
        let result = gelu(1.0);
        assert!(result > 0.0, "GELU(1) should be positive");
        assert!(result < 1.0, "GELU(1) should be less than 1");
    }

    #[test]
    fn test_gelu_negative() {
        let result = gelu(-1.0);
        assert!(result < 0.0, "GELU(-1) should be negative");
        assert!(result > -0.2, "GELU(-1) should be > -0.2");
    }

    // =========================================================================
    // Encoder Block Tests
    // =========================================================================

    #[test]
    fn test_encoder_block_new() {
        let block = EncoderBlock::new(64, 4, 256);
        assert_eq!(block.self_attn.d_model(), 64);
        assert_eq!(block.ffn.d_model, 64);
    }

    #[test]
    fn test_encoder_block_forward() {
        let block = EncoderBlock::new(8, 2, 32);
        let input = vec![0.1_f32; 16]; // seq_len=2, d_model=8
        let output = block.forward(&input).expect("forward should succeed");

        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_encoder_block_residual() {
        let block = EncoderBlock::new(8, 2, 32);
        let input = vec![1.0_f32; 8]; // seq_len=1, d_model=8

        // With zero weights, residual should dominate
        let output = block.forward(&input).expect("forward should succeed");

        // Output should be close to input (residual connection)
        // Not exact due to layer norm
        assert_eq!(output.len(), 8);
    }

    // =========================================================================
    // Encoder Tests
    // =========================================================================

    #[test]
    fn test_encoder_new() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        assert_eq!(encoder.n_layers(), 4);
        assert_eq!(encoder.d_model(), 384);
        assert_eq!(encoder.n_heads(), 6);
    }

    #[test]
    fn test_encoder_positional_embedding_shape() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        assert_eq!(
            encoder.positional_embedding.len(),
            encoder.max_len * encoder.d_model
        );
    }

    #[test]
    fn test_encoder_forward() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        // Input: seq_len=10, d_model=384
        let input = vec![0.0_f32; 10 * 384];
        let output = encoder.forward(&input).expect("forward should succeed");

        assert_eq!(output.len(), 10 * 384);
    }

    #[test]
    fn test_encoder_forward_size_mismatch() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        let input = vec![0.0_f32; 100]; // Not divisible by d_model
        let result = encoder.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_encoder_forward_too_long() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        // Sequence longer than max_len (1500)
        let input = vec![0.0_f32; 2000 * 384];
        let result = encoder.forward(&input);
        assert!(result.is_err());
    }

    // =========================================================================
    // Positional Embedding Tests
    // =========================================================================

    #[test]
    fn test_positional_embedding_sinusoidal() {
        let pe = Encoder::create_positional_embedding(100, 64);

        // First position should have sin(0)=0 and cos(0)=1 for first two dimensions
        assert!(pe[0].abs() < 1e-5, "sin(0) should be 0");
        assert!((pe[1] - 1.0).abs() < 1e-5, "cos(0) should be 1");
    }

    #[test]
    fn test_positional_embedding_different_positions() {
        let pe = Encoder::create_positional_embedding(100, 64);

        // Different positions should have different embeddings
        let pos0: Vec<f32> = pe[0..64].to_vec();
        let pos1: Vec<f32> = pe[64..128].to_vec();

        let diff: f32 = pos0
            .iter()
            .zip(pos1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.1,
            "Different positions should have different embeddings"
        );
    }

    // =========================================================================
    // Encoder Accessor Tests
    // =========================================================================

    #[test]
    fn test_encoder_positional_embedding() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        let pe = encoder.positional_embedding();
        assert_eq!(pe.len(), encoder.max_len() * encoder.d_model());
    }

    #[test]
    fn test_encoder_positional_embedding_mut() {
        let config = ModelConfig::tiny();
        let mut encoder = Encoder::new(&config);

        // Modify positional embedding
        encoder.positional_embedding_mut()[0] = 100.0;
        assert!((encoder.positional_embedding()[0] - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_encoder_blocks() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        assert_eq!(encoder.blocks().len(), 4);
    }

    #[test]
    fn test_encoder_blocks_mut() {
        let config = ModelConfig::tiny();
        let mut encoder = Encoder::new(&config);

        // Should be mutable
        let blocks = encoder.blocks_mut();
        assert_eq!(blocks.len(), 4);
    }

    #[test]
    fn test_encoder_ln_post() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        let ln = encoder.ln_post();
        assert_eq!(ln.normalized_shape, encoder.d_model());
    }

    #[test]
    fn test_encoder_ln_post_mut() {
        let config = ModelConfig::tiny();
        let mut encoder = Encoder::new(&config);

        // Modify ln_post weight
        encoder.ln_post_mut().weight[0] = 2.0;
        assert!((encoder.ln_post().weight[0] - 2.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Conv1d Tests
    // =========================================================================

    #[test]
    fn test_conv1d_new() {
        let conv = Conv1d::new(80, 384, 3, 1, 1);
        assert_eq!(conv.in_channels, 80);
        assert_eq!(conv.out_channels, 384);
        assert_eq!(conv.kernel_size, 3);
        assert_eq!(conv.stride, 1);
        assert_eq!(conv.padding, 1);
    }

    #[test]
    fn test_conv1d_weight_size() {
        let conv = Conv1d::new(80, 384, 3, 1, 1);
        assert_eq!(conv.weight.len(), 384 * 80 * 3);
        assert_eq!(conv.bias.len(), 384);
    }

    #[test]
    fn test_conv1d_forward_stride1() {
        let conv = Conv1d::new(4, 8, 3, 1, 1);
        let input = vec![0.0_f32; 10 * 4]; // 10 frames, 4 channels

        let output = conv.forward(&input).expect("forward should succeed");

        // With stride=1 and padding=1, output length equals input length
        assert_eq!(output.len(), 10 * 8);
    }

    #[test]
    fn test_conv1d_forward_stride2() {
        let conv = Conv1d::new(4, 8, 3, 2, 1);
        let input = vec![0.0_f32; 10 * 4]; // 10 frames, 4 channels

        let output = conv.forward(&input).expect("forward should succeed");

        // With stride=2, output length is roughly half
        let expected_len = (10 + 2 * 1 - 3) / 2 + 1;
        assert_eq!(output.len(), expected_len * 8);
    }

    #[test]
    fn test_conv1d_forward_size_mismatch() {
        let conv = Conv1d::new(4, 8, 3, 1, 1);
        let input = vec![0.0_f32; 13]; // Not divisible by in_channels

        let result = conv.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv1d_weight_mut() {
        let mut conv = Conv1d::new(4, 8, 3, 1, 1);
        conv.weight_mut()[0] = 1.0;
        assert!((conv.weight[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_conv1d_bias_mut() {
        let mut conv = Conv1d::new(4, 8, 3, 1, 1);
        conv.bias_mut()[0] = 2.0;
        assert!((conv.bias[0] - 2.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // ConvFrontend Tests
    // =========================================================================

    #[test]
    fn test_conv_frontend_new() {
        let frontend = ConvFrontend::new(80, 384);
        assert_eq!(frontend.n_mels, 80);
        assert_eq!(frontend.d_model, 384);
    }

    #[test]
    fn test_conv_frontend_conv1_params() {
        let frontend = ConvFrontend::new(80, 384);
        assert_eq!(frontend.conv1.in_channels, 80);
        assert_eq!(frontend.conv1.out_channels, 384);
        assert_eq!(frontend.conv1.stride, 1);
    }

    #[test]
    fn test_conv_frontend_conv2_params() {
        let frontend = ConvFrontend::new(80, 384);
        assert_eq!(frontend.conv2.in_channels, 384);
        assert_eq!(frontend.conv2.out_channels, 384);
        assert_eq!(frontend.conv2.stride, 2);
    }

    #[test]
    fn test_conv_frontend_forward() {
        let frontend = ConvFrontend::new(4, 8);
        let input = vec![0.0_f32; 100 * 4]; // 100 frames, 4 mel channels

        let output = frontend.forward(&input).expect("forward should succeed");

        // Output length should be about half (due to stride 2 in conv2)
        let expected_frames = frontend.output_length(100);
        assert_eq!(output.len(), expected_frames * 8);
    }

    #[test]
    fn test_conv_frontend_output_length() {
        let frontend = ConvFrontend::new(80, 384);

        // 3000 frames -> ~1500 after stride 2
        let out_len = frontend.output_length(3000);
        assert_eq!(out_len, 1500);
    }

    // =========================================================================
    // Encoder with ConvFrontend Tests
    // =========================================================================

    #[test]
    fn test_encoder_n_mels() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        assert_eq!(encoder.n_mels(), 80);
    }

    #[test]
    fn test_encoder_conv_frontend() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        let frontend = encoder.conv_frontend();
        assert_eq!(frontend.n_mels, 80);
        assert_eq!(frontend.d_model, 384);
    }

    #[test]
    fn test_encoder_conv_frontend_mut() {
        let config = ModelConfig::tiny();
        let mut encoder = Encoder::new(&config);

        encoder.conv_frontend_mut().conv1.bias_mut()[0] = 5.0;
        assert!((encoder.conv_frontend().conv1.bias[0] - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_encoder_forward_mel() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        // Mel spectrogram: n_frames x n_mels
        // For Whisper tiny: 3000 frames (30s audio), 80 mels
        let mel = vec![0.0_f32; 100 * 80]; // 100 frames for faster test

        let output = encoder
            .forward_mel(&mel)
            .expect("forward_mel should succeed");

        // Output should be (out_frames x d_model)
        let expected_frames = encoder.conv_frontend().output_length(100);
        assert_eq!(output.len(), expected_frames * encoder.d_model());
    }

    #[test]
    fn test_encoder_forward_mel_size_mismatch() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        let mel = vec![0.0_f32; 123]; // Not divisible by n_mels (80)
        let result = encoder.forward_mel(&mel);
        assert!(result.is_err());
    }

    // =========================================================================
    // Batch Processing Tests (WAPR-081)
    // =========================================================================

    #[test]
    fn test_encoder_forward_batch_empty() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        let batch: Vec<Vec<f32>> = Vec::new();
        let results = encoder.forward_batch(&batch).expect("forward_batch");

        assert!(results.is_empty());
    }

    #[test]
    fn test_encoder_forward_batch_single() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        let mel = vec![0.0_f32; 100 * 80]; // 100 frames, 80 mels
        let batch = vec![mel];

        let results = encoder.forward_batch(&batch).expect("forward_batch");

        assert_eq!(results.len(), 1);
        let expected_frames = encoder.conv_frontend().output_length(100);
        assert_eq!(results[0].len(), expected_frames * encoder.d_model());
    }

    #[test]
    fn test_encoder_forward_batch_multiple() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        let mel1 = vec![0.0_f32; 100 * 80]; // 100 frames
        let mel2 = vec![0.1_f32; 50 * 80]; // 50 frames
        let batch = vec![mel1, mel2];

        let results = encoder.forward_batch(&batch).expect("forward_batch");

        assert_eq!(results.len(), 2);

        let expected_frames1 = encoder.conv_frontend().output_length(100);
        let expected_frames2 = encoder.conv_frontend().output_length(50);

        assert_eq!(results[0].len(), expected_frames1 * encoder.d_model());
        assert_eq!(results[1].len(), expected_frames2 * encoder.d_model());
    }

    #[test]
    fn test_encoder_forward_batch_padded_empty() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        let batch: Vec<Vec<f32>> = Vec::new();
        let output = encoder
            .forward_batch_padded(&batch)
            .expect("forward_batch_padded");

        assert!(output.is_empty());
        assert_eq!(output.batch_size, 0);
    }

    #[test]
    fn test_encoder_forward_batch_padded_single() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        let mel = vec![0.0_f32; 100 * 80];
        let batch = vec![mel];

        let output = encoder
            .forward_batch_padded(&batch)
            .expect("forward_batch_padded");

        assert_eq!(output.batch_size, 1);
        assert_eq!(output.seq_lengths.len(), 1);
    }

    #[test]
    fn test_encoder_forward_batch_padded_multiple() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        let mel1 = vec![0.0_f32; 100 * 80]; // 100 frames
        let mel2 = vec![0.0_f32; 50 * 80]; // 50 frames
        let batch = vec![mel1, mel2];

        let output = encoder
            .forward_batch_padded(&batch)
            .expect("forward_batch_padded");

        assert_eq!(output.batch_size, 2);
        assert_eq!(output.seq_lengths.len(), 2);

        // First should have more frames than second
        assert!(output.seq_lengths[0] >= output.seq_lengths[1]);

        // max_seq_len should equal longest sequence
        assert_eq!(output.max_seq_len, output.seq_lengths[0]);
    }

    #[test]
    fn test_batch_encoder_output_get() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        let mel1 = vec![0.1_f32; 100 * 80];
        let mel2 = vec![0.2_f32; 50 * 80];
        let batch = vec![mel1, mel2];

        let output = encoder
            .forward_batch_padded(&batch)
            .expect("forward_batch_padded");

        // Should be able to get both items
        assert!(output.get(0).is_some());
        assert!(output.get(1).is_some());
        assert!(output.get(2).is_none());
    }

    #[test]
    fn test_batch_encoder_output_total_tokens() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);

        let mel1 = vec![0.0_f32; 100 * 80];
        let mel2 = vec![0.0_f32; 50 * 80];
        let batch = vec![mel1, mel2];

        let output = encoder
            .forward_batch_padded(&batch)
            .expect("forward_batch_padded");

        let total = output.total_tokens();
        let expected = output.seq_lengths.iter().sum::<usize>();

        assert_eq!(total, expected);
    }
}
