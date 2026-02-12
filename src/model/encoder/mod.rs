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

mod block;
mod conv;
mod layers;

pub use block::EncoderBlock;
pub use conv::{Conv1d, ConvFrontend};
pub use layers::{FeedForward, LayerNorm};

use super::ModelConfig;
use crate::error::{WhisperError, WhisperResult};

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

        let conv_frontend = ConvFrontend::new(n_mels, d_model);

        let blocks: Vec<EncoderBlock> = (0..n_layers)
            .map(|_| EncoderBlock::new(d_model, n_heads, d_ff))
            .collect();

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

    /// Get mutable positional embedding reference
    pub fn positional_embedding_mut(&mut self) -> &mut [f32] {
        &mut self.positional_embedding
    }

    /// Get encoder blocks reference
    #[must_use]
    pub fn blocks(&self) -> &[EncoderBlock] {
        &self.blocks
    }

    /// Get mutable encoder blocks reference
    pub fn blocks_mut(&mut self) -> &mut [EncoderBlock] {
        &mut self.blocks
    }

    /// Get layer norm reference
    #[must_use]
    pub fn ln_post(&self) -> &LayerNorm {
        &self.ln_post
    }

    /// Get mutable layer norm reference
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

    /// Get mutable convolutional frontend reference
    pub fn conv_frontend_mut(&mut self) -> &mut ConvFrontend {
        &mut self.conv_frontend
    }

    /// Forward pass from raw mel spectrogram
    pub fn forward_mel(&self, mel: &[f32]) -> WhisperResult<Vec<f32>> {
        if mel.len() % self.n_mels != 0 {
            return Err(WhisperError::Model(format!(
                "mel size {} not divisible by n_mels {}",
                mel.len(),
                self.n_mels
            )));
        }

        let conv_output = self.conv_frontend.forward(mel)?;
        self.forward(&conv_output)
    }

    /// Forward pass for a batch of mel spectrograms
    pub fn forward_batch(&self, batch: &[Vec<f32>]) -> WhisperResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(batch.len());

        for mel in batch {
            let encoded = self.forward_mel(mel)?;
            results.push(encoded);
        }

        Ok(results)
    }

    /// Forward pass for batch with padding
    pub fn forward_batch_padded(&self, batch: &[Vec<f32>]) -> WhisperResult<BatchEncoderOutput> {
        let encoded = self.forward_batch(batch)?;

        let max_seq_len = encoded
            .iter()
            .map(|e| e.len() / self.d_model)
            .max()
            .unwrap_or(0);

        let seq_lengths: Vec<usize> = encoded.iter().map(|e| e.len() / self.d_model).collect();

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

    /// Get the total number of tokens
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.seq_lengths.iter().sum()
    }
}

// Fused FFN for realizar-inference feature
#[cfg(feature = "realizar-inference")]
pub use fused::FusedFFN;

#[cfg(feature = "realizar-inference")]
mod fused {
    use crate::error::WhisperResult;

    /// Fused Feed-Forward Network
    ///
    /// Combines layer normalization with the first linear layer
    /// for improved performance when using fused kernels.
    #[derive(Debug, Clone)]
    pub struct FusedFFN {
        /// Fused weight matrix (LN + FC1)
        pub fused_weight: Vec<f32>,
        /// First linear layer bias
        pub fc1_bias: Vec<f32>,
        /// Second linear layer weight
        pub fc2_weight: Vec<f32>,
        /// Second linear layer bias
        pub fc2_bias: Vec<f32>,
        /// Model dimension
        pub d_model: usize,
        /// Feed-forward hidden dimension
        pub d_ff: usize,
    }

    impl FusedFFN {
        /// Create a new fused FFN layer
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

        /// Forward pass through fused FFN
        ///
        /// Computes: fc2(GELU(fused_weight @ input + fc1_bias)) + fc2_bias
        /// where fused_weight = fc1_weight * ln_weight (pre-fused)
        pub fn forward(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
            let seq_len = input.len() / self.d_model;
            let mut output = vec![0.0f32; seq_len * self.d_model];

            for s in 0..seq_len {
                let x = &input[s * self.d_model..(s + 1) * self.d_model];

                // FC1: fused_weight (d_ff x d_model) @ x (d_model) + fc1_bias -> hidden (d_ff)
                let mut hidden = vec![0.0f32; self.d_ff];
                for (i, h) in hidden.iter_mut().enumerate() {
                    let mut sum = self.fc1_bias[i];
                    for (j, &xj) in x.iter().enumerate() {
                        sum += self.fused_weight[i * self.d_model + j] * xj;
                    }
                    // GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    let g = sum
                        * 0.5
                        * (1.0 + (0.797_884_6_f32 * (sum + 0.044715 * sum * sum * sum)).tanh());
                    *h = g;
                }

                // FC2: fc2_weight (d_model x d_ff) @ hidden (d_ff) + fc2_bias -> out (d_model)
                let out = &mut output[s * self.d_model..(s + 1) * self.d_model];
                for (i, o) in out.iter_mut().enumerate() {
                    let mut sum = self.fc2_bias[i];
                    for (j, &hj) in hidden.iter().enumerate() {
                        sum += self.fc2_weight[i * self.d_ff + j] * hj;
                    }
                    *o = sum;
                }
            }

            Ok(output)
        }

        /// Set fused weights combining layer norm and FC1
        pub fn set_fused_weights(
            &mut self,
            ln_weight: &[f32],
            ln_bias: &[f32],
            fc1_weight: &[f32],
        ) {
            for i in 0..self.d_ff {
                for j in 0..self.d_model {
                    self.fused_weight[i * self.d_model + j] =
                        fc1_weight[i * self.d_model + j] * ln_weight[j];
                }
            }
            let _ = ln_bias;
        }

        /// Set first linear layer weights and bias
        pub fn set_fc1_weights(&mut self, weight: &[f32], bias: &[f32]) {
            let len = weight.len().min(self.fused_weight.len());
            self.fused_weight[..len].copy_from_slice(&weight[..len]);
            let bias_len = bias.len().min(self.fc1_bias.len());
            self.fc1_bias[..bias_len].copy_from_slice(&bias[..bias_len]);
        }

        /// Set second linear layer weights and bias
        pub fn set_fc2_weights(&mut self, weight: &[f32], bias: &[f32]) {
            let len = weight.len().min(self.fc2_weight.len());
            self.fc2_weight[..len].copy_from_slice(&weight[..len]);
            let bias_len = bias.len().min(self.fc2_bias.len());
            self.fc2_bias[..bias_len].copy_from_slice(&bias[..bias_len]);
        }

        /// Set layer normalization weights (used in fusion)
        pub fn set_norm_weights(&mut self, _weight: &[f32], _bias: &[f32]) {
            // LayerNorm weights are used during fusion with FC1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let input = vec![0.0_f32; 10 * 384];
        let output = encoder.forward(&input).expect("forward should succeed");
        assert_eq!(output.len(), 10 * 384);
    }

    #[test]
    fn test_encoder_forward_size_mismatch() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let input = vec![0.0_f32; 100];
        let result = encoder.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_encoder_forward_too_long() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let input = vec![0.0_f32; 2000 * 384];
        let result = encoder.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_positional_embedding_sinusoidal() {
        let pe = Encoder::create_positional_embedding(100, 64);
        assert!(pe[0].abs() < 1e-5, "sin(0) should be 0");
        assert!((pe[1] - 1.0).abs() < 1e-5, "cos(0) should be 1");
    }

    #[test]
    fn test_positional_embedding_different_positions() {
        let pe = Encoder::create_positional_embedding(100, 64);
        let pos0: Vec<f32> = pe[0..64].to_vec();
        let pos1: Vec<f32> = pe[64..128].to_vec();

        let diff: f32 = pos0
            .iter()
            .zip(pos1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1);
    }

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
        encoder.ln_post_mut().weight[0] = 2.0;
        assert!((encoder.ln_post().weight[0] - 2.0).abs() < f32::EPSILON);
    }

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
        let mel = vec![0.0_f32; 100 * 80];
        let output = encoder.forward_mel(&mel).expect("forward_mel");
        let expected_frames = encoder.conv_frontend().output_length(100);
        assert_eq!(output.len(), expected_frames * encoder.d_model());
    }

    #[test]
    fn test_encoder_forward_mel_size_mismatch() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let mel = vec![0.0_f32; 123];
        let result = encoder.forward_mel(&mel);
        assert!(result.is_err());
    }

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
        let mel = vec![0.0_f32; 100 * 80];
        let batch = vec![mel];
        let results = encoder.forward_batch(&batch).expect("forward_batch");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_encoder_forward_batch_padded_empty() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let batch: Vec<Vec<f32>> = Vec::new();
        let output = encoder.forward_batch_padded(&batch).expect("padded");
        assert!(output.is_empty());
        assert_eq!(output.batch_size, 0);
    }

    #[test]
    fn test_batch_encoder_output_get() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let mel1 = vec![0.1_f32; 100 * 80];
        let mel2 = vec![0.2_f32; 50 * 80];
        let batch = vec![mel1, mel2];

        let output = encoder.forward_batch_padded(&batch).expect("padded");
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

        let output = encoder.forward_batch_padded(&batch).expect("padded");
        let total = output.total_tokens();
        let expected = output.seq_lengths.iter().sum::<usize>();
        assert_eq!(total, expected);
    }
}
