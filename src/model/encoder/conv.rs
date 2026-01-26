//! Convolutional layers for Whisper encoder
//!
//! 1D convolutions for processing mel spectrograms.

use crate::error::{WhisperError, WhisperResult};

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

    /// Forward pass using SIMD-accelerated im2col + matmul
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
        let patch_size = self.kernel_size * self.in_channels;

        // im2col: Extract patches into matrix (out_seq_len × patch_size)
        let mut patches = vec![0.0_f32; out_seq_len * patch_size];

        for out_pos in 0..out_seq_len {
            let in_start = out_pos as isize * self.stride as isize - self.padding as isize;
            let patch_row = out_pos * patch_size;

            for k in 0..self.kernel_size {
                let in_pos = in_start + k as isize;
                let patch_col = k * self.in_channels;

                if in_pos >= 0 && (in_pos as usize) < seq_len {
                    let input_row = (in_pos as usize) * self.in_channels;
                    patches[patch_row + patch_col..patch_row + patch_col + self.in_channels]
                        .copy_from_slice(&input[input_row..input_row + self.in_channels]);
                }
            }
        }

        // Reshape weights
        let mut weight_reshaped = vec![0.0_f32; self.out_channels * patch_size];
        for out_ch in 0..self.out_channels {
            for k in 0..self.kernel_size {
                for in_ch in 0..self.in_channels {
                    let old_idx =
                        out_ch * self.in_channels * self.kernel_size + in_ch * self.kernel_size + k;
                    let new_idx = out_ch * patch_size + k * self.in_channels + in_ch;
                    weight_reshaped[new_idx] = self.weight[old_idx];
                }
            }
        }

        // Transpose weights
        let weight_t = crate::simd::transpose(&weight_reshaped, self.out_channels, patch_size);

        // SIMD matmul
        let mut output = crate::simd::matmul(
            &patches,
            &weight_t,
            out_seq_len,
            patch_size,
            self.out_channels,
        );

        // Add bias
        crate::simd::broadcast_add_inplace(&mut output, &self.bias, out_seq_len, self.out_channels);

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
            conv1: Conv1d::new(n_mels, d_model, 3, 1, 1),
            conv2: Conv1d::new(d_model, d_model, 3, 2, 1),
            n_mels,
            d_model,
        }
    }

    /// Forward pass through convolutional frontend
    pub fn forward(&self, mel: &[f32]) -> WhisperResult<Vec<f32>> {
        // First conv + GELU
        let mut x = self.conv1.forward(mel)?;
        for v in &mut x {
            *v = super::layers::gelu(*v);
        }

        // Second conv (with stride 2) + GELU
        let mut x = self.conv2.forward(&x)?;
        for v in &mut x {
            *v = super::layers::gelu(*v);
        }

        Ok(x)
    }

    /// Get expected output sequence length for given input length
    #[must_use]
    pub fn output_length(&self, input_len: usize) -> usize {
        let after_conv1 = input_len;
        (after_conv1 + 2 - 3) / 2 + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let input = vec![0.0_f32; 10 * 4];
        let output = conv.forward(&input).expect("forward should succeed");
        assert_eq!(output.len(), 10 * 8);
    }

    #[test]
    fn test_conv1d_forward_stride2() {
        let conv = Conv1d::new(4, 8, 3, 2, 1);
        let input = vec![0.0_f32; 10 * 4];
        let output = conv.forward(&input).expect("forward should succeed");
        let expected_len = (10 + 2 * 1 - 3) / 2 + 1;
        assert_eq!(output.len(), expected_len * 8);
    }

    #[test]
    fn test_conv1d_forward_size_mismatch() {
        let conv = Conv1d::new(4, 8, 3, 1, 1);
        let input = vec![0.0_f32; 13];
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
        let input = vec![0.0_f32; 100 * 4];
        let output = frontend.forward(&input).expect("forward should succeed");
        let expected_frames = frontend.output_length(100);
        assert_eq!(output.len(), expected_frames * 8);
    }

    #[test]
    fn test_conv_frontend_output_length() {
        let frontend = ConvFrontend::new(80, 384);
        let out_len = frontend.output_length(3000);
        assert_eq!(out_len, 1500);
    }
}
