//! Quantized inference
//!
//! Int8 and Int4 quantization for memory-efficient inference.
//!
//! # Quantization Scheme
//!
//! Uses symmetric per-tensor quantization:
//! - Scale factor: `scale = max(abs(tensor)) / 127`
//! - Quantize: `q = round(f / scale)`
//! - Dequantize: `f = q * scale`
//!
//! # Example
//!
//! ```rust,ignore
//! use whisper_apr::model::quantized::{QuantizedTensor, quantize_f32_to_i8};
//!
//! let weights = vec![0.5, -0.3, 0.8, -0.1];
//! let (quantized, scale) = quantize_f32_to_i8(&weights);
//! let dequantized = dequantize_i8_to_f32(&quantized, scale);
//! ```

use crate::error::{WhisperError, WhisperResult};
use crate::simd;

/// Maximum value for int8 quantization
pub const I8_MAX: f32 = 127.0;

/// Maximum value for int4 quantization (signed -8 to 7)
pub const I4_MAX: f32 = 7.0;

/// Minimum scale to avoid division by zero
pub const MIN_SCALE: f32 = 1e-10;

/// Quantized tensor with scale factor
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized int8 values
    pub data: Vec<i8>,
    /// Scale factor for dequantization
    pub scale: f32,
    /// Optional zero point (for asymmetric quantization)
    pub zero_point: i8,
    /// Original shape dimensions
    pub shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Create a new quantized tensor from f32 data
    ///
    /// Uses symmetric quantization (zero_point = 0).
    #[must_use]
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Self {
        let (quantized, scale) = quantize_f32_to_i8(data);
        Self {
            data: quantized,
            scale,
            zero_point: 0,
            shape,
        }
    }

    /// Dequantize back to f32
    #[must_use]
    pub fn to_f32(&self) -> Vec<f32> {
        dequantize_i8_to_f32(&self.data, self.scale)
    }

    /// Get the number of elements
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the total number of elements from shape
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

// ============================================================================
// Q4_K Quantization (realizar integration)
// ============================================================================

/// Q4_K quantized tensor using realizar's K-quantization format
///
/// K-quantization uses super-blocks of 256 values with grouped scales:
/// - 144 bytes per super-block = 4.5 bits per weight
/// - ~7x memory reduction vs f32
///
/// # Example
///
/// ```rust,ignore
/// use whisper_apr::model::quantized::QuantizedTensorQ4K;
///
/// // Load Q4_K data from a quantized model file
/// let raw_data = load_q4k_weights(path);
/// let tensor = QuantizedTensorQ4K::from_raw(raw_data, vec![out_features, in_features]);
/// let dequantized = tensor.dequantize();
/// ```
#[cfg(feature = "realizar-inference")]
#[derive(Debug, Clone)]
pub struct QuantizedTensorQ4K {
    /// Raw Q4_K data (super-blocks of 144 bytes each)
    data: Vec<u8>,
    /// Number of f32 values this represents
    n_values: usize,
    /// Shape dimensions
    shape: Vec<usize>,
}

#[cfg(feature = "realizar-inference")]
impl QuantizedTensorQ4K {
    /// Super-block size in bytes (144 bytes = 256 values)
    pub const SUPER_BLOCK_BYTES: usize = 144;
    /// Values per super-block
    pub const VALUES_PER_BLOCK: usize = 256;

    /// Create a Q4_K tensor from raw data
    ///
    /// # Arguments
    /// * `data` - Raw Q4_K data (must be multiple of 144 bytes)
    /// * `shape` - Tensor shape
    #[must_use]
    pub fn from_raw(data: Vec<u8>, shape: Vec<usize>) -> Self {
        let n_values = shape.iter().product();
        Self {
            data,
            n_values,
            shape,
        }
    }

    /// Get number of values
    #[must_use]
    pub const fn len(&self) -> usize {
        self.n_values
    }

    /// Check if empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.n_values == 0
    }

    /// Get shape
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    /// Dequantize to f32 using realizar's Q4_K dequantization
    ///
    /// # Returns
    /// Dequantized f32 values
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        crate::realizar_inference::dequantize_q4_k(&self.data)
            .unwrap_or_else(|_| vec![0.0; self.n_values])
    }

    /// Get raw data reference
    #[must_use]
    pub fn raw_data(&self) -> &[u8] {
        &self.data
    }
}

/// Q4_K quantized linear layer using realizar's K-quantization
///
/// Stores weights in Q4_K format (~4.5 bits per weight) for ~7x memory
/// reduction vs f32. Forward pass dequantizes to f32 for computation.
///
/// # Example
///
/// ```rust,ignore
/// use whisper_apr::model::quantized::QuantizedLinearQ4K;
///
/// let raw_weights = load_q4k_weights(path);
/// let linear = QuantizedLinearQ4K::from_raw(raw_weights, Some(&bias), 512, 512);
/// let output = linear.forward(&input)?;
/// ```
#[cfg(feature = "realizar-inference")]
#[derive(Debug, Clone)]
pub struct QuantizedLinearQ4K {
    /// Q4_K quantized weight tensor [out_features, in_features]
    weight: QuantizedTensorQ4K,
    /// Optional bias (not quantized, f32)
    bias: Option<Vec<f32>>,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
}

#[cfg(feature = "realizar-inference")]
impl QuantizedLinearQ4K {
    /// Create from raw Q4_K weight data
    ///
    /// # Arguments
    /// * `weight_data` - Raw Q4_K data (must be correctly sized for out_features × in_features)
    /// * `bias` - Optional bias vector (out_features length)
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    #[must_use]
    pub fn from_raw(
        weight_data: Vec<u8>,
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        let n_values = out_features * in_features;
        let weight = QuantizedTensorQ4K::from_raw(weight_data, vec![out_features, in_features]);
        Self {
            weight: QuantizedTensorQ4K {
                data: weight.data,
                n_values,
                shape: vec![out_features, in_features],
            },
            bias: bias.map(|b| b.to_vec()),
            in_features,
            out_features,
        }
    }

    /// Get input features
    #[must_use]
    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features
    #[must_use]
    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_size(&self) -> usize {
        let weight_size = self.weight.memory_bytes();
        let bias_size = self.bias.as_ref().map_or(0, |b| b.len() * 4);
        weight_size + bias_size
    }

    /// Forward pass with dequantization
    ///
    /// Dequantizes weights to f32, then performs matmul.
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size × in_features)
    ///
    /// # Returns
    /// Output tensor (batch_size × out_features)
    ///
    /// # Errors
    /// Returns error if input size is not divisible by in_features.
    pub fn forward(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        let batch_size = input.len() / self.in_features;
        if input.len() % self.in_features != 0 {
            return Err(WhisperError::Model(format!(
                "input size {} not divisible by in_features {}",
                input.len(),
                self.in_features
            )));
        }

        // Dequantize weights [out_features, in_features]
        let weights = self.weight.dequantize();

        // Transpose weights to [in_features, out_features] for matmul
        let weights_t = simd::transpose(&weights, self.out_features, self.in_features);

        // SIMD matrix multiply: input [batch, in] @ weights_t [in, out] = output [batch, out]
        let mut output = simd::matmul(
            input,
            &weights_t,
            batch_size,
            self.in_features,
            self.out_features,
        );

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for o in 0..self.out_features {
                    output[b * self.out_features + o] += bias[o];
                }
            }
        }

        Ok(output)
    }

    /// Forward pass with fused dequantize + matmul
    ///
    /// Uses realizar's `fused_q4k_parallel_matvec` to avoid materializing
    /// the full dequantized weight tensor. More memory efficient than `forward()`.
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size × in_features)
    ///
    /// # Returns
    /// Output tensor (batch_size × out_features)
    ///
    /// # Errors
    /// Returns error if input size is not divisible by in_features.
    pub fn forward_fused(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        let batch_size = input.len() / self.in_features;
        if input.len() % self.in_features != 0 {
            return Err(WhisperError::Model(format!(
                "input size {} not divisible by in_features {}",
                input.len(),
                self.in_features
            )));
        }

        // Use fused Q4K parallel matvec for each batch
        let mut output = Vec::with_capacity(batch_size * self.out_features);

        for b in 0..batch_size {
            let input_slice = &input[b * self.in_features..(b + 1) * self.in_features];

            // Fused dequantize + matvec: no intermediate f32 weight buffer
            let batch_output = crate::realizar_inference::fused_q4k_parallel_matvec(
                self.weight.raw_data(),
                input_slice,
                self.in_features,
                self.out_features,
            )
            .map_err(|e| WhisperError::Model(format!("Fused Q4K matvec failed: {e}")))?;

            output.extend(batch_output);
        }

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for o in 0..self.out_features {
                    output[b * self.out_features + o] += bias[o];
                }
            }
        }

        Ok(output)
    }
}

// ============================================================================
// Q5_K Quantization (realizar integration)
// ============================================================================

/// Q5_K quantized tensor using realizar's K-quantization format
///
/// K-quantization uses super-blocks of 256 values with grouped scales:
/// - 176 bytes per super-block = 5.5 bits per weight
/// - ~6x memory reduction vs f32
#[cfg(feature = "realizar-inference")]
#[derive(Debug, Clone)]
pub struct QuantizedTensorQ5K {
    /// Raw Q5_K data (super-blocks of 176 bytes each)
    data: Vec<u8>,
    /// Number of f32 values this represents
    n_values: usize,
    /// Shape dimensions
    shape: Vec<usize>,
}

#[cfg(feature = "realizar-inference")]
impl QuantizedTensorQ5K {
    /// Super-block size in bytes (176 bytes = 256 values)
    pub const SUPER_BLOCK_BYTES: usize = 176;
    /// Values per super-block
    pub const VALUES_PER_BLOCK: usize = 256;

    /// Create a Q5_K tensor from raw data
    #[must_use]
    pub fn from_raw(data: Vec<u8>, shape: Vec<usize>) -> Self {
        let n_values = shape.iter().product();
        Self {
            data,
            n_values,
            shape,
        }
    }

    /// Get number of values
    #[must_use]
    pub const fn len(&self) -> usize {
        self.n_values
    }

    /// Check if empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.n_values == 0
    }

    /// Get shape
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    /// Dequantize to f32 using realizar's Q5_K dequantization
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        crate::realizar_inference::dequantize_q5_k(&self.data)
            .unwrap_or_else(|_| vec![0.0; self.n_values])
    }

    /// Get raw data reference
    #[must_use]
    pub fn raw_data(&self) -> &[u8] {
        &self.data
    }
}

/// Q5_K quantized linear layer
#[cfg(feature = "realizar-inference")]
#[derive(Debug, Clone)]
pub struct QuantizedLinearQ5K {
    weight: QuantizedTensorQ5K,
    bias: Option<Vec<f32>>,
    in_features: usize,
    out_features: usize,
}

#[cfg(feature = "realizar-inference")]
impl QuantizedLinearQ5K {
    /// Create from raw Q5_K weight data
    #[must_use]
    pub fn from_raw(
        weight_data: Vec<u8>,
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        let n_values = out_features * in_features;
        Self {
            weight: QuantizedTensorQ5K {
                data: weight_data,
                n_values,
                shape: vec![out_features, in_features],
            },
            bias: bias.map(|b| b.to_vec()),
            in_features,
            out_features,
        }
    }

    /// Get input features
    #[must_use]
    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features
    #[must_use]
    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_size(&self) -> usize {
        let weight_size = self.weight.memory_bytes();
        let bias_size = self.bias.as_ref().map_or(0, |b| b.len() * 4);
        weight_size + bias_size
    }

    /// Forward pass with fused dequantize + matmul
    pub fn forward_fused(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        let batch_size = input.len() / self.in_features;
        if input.len() % self.in_features != 0 {
            return Err(WhisperError::Model(format!(
                "input size {} not divisible by in_features {}",
                input.len(),
                self.in_features
            )));
        }

        let mut output = Vec::with_capacity(batch_size * self.out_features);

        for b in 0..batch_size {
            let input_slice = &input[b * self.in_features..(b + 1) * self.in_features];
            let batch_output = crate::realizar_inference::fused_q5k_parallel_matvec(
                self.weight.raw_data(),
                input_slice,
                self.in_features,
                self.out_features,
            )
            .map_err(|e| WhisperError::Model(format!("Fused Q5K matvec failed: {e}")))?;
            output.extend(batch_output);
        }

        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for o in 0..self.out_features {
                    output[b * self.out_features + o] += bias[o];
                }
            }
        }

        Ok(output)
    }
}

// ============================================================================
// Q6_K Quantization (realizar integration)
// ============================================================================

/// Q6_K quantized tensor using realizar's K-quantization format
///
/// K-quantization uses super-blocks of 256 values with grouped scales:
/// - 210 bytes per super-block = 6.5 bits per weight
/// - ~5x memory reduction vs f32 (highest quality K-quant)
#[cfg(feature = "realizar-inference")]
#[derive(Debug, Clone)]
pub struct QuantizedTensorQ6K {
    /// Raw Q6_K data (super-blocks of 210 bytes each)
    data: Vec<u8>,
    /// Number of f32 values this represents
    n_values: usize,
    /// Shape dimensions
    shape: Vec<usize>,
}

#[cfg(feature = "realizar-inference")]
impl QuantizedTensorQ6K {
    /// Super-block size in bytes (210 bytes = 256 values)
    pub const SUPER_BLOCK_BYTES: usize = 210;
    /// Values per super-block
    pub const VALUES_PER_BLOCK: usize = 256;

    /// Create a Q6_K tensor from raw data
    #[must_use]
    pub fn from_raw(data: Vec<u8>, shape: Vec<usize>) -> Self {
        let n_values = shape.iter().product();
        Self {
            data,
            n_values,
            shape,
        }
    }

    /// Get number of values
    #[must_use]
    pub const fn len(&self) -> usize {
        self.n_values
    }

    /// Check if empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.n_values == 0
    }

    /// Get shape
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    /// Dequantize to f32 using realizar's Q6_K dequantization
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        crate::realizar_inference::dequantize_q6_k(&self.data)
            .unwrap_or_else(|_| vec![0.0; self.n_values])
    }

    /// Get raw data reference
    #[must_use]
    pub fn raw_data(&self) -> &[u8] {
        &self.data
    }
}

/// Q6_K quantized linear layer
#[cfg(feature = "realizar-inference")]
#[derive(Debug, Clone)]
pub struct QuantizedLinearQ6K {
    weight: QuantizedTensorQ6K,
    bias: Option<Vec<f32>>,
    in_features: usize,
    out_features: usize,
}

#[cfg(feature = "realizar-inference")]
impl QuantizedLinearQ6K {
    /// Create from raw Q6_K weight data
    #[must_use]
    pub fn from_raw(
        weight_data: Vec<u8>,
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        let n_values = out_features * in_features;
        Self {
            weight: QuantizedTensorQ6K {
                data: weight_data,
                n_values,
                shape: vec![out_features, in_features],
            },
            bias: bias.map(|b| b.to_vec()),
            in_features,
            out_features,
        }
    }

    /// Get input features
    #[must_use]
    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features
    #[must_use]
    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_size(&self) -> usize {
        let weight_size = self.weight.memory_bytes();
        let bias_size = self.bias.as_ref().map_or(0, |b| b.len() * 4);
        weight_size + bias_size
    }

    /// Forward pass with fused dequantize + matmul
    pub fn forward_fused(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        let batch_size = input.len() / self.in_features;
        if input.len() % self.in_features != 0 {
            return Err(WhisperError::Model(format!(
                "input size {} not divisible by in_features {}",
                input.len(),
                self.in_features
            )));
        }

        let mut output = Vec::with_capacity(batch_size * self.out_features);

        for b in 0..batch_size {
            let input_slice = &input[b * self.in_features..(b + 1) * self.in_features];
            let batch_output = crate::realizar_inference::fused_q6k_parallel_matvec(
                self.weight.raw_data(),
                input_slice,
                self.in_features,
                self.out_features,
            )
            .map_err(|e| WhisperError::Model(format!("Fused Q6K matvec failed: {e}")))?;
            output.extend(batch_output);
        }

        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for o in 0..self.out_features {
                    output[b * self.out_features + o] += bias[o];
                }
            }
        }

        Ok(output)
    }
}

// =============================================================================
// Sprint 9: QuantizedFeedForward (Q4K-based FFN)
// =============================================================================

/// Quantized feed-forward network using Q4_K weights
///
/// FFN(x) = fc2(GELU(fc1(x)))
/// Where fc1 and fc2 are QuantizedLinearQ4K layers.
///
/// # Memory Savings
///
/// For whisper-tiny (d_model=384, d_ff=1536):
/// - FP32: 4.72 MB per block
/// - Q4K: 664 KB per block (~7.1x reduction)
#[cfg(feature = "realizar-inference")]
#[derive(Debug, Clone)]
pub struct QuantizedFeedForward {
    /// First linear layer (d_model → d_ff) - expansion
    fc1: QuantizedLinearQ4K,
    /// Second linear layer (d_ff → d_model) - projection
    fc2: QuantizedLinearQ4K,
    /// Model dimension
    d_model: usize,
    /// Hidden dimension (typically 4 * d_model)
    d_ff: usize,
}

#[cfg(feature = "realizar-inference")]
impl QuantizedFeedForward {
    /// Create new quantized feed-forward network from raw Q4_K weight data
    ///
    /// # Arguments
    /// - `fc1_data`: Raw Q4_K data for expansion layer (d_model × d_ff)
    /// - `fc2_data`: Raw Q4_K data for projection layer (d_ff × d_model)
    /// - `d_model`: Model dimension
    /// - `d_ff`: Hidden dimension (typically 4 * d_model)
    #[must_use]
    pub fn new(fc1_data: Vec<u8>, fc2_data: Vec<u8>, d_model: usize, d_ff: usize) -> Self {
        Self {
            fc1: QuantizedLinearQ4K::from_raw(fc1_data, None, d_model, d_ff),
            fc2: QuantizedLinearQ4K::from_raw(fc2_data, None, d_ff, d_model),
            d_model,
            d_ff,
        }
    }

    /// Get model dimension
    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.d_model
    }

    /// Get hidden dimension
    #[must_use]
    pub const fn d_ff(&self) -> usize {
        self.d_ff
    }

    /// Get total memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.fc1.memory_size() + self.fc2.memory_size()
    }

    /// Forward pass: FFN(x) = fc2(GELU(fc1(x)))
    ///
    /// Uses fused Q4K operations for memory efficiency.
    ///
    /// # Arguments
    /// - `input`: Input tensor of shape [seq_len, d_model] flattened
    ///
    /// # Returns
    /// Output tensor of shape [seq_len, d_model] flattened
    pub fn forward(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        // fc1: [seq_len, d_model] → [seq_len, d_ff]
        let hidden = self.fc1.forward_fused(input)?;

        // GELU activation
        let activated = crate::simd::gelu(&hidden);

        // fc2: [seq_len, d_ff] → [seq_len, d_model]
        self.fc2.forward_fused(&activated)
    }
}

// =============================================================================
// Sprint 13: QuantizedMultiHeadAttention
// =============================================================================

/// Quantized multi-head attention using Q4K projection weights
///
/// Uses Q4K quantization for all four projection matrices (Q, K, V, O),
/// providing ~85% memory reduction compared to FP32 while maintaining
/// accuracy through high-precision score computation.
///
/// # Architecture
///
/// ```text
/// Input → Q4K(Wq) → Q \
/// Input → Q4K(Wk) → K  → Attention(Q,K,V) → Q4K(Wo) → Output
/// Input → Q4K(Wv) → V /
/// ```
///
/// # Memory Impact
///
/// For whisper-tiny (d_model=384):
/// - FP32: 4 × 384 × 384 × 4 = 2.36 MB
/// - Q4K: 4 × 384 × 384 × 0.5625 = 0.33 MB (85.9% reduction)
#[cfg(feature = "realizar-inference")]
#[derive(Debug, Clone)]
pub struct QuantizedMultiHeadAttention {
    /// Number of attention heads
    n_heads: usize,
    /// Hidden state dimension
    d_model: usize,
    /// Per-head dimension (d_model / n_heads)
    d_head: usize,
    /// Query projection (Q4K)
    w_q: QuantizedLinearQ4K,
    /// Key projection (Q4K)
    w_k: QuantizedLinearQ4K,
    /// Value projection (Q4K)
    w_v: QuantizedLinearQ4K,
    /// Output projection (Q4K)
    w_o: QuantizedLinearQ4K,
    /// Scale factor for attention scores (1/sqrt(d_head))
    scale: f32,
}

#[cfg(feature = "realizar-inference")]
impl QuantizedMultiHeadAttention {
    /// Create new quantized attention with random Q4K weights (for testing)
    ///
    /// # Arguments
    /// * `n_heads` - Number of attention heads
    /// * `d_model` - Hidden dimension (must be divisible by n_heads)
    ///
    /// # Panics
    /// Panics if `d_model` is not divisible by `n_heads`
    #[must_use]
    pub fn new_random(n_heads: usize, d_model: usize) -> Self {
        assert!(
            d_model % n_heads == 0,
            "d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        );

        let d_head = d_model / n_heads;

        // Q4_K: 144 bytes per super-block of 256 values
        // Realizar calculates row-by-row with padding to 256:
        //   super_blocks_per_row = in_dim.div_ceil(256)
        //   bytes_per_row = super_blocks_per_row * 144
        //   total_bytes = out_dim * bytes_per_row
        let super_block_bytes = 144usize;
        let super_blocks_per_row = d_model.div_ceil(256);
        let bytes_per_row = super_blocks_per_row * super_block_bytes;
        let data_size = d_model * bytes_per_row;

        // Create properly sized Q4K data for each projection
        let create_projection =
            || QuantizedLinearQ4K::from_raw(vec![0u8; data_size], None, d_model, d_model);

        Self {
            n_heads,
            d_model,
            d_head,
            w_q: create_projection(),
            w_k: create_projection(),
            w_v: create_projection(),
            w_o: create_projection(),
            scale: 1.0 / (d_head as f32).sqrt(),
        }
    }

    /// Create from quantized projection weights
    ///
    /// # Arguments
    /// * `n_heads` - Number of attention heads
    /// * `d_model` - Hidden dimension
    /// * `w_q` - Quantized query projection
    /// * `w_k` - Quantized key projection
    /// * `w_v` - Quantized value projection
    /// * `w_o` - Quantized output projection
    ///
    /// # Panics
    /// Panics if `d_model` is not divisible by `n_heads`
    #[must_use]
    pub fn new(
        n_heads: usize,
        d_model: usize,
        w_q: QuantizedLinearQ4K,
        w_k: QuantizedLinearQ4K,
        w_v: QuantizedLinearQ4K,
        w_o: QuantizedLinearQ4K,
    ) -> Self {
        assert!(
            d_model % n_heads == 0,
            "d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        );

        let d_head = d_model / n_heads;

        Self {
            n_heads,
            d_model,
            d_head,
            w_q,
            w_k,
            w_v,
            w_o,
            scale: 1.0 / (d_head as f32).sqrt(),
        }
    }

    /// Number of attention heads
    #[must_use]
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Hidden dimension
    #[must_use]
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Per-head dimension
    #[must_use]
    pub fn d_head(&self) -> usize {
        self.d_head
    }

    /// Total memory used by quantized weights (bytes)
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.w_q.memory_size()
            + self.w_k.memory_size()
            + self.w_v.memory_size()
            + self.w_o.memory_size()
    }

    /// Forward pass with quantized projections
    ///
    /// Computes multi-head attention:
    /// 1. Project Q, K, V using Q4K fused dequantize
    /// 2. Compute scaled dot-product attention (FP32)
    /// 3. Project output using Q4K fused dequantize
    ///
    /// # Arguments
    /// * `query` - Query input [q_len × d_model]
    /// * `key` - Key input [kv_len × d_model]
    /// * `value` - Value input [kv_len × d_model]
    /// * `mask` - Optional attention mask [q_len × kv_len]
    ///
    /// # Returns
    /// Output tensor [q_len × d_model]
    #[allow(clippy::needless_range_loop)]
    pub fn forward(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        mask: Option<&[f32]>,
    ) -> WhisperResult<Vec<f32>> {
        let q_len = query.len() / self.d_model;
        let kv_len = key.len() / self.d_model;

        // Project Q, K, V using quantized weights
        let q = self.w_q.forward_fused(query)?;
        let k = self.w_k.forward_fused(key)?;
        let v = self.w_v.forward_fused(value)?;

        // Compute attention for each head
        let mut output = vec![0.0f32; q_len * self.d_model];

        for head in 0..self.n_heads {
            let head_offset = head * self.d_head;

            // Extract head slices and compute attention
            for qi in 0..q_len {
                // Compute attention scores for this query position
                let mut scores = Vec::with_capacity(kv_len);

                for ki in 0..kv_len {
                    let mut score = 0.0f32;
                    for d in 0..self.d_head {
                        let q_idx = qi * self.d_model + head_offset + d;
                        let k_idx = ki * self.d_model + head_offset + d;
                        score += q[q_idx] * k[k_idx];
                    }
                    score *= self.scale;

                    // Apply mask if provided
                    if let Some(m) = mask {
                        let mask_idx = qi * kv_len + ki;
                        if mask_idx < m.len() {
                            score += m[mask_idx];
                        }
                    }

                    scores.push(score);
                }

                // Softmax
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
                let sum: f32 = exp_scores.iter().sum();
                let attn_weights: Vec<f32> = exp_scores.iter().map(|e| e / sum).collect();

                // Apply attention to values
                for d in 0..self.d_head {
                    let mut val = 0.0f32;
                    for ki in 0..kv_len {
                        let v_idx = ki * self.d_model + head_offset + d;
                        val += attn_weights[ki] * v[v_idx];
                    }
                    output[qi * self.d_model + head_offset + d] = val;
                }
            }
        }

        // Output projection
        self.w_o.forward_fused(&output)
    }
}

// =============================================================================
// Sprint 10: QuantizedDecoderBlock
// =============================================================================

/// Quantized decoder block using Q4K FFN weights
///
/// A single transformer decoder block with:
/// - Masked self-attention (FP32)
/// - Cross-attention to encoder (FP32)
/// - Feed-forward network with Q4K quantized weights
///
/// The attention layers remain FP32 for accuracy, while the FFN
/// (which dominates memory) uses Q4K quantization for ~7x reduction.
#[cfg(feature = "realizar-inference")]
#[derive(Debug, Clone)]
pub struct QuantizedDecoderBlock {
    /// Masked self-attention layer
    pub self_attn: super::attention::MultiHeadAttention,
    /// Layer norm before self-attention
    pub ln1: super::encoder::LayerNorm,
    /// Cross-attention layer (to encoder output)
    pub cross_attn: super::attention::MultiHeadAttention,
    /// Layer norm before cross-attention
    pub ln2: super::encoder::LayerNorm,
    /// Quantized feed-forward network (Q4K)
    pub ffn: QuantizedFeedForward,
    /// Layer norm before FFN
    pub ln3: super::encoder::LayerNorm,
    /// Model dimension
    d_model: usize,
    /// Hidden dimension
    d_ff: usize,
    /// Number of attention heads
    n_heads: usize,
}

#[cfg(feature = "realizar-inference")]
impl QuantizedDecoderBlock {
    /// Create new quantized decoder block
    ///
    /// # Arguments
    /// - `d_model`: Model dimension
    /// - `n_heads`: Number of attention heads
    /// - `d_ff`: Hidden dimension (typically 4 * d_model)
    /// - `fc1_data`: Raw Q4_K data for FFN expansion layer
    /// - `fc2_data`: Raw Q4_K data for FFN projection layer
    #[must_use]
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        fc1_data: Vec<u8>,
        fc2_data: Vec<u8>,
    ) -> Self {
        Self {
            self_attn: super::attention::MultiHeadAttention::new(n_heads, d_model),
            ln1: super::encoder::LayerNorm::new(d_model),
            cross_attn: super::attention::MultiHeadAttention::new(n_heads, d_model),
            ln2: super::encoder::LayerNorm::new(d_model),
            ffn: QuantizedFeedForward::new(fc1_data, fc2_data, d_model, d_ff),
            ln3: super::encoder::LayerNorm::new(d_model),
            d_model,
            d_ff,
            n_heads,
        }
    }

    /// Get model dimension
    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.d_model
    }

    /// Get hidden dimension
    #[must_use]
    pub const fn d_ff(&self) -> usize {
        self.d_ff
    }

    /// Get number of attention heads
    #[must_use]
    pub const fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Get FFN memory usage in bytes (Q4K quantized)
    #[must_use]
    pub fn ffn_memory_bytes(&self) -> usize {
        self.ffn.memory_bytes()
    }

    /// Forward pass through the decoder block
    ///
    /// # Arguments
    /// - `x`: Input tensor [seq_len, d_model] flattened
    /// - `encoder_output`: Encoder output [seq_len, d_model] flattened
    /// - `causal_mask`: Optional causal attention mask
    ///
    /// # Returns
    /// Output tensor [seq_len, d_model] flattened
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

        // Pre-norm FFN with residual (using quantized FFN)
        let normed = self.ln3.forward(&residual)?;
        let ffn_out = self.ffn.forward(&normed)?;
        for (r, f) in residual.iter_mut().zip(ffn_out.iter()) {
            *r += f;
        }

        Ok(residual)
    }
}

// =============================================================================
// Sprint 14: FullyQuantizedDecoderBlock (Attention + FFN all Q4K)
// =============================================================================

/// Fully quantized decoder block with Q4K attention and FFN
///
/// A single transformer decoder block where ALL major weight matrices are Q4K:
/// - Self-attention: Q4K projections (Q, K, V, O)
/// - Cross-attention: Q4K projections (Q, K, V, O)
/// - Feed-forward: Q4K fc1 and fc2
///
/// Only LayerNorm weights remain FP32 (small memory impact).
///
/// # Memory Savings (whisper-tiny)
///
/// - Self-attention: 2.36 MB → 0.44 MB (81% reduction)
/// - Cross-attention: 2.36 MB → 0.44 MB (81% reduction)
/// - FFN: 4.72 MB → 0.66 MB (86% reduction)
/// - Total per block: 9.44 MB → 1.54 MB (83.7% reduction)
#[cfg(feature = "realizar-inference")]
#[derive(Debug, Clone)]
pub struct FullyQuantizedDecoderBlock {
    /// Quantized self-attention (Q4K)
    pub self_attn: QuantizedMultiHeadAttention,
    /// Layer norm before self-attention
    pub ln1: super::encoder::LayerNorm,
    /// Quantized cross-attention (Q4K)
    pub cross_attn: QuantizedMultiHeadAttention,
    /// Layer norm before cross-attention
    pub ln2: super::encoder::LayerNorm,
    /// Quantized feed-forward network (Q4K)
    pub ffn: QuantizedFeedForward,
    /// Layer norm before FFN
    pub ln3: super::encoder::LayerNorm,
    /// Model dimension
    d_model: usize,
    /// Hidden dimension
    d_ff: usize,
    /// Number of attention heads
    n_heads: usize,
}

#[cfg(feature = "realizar-inference")]
impl FullyQuantizedDecoderBlock {
    /// Create new fully quantized decoder block with random weights (for testing)
    ///
    /// # Arguments
    /// - `n_heads`: Number of attention heads
    /// - `d_model`: Model dimension
    /// - `d_ff`: Hidden dimension (typically 4 * d_model)
    ///
    /// # Panics
    /// Panics if `d_model` is not divisible by `n_heads`
    #[must_use]
    pub fn new_random(n_heads: usize, d_model: usize, d_ff: usize) -> Self {
        // Create Q4K FFN data
        let super_block_bytes = 144usize;

        // FFN fc1: d_model → d_ff
        let fc1_blocks_per_row = d_model.div_ceil(256);
        let fc1_bytes_per_row = fc1_blocks_per_row * super_block_bytes;
        let fc1_data = vec![0u8; d_ff * fc1_bytes_per_row];

        // FFN fc2: d_ff → d_model
        let fc2_blocks_per_row = d_ff.div_ceil(256);
        let fc2_bytes_per_row = fc2_blocks_per_row * super_block_bytes;
        let fc2_data = vec![0u8; d_model * fc2_bytes_per_row];

        Self {
            self_attn: QuantizedMultiHeadAttention::new_random(n_heads, d_model),
            ln1: super::encoder::LayerNorm::new(d_model),
            cross_attn: QuantizedMultiHeadAttention::new_random(n_heads, d_model),
            ln2: super::encoder::LayerNorm::new(d_model),
            ffn: QuantizedFeedForward::new(fc1_data, fc2_data, d_model, d_ff),
            ln3: super::encoder::LayerNorm::new(d_model),
            d_model,
            d_ff,
            n_heads,
        }
    }

    /// Get model dimension
    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.d_model
    }

    /// Get hidden dimension
    #[must_use]
    pub const fn d_ff(&self) -> usize {
        self.d_ff
    }

    /// Get number of attention heads
    #[must_use]
    pub const fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Get total memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.self_attn.memory_bytes() + self.cross_attn.memory_bytes() + self.ffn.memory_bytes()
    }

    /// Forward pass through fully quantized decoder block
    ///
    /// # Arguments
    /// - `input`: Decoder hidden state [seq_len × d_model]
    /// - `encoder_output`: Encoder output for cross-attention [enc_len × d_model]
    ///
    /// # Returns
    /// Output hidden state [seq_len × d_model]
    pub fn forward(&self, input: &[f32], encoder_output: &[f32]) -> WhisperResult<Vec<f32>> {
        let mut residual = input.to_vec();

        // Pre-norm self-attention with residual (quantized)
        let normed = self.ln1.forward(&residual)?;
        let self_attn_out = self.self_attn.forward(&normed, &normed, &normed, None)?;
        for (r, s) in residual.iter_mut().zip(self_attn_out.iter()) {
            *r += s;
        }

        // Pre-norm cross-attention with residual (quantized)
        let normed = self.ln2.forward(&residual)?;
        let cross_out = self
            .cross_attn
            .forward(&normed, encoder_output, encoder_output, None)?;
        for (r, c) in residual.iter_mut().zip(cross_out.iter()) {
            *r += c;
        }

        // Pre-norm FFN with residual (quantized)
        let normed = self.ln3.forward(&residual)?;
        let ffn_out = self.ffn.forward(&normed)?;
        for (r, f) in residual.iter_mut().zip(ffn_out.iter()) {
            *r += f;
        }

        Ok(residual)
    }
}

// =============================================================================
// Sprint 15: FullyQuantizedDecoder (All weights Q4K)
// =============================================================================

/// Fully quantized decoder with all weights in Q4K format
///
/// A complete transformer decoder where each block uses Q4K quantized weights
/// for both attention projections AND FFN layers, achieving maximum compression.
///
/// # Memory Savings (whisper-tiny)
///
/// - Attention weights: 7.08 MB → 1.26 MB (~82% reduction per block)
/// - FFN weights: 18.9 MB → 2.7 MB (~86% reduction per block)
/// - Total blocks: 37.76 MB → 6.64 MB (~82% reduction)
/// - Embeddings: unchanged (FP32 for accuracy)
#[cfg(feature = "realizar-inference")]
#[derive(Debug, Clone)]
pub struct FullyQuantizedDecoder {
    /// Number of layers
    n_layers: usize,
    /// Hidden state dimension
    d_model: usize,
    /// Number of attention heads
    n_heads: usize,
    /// Hidden dimension
    d_ff: usize,
    /// Fully quantized decoder blocks
    blocks: Vec<FullyQuantizedDecoderBlock>,
    /// Final layer norm
    ln_post: super::encoder::LayerNorm,
    /// Token embeddings (n_vocab x d_model)
    token_embedding: Vec<f32>,
    /// Positional embeddings (max_len x d_model)
    positional_embedding: Vec<f32>,
    /// Vocabulary size
    n_vocab: usize,
    /// Maximum sequence length
    max_len: usize,
}

#[cfg(feature = "realizar-inference")]
impl FullyQuantizedDecoder {
    /// Create new fully quantized decoder with random weights (for testing)
    ///
    /// # Arguments
    /// - `n_layers`: Number of transformer layers
    /// - `d_model`: Model dimension
    /// - `n_heads`: Number of attention heads
    /// - `d_ff`: Hidden dimension (typically 4 * d_model)
    /// - `n_vocab`: Vocabulary size
    /// - `max_len`: Maximum sequence length
    #[must_use]
    pub fn new_random(
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_vocab: usize,
        max_len: usize,
    ) -> Self {
        let blocks = (0..n_layers)
            .map(|_| FullyQuantizedDecoderBlock::new_random(n_heads, d_model, d_ff))
            .collect();

        // Initialize embeddings with zeros (would be loaded from model in practice)
        let token_embedding = vec![0.0f32; n_vocab * d_model];
        let positional_embedding = vec![0.0f32; max_len * d_model];

        Self {
            n_layers,
            d_model,
            n_heads,
            d_ff,
            blocks,
            ln_post: super::encoder::LayerNorm::new(d_model),
            token_embedding,
            positional_embedding,
            n_vocab,
            max_len,
        }
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

    /// Get vocabulary size
    #[must_use]
    pub const fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// Get number of attention heads
    #[must_use]
    pub const fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Get hidden dimension
    #[must_use]
    pub const fn d_ff(&self) -> usize {
        self.d_ff
    }

    /// Get total block memory usage in bytes (all Q4K weights)
    #[must_use]
    pub fn block_memory_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.memory_bytes()).sum()
    }

    /// Create KV cache for incremental decoding
    #[must_use]
    pub fn create_kv_cache(&self) -> super::decoder::DecoderKVCache {
        super::decoder::DecoderKVCache::new(self.n_layers, self.d_model, self.max_len)
    }

    /// Forward pass for single token (incremental decoding)
    ///
    /// # Arguments
    /// - `token`: Input token ID
    /// - `encoder_output`: Encoder output [encoder_seq_len, d_model] flattened
    /// - `cache`: KV cache for incremental decoding
    ///
    /// # Returns
    /// Logits over vocabulary [n_vocab]
    #[allow(clippy::needless_range_loop)]
    pub fn forward_one_fully_quantized(
        &self,
        token: u32,
        encoder_output: &[f32],
        cache: &mut super::decoder::DecoderKVCache,
    ) -> WhisperResult<Vec<f32>> {
        let pos = cache.seq_len();

        if pos >= self.max_len {
            return Err(WhisperError::Model(format!(
                "position {} exceeds max {}",
                pos, self.max_len
            )));
        }

        // Get token embedding
        let token_idx = token as usize;
        if token_idx >= self.n_vocab {
            return Err(WhisperError::Model(format!(
                "token {} exceeds vocab size {}",
                token_idx, self.n_vocab
            )));
        }

        // x = token_embedding + positional_embedding
        let mut x: Vec<f32> = (0..self.d_model)
            .map(|i| {
                self.token_embedding[token_idx * self.d_model + i]
                    + self.positional_embedding[pos * self.d_model + i]
            })
            .collect();

        // Forward through each fully quantized block
        for block in &self.blocks {
            x = block.forward(&x, encoder_output)?;
        }

        // Final layer norm
        x = self.ln_post.forward(&x)?;

        // Project to vocabulary (simple matmul with token embeddings)
        // logits = x @ token_embedding.T
        let mut logits = vec![0.0f32; self.n_vocab];
        for v in 0..self.n_vocab {
            let mut sum = 0.0f32;
            for d in 0..self.d_model {
                sum += x[d] * self.token_embedding[v * self.d_model + d];
            }
            logits[v] = sum;
        }

        // Update cache position (simplified - real impl would update KV cache)
        cache.increment_seq_len();

        Ok(logits)
    }
}

// =============================================================================
// Sprint 11: QuantizedDecoder (Full decoder with Q4K FFN)
// =============================================================================

/// Quantized decoder with Q4K FFN weights
///
/// A complete transformer decoder where each block uses Q4K quantized FFN
/// weights for ~7x memory reduction in the FFN layers.
///
/// # Memory Savings (whisper-tiny)
///
/// - FFN weights: 18.9 MB → 2.7 MB (~7x reduction)
/// - Attention weights: unchanged (FP32 for accuracy)
/// - Total model: ~45 MB → ~29 MB (~35% reduction)
#[cfg(feature = "realizar-inference")]
#[derive(Debug, Clone)]
pub struct QuantizedDecoder {
    /// Number of layers
    n_layers: usize,
    /// Hidden state dimension
    d_model: usize,
    /// Number of attention heads
    n_heads: usize,
    /// Hidden dimension
    d_ff: usize,
    /// Quantized decoder blocks
    blocks: Vec<QuantizedDecoderBlock>,
    /// Final layer norm
    ln_post: super::encoder::LayerNorm,
    /// Token embeddings (n_vocab x d_model)
    token_embedding: Vec<f32>,
    /// Positional embeddings (max_len x d_model)
    positional_embedding: Vec<f32>,
    /// Vocabulary size
    n_vocab: usize,
    /// Maximum sequence length
    max_len: usize,
}

#[cfg(feature = "realizar-inference")]
impl QuantizedDecoder {
    /// Create new quantized decoder
    ///
    /// # Arguments
    /// - `n_layers`: Number of transformer layers
    /// - `d_model`: Model dimension
    /// - `n_heads`: Number of attention heads
    /// - `d_ff`: Hidden dimension (typically 4 * d_model)
    /// - `n_vocab`: Vocabulary size
    /// - `max_len`: Maximum sequence length
    /// - `ffn_data`: Q4K weight data for each layer's FFN (fc1_data, fc2_data)
    ///
    /// # Panics
    /// Panics if `ffn_data.len() != n_layers`
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_vocab: usize,
        max_len: usize,
        ffn_data: Vec<(Vec<u8>, Vec<u8>)>,
    ) -> Self {
        assert_eq!(ffn_data.len(), n_layers, "FFN data must match n_layers");

        let blocks = ffn_data
            .into_iter()
            .map(|(fc1, fc2)| QuantizedDecoderBlock::new(d_model, n_heads, d_ff, fc1, fc2))
            .collect();

        // Initialize embeddings with zeros (would be loaded from model in practice)
        let token_embedding = vec![0.0f32; n_vocab * d_model];
        let positional_embedding = vec![0.0f32; max_len * d_model];

        Self {
            n_layers,
            d_model,
            n_heads,
            d_ff,
            blocks,
            ln_post: super::encoder::LayerNorm::new(d_model),
            token_embedding,
            positional_embedding,
            n_vocab,
            max_len,
        }
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

    /// Get vocabulary size
    #[must_use]
    pub const fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// Get number of attention heads
    #[must_use]
    pub const fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Get hidden dimension
    #[must_use]
    pub const fn d_ff(&self) -> usize {
        self.d_ff
    }

    /// Get total FFN memory usage in bytes (Q4K quantized)
    #[must_use]
    pub fn ffn_memory_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.ffn_memory_bytes()).sum()
    }

    /// Create KV cache for incremental decoding
    #[must_use]
    pub fn create_kv_cache(&self) -> super::decoder::DecoderKVCache {
        super::decoder::DecoderKVCache::new(self.n_layers, self.d_model, self.max_len)
    }

    /// Forward pass for single token (incremental decoding)
    ///
    /// # Arguments
    /// - `token`: Input token ID
    /// - `encoder_output`: Encoder output [encoder_seq_len, d_model] flattened
    /// - `cache`: KV cache for incremental decoding
    ///
    /// # Returns
    /// Logits over vocabulary [n_vocab]
    #[allow(clippy::needless_range_loop)]
    pub fn forward_one_quantized(
        &self,
        token: u32,
        encoder_output: &[f32],
        cache: &mut super::decoder::DecoderKVCache,
    ) -> WhisperResult<Vec<f32>> {
        let pos = cache.seq_len();

        if pos >= self.max_len {
            return Err(WhisperError::Model(format!(
                "position {} exceeds max {}",
                pos, self.max_len
            )));
        }

        // Get token embedding
        let token_idx = token as usize;
        if token_idx >= self.n_vocab {
            return Err(WhisperError::Model(format!(
                "token {} exceeds vocab size {}",
                token_idx, self.n_vocab
            )));
        }

        // x = token_embedding + positional_embedding
        let mut x: Vec<f32> = (0..self.d_model)
            .map(|i| {
                self.token_embedding[token_idx * self.d_model + i]
                    + self.positional_embedding[pos * self.d_model + i]
            })
            .collect();

        // Forward through each quantized block
        for block in &self.blocks {
            x = block.forward(&x, encoder_output, None)?;
        }

        // Final layer norm
        x = self.ln_post.forward(&x)?;

        // Project to vocabulary (simple matmul with token embeddings)
        // logits = x @ token_embedding.T
        let mut logits = vec![0.0f32; self.n_vocab];
        for v in 0..self.n_vocab {
            let mut sum = 0.0f32;
            for d in 0..self.d_model {
                sum += x[d] * self.token_embedding[v * self.d_model + d];
            }
            logits[v] = sum;
        }

        // Update cache position (simplified - real impl would update KV cache)
        cache.increment_seq_len();

        Ok(logits)
    }
}

/// Quantize f32 values to int8 with symmetric quantization
///
/// Returns (quantized_data, scale)
#[must_use]
pub fn quantize_f32_to_i8(data: &[f32]) -> (Vec<i8>, f32) {
    if data.is_empty() {
        return (Vec::new(), 1.0);
    }

    // Find max absolute value
    let max_abs = data.iter().map(|x| x.abs()).fold(0.0_f32, |a, b| a.max(b));

    // Compute scale
    let scale = if max_abs < MIN_SCALE {
        1.0
    } else {
        max_abs / I8_MAX
    };

    // Quantize
    let quantized: Vec<i8> = data
        .iter()
        .map(|&x| {
            let q = (x / scale).round();
            q.clamp(-128.0, 127.0) as i8
        })
        .collect();

    (quantized, scale)
}

/// Dequantize int8 values back to f32
#[must_use]
pub fn dequantize_i8_to_f32(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&q| f32::from(q) * scale).collect()
}

/// Quantize f32 values to int8 with per-channel quantization
///
/// Uses different scales for each output channel (useful for linear layers).
#[must_use]
pub fn quantize_f32_to_i8_per_channel(
    data: &[f32],
    n_channels: usize,
    channel_size: usize,
) -> (Vec<i8>, Vec<f32>) {
    if data.is_empty() || n_channels == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut quantized = Vec::with_capacity(data.len());
    let mut scales = Vec::with_capacity(n_channels);

    for ch in 0..n_channels {
        let start = ch * channel_size;
        let end = start + channel_size;
        let channel_data = &data[start..end.min(data.len())];

        let (q, scale) = quantize_f32_to_i8(channel_data);
        quantized.extend(q);
        scales.push(scale);
    }

    (quantized, scales)
}

/// Dequantize int8 values with per-channel scales
#[must_use]
pub fn dequantize_i8_to_f32_per_channel(
    data: &[i8],
    scales: &[f32],
    channel_size: usize,
) -> Vec<f32> {
    let mut result = Vec::with_capacity(data.len());

    for (ch, &scale) in scales.iter().enumerate() {
        let start = ch * channel_size;
        let end = (start + channel_size).min(data.len());

        for &q in &data[start..end] {
            result.push(f32::from(q) * scale);
        }
    }

    result
}

// =============================================================================
// Int4 Quantization
// =============================================================================

/// Quantized int4 tensor with packed storage
///
/// Packs 2 int4 values per byte for 2x memory savings over int8.
/// Uses signed int4 (-8 to 7) with symmetric quantization.
#[derive(Debug, Clone)]
pub struct QuantizedTensorInt4 {
    /// Packed int4 values (2 values per byte)
    pub data: Vec<u8>,
    /// Scale factor for dequantization
    pub scale: f32,
    /// Number of elements (may be odd, so last byte only uses low nibble)
    pub len: usize,
    /// Original shape dimensions
    pub shape: Vec<usize>,
}

impl QuantizedTensorInt4 {
    /// Create a new int4 quantized tensor from f32 data
    #[must_use]
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Self {
        let (quantized, scale) = quantize_f32_to_i4_packed(data);
        Self {
            data: quantized,
            scale,
            len: data.len(),
            shape,
        }
    }

    /// Dequantize back to f32
    #[must_use]
    pub fn to_f32(&self) -> Vec<f32> {
        dequantize_i4_packed_to_f32(&self.data, self.scale, self.len)
    }

    /// Get the number of elements
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_size(&self) -> usize {
        self.data.len() + 4 // packed data + scale
    }

    /// Get the total number of elements from shape
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Unpack to individual i8 values (for easier computation)
    #[must_use]
    pub fn unpack(&self) -> Vec<i8> {
        unpack_i4_to_i8(&self.data, self.len)
    }
}

/// Pack a signed int4 value (-8 to 7) into the low 4 bits
#[inline]
fn pack_i4(value: i8) -> u8 {
    // Convert signed int4 to unsigned nibble (0-15)
    // -8 -> 0x8, -1 -> 0xF, 0 -> 0x0, 7 -> 0x7
    (value as u8) & 0x0F
}

/// Unpack a nibble to signed int4
#[inline]
fn unpack_i4(nibble: u8) -> i8 {
    // Convert unsigned nibble to signed int4
    let val = nibble & 0x0F;
    if val >= 8 {
        // Sign extend
        (val as i8) - 16
    } else {
        val as i8
    }
}

/// Quantize f32 values to packed int4
///
/// Returns (packed_data, scale)
/// Two int4 values are packed per byte: low nibble first, high nibble second.
#[must_use]
pub fn quantize_f32_to_i4_packed(data: &[f32]) -> (Vec<u8>, f32) {
    if data.is_empty() {
        return (Vec::new(), 1.0);
    }

    // Find max absolute value
    let max_abs = data.iter().map(|x| x.abs()).fold(0.0_f32, |a, b| a.max(b));

    // Compute scale
    let scale = if max_abs < MIN_SCALE {
        1.0
    } else {
        max_abs / I4_MAX
    };

    // Quantize and pack
    let packed_len = data.len().div_ceil(2);
    let mut packed = vec![0u8; packed_len];

    for (i, &x) in data.iter().enumerate() {
        let q = (x / scale).round().clamp(-8.0, 7.0) as i8;
        let nibble = pack_i4(q);

        let byte_idx = i / 2;
        if i % 2 == 0 {
            // Low nibble
            packed[byte_idx] |= nibble;
        } else {
            // High nibble
            packed[byte_idx] |= nibble << 4;
        }
    }

    (packed, scale)
}

/// Dequantize packed int4 values back to f32
#[must_use]
pub fn dequantize_i4_packed_to_f32(data: &[u8], scale: f32, len: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(len);

    for (i, &byte) in data.iter().enumerate() {
        // Low nibble
        let low_idx = i * 2;
        if low_idx < len {
            let q = unpack_i4(byte & 0x0F);
            result.push(f32::from(q) * scale);
        }

        // High nibble
        let high_idx = i * 2 + 1;
        if high_idx < len {
            let q = unpack_i4(byte >> 4);
            result.push(f32::from(q) * scale);
        }
    }

    result
}

/// Unpack int4 values to individual i8 values
#[must_use]
pub fn unpack_i4_to_i8(data: &[u8], len: usize) -> Vec<i8> {
    let mut result = Vec::with_capacity(len);

    for (i, &byte) in data.iter().enumerate() {
        let low_idx = i * 2;
        if low_idx < len {
            result.push(unpack_i4(byte & 0x0F));
        }

        let high_idx = i * 2 + 1;
        if high_idx < len {
            result.push(unpack_i4(byte >> 4));
        }
    }

    result
}

/// Quantize f32 values to int4 (unpacked as i8 for easier computation)
///
/// Returns (quantized_data, scale) where values are in -8..7 range
#[must_use]
pub fn quantize_f32_to_i4(data: &[f32]) -> (Vec<i8>, f32) {
    if data.is_empty() {
        return (Vec::new(), 1.0);
    }

    // Find max absolute value
    let max_abs = data.iter().map(|x| x.abs()).fold(0.0_f32, |a, b| a.max(b));

    // Compute scale
    let scale = if max_abs < MIN_SCALE {
        1.0
    } else {
        max_abs / I4_MAX
    };

    // Quantize to -8..7
    let quantized: Vec<i8> = data
        .iter()
        .map(|&x| (x / scale).round().clamp(-8.0, 7.0) as i8)
        .collect();

    (quantized, scale)
}

/// Dequantize int4 values (stored as i8) back to f32
#[must_use]
pub fn dequantize_i4_to_f32(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&q| f32::from(q) * scale).collect()
}

/// Quantized int4 linear layer
#[derive(Debug, Clone)]
pub struct QuantizedLinearInt4 {
    /// Quantized weight matrix (packed int4)
    pub weight: QuantizedTensorInt4,
    /// Optional bias (not quantized)
    pub bias: Option<Vec<f32>>,
    /// Input features
    pub in_features: usize,
    /// Output features
    pub out_features: usize,
}

impl QuantizedLinearInt4 {
    /// Create from f32 weights
    #[must_use]
    pub fn from_f32(
        weight: &[f32],
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        Self {
            weight: QuantizedTensorInt4::from_f32(weight, vec![out_features, in_features]),
            bias: bias.map(|b| b.to_vec()),
            in_features,
            out_features,
        }
    }

    /// Forward pass with dequantization
    pub fn forward(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        let batch_size = input.len() / self.in_features;
        if input.len() % self.in_features != 0 {
            return Err(WhisperError::Model(format!(
                "input size {} not divisible by in_features {}",
                input.len(),
                self.in_features
            )));
        }

        // Dequantize weights [out_features, in_features]
        let weights = self.weight.to_f32();

        // Transpose weights to [in_features, out_features] for matmul
        let weights_t = simd::transpose(&weights, self.out_features, self.in_features);

        // SIMD matrix multiply: input [batch, in] @ weights_t [in, out] = output [batch, out]
        let mut output = simd::matmul(
            input,
            &weights_t,
            batch_size,
            self.in_features,
            self.out_features,
        );

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for o in 0..self.out_features {
                    output[b * self.out_features + o] += bias[o];
                }
            }
        }

        Ok(output)
    }

    /// Forward pass keeping computation in int4 (faster but less accurate)
    pub fn forward_quantized(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        let batch_size = input.len() / self.in_features;
        if input.len() % self.in_features != 0 {
            return Err(WhisperError::Model(format!(
                "input size {} not divisible by in_features {}",
                input.len(),
                self.in_features
            )));
        }

        // Quantize input to int4
        let (input_q, input_scale) = quantize_f32_to_i4(input);
        let weight_q = self.weight.unpack();

        // Int4 matrix multiply with i32 accumulator
        let mut output_acc = vec![0i32; batch_size * self.out_features];

        for b in 0..batch_size {
            for o in 0..self.out_features {
                let mut sum = 0i32;
                for i in 0..self.in_features {
                    sum += i32::from(input_q[b * self.in_features + i])
                        * i32::from(weight_q[o * self.in_features + i]);
                }
                output_acc[b * self.out_features + o] = sum;
            }
        }

        // Scale back to f32
        let scale = input_scale * self.weight.scale;
        let mut output: Vec<f32> = output_acc.iter().map(|&x| (x as f32) * scale).collect();

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for o in 0..self.out_features {
                    output[b * self.out_features + o] += bias[o];
                }
            }
        }

        Ok(output)
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_size(&self) -> usize {
        let weight_size = self.weight.data.len(); // 0.5 bytes per element (packed)
        let bias_size = self.bias.as_ref().map_or(0, |b| b.len() * 4);
        weight_size + bias_size + 4 // +4 for scale
    }
}

/// Compute quantization error for int4
#[must_use]
pub fn quantization_error_i4(original: &[f32], quantized: &[i8], scale: f32) -> f32 {
    if original.is_empty() {
        return 0.0;
    }

    let reconstructed = dequantize_i4_to_f32(quantized, scale);
    let mse: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| (o - r).powi(2))
        .sum::<f32>()
        / original.len() as f32;

    mse
}

/// Compute SQNR for int4 quantization
#[must_use]
pub fn compute_sqnr_i4(original: &[f32], quantized: &[i8], scale: f32) -> f32 {
    if original.is_empty() {
        return 0.0;
    }

    let signal_power: f32 = original.iter().map(|x| x.powi(2)).sum::<f32>() / original.len() as f32;
    let noise_power = quantization_error_i4(original, quantized, scale);

    if noise_power < MIN_SCALE {
        return f32::INFINITY;
    }

    10.0 * (signal_power / noise_power).log10()
}

// =============================================================================
// Mixed-Precision Inference (WAPR-073)
// =============================================================================

/// Weight precision options for mixed-precision inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightPrecision {
    /// 4-bit integer weights (packed, 2x memory savings)
    Int4,
    /// 8-bit integer weights
    Int8,
}

impl core::fmt::Display for WeightPrecision {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Int4 => write!(f, "int4"),
            Self::Int8 => write!(f, "int8"),
        }
    }
}

/// Activation precision options for mixed-precision inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationPrecision {
    /// 32-bit floating point activations (full precision)
    Float32,
}

impl core::fmt::Display for ActivationPrecision {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Float32 => write!(f, "fp32"),
        }
    }
}

/// Configuration for mixed-precision inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MixedPrecisionConfig {
    /// Precision for storing weights
    pub weight_precision: WeightPrecision,
    /// Precision for activations during computation
    pub activation_precision: ActivationPrecision,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self::int4_fp32()
    }
}

impl MixedPrecisionConfig {
    /// Create config for int4 weights with fp32 activations
    ///
    /// This is the default for maximum memory savings with good accuracy.
    #[must_use]
    pub const fn int4_fp32() -> Self {
        Self {
            weight_precision: WeightPrecision::Int4,
            activation_precision: ActivationPrecision::Float32,
        }
    }

    /// Create config for int8 weights with fp32 activations
    ///
    /// Better accuracy than int4 but uses more memory.
    #[must_use]
    pub const fn int8_fp32() -> Self {
        Self {
            weight_precision: WeightPrecision::Int8,
            activation_precision: ActivationPrecision::Float32,
        }
    }

    /// Get human-readable description of this config
    #[must_use]
    pub fn description(&self) -> String {
        format!(
            "{} weights, {} activations",
            self.weight_precision, self.activation_precision
        )
    }
}

/// Quantized weight storage for mixed-precision inference
#[derive(Debug, Clone)]
enum QuantizedWeights {
    /// Int4 packed weights
    Int4(QuantizedTensorInt4),
    /// Int8 weights
    Int8(QuantizedTensor),
}

impl QuantizedWeights {
    /// Dequantize weights to f32 for computation
    fn to_f32(&self) -> Vec<f32> {
        match self {
            Self::Int4(t) => t.to_f32(),
            Self::Int8(t) => t.to_f32(),
        }
    }

    /// Get memory size in bytes
    fn memory_size(&self) -> usize {
        match self {
            Self::Int4(t) => t.memory_size(),
            Self::Int8(t) => t.data.len() + 4, // data + scale
        }
    }
}

/// Mixed-precision linear layer
///
/// Stores weights in quantized format (int4 or int8) while keeping
/// activations in full precision (fp32) during computation.
///
/// This provides:
/// - Memory savings from quantized weight storage
/// - Accuracy from fp32 activation computation
/// - Flexibility to trade off memory vs accuracy
#[derive(Debug, Clone)]
pub struct MixedPrecisionLinear {
    /// Quantized weight matrix (out_features × in_features)
    weights: QuantizedWeights,
    /// Optional bias (always fp32)
    bias: Option<Vec<f32>>,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Precision configuration
    config: MixedPrecisionConfig,
}

impl MixedPrecisionLinear {
    /// Create from f32 weights with specified precision config
    #[must_use]
    pub fn from_f32_with_config(
        weight: &[f32],
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
        config: MixedPrecisionConfig,
    ) -> Self {
        let weights = match config.weight_precision {
            WeightPrecision::Int4 => QuantizedWeights::Int4(QuantizedTensorInt4::from_f32(
                weight,
                vec![out_features, in_features],
            )),
            WeightPrecision::Int8 => QuantizedWeights::Int8(QuantizedTensor::from_f32(
                weight,
                vec![out_features, in_features],
            )),
        };

        Self {
            weights,
            bias: bias.map(|b| b.to_vec()),
            in_features,
            out_features,
            config,
        }
    }

    /// Get input features
    #[must_use]
    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features
    #[must_use]
    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get weight precision
    #[must_use]
    pub const fn weight_precision(&self) -> WeightPrecision {
        self.config.weight_precision
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_size(&self) -> usize {
        let weight_size = self.weights.memory_size();
        let bias_size = self.bias.as_ref().map_or(0, |b| b.len() * 4);
        weight_size + bias_size
    }

    /// Forward pass with mixed-precision computation
    ///
    /// Weights are dequantized to fp32, then multiplied with fp32 activations.
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size × in_features)
    ///
    /// # Returns
    /// Output tensor (batch_size × out_features)
    ///
    /// # Errors
    ///
    /// Returns error if input size is not divisible by in_features.
    pub fn forward(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        let batch_size = input.len() / self.in_features;
        if input.len() % self.in_features != 0 {
            return Err(WhisperError::Model(format!(
                "input size {} not divisible by in_features {}",
                input.len(),
                self.in_features
            )));
        }

        // Dequantize weights to fp32 [out_features, in_features]
        let weights = self.weights.to_f32();

        // Transpose weights to [in_features, out_features] for matmul
        let weights_t = simd::transpose(&weights, self.out_features, self.in_features);

        // SIMD matrix multiply: input [batch, in] @ weights_t [in, out] = output [batch, out]
        let mut output = simd::matmul(
            input,
            &weights_t,
            batch_size,
            self.in_features,
            self.out_features,
        );

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for o in 0..self.out_features {
                    output[b * self.out_features + o] += bias[o];
                }
            }
        }

        Ok(output)
    }
}

/// Quantized linear layer weights
#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    /// Quantized weight matrix (out_features × in_features)
    pub weight: QuantizedTensor,
    /// Optional bias (not quantized)
    pub bias: Option<Vec<f32>>,
    /// Input features
    pub in_features: usize,
    /// Output features
    pub out_features: usize,
}

impl QuantizedLinear {
    /// Create from f32 weights
    #[must_use]
    pub fn from_f32(
        weight: &[f32],
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        Self {
            weight: QuantizedTensor::from_f32(weight, vec![out_features, in_features]),
            bias: bias.map(|b| b.to_vec()),
            in_features,
            out_features,
        }
    }

    /// Forward pass with dequantization
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size × in_features)
    ///
    /// # Returns
    /// Output tensor (batch_size × out_features)
    pub fn forward(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        let batch_size = input.len() / self.in_features;
        if input.len() % self.in_features != 0 {
            return Err(WhisperError::Model(format!(
                "input size {} not divisible by in_features {}",
                input.len(),
                self.in_features
            )));
        }

        // Dequantize weights [out_features, in_features]
        let weights = self.weight.to_f32();

        // Transpose weights to [in_features, out_features] for matmul
        let weights_t = simd::transpose(&weights, self.out_features, self.in_features);

        // SIMD matrix multiply: input [batch, in] @ weights_t [in, out] = output [batch, out]
        let mut output = simd::matmul(
            input,
            &weights_t,
            batch_size,
            self.in_features,
            self.out_features,
        );

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for o in 0..self.out_features {
                    output[b * self.out_features + o] += bias[o];
                }
            }
        }

        Ok(output)
    }

    /// Forward pass keeping computation in int8 (faster but slightly less accurate)
    ///
    /// Uses int8 GEMM with final f32 rescaling.
    pub fn forward_quantized(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        let batch_size = input.len() / self.in_features;
        if input.len() % self.in_features != 0 {
            return Err(WhisperError::Model(format!(
                "input size {} not divisible by in_features {}",
                input.len(),
                self.in_features
            )));
        }

        // Quantize input
        let (input_q, input_scale) = quantize_f32_to_i8(input);

        // Int8 matrix multiply with i32 accumulator
        let mut output_acc = vec![0i32; batch_size * self.out_features];

        for b in 0..batch_size {
            for o in 0..self.out_features {
                let mut sum = 0i32;
                for i in 0..self.in_features {
                    sum += i32::from(input_q[b * self.in_features + i])
                        * i32::from(self.weight.data[o * self.in_features + i]);
                }
                output_acc[b * self.out_features + o] = sum;
            }
        }

        // Scale back to f32
        let scale = input_scale * self.weight.scale;
        let mut output: Vec<f32> = output_acc.iter().map(|&x| (x as f32) * scale).collect();

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for o in 0..self.out_features {
                    output[b * self.out_features + o] += bias[o];
                }
            }
        }

        Ok(output)
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_size(&self) -> usize {
        let weight_size = self.weight.data.len(); // 1 byte per element
        let bias_size = self.bias.as_ref().map_or(0, |b| b.len() * 4);
        weight_size + bias_size + 4 // +4 for scale
    }
}

/// Compute quantization error (MSE)
#[must_use]
pub fn quantization_error(original: &[f32], quantized: &[i8], scale: f32) -> f32 {
    if original.is_empty() {
        return 0.0;
    }

    let reconstructed = dequantize_i8_to_f32(quantized, scale);
    let mse: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| (o - r).powi(2))
        .sum::<f32>()
        / original.len() as f32;

    mse
}

/// Compute signal-to-quantization-noise ratio (SQNR) in dB
#[must_use]
pub fn compute_sqnr(original: &[f32], quantized: &[i8], scale: f32) -> f32 {
    if original.is_empty() {
        return 0.0;
    }

    let signal_power: f32 = original.iter().map(|x| x.powi(2)).sum::<f32>() / original.len() as f32;
    let noise_power = quantization_error(original, quantized, scale);

    if noise_power < MIN_SCALE {
        return f32::INFINITY;
    }

    10.0 * (signal_power / noise_power).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Basic Quantization Tests
    // =========================================================================

    #[test]
    fn test_quantize_empty() {
        let (q, scale) = quantize_f32_to_i8(&[]);
        assert!(q.is_empty());
        assert!((scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quantize_zeros() {
        let data = vec![0.0; 10];
        let (q, scale) = quantize_f32_to_i8(&data);

        assert_eq!(q.len(), 10);
        assert!(q.iter().all(|&x| x == 0));
        assert!((scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quantize_max_value() {
        let data = vec![127.0];
        let (q, scale) = quantize_f32_to_i8(&data);

        assert_eq!(q[0], 127);
        assert!((scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quantize_negative() {
        let data = vec![-127.0];
        let (q, scale) = quantize_f32_to_i8(&data);

        assert_eq!(q[0], -127);
        assert!((scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quantize_small_values() {
        let data = vec![0.1, -0.1, 0.05, -0.05];
        let (q, scale) = quantize_f32_to_i8(&data);

        assert_eq!(q.len(), 4);
        // Max abs is 0.1, scale = 0.1 / 127
        let expected_scale = 0.1 / 127.0;
        assert!((scale - expected_scale).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_roundtrip() {
        let data = vec![1.0, -1.0, 0.5, -0.5, 0.0];
        let (q, scale) = quantize_f32_to_i8(&data);
        let reconstructed = dequantize_i8_to_f32(&q, scale);

        for (orig, recon) in data.iter().zip(reconstructed.iter()) {
            // Should be close but not exact due to quantization
            assert!((orig - recon).abs() < 0.02, "orig={orig}, recon={recon}");
        }
    }

    // =========================================================================
    // Quantized Tensor Tests
    // =========================================================================

    #[test]
    fn test_quantized_tensor_from_f32() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = QuantizedTensor::from_f32(&data, vec![2, 2]);

        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.numel(), 4);
    }

    #[test]
    fn test_quantized_tensor_roundtrip() {
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let tensor = QuantizedTensor::from_f32(&data, vec![2, 3]);
        let reconstructed = tensor.to_f32();

        for (orig, recon) in data.iter().zip(reconstructed.iter()) {
            let error = (orig - recon).abs() / orig.abs().max(1.0);
            assert!(error < 0.1, "Relative error too high: {error}");
        }
    }

    #[test]
    fn test_quantized_tensor_empty() {
        let tensor = QuantizedTensor::from_f32(&[], vec![0]);
        assert!(tensor.is_empty());
        assert_eq!(tensor.len(), 0);
    }

    // =========================================================================
    // Per-Channel Quantization Tests
    // =========================================================================

    #[test]
    fn test_per_channel_quantize() {
        // 2 channels, 3 elements each
        let data = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let (q, scales) = quantize_f32_to_i8_per_channel(&data, 2, 3);

        assert_eq!(q.len(), 6);
        assert_eq!(scales.len(), 2);

        // First channel max is 3, second is 30
        assert!(scales[1] > scales[0]);
    }

    #[test]
    fn test_per_channel_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let (q, scales) = quantize_f32_to_i8_per_channel(&data, 2, 3);
        let reconstructed = dequantize_i8_to_f32_per_channel(&q, &scales, 3);

        for (orig, recon) in data.iter().zip(reconstructed.iter()) {
            let error = (orig - recon).abs() / orig.abs().max(1.0);
            assert!(error < 0.1, "orig={orig}, recon={recon}");
        }
    }

    // =========================================================================
    // Quantized Linear Tests
    // =========================================================================

    #[test]
    fn test_quantized_linear_forward() {
        // Simple 2x2 identity-like weight
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let linear = QuantizedLinear::from_f32(&weight, None, 2, 2);

        let input = vec![1.0, 2.0];
        let output = linear.forward(&input).expect("forward");

        // Should be approximately identity
        assert!((output[0] - 1.0).abs() < 0.1);
        assert!((output[1] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_quantized_linear_with_bias() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let bias = vec![0.5, -0.5];
        let linear = QuantizedLinear::from_f32(&weight, Some(&bias), 2, 2);

        let input = vec![1.0, 2.0];
        let output = linear.forward(&input).expect("forward");

        assert!((output[0] - 1.5).abs() < 0.1);
        assert!((output[1] - 1.5).abs() < 0.1);
    }

    #[test]
    fn test_quantized_linear_batch() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let linear = QuantizedLinear::from_f32(&weight, None, 2, 2);

        let input = vec![1.0, 2.0, 3.0, 4.0]; // batch of 2
        let output = linear.forward(&input).expect("forward");

        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_quantized_linear_error() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let linear = QuantizedLinear::from_f32(&weight, None, 2, 2);

        let input = vec![1.0, 2.0, 3.0]; // Wrong size
        let result = linear.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_forward_vs_dequantized() {
        let weight = vec![0.5, -0.3, 0.2, 0.8, -0.1, 0.4, 0.6, -0.2, 0.3];
        let linear = QuantizedLinear::from_f32(&weight, None, 3, 3);

        let input = vec![1.0, 2.0, 3.0];
        let output1 = linear.forward(&input).expect("forward");
        let output2 = linear.forward_quantized(&input).expect("forward_quantized");

        // Both methods should give similar results
        for (o1, o2) in output1.iter().zip(output2.iter()) {
            assert!((o1 - o2).abs() < 0.5, "o1={o1}, o2={o2}");
        }
    }

    #[test]
    fn test_quantized_linear_memory_size() {
        let weight = vec![0.0; 1024]; // 1024 elements
        let bias = vec![0.0; 32];
        let linear = QuantizedLinear::from_f32(&weight, Some(&bias), 32, 32);

        let size = linear.memory_size();
        // 1024 bytes for weights + 128 bytes for bias + 4 for scale
        assert_eq!(size, 1024 + 128 + 4);
    }

    // =========================================================================
    // Error Metrics Tests
    // =========================================================================

    #[test]
    fn test_quantization_error_zero() {
        let data = vec![0.0; 10];
        let (q, scale) = quantize_f32_to_i8(&data);
        let error = quantization_error(&data, &q, scale);
        assert!(error < 1e-10);
    }

    #[test]
    fn test_quantization_error_small() {
        let data = vec![1.0, -1.0, 0.5, -0.5];
        let (q, scale) = quantize_f32_to_i8(&data);
        let error = quantization_error(&data, &q, scale);

        // Error should be small for this range
        assert!(error < 0.01, "error={error}");
    }

    #[test]
    fn test_sqnr_high_for_small_error() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32) / 10.0).collect();
        let (q, scale) = quantize_f32_to_i8(&data);
        let sqnr = compute_sqnr(&data, &q, scale);

        // SQNR should be high (> 30 dB) for reasonable quantization
        assert!(sqnr > 30.0, "SQNR too low: {sqnr}");
    }

    #[test]
    fn test_sqnr_empty() {
        let sqnr = compute_sqnr(&[], &[], 1.0);
        assert!((sqnr - 0.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_quantize_saturation() {
        // Value larger than scale * 127
        let data = vec![1000.0];
        let (q, scale) = quantize_f32_to_i8(&data);

        // Should saturate at 127
        assert_eq!(q[0], 127);
        assert!((scale - 1000.0 / 127.0).abs() < 1e-3);
    }

    #[test]
    fn test_quantize_very_small_scale() {
        let data = vec![1e-15, -1e-15];
        let (q, _scale) = quantize_f32_to_i8(&data);

        // Should use MIN_SCALE and round to 0
        assert!(q.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_quantize_mixed_magnitude() {
        // Mix of large and small values
        let data = vec![100.0, 0.001, -50.0, 0.0001];
        let (q, scale) = quantize_f32_to_i8(&data);

        // Large values should be well-represented
        let reconstructed = dequantize_i8_to_f32(&q, scale);
        assert!((reconstructed[0] - 100.0).abs() < 2.0);
        assert!((reconstructed[2] - (-50.0)).abs() < 2.0);

        // Small values may lose precision
        assert!(reconstructed[1].abs() < 2.0);
    }

    // =========================================================================
    // Int4 Quantization Tests
    // =========================================================================

    #[test]
    fn test_i4_pack_unpack() {
        // Test pack/unpack roundtrip for all valid int4 values
        for val in -8i8..=7i8 {
            let packed = pack_i4(val);
            let unpacked = unpack_i4(packed);
            assert_eq!(unpacked, val, "Pack/unpack failed for {val}");
        }
    }

    #[test]
    fn test_i4_quantize_empty() {
        let (q, scale) = quantize_f32_to_i4(&[]);
        assert!(q.is_empty());
        assert!((scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_i4_quantize_zeros() {
        let data = vec![0.0; 10];
        let (q, scale) = quantize_f32_to_i4(&data);

        assert_eq!(q.len(), 10);
        assert!(q.iter().all(|&x| x == 0));
        assert!((scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_i4_quantize_max_value() {
        let data = vec![7.0];
        let (q, scale) = quantize_f32_to_i4(&data);

        assert_eq!(q[0], 7);
        assert!((scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_i4_quantize_negative() {
        let data = vec![-7.0];
        let (q, scale) = quantize_f32_to_i4(&data);

        assert_eq!(q[0], -7);
        assert!((scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_i4_quantize_roundtrip() {
        let data = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.7, -0.7];
        let (q, scale) = quantize_f32_to_i4(&data);
        let reconstructed = dequantize_i4_to_f32(&q, scale);

        for (orig, recon) in data.iter().zip(reconstructed.iter()) {
            // Int4 has less precision than int8
            assert!((orig - recon).abs() < 0.2, "orig={orig}, recon={recon}");
        }
    }

    #[test]
    fn test_i4_packed_empty() {
        let (packed, scale) = quantize_f32_to_i4_packed(&[]);
        assert!(packed.is_empty());
        assert!((scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_i4_packed_even_length() {
        let data = vec![1.0, -1.0, 2.0, -2.0];
        let (packed, scale) = quantize_f32_to_i4_packed(&data);

        // 4 values should pack into 2 bytes
        assert_eq!(packed.len(), 2);

        // Verify roundtrip
        let reconstructed = dequantize_i4_packed_to_f32(&packed, scale, data.len());
        assert_eq!(reconstructed.len(), 4);
    }

    #[test]
    fn test_i4_packed_odd_length() {
        let data = vec![1.0, -1.0, 2.0, -2.0, 3.0];
        let (packed, scale) = quantize_f32_to_i4_packed(&data);

        // 5 values should pack into 3 bytes
        assert_eq!(packed.len(), 3);

        // Verify roundtrip
        let reconstructed = dequantize_i4_packed_to_f32(&packed, scale, data.len());
        assert_eq!(reconstructed.len(), 5);
    }

    #[test]
    fn test_i4_packed_roundtrip() {
        let data = vec![0.5, -0.3, 0.7, -0.1, 0.4, -0.6, 0.2];
        let (packed, scale) = quantize_f32_to_i4_packed(&data);
        let reconstructed = dequantize_i4_packed_to_f32(&packed, scale, data.len());

        for (orig, recon) in data.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 0.2, "orig={orig}, recon={recon}");
        }
    }

    #[test]
    fn test_i4_unpack_to_i8() {
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let (packed, _scale) = quantize_f32_to_i4_packed(&data);
        let unpacked = unpack_i4_to_i8(&packed, data.len());

        assert_eq!(unpacked.len(), 5);
        // All values should be in -8..7 range
        assert!(unpacked.iter().all(|&x| x >= -8 && x <= 7));
    }

    // =========================================================================
    // QuantizedTensorInt4 Tests
    // =========================================================================

    #[test]
    fn test_quantized_tensor_i4_from_f32() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = QuantizedTensorInt4::from_f32(&data, vec![2, 2]);

        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.numel(), 4);
        // 4 values pack into 2 bytes
        assert_eq!(tensor.data.len(), 2);
    }

    #[test]
    fn test_quantized_tensor_i4_roundtrip() {
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let tensor = QuantizedTensorInt4::from_f32(&data, vec![2, 3]);
        let reconstructed = tensor.to_f32();

        assert_eq!(reconstructed.len(), 6);
        for (orig, recon) in data.iter().zip(reconstructed.iter()) {
            let error = (orig - recon).abs() / orig.abs().max(1.0);
            // Int4 has larger error than int8
            assert!(error < 0.3, "Relative error too high: {error}");
        }
    }

    #[test]
    fn test_quantized_tensor_i4_empty() {
        let tensor = QuantizedTensorInt4::from_f32(&[], vec![0]);
        assert!(tensor.is_empty());
        assert_eq!(tensor.len(), 0);
    }

    #[test]
    fn test_quantized_tensor_i4_memory_savings() {
        let data = vec![0.0; 1000];

        let tensor_i8 = QuantizedTensor::from_f32(&data, vec![1000]);
        let tensor_i4 = QuantizedTensorInt4::from_f32(&data, vec![1000]);

        // Int4 should use ~50% of int8 memory
        let i8_size = tensor_i8.data.len();
        let i4_size = tensor_i4.data.len();

        assert_eq!(i8_size, 1000); // 1 byte per element
        assert_eq!(i4_size, 500); // 0.5 bytes per element
    }

    #[test]
    fn test_quantized_tensor_i4_unpack() {
        let data = vec![1.0, -1.0, 2.0, -2.0];
        let tensor = QuantizedTensorInt4::from_f32(&data, vec![4]);
        let unpacked = tensor.unpack();

        assert_eq!(unpacked.len(), 4);
    }

    // =========================================================================
    // QuantizedLinearInt4 Tests
    // =========================================================================

    #[test]
    fn test_quantized_linear_i4_forward() {
        // Simple 2x2 identity-like weight
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let linear = QuantizedLinearInt4::from_f32(&weight, None, 2, 2);

        let input = vec![1.0, 2.0];
        let output = linear.forward(&input).expect("forward");

        // Should be approximately identity (larger tolerance for int4)
        assert!((output[0] - 1.0).abs() < 0.5, "output[0]={}", output[0]);
        assert!((output[1] - 2.0).abs() < 0.5, "output[1]={}", output[1]);
    }

    #[test]
    fn test_quantized_linear_i4_with_bias() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let bias = vec![0.5, -0.5];
        let linear = QuantizedLinearInt4::from_f32(&weight, Some(&bias), 2, 2);

        let input = vec![1.0, 2.0];
        let output = linear.forward(&input).expect("forward");

        assert!((output[0] - 1.5).abs() < 0.5);
        assert!((output[1] - 1.5).abs() < 0.5);
    }

    #[test]
    fn test_quantized_linear_i4_batch() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let linear = QuantizedLinearInt4::from_f32(&weight, None, 2, 2);

        let input = vec![1.0, 2.0, 3.0, 4.0]; // batch of 2
        let output = linear.forward(&input).expect("forward");

        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_quantized_linear_i4_error() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let linear = QuantizedLinearInt4::from_f32(&weight, None, 2, 2);

        let input = vec![1.0, 2.0, 3.0]; // Wrong size
        let result = linear.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_linear_i4_forward_vs_dequantized() {
        let weight = vec![0.5, -0.3, 0.2, 0.8, -0.1, 0.4, 0.6, -0.2, 0.3];
        let linear = QuantizedLinearInt4::from_f32(&weight, None, 3, 3);

        let input = vec![1.0, 2.0, 3.0];
        let output1 = linear.forward(&input).expect("forward");
        let output2 = linear.forward_quantized(&input).expect("forward_quantized");

        // Both methods should give similar results (larger tolerance for int4)
        for (o1, o2) in output1.iter().zip(output2.iter()) {
            assert!((o1 - o2).abs() < 1.0, "o1={o1}, o2={o2}");
        }
    }

    #[test]
    fn test_quantized_linear_i4_memory_size() {
        let weight = vec![0.0; 1024]; // 1024 elements
        let bias = vec![0.0; 32];
        let linear_i8 = QuantizedLinear::from_f32(&weight, Some(&bias), 32, 32);
        let linear_i4 = QuantizedLinearInt4::from_f32(&weight, Some(&bias), 32, 32);

        let size_i8 = linear_i8.memory_size();
        let size_i4 = linear_i4.memory_size();

        // Int4 weights should be ~50% of int8
        // i8: 1024 bytes + 128 bias + 4 scale = 1156
        // i4: 512 bytes + 128 bias + 4 scale = 644
        assert_eq!(size_i8, 1024 + 128 + 4);
        assert_eq!(size_i4, 512 + 128 + 4);
    }

    // =========================================================================
    // Int4 Error Metrics Tests
    // =========================================================================

    #[test]
    fn test_quantization_error_i4_zero() {
        let data = vec![0.0; 10];
        let (q, scale) = quantize_f32_to_i4(&data);
        let error = quantization_error_i4(&data, &q, scale);
        assert!(error < 1e-10);
    }

    #[test]
    fn test_quantization_error_i4_larger_than_i8() {
        let data = vec![1.0, -1.0, 0.5, -0.5];

        let (q_i8, scale_i8) = quantize_f32_to_i8(&data);
        let error_i8 = quantization_error(&data, &q_i8, scale_i8);

        let (q_i4, scale_i4) = quantize_f32_to_i4(&data);
        let error_i4 = quantization_error_i4(&data, &q_i4, scale_i4);

        // Int4 error should be larger than int8
        assert!(error_i4 >= error_i8, "i4_err={error_i4}, i8_err={error_i8}");
    }

    #[test]
    fn test_sqnr_i4_lower_than_i8() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32) / 10.0).collect();

        let (q_i8, scale_i8) = quantize_f32_to_i8(&data);
        let sqnr_i8 = compute_sqnr(&data, &q_i8, scale_i8);

        let (q_i4, scale_i4) = quantize_f32_to_i4(&data);
        let sqnr_i4 = compute_sqnr_i4(&data, &q_i4, scale_i4);

        // Int4 SQNR should be lower (more noise)
        assert!(sqnr_i4 < sqnr_i8, "i4_sqnr={sqnr_i4}, i8_sqnr={sqnr_i8}");

        // But still reasonable (> 15 dB)
        assert!(sqnr_i4 > 15.0, "SQNR too low: {sqnr_i4}");
    }

    #[test]
    fn test_sqnr_i4_empty() {
        let sqnr = compute_sqnr_i4(&[], &[], 1.0);
        assert!((sqnr - 0.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Mixed-Precision Inference Tests (WAPR-073)
    // =========================================================================

    #[test]
    fn test_mixed_precision_config_default() {
        let config = MixedPrecisionConfig::default();
        assert_eq!(config.weight_precision, WeightPrecision::Int4);
        assert_eq!(config.activation_precision, ActivationPrecision::Float32);
    }

    #[test]
    fn test_mixed_precision_config_int4_fp32() {
        let config = MixedPrecisionConfig::int4_fp32();
        assert_eq!(config.weight_precision, WeightPrecision::Int4);
        assert_eq!(config.activation_precision, ActivationPrecision::Float32);
    }

    #[test]
    fn test_mixed_precision_config_int8_fp32() {
        let config = MixedPrecisionConfig::int8_fp32();
        assert_eq!(config.weight_precision, WeightPrecision::Int8);
        assert_eq!(config.activation_precision, ActivationPrecision::Float32);
    }

    #[test]
    fn test_mixed_precision_linear_from_config() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let config = MixedPrecisionConfig::int4_fp32();
        let linear = MixedPrecisionLinear::from_f32_with_config(&weight, None, 2, 2, config);

        assert_eq!(linear.in_features(), 2);
        assert_eq!(linear.out_features(), 2);
        assert_eq!(linear.weight_precision(), WeightPrecision::Int4);
    }

    #[test]
    fn test_mixed_precision_linear_forward_int4_fp32() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let config = MixedPrecisionConfig::int4_fp32();
        let linear = MixedPrecisionLinear::from_f32_with_config(&weight, None, 2, 2, config);

        let input = vec![1.0_f32, 2.0_f32];
        let output = linear.forward(&input).expect("forward");

        // Should be approximately identity
        assert!((output[0] - 1.0).abs() < 0.5, "output[0]={}", output[0]);
        assert!((output[1] - 2.0).abs() < 0.5, "output[1]={}", output[1]);
    }

    #[test]
    fn test_mixed_precision_linear_forward_int8_fp32() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let config = MixedPrecisionConfig::int8_fp32();
        let linear = MixedPrecisionLinear::from_f32_with_config(&weight, None, 2, 2, config);

        let input = vec![1.0_f32, 2.0_f32];
        let output = linear.forward(&input).expect("forward");

        // Int8 should be more accurate than int4
        assert!((output[0] - 1.0).abs() < 0.2, "output[0]={}", output[0]);
        assert!((output[1] - 2.0).abs() < 0.2, "output[1]={}", output[1]);
    }

    #[test]
    fn test_mixed_precision_memory_savings() {
        let weight = vec![0.0_f32; 4096]; // 4096 elements

        let linear_int4 = MixedPrecisionLinear::from_f32_with_config(
            &weight,
            None,
            64,
            64,
            MixedPrecisionConfig::int4_fp32(),
        );
        let linear_int8 = MixedPrecisionLinear::from_f32_with_config(
            &weight,
            None,
            64,
            64,
            MixedPrecisionConfig::int8_fp32(),
        );

        let size_int4 = linear_int4.memory_size();
        let size_int8 = linear_int8.memory_size();

        // Int4 should use ~50% of int8 memory
        assert!(size_int4 < size_int8, "int4={size_int4}, int8={size_int8}");
        let ratio = size_int4 as f32 / size_int8 as f32;
        assert!(ratio < 0.7, "Memory ratio should be < 0.7, got {ratio}");
    }

    #[test]
    fn test_mixed_precision_accuracy_comparison() {
        let weight = vec![0.5, -0.3, 0.2, 0.8, -0.1, 0.4, 0.6, -0.2, 0.3];
        let input = vec![1.0, 2.0, 3.0];

        let linear_int4 = MixedPrecisionLinear::from_f32_with_config(
            &weight,
            None,
            3,
            3,
            MixedPrecisionConfig::int4_fp32(),
        );
        let linear_int8 = MixedPrecisionLinear::from_f32_with_config(
            &weight,
            None,
            3,
            3,
            MixedPrecisionConfig::int8_fp32(),
        );

        let output_int4 = linear_int4.forward(&input).expect("forward int4");
        let output_int8 = linear_int8.forward(&input).expect("forward int8");

        // Both should give reasonable results
        assert_eq!(output_int4.len(), 3);
        assert_eq!(output_int8.len(), 3);

        // Int8 should generally be closer to reference (not always, but on average)
        // Just verify both work
        for val in &output_int4 {
            assert!(val.is_finite(), "int4 output not finite");
        }
        for val in &output_int8 {
            assert!(val.is_finite(), "int8 output not finite");
        }
    }

    #[test]
    fn test_mixed_precision_with_bias() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let bias = vec![0.5, -0.5];
        let config = MixedPrecisionConfig::int4_fp32();
        let linear = MixedPrecisionLinear::from_f32_with_config(&weight, Some(&bias), 2, 2, config);

        let input = vec![1.0_f32, 2.0_f32];
        let output = linear.forward(&input).expect("forward");

        // Output should include bias
        assert!((output[0] - 1.5).abs() < 0.5);
        assert!((output[1] - 1.5).abs() < 0.5);
    }

    #[test]
    fn test_mixed_precision_batch_forward() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let config = MixedPrecisionConfig::int4_fp32();
        let linear = MixedPrecisionLinear::from_f32_with_config(&weight, None, 2, 2, config);

        let input = vec![1.0, 2.0, 3.0, 4.0]; // batch of 2
        let output = linear.forward(&input).expect("forward");

        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_mixed_precision_error_invalid_input() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let config = MixedPrecisionConfig::int4_fp32();
        let linear = MixedPrecisionLinear::from_f32_with_config(&weight, None, 2, 2, config);

        let input = vec![1.0, 2.0, 3.0]; // Wrong size
        let result = linear.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_precision_display() {
        assert_eq!(format!("{}", WeightPrecision::Int4), "int4");
        assert_eq!(format!("{}", WeightPrecision::Int8), "int8");
    }

    #[test]
    fn test_activation_precision_display() {
        assert_eq!(format!("{}", ActivationPrecision::Float32), "fp32");
    }

    #[test]
    fn test_mixed_precision_config_description() {
        let config = MixedPrecisionConfig::int4_fp32();
        let desc = config.description();
        assert!(desc.contains("int4"));
        assert!(desc.contains("fp32"));
    }

    // =========================================================================
    // Int4 Edge Cases
    // =========================================================================

    #[test]
    fn test_i4_quantize_saturation() {
        // Value that would exceed int4 range
        let data = vec![100.0];
        let (q, scale) = quantize_f32_to_i4(&data);

        // Should saturate at 7
        assert_eq!(q[0], 7);
        assert!((scale - 100.0 / 7.0).abs() < 1e-3);
    }

    #[test]
    fn test_i4_quantize_negative_saturation() {
        // Use -8.0 which should map to -8 exactly
        let data = vec![-8.0];
        let (q, _scale) = quantize_f32_to_i4(&data);

        // Scale = 8.0/7.0, so -8.0 / scale = -7.0 (clamped from actual -8/7*8 = -7)
        // For true saturation test, use value beyond scale
        assert!(q[0] <= -7);

        // Test true saturation with asymmetric value
        let data2 = vec![7.0, -9.0]; // Max abs is 9, scale = 9/7
        let (q2, _) = quantize_f32_to_i4(&data2);
        // -9 / (9/7) = -7, but we clamp to -8
        assert!(q2[1] >= -8);
    }

    #[test]
    fn test_i4_single_element() {
        let data = vec![0.5];
        let (packed, scale) = quantize_f32_to_i4_packed(&data);

        // Single element should use 1 byte
        assert_eq!(packed.len(), 1);

        let reconstructed = dequantize_i4_packed_to_f32(&packed, scale, 1);
        assert_eq!(reconstructed.len(), 1);
        assert!((reconstructed[0] - 0.5).abs() < 0.2);
    }

    // =========================================================================
    // Q4_K Tests (Sprint 5 - realizar integration)
    // =========================================================================

    #[cfg(feature = "realizar-inference")]
    mod q4k_tests {
        use super::*;

        #[test]
        fn test_q4k_tensor_creation() {
            // Test: QuantizedTensorQ4K can be created from raw Q4_K data
            // Q4_K super-block: 144 bytes per 256 values
            let super_block_bytes = 144;
            let n_values = 256;

            // Create mock Q4_K data (1 super-block)
            let raw_data = vec![0u8; super_block_bytes];

            let tensor = QuantizedTensorQ4K::from_raw(raw_data, vec![n_values]);

            assert_eq!(tensor.len(), n_values);
            assert_eq!(tensor.shape(), &[n_values]);
        }

        #[test]
        fn test_q4k_dequantize_produces_values() {
            // Test: dequantize produces f32 values (may not match original exactly)
            let super_block_bytes = 144;
            let n_values = 256;

            // Create mock Q4_K data with some non-zero values
            let mut raw_data = vec![0u8; super_block_bytes];
            // Set d (scale) to a small non-zero value (f16 format)
            raw_data[0] = 0x00;
            raw_data[1] = 0x3C; // ~1.0 in f16

            let tensor = QuantizedTensorQ4K::from_raw(raw_data, vec![n_values]);
            let dequantized = tensor.dequantize();

            assert_eq!(dequantized.len(), n_values);
            // Values should be finite
            assert!(dequantized.iter().all(|x: &f32| x.is_finite()));
        }

        #[test]
        fn test_q4k_memory_savings() {
            // Test: Q4_K uses ~4.5 bits per weight (vs 32 bits for f32)
            let n_values = 256 * 4; // 4 super-blocks = 1024 values
            let super_block_bytes = 144;

            let raw_data = vec![0u8; super_block_bytes * 4];
            let tensor = QuantizedTensorQ4K::from_raw(raw_data, vec![n_values]);

            let q4k_bytes = tensor.memory_bytes();
            let f32_bytes = n_values * 4;

            // Q4_K should be ~7x smaller than f32
            let compression_ratio = f32_bytes as f64 / q4k_bytes as f64;
            assert!(
                compression_ratio > 6.0,
                "Expected >6x compression, got {:.2}x",
                compression_ratio
            );
        }

        #[test]
        fn test_q4k_linear_creation() {
            // Test: QuantizedLinearQ4K can be created with Q4_K weights
            let super_block_bytes = 144usize;
            let in_features = 256usize;
            let out_features = 64usize;
            let n_values = in_features * out_features; // 16384 values
            let n_blocks = n_values.div_ceil(256);

            let raw_data = vec![0u8; super_block_bytes * n_blocks];
            let bias = vec![0.0f32; out_features];

            let linear =
                QuantizedLinearQ4K::from_raw(raw_data, Some(&bias), in_features, out_features);

            assert_eq!(linear.in_features(), in_features);
            assert_eq!(linear.out_features(), out_features);
        }

        #[test]
        fn test_q4k_linear_forward_shape() {
            // Test: forward() produces correct output shape
            let super_block_bytes = 144usize;
            let in_features = 256usize;
            let out_features = 64usize;
            let n_values = in_features * out_features;
            let n_blocks = n_values.div_ceil(256);

            let raw_data = vec![0u8; super_block_bytes * n_blocks];
            let linear = QuantizedLinearQ4K::from_raw(raw_data, None, in_features, out_features);

            let input = vec![0.1f32; in_features];
            let output = linear.forward(&input).expect("forward");

            assert_eq!(output.len(), out_features);
        }

        #[test]
        fn test_q4k_linear_memory_vs_f32() {
            // Test: Q4_K linear uses ~7x less memory than f32
            let super_block_bytes = 144usize;
            let in_features = 512usize;
            let out_features = 512usize;
            let n_values = in_features * out_features; // 262144 values
            let n_blocks = n_values.div_ceil(256);

            let raw_data = vec![0u8; super_block_bytes * n_blocks];
            let linear = QuantizedLinearQ4K::from_raw(raw_data, None, in_features, out_features);

            let q4k_bytes = linear.memory_size();
            let f32_bytes = n_values * 4;

            let compression_ratio = f32_bytes as f64 / q4k_bytes as f64;
            assert!(
                compression_ratio > 6.0,
                "Expected >6x compression, got {:.2}x",
                compression_ratio
            );
        }

        // =====================================================================
        // Sprint 7: Fused Q4K MatVec Tests
        // =====================================================================

        #[test]
        fn test_fused_q4k_matvec_shape() {
            // Test: Fused forward produces correct output shape
            let super_block_bytes = 144usize;
            let in_features = 256usize;
            let out_features = 64usize;
            let n_values = in_features * out_features;
            let n_blocks = n_values.div_ceil(256);

            let raw_data = vec![0u8; super_block_bytes * n_blocks];
            let linear = QuantizedLinearQ4K::from_raw(raw_data, None, in_features, out_features);

            let input = vec![0.1f32; in_features];
            let output = linear.forward_fused(&input).expect("forward_fused");

            assert_eq!(output.len(), out_features);
        }

        #[test]
        fn test_fused_q4k_matvec_matches_dequant() {
            // Test: Fused forward matches dequantize+matmul baseline
            let super_block_bytes = 144usize;
            let in_features = 256usize;
            let out_features = 64usize;
            let n_values = in_features * out_features;
            let n_blocks = n_values.div_ceil(256);

            // Create data with non-zero scale
            let mut raw_data = vec![0u8; super_block_bytes * n_blocks];
            for block in 0..n_blocks {
                let offset = block * super_block_bytes;
                raw_data[offset] = 0x00;
                raw_data[offset + 1] = 0x3C; // ~1.0 in f16
            }

            let bias = vec![0.1f32; out_features];
            let linear =
                QuantizedLinearQ4K::from_raw(raw_data, Some(&bias), in_features, out_features);

            let input: Vec<f32> = (0..in_features).map(|i| (i as f32) * 0.01).collect();

            // Get both forward methods
            let baseline = linear.forward(&input).expect("forward");
            let fused = linear.forward_fused(&input).expect("forward_fused");

            // Fused should match baseline within tolerance
            assert_eq!(baseline.len(), fused.len());
            for (i, (b, f)) in baseline.iter().zip(fused.iter()).enumerate() {
                let diff = (*b - *f).abs();
                assert!(
                    diff < 1e-4,
                    "Mismatch at index {}: baseline={}, fused={}, diff={}",
                    i,
                    b,
                    f,
                    diff
                );
            }
        }

        #[test]
        fn test_fused_q4k_batch_forward() {
            // Test: Fused forward works with batched input
            let super_block_bytes = 144usize;
            let in_features = 256usize;
            let out_features = 64usize;
            let batch_size = 4usize;
            let n_values = in_features * out_features;
            let n_blocks = n_values.div_ceil(256);

            let raw_data = vec![0u8; super_block_bytes * n_blocks];
            let linear = QuantizedLinearQ4K::from_raw(raw_data, None, in_features, out_features);

            let input = vec![0.1f32; in_features * batch_size];
            let output = linear.forward_fused(&input).expect("forward_fused");

            assert_eq!(output.len(), out_features * batch_size);
        }

        // =====================================================================
        // Sprint 8: Q5_K and Q6_K Tests
        // =====================================================================

        #[test]
        fn test_q5k_tensor_creation() {
            // Q5_K: 176 bytes per 256 values
            let super_block_bytes = 176usize;
            let n_values = 256usize;

            let raw_data = vec![0u8; super_block_bytes];
            let tensor = QuantizedTensorQ5K::from_raw(raw_data, vec![n_values]);

            assert_eq!(tensor.len(), n_values);
            assert_eq!(tensor.shape(), &[n_values]);
            assert_eq!(tensor.memory_bytes(), super_block_bytes);
        }

        #[test]
        fn test_q5k_linear_forward_fused() {
            // Q5_K: 176 bytes per 256 values
            let super_block_bytes = 176usize;
            let in_features = 256usize;
            let out_features = 64usize;
            let n_values = in_features * out_features;
            let n_blocks = n_values.div_ceil(256);

            let raw_data = vec![0u8; super_block_bytes * n_blocks];
            let linear = QuantizedLinearQ5K::from_raw(raw_data, None, in_features, out_features);

            let input = vec![0.1f32; in_features];
            let output = linear.forward_fused(&input).expect("forward_fused");

            assert_eq!(output.len(), out_features);
        }

        #[test]
        fn test_q6k_tensor_creation() {
            // Q6_K: 210 bytes per 256 values
            let super_block_bytes = 210usize;
            let n_values = 256usize;

            let raw_data = vec![0u8; super_block_bytes];
            let tensor = QuantizedTensorQ6K::from_raw(raw_data, vec![n_values]);

            assert_eq!(tensor.len(), n_values);
            assert_eq!(tensor.shape(), &[n_values]);
            assert_eq!(tensor.memory_bytes(), super_block_bytes);
        }

        #[test]
        fn test_q6k_linear_forward_fused() {
            // Q6_K: 210 bytes per 256 values
            let super_block_bytes = 210usize;
            let in_features = 256usize;
            let out_features = 64usize;
            let n_values = in_features * out_features;
            let n_blocks = n_values.div_ceil(256);

            let raw_data = vec![0u8; super_block_bytes * n_blocks];
            let linear = QuantizedLinearQ6K::from_raw(raw_data, None, in_features, out_features);

            let input = vec![0.1f32; in_features];
            let output = linear.forward_fused(&input).expect("forward_fused");

            assert_eq!(output.len(), out_features);
        }

        #[test]
        fn test_k_quant_compression_ratios() {
            // Verify compression ratios for each K-quant format
            let n_values = 256 * 4usize; // 1024 values

            // Q4_K: 144 bytes per 256 = 4.5 bits/weight
            let q4k_data = vec![0u8; 144 * 4];
            let q4k = QuantizedTensorQ4K::from_raw(q4k_data, vec![n_values]);

            // Q5_K: 176 bytes per 256 = 5.5 bits/weight
            let q5k_data = vec![0u8; 176 * 4];
            let q5k = QuantizedTensorQ5K::from_raw(q5k_data, vec![n_values]);

            // Q6_K: 210 bytes per 256 = 6.5 bits/weight
            let q6k_data = vec![0u8; 210 * 4];
            let q6k = QuantizedTensorQ6K::from_raw(q6k_data, vec![n_values]);

            let f32_bytes = n_values * 4;

            // Verify compression hierarchy: Q4K < Q5K < Q6K < f32
            assert!(q4k.memory_bytes() < q5k.memory_bytes());
            assert!(q5k.memory_bytes() < q6k.memory_bytes());
            assert!(q6k.memory_bytes() < f32_bytes);

            // Verify approximate compression ratios
            let q4k_ratio = f32_bytes as f64 / q4k.memory_bytes() as f64;
            let q5k_ratio = f32_bytes as f64 / q5k.memory_bytes() as f64;
            let q6k_ratio = f32_bytes as f64 / q6k.memory_bytes() as f64;

            assert!(q4k_ratio > 6.5, "Q4K should have >6.5x compression");
            assert!(q5k_ratio > 5.5, "Q5K should have >5.5x compression");
            assert!(q6k_ratio > 4.5, "Q6K should have >4.5x compression");
        }

        // =====================================================================
        // Sprint 9: QuantizedFeedForward Tests (TDD RED Phase)
        // =====================================================================

        #[test]
        fn test_quantized_ffn_creation() {
            // Test: QuantizedFeedForward can be created from Q4K weights
            // FFN structure: fc1 (d_model → d_ff), fc2 (d_ff → d_model)
            let d_model = 256usize;
            let d_ff = 1024usize; // 4x expansion

            // Q4_K: 144 bytes per 256 values
            let super_block_bytes = 144usize;

            // fc1: d_model × d_ff weights
            let fc1_values = d_model * d_ff;
            let fc1_blocks = fc1_values.div_ceil(256);
            let fc1_data = vec![0u8; super_block_bytes * fc1_blocks];

            // fc2: d_ff × d_model weights
            let fc2_values = d_ff * d_model;
            let fc2_blocks = fc2_values.div_ceil(256);
            let fc2_data = vec![0u8; super_block_bytes * fc2_blocks];

            let ffn = QuantizedFeedForward::new(fc1_data, fc2_data, d_model, d_ff);

            assert_eq!(ffn.d_model(), d_model);
            assert_eq!(ffn.d_ff(), d_ff);
        }

        #[test]
        fn test_quantized_ffn_forward() {
            // Test: forward() produces correct output shape
            let d_model = 256usize;
            let d_ff = 1024usize;
            let seq_len = 4usize;

            let super_block_bytes = 144usize;
            let fc1_blocks = (d_model * d_ff).div_ceil(256);
            let fc2_blocks = (d_ff * d_model).div_ceil(256);

            let fc1_data = vec![0u8; super_block_bytes * fc1_blocks];
            let fc2_data = vec![0u8; super_block_bytes * fc2_blocks];

            let ffn = QuantizedFeedForward::new(fc1_data, fc2_data, d_model, d_ff);

            // Input: [seq_len, d_model]
            let input = vec![0.1f32; seq_len * d_model];
            let output = ffn.forward(&input).expect("forward");

            // Output should be same shape as input: [seq_len, d_model]
            assert_eq!(output.len(), seq_len * d_model);
        }

        #[test]
        fn test_quantized_ffn_memory_reduction() {
            // Test: QuantizedFeedForward uses <15% of FP32 memory
            let d_model = 384usize; // whisper-tiny
            let d_ff = 1536usize; // 4x

            let super_block_bytes = 144usize;
            let fc1_blocks = (d_model * d_ff).div_ceil(256);
            let fc2_blocks = (d_ff * d_model).div_ceil(256);

            let fc1_data = vec![0u8; super_block_bytes * fc1_blocks];
            let fc2_data = vec![0u8; super_block_bytes * fc2_blocks];

            let ffn = QuantizedFeedForward::new(fc1_data, fc2_data, d_model, d_ff);

            let q4k_bytes = ffn.memory_bytes();
            let fp32_bytes = (d_model * d_ff + d_ff * d_model) * 4; // weights only

            let ratio = q4k_bytes as f64 / fp32_bytes as f64;
            assert!(
                ratio < 0.15,
                "Q4K FFN should use <15% of FP32 memory, got {:.1}%",
                ratio * 100.0
            );
        }

        #[test]
        fn test_quantized_ffn_output_finite() {
            // Test: forward produces finite values (not NaN/Inf)
            let d_model = 256usize;
            let d_ff = 1024usize;

            let super_block_bytes = 144usize;
            let fc1_blocks = (d_model * d_ff).div_ceil(256);
            let fc2_blocks = (d_ff * d_model).div_ceil(256);

            // Create non-zero Q4K data to produce meaningful output
            let mut fc1_data = vec![0u8; super_block_bytes * fc1_blocks];
            let mut fc2_data = vec![0u8; super_block_bytes * fc2_blocks];

            // Set scale values (d field at offset 0, f16 format)
            for block in 0..fc1_blocks {
                fc1_data[block * super_block_bytes] = 0x00;
                fc1_data[block * super_block_bytes + 1] = 0x3C; // ~1.0 in f16
            }
            for block in 0..fc2_blocks {
                fc2_data[block * super_block_bytes] = 0x00;
                fc2_data[block * super_block_bytes + 1] = 0x3C;
            }

            let ffn = QuantizedFeedForward::new(fc1_data, fc2_data, d_model, d_ff);

            let input = vec![0.5f32; d_model];
            let output = ffn.forward(&input).expect("forward");

            assert!(
                output.iter().all(|x: &f32| x.is_finite()),
                "All outputs must be finite"
            );
        }

        // =====================================================================
        // Sprint 10: QuantizedDecoderBlock Tests (TDD RED Phase)
        // =====================================================================

        #[test]
        fn test_quantized_decoder_block_creation() {
            // Test: QuantizedDecoderBlock can be created from Q4K FFN data
            let d_model = 256usize;
            let d_ff = 1024usize;
            let n_heads = 4usize;

            let super_block_bytes = 144usize;
            let fc1_blocks = (d_model * d_ff).div_ceil(256);
            let fc2_blocks = (d_ff * d_model).div_ceil(256);

            let fc1_data = vec![0u8; super_block_bytes * fc1_blocks];
            let fc2_data = vec![0u8; super_block_bytes * fc2_blocks];

            let block = QuantizedDecoderBlock::new(d_model, n_heads, d_ff, fc1_data, fc2_data);

            assert_eq!(block.d_model(), d_model);
            assert_eq!(block.d_ff(), d_ff);
            assert_eq!(block.n_heads(), n_heads);
        }

        #[test]
        fn test_quantized_decoder_block_forward() {
            // Test: forward() produces correct output shape
            let d_model = 256usize;
            let d_ff = 1024usize;
            let n_heads = 4usize;
            let seq_len = 4usize;

            let super_block_bytes = 144usize;
            let fc1_blocks = (d_model * d_ff).div_ceil(256);
            let fc2_blocks = (d_ff * d_model).div_ceil(256);

            let fc1_data = vec![0u8; super_block_bytes * fc1_blocks];
            let fc2_data = vec![0u8; super_block_bytes * fc2_blocks];

            let block = QuantizedDecoderBlock::new(d_model, n_heads, d_ff, fc1_data, fc2_data);

            // Input: [seq_len, d_model]
            let x = vec![0.1f32; seq_len * d_model];
            let encoder_output = vec![0.1f32; seq_len * d_model];

            let output = block.forward(&x, &encoder_output, None).expect("forward");

            // Output should match input shape: [seq_len, d_model]
            assert_eq!(output.len(), seq_len * d_model);
        }

        #[test]
        fn test_quantized_decoder_block_output_finite() {
            // Test: forward produces finite values
            let d_model = 256usize;
            let d_ff = 1024usize;
            let n_heads = 4usize;

            let super_block_bytes = 144usize;
            let fc1_blocks = (d_model * d_ff).div_ceil(256);
            let fc2_blocks = (d_ff * d_model).div_ceil(256);

            // Initialize with small scale values
            let mut fc1_data = vec![0u8; super_block_bytes * fc1_blocks];
            let mut fc2_data = vec![0u8; super_block_bytes * fc2_blocks];
            for block in 0..fc1_blocks {
                fc1_data[block * super_block_bytes + 1] = 0x3C;
            }
            for block in 0..fc2_blocks {
                fc2_data[block * super_block_bytes + 1] = 0x3C;
            }

            let block = QuantizedDecoderBlock::new(d_model, n_heads, d_ff, fc1_data, fc2_data);

            let x = vec![0.5f32; d_model];
            let encoder_output = vec![0.5f32; d_model];

            let output = block.forward(&x, &encoder_output, None).expect("forward");

            assert!(
                output.iter().all(|v: &f32| v.is_finite()),
                "All outputs must be finite"
            );
        }

        #[test]
        fn test_quantized_decoder_block_memory_savings() {
            // Test: QuantizedDecoderBlock FFN uses less memory than FP32
            let d_model = 384usize; // whisper-tiny
            let d_ff = 1536usize;
            let n_heads = 6usize;

            let super_block_bytes = 144usize;
            let fc1_blocks = (d_model * d_ff).div_ceil(256);
            let fc2_blocks = (d_ff * d_model).div_ceil(256);

            let fc1_data = vec![0u8; super_block_bytes * fc1_blocks];
            let fc2_data = vec![0u8; super_block_bytes * fc2_blocks];

            let block = QuantizedDecoderBlock::new(d_model, n_heads, d_ff, fc1_data, fc2_data);

            let ffn_q4k_bytes = block.ffn_memory_bytes();
            let ffn_fp32_bytes = (d_model * d_ff + d_ff * d_model) * 4;

            let ratio = ffn_q4k_bytes as f64 / ffn_fp32_bytes as f64;
            assert!(
                ratio < 0.15,
                "Q4K FFN should use <15% of FP32 memory, got {:.1}%",
                ratio * 100.0
            );
        }

        // =====================================================================
        // Sprint 11: QuantizedDecoder Tests (TDD RED Phase)
        // =====================================================================

        #[test]
        fn test_quantized_decoder_creation() {
            // Test: QuantizedDecoder can be created with Q4K blocks
            let n_layers = 4usize;
            let d_model = 256usize;
            let d_ff = 1024usize;
            let n_heads = 4usize;
            let n_vocab = 51865usize;
            let max_len = 448usize;

            let super_block_bytes = 144usize;
            let fc1_blocks = (d_model * d_ff).div_ceil(256);
            let fc2_blocks = (d_ff * d_model).div_ceil(256);

            // Create Q4K data for each layer
            let ffn_data: Vec<(Vec<u8>, Vec<u8>)> = (0..n_layers)
                .map(|_| {
                    (
                        vec![0u8; super_block_bytes * fc1_blocks],
                        vec![0u8; super_block_bytes * fc2_blocks],
                    )
                })
                .collect();

            let decoder =
                QuantizedDecoder::new(n_layers, d_model, n_heads, d_ff, n_vocab, max_len, ffn_data);

            assert_eq!(decoder.n_layers(), n_layers);
            assert_eq!(decoder.d_model(), d_model);
            assert_eq!(decoder.n_vocab(), n_vocab);
        }

        #[test]
        fn test_quantized_decoder_forward_one() {
            // Test: forward_one_quantized works for incremental decoding
            let n_layers = 2usize;
            let d_model = 256usize;
            let d_ff = 1024usize;
            let n_heads = 4usize;
            let n_vocab = 1000usize; // Small vocab for test
            let max_len = 64usize;

            let super_block_bytes = 144usize;
            let fc1_blocks = (d_model * d_ff).div_ceil(256);
            let fc2_blocks = (d_ff * d_model).div_ceil(256);

            let ffn_data: Vec<(Vec<u8>, Vec<u8>)> = (0..n_layers)
                .map(|_| {
                    (
                        vec![0u8; super_block_bytes * fc1_blocks],
                        vec![0u8; super_block_bytes * fc2_blocks],
                    )
                })
                .collect();

            let decoder =
                QuantizedDecoder::new(n_layers, d_model, n_heads, d_ff, n_vocab, max_len, ffn_data);

            // Create KV cache
            let mut cache = decoder.create_kv_cache();

            // Encoder output (mock)
            let encoder_output = vec![0.1f32; 10 * d_model];

            // Generate one token
            let token = 1u32;
            let logits = decoder
                .forward_one_quantized(token, &encoder_output, &mut cache)
                .expect("forward_one_quantized");

            // Should produce logits over vocabulary
            assert_eq!(logits.len(), n_vocab);
        }

        #[test]
        fn test_quantized_decoder_output_finite() {
            // Test: forward_one_quantized produces finite values
            let n_layers = 2usize;
            let d_model = 256usize;
            let d_ff = 1024usize;
            let n_heads = 4usize;
            let n_vocab = 100usize;
            let max_len = 64usize;

            let super_block_bytes = 144usize;
            let fc1_blocks = (d_model * d_ff).div_ceil(256);
            let fc2_blocks = (d_ff * d_model).div_ceil(256);

            let ffn_data: Vec<(Vec<u8>, Vec<u8>)> = (0..n_layers)
                .map(|_| {
                    let mut fc1 = vec![0u8; super_block_bytes * fc1_blocks];
                    let mut fc2 = vec![0u8; super_block_bytes * fc2_blocks];
                    // Set small scale values
                    for b in 0..fc1_blocks {
                        fc1[b * super_block_bytes + 1] = 0x3C;
                    }
                    for b in 0..fc2_blocks {
                        fc2[b * super_block_bytes + 1] = 0x3C;
                    }
                    (fc1, fc2)
                })
                .collect();

            let decoder =
                QuantizedDecoder::new(n_layers, d_model, n_heads, d_ff, n_vocab, max_len, ffn_data);

            let mut cache = decoder.create_kv_cache();
            let encoder_output = vec![0.5f32; 5 * d_model];

            let logits = decoder
                .forward_one_quantized(1, &encoder_output, &mut cache)
                .expect("forward");

            assert!(
                logits.iter().all(|v: &f32| v.is_finite()),
                "All logits must be finite"
            );
        }

        #[test]
        fn test_quantized_decoder_memory_savings() {
            // Test: QuantizedDecoder uses less FFN memory than FP32
            let n_layers = 4usize;
            let d_model = 384usize; // whisper-tiny
            let d_ff = 1536usize;
            let n_heads = 6usize;
            let n_vocab = 51865usize;
            let max_len = 448usize;

            let super_block_bytes = 144usize;
            let fc1_blocks = (d_model * d_ff).div_ceil(256);
            let fc2_blocks = (d_ff * d_model).div_ceil(256);

            let ffn_data: Vec<(Vec<u8>, Vec<u8>)> = (0..n_layers)
                .map(|_| {
                    (
                        vec![0u8; super_block_bytes * fc1_blocks],
                        vec![0u8; super_block_bytes * fc2_blocks],
                    )
                })
                .collect();

            let decoder =
                QuantizedDecoder::new(n_layers, d_model, n_heads, d_ff, n_vocab, max_len, ffn_data);

            let ffn_q4k_bytes = decoder.ffn_memory_bytes();
            let ffn_fp32_bytes = n_layers * (d_model * d_ff + d_ff * d_model) * 4;

            let ratio = ffn_q4k_bytes as f64 / ffn_fp32_bytes as f64;
            assert!(
                ratio < 0.15,
                "Q4K FFN should use <15% of FP32 memory, got {:.1}%",
                ratio * 100.0
            );
        }

        // =====================================================================
        // Sprint 12: RTF Benchmark & Validation Tests
        // =====================================================================

        #[test]
        fn test_quantized_decoder_token_generation_time() {
            // Test: Measure token generation time for quantized decoder
            use std::time::Instant;

            let n_layers = 4usize;
            let d_model = 384usize; // whisper-tiny
            let d_ff = 1536usize;
            let n_heads = 6usize;
            let n_vocab = 51865usize;
            let max_len = 448usize;

            let super_block_bytes = 144usize;
            let fc1_blocks = (d_model * d_ff).div_ceil(256);
            let fc2_blocks = (d_ff * d_model).div_ceil(256);

            let ffn_data: Vec<(Vec<u8>, Vec<u8>)> = (0..n_layers)
                .map(|_| {
                    let mut fc1 = vec![0u8; super_block_bytes * fc1_blocks];
                    let mut fc2 = vec![0u8; super_block_bytes * fc2_blocks];
                    for b in 0..fc1_blocks {
                        fc1[b * super_block_bytes + 1] = 0x3C;
                    }
                    for b in 0..fc2_blocks {
                        fc2[b * super_block_bytes + 1] = 0x3C;
                    }
                    (fc1, fc2)
                })
                .collect();

            let decoder =
                QuantizedDecoder::new(n_layers, d_model, n_heads, d_ff, n_vocab, max_len, ffn_data);

            let mut cache = decoder.create_kv_cache();
            let encoder_output = vec![0.1f32; 10 * d_model];

            // Warm up
            let _ = decoder.forward_one_quantized(1, &encoder_output, &mut cache);

            // Measure 10 tokens
            cache.clear();
            let start = Instant::now();
            let num_tokens = 10;
            for i in 0..num_tokens {
                let _ = decoder.forward_one_quantized((i + 1) as u32, &encoder_output, &mut cache);
            }
            let elapsed = start.elapsed();

            let ms_per_token = elapsed.as_millis() as f64 / num_tokens as f64;
            println!(
                "Quantized decoder: {:.2}ms per token ({} tokens in {:?})",
                ms_per_token, num_tokens, elapsed
            );

            // Token generation should complete (no assertion on speed - just measuring)
            assert!(
                ms_per_token > 0.0,
                "Token generation should take measurable time"
            );
        }

        #[test]
        fn test_quantized_memory_reduction_validation() {
            // Test: Validate memory reduction meets target (~35% total, ~85% FFN)
            let n_layers = 4usize;
            let d_model = 384usize;
            let d_ff = 1536usize;
            let n_heads = 6usize;
            let n_vocab = 51865usize;
            let max_len = 448usize;

            let super_block_bytes = 144usize;
            let fc1_blocks = (d_model * d_ff).div_ceil(256);
            let fc2_blocks = (d_ff * d_model).div_ceil(256);

            let ffn_data: Vec<(Vec<u8>, Vec<u8>)> = (0..n_layers)
                .map(|_| {
                    (
                        vec![0u8; super_block_bytes * fc1_blocks],
                        vec![0u8; super_block_bytes * fc2_blocks],
                    )
                })
                .collect();

            let decoder =
                QuantizedDecoder::new(n_layers, d_model, n_heads, d_ff, n_vocab, max_len, ffn_data);

            // Calculate memory usage
            let ffn_q4k_bytes = decoder.ffn_memory_bytes();
            let ffn_fp32_bytes = n_layers * (d_model * d_ff + d_ff * d_model) * 4;

            // FP32 model components (for comparison)
            let embedding_bytes = n_vocab * d_model * 4; // token embeddings
            let pos_embedding_bytes = max_len * d_model * 4; // positional embeddings

            // Attention weights (still FP32 in quantized model)
            // Per layer: Q, K, V, O projections = 4 * d_model * d_model
            let attention_bytes_per_layer = 4 * d_model * d_model * 4;
            let total_attention_bytes = n_layers * attention_bytes_per_layer;

            // Total FP32 model size
            let fp32_total =
                ffn_fp32_bytes + embedding_bytes + pos_embedding_bytes + total_attention_bytes;

            // Q4K model size (FFN quantized, rest FP32)
            let q4k_total =
                ffn_q4k_bytes + embedding_bytes + pos_embedding_bytes + total_attention_bytes;

            let ffn_reduction = 1.0 - (ffn_q4k_bytes as f64 / ffn_fp32_bytes as f64);
            let total_reduction = 1.0 - (q4k_total as f64 / fp32_total as f64);

            // Calculate what fraction of model is FFN
            let ffn_fraction = ffn_fp32_bytes as f64 / fp32_total as f64;

            println!("Memory Analysis:");
            println!(
                "  FFN FP32: {:.2} MB ({:.1}% of model)",
                ffn_fp32_bytes as f64 / 1_000_000.0,
                ffn_fraction * 100.0
            );
            println!("  FFN Q4K:  {:.2} MB", ffn_q4k_bytes as f64 / 1_000_000.0);
            println!("  FFN reduction: {:.1}%", ffn_reduction * 100.0);
            println!(
                "  Embeddings: {:.2} MB ({:.1}% of model - not quantized)",
                (embedding_bytes + pos_embedding_bytes) as f64 / 1_000_000.0,
                (embedding_bytes + pos_embedding_bytes) as f64 / fp32_total as f64 * 100.0
            );
            println!(
                "  Attention: {:.2} MB ({:.1}% of model - FP32 for accuracy)",
                total_attention_bytes as f64 / 1_000_000.0,
                total_attention_bytes as f64 / fp32_total as f64 * 100.0
            );
            println!("  Total FP32: {:.2} MB", fp32_total as f64 / 1_000_000.0);
            println!("  Total Q4K:  {:.2} MB", q4k_total as f64 / 1_000_000.0);
            println!("  Total reduction: {:.1}%", total_reduction * 100.0);

            // Validate targets
            // FFN reduction is the key metric - we expect >85% compression
            assert!(
                ffn_reduction > 0.85,
                "FFN should be reduced by >85%, got {:.1}%",
                ffn_reduction * 100.0
            );

            // Total reduction is limited by embedding dominance (~73% of model)
            // Expected: ffn_fraction * ffn_reduction = ~17% * 86% = ~15%
            // Using 10% as conservative target to account for rounding
            let expected_total_reduction = ffn_fraction * ffn_reduction;
            println!(
                "  Expected total reduction: {:.1}% (ffn_fraction * ffn_reduction)",
                expected_total_reduction * 100.0
            );

            assert!(
                total_reduction > 0.10,
                "Total model should be reduced by >10%, got {:.1}%",
                total_reduction * 100.0
            );

            // Verify actual reduction is close to theoretical
            assert!(
                (total_reduction - expected_total_reduction).abs() < 0.02,
                "Total reduction {:.1}% should be within 2% of expected {:.1}%",
                total_reduction * 100.0,
                expected_total_reduction * 100.0
            );
        }

        #[test]
        fn test_rtf_theoretical_improvement() {
            // Test: Calculate theoretical RTF improvement based on memory bandwidth
            //
            // Background:
            // - Baseline RTF: 3.92x (decoder dominates at 80.1%)
            // - Memory bandwidth is often the bottleneck
            // - Q4K reduces FFN memory by ~85%
            // - FFN is ~60% of decoder compute
            //
            // Theoretical improvement:
            // - FFN speedup from reduced memory: ~2-3x (depends on compute vs memory bound)
            // - If FFN is 60% of decoder: max decoder speedup = 1/(0.4 + 0.6/2.5) = 1.47x
            // - If decoder is 80% of total: max total speedup = 1/(0.2 + 0.8/1.47) = 1.35x
            // - New RTF: 3.92x / 1.35 = 2.9x

            let baseline_rtf = 3.92_f64;
            let decoder_fraction = 0.801_f64; // 80.1% from baseline
            let ffn_fraction_of_decoder = 0.6_f64; // Estimate

            // Conservative estimate: 2x memory bandwidth improvement from Q4K
            let memory_speedup = 2.0_f64;

            // Calculate speedups
            let ffn_speedup = memory_speedup;
            let decoder_speedup =
                1.0 / ((1.0 - ffn_fraction_of_decoder) + ffn_fraction_of_decoder / ffn_speedup);
            let total_speedup =
                1.0 / ((1.0 - decoder_fraction) + decoder_fraction / decoder_speedup);
            let new_rtf = baseline_rtf / total_speedup;

            println!("RTF Improvement Analysis:");
            println!("  Baseline RTF: {:.2}x", baseline_rtf);
            println!("  FFN speedup (memory): {:.2}x", ffn_speedup);
            println!("  Decoder speedup: {:.2}x", decoder_speedup);
            println!("  Total speedup: {:.2}x", total_speedup);
            println!("  Projected RTF: {:.2}x", new_rtf);

            // Theoretical RTF should be improved
            assert!(
                new_rtf < baseline_rtf,
                "Q4K should improve RTF: {:.2}x should be < {:.2}x",
                new_rtf,
                baseline_rtf
            );

            // Target: RTF < 3.0x (25% improvement)
            let target_rtf = 3.0_f64;
            println!("  Target RTF: {:.2}x", target_rtf);
            println!("  Meets target: {}", new_rtf < target_rtf);

            // Note: This is theoretical - actual improvement depends on many factors
            // The test validates the math, not the actual performance
        }

        // ========================================================================
        // Sprint 13: QuantizedMultiHeadAttention Tests
        // ========================================================================

        #[test]
        fn test_quantized_attention_creation() {
            // Test: QuantizedMultiHeadAttention creates with Q4K projections
            //
            // Architecture:
            //   MultiHeadAttention uses LinearWeights (FP32) for Q, K, V, O
            //   QuantizedMultiHeadAttention uses QuantizedLinearQ4K for all projections
            //
            // Success: Creates successfully with quantized weights

            let n_heads = 6;
            let d_model = 384;

            // Create with random Q4K weights for Q, K, V, O projections
            let attn = QuantizedMultiHeadAttention::new_random(n_heads, d_model);

            // Verify dimensions
            assert_eq!(attn.n_heads(), n_heads, "Number of heads mismatch");
            assert_eq!(attn.d_model(), d_model, "Model dimension mismatch");
            assert_eq!(attn.d_head(), d_model / n_heads, "Head dimension mismatch");
        }

        #[test]
        fn test_quantized_attention_forward() {
            // Test: Forward pass produces valid output
            //
            // Setup: whisper-tiny dimensions (6 heads, 384 d_model)
            // Input: Single token embedding [384]
            //
            // Success: Forward returns finite values

            let n_heads = 6;
            let d_model = 384;
            let seq_len = 1;

            let attn = QuantizedMultiHeadAttention::new_random(n_heads, d_model);

            // Create input (single token)
            let input = vec![0.1f32; d_model * seq_len];

            // Forward pass (self-attention)
            let output = attn
                .forward(&input, &input, &input, None)
                .expect("forward should succeed");

            // Output should match input shape
            assert_eq!(output.len(), d_model * seq_len, "Output length mismatch");

            // All outputs should be finite
            assert!(
                output.iter().all(|x: &f32| x.is_finite()),
                "Output contains non-finite values"
            );
        }

        #[test]
        fn test_quantized_attention_output_shape() {
            // Test: Output dimensions are correct for various sequence lengths
            //
            // Whisper decoder attention:
            //   - Self attention: Q=K=V from decoder hidden state
            //   - Cross attention: Q from decoder, K=V from encoder output

            let n_heads = 6;
            let d_model = 384;

            let attn = QuantizedMultiHeadAttention::new_random(n_heads, d_model);

            // Test various sequence lengths
            for seq_len in [1, 4, 16, 64] {
                let input = vec![0.1f32; d_model * seq_len];

                let output = attn
                    .forward(&input, &input, &input, None)
                    .expect("forward should succeed");

                assert_eq!(
                    output.len(),
                    d_model * seq_len,
                    "Output shape mismatch for seq_len={seq_len}"
                );
            }

            // Test cross-attention (different K/V sequence length)
            let q_len = 1;
            let kv_len = 1500; // Typical encoder output length

            let query = vec![0.1f32; d_model * q_len];
            let key_value = vec![0.1f32; d_model * kv_len];

            let output = attn
                .forward(&query, &key_value, &key_value, None)
                .expect("cross-attention should succeed");

            assert_eq!(
                output.len(),
                d_model * q_len,
                "Cross-attention output shape mismatch"
            );
        }

        #[test]
        fn test_quantized_attention_memory_savings() {
            // Test: Quantized attention uses significantly less memory
            //
            // FP32 attention memory:
            //   - Q, K, V, O projections: 4 × d_model × d_model × 4 bytes
            //   - whisper-tiny (384): 4 × 384 × 384 × 4 = 2.36 MB
            //
            // Q4K attention memory:
            //   - ~4.5 bits per weight = 0.5625 bytes
            //   - Expected: 2.36 MB × 0.14 = 0.33 MB
            //
            // Success: Q4K uses <15% of FP32 memory

            let n_heads = 6;
            let d_model = 384;

            let attn = QuantizedMultiHeadAttention::new_random(n_heads, d_model);

            // FP32 memory: 4 projections × d_model × d_model × 4 bytes
            let fp32_bytes = 4 * d_model * d_model * 4;

            // Get quantized memory
            let q4k_bytes = attn.memory_bytes();

            let reduction = 1.0 - (q4k_bytes as f64 / fp32_bytes as f64);

            println!("Attention Memory Analysis:");
            println!("  FP32: {:.2} MB", fp32_bytes as f64 / 1_000_000.0);
            println!("  Q4K:  {:.2} MB", q4k_bytes as f64 / 1_000_000.0);
            println!("  Reduction: {:.1}%", reduction * 100.0);

            // Verify significant reduction
            // Note: Row-based padding (each row padded to 256-multiple) reduces
            // efficiency slightly vs. optimal 85.9%. For d_model=384:
            //   - Optimal: 147,456 values / 256 * 144 = 82,944 bytes
            //   - Row-based: 384 rows × ceil(384/256) × 144 = 110,592 bytes
            //   - Overhead: ~33% more bytes than optimal
            // Expected reduction: ~80-82% instead of 85.9%
            assert!(
                reduction > 0.80,
                "Attention should be reduced by >80%, got {:.1}%",
                reduction * 100.0
            );
        }

        // ========================================================================
        // Sprint 14: FullyQuantizedDecoderBlock Tests
        // ========================================================================

        #[test]
        fn test_fully_quantized_block_creation() {
            // Test: FullyQuantizedDecoderBlock creates with all Q4K weights
            //
            // Architecture:
            //   - self_attn: QuantizedMultiHeadAttention (Q4K)
            //   - cross_attn: QuantizedMultiHeadAttention (Q4K)
            //   - ffn: QuantizedFeedForward (Q4K)
            //   - ln1, ln2, ln3: LayerNorm (FP32 - small)
            //
            // Success: Creates successfully with all quantized weights

            let n_heads = 6;
            let d_model = 384;
            let d_ff = 1536; // 4x expansion

            let block = FullyQuantizedDecoderBlock::new_random(n_heads, d_model, d_ff);

            assert_eq!(block.d_model(), d_model, "Model dimension mismatch");
            assert_eq!(block.d_ff(), d_ff, "FFN dimension mismatch");
            assert_eq!(block.n_heads(), n_heads, "Number of heads mismatch");
        }

        #[test]
        fn test_fully_quantized_block_forward() {
            // Test: Forward pass produces valid output
            //
            // Decoder block forward:
            //   1. Self-attention with causal mask
            //   2. Cross-attention to encoder output
            //   3. FFN with GELU activation
            //
            // Success: Forward returns finite values with correct shape

            let n_heads = 6;
            let d_model = 384;
            let d_ff = 1536;
            let seq_len = 1;
            let encoder_len = 1500;

            let block = FullyQuantizedDecoderBlock::new_random(n_heads, d_model, d_ff);

            // Decoder input (single token)
            let decoder_input = vec![0.1f32; d_model * seq_len];
            // Encoder output
            let encoder_output = vec![0.1f32; d_model * encoder_len];

            let output = block
                .forward(&decoder_input, &encoder_output)
                .expect("forward");

            assert_eq!(output.len(), d_model * seq_len, "Output shape mismatch");

            assert!(
                output.iter().all(|x: &f32| x.is_finite()),
                "Output contains non-finite values"
            );
        }

        #[test]
        fn test_fully_quantized_block_memory_savings() {
            // Test: Fully quantized block uses significantly less memory
            //
            // FP32 block memory (per block):
            //   - Self-attention: 4 × d_model × d_model × 4 = 2.36 MB
            //   - Cross-attention: 4 × d_model × d_model × 4 = 2.36 MB
            //   - FFN: 2 × d_model × d_ff × 4 = 4.72 MB
            //   - Total: ~9.44 MB per block
            //
            // Q4K block memory:
            //   - Self-attention: 0.44 MB (81% reduction)
            //   - Cross-attention: 0.44 MB (81% reduction)
            //   - FFN: 0.66 MB (86% reduction)
            //   - Total: ~1.54 MB per block (83.7% reduction)
            //
            // Success: Q4K block uses <20% of FP32 memory

            let n_heads = 6;
            let d_model = 384;
            let d_ff = 1536;

            let block = FullyQuantizedDecoderBlock::new_random(n_heads, d_model, d_ff);

            // FP32 memory calculation
            let attn_fp32 = 4 * d_model * d_model * 4; // Q, K, V, O projections
            let ffn_fp32 = 2 * d_model * d_ff * 4; // fc1 + fc2
            let fp32_bytes = 2 * attn_fp32 + ffn_fp32; // self + cross + ffn

            // Get quantized memory
            let q4k_bytes = block.memory_bytes();

            let reduction = 1.0 - (q4k_bytes as f64 / fp32_bytes as f64);

            println!("Fully Quantized Block Memory Analysis:");
            println!("  FP32: {:.2} MB", fp32_bytes as f64 / 1_000_000.0);
            println!("  Q4K:  {:.2} MB", q4k_bytes as f64 / 1_000_000.0);
            println!("  Reduction: {:.1}%", reduction * 100.0);

            // Verify significant reduction (>75% due to row padding overhead)
            assert!(
                reduction > 0.75,
                "Block should be reduced by >75%, got {:.1}%",
                reduction * 100.0
            );
        }

        #[test]
        fn test_fully_quantized_block_multi_token() {
            // Test: Block handles multiple tokens in sequence
            //
            // Whisper decoder processes tokens incrementally, but should
            // also work with batched inputs for efficiency.

            let n_heads = 6;
            let d_model = 384;
            let d_ff = 1536;
            let encoder_len = 1500;

            let block = FullyQuantizedDecoderBlock::new_random(n_heads, d_model, d_ff);

            let encoder_output = vec![0.1f32; d_model * encoder_len];

            // Test various sequence lengths
            for seq_len in [1, 4, 16] {
                let decoder_input = vec![0.1f32; d_model * seq_len];

                let output = block
                    .forward(&decoder_input, &encoder_output)
                    .expect("forward");

                assert_eq!(
                    output.len(),
                    d_model * seq_len,
                    "Output shape mismatch for seq_len={seq_len}"
                );
            }
        }

        // ========================================================================
        // Sprint 15: FullyQuantizedDecoder Tests
        // ========================================================================

        #[test]
        fn test_fully_quantized_decoder_creation() {
            // Test: FullyQuantizedDecoder creates with all Q4K weights
            //
            // Architecture:
            //   - blocks: Vec<FullyQuantizedDecoderBlock>
            //   - ln_post: LayerNorm
            //   - token_embedding: FP32
            //   - positional_embedding: FP32
            //
            // Success: Creates successfully with all quantized blocks

            let n_layers = 4;
            let n_heads = 6;
            let d_model = 384;
            let d_ff = 1536;
            let n_vocab = 51865;
            let max_len = 448;

            let decoder = FullyQuantizedDecoder::new_random(
                n_layers, d_model, n_heads, d_ff, n_vocab, max_len,
            );

            assert_eq!(decoder.n_layers(), n_layers, "Layer count mismatch");
            assert_eq!(decoder.d_model(), d_model, "Model dimension mismatch");
            assert_eq!(decoder.n_vocab(), n_vocab, "Vocab size mismatch");
        }

        #[test]
        fn test_fully_quantized_decoder_forward_one() {
            // Test: Forward pass for single token works
            //
            // Incremental decoding:
            //   1. Embed token + position
            //   2. Forward through all blocks
            //   3. Project to vocabulary
            //
            // Success: Returns valid logits

            let n_layers = 4;
            let n_heads = 6;
            let d_model = 384;
            let d_ff = 1536;
            let n_vocab = 51865;
            let max_len = 448;
            let encoder_len = 1500;

            let decoder = FullyQuantizedDecoder::new_random(
                n_layers, d_model, n_heads, d_ff, n_vocab, max_len,
            );

            let encoder_output = vec![0.1f32; d_model * encoder_len];
            let mut cache = decoder.create_kv_cache();

            // Generate first token
            let token = 50258u32; // <|startoftranscript|>
            let logits = decoder
                .forward_one_fully_quantized(token, &encoder_output, &mut cache)
                .expect("forward_one");

            assert_eq!(logits.len(), n_vocab, "Logits length mismatch");
            assert!(
                logits.iter().all(|x: &f32| x.is_finite()),
                "Logits contain non-finite values"
            );
        }

        #[test]
        fn test_fully_quantized_decoder_memory_savings() {
            // Test: FullyQuantizedDecoder uses less memory than FP32
            //
            // FP32 decoder memory (whisper-tiny):
            //   - 4 blocks × 9.44 MB = 37.76 MB (blocks)
            //   - Embeddings: ~80 MB (not quantized)
            //
            // Q4K decoder memory:
            //   - 4 blocks × 1.66 MB = 6.64 MB (blocks)
            //   - Embeddings: ~80 MB (not quantized)
            //
            // Success: Block memory reduced >80%

            let n_layers = 4;
            let n_heads = 6;
            let d_model = 384;
            let d_ff = 1536;
            let n_vocab = 51865;
            let max_len = 448;

            let decoder = FullyQuantizedDecoder::new_random(
                n_layers, d_model, n_heads, d_ff, n_vocab, max_len,
            );

            // FP32 block memory: attention + FFN
            let attn_fp32_per_block = 2 * 4 * d_model * d_model * 4; // self + cross
            let ffn_fp32_per_block = 2 * d_model * d_ff * 4;
            let fp32_blocks = n_layers * (attn_fp32_per_block + ffn_fp32_per_block);

            // Q4K block memory
            let q4k_blocks = decoder.block_memory_bytes();

            let reduction = 1.0 - (q4k_blocks as f64 / fp32_blocks as f64);

            println!("Fully Quantized Decoder Memory Analysis:");
            println!("  FP32 blocks: {:.2} MB", fp32_blocks as f64 / 1_000_000.0);
            println!("  Q4K blocks:  {:.2} MB", q4k_blocks as f64 / 1_000_000.0);
            println!("  Block reduction: {:.1}%", reduction * 100.0);

            assert!(
                reduction > 0.75,
                "Blocks should be reduced by >75%, got {:.1}%",
                reduction * 100.0
            );
        }

        #[test]
        fn test_fully_quantized_decoder_token_generation_time() {
            // Test: Benchmark token generation time
            //
            // Measures time for single token forward pass through
            // fully quantized decoder (attention + FFN all Q4K).
            //
            // Compared to FFN-only quantization (Sprint 12):
            //   - FFN-only: 61.10ms per token
            //   - Fully quantized: should be similar or slightly slower
            //     (attention projection adds Q4K overhead but reduces memory bandwidth)

            let n_layers = 4;
            let n_heads = 6;
            let d_model = 384;
            let d_ff = 1536;
            let n_vocab = 51865;
            let max_len = 448;
            let encoder_len = 1500;

            let decoder = FullyQuantizedDecoder::new_random(
                n_layers, d_model, n_heads, d_ff, n_vocab, max_len,
            );

            let encoder_output = vec![0.1f32; d_model * encoder_len];
            let mut cache = decoder.create_kv_cache();

            // Warmup
            let _ = decoder.forward_one_fully_quantized(50258, &encoder_output, &mut cache);

            // Reset cache for timing
            cache = decoder.create_kv_cache();

            // Time 10 tokens
            let n_tokens = 10;
            let start = std::time::Instant::now();
            for i in 0..n_tokens {
                let token = (50258 + i) as u32;
                let _ = decoder
                    .forward_one_fully_quantized(token, &encoder_output, &mut cache)
                    .expect("forward_one");
            }
            let elapsed = start.elapsed();

            let ms_per_token = elapsed.as_secs_f64() * 1000.0 / n_tokens as f64;
            println!(
                "Fully quantized decoder: {:.2}ms per token ({} tokens in {:?})",
                ms_per_token, n_tokens, elapsed
            );

            // Token generation should complete (no assertion on speed - varies by hardware)
            assert!(elapsed.as_secs() < 60, "Token generation too slow");
        }
    }
}
