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
    let packed_len = (data.len() + 1) / 2;
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

        // Dequantize weights
        let weights = self.weight.to_f32();

        // Matrix multiply
        let mut output = vec![0.0_f32; batch_size * self.out_features];

        for b in 0..batch_size {
            for o in 0..self.out_features {
                let mut sum = 0.0_f32;
                for i in 0..self.in_features {
                    sum += input[b * self.in_features + i] * weights[o * self.in_features + i];
                }
                output[b * self.out_features + o] = sum;
            }
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
            WeightPrecision::Int4 => {
                QuantizedWeights::Int4(QuantizedTensorInt4::from_f32(
                    weight,
                    vec![out_features, in_features],
                ))
            }
            WeightPrecision::Int8 => {
                QuantizedWeights::Int8(QuantizedTensor::from_f32(
                    weight,
                    vec![out_features, in_features],
                ))
            }
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

        // Dequantize weights to fp32
        let weights = self.weights.to_f32();

        // Matrix multiply in fp32
        let mut output = vec![0.0_f32; batch_size * self.out_features];

        for b in 0..batch_size {
            for o in 0..self.out_features {
                let mut sum = 0.0_f32;
                for i in 0..self.in_features {
                    sum += input[b * self.in_features + i] * weights[o * self.in_features + i];
                }
                output[b * self.out_features + o] = sum;
            }
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

        // Dequantize weights
        let weights = self.weight.to_f32();

        // Matrix multiply
        let mut output = vec![0.0_f32; batch_size * self.out_features];

        for b in 0..batch_size {
            for o in 0..self.out_features {
                let mut sum = 0.0_f32;
                for i in 0..self.in_features {
                    sum += input[b * self.in_features + i] * weights[o * self.in_features + i];
                }
                output[b * self.out_features + o] = sum;
            }
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
        let linear = MixedPrecisionLinear::from_f32_with_config(
            &weight, None, 2, 2, config
        );

        assert_eq!(linear.in_features(), 2);
        assert_eq!(linear.out_features(), 2);
        assert_eq!(linear.weight_precision(), WeightPrecision::Int4);
    }

    #[test]
    fn test_mixed_precision_linear_forward_int4_fp32() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let config = MixedPrecisionConfig::int4_fp32();
        let linear = MixedPrecisionLinear::from_f32_with_config(
            &weight, None, 2, 2, config
        );

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
        let linear = MixedPrecisionLinear::from_f32_with_config(
            &weight, None, 2, 2, config
        );

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
            &weight, None, 64, 64, MixedPrecisionConfig::int4_fp32()
        );
        let linear_int8 = MixedPrecisionLinear::from_f32_with_config(
            &weight, None, 64, 64, MixedPrecisionConfig::int8_fp32()
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
            &weight, None, 3, 3, MixedPrecisionConfig::int4_fp32()
        );
        let linear_int8 = MixedPrecisionLinear::from_f32_with_config(
            &weight, None, 3, 3, MixedPrecisionConfig::int8_fp32()
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
        let linear = MixedPrecisionLinear::from_f32_with_config(
            &weight, Some(&bias), 2, 2, config
        );

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
        let linear = MixedPrecisionLinear::from_f32_with_config(
            &weight, None, 2, 2, config
        );

        let input = vec![1.0, 2.0, 3.0, 4.0]; // batch of 2
        let output = linear.forward(&input).expect("forward");

        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_mixed_precision_error_invalid_input() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let config = MixedPrecisionConfig::int4_fp32();
        let linear = MixedPrecisionLinear::from_f32_with_config(
            &weight, None, 2, 2, config
        );

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
}
