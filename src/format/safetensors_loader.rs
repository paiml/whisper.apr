//! SafeTensors Loader for HuggingFace Models
//!
//! This module provides utilities for loading model weights from HuggingFace
//! safetensors format and converting them to APR2 format.
//!
//! # Usage
//!
//! ```rust,ignore
//! use whisper_apr::format::safetensors_loader::SafeTensorsLoader;
//!
//! let loader = SafeTensorsLoader::load("model.safetensors")?;
//! let weights = loader.get_tensor("model.embed_tokens.weight")?;
//! ```
//!
//! # Spec Reference
//!
//! See `docs/specifications/1.0-whisper-apr.md` Section 18.8 for conversion pipeline.

#[cfg(feature = "cli")]
use crate::error::{WhisperError, WhisperResult};
#[cfg(feature = "cli")]
use crate::format::apr2::{Apr2Writer, Lfm2Config, QuantConfig};

#[cfg(feature = "cli")]
use safetensors::SafeTensors;
#[cfg(feature = "cli")]
use std::path::Path;

/// Weight name mapping from HuggingFace to internal format
#[derive(Debug, Clone)]
pub struct WeightMapping {
    /// HuggingFace tensor name pattern
    pub hf_pattern: String,
    /// Internal tensor name pattern
    pub internal_pattern: String,
}

impl WeightMapping {
    /// Create a new weight mapping
    #[must_use]
    pub fn new(hf: impl Into<String>, internal: impl Into<String>) -> Self {
        Self {
            hf_pattern: hf.into(),
            internal_pattern: internal.into(),
        }
    }
}

/// LFM2 weight mappings from HuggingFace naming convention
///
/// HuggingFace LFM2 model uses names like:
/// - `model.embed_tokens.weight`
/// - `model.layers.{n}.self_attn.q_proj.weight`
/// - `model.layers.{n}.self_attn.k_proj.weight`
/// - `model.layers.{n}.self_attn.v_proj.weight`
/// - `model.layers.{n}.self_attn.o_proj.weight`
/// - `model.layers.{n}.mlp.gate_proj.weight`
/// - `model.layers.{n}.mlp.up_proj.weight`
/// - `model.layers.{n}.mlp.down_proj.weight`
/// - `model.layers.{n}.input_layernorm.weight`
/// - `model.layers.{n}.post_attention_layernorm.weight`
/// - `model.norm.weight`
/// - `lm_head.weight`
#[must_use]
pub fn lfm2_weight_mappings() -> Vec<WeightMapping> {
    vec![
        // Embeddings
        WeightMapping::new("model.embed_tokens.weight", "embed.weight"),
        // Output
        WeightMapping::new("model.norm.weight", "norm.weight"),
        WeightMapping::new("lm_head.weight", "lm_head.weight"),
        // Layer patterns are handled dynamically
    ]
}

/// Convert HuggingFace tensor name to internal format
///
/// # Examples
///
/// ```rust,ignore
/// let internal = map_tensor_name("model.layers.0.self_attn.q_proj.weight");
/// assert_eq!(internal, "layers.0.attn.q.weight");
/// ```
#[must_use]
pub fn map_tensor_name(hf_name: &str) -> String {
    // Direct mappings
    match hf_name {
        "model.embed_tokens.weight" => return "embed.weight".to_string(),
        "model.norm.weight" => return "norm.weight".to_string(),
        "lm_head.weight" => return "lm_head.weight".to_string(),
        _ => {}
    }

    // Layer patterns
    if let Some(rest) = hf_name.strip_prefix("model.layers.") {
        // Parse layer number
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];

            // Map attention weights
            let mapped_suffix = match suffix {
                "self_attn.q_proj.weight" => "attn.q.weight",
                "self_attn.k_proj.weight" => "attn.k.weight",
                "self_attn.v_proj.weight" => "attn.v.weight",
                "self_attn.o_proj.weight" => "attn.o.weight",
                "self_attn.q_proj.bias" => "attn.q.bias",
                "self_attn.k_proj.bias" => "attn.k.bias",
                "self_attn.v_proj.bias" => "attn.v.bias",
                "self_attn.o_proj.bias" => "attn.o.bias",
                // Map MLP/FFN weights (SwiGLU)
                "mlp.gate_proj.weight" => "ffn.gate.weight",
                "mlp.up_proj.weight" => "ffn.up.weight",
                "mlp.down_proj.weight" => "ffn.down.weight",
                "mlp.gate_proj.bias" => "ffn.gate.bias",
                "mlp.up_proj.bias" => "ffn.up.bias",
                "mlp.down_proj.bias" => "ffn.down.bias",
                // Map layer norms
                "input_layernorm.weight" => "ln1.weight",
                "post_attention_layernorm.weight" => "ln2.weight",
                "input_layernorm.bias" => "ln1.bias",
                "post_attention_layernorm.bias" => "ln2.bias",
                // Conv layers (LFM2 specific)
                "conv.weight" => "conv.weight",
                "conv.bias" => "conv.bias",
                // Pass through unknown
                other => other,
            };

            return format!("layers.{layer_num}.{mapped_suffix}");
        }
    }

    // Unknown pattern - pass through
    hf_name.to_string()
}

/// SafeTensors file loader
#[cfg(feature = "cli")]
#[derive(Debug)]
pub struct SafeTensorsLoader {
    /// Raw file data
    data: Vec<u8>,
    /// Tensor metadata cache
    tensor_names: Vec<String>,
}

#[cfg(feature = "cli")]
impl SafeTensorsLoader {
    /// Load safetensors file from path
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed
    pub fn load(path: impl AsRef<Path>) -> WhisperResult<Self> {
        let data = std::fs::read(path.as_ref())?;

        // Parse to get tensor names
        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| WhisperError::Format(format!("safetensors parse error: {e}")))?;

        let tensor_names: Vec<String> = tensors.names().into_iter().map(String::from).collect();

        Ok(Self { data, tensor_names })
    }

    /// Get list of tensor names
    #[must_use]
    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_names
    }

    /// Get tensor data as f32
    ///
    /// # Errors
    /// Returns error if tensor not found or conversion fails
    pub fn get_tensor_f32(&self, name: &str) -> WhisperResult<(Vec<usize>, Vec<f32>)> {
        let tensors = SafeTensors::deserialize(&self.data)
            .map_err(|e| WhisperError::Format(format!("safetensors parse error: {e}")))?;

        let tensor = tensors
            .tensor(name)
            .map_err(|e| WhisperError::Format(format!("tensor not found: {name}: {e}")))?;

        let shape: Vec<usize> = tensor.shape().to_vec();
        let dtype = tensor.dtype();
        let raw_data = tensor.data();

        // Convert to f32 based on dtype
        let f32_data = match dtype {
            safetensors::Dtype::F32 => raw_data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            safetensors::Dtype::F16 => raw_data
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half_to_f32(bits)
                })
                .collect(),
            safetensors::Dtype::BF16 => raw_data
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    bf16_to_f32(bits)
                })
                .collect(),
            other => {
                return Err(WhisperError::Format(format!(
                    "unsupported dtype: {other:?}"
                )));
            }
        };

        Ok((shape, f32_data))
    }

    /// Convert to APR2 format
    ///
    /// # Arguments
    /// * `config` - LFM2 model configuration
    /// * `quant` - Quantization configuration
    /// * `quantize` - Whether to quantize weights to int8
    ///
    /// # Errors
    /// Returns error if conversion fails
    pub fn to_apr2(
        &self,
        config: Lfm2Config,
        quant: QuantConfig,
        quantize: bool,
    ) -> WhisperResult<Apr2Writer> {
        let mut writer = Apr2Writer::lfm2(config, quant);

        for hf_name in &self.tensor_names {
            let internal_name = map_tensor_name(hf_name);
            let (shape, f32_data) = self.get_tensor_f32(hf_name)?;

            if quantize {
                writer.add_int8_quantized(&internal_name, shape, &f32_data);
            } else {
                writer.add_f32(&internal_name, shape, &f32_data);
            }
        }

        Ok(writer)
    }

    /// Get model config from safetensors metadata
    ///
    /// Note: This requires the config.json to be loaded separately.
    /// This method just returns default LFM2 config.
    #[must_use]
    pub fn default_lfm2_config() -> Lfm2Config {
        Lfm2Config::lfm2_2_6b()
    }

    /// Calculate total number of parameters across all tensors
    ///
    /// Returns the sum of products of all tensor shapes.
    ///
    /// # Errors
    /// Returns error if tensors cannot be parsed
    pub fn total_params(&self) -> WhisperResult<u64> {
        let tensors = SafeTensors::deserialize(&self.data)
            .map_err(|e| WhisperError::Format(format!("safetensors parse error: {e}")))?;

        let mut total: u64 = 0;
        for name in tensors.names() {
            if let Ok(tensor) = tensors.tensor(name) {
                let shape = tensor.shape();
                let params: u64 = shape.iter().map(|&d| d as u64).product();
                total = total.saturating_add(params);
            }
        }

        Ok(total)
    }
}

/// Convert half-precision (f16) bits to f32
#[cfg(any(feature = "cli", test))]
#[inline]
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut m = mant;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let new_exp = (127 - 15 + 1 + e) as u32;
            f32::from_bits((sign << 31) | (new_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        let new_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (new_exp << 23) | (mant << 13))
    }
}

/// Convert bfloat16 bits to f32
#[cfg(any(feature = "cli", test))]
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// =============================================================================
// Conversion Statistics
// =============================================================================

/// Statistics from conversion
#[derive(Debug, Clone, Default)]
pub struct ConversionStats {
    /// Number of tensors converted
    pub n_tensors: usize,
    /// Total parameters
    pub n_params: u64,
    /// Input size in bytes
    pub input_bytes: u64,
    /// Output size in bytes
    pub output_bytes: u64,
    /// Compression ratio
    pub compression_ratio: f32,
}

impl ConversionStats {
    /// Calculate compression ratio
    #[must_use]
    pub fn with_compression(mut self) -> Self {
        if self.input_bytes > 0 {
            self.compression_ratio = self.output_bytes as f32 / self.input_bytes as f32;
        }
        self
    }
}

impl core::fmt::Display for ConversionStats {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Converted {} tensors ({} params): {:.2} MB → {:.2} MB ({:.1}x compression)",
            self.n_tensors,
            self.n_params,
            self.input_bytes as f64 / (1024.0 * 1024.0),
            self.output_bytes as f64 / (1024.0 * 1024.0),
            1.0 / self.compression_ratio
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_tensor_name_embeddings() {
        assert_eq!(map_tensor_name("model.embed_tokens.weight"), "embed.weight");
        assert_eq!(map_tensor_name("model.norm.weight"), "norm.weight");
        assert_eq!(map_tensor_name("lm_head.weight"), "lm_head.weight");
    }

    #[test]
    fn test_map_tensor_name_attention() {
        assert_eq!(
            map_tensor_name("model.layers.0.self_attn.q_proj.weight"),
            "layers.0.attn.q.weight"
        );
        assert_eq!(
            map_tensor_name("model.layers.5.self_attn.k_proj.weight"),
            "layers.5.attn.k.weight"
        );
        assert_eq!(
            map_tensor_name("model.layers.29.self_attn.v_proj.weight"),
            "layers.29.attn.v.weight"
        );
        assert_eq!(
            map_tensor_name("model.layers.0.self_attn.o_proj.weight"),
            "layers.0.attn.o.weight"
        );
    }

    #[test]
    fn test_map_tensor_name_ffn() {
        assert_eq!(
            map_tensor_name("model.layers.0.mlp.gate_proj.weight"),
            "layers.0.ffn.gate.weight"
        );
        assert_eq!(
            map_tensor_name("model.layers.0.mlp.up_proj.weight"),
            "layers.0.ffn.up.weight"
        );
        assert_eq!(
            map_tensor_name("model.layers.0.mlp.down_proj.weight"),
            "layers.0.ffn.down.weight"
        );
    }

    #[test]
    fn test_map_tensor_name_layernorm() {
        assert_eq!(
            map_tensor_name("model.layers.0.input_layernorm.weight"),
            "layers.0.ln1.weight"
        );
        assert_eq!(
            map_tensor_name("model.layers.0.post_attention_layernorm.weight"),
            "layers.0.ln2.weight"
        );
    }

    #[test]
    fn test_map_tensor_name_unknown() {
        // Unknown patterns pass through
        assert_eq!(
            map_tensor_name("some.unknown.tensor"),
            "some.unknown.tensor"
        );
    }

    #[test]
    fn test_weight_mapping_new() {
        let mapping = WeightMapping::new("hf.name", "internal.name");
        assert_eq!(mapping.hf_pattern, "hf.name");
        assert_eq!(mapping.internal_pattern, "internal.name");
    }

    #[test]
    fn test_lfm2_weight_mappings() {
        let mappings = lfm2_weight_mappings();
        assert!(!mappings.is_empty());
        assert!(mappings
            .iter()
            .any(|m| m.hf_pattern == "model.embed_tokens.weight"));
    }

    #[test]
    fn test_conversion_stats_display() {
        let stats = ConversionStats {
            n_tensors: 100,
            n_params: 2_600_000_000,
            input_bytes: 5_200_000_000,
            output_bytes: 1_300_000_000,
            compression_ratio: 0.25,
        };

        let display = format!("{stats}");
        assert!(display.contains("100 tensors"));
        assert!(display.contains("4.0x compression"));
    }

    #[test]
    fn test_half_to_f32_values() {
        // Test zero
        assert_eq!(half_to_f32(0x0000), 0.0);

        // Test one (0x3C00)
        let one = half_to_f32(0x3C00);
        assert!((one - 1.0).abs() < 1e-6);

        // Test negative one (0xBC00)
        let neg_one = half_to_f32(0xBC00);
        assert!((neg_one + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32_values() {
        // bf16 1.0 = 0x3F80
        let one = bf16_to_f32(0x3F80);
        assert!((one - 1.0).abs() < 1e-6);

        // bf16 0.0 = 0x0000
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }
}
