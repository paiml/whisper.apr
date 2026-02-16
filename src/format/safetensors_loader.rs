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

/// Convert Moonshine HuggingFace tensor name to internal format
///
/// Maps 160 tensor names from `usefulsensors/moonshine-*` SafeTensors to
/// the internal naming used by `core_generated.rs` weight loading.
///
/// HF naming:
/// - `model.encoder.layers.{n}.self_attn.q_proj.weight`
/// - `model.decoder.layers.{n}.encoder_attn.k_proj.weight`
///
/// Internal naming:
/// - `encoder.blocks.{n}.attn.q.weight`
/// - `decoder.blocks.{n}.cross_attn.k.weight`
#[must_use]
pub fn map_moonshine_tensor_name(hf_name: &str) -> String {
    // Direct mappings (non-layer tensors)
    match hf_name {
        // Encoder conv stem
        "model.encoder.conv1.weight" => return "encoder.conv1.weight".into(),
        "model.encoder.conv2.weight" => return "encoder.conv2.weight".into(),
        "model.encoder.conv2.bias" => return "encoder.conv2.bias".into(),
        "model.encoder.conv3.weight" => return "encoder.conv3.weight".into(),
        "model.encoder.conv3.bias" => return "encoder.conv3.bias".into(),
        "model.encoder.groupnorm.weight" => return "encoder.groupnorm.weight".into(),
        "model.encoder.groupnorm.bias" => return "encoder.groupnorm.bias".into(),
        "model.encoder.layer_norm.weight" => return "encoder.layer_norm.weight".into(),
        // Decoder embeddings and final norm
        "model.decoder.embed_tokens.weight" => return "decoder.token_embedding.weight".into(),
        "model.decoder.norm.weight" => return "decoder.ln_post.weight".into(),
        // Output projection (tied to embed_tokens, may not exist in SafeTensors)
        "proj_out.weight" => return "decoder.proj_out.weight".into(),
        _ => {}
    }

    // Encoder layer patterns: model.encoder.layers.{n}.suffix
    if let Some(rest) = hf_name.strip_prefix("model.encoder.layers.") {
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];

            let mapped = match suffix {
                "input_layernorm.weight" => "ln1.weight",
                "self_attn.q_proj.weight" => "attn.q.weight",
                "self_attn.k_proj.weight" => "attn.k.weight",
                "self_attn.v_proj.weight" => "attn.v.weight",
                "self_attn.o_proj.weight" => "attn.o.weight",
                "post_attention_layernorm.weight" => "ln2.weight",
                "mlp.fc1.weight" => "ffn.fc1.weight",
                "mlp.fc1.bias" => "ffn.fc1.bias",
                "mlp.fc2.weight" => "ffn.fc2.weight",
                "mlp.fc2.bias" => "ffn.fc2.bias",
                other => other,
            };
            return format!("encoder.blocks.{layer_num}.{mapped}");
        }
    }

    // Decoder layer patterns: model.decoder.layers.{n}.suffix
    if let Some(rest) = hf_name.strip_prefix("model.decoder.layers.") {
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];

            let mapped = match suffix {
                // Self-attention
                "input_layernorm.weight" => "ln1.weight",
                "self_attn.q_proj.weight" => "attn.q.weight",
                "self_attn.k_proj.weight" => "attn.k.weight",
                "self_attn.v_proj.weight" => "attn.v.weight",
                "self_attn.o_proj.weight" => "attn.o.weight",
                // Cross-attention
                "post_attention_layernorm.weight" => "ln_cross.weight",
                "encoder_attn.q_proj.weight" => "cross_attn.q.weight",
                "encoder_attn.k_proj.weight" => "cross_attn.k.weight",
                "encoder_attn.v_proj.weight" => "cross_attn.v.weight",
                "encoder_attn.o_proj.weight" => "cross_attn.o.weight",
                // FFN
                "final_layernorm.weight" => "ln2.weight",
                "mlp.fc1.weight" => "ffn.fc1.weight",
                "mlp.fc1.bias" => "ffn.fc1.bias",
                "mlp.fc2.weight" => "ffn.fc2.weight",
                "mlp.fc2.bias" => "ffn.fc2.bias",
                other => other,
            };
            return format!("decoder.blocks.{layer_num}.{mapped}");
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
    /// Load safetensors from a path (file or directory)
    ///
    /// If path is a directory, loads sharded safetensors using index.json.
    /// If path is a file, loads single safetensors file.
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed
    pub fn load(path: impl AsRef<Path>) -> WhisperResult<Self> {
        let path = path.as_ref();

        // Check if path is a directory with sharded safetensors
        if path.is_dir() {
            return Self::load_directory(path);
        }

        let data = std::fs::read(path)?;

        // Parse to get tensor names
        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| WhisperError::Format(format!("safetensors parse error: {e}")))?;

        let tensor_names: Vec<String> = tensors.names().into_iter().map(String::from).collect();

        Ok(Self { data, tensor_names })
    }

    /// Load safetensors from a directory with sharded files
    ///
    /// This creates a ShardedSafeTensorsLoader internally and converts tensors
    /// one at a time to avoid loading all shards into a single memory buffer.
    ///
    /// # Errors
    /// Returns error if directory is missing required files
    fn load_directory(dir: &Path) -> WhisperResult<Self> {
        // For sharded models, we load the first shard to get started
        // The get_tensor_f32 method needs to be overridden for sharded loading
        // For now, return an error suggesting to use ShardedSafeTensorsLoader

        // Look for index file
        let index_path = dir.join("model.safetensors.index.json");
        if !index_path.exists() {
            return Err(WhisperError::Format(
                "Directory missing model.safetensors.index.json. Use ShardedSafeTensorsLoader for sharded models.".to_string(),
            ));
        }

        // Create the sharded loader
        let sharded = ShardedSafeTensorsLoader::load(dir)?;

        // For compatibility, we can't easily convert sharded to single-buffer
        // Return error with helpful message
        Err(WhisperError::Format(format!(
            "This is a sharded model with {} tensors across multiple files. Use ShardedSafeTensorsLoader::load() instead.",
            sharded.tensor_names().len()
        )))
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
// Sharded SafeTensors Loader
// =============================================================================

/// Loader for sharded HuggingFace safetensors models
///
/// Handles models split across multiple files (e.g., model-00001-of-00002.safetensors).
#[cfg(feature = "cli")]
#[derive(Debug)]
pub struct ShardedSafeTensorsLoader {
    /// Base directory containing shards
    base_dir: std::path::PathBuf,
    /// Tensor name to shard file mapping
    tensor_to_shard: std::collections::HashMap<String, String>,
    /// All tensor names
    tensor_names: Vec<String>,
}

#[cfg(feature = "cli")]
impl ShardedSafeTensorsLoader {
    /// Load sharded safetensors from a directory
    ///
    /// # Errors
    /// Returns error if directory is missing required files
    pub fn load(dir: impl AsRef<Path>) -> WhisperResult<Self> {
        let dir = dir.as_ref().to_path_buf();

        // Look for index file
        let index_path = dir.join("model.safetensors.index.json");
        if !index_path.exists() {
            return Err(WhisperError::Format(
                "Directory missing model.safetensors.index.json".to_string(),
            ));
        }

        // Parse index file
        let index_content = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_content)
            .map_err(|e| WhisperError::Format(format!("Invalid index.json: {e}")))?;

        let weight_map = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| WhisperError::Format("Missing weight_map in index.json".to_string()))?;

        // Build tensor to shard mapping
        let mut tensor_to_shard = std::collections::HashMap::new();
        let mut tensor_names = Vec::new();

        for (tensor_name, shard_file) in weight_map {
            if let Some(shard) = shard_file.as_str() {
                tensor_to_shard.insert(tensor_name.clone(), shard.to_string());
                tensor_names.push(tensor_name.clone());
            }
        }

        tensor_names.sort();

        Ok(Self {
            base_dir: dir,
            tensor_to_shard,
            tensor_names,
        })
    }

    /// Get list of tensor names
    #[must_use]
    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_names
    }

    /// Get total parameter count from metadata
    #[must_use]
    pub fn total_params(&self) -> Option<u64> {
        // Read from index.json metadata
        let index_path = self.base_dir.join("model.safetensors.index.json");
        let content = std::fs::read_to_string(&index_path).ok()?;
        let index: serde_json::Value = serde_json::from_str(&content).ok()?;
        index
            .get("metadata")
            .and_then(|m| m.get("total_parameters"))
            .and_then(|p| p.as_u64())
    }

    /// Get tensor data as f32
    ///
    /// # Errors
    /// Returns error if tensor not found or conversion fails
    pub fn get_tensor_f32(&self, name: &str) -> WhisperResult<(Vec<usize>, Vec<f32>)> {
        let shard_file = self
            .tensor_to_shard
            .get(name)
            .ok_or_else(|| WhisperError::Format(format!("Tensor not found: {name}")))?;

        let shard_path = self.base_dir.join(shard_file);
        let shard_data = std::fs::read(&shard_path)?;

        let tensors = SafeTensors::deserialize(&shard_data)
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
    /// * `quantize` - Whether to quantize weights
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
    fn test_map_tensor_name_attention_biases() {
        assert_eq!(
            map_tensor_name("model.layers.0.self_attn.q_proj.bias"),
            "layers.0.attn.q.bias"
        );
        assert_eq!(
            map_tensor_name("model.layers.0.self_attn.k_proj.bias"),
            "layers.0.attn.k.bias"
        );
        assert_eq!(
            map_tensor_name("model.layers.0.self_attn.v_proj.bias"),
            "layers.0.attn.v.bias"
        );
        assert_eq!(
            map_tensor_name("model.layers.0.self_attn.o_proj.bias"),
            "layers.0.attn.o.bias"
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
    fn test_map_tensor_name_ffn_biases() {
        assert_eq!(
            map_tensor_name("model.layers.0.mlp.gate_proj.bias"),
            "layers.0.ffn.gate.bias"
        );
        assert_eq!(
            map_tensor_name("model.layers.0.mlp.up_proj.bias"),
            "layers.0.ffn.up.bias"
        );
        assert_eq!(
            map_tensor_name("model.layers.0.mlp.down_proj.bias"),
            "layers.0.ffn.down.bias"
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
    fn test_map_tensor_name_layernorm_biases() {
        assert_eq!(
            map_tensor_name("model.layers.0.input_layernorm.bias"),
            "layers.0.ln1.bias"
        );
        assert_eq!(
            map_tensor_name("model.layers.0.post_attention_layernorm.bias"),
            "layers.0.ln2.bias"
        );
    }

    #[test]
    fn test_map_tensor_name_conv() {
        assert_eq!(
            map_tensor_name("model.layers.0.conv.weight"),
            "layers.0.conv.weight"
        );
        assert_eq!(
            map_tensor_name("model.layers.0.conv.bias"),
            "layers.0.conv.bias"
        );
    }

    #[test]
    fn test_map_tensor_name_layer_unknown_suffix() {
        // Unknown suffix within a layer passes through as-is
        assert_eq!(
            map_tensor_name("model.layers.0.some_new_thing.weight"),
            "layers.0.some_new_thing.weight"
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

    // =========================================================================
    // half_to_f32 edge cases for full branch coverage (PMAT-023)
    // =========================================================================

    #[test]
    fn test_half_to_f32_subnormal() {
        // Subnormal: exp == 0, mant != 0
        // Smallest subnormal: 0x0001 = 2^(-14) * 2^(-10) = 2^(-24)
        let val = half_to_f32(0x0001);
        assert!(val > 0.0);
        assert!(val < 1e-6);

        // Larger subnormal: 0x0200 = 2^(-14) * 0.5 = 2^(-15)
        let val2 = half_to_f32(0x0200);
        assert!(val2 > val);
    }

    #[test]
    fn test_half_to_f32_infinity_nan() {
        // Positive infinity: exp=31, mant=0 -> 0x7C00
        let inf = half_to_f32(0x7C00);
        assert!(inf.is_infinite());
        assert!(inf > 0.0);

        // Negative infinity: 0xFC00
        let neg_inf = half_to_f32(0xFC00);
        assert!(neg_inf.is_infinite());
        assert!(neg_inf < 0.0);

        // NaN: exp=31, mant!=0 -> 0x7C01
        let nan = half_to_f32(0x7C01);
        assert!(nan.is_nan());
    }

    #[test]
    fn test_half_to_f32_negative_zero() {
        // Negative zero: sign=1, exp=0, mant=0 -> 0x8000
        let neg_zero = half_to_f32(0x8000);
        assert_eq!(neg_zero, 0.0);
        assert!(neg_zero.is_sign_negative());
    }

    #[test]
    fn test_half_to_f32_negative_subnormal() {
        // Negative subnormal: sign=1, exp=0, mant!=0 -> 0x8001
        let val = half_to_f32(0x8001);
        assert!(val < 0.0);
        assert!(val > -1e-6);
    }

    #[test]
    fn test_bf16_to_f32_negative() {
        // bf16 -1.0 = 0xBF80
        let neg_one = bf16_to_f32(0xBF80);
        assert!((neg_one + 1.0).abs() < 1e-6);
    }

    // =========================================================================
    // ConversionStats Tests (WAPR-QA-003)
    // =========================================================================

    #[test]
    fn test_conversion_stats_with_compression() {
        let stats = ConversionStats {
            n_tensors: 10,
            n_params: 1000,
            input_bytes: 4000,
            output_bytes: 2000,
            compression_ratio: 0.0,
        };
        let stats = stats.with_compression();
        assert!((stats.compression_ratio - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_conversion_stats_with_compression_zero_input() {
        let stats = ConversionStats {
            n_tensors: 0,
            n_params: 0,
            input_bytes: 0,
            output_bytes: 0,
            compression_ratio: 0.0,
        };
        let stats = stats.with_compression();
        // Should not divide by zero
        assert!((stats.compression_ratio - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_conversion_stats_display_with_compression() {
        let stats = ConversionStats {
            n_tensors: 5,
            n_params: 500,
            input_bytes: 1024 * 1024,
            output_bytes: 512 * 1024,
            compression_ratio: 0.5,
        };
        let display = format!("{stats}");
        assert!(display.contains("5 tensors"));
        assert!(display.contains("500 params"));
        assert!(display.contains("compression"));
    }

    // =========================================================================
    // ConversionStats::with_compression additional coverage (WAPR-QA-005)
    // =========================================================================

    #[test]
    fn test_conversion_stats_with_compression_high_ratio() {
        let stats = ConversionStats {
            n_tensors: 50,
            n_params: 1_000_000,
            input_bytes: 10_000,
            output_bytes: 1_000,
            compression_ratio: 0.0, // unset
        };
        let stats = stats.with_compression();
        assert!(
            (stats.compression_ratio - 0.1).abs() < f32::EPSILON,
            "expected 0.1, got {}",
            stats.compression_ratio
        );
    }

    #[test]
    fn test_conversion_stats_with_compression_expansion() {
        // Output larger than input (expansion, not compression)
        let stats = ConversionStats {
            n_tensors: 1,
            n_params: 100,
            input_bytes: 100,
            output_bytes: 200,
            compression_ratio: 0.0,
        };
        let stats = stats.with_compression();
        assert!(
            (stats.compression_ratio - 2.0).abs() < f32::EPSILON,
            "expected 2.0 for expansion, got {}",
            stats.compression_ratio
        );
    }

    #[test]
    fn test_conversion_stats_default_then_with_compression() {
        let stats = ConversionStats::default().with_compression();
        // Default has input_bytes=0, so compression_ratio stays 0
        assert!(
            (stats.compression_ratio - 0.0).abs() < f32::EPSILON,
            "default with zero input should yield 0.0 ratio"
        );
    }

    // =========================================================================
    // Moonshine tensor name mapping tests (WAPR-MOONSHINE-011)
    // =========================================================================

    #[test]
    fn test_moonshine_map_encoder_conv_stem() {
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.conv1.weight"),
            "encoder.conv1.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.conv2.weight"),
            "encoder.conv2.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.conv2.bias"),
            "encoder.conv2.bias"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.conv3.weight"),
            "encoder.conv3.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.conv3.bias"),
            "encoder.conv3.bias"
        );
    }

    #[test]
    fn test_moonshine_map_encoder_norms() {
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.groupnorm.weight"),
            "encoder.groupnorm.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.groupnorm.bias"),
            "encoder.groupnorm.bias"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.layer_norm.weight"),
            "encoder.layer_norm.weight"
        );
    }

    #[test]
    fn test_moonshine_map_decoder_direct() {
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.embed_tokens.weight"),
            "decoder.token_embedding.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.norm.weight"),
            "decoder.ln_post.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("proj_out.weight"),
            "decoder.proj_out.weight"
        );
    }

    #[test]
    fn test_moonshine_map_encoder_layer_attention() {
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.layers.0.self_attn.q_proj.weight"),
            "encoder.blocks.0.attn.q.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.layers.0.self_attn.k_proj.weight"),
            "encoder.blocks.0.attn.k.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.layers.0.self_attn.v_proj.weight"),
            "encoder.blocks.0.attn.v.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.layers.0.self_attn.o_proj.weight"),
            "encoder.blocks.0.attn.o.weight"
        );
    }

    #[test]
    fn test_moonshine_map_encoder_layer_norms_mlp() {
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.layers.2.input_layernorm.weight"),
            "encoder.blocks.2.ln1.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.layers.2.post_attention_layernorm.weight"),
            "encoder.blocks.2.ln2.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.layers.3.mlp.fc1.weight"),
            "encoder.blocks.3.ffn.fc1.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.layers.3.mlp.fc1.bias"),
            "encoder.blocks.3.ffn.fc1.bias"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.layers.3.mlp.fc2.weight"),
            "encoder.blocks.3.ffn.fc2.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.layers.3.mlp.fc2.bias"),
            "encoder.blocks.3.ffn.fc2.bias"
        );
    }

    #[test]
    fn test_moonshine_map_decoder_layer_self_attn() {
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.0.input_layernorm.weight"),
            "decoder.blocks.0.ln1.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.1.self_attn.q_proj.weight"),
            "decoder.blocks.1.attn.q.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.1.self_attn.k_proj.weight"),
            "decoder.blocks.1.attn.k.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.1.self_attn.v_proj.weight"),
            "decoder.blocks.1.attn.v.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.1.self_attn.o_proj.weight"),
            "decoder.blocks.1.attn.o.weight"
        );
    }

    #[test]
    fn test_moonshine_map_decoder_layer_cross_attn() {
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.0.post_attention_layernorm.weight"),
            "decoder.blocks.0.ln_cross.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.0.encoder_attn.q_proj.weight"),
            "decoder.blocks.0.cross_attn.q.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.0.encoder_attn.k_proj.weight"),
            "decoder.blocks.0.cross_attn.k.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.0.encoder_attn.v_proj.weight"),
            "decoder.blocks.0.cross_attn.v.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.0.encoder_attn.o_proj.weight"),
            "decoder.blocks.0.cross_attn.o.weight"
        );
    }

    #[test]
    fn test_moonshine_map_decoder_layer_ffn() {
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.5.final_layernorm.weight"),
            "decoder.blocks.5.ln2.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.5.mlp.fc1.weight"),
            "decoder.blocks.5.ffn.fc1.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.5.mlp.fc1.bias"),
            "decoder.blocks.5.ffn.fc1.bias"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.5.mlp.fc2.weight"),
            "decoder.blocks.5.ffn.fc2.weight"
        );
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.5.mlp.fc2.bias"),
            "decoder.blocks.5.ffn.fc2.bias"
        );
    }

    #[test]
    fn test_moonshine_map_multi_digit_layer() {
        // Layer indices beyond single digits
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.layers.11.self_attn.q_proj.weight"),
            "encoder.blocks.11.attn.q.weight"
        );
    }

    #[test]
    fn test_moonshine_map_unknown_passthrough() {
        assert_eq!(
            map_moonshine_tensor_name("some.random.tensor"),
            "some.random.tensor"
        );
    }

    #[test]
    fn test_moonshine_map_unknown_encoder_suffix_passthrough() {
        // Unknown suffix within encoder layer passes through as-is
        assert_eq!(
            map_moonshine_tensor_name("model.encoder.layers.0.unknown_thing.weight"),
            "encoder.blocks.0.unknown_thing.weight"
        );
    }

    #[test]
    fn test_moonshine_map_unknown_decoder_suffix_passthrough() {
        // Unknown suffix within decoder layer passes through as-is
        assert_eq!(
            map_moonshine_tensor_name("model.decoder.layers.0.unknown_thing.weight"),
            "decoder.blocks.0.unknown_thing.weight"
        );
    }

    #[test]
    fn test_moonshine_map_all_160_tensor_coverage() {
        // Verify coverage of the full Moonshine tiny tensor set:
        // 68 encoder tensors + 92 decoder tensors = 160 total
        // Encoder per-layer: 10 tensors * 6 layers = 60, + 8 direct = 68
        // Decoder per-layer: 15 tensors * 6 layers = 90, + 2 direct = 92

        let encoder_per_layer_suffixes = [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ];

        let decoder_per_layer_suffixes = [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "encoder_attn.q_proj.weight",
            "encoder_attn.k_proj.weight",
            "encoder_attn.v_proj.weight",
            "encoder_attn.o_proj.weight",
            "final_layernorm.weight",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ];

        // Verify all encoder layers map without passthrough
        for layer in 0..6 {
            for suffix in &encoder_per_layer_suffixes {
                let hf = format!("model.encoder.layers.{layer}.{suffix}");
                let mapped = map_moonshine_tensor_name(&hf);
                assert!(
                    !mapped.contains("self_attn")
                        && !mapped.contains("encoder_attn")
                        && !mapped.contains("input_layernorm")
                        && !mapped.contains("post_attention_layernorm")
                        && !mapped.contains("final_layernorm")
                        && !mapped.contains("mlp."),
                    "encoder tensor {hf} was not properly mapped: {mapped}"
                );
            }
        }

        // Verify all decoder layers map without passthrough
        for layer in 0..6 {
            for suffix in &decoder_per_layer_suffixes {
                let hf = format!("model.decoder.layers.{layer}.{suffix}");
                let mapped = map_moonshine_tensor_name(&hf);
                assert!(
                    !mapped.contains("self_attn")
                        && !mapped.contains("encoder_attn")
                        && !mapped.contains("input_layernorm")
                        && !mapped.contains("post_attention_layernorm")
                        && !mapped.contains("final_layernorm")
                        && !mapped.contains("mlp."),
                    "decoder tensor {hf} was not properly mapped: {mapped}"
                );
            }
        }
    }
}
