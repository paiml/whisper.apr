//! Model loading and caching for whisper-apr CLI
//!
//! This module handles automatic downloading of Whisper models from HuggingFace
//! when no local model path is provided.
//!
//! # Cache Location
//!
//! Models are cached in `~/.cache/whisper-apr/models/` following XDG conventions.
//!
//! # Supported Models
//!
//! - `openai/whisper-tiny` → tiny.apr
//! - `openai/whisper-base` → base.apr
//! - `openai/whisper-small` → small.apr
//! - `openai/whisper-medium` → medium.apr
//! - `openai/whisper-large-v3` → large.apr

use std::fs;
use std::path::PathBuf;

use super::args::ModelSize;
use crate::WhisperApr;

/// Model loader error
#[derive(Debug, thiserror::Error)]
pub enum ModelLoaderError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Model load error
    #[error("Model load error: {0}")]
    ModelLoad(#[from] crate::WhisperError),

    /// Download error
    #[error("Download error: {0}")]
    Download(String),

    /// Cache error
    #[error("Cache error: {0}")]
    Cache(String),
}

/// Result type for model loader operations
pub type ModelLoaderResult<T> = Result<T, ModelLoaderError>;

/// HuggingFace repository IDs for Whisper models
fn get_hf_repo_id(size: ModelSize) -> &'static str {
    match size {
        ModelSize::Tiny => "openai/whisper-tiny",
        ModelSize::Base => "openai/whisper-base",
        ModelSize::Small => "openai/whisper-small",
        ModelSize::Medium => "openai/whisper-medium",
        ModelSize::Large => "openai/whisper-large-v3",
    }
}

/// Get the filename for a model size
fn get_model_filename(size: ModelSize) -> &'static str {
    match size {
        ModelSize::Tiny => "tiny.apr",
        ModelSize::Base => "base.apr",
        ModelSize::Small => "small.apr",
        ModelSize::Medium => "medium.apr",
        ModelSize::Large => "large.apr",
    }
}

/// Get the cache directory for whisper-apr models
pub fn get_cache_dir() -> PathBuf {
    // Follow XDG conventions
    if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(xdg_cache).join("whisper-apr").join("models")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home)
            .join(".cache")
            .join("whisper-apr")
            .join("models")
    } else {
        // Fallback to current directory
        PathBuf::from(".cache").join("whisper-apr").join("models")
    }
}

/// Get the cache path for a specific model size
pub fn get_model_cache_path(size: ModelSize) -> PathBuf {
    get_cache_dir().join(get_model_filename(size))
}

/// Check if a model is already cached
pub fn is_model_cached(size: ModelSize) -> bool {
    let path = get_model_cache_path(size);
    path.exists() && path.metadata().map(|m| m.len() > 0).unwrap_or(false)
}

/// Download a model from HuggingFace Hub
///
/// This downloads the safetensors weights and converts them to .apr format.
fn download_model(size: ModelSize, verbose: bool) -> ModelLoaderResult<PathBuf> {
    use hf_hub::api::sync::Api;

    let repo_id = get_hf_repo_id(size);
    let cache_path = get_model_cache_path(size);

    if verbose {
        eprintln!("[INFO] Downloading model from HuggingFace: {}", repo_id);
    }

    // Ensure cache directory exists
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Use hf_hub to download
    let api = Api::new().map_err(|e| ModelLoaderError::Download(e.to_string()))?;
    let repo = api.model(repo_id.to_string());

    // Download model.safetensors
    let safetensors_path = repo
        .get("model.safetensors")
        .map_err(|e| ModelLoaderError::Download(format!("Failed to download model: {}", e)))?;

    if verbose {
        eprintln!(
            "[INFO] Downloaded safetensors to: {}",
            safetensors_path.display()
        );
    }

    // Convert safetensors to .apr format
    convert_safetensors_to_apr(&safetensors_path, &cache_path, size, verbose)?;

    Ok(cache_path)
}

/// Convert safetensors to .apr format
fn convert_safetensors_to_apr(
    safetensors_path: &std::path::Path,
    apr_path: &std::path::Path,
    size: ModelSize,
    verbose: bool,
) -> ModelLoaderResult<()> {
    use crate::format::AprWriter;
    use crate::model::ModelConfig;
    use safetensors::SafeTensors;

    if verbose {
        eprintln!("[INFO] Converting to .apr format...");
    }

    // Read safetensors file
    let data = fs::read(safetensors_path)?;
    let tensors =
        SafeTensors::deserialize(&data).map_err(|e| ModelLoaderError::Download(e.to_string()))?;

    // Get model config
    let config = match size {
        ModelSize::Tiny => ModelConfig::tiny(),
        ModelSize::Base => ModelConfig::base(),
        ModelSize::Small => ModelConfig::small(),
        ModelSize::Medium => ModelConfig::medium(),
        ModelSize::Large => ModelConfig::large(),
    };

    // Create APR writer
    let mut writer = AprWriter::from_config(&config);

    // Map tensor names from HuggingFace format to our format and write
    for (name, tensor) in tensors.tensors() {
        // Convert tensor data to f32
        let f32_data: Vec<f32> = match tensor.dtype() {
            safetensors::Dtype::F32 => tensor
                .data()
                .chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            safetensors::Dtype::F16 => tensor
                .data()
                .chunks(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect(),
            safetensors::Dtype::BF16 => tensor
                .data()
                .chunks(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect(),
            _ => {
                if verbose {
                    eprintln!("[WARN] Skipping tensor {} with unsupported dtype", name);
                }
                continue;
            }
        };

        // Map HuggingFace tensor name to our format
        let our_name = map_tensor_name(&name);

        // Get shape
        let shape: Vec<usize> = tensor.shape().to_vec();

        // Use the add() method which takes name, shape, data
        writer.add(our_name, shape, f32_data);
    }

    // Write to file
    let apr_data = writer
        .to_bytes()
        .map_err(|e| ModelLoaderError::Download(format!("Failed to write APR: {}", e)))?;
    fs::write(apr_path, apr_data)?;

    if verbose {
        eprintln!("[INFO] Saved model to: {}", apr_path.display());
    }

    Ok(())
}

/// Map HuggingFace tensor names to our internal format
fn map_tensor_name(hf_name: &str) -> String {
    // HuggingFace uses names like:
    // - encoder.conv1.weight
    // - decoder.layers.0.self_attn.k_proj.weight
    //
    // We use similar naming, just pass through
    hf_name.to_string()
}

/// Load a model, downloading from HuggingFace if not cached
///
/// # Arguments
///
/// * `size` - Model size to load
/// * `model_path` - Optional explicit path to .apr file
/// * `verbose` - Whether to print progress messages
///
/// # Returns
///
/// Loaded WhisperApr model with weights
pub fn load_or_download_model(
    size: ModelSize,
    model_path: Option<&std::path::Path>,
    verbose: bool,
) -> ModelLoaderResult<WhisperApr> {
    // If explicit path provided, use it
    if let Some(path) = model_path {
        if verbose {
            eprintln!("[INFO] Loading model from: {}", path.display());
        }
        let bytes = fs::read(path)?;
        let model = WhisperApr::load_from_apr(&bytes)?;
        return Ok(model);
    }

    // Check cache
    let cache_path = get_model_cache_path(size);

    if is_model_cached(size) {
        if verbose {
            eprintln!("[INFO] Loading cached model: {}", cache_path.display());
        }
        let bytes = fs::read(&cache_path)?;
        let model = WhisperApr::load_from_apr(&bytes)?;
        return Ok(model);
    }

    // Download and cache
    if verbose {
        eprintln!("[INFO] Model not cached, downloading...");
    }
    let downloaded_path = download_model(size, verbose)?;

    // Load the downloaded model
    let bytes = fs::read(&downloaded_path)?;
    let model = WhisperApr::load_from_apr(&bytes)?;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_cache_dir() {
        let cache_dir = get_cache_dir();
        let path_str = cache_dir.to_string_lossy();
        assert!(
            path_str.contains("whisper-apr"),
            "Cache dir should contain whisper-apr: {}",
            path_str
        );
    }

    #[test]
    fn test_get_model_cache_path() {
        let path = get_model_cache_path(ModelSize::Tiny);
        assert!(path.ends_with("tiny.apr"), "Path should end with tiny.apr");
    }

    #[test]
    fn test_get_hf_repo_id() {
        assert_eq!(get_hf_repo_id(ModelSize::Tiny), "openai/whisper-tiny");
        assert_eq!(get_hf_repo_id(ModelSize::Base), "openai/whisper-base");
        assert_eq!(get_hf_repo_id(ModelSize::Large), "openai/whisper-large-v3");
    }

    #[test]
    fn test_get_model_filename() {
        assert_eq!(get_model_filename(ModelSize::Tiny), "tiny.apr");
        assert_eq!(get_model_filename(ModelSize::Medium), "medium.apr");
    }
}
