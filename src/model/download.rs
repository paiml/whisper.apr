//! Model Download and Registry
//!
//! This module provides functionality for downloading models from HuggingFace
//! and managing the local model cache.
//!
//! # Supported Models
//!
//! ## Whisper (ASR)
//! - `openai/whisper-tiny` - 39M params
//! - `openai/whisper-base` - 74M params
//! - `openai/whisper-small` - 244M params
//!
//! ## LFM2 (Post-Transcription)
//! - `LiquidAI/LFM2-2.6B-Transcript` - 2.6B params for transcript summarization
//!
//! # Spec Reference
//!
//! See `docs/specifications/1.0-whisper-apr.md` Section 18 for LFM2 integration.

use std::fmt;

/// Model registry entry
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name (user-friendly)
    pub name: &'static str,
    /// HuggingFace repository ID
    pub repo_id: &'static str,
    /// Model family
    pub family: ModelFamily,
    /// Number of parameters
    pub params: &'static str,
    /// Description
    pub description: &'static str,
    /// Recommended quantization for WASM
    pub wasm_quant: &'static str,
    /// Estimated download size (fp16)
    pub size_fp16: &'static str,
    /// Estimated download size (int4)
    pub size_int4: &'static str,
}

/// Model family
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Whisper ASR models
    Whisper,
    /// LFM2 transcript summarization
    Lfm2,
    /// Moonshine ASR models (variable-length input)
    Moonshine,
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Whisper => write!(f, "whisper"),
            Self::Lfm2 => write!(f, "lfm2"),
            Self::Moonshine => write!(f, "moonshine"),
        }
    }
}

/// Available models registry
pub const MODELS: &[ModelInfo] = &[
    // Moonshine models (default ASR family - variable-length, efficient for short audio)
    ModelInfo {
        name: "moonshine-tiny",
        repo_id: "usefulsensors/moonshine-tiny",
        family: ModelFamily::Moonshine,
        params: "27.1M",
        description: "Default. Fast variable-length ASR, efficient for short audio",
        wasm_quant: "fp16",
        size_fp16: "54MB",
        size_int4: "14MB",
    },
    ModelInfo {
        name: "moonshine-base",
        repo_id: "usefulsensors/moonshine-base",
        family: ModelFamily::Moonshine,
        params: "61.5M",
        description: "Higher accuracy variable-length ASR",
        wasm_quant: "fp16",
        size_fp16: "123MB",
        size_int4: "31MB",
    },
    // Whisper models
    ModelInfo {
        name: "whisper-tiny",
        repo_id: "openai/whisper-tiny",
        family: ModelFamily::Whisper,
        params: "39M",
        description: "Fastest, lowest accuracy",
        wasm_quant: "fp16",
        size_fp16: "78MB",
        size_int4: "20MB",
    },
    ModelInfo {
        name: "whisper-base",
        repo_id: "openai/whisper-base",
        family: ModelFamily::Whisper,
        params: "74M",
        description: "Good balance of speed and accuracy",
        wasm_quant: "fp16",
        size_fp16: "148MB",
        size_int4: "37MB",
    },
    ModelInfo {
        name: "whisper-small",
        repo_id: "openai/whisper-small",
        family: ModelFamily::Whisper,
        params: "244M",
        description: "Higher accuracy, slower",
        wasm_quant: "int8",
        size_fp16: "488MB",
        size_int4: "122MB",
    },
    ModelInfo {
        name: "whisper-medium",
        repo_id: "openai/whisper-medium",
        family: ModelFamily::Whisper,
        params: "769M",
        description: "High accuracy (not recommended for WASM)",
        wasm_quant: "int4",
        size_fp16: "1.5GB",
        size_int4: "385MB",
    },
    ModelInfo {
        name: "whisper-large",
        repo_id: "openai/whisper-large-v3",
        family: ModelFamily::Whisper,
        params: "1.5B",
        description: "Best accuracy (not for WASM)",
        wasm_quant: "int4",
        size_fp16: "3.0GB",
        size_int4: "750MB",
    },
    ModelInfo {
        name: "whisper-large-v3-turbo",
        repo_id: "openai/whisper-large-v3-turbo",
        family: ModelFamily::Whisper,
        params: "809M",
        description: "Fast large model (32 enc + 4 dec layers)",
        wasm_quant: "int4",
        size_fp16: "1.6GB",
        size_int4: "404MB",
    },
    // LFM2 models
    ModelInfo {
        name: "lfm2-2.6b-transcript",
        repo_id: "LiquidAI/LFM2-2.6B-Transcript",
        family: ModelFamily::Lfm2,
        params: "2.6B",
        description: "Post-transcription summarization (WASM with int4)",
        wasm_quant: "int4-awq",
        size_fp16: "5.2GB",
        size_int4: "1.3GB",
    },
];

/// Find model by name
#[must_use]
pub fn find_model(name: &str) -> Option<&'static ModelInfo> {
    let name_lower = name.to_lowercase();
    MODELS.iter().find(|m| {
        m.name.to_lowercase() == name_lower
            || m.repo_id.to_lowercase() == name_lower
            || m.repo_id
                .to_lowercase()
                .ends_with(&format!("/{name_lower}"))
    })
}

/// List all models
#[must_use]
pub fn list_models() -> &'static [ModelInfo] {
    MODELS
}

/// List models by family
#[must_use]
pub fn list_models_by_family(family: ModelFamily) -> Vec<&'static ModelInfo> {
    MODELS.iter().filter(|m| m.family == family).collect()
}

/// Get default cache directory for models
#[must_use]
#[cfg(feature = "cli")]
pub fn default_cache_dir() -> std::path::PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("whisper-apr")
        .join("models")
}

/// Download status
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Bytes downloaded so far
    pub downloaded: u64,
    /// Total bytes (if known)
    pub total: Option<u64>,
    /// Current file being downloaded
    pub current_file: String,
}

/// Model downloader using HuggingFace Hub
#[cfg(feature = "cli")]
pub struct ModelDownloader {
    /// HuggingFace Hub API
    api: hf_hub::api::sync::Api,
    /// Cache directory
    cache_dir: std::path::PathBuf,
}

#[cfg(feature = "cli")]
impl ModelDownloader {
    /// Create new downloader with default cache
    ///
    /// # Errors
    /// Returns error if cache directory cannot be created
    pub fn new() -> crate::error::WhisperResult<Self> {
        let cache_dir = default_cache_dir();
        std::fs::create_dir_all(&cache_dir)?;

        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::error::WhisperError::Model(e.to_string()))?;

        Ok(Self { api, cache_dir })
    }

    /// Create downloader with custom cache directory
    ///
    /// # Errors
    /// Returns error if cache directory cannot be created
    pub fn with_cache_dir(cache_dir: std::path::PathBuf) -> crate::error::WhisperResult<Self> {
        std::fs::create_dir_all(&cache_dir)?;

        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::error::WhisperError::Model(e.to_string()))?;

        Ok(Self { api, cache_dir })
    }

    /// Download model from HuggingFace
    ///
    /// # Arguments
    /// * `model` - Model info to download
    /// * `filename` - Specific file to download (e.g., "model.safetensors")
    ///
    /// # Returns
    /// Path to downloaded file
    ///
    /// # Errors
    /// Returns error if download fails
    pub fn download(
        &self,
        model: &ModelInfo,
        filename: &str,
    ) -> crate::error::WhisperResult<std::path::PathBuf> {
        let repo = self.api.model(model.repo_id.to_string());

        let path = repo.get(filename).map_err(|e| {
            crate::error::WhisperError::Model(format!(
                "Failed to download {}/{}: {}",
                model.repo_id, filename, e
            ))
        })?;

        Ok(path)
    }

    /// Download all safetensors files for a model
    ///
    /// # Errors
    /// Returns error if download fails
    pub fn download_safetensors(
        &self,
        model: &ModelInfo,
    ) -> crate::error::WhisperResult<Vec<std::path::PathBuf>> {
        let repo = self.api.model(model.repo_id.to_string());

        // Try common safetensors filenames
        let filenames = [
            "model.safetensors",
            "pytorch_model.safetensors",
            "model-00001-of-00002.safetensors",
            "model-00001-of-00003.safetensors",
        ];

        let mut downloaded = Vec::new();

        for filename in filenames {
            if let Ok(path) = repo.get(filename) {
                downloaded.push(path);
            }
        }

        if downloaded.is_empty() {
            return Err(crate::error::WhisperError::Model(format!(
                "No safetensors files found in {}",
                model.repo_id
            )));
        }

        Ok(downloaded)
    }

    /// Get cache directory
    #[must_use]
    pub fn cache_dir(&self) -> &std::path::Path {
        &self.cache_dir
    }

    /// Check if model is cached
    #[must_use]
    pub fn is_cached(&self, model: &ModelInfo) -> bool {
        let model_dir = self.cache_dir.join(model.name);
        model_dir.exists() && model_dir.is_dir()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_model_by_name() {
        let model = find_model("whisper-tiny");
        assert!(model.is_some());
        let m = model.unwrap();
        assert_eq!(m.name, "whisper-tiny");
        assert_eq!(m.params, "39M");
    }

    #[test]
    fn test_find_model_by_repo_id() {
        let model = find_model("openai/whisper-base");
        assert!(model.is_some());
        assert_eq!(model.unwrap().name, "whisper-base");
    }

    #[test]
    fn test_find_model_case_insensitive() {
        let model = find_model("WHISPER-TINY");
        assert!(model.is_some());
        assert_eq!(model.unwrap().name, "whisper-tiny");
    }

    #[test]
    fn test_find_model_not_found() {
        let model = find_model("nonexistent-model");
        assert!(model.is_none());
    }

    #[test]
    fn test_list_models() {
        let models = list_models();
        assert!(!models.is_empty());
        assert!(models.len() >= 9); // 2 moonshine + 6 whisper + 1 lfm2
    }

    #[test]
    fn test_list_models_by_family_whisper() {
        let whisper_models = list_models_by_family(ModelFamily::Whisper);
        assert_eq!(whisper_models.len(), 6);
        for m in whisper_models {
            assert_eq!(m.family, ModelFamily::Whisper);
        }
    }

    #[test]
    fn test_list_models_by_family_moonshine() {
        let moonshine_models = list_models_by_family(ModelFamily::Moonshine);
        assert_eq!(moonshine_models.len(), 2);
        assert_eq!(moonshine_models[0].name, "moonshine-tiny");
        assert_eq!(moonshine_models[0].params, "27.1M");
        assert_eq!(moonshine_models[1].name, "moonshine-base");
        assert_eq!(moonshine_models[1].params, "61.5M");
    }

    #[test]
    fn test_list_models_by_family_lfm2() {
        let lfm2_models = list_models_by_family(ModelFamily::Lfm2);
        assert_eq!(lfm2_models.len(), 1);
        assert_eq!(lfm2_models[0].name, "lfm2-2.6b-transcript");
        assert_eq!(lfm2_models[0].params, "2.6B");
    }

    #[test]
    fn test_lfm2_model_info() {
        let model = find_model("lfm2-2.6b-transcript");
        assert!(model.is_some());
        let m = model.unwrap();
        assert_eq!(m.family, ModelFamily::Lfm2);
        assert_eq!(m.repo_id, "LiquidAI/LFM2-2.6B-Transcript");
        assert_eq!(m.wasm_quant, "int4-awq");
    }

    #[test]
    fn test_find_moonshine_model() {
        let model = find_model("moonshine-tiny");
        assert!(model.is_some());
        let m = model.map(|m| m).expect("should find moonshine-tiny");
        assert_eq!(m.name, "moonshine-tiny");
        assert_eq!(m.family, ModelFamily::Moonshine);
        assert_eq!(m.params, "27.1M");
    }

    #[test]
    fn test_model_family_display() {
        assert_eq!(format!("{}", ModelFamily::Whisper), "whisper");
        assert_eq!(format!("{}", ModelFamily::Lfm2), "lfm2");
        assert_eq!(format!("{}", ModelFamily::Moonshine), "moonshine");
    }

    #[test]
    fn test_all_models_have_required_fields() {
        for model in MODELS {
            assert!(!model.name.is_empty());
            assert!(!model.repo_id.is_empty());
            assert!(!model.params.is_empty());
            assert!(!model.description.is_empty());
            assert!(!model.wasm_quant.is_empty());
        }
    }
}
