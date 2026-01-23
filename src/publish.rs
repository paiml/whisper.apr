//! HuggingFace Hub Publishing Module (WAPR-PUB-001)
//!
//! Orchestrates publishing whisper.apr models to HuggingFace Hub.
//! Uses batuta::hf patterns for Hub API integration.
//!
//! # Stack Integration
//!
//! This module follows the patterns from:
//! - `batuta::hf::HubClient` for API interactions
//! - `pacha` for model signing
//!
//! # Workflow
//!
//! ```text
//! 1. Load APR model
//! 2. Export to SafeTensors (format::export)
//! 3. Verify model (verify module)
//! 4. Sign model (optional, via pacha)
//! 5. Upload to Hub
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{WhisperError, WhisperResult};
use crate::format::export::{SafeTensorsExporter, TensorData};

/// Configuration for publishing to HuggingFace Hub.
#[derive(Debug, Clone)]
pub struct PublishConfig {
    /// Repository ID (e.g., "paiml/whisper-apr-tiny")
    pub repo_id: String,
    /// Commit message
    pub commit_message: String,
    /// Whether to create the repo if it doesn't exist
    pub create_repo: bool,
    /// Whether to make the repo private
    pub private: bool,
    /// Model card content (README.md)
    pub model_card: Option<String>,
    /// Additional files to upload
    pub extra_files: Vec<PathBuf>,
}

impl Default for PublishConfig {
    fn default() -> Self {
        Self {
            repo_id: String::new(),
            commit_message: "Upload model".to_string(),
            create_repo: true,
            private: false,
            model_card: None,
            extra_files: Vec::new(),
        }
    }
}

impl PublishConfig {
    /// Create a new publish configuration.
    #[must_use]
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            ..Default::default()
        }
    }

    /// Set commit message.
    #[must_use]
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.commit_message = message.into();
        self
    }

    /// Set model card content.
    #[must_use]
    pub fn with_model_card(mut self, card: impl Into<String>) -> Self {
        self.model_card = Some(card.into());
        self
    }

    /// Set private flag.
    #[must_use]
    pub fn private(mut self, is_private: bool) -> Self {
        self.private = is_private;
        self
    }

    /// Add extra file to upload.
    #[must_use]
    pub fn with_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.extra_files.push(path.into());
        self
    }
}

/// Format options for publishing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishFormat {
    /// APR format only
    Apr,
    /// SafeTensors format only
    SafeTensors,
    /// Both formats
    Both,
}

impl Default for PublishFormat {
    fn default() -> Self {
        Self::Both
    }
}

/// Result of a publish operation.
#[derive(Debug, Clone)]
pub struct PublishResult {
    /// Repository URL
    pub repo_url: String,
    /// Commit SHA
    pub commit_sha: String,
    /// Files uploaded
    pub files_uploaded: Vec<String>,
    /// Total bytes uploaded
    pub total_bytes: usize,
}

/// Publisher for whisper.apr models.
pub struct Publisher {
    /// HuggingFace API token (from HF_TOKEN env var)
    token: Option<String>,
    /// Base API URL
    api_url: String,
}

impl Default for Publisher {
    fn default() -> Self {
        Self::new()
    }
}

impl Publisher {
    /// Create a new publisher, reading token from HF_TOKEN env var.
    #[must_use]
    pub fn new() -> Self {
        Self {
            token: std::env::var("HF_TOKEN").ok(),
            api_url: "https://huggingface.co/api".to_string(),
        }
    }

    /// Create publisher with explicit token.
    #[must_use]
    pub fn with_token(token: impl Into<String>) -> Self {
        Self {
            token: Some(token.into()),
            api_url: "https://huggingface.co/api".to_string(),
        }
    }

    /// Check if authentication is configured.
    #[must_use]
    pub fn is_authenticated(&self) -> bool {
        self.token.is_some()
    }

    /// Prepare model files for publishing.
    ///
    /// This exports the APR model to the requested formats and prepares
    /// all files for upload.
    pub fn prepare<P: AsRef<Path>>(
        &self,
        apr_path: P,
        output_dir: P,
        format: PublishFormat,
    ) -> WhisperResult<PreparedPublish> {
        let apr_path = apr_path.as_ref();
        let output_dir = output_dir.as_ref();

        // Create output directory
        std::fs::create_dir_all(output_dir).map_err(WhisperError::Io)?;

        let mut files = Vec::new();
        let mut total_bytes = 0usize;

        // Copy APR file if requested
        if matches!(format, PublishFormat::Apr | PublishFormat::Both) {
            let apr_name = apr_path
                .file_name()
                .ok_or_else(|| WhisperError::Format("Invalid APR path".to_string()))?;
            let dest = output_dir.join(apr_name);
            std::fs::copy(apr_path, &dest).map_err(WhisperError::Io)?;
            let size = std::fs::metadata(&dest).map_err(WhisperError::Io)?.len() as usize;
            total_bytes += size;
            files.push(dest);
        }

        // Export SafeTensors if requested
        if matches!(format, PublishFormat::SafeTensors | PublishFormat::Both) {
            let st_path = output_dir.join("model.safetensors");

            // For now, create a minimal SafeTensors with placeholder
            // In production, this would extract weights from the APR file
            let mut tensors = BTreeMap::new();
            let mut metadata = BTreeMap::new();

            // Add metadata
            metadata.insert("format".to_string(), "whisper.apr".to_string());
            metadata.insert(
                "source".to_string(),
                apr_path.to_string_lossy().to_string(),
            );

            // Placeholder tensor (in production, extract from APR)
            tensors.insert(
                "model.version".to_string(),
                TensorData::new(vec![0.2, 0.0], vec![2]),
            );

            SafeTensorsExporter::save_with_metadata(&st_path, &tensors, Some(metadata))?;

            let size = std::fs::metadata(&st_path).map_err(WhisperError::Io)?.len() as usize;
            total_bytes += size;
            files.push(st_path);
        }

        Ok(PreparedPublish {
            files,
            total_bytes,
            apr_path: apr_path.to_path_buf(),
        })
    }

    /// Publish prepared files to HuggingFace Hub.
    ///
    /// Note: This is a placeholder implementation. Full implementation
    /// requires the batuta::hf crate or direct HTTP API calls.
    pub fn publish(
        &self,
        _prepared: &PreparedPublish,
        config: &PublishConfig,
    ) -> WhisperResult<PublishResult> {
        // Verify authentication
        if !self.is_authenticated() {
            return Err(WhisperError::Auth(
                "HF_TOKEN not set. Set environment variable or use Publisher::with_token()"
                    .to_string(),
            ));
        }

        // Validate config
        if config.repo_id.is_empty() {
            return Err(WhisperError::Config("repo_id is required".to_string()));
        }

        // In production, this would:
        // 1. Create repo if needed (POST /api/repos/create)
        // 2. Upload files via Git LFS
        // 3. Create commit

        // For now, return a placeholder result
        // Full implementation uses batuta::hf::HubClient
        Ok(PublishResult {
            repo_url: format!("https://huggingface.co/{}", config.repo_id),
            commit_sha: "placeholder".to_string(),
            files_uploaded: _prepared
                .files
                .iter()
                .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
                .collect(),
            total_bytes: _prepared.total_bytes,
        })
    }
}

/// Prepared files ready for publishing.
#[derive(Debug, Clone)]
pub struct PreparedPublish {
    /// Files to upload
    pub files: Vec<PathBuf>,
    /// Total size in bytes
    pub total_bytes: usize,
    /// Original APR path
    pub apr_path: PathBuf,
}

/// Generate a default model card for whisper.apr models.
#[must_use]
pub fn generate_model_card(model_name: &str, model_size: &str) -> String {
    format!(
        r#"---
license: mit
language:
  - en
  - multilingual
tags:
  - whisper
  - speech-recognition
  - audio
  - automatic-speech-recognition
  - rust
  - wasm
library_name: whisper-apr
pipeline_tag: automatic-speech-recognition
---

# {model_name}

Pure Rust implementation of OpenAI Whisper ({model_size}) optimized for WebAssembly.

## Formats Available

| Format | Description |
|--------|-------------|
| `*.apr` | Native APR format (quantized, streaming) |
| `model.safetensors` | HuggingFace standard format |

## Usage (Rust/WASM)

```rust
use whisper_apr::WhisperApr;

let model = WhisperApr::from_file("model.apr")?;
let result = model.transcribe(&audio)?;
println!("{{}}", result.text);
```

## Provenance

- **Stack**: PAIML Sovereign AI (whisper.apr, trueno, aprender)
- **Format**: APR v2 with Int8 quantization

## Citation

```bibtex
@software{{whisper_apr,
  title = {{whisper.apr: WASM-First Whisper Implementation}},
  author = {{PAIML}},
  year = {{2024}},
  url = {{https://github.com/paiml/whisper.apr}}
}}
```
"#,
        model_name = model_name,
        model_size = model_size
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_publish_config_builder() {
        let config = PublishConfig::new("paiml/test-model")
            .with_message("Initial upload")
            .private(true)
            .with_file("extra.txt");

        assert_eq!(config.repo_id, "paiml/test-model");
        assert_eq!(config.commit_message, "Initial upload");
        assert!(config.private);
        assert_eq!(config.extra_files.len(), 1);
    }

    #[test]
    fn test_publish_config_default() {
        let config = PublishConfig::default();
        assert!(config.repo_id.is_empty());
        assert_eq!(config.commit_message, "Upload model");
        assert!(config.create_repo);
        assert!(!config.private);
        assert!(config.model_card.is_none());
        assert!(config.extra_files.is_empty());
    }

    #[test]
    fn test_publish_config_with_model_card() {
        let config = PublishConfig::new("test/repo")
            .with_model_card("# My Model Card");
        assert_eq!(config.model_card.as_deref(), Some("# My Model Card"));
    }

    #[test]
    fn test_publish_config_multiple_files() {
        let config = PublishConfig::new("test/repo")
            .with_file("file1.txt")
            .with_file("file2.txt")
            .with_file("file3.txt");
        assert_eq!(config.extra_files.len(), 3);
    }

    #[test]
    fn test_publisher_default() {
        let pub1 = Publisher::default();
        // Default should be same as new()
        let pub2 = Publisher::new();
        // Both may or may not be authenticated depending on env
        assert_eq!(pub1.is_authenticated(), pub2.is_authenticated());
    }

    #[test]
    fn test_publisher_auth_check() {
        // With explicit token
        let pub2 = Publisher::with_token("test_token");
        assert!(pub2.is_authenticated());
    }

    #[test]
    fn test_publisher_empty_token() {
        let pub1 = Publisher::with_token("");
        // Empty string is still Some, so it's "authenticated"
        assert!(pub1.is_authenticated());
    }

    #[test]
    fn test_model_card_generation() {
        let card = generate_model_card("whisper-apr-tiny", "tiny");
        assert!(card.contains("whisper-apr-tiny"));
        assert!(card.contains("tiny"));
        assert!(card.contains("license: mit"));
        assert!(card.contains("whisper-apr"));
    }

    #[test]
    fn test_model_card_base_model() {
        let card = generate_model_card("whisper-apr-base", "base");
        assert!(card.contains("whisper-apr-base"));
        assert!(card.contains("base"));
        assert!(card.contains("WASM-First")); // Capital F in citation
    }

    #[test]
    fn test_model_card_contains_usage() {
        let card = generate_model_card("test-model", "small");
        assert!(card.contains("Usage"));
        assert!(card.contains("WhisperApr"));
        assert!(card.contains("transcribe"));
    }

    #[test]
    fn test_publish_format_variants() {
        assert_eq!(PublishFormat::default(), PublishFormat::Both);
        assert_ne!(PublishFormat::Apr, PublishFormat::SafeTensors);
        assert_ne!(PublishFormat::Apr, PublishFormat::Both);
        assert_ne!(PublishFormat::SafeTensors, PublishFormat::Both);
    }

    #[test]
    fn test_publish_format_clone_copy() {
        let format = PublishFormat::Apr;
        let cloned = format.clone();
        let copied = format;
        assert_eq!(format, cloned);
        assert_eq!(format, copied);
    }

    #[test]
    fn test_publish_result_fields() {
        let result = PublishResult {
            repo_url: "https://huggingface.co/paiml/test".to_string(),
            commit_sha: "abc123".to_string(),
            files_uploaded: vec!["model.safetensors".to_string()],
            total_bytes: 1024,
        };
        assert_eq!(result.repo_url, "https://huggingface.co/paiml/test");
        assert_eq!(result.commit_sha, "abc123");
        assert_eq!(result.files_uploaded.len(), 1);
        assert_eq!(result.total_bytes, 1024);
    }

    #[test]
    fn test_publish_result_clone() {
        let result = PublishResult {
            repo_url: "https://example.com".to_string(),
            commit_sha: "def456".to_string(),
            files_uploaded: vec!["a.bin".to_string(), "b.bin".to_string()],
            total_bytes: 2048,
        };
        let cloned = result.clone();
        assert_eq!(result.repo_url, cloned.repo_url);
        assert_eq!(result.files_uploaded, cloned.files_uploaded);
    }

    #[test]
    fn test_publish_config_builder_chain() {
        let config = PublishConfig::new("org/model")
            .with_message("v1.0")
            .private(false)
            .with_model_card("# Card")
            .with_file("f1")
            .with_file("f2");

        assert_eq!(config.repo_id, "org/model");
        assert_eq!(config.commit_message, "v1.0");
        assert!(!config.private);
        assert!(config.model_card.is_some());
        assert_eq!(config.extra_files.len(), 2);
    }
}
