//! Tests for HuggingFace Hub publishing module

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
    let config = PublishConfig::new("test/repo").with_model_card("# My Model Card");
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

#[test]
fn test_prepare_apr_only() {
    let temp_dir = tempfile::tempdir().unwrap();
    let apr_path = temp_dir.path().join("test.apr");
    std::fs::write(&apr_path, b"APR1test_content").unwrap();

    let output_dir = temp_dir.path().join("output");
    let publisher = Publisher::with_token("test");

    let prepared = publisher
        .prepare(&apr_path, &output_dir, PublishFormat::Apr)
        .unwrap();

    assert_eq!(prepared.files.len(), 1);
    assert!(prepared.files[0].to_string_lossy().contains("test.apr"));
    assert!(prepared.total_bytes > 0);
    assert!(output_dir.join("test.apr").exists());
}

#[test]
fn test_prepare_safetensors_only() {
    let temp_dir = tempfile::tempdir().unwrap();
    let apr_path = temp_dir.path().join("model.apr");
    std::fs::write(&apr_path, b"APR1test").unwrap();

    let output_dir = temp_dir.path().join("output");
    let publisher = Publisher::new();

    let prepared = publisher
        .prepare(&apr_path, &output_dir, PublishFormat::SafeTensors)
        .unwrap();

    assert_eq!(prepared.files.len(), 1);
    assert!(prepared.files[0]
        .to_string_lossy()
        .contains("model.safetensors"));
    assert!(output_dir.join("model.safetensors").exists());
}

#[test]
fn test_prepare_both_formats() {
    let temp_dir = tempfile::tempdir().unwrap();
    let apr_path = temp_dir.path().join("whisper.apr");
    std::fs::write(&apr_path, b"APR1content").unwrap();

    let output_dir = temp_dir.path().join("out");
    let publisher = Publisher::with_token("tok");

    let prepared = publisher
        .prepare(&apr_path, &output_dir, PublishFormat::Both)
        .unwrap();

    assert_eq!(prepared.files.len(), 2);
    assert!(output_dir.join("whisper.apr").exists());
    assert!(output_dir.join("model.safetensors").exists());
}

#[test]
fn test_prepare_creates_output_dir() {
    let temp_dir = tempfile::tempdir().unwrap();
    let apr_path = temp_dir.path().join("model.apr");
    std::fs::write(&apr_path, b"APR1data").unwrap();

    let nested_output = temp_dir.path().join("a").join("b").join("c");
    let publisher = Publisher::new();

    let result = publisher.prepare(&apr_path, &nested_output, PublishFormat::Apr);
    assert!(result.is_ok());
    assert!(nested_output.exists());
}

#[test]
fn test_prepare_invalid_apr_path() {
    let temp_dir = tempfile::tempdir().unwrap();
    // Path without filename (just directory)
    let apr_path = temp_dir.path();
    let output_dir = temp_dir.path().join("out");

    let publisher = Publisher::new();
    let result = publisher.prepare(apr_path, &output_dir, PublishFormat::Apr);

    // Should fail because apr_path has no filename
    assert!(result.is_err());
}

#[test]
fn test_prepare_nonexistent_apr() {
    let temp_dir = tempfile::tempdir().unwrap();
    let apr_path = temp_dir.path().join("nonexistent.apr");
    let output_dir = temp_dir.path().join("out");

    let publisher = Publisher::new();
    let result = publisher.prepare(&apr_path, &output_dir, PublishFormat::Apr);

    assert!(result.is_err());
}

#[test]
fn test_prepared_publish_fields() {
    let prepared = PreparedPublish {
        files: vec![PathBuf::from("a.apr"), PathBuf::from("b.safetensors")],
        total_bytes: 4096,
        apr_path: PathBuf::from("/original/model.apr"),
    };
    assert_eq!(prepared.files.len(), 2);
    assert_eq!(prepared.total_bytes, 4096);
    assert_eq!(prepared.apr_path.to_string_lossy(), "/original/model.apr");
}

#[test]
fn test_prepared_publish_clone() {
    let prepared = PreparedPublish {
        files: vec![PathBuf::from("test.apr")],
        total_bytes: 100,
        apr_path: PathBuf::from("src.apr"),
    };
    let cloned = prepared.clone();
    assert_eq!(prepared.files, cloned.files);
    assert_eq!(prepared.total_bytes, cloned.total_bytes);
}

#[test]
fn test_publish_requires_auth() {
    // Create a publisher without token by clearing env
    let publisher = Publisher {
        token: None,
        _api_url: "https://example.com".to_string(),
    };

    let prepared = PreparedPublish {
        files: vec![],
        total_bytes: 0,
        apr_path: PathBuf::from("test.apr"),
    };
    let config = PublishConfig::new("paiml/test");

    let result = publisher.publish(&prepared, &config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("HF_TOKEN"));
}

#[test]
fn test_publish_requires_repo_id() {
    let publisher = Publisher::with_token("valid_token");
    let prepared = PreparedPublish {
        files: vec![],
        total_bytes: 0,
        apr_path: PathBuf::from("test.apr"),
    };
    let config = PublishConfig::default(); // Empty repo_id

    let result = publisher.publish(&prepared, &config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("repo_id"));
}

#[test]
fn test_publish_success() {
    let publisher = Publisher::with_token("test_token");
    let prepared = PreparedPublish {
        files: vec![
            PathBuf::from("model.apr"),
            PathBuf::from("model.safetensors"),
        ],
        total_bytes: 2048,
        apr_path: PathBuf::from("original.apr"),
    };
    let config = PublishConfig::new("paiml/whisper-test");

    let result = publisher.publish(&prepared, &config).unwrap();
    assert!(result.repo_url.contains("paiml/whisper-test"));
    assert_eq!(result.files_uploaded.len(), 2);
    assert_eq!(result.total_bytes, 2048);
}

#[test]
fn test_publish_result_debug() {
    let result = PublishResult {
        repo_url: "https://hf.co/test".to_string(),
        commit_sha: "abc".to_string(),
        files_uploaded: vec!["f1".to_string()],
        total_bytes: 100,
    };
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("PublishResult"));
    assert!(debug_str.contains("abc"));
}

#[test]
fn test_publish_config_debug() {
    let config = PublishConfig::new("test/repo");
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("PublishConfig"));
    assert!(debug_str.contains("test/repo"));
}

#[test]
fn test_prepared_publish_debug() {
    let prepared = PreparedPublish {
        files: vec![PathBuf::from("test.apr")],
        total_bytes: 50,
        apr_path: PathBuf::from("src.apr"),
    };
    let debug_str = format!("{:?}", prepared);
    assert!(debug_str.contains("PreparedPublish"));
}

#[test]
fn test_publish_format_debug() {
    let format = PublishFormat::Both;
    let debug_str = format!("{:?}", format);
    assert!(debug_str.contains("Both"));
}

#[test]
fn test_model_card_special_chars() {
    let card = generate_model_card("model-with-special", "tiny-en");
    assert!(card.contains("model-with-special"));
    assert!(card.contains("tiny-en"));
}
