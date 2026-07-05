//! Integration Tests for HuggingFace Publishing (WAPR-PUB-001)
#![cfg(feature = "integration-tests")]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::no_effect_underscore_binding
)]
//!
//! End-to-end tests for the publish workflow:
//! 1. APR → SafeTensors export
//! 2. Pre-publish verification
//! 3. Model card generation
//! 4. Publish orchestration

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use whisper_apr::format::export::{SafeTensorsExporter, TensorData};
use whisper_apr::publish::{generate_model_card, PublishConfig, PublishFormat, Publisher};
use whisper_apr::verify::{verify_safetensors, CheckResult, VerificationReport, Verifier};

/// Test fixture: create a unique temporary directory per call
fn temp_dir() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "whisper_apr_publish_test_{}_{}",
        std::process::id(),
        id
    ));
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Cleanup test directory
fn cleanup(dir: &PathBuf) {
    let _ = fs::remove_dir_all(dir);
}

/// Test the complete export → verify → publish workflow
#[test]
fn test_export_verify_workflow() {
    let dir = temp_dir();
    let st_path = dir.join("test_model.safetensors");

    // Step 1: Create test tensors
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "encoder.conv1.weight".to_string(),
        TensorData::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
    );
    tensors.insert(
        "encoder.conv2.weight".to_string(),
        TensorData::new(vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0], vec![2, 3]),
    );
    tensors.insert(
        "decoder.token_embedding".to_string(),
        TensorData::new(vec![1.0; 100], vec![10, 10]),
    );

    // Step 2: Export to SafeTensors
    let mut metadata = BTreeMap::new();
    metadata.insert("format".to_string(), "whisper.apr".to_string());
    metadata.insert("version".to_string(), "0.2.0".to_string());

    SafeTensorsExporter::save_with_metadata(&st_path, &tensors, Some(metadata))
        .expect("Export should succeed");

    assert!(st_path.exists(), "SafeTensors file should exist");

    // Step 3: Verify the exported file
    let report = verify_safetensors(&st_path).expect("Verification should complete");

    assert!(report.passed, "Verification should pass");
    assert!(report.pass_rate() >= 80.0, "Pass rate should be >= 80%");

    // Step 4: Verify tensors don't have NaN/Inf
    let verifier = Verifier::new();
    let tensor_report = verifier
        .verify_tensors(&tensors)
        .expect("Tensor verification should complete");

    assert!(tensor_report.passed, "Tensor verification should pass");

    cleanup(&dir);
}

/// Test SafeTensors format compliance
#[test]
fn test_safetensors_format_compliance() {
    let dir = temp_dir();
    let path = dir.join("format_test.safetensors");

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "weight".to_string(),
        TensorData::new(vec![1.0, 2.0], vec![2]),
    );

    SafeTensorsExporter::save(&path, &tensors).expect("Export should succeed");

    // Read raw bytes and verify format
    let data = fs::read(&path).expect("Should read file");

    // Check header length (first 8 bytes, u64 LE)
    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    assert!(header_len > 0, "Header length should be > 0");
    assert_eq!(header_len % 8, 0, "Header should be 8-byte aligned");

    // Check header is valid JSON
    let header_str = std::str::from_utf8(&data[8..8 + header_len])
        .expect("Header should be UTF-8")
        .trim();
    assert!(
        header_str.starts_with('{'),
        "Header should start with open brace"
    );
    assert!(
        header_str.ends_with('}'),
        "Header should end with close brace"
    );

    // Check required fields
    assert!(
        header_str.contains(r#""dtype":"F32""#),
        "Should have F32 dtype"
    );
    assert!(
        header_str.contains(r#""shape":[2]"#),
        "Should have correct shape"
    );
    assert!(
        header_str.contains(r#""data_offsets""#),
        "Should have data_offsets"
    );

    // Check tensor data follows header
    let data_start = 8 + header_len;
    assert!(
        data.len() >= data_start + 8,
        "File should contain tensor data"
    );

    // Verify first tensor value (1.0f32 in LE)
    let first_value = f32::from_le_bytes(data[data_start..data_start + 4].try_into().unwrap());
    assert!(
        (first_value - 1.0).abs() < 0.0001,
        "First value should be 1.0"
    );

    cleanup(&dir);
}

/// Test deterministic output (same input = same output)
#[test]
fn test_deterministic_export() {
    let dir = temp_dir();
    let path1 = dir.join("det1.safetensors");
    let path2 = dir.join("det2.safetensors");

    // Create tensors with specific order
    let mut tensors = BTreeMap::new();
    tensors.insert("z_weight".to_string(), TensorData::new(vec![3.0], vec![1]));
    tensors.insert("a_weight".to_string(), TensorData::new(vec![1.0], vec![1]));
    tensors.insert("m_weight".to_string(), TensorData::new(vec![2.0], vec![1]));

    // Export twice
    SafeTensorsExporter::save(&path1, &tensors).expect("First export");
    SafeTensorsExporter::save(&path2, &tensors).expect("Second export");

    // Compare outputs
    let data1 = fs::read(&path1).expect("Read first");
    let data2 = fs::read(&path2).expect("Read second");

    assert_eq!(data1, data2, "Exports should be byte-identical");

    // Verify alphabetical ordering in header (BTreeMap guarantees this)
    let header_len = u64::from_le_bytes(data1[0..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&data1[8..8 + header_len]).unwrap();

    let a_pos = header.find("a_weight").unwrap();
    let m_pos = header.find("m_weight").unwrap();
    let z_pos = header.find("z_weight").unwrap();

    assert!(a_pos < m_pos, "a_weight should come before m_weight");
    assert!(m_pos < z_pos, "m_weight should come before z_weight");

    cleanup(&dir);
}

/// Test verification catches NaN values
#[test]
fn test_verify_catches_nan() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "has_nan".to_string(),
        TensorData::new(vec![1.0, f32::NAN, 3.0], vec![3]),
    );

    let verifier = Verifier::new();
    let report = verifier.verify_tensors(&tensors).expect("Should verify");

    assert!(!report.passed, "Should fail due to NaN");

    let nan_check = report
        .checks
        .iter()
        .find(|c| c.name.contains("no_nan"))
        .expect("Should have NaN check");
    assert!(!nan_check.passed, "NaN check should fail");
}

/// Test verification catches Inf values
#[test]
fn test_verify_catches_inf() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "has_inf".to_string(),
        TensorData::new(vec![1.0, f32::INFINITY, 3.0], vec![3]),
    );

    let verifier = Verifier::new();
    let report = verifier.verify_tensors(&tensors).expect("Should verify");

    assert!(!report.passed, "Should fail due to Inf");

    let inf_check = report
        .checks
        .iter()
        .find(|c| c.name.contains("no_inf"))
        .expect("Should have Inf check");
    assert!(!inf_check.passed, "Inf check should fail");
}

/// Test model card generation
#[test]
fn test_model_card_generation() {
    let card = generate_model_card("whisper-apr-tiny", "tiny");

    // Check required YAML frontmatter
    assert!(card.starts_with("---"), "Should have YAML frontmatter");
    assert!(card.contains("license: mit"), "Should have license");
    assert!(card.contains("pipeline_tag: automatic-speech-recognition"));
    assert!(card.contains("library_name: whisper-apr"));

    // Check content
    assert!(
        card.contains("whisper-apr-tiny"),
        "Should contain model name"
    );
    assert!(card.contains("tiny"), "Should contain model size");
    assert!(card.contains("Rust"), "Should mention Rust");
    assert!(card.contains("WASM") || card.contains("WebAssembly"));

    // Check usage example
    assert!(card.contains("```rust"), "Should have Rust code example");
    assert!(card.contains("WhisperApr"), "Should reference main type");
}

/// Test publisher configuration
#[test]
fn test_publisher_config() {
    let config = PublishConfig::new("paiml/whisper-apr-tiny")
        .with_message("Release v0.2.0")
        .with_model_card("# Test Model")
        .private(false)
        .with_file("tokenizer.json");

    assert_eq!(config.repo_id, "paiml/whisper-apr-tiny");
    assert_eq!(config.commit_message, "Release v0.2.0");
    assert_eq!(config.model_card, Some("# Test Model".to_string()));
    assert!(!config.private);
    assert_eq!(config.extra_files.len(), 1);
}

/// Test publish format options
#[test]
fn test_publish_formats() {
    assert_eq!(PublishFormat::default(), PublishFormat::Both);

    // Test all variants exist
    let _apr = PublishFormat::Apr;
    let _st = PublishFormat::SafeTensors;
    let _both = PublishFormat::Both;
}

/// Test verification report scoring
#[test]
fn test_verification_scoring() {
    let mut report = VerificationReport::new();

    // 88% pass rate (spec threshold)
    for i in 0..88 {
        report.add(CheckResult::pass(format!("check_{}", i), "ok"));
    }
    for i in 88..100 {
        report.add(CheckResult::fail(format!("check_{}", i), "fail"));
    }

    assert_eq!(report.total_checks, 100);
    assert_eq!(report.passed_checks, 88);
    assert!((report.pass_rate() - 88.0).abs() < 0.01);

    let verifier = Verifier::new(); // Default 88% threshold
    assert!(verifier.meets_threshold(&report));
}

/// Test metadata in SafeTensors export
#[test]
fn test_safetensors_metadata() {
    let dir = temp_dir();
    let path = dir.join("with_meta.safetensors");

    let mut tensors = BTreeMap::new();
    tensors.insert("w".to_string(), TensorData::new(vec![1.0], vec![1]));

    let mut meta = BTreeMap::new();
    meta.insert("format".to_string(), "whisper.apr".to_string());
    meta.insert("version".to_string(), "0.2.0".to_string());
    meta.insert("architecture".to_string(), "whisper".to_string());

    SafeTensorsExporter::save_with_metadata(&path, &tensors, Some(meta))
        .expect("Export with metadata");

    let data = fs::read(&path).expect("Read file");
    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&data[8..8 + header_len]).unwrap();

    assert!(
        header.contains("__metadata__"),
        "Should have metadata block"
    );
    assert!(header.contains("whisper.apr"), "Should have format");
    assert!(header.contains("0.2.0"), "Should have version");
    assert!(header.contains("whisper"), "Should have architecture");

    cleanup(&dir);
}

/// Test publisher prepare (without actual upload)
#[test]
fn test_publisher_prepare() {
    let dir = temp_dir();

    // Create a minimal APR-like file
    let apr_path = dir.join("test.apr");
    fs::write(&apr_path, b"APR\0\x01\x00\x00\x00").expect("Create test APR");

    let publisher = Publisher::new();
    let output_dir = dir.join("output");

    let prepared = publisher
        .prepare(&apr_path, &output_dir, PublishFormat::Apr)
        .expect("Prepare should succeed");

    assert!(!prepared.files.is_empty(), "Should have files to upload");
    assert!(prepared.total_bytes > 0, "Should have bytes");
    assert!(output_dir.exists(), "Output dir should be created");

    cleanup(&dir);
}
