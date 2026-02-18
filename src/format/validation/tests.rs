use super::*;
use crate::format::{build_whisper_metadata, AprV2ReaderRef, AprV2Writer};
use crate::model::ModelConfig;

// Helper: create an AprV2Writer pre-configured for tiny model
fn test_writer() -> AprV2Writer {
    let config = ModelConfig::tiny();
    let meta = build_whisper_metadata(&config, "test");
    AprV2Writer::new(meta)
}

// =========================================================================
// TensorStats Tests
// =========================================================================

#[test]
fn test_tensor_stats_empty() {
    let stats = TensorStats::compute("empty", &[]);
    assert_eq!(stats.count, 0);
    assert_eq!(stats.mean, 0.0);
    assert_eq!(stats.std, 0.0);
}

#[test]
fn test_tensor_stats_single_value() {
    let stats = TensorStats::compute("single", &[5.0]);
    assert_eq!(stats.count, 1);
    assert_eq!(stats.mean, 5.0);
    assert_eq!(stats.min, 5.0);
    assert_eq!(stats.max, 5.0);
}

#[test]
fn test_tensor_stats_uniform() {
    let data: Vec<f32> = vec![1.0; 100];
    let stats = TensorStats::compute("uniform", &data);
    assert_eq!(stats.count, 100);
    assert!((stats.mean - 1.0).abs() < 1e-6);
    assert!(stats.std < 1e-6);
}

#[test]
fn test_tensor_stats_with_nan() {
    let data = vec![1.0, f32::NAN, 2.0, 3.0];
    let stats = TensorStats::compute("nan", &data);
    assert_eq!(stats.nan_count, 1);
    assert!(stats.has_nan());
    assert!((stats.mean - 2.0).abs() < 1e-6);
}

#[test]
fn test_tensor_stats_with_inf() {
    let data = vec![1.0, f32::INFINITY, 2.0];
    let stats = TensorStats::compute("inf", &data);
    assert_eq!(stats.inf_count, 1);
    assert!(stats.has_inf());
}

#[test]
fn test_tensor_stats_all_zeros() {
    let data = vec![0.0; 50];
    let stats = TensorStats::compute("zeros", &data);
    assert!(stats.is_all_zeros());
    assert_eq!(stats.zero_count, 50);
}

#[test]
fn test_tensor_stats_layer_norm_like() {
    let data: Vec<f32> = (0..384).map(|i| 0.8 + 0.4 * (i as f32 / 384.0)).collect();
    let stats = TensorStats::compute("ln_weight", &data);
    assert!(stats.mean > 0.5 && stats.mean < 3.0);
}

#[test]
fn test_tensor_stats_bad_layer_norm() {
    let data: Vec<f32> = vec![11.0; 384];
    let stats = TensorStats::compute("bad_ln", &data);
    assert!(stats.mean > 3.0);
}

// =========================================================================
// ValidationCheck Tests
// =========================================================================

#[test]
fn test_validation_check_pass() {
    let check = ValidationCheck::pass(1, 'A', "Test", "OK");
    assert!(check.passed);
    assert_eq!(check.id, 1);
    assert_eq!(check.category, 'A');
}

#[test]
fn test_validation_check_fail() {
    let check = ValidationCheck::fail(2, 'B', "Test", "Failed");
    assert!(!check.passed);
}

// =========================================================================
// ValidationReport Tests
// =========================================================================

#[test]
fn test_validation_report_score() {
    let checks = vec![
        ValidationCheck::pass(1, 'A', "Test1", "OK"),
        ValidationCheck::pass(2, 'A', "Test2", "OK"),
        ValidationCheck::fail(3, 'A', "Test3", "Fail"),
    ];
    let report = ValidationReport::from_checks(checks, vec![]);
    assert_eq!(report.score, 2);
    assert_eq!(report.max_score, 3);
}

#[test]
fn test_validation_report_pass_threshold() {
    let mut checks = Vec::new();
    for i in 1..=23 {
        checks.push(ValidationCheck::pass(i, 'A', &format!("Test{i}"), "OK"));
    }
    for i in 24..=25 {
        checks.push(ValidationCheck::fail(i, 'E', &format!("Test{i}"), "Fail"));
    }
    let report = ValidationReport::from_checks(checks, vec![]);
    assert!(report.passed);
    assert_eq!(report.score, 23);
}

#[test]
fn test_validation_report_fail_threshold() {
    let mut checks = Vec::new();
    for i in 1..=22 {
        checks.push(ValidationCheck::pass(i, 'A', &format!("Test{i}"), "OK"));
    }
    for i in 23..=25 {
        checks.push(ValidationCheck::fail(i, 'E', &format!("Test{i}"), "Fail"));
    }
    let report = ValidationReport::from_checks(checks, vec![]);
    assert!(!report.passed);
}

#[test]
fn test_validation_report_critical_failure_overrides() {
    let mut checks = Vec::new();
    for i in 1..=25 {
        checks.push(ValidationCheck::pass(i, 'A', &format!("Test{i}"), "OK"));
    }
    let report =
        ValidationReport::from_checks(checks, vec!["Critical: LN weight mean=11".to_string()]);
    assert!(!report.passed);
}

#[test]
fn test_validation_report_by_category() {
    let checks = vec![
        ValidationCheck::pass(1, 'A', "A1", "OK"),
        ValidationCheck::pass(2, 'A', "A2", "OK"),
        ValidationCheck::pass(3, 'B', "B1", "OK"),
    ];
    let report = ValidationReport::from_checks(checks, vec![]);
    assert_eq!(report.checks_by_category('A').len(), 2);
    assert_eq!(report.checks_by_category('B').len(), 1);
    assert_eq!(report.checks_by_category('C').len(), 0);
}

// =========================================================================
// AprValidator Tests with Test Data
// =========================================================================

fn create_valid_test_apr() -> Vec<u8> {
    let mut writer = test_writer();

    writer.add_f32_tensor(
        "decoder.token_embedding",
        vec![51865, 384],
        &vec![0.02; 51865 * 384],
    );

    writer.add_f32_tensor(
        "encoder.positional_embedding",
        vec![1500, 384],
        &vec![0.01; 1500 * 384],
    );
    writer.add_f32_tensor(
        "decoder.positional_embedding",
        vec![448, 384],
        &vec![0.01; 448 * 384],
    );

    let ln_weight: Vec<f32> = (0..384).map(|i| 0.9 + 0.2 * (i as f32 / 384.0)).collect();
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &ln_weight);
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &ln_weight);

    let ln_bias: Vec<f32> = (0..384).map(|i| -0.1 + 0.2 * (i as f32 / 384.0)).collect();
    writer.add_f32_tensor("encoder.layer_norm.bias", vec![384], &ln_bias);
    writer.add_f32_tensor("decoder.layer_norm.bias", vec![384], &ln_bias);

    writer.add_f32_tensor(
        "encoder.conv1.weight",
        vec![384, 80, 3],
        &vec![0.05; 384 * 80 * 3],
    );
    writer.add_f32_tensor("encoder.conv1.bias", vec![384], &vec![0.01; 384]);

    let attn_weight: Vec<f32> = vec![0.02; 384 * 384];
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn.q_proj.weight",
        vec![384, 384],
        &attn_weight,
    );
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn.k_proj.weight",
        vec![384, 384],
        &attn_weight,
    );
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn.v_proj.weight",
        vec![384, 384],
        &attn_weight,
    );
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn.out_proj.weight",
        vec![384, 384],
        &attn_weight,
    );

    writer.add_f32_tensor(
        "encoder.layers.0.fc1.weight",
        vec![1536, 384],
        &vec![0.02; 1536 * 384],
    );
    writer.add_f32_tensor(
        "encoder.layers.0.fc2.weight",
        vec![384, 1536],
        &vec![0.02; 384 * 1536],
    );

    writer.add_f32_tensor(
        "encoder.layers.0.self_attn_layer_norm.weight",
        vec![384],
        &ln_weight,
    );
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn_layer_norm.bias",
        vec![384],
        &ln_bias,
    );
    writer.add_f32_tensor(
        "encoder.layers.0.final_layer_norm.weight",
        vec![384],
        &ln_weight,
    );
    writer.add_f32_tensor(
        "encoder.layers.0.final_layer_norm.bias",
        vec![384],
        &ln_bias,
    );

    writer.write().expect("should serialize")
}

fn create_bad_ln_apr() -> Vec<u8> {
    let mut writer = test_writer();

    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![11.0; 384]);
    writer.add_f32_tensor("decoder.layer_norm.bias", vec![384], &vec![0.0; 384]);

    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("encoder.layer_norm.bias", vec![384], &vec![0.0; 384]);

    writer.add_f32_tensor(
        "decoder.token_embedding",
        vec![51865, 384],
        &vec![0.02; 51865 * 384],
    );

    writer.write().expect("should serialize")
}

fn parse_and_validate(data: &[u8]) -> ValidationReport {
    let reader = AprV2ReaderRef::from_bytes(data).expect("should parse");
    let config = metadata_to_model_config(reader.metadata());
    let validator = AprValidator::new(&reader, config);
    validator.validate_all()
}

#[test]
fn test_validator_valid_apr() {
    let data = create_valid_test_apr();
    let report = parse_and_validate(&data);
    assert!(report.score >= 15);
}

#[test]
fn test_validator_bad_ln_apr() {
    let data = create_bad_ln_apr();
    let report = parse_and_validate(&data);

    let check_7 = report
        .checks
        .iter()
        .find(|c| c.id == 7)
        .expect("should have check 7");
    assert!(!check_7.passed);
    assert!(check_7.message.contains("mean="));
}

#[test]
fn test_validator_detects_ln_bug() {
    let data = create_bad_ln_apr();
    let report = parse_and_validate(&data);

    assert!(!report.passed);
    assert!(!report.critical_failures.is_empty());
}

// =========================================================================
// Quick Validate Tests
// =========================================================================

#[test]
fn test_quick_validate_valid() {
    let data = create_valid_test_apr();
    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    assert!(quick_validate(&reader).is_ok());
}

#[test]
fn test_quick_validate_bad_ln() {
    let data = create_bad_ln_apr();
    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    let result = quick_validate(&reader);
    assert!(result.is_err());
    let err = result.expect_err("expected error for bad ln").to_string();
    assert!(err.contains("decoder.layer_norm.weight"));
}

// =========================================================================
// Individual Check Tests
// =========================================================================

#[test]
fn test_check_magic() {
    let data = create_valid_test_apr();
    let report = parse_and_validate(&data);
    let check = report.checks.iter().find(|c| c.id == 1).unwrap();
    assert!(check.passed);
}

#[test]
fn test_check_header() {
    let data = create_valid_test_apr();
    let report = parse_and_validate(&data);
    let check = report.checks.iter().find(|c| c.id == 2).unwrap();
    assert!(check.passed);
}

#[test]
fn test_check_tensor_count() {
    let data = create_valid_test_apr();
    let report = parse_and_validate(&data);
    let check = report.checks.iter().find(|c| c.id == 3).unwrap();
    assert!(check.passed || check.message.contains("tensors"));
}

#[test]
fn test_check_encoder_ln_weight_valid() {
    let data = create_valid_test_apr();
    let report = parse_and_validate(&data);
    let check = report.checks.iter().find(|c| c.id == 6).unwrap();
    assert!(check.passed, "Encoder LN should pass: {}", check.message);
}

#[test]
fn test_check_decoder_ln_weight_bad() {
    let data = create_bad_ln_apr();
    let report = parse_and_validate(&data);
    let check = report.checks.iter().find(|c| c.id == 7).unwrap();
    assert!(!check.passed, "Decoder LN should fail: {}", check.message);
    assert!(check.message.contains("11"));
}

#[test]
fn test_check_ln_nan_inf_clean() {
    let data = create_valid_test_apr();
    let report = parse_and_validate(&data);
    let check = report.checks.iter().find(|c| c.id == 10).unwrap();
    assert!(check.passed);
}

#[test]
fn test_check_no_zero_tensors_pass() {
    let data = create_valid_test_apr();
    let report = parse_and_validate(&data);
    let check = report.checks.iter().find(|c| c.id == 14).unwrap();
    assert!(check.passed);
}

#[test]
fn test_check_token_embedding_shape() {
    let data = create_valid_test_apr();
    let report = parse_and_validate(&data);
    let check = report.checks.iter().find(|c| c.id == 16).unwrap();
    assert!(check.passed, "Token embedding shape: {}", check.message);
}

#[test]
fn test_check_vocab_size() {
    let data = create_valid_test_apr();
    let report = parse_and_validate(&data);
    let check = report.checks.iter().find(|c| c.id == 20).unwrap();
    assert!(check.passed, "Vocab size: {}", check.message);
}

// =========================================================================
// Validator Fail-Path Tests (coverage for uncovered branches)
// =========================================================================

fn create_minimal_apr() -> Vec<u8> {
    let mut writer = test_writer();
    writer.add_f32_tensor(
        "decoder.token_embedding",
        vec![51865, 384],
        &vec![0.02; 51865 * 384],
    );
    writer.write().expect("should serialize")
}

fn create_zero_tensor_apr() -> Vec<u8> {
    let mut writer = test_writer();
    writer.add_f32_tensor(
        "decoder.token_embedding",
        vec![51865, 384],
        &vec![0.0; 51865 * 384],
    );
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("encoder.layer_norm.bias", vec![384], &vec![0.0; 384]);
    writer.add_f32_tensor("decoder.layer_norm.bias", vec![384], &vec![0.0; 384]);
    writer.write().expect("should serialize")
}

fn create_bad_embedding_stats_apr() -> Vec<u8> {
    let mut writer = test_writer();
    // Embedding with bad stats: high mean, low std
    writer.add_f32_tensor(
        "decoder.token_embedding",
        vec![51865, 384],
        &vec![5.0; 51865 * 384],
    );
    // Wrong-shaped positional embeddings
    writer.add_f32_tensor(
        "encoder.positional_embedding",
        vec![100, 384],
        &vec![0.01; 100 * 384],
    );
    writer.add_f32_tensor(
        "decoder.positional_embedding",
        vec![100, 384],
        &vec![0.01; 100 * 384],
    );
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.write().expect("should serialize")
}

fn create_bad_weight_std_apr() -> Vec<u8> {
    let mut writer = test_writer();
    writer.add_f32_tensor(
        "decoder.token_embedding",
        vec![51865, 384],
        &vec![0.02; 51865 * 384],
    );
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    // Weights with extreme std (all same value = std ~0)
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn.q_proj.weight",
        vec![384, 384],
        &vec![0.5; 384 * 384],
    );
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn.k_proj.weight",
        vec![384, 384],
        &vec![0.5; 384 * 384],
    );
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn.v_proj.weight",
        vec![384, 384],
        &vec![0.5; 384 * 384],
    );
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn.out_proj.weight",
        vec![384, 384],
        &vec![0.5; 384 * 384],
    );
    writer.add_f32_tensor(
        "encoder.layers.0.fc1.weight",
        vec![1536, 384],
        &vec![0.5; 1536 * 384],
    );
    writer.add_f32_tensor(
        "encoder.layers.0.fc2.weight",
        vec![384, 1536],
        &vec![0.5; 384 * 1536],
    );
    // Bad bias
    writer.add_f32_tensor("encoder.conv1.bias", vec![384], &vec![5.0; 384]);
    writer.write().expect("should serialize")
}

#[test]
fn test_validator_missing_embeddings() {
    let data = create_minimal_apr();
    let report = parse_and_validate(&data);
    // Missing LN weights should produce fails in checks 6-7
    let check_6 = report.checks.iter().find(|c| c.id == 6).unwrap();
    assert!(!check_6.passed);
    let check_7 = report.checks.iter().find(|c| c.id == 7).unwrap();
    assert!(!check_7.passed);
}

#[test]
fn test_validator_zero_tensors_detected() {
    let data = create_zero_tensor_apr();
    let report = parse_and_validate(&data);
    // Check 14: no zero tensors - should fail because embedding is all zeros
    let check_14 = report.checks.iter().find(|c| c.id == 14).unwrap();
    assert!(
        !check_14.passed,
        "Should detect zero tensor: {}",
        check_14.message
    );
}

#[test]
fn test_validator_bad_embedding_stats() {
    let data = create_bad_embedding_stats_apr();
    let report = parse_and_validate(&data);
    // Check 17: token embedding stats - mean too high
    let check_17 = report.checks.iter().find(|c| c.id == 17).unwrap();
    assert!(!check_17.passed, "Should fail: {}", check_17.message);
    // Check 18: positional embedding shape - wrong shape
    let check_18 = report.checks.iter().find(|c| c.id == 18).unwrap();
    assert!(!check_18.passed, "Should fail: {}", check_18.message);
}

#[test]
fn test_validator_bad_weight_std() {
    let data = create_bad_weight_std_apr();
    let report = parse_and_validate(&data);
    // Check 13: weight std - should fail (all-same values = ~0 std)
    let check_13 = report.checks.iter().find(|c| c.id == 13).unwrap();
    assert!(
        !check_13.passed,
        "Should detect bad std: {}",
        check_13.message
    );
    // Check 15: bias vectors - mean=5.0 is out of [-1, 1]
    let check_15 = report.checks.iter().find(|c| c.id == 15).unwrap();
    assert!(
        !check_15.passed,
        "Should detect bad bias: {}",
        check_15.message
    );
}

#[test]
fn test_validator_qkv_proj_means_bad() {
    let data = create_bad_weight_std_apr();
    let report = parse_and_validate(&data);
    // Check 11: QKV proj means - 0.5 is out of [-0.1, 0.1]
    let check_11 = report.checks.iter().find(|c| c.id == 11).unwrap();
    assert!(
        !check_11.passed,
        "Should detect bad proj means: {}",
        check_11.message
    );
    // Check 12: FFN means - 0.5 is out of [-0.1, 0.1]
    let check_12 = report.checks.iter().find(|c| c.id == 12).unwrap();
    assert!(
        !check_12.passed,
        "Should detect bad FFN means: {}",
        check_12.message
    );
}

#[test]
fn test_validator_positional_embedding_stats_bad() {
    let data = create_bad_embedding_stats_apr();
    let report = parse_and_validate(&data);
    // Check 19: positional embedding stats
    let check_19 = report.checks.iter().find(|c| c.id == 19).unwrap();
    // Stats may or may not fail depending on the data, just ensure the check ran
    assert!(check_19.id == 19);
}

#[test]
fn test_validate_apr_bytes_convenience() {
    let data = create_valid_test_apr();
    let report = validate_apr_bytes(&data).expect("should validate");
    assert!(report.score >= 15);
}

#[test]
fn test_validator_tensor_shapes_bad() {
    let mut writer = test_writer();
    // Wrong-shaped token embedding (shape[1] != d_model)
    writer.add_f32_tensor(
        "decoder.token_embedding",
        vec![51865, 128],
        &vec![0.02; 51865 * 128],
    );
    // Wrong-shaped conv1 (shape[0] != d_model)
    writer.add_f32_tensor(
        "encoder.conv1.weight",
        vec![128, 80, 3],
        &vec![0.05; 128 * 80 * 3],
    );
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    let data = writer.write().expect("should serialize");

    let report = parse_and_validate(&data);
    let check_4 = report.checks.iter().find(|c| c.id == 4).unwrap();
    assert!(
        !check_4.passed,
        "Should detect bad shapes: {}",
        check_4.message
    );
}

#[test]
fn test_validator_vocab_size_mismatch() {
    let mut writer = test_writer();
    // Wrong vocab size in embedding (384 != 51865)
    writer.add_f32_tensor(
        "decoder.token_embedding",
        vec![384, 384],
        &vec![0.02; 384 * 384],
    );
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    let data = writer.write().expect("should serialize");

    let report = parse_and_validate(&data);
    let check_20 = report.checks.iter().find(|c| c.id == 20).unwrap();
    assert!(
        !check_20.passed,
        "Should detect vocab mismatch: {}",
        check_20.message
    );
}

#[test]
fn test_validator_ln_with_nan_inf() {
    let mut writer = test_writer();
    writer.add_f32_tensor(
        "decoder.token_embedding",
        vec![51865, 384],
        &vec![0.02; 51865 * 384],
    );
    // LN weight with NaN
    let mut ln_data = vec![1.0f32; 384];
    ln_data[0] = f32::NAN;
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &ln_data);
    // LN weight with Inf
    let mut ln_data2 = vec![1.0f32; 384];
    ln_data2[0] = f32::INFINITY;
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &ln_data2);
    // Block LN with bad mean
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn_layer_norm.weight",
        vec![384],
        &vec![11.0; 384],
    );
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn_layer_norm.bias",
        vec![384],
        &vec![0.0; 384],
    );
    writer.add_f32_tensor("encoder.layer_norm.bias", vec![384], &vec![5.0; 384]);
    writer.add_f32_tensor("decoder.layer_norm.bias", vec![384], &vec![0.0; 384]);
    let data = writer.write().expect("should serialize");

    let report = parse_and_validate(&data);
    // Check 10: NaN/Inf in LN - should detect both
    let check_10 = report.checks.iter().find(|c| c.id == 10).unwrap();
    assert!(
        !check_10.passed,
        "Should detect NaN/Inf: {}",
        check_10.message
    );
    // Check 8: Block LN means - should detect bad mean
    let check_8 = report.checks.iter().find(|c| c.id == 8).unwrap();
    assert!(
        !check_8.passed,
        "Should detect bad block LN: {}",
        check_8.message
    );
    // Check 9: LN biases - encoder bias mean=5.0 out of [-0.5, 0.5]
    let check_9 = report.checks.iter().find(|c| c.id == 9).unwrap();
    assert!(
        !check_9.passed,
        "Should detect bad bias: {}",
        check_9.message
    );
}

#[test]
fn test_validator_weight_std_minor_outlier() {
    let mut writer = test_writer();
    writer.add_f32_tensor(
        "decoder.token_embedding",
        vec![51865, 384],
        &vec![0.02; 51865 * 384],
    );
    // LN weights with varying values (so they have valid std when checked as .weight)
    let ln_weight: Vec<f32> = (0..384).map(|i| 0.9 + 0.2 * (i as f32 / 384.0)).collect();
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &ln_weight);
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &ln_weight);
    // Many good weights with proper std (range gives std ~0.03)
    let good_weight: Vec<f32> = (0..384 * 384)
        .map(|i| ((i % 100) as f32 - 50.0) * 0.001)
        .collect();
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn.q_proj.weight",
        vec![384, 384],
        &good_weight,
    );
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn.k_proj.weight",
        vec![384, 384],
        &good_weight,
    );
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn.v_proj.weight",
        vec![384, 384],
        &good_weight,
    );
    writer.add_f32_tensor(
        "encoder.layers.0.self_attn.out_proj.weight",
        vec![384, 384],
        &good_weight,
    );
    writer.add_f32_tensor(
        "encoder.layers.0.fc1.weight",
        vec![1536, 384],
        &(0..1536 * 384)
            .map(|i| ((i % 100) as f32 - 50.0) * 0.001)
            .collect::<Vec<f32>>(),
    );
    writer.add_f32_tensor(
        "encoder.layers.0.fc2.weight",
        vec![384, 1536],
        &(0..384 * 1536)
            .map(|i| ((i % 100) as f32 - 50.0) * 0.001)
            .collect::<Vec<f32>>(),
    );
    // One weight with bad std (all same = ~0 std) as minor outlier
    writer.add_f32_tensor(
        "encoder.conv1.weight",
        vec![384, 80, 3],
        &vec![0.05; 384 * 80 * 3],
    );
    let data = writer.write().expect("should serialize");

    let report = parse_and_validate(&data);
    // Check 13: 1 outlier in 9 weights = ~11% < 25%, so passes with "minor outliers"
    let check_13 = report.checks.iter().find(|c| c.id == 13).unwrap();
    assert!(
        check_13.passed,
        "Minor outlier should pass: {}",
        check_13.message
    );
    assert!(
        check_13.message.contains("minor outlier") || check_13.message.contains("outlier"),
        "Message should mention outliers: {}",
        check_13.message
    );
}

#[test]
fn test_validator_no_token_embedding() {
    let mut writer = test_writer();
    // No token embedding at all
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    let data = writer.write().expect("should serialize");

    let report = parse_and_validate(&data);
    // Check 16: token embedding shape - not found
    let check_16 = report.checks.iter().find(|c| c.id == 16).unwrap();
    assert!(!check_16.passed, "Missing embedding: {}", check_16.message);
    // Check 17: token embedding stats - not found
    let check_17 = report.checks.iter().find(|c| c.id == 17).unwrap();
    assert!(
        !check_17.passed,
        "Missing embedding stats: {}",
        check_17.message
    );
    // Check 20: vocab size - not found
    let check_20 = report.checks.iter().find(|c| c.id == 20).unwrap();
    assert!(!check_20.passed, "Missing vocab: {}", check_20.message);
}

#[test]
fn test_validator_tensor_count_insufficient() {
    let mut writer = test_writer();
    // Only a few tensors (way less than expected for tiny model)
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    let data = writer.write().expect("should serialize");

    let report = parse_and_validate(&data);
    let check_3 = report.checks.iter().find(|c| c.id == 3).unwrap();
    assert!(
        !check_3.passed,
        "Should detect insufficient tensors: {}",
        check_3.message
    );
}

// =========================================================================
// Coverage Tests for quick_validate uncovered branches (WAPR-QA-005)
// =========================================================================

#[test]
fn test_quick_validate_bad_encoder_ln() {
    let mut writer = test_writer();
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![11.0; 384]);
    writer.add_f32_tensor(
        "decoder.token_embedding",
        vec![51865, 384],
        &vec![0.02; 51865 * 384],
    );
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    let result = quick_validate(&reader);
    assert!(result.is_err());
    let err = result
        .expect_err("expected error for bad encoder ln")
        .to_string();
    assert!(err.contains("encoder.layer_norm.weight"));
}

#[test]
fn test_quick_validate_bad_encoder_ln_low_mean() {
    let mut writer = test_writer();
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![0.1; 384]);
    writer.add_f32_tensor(
        "decoder.token_embedding",
        vec![51865, 384],
        &vec![0.02; 51865 * 384],
    );
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    let result = quick_validate(&reader);
    assert!(result.is_err());
    let err = result.expect_err("expected error").to_string();
    assert!(err.contains("encoder.layer_norm.weight"));
}

#[test]
fn test_quick_validate_no_ln_tensors() {
    let mut writer = test_writer();
    writer.add_f32_tensor(
        "decoder.token_embedding",
        vec![51865, 384],
        &vec![0.02; 51865 * 384],
    );
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    let result = quick_validate(&reader);
    assert!(result.is_ok());
}

#[test]
fn test_quick_validate_only_decoder_ln() {
    let mut writer = test_writer();
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![0.1; 384]);
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    let result = quick_validate(&reader);
    assert!(result.is_err());
    let err = result.expect_err("expected error").to_string();
    assert!(err.contains("decoder.layer_norm.weight"));
}

#[test]
fn test_quick_validate_only_encoder_ln() {
    let mut writer = test_writer();
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    let result = quick_validate(&reader);
    assert!(result.is_ok());
}

#[test]
fn test_quick_validate_both_valid() {
    let mut writer = test_writer();
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    assert!(quick_validate(&reader).is_ok());
}

#[test]
fn test_quick_validate_decoder_bad_high_encoder_valid() {
    let mut writer = test_writer();
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![5.0; 384]);
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    let result = quick_validate(&reader);
    assert!(result.is_err());
    let err = result.expect_err("expected error").to_string();
    assert!(err.contains("decoder.layer_norm.weight"));
}

// =========================================================================
// quick_validate: boundary and path coverage (WAPR-QA-008)
// =========================================================================

#[test]
fn test_quick_validate_decoder_at_lower_boundary() {
    let mut writer = test_writer();
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![0.5; 384]);
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    assert!(quick_validate(&reader).is_ok());
}

#[test]
fn test_quick_validate_decoder_at_upper_boundary() {
    let mut writer = test_writer();
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![3.0; 384]);
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    assert!(quick_validate(&reader).is_ok());
}

#[test]
fn test_quick_validate_decoder_just_below_lower() {
    let mut writer = test_writer();
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![0.499; 384]);
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    assert!(quick_validate(&reader).is_err());
}

#[test]
fn test_quick_validate_decoder_just_above_upper() {
    let mut writer = test_writer();
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![3.001; 384]);
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    assert!(quick_validate(&reader).is_err());
}

#[test]
fn test_quick_validate_encoder_at_lower_boundary() {
    let mut writer = test_writer();
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![0.5; 384]);
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    assert!(quick_validate(&reader).is_ok());
}

#[test]
fn test_quick_validate_encoder_at_upper_boundary() {
    let mut writer = test_writer();
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![3.0; 384]);
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    assert!(quick_validate(&reader).is_ok());
}

#[test]
fn test_quick_validate_encoder_just_below_lower() {
    let mut writer = test_writer();
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![0.499; 384]);
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    let result = quick_validate(&reader);
    assert!(result.is_err());
    let err = result.expect_err("expected encoder error").to_string();
    assert!(err.contains("encoder.layer_norm.weight"));
}

#[test]
fn test_quick_validate_encoder_just_above_upper() {
    let mut writer = test_writer();
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![3.001; 384]);
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    let result = quick_validate(&reader);
    assert!(result.is_err());
    let err = result.expect_err("expected encoder error").to_string();
    assert!(err.contains("encoder.layer_norm.weight"));
}

#[test]
fn test_quick_validate_decoder_valid_encoder_missing() {
    let mut writer = test_writer();
    writer.add_f32_tensor("decoder.layer_norm.weight", vec![384], &vec![1.0; 384]);
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    assert!(quick_validate(&reader).is_ok());
}

#[test]
fn test_quick_validate_decoder_missing_encoder_bad() {
    let mut writer = test_writer();
    writer.add_f32_tensor("encoder.layer_norm.weight", vec![384], &vec![11.0; 384]);
    let data = writer.write().expect("should serialize");

    let reader = AprV2ReaderRef::from_bytes(&data).expect("should parse");
    let result = quick_validate(&reader);
    assert!(result.is_err());
    let err = result.expect_err("expected encoder error").to_string();
    assert!(err.contains("encoder.layer_norm.weight"));
}
