//! Tests for pre-publish verification module
#![allow(clippy::unwrap_used, clippy::large_stack_arrays)]
use super::*;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_check_result() {
    let pass = CheckResult::pass("test", "ok");
    assert!(pass.passed);
    assert_eq!(pass.name, "test");
    assert_eq!(pass.message, "ok");

    let fail = CheckResult::fail("test", "error");
    assert!(!fail.passed);
    assert_eq!(fail.name, "test");
    assert_eq!(fail.message, "error");
}

#[test]
fn test_check_result_clone() {
    let result = CheckResult::pass("name", "msg");
    let cloned = result.clone();
    assert_eq!(result.name, cloned.name);
    assert_eq!(result.passed, cloned.passed);
}

#[test]
fn test_verification_report() {
    let mut report = VerificationReport::new();
    assert!(report.passed);

    report.add(CheckResult::pass("a", "ok"));
    report.add(CheckResult::pass("b", "ok"));
    report.add(CheckResult::fail("c", "error"));

    assert!(!report.passed);
    assert_eq!(report.total_checks, 3);
    assert_eq!(report.passed_checks, 2);
    assert!((report.pass_rate() - 66.67).abs() < 0.1);
}

#[test]
fn test_verification_report_empty() {
    let report = VerificationReport::new();
    assert!(report.passed);
    assert_eq!(report.total_checks, 0);
    assert_eq!(report.passed_checks, 0);
    assert!((report.pass_rate() - 100.0).abs() < 0.01);
}

#[test]
fn test_verification_report_all_pass() {
    let mut report = VerificationReport::new();
    report.add(CheckResult::pass("a", "ok"));
    report.add(CheckResult::pass("b", "ok"));
    report.add(CheckResult::pass("c", "ok"));

    assert!(report.passed);
    assert_eq!(report.total_checks, 3);
    assert_eq!(report.passed_checks, 3);
    assert!((report.pass_rate() - 100.0).abs() < 0.01);
}

#[test]
fn test_verification_report_all_fail() {
    let mut report = VerificationReport::new();
    report.add(CheckResult::fail("a", "err"));
    report.add(CheckResult::fail("b", "err"));

    assert!(!report.passed);
    assert_eq!(report.total_checks, 2);
    assert_eq!(report.passed_checks, 0);
    assert!((report.pass_rate() - 0.0).abs() < 0.01);
}

#[test]
fn test_verification_report_clone() {
    let mut report = VerificationReport::new();
    report.add(CheckResult::pass("a", "ok"));
    let cloned = report.clone();
    assert_eq!(report.total_checks, cloned.total_checks);
    assert_eq!(report.passed, cloned.passed);
}

#[test]
fn test_verification_report_default() {
    let report = VerificationReport::default();
    assert!(report.passed);
    assert_eq!(report.total_checks, 0);
}

#[test]
fn test_verifier_default() {
    let v1 = Verifier::new();
    let v2 = Verifier::default();
    // Both should have same default pass rate
    assert!((v1.min_pass_rate - v2.min_pass_rate).abs() < 0.01);
}

#[test]
fn test_verifier_threshold() {
    let verifier = Verifier::new().with_min_pass_rate(80.0);

    let mut report = VerificationReport::new();
    report.add(CheckResult::pass("a", "ok"));
    report.add(CheckResult::pass("b", "ok"));
    report.add(CheckResult::pass("c", "ok"));
    report.add(CheckResult::pass("d", "ok"));
    report.add(CheckResult::fail("e", "error"));

    // 4/5 = 80%
    assert!(verifier.meets_threshold(&report));
}

#[test]
fn test_verifier_threshold_below() {
    let verifier = Verifier::new().with_min_pass_rate(90.0);

    let mut report = VerificationReport::new();
    report.add(CheckResult::pass("a", "ok"));
    report.add(CheckResult::fail("b", "err"));

    // 50% < 90%
    assert!(!verifier.meets_threshold(&report));
}

#[test]
fn test_tensor_verification() {
    let verifier = Verifier::new();

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "good".to_string(),
        TensorData::new(vec![1.0, 2.0, 3.0], vec![3]),
    );
    tensors.insert(
        "has_nan".to_string(),
        TensorData::new(vec![1.0, f32::NAN, 3.0], vec![3]),
    );

    let report = verifier.verify_tensors(&tensors).unwrap();

    // Should have checks for each tensor
    assert!(report.total_checks >= 4); // nan + inf checks for each

    // Should fail due to NaN
    assert!(!report.passed);
}

#[test]
fn test_tensor_verification_with_inf() {
    let verifier = Verifier::new();

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "has_inf".to_string(),
        TensorData::new(vec![1.0, f32::INFINITY, 3.0], vec![3]),
    );

    let report = verifier.verify_tensors(&tensors).unwrap();
    assert!(!report.passed);
}

#[test]
fn test_tensor_verification_with_neg_inf() {
    let verifier = Verifier::new();

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "has_neg_inf".to_string(),
        TensorData::new(vec![f32::NEG_INFINITY, 2.0, 3.0], vec![3]),
    );

    let report = verifier.verify_tensors(&tensors).unwrap();
    assert!(!report.passed);
}

#[test]
fn test_tensor_verification_all_good() {
    let verifier = Verifier::new();

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "tensor1".to_string(),
        TensorData::new(vec![1.0, 2.0, 3.0], vec![3]),
    );
    tensors.insert(
        "tensor2".to_string(),
        TensorData::new(vec![4.0, 5.0, 6.0, 7.0], vec![2, 2]),
    );

    let report = verifier.verify_tensors(&tensors).unwrap();
    assert!(report.passed);
}

#[test]
fn test_tensor_verification_shape_mismatch() {
    let verifier = Verifier::new();

    let mut tensors = BTreeMap::new();
    // Data has 3 elements but shape says 4
    tensors.insert(
        "bad_shape".to_string(),
        TensorData::new(vec![1.0, 2.0, 3.0], vec![2, 2]),
    );

    let report = verifier.verify_tensors(&tensors).unwrap();
    assert!(!report.passed);
}

#[test]
fn test_nonexistent_file() {
    let report = verify_apr("/nonexistent/path/model.apr").unwrap();
    assert!(!report.passed);
    assert!(report.checks[0].message.contains("not found"));
}

#[test]
fn test_verify_safetensors_nonexistent() {
    let report = verify_safetensors("/nonexistent/model.safetensors").unwrap();
    assert!(!report.passed);
}

#[test]
fn test_verify_safetensors_too_small() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&[0u8; 4]).unwrap();
    file.flush().unwrap();

    let report = verify_safetensors(file.path()).unwrap();
    assert!(!report.passed);
}

#[test]
fn test_verify_safetensors_valid_header() {
    let mut file = NamedTempFile::new().unwrap();
    // Header length (2 bytes as u64)
    let header = b"{}";
    let header_len = (header.len() as u64).to_le_bytes();
    file.write_all(&header_len).unwrap();
    file.write_all(header).unwrap();
    file.flush().unwrap();

    let report = verify_safetensors(file.path()).unwrap();
    // Should pass basic checks
    assert!(report
        .checks
        .iter()
        .any(|c| c.name == "A1_file_exists" && c.passed));
}

#[test]
fn test_verify_safetensors_header_too_large() {
    let mut file = NamedTempFile::new().unwrap();
    // Claim header is 200MB (too large)
    let header_len = (200_000_000u64).to_le_bytes();
    file.write_all(&header_len).unwrap();
    file.write_all(&[0u8; 100]).unwrap();
    file.flush().unwrap();

    let report = verify_safetensors(file.path()).unwrap();
    // Should have header limit check fail
    let has_limit_fail = report
        .checks
        .iter()
        .any(|c| c.name == "C11_header_limit" && !c.passed);
    assert!(has_limit_fail);
}

#[test]
fn test_verify_apr_small_file() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&[0u8; 10]).unwrap();
    file.flush().unwrap();

    let report = verify_apr(file.path()).unwrap();
    // File exists but is too small
    assert!(report
        .checks
        .iter()
        .any(|c| c.name == "A1_file_exists" && c.passed));
}

#[test]
fn test_verify_apr_wrong_magic() {
    let mut file = NamedTempFile::new().unwrap();
    // Write wrong magic bytes
    file.write_all(b"XXXX").unwrap();
    file.write_all(&[0u8; 60]).unwrap();
    file.flush().unwrap();

    let report = verify_apr(file.path()).unwrap();
    // Magic check should fail
    let has_magic_fail = report
        .checks
        .iter()
        .any(|c| c.name.contains("magic") && !c.passed);
    assert!(has_magic_fail);
}

#[test]
fn test_verify_apr_correct_magic() {
    let mut file = NamedTempFile::new().unwrap();
    // Write correct APR magic
    file.write_all(b"APR\0").unwrap();
    file.write_all(&[0u8; 60]).unwrap();
    file.flush().unwrap();

    let report = verify_apr(file.path()).unwrap();
    // Magic check should pass
    let has_magic_pass = report
        .checks
        .iter()
        .any(|c| c.name.contains("magic") && c.passed);
    assert!(has_magic_pass);
}

#[test]
fn test_verify_apr_potential_secret() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"APR\0").unwrap();
    file.write_all(&[0u8; 56]).unwrap();
    // Add something that looks like a secret
    file.write_all(b"api_key=secret123").unwrap();
    file.flush().unwrap();

    let report = verify_apr(file.path()).unwrap();
    // Should have secret check
    let has_secret_check = report.checks.iter().any(|c| c.name.contains("secret"));
    assert!(has_secret_check);
}

#[test]
fn test_convenience_functions() {
    // Test that convenience functions work
    let apr_result = verify_apr("/nonexistent.apr");
    assert!(apr_result.is_ok());

    let st_result = verify_safetensors("/nonexistent.safetensors");
    assert!(st_result.is_ok());
}

#[test]
fn test_verify_apr_secret_patterns() {
    // Test various secret patterns

    // PASSWORD pattern
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"APR\0").unwrap();
    file.write_all(&[0u8; 56]).unwrap();
    file.write_all(b"PASSWORD=hunter2").unwrap();
    file.flush().unwrap();
    let report = verify_apr(file.path()).unwrap();
    let has_fail = report
        .checks
        .iter()
        .any(|c| c.name.contains("secret") && !c.passed);
    assert!(has_fail);
}

#[test]
fn test_verify_apr_secret_sk_pattern() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"APR\0").unwrap();
    file.write_all(&[0u8; 56]).unwrap();
    file.write_all(b"sk-abc123def456").unwrap();
    file.flush().unwrap();
    let report = verify_apr(file.path()).unwrap();
    let has_fail = report
        .checks
        .iter()
        .any(|c| c.name.contains("secret") && !c.passed);
    assert!(has_fail);
}

#[test]
fn test_verify_apr_secret_private_key() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"APR\0").unwrap();
    file.write_all(&[0u8; 56]).unwrap();
    file.write_all(b"-----BEGIN PRIVATE KEY-----").unwrap();
    file.flush().unwrap();
    let report = verify_apr(file.path()).unwrap();
    let has_fail = report
        .checks
        .iter()
        .any(|c| c.name.contains("secret") && !c.passed);
    assert!(has_fail);
}

#[test]
fn test_verify_apr_secret_token() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"APR\0").unwrap();
    file.write_all(&[0u8; 56]).unwrap();
    file.write_all(b"auth_token=xyz").unwrap();
    file.flush().unwrap();
    let report = verify_apr(file.path()).unwrap();
    let has_fail = report
        .checks
        .iter()
        .any(|c| c.name.contains("secret") && !c.passed);
    assert!(has_fail);
}

#[test]
fn test_verify_apr_no_secrets_clean_file() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"APR\0").unwrap();
    file.write_all(&[0u8; 100]).unwrap();
    file.flush().unwrap();
    let report = verify_apr(file.path()).unwrap();
    let has_pass = report
        .checks
        .iter()
        .any(|c| c.name.contains("secret") && c.passed);
    assert!(has_pass);
}

#[test]
fn test_verify_safetensors_invalid_utf8() {
    let mut file = NamedTempFile::new().unwrap();
    // Valid header length
    let header_len = 10u64.to_le_bytes();
    file.write_all(&header_len).unwrap();
    // Invalid UTF-8 bytes
    file.write_all(&[0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89])
        .unwrap();
    file.flush().unwrap();

    let report = verify_safetensors(file.path()).unwrap();
    let has_utf8_fail = report
        .checks
        .iter()
        .any(|c| c.name.contains("json") && !c.passed);
    assert!(has_utf8_fail);
}

#[test]
fn test_verify_safetensors_truncated() {
    let mut file = NamedTempFile::new().unwrap();
    // Claim header is 100 bytes but only provide 8 bytes total
    let header_len = 100u64.to_le_bytes();
    file.write_all(&header_len).unwrap();
    file.flush().unwrap();

    let report = verify_safetensors(file.path()).unwrap();
    let has_truncated_fail = report
        .checks
        .iter()
        .any(|c| c.message.contains("truncated"));
    assert!(has_truncated_fail);
}

#[test]
fn test_verify_safetensors_invalid_json_structure() {
    let mut file = NamedTempFile::new().unwrap();
    // Valid UTF-8 but not valid JSON object structure
    let header = b"not a json object";
    let header_len = (header.len() as u64).to_le_bytes();
    file.write_all(&header_len).unwrap();
    file.write_all(header).unwrap();
    file.flush().unwrap();

    let report = verify_safetensors(file.path()).unwrap();
    let has_json_fail = report
        .checks
        .iter()
        .any(|c| c.name.contains("json") && !c.passed);
    assert!(has_json_fail);
}

#[test]
fn test_verifier_threshold_exact_boundary() {
    let verifier = Verifier::new().with_min_pass_rate(88.0);

    // Create report with exactly 88% pass rate (22 pass, 3 fail = 22/25 = 88%)
    let mut report = VerificationReport::new();
    for i in 0..22 {
        report.add(CheckResult::pass(format!("pass_{}", i), "ok"));
    }
    for i in 0..3 {
        report.add(CheckResult::fail(format!("fail_{}", i), "err"));
    }

    assert!(verifier.meets_threshold(&report));
}

#[test]
fn test_verifier_threshold_just_below() {
    let verifier = Verifier::new().with_min_pass_rate(88.0);

    // Create report with 87.5% pass rate (7 pass, 1 fail = 7/8 = 87.5%)
    let mut report = VerificationReport::new();
    for i in 0..7 {
        report.add(CheckResult::pass(format!("pass_{}", i), "ok"));
    }
    report.add(CheckResult::fail("fail", "err"));

    assert!(!verifier.meets_threshold(&report));
}

#[test]
fn test_check_result_debug() {
    let result = CheckResult::pass("test_name", "test message");
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("CheckResult"));
    assert!(debug_str.contains("test_name"));
}

#[test]
fn test_verification_report_debug() {
    let report = VerificationReport::new();
    let debug_str = format!("{:?}", report);
    assert!(debug_str.contains("VerificationReport"));
}

#[test]
fn test_tensor_empty_collection() {
    let verifier = Verifier::new();
    let tensors = BTreeMap::new();
    let report = verifier.verify_tensors(&tensors).unwrap();
    assert!(report.passed);
    assert_eq!(report.total_checks, 0);
}

#[test]
fn test_tensor_single_element() {
    let verifier = Verifier::new();
    let mut tensors = BTreeMap::new();
    tensors.insert("single".to_string(), TensorData::new(vec![42.0], vec![1]));
    let report = verifier.verify_tensors(&tensors).unwrap();
    assert!(report.passed);
}

#[test]
fn test_tensor_large_shape() {
    let verifier = Verifier::new();
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "large".to_string(),
        TensorData::new(vec![1.0; 1000], vec![10, 10, 10]),
    );
    let report = verifier.verify_tensors(&tensors).unwrap();
    assert!(report.passed);
}

#[test]
fn test_tensor_multiple_issues() {
    let verifier = Verifier::new();
    let mut tensors = BTreeMap::new();
    // One tensor with NaN
    tensors.insert(
        "has_nan".to_string(),
        TensorData::new(vec![f32::NAN], vec![1]),
    );
    // One tensor with Inf
    tensors.insert(
        "has_inf".to_string(),
        TensorData::new(vec![f32::INFINITY], vec![1]),
    );
    // One tensor with shape mismatch
    tensors.insert("bad_shape".to_string(), TensorData::new(vec![1.0], vec![2]));

    let report = verifier.verify_tensors(&tensors).unwrap();
    assert!(!report.passed);
    // Should have multiple failures
    let fail_count = report.checks.iter().filter(|c| !c.passed).count();
    assert!(fail_count >= 3);
}

#[test]
fn test_verify_apr_size_exactly_64() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"APR\0").unwrap();
    file.write_all(&[0u8; 60]).unwrap(); // Total 64 bytes
    file.flush().unwrap();

    let report = verify_apr(file.path()).unwrap();
    let size_check = report.checks.iter().find(|c| c.name.contains("size"));
    assert!(size_check.is_some());
    assert!(size_check.unwrap().passed);
}

#[test]
fn test_verify_apr_size_below_64() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"APR\0").unwrap();
    file.write_all(&[0u8; 50]).unwrap(); // Total 54 bytes (< 64)
    file.flush().unwrap();

    let report = verify_apr(file.path()).unwrap();
    let size_check = report.checks.iter().find(|c| c.name.contains("size"));
    assert!(size_check.is_some());
    assert!(!size_check.unwrap().passed);
}

#[test]
fn test_verifier_with_custom_pass_rate() {
    let verifier = Verifier::new().with_min_pass_rate(50.0);
    assert!((verifier.min_pass_rate - 50.0).abs() < 0.01);
}

#[test]
fn test_verifier_meets_threshold_empty_report() {
    let verifier = Verifier::new();
    let report = VerificationReport::new();
    // Empty report has 100% pass rate
    assert!(verifier.meets_threshold(&report));
}

#[cfg(unix)]
#[test]
fn test_verify_apr_permission_denied() {
    use std::os::unix::fs::PermissionsExt;

    // Root bypasses file permission checks, so this test is meaningless as root
    if std::env::var("USER").unwrap_or_default() == "root"
        || std::fs::read_to_string("/proc/self/status")
            .map(|s| s.lines().any(|l| l.starts_with("Uid:\t0\t")))
            .unwrap_or(false)
    {
        eprintln!("skipping test_verify_apr_permission_denied: running as root");
        return;
    }

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"APR\0").unwrap();
    file.write_all(&[0u8; 60]).unwrap();
    file.flush().unwrap();
    let path = file.path().to_path_buf();

    // Remove all permissions so File::open fails (A2_readable)
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();

    let report = verify_apr(&path).unwrap();

    // Restore permissions for cleanup
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

    // A1 should pass (file exists), A2 should fail (can't open)
    let a1 = report.checks.iter().find(|c| c.name == "A1_file_exists");
    assert!(a1.is_some());
    assert!(a1.unwrap().passed);

    let a2 = report.checks.iter().find(|c| c.name == "A2_readable");
    assert!(a2.is_some());
    assert!(!a2.unwrap().passed);
    assert!(a2.unwrap().message.contains("Cannot read"));
}
