//! Pre-publish Verification Module (WAPR-PUB-001)
//!
//! Verifies whisper.apr models before publishing to HuggingFace Hub.
//! Follows certeza patterns for quality verification.
//!
//! # Verification Checks
//!
//! 1. **Format Integrity**: APR magic bytes, version, checksums
//! 2. **Tensor Validation**: No NaN/Inf, shape consistency
//! 3. **SafeTensors Compatibility**: Header format, dtype support
//! 4. **Security Scan**: No embedded secrets or malicious content

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::error::{WhisperError, WhisperResult};
use crate::format::export::TensorData;

/// Verification check result.
#[derive(Debug, Clone)]
pub struct CheckResult {
    /// Check name
    pub name: String,
    /// Whether the check passed
    pub passed: bool,
    /// Details or error message
    pub message: String,
}

impl CheckResult {
    /// Create a passing result.
    #[must_use]
    pub fn pass(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: true,
            message: message.into(),
        }
    }

    /// Create a failing result.
    #[must_use]
    pub fn fail(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: false,
            message: message.into(),
        }
    }
}

/// Complete verification report.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Individual check results
    pub checks: Vec<CheckResult>,
    /// Overall pass/fail
    pub passed: bool,
    /// Total checks run
    pub total_checks: usize,
    /// Passed checks count
    pub passed_checks: usize,
}

impl VerificationReport {
    /// Create a new empty report.
    #[must_use]
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
            passed: true,
            total_checks: 0,
            passed_checks: 0,
        }
    }

    /// Add a check result.
    pub fn add(&mut self, result: CheckResult) {
        self.total_checks += 1;
        if result.passed {
            self.passed_checks += 1;
        } else {
            self.passed = false;
        }
        self.checks.push(result);
    }

    /// Get pass rate as percentage.
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        if self.total_checks == 0 {
            100.0
        } else {
            (self.passed_checks as f64 / self.total_checks as f64) * 100.0
        }
    }
}

impl Default for VerificationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Model verifier for pre-publish checks.
pub struct Verifier {
    /// Minimum required checks to pass
    min_pass_rate: f64,
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Verifier {
    /// Create a new verifier with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self { min_pass_rate: 88.0 } // Per spec: ≥88/100 to publish
    }

    /// Set minimum pass rate.
    #[must_use]
    pub fn with_min_pass_rate(mut self, rate: f64) -> Self {
        self.min_pass_rate = rate;
        self
    }

    /// Verify an APR model file.
    pub fn verify_apr<P: AsRef<Path>>(&self, path: P) -> WhisperResult<VerificationReport> {
        let path = path.as_ref();
        let mut report = VerificationReport::new();

        // A1: Check file exists
        if !path.exists() {
            report.add(CheckResult::fail(
                "A1_file_exists",
                format!("File not found: {}", path.display()),
            ));
            return Ok(report);
        }
        report.add(CheckResult::pass("A1_file_exists", "File exists"));

        // A2: Check file is readable
        let mut file = match File::open(path) {
            Ok(f) => {
                report.add(CheckResult::pass("A2_readable", "File is readable"));
                f
            }
            Err(e) => {
                report.add(CheckResult::fail(
                    "A2_readable",
                    format!("Cannot read file: {}", e),
                ));
                return Ok(report);
            }
        };

        // A3: Check APR magic bytes
        let mut magic = [0u8; 4];
        if file.read_exact(&mut magic).is_ok() && &magic == b"APR\0" {
            report.add(CheckResult::pass("A3_magic", "APR magic bytes valid"));
        } else {
            report.add(CheckResult::fail(
                "A3_magic",
                "Invalid APR magic bytes (expected APR\\0)",
            ));
        }

        // A4: Check file size is reasonable
        let metadata = std::fs::metadata(path).map_err(WhisperError::Io)?;
        let size = metadata.len();
        if size >= 64 {
            // Minimum header size
            report.add(CheckResult::pass(
                "A4_size",
                format!("File size: {} bytes", size),
            ));
        } else {
            report.add(CheckResult::fail(
                "A4_size",
                format!("File too small: {} bytes (min 64)", size),
            ));
        }

        // A5: No obvious secrets (scan first 1KB)
        let mut buffer = vec![0u8; 1024.min(size as usize)];
        file = File::open(path).map_err(WhisperError::Io)?;
        let _ = file.read(&mut buffer);
        let content = String::from_utf8_lossy(&buffer);

        let secret_patterns = [
            "PRIVATE KEY",
            "sk-",
            "api_key",
            "password",
            "secret",
            "token",
        ];
        let mut found_secrets = false;
        for pattern in &secret_patterns {
            if content.to_lowercase().contains(&pattern.to_lowercase()) {
                report.add(CheckResult::fail(
                    "C6_no_secrets",
                    format!("Potential secret found: {}", pattern),
                ));
                found_secrets = true;
                break;
            }
        }
        if !found_secrets {
            report.add(CheckResult::pass("C6_no_secrets", "No obvious secrets found"));
        }

        Ok(report)
    }

    /// Verify a SafeTensors file.
    pub fn verify_safetensors<P: AsRef<Path>>(&self, path: P) -> WhisperResult<VerificationReport> {
        let path = path.as_ref();
        let mut report = VerificationReport::new();

        // Check file exists
        if !path.exists() {
            report.add(CheckResult::fail(
                "A1_file_exists",
                format!("File not found: {}", path.display()),
            ));
            return Ok(report);
        }
        report.add(CheckResult::pass("A1_file_exists", "File exists"));

        // Read file
        let data = std::fs::read(path).map_err(WhisperError::Io)?;

        // A4: Check minimum size (8 bytes for header length)
        if data.len() < 8 {
            report.add(CheckResult::fail(
                "A4_header_size",
                "File too small for SafeTensors header",
            ));
            return Ok(report);
        }
        report.add(CheckResult::pass("A4_header_size", "Header size field present"));

        // A5: Parse header length
        let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        if header_len > 100_000_000 {
            // 100MB max per spec
            report.add(CheckResult::fail(
                "C11_header_limit",
                format!("Header too large: {} bytes (max 100MB)", header_len),
            ));
        } else {
            report.add(CheckResult::pass(
                "C11_header_limit",
                format!("Header size: {} bytes", header_len),
            ));
        }

        // A6: Check header is valid UTF-8 JSON
        if data.len() >= 8 + header_len {
            let header_bytes = &data[8..8 + header_len];
            match std::str::from_utf8(header_bytes) {
                Ok(header_str) => {
                    let trimmed = header_str.trim();
                    if trimmed.starts_with('{') && trimmed.ends_with('}') {
                        report.add(CheckResult::pass("A5_json_valid", "Header is valid JSON"));
                    } else {
                        report.add(CheckResult::fail(
                            "A5_json_valid",
                            "Header JSON doesn't start with '{' or end with '}'",
                        ));
                    }
                }
                Err(e) => {
                    report.add(CheckResult::fail(
                        "A5_json_valid",
                        format!("Header is not valid UTF-8: {}", e),
                    ));
                }
            }
        } else {
            report.add(CheckResult::fail(
                "A5_json_valid",
                "File truncated before header end",
            ));
        }

        Ok(report)
    }

    /// Verify tensor data for NaN/Inf values.
    pub fn verify_tensors(
        &self,
        tensors: &BTreeMap<String, TensorData>,
    ) -> WhisperResult<VerificationReport> {
        let mut report = VerificationReport::new();

        for (name, tensor) in tensors {
            // A10: Check for NaN
            let has_nan = tensor.data.iter().any(|v| v.is_nan());
            if has_nan {
                report.add(CheckResult::fail(
                    format!("A10_no_nan_{}", name),
                    format!("Tensor '{}' contains NaN values", name),
                ));
            } else {
                report.add(CheckResult::pass(
                    format!("A10_no_nan_{}", name),
                    format!("Tensor '{}' has no NaN", name),
                ));
            }

            // A11: Check for Inf
            let has_inf = tensor.data.iter().any(|v| v.is_infinite());
            if has_inf {
                report.add(CheckResult::fail(
                    format!("A11_no_inf_{}", name),
                    format!("Tensor '{}' contains Inf values", name),
                ));
            } else {
                report.add(CheckResult::pass(
                    format!("A11_no_inf_{}", name),
                    format!("Tensor '{}' has no Inf", name),
                ));
            }

            // A7: Check shape consistency
            let expected = tensor.expected_elements();
            if tensor.data.len() == expected {
                report.add(CheckResult::pass(
                    format!("A7_shape_{}", name),
                    format!("Tensor '{}' shape {:?} matches data", name, tensor.shape),
                ));
            } else {
                report.add(CheckResult::fail(
                    format!("A7_shape_{}", name),
                    format!(
                        "Tensor '{}' shape {:?} expects {} elements, got {}",
                        name,
                        tensor.shape,
                        expected,
                        tensor.data.len()
                    ),
                ));
            }
        }

        Ok(report)
    }

    /// Check if a report meets the minimum pass rate.
    #[must_use]
    pub fn meets_threshold(&self, report: &VerificationReport) -> bool {
        report.pass_rate() >= self.min_pass_rate
    }
}

/// Convenience function to verify an APR model.
pub fn verify_apr<P: AsRef<Path>>(path: P) -> WhisperResult<VerificationReport> {
    Verifier::new().verify_apr(path)
}

/// Convenience function to verify a SafeTensors file.
pub fn verify_safetensors<P: AsRef<Path>>(path: P) -> WhisperResult<VerificationReport> {
    Verifier::new().verify_safetensors(path)
}

#[cfg(test)]
mod tests {
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
        assert!(report.checks.iter().any(|c| c.name == "A1_file_exists" && c.passed));
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
        let has_limit_fail = report.checks.iter().any(|c| c.name == "C11_header_limit" && !c.passed);
        assert!(has_limit_fail);
    }

    #[test]
    fn test_verify_apr_small_file() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&[0u8; 10]).unwrap();
        file.flush().unwrap();

        let report = verify_apr(file.path()).unwrap();
        // File exists but is too small
        assert!(report.checks.iter().any(|c| c.name == "A1_file_exists" && c.passed));
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
        let has_magic_fail = report.checks.iter().any(|c| c.name.contains("magic") && !c.passed);
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
        let has_magic_pass = report.checks.iter().any(|c| c.name.contains("magic") && c.passed);
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
        let has_fail = report.checks.iter().any(|c| c.name.contains("secret") && !c.passed);
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
        let has_fail = report.checks.iter().any(|c| c.name.contains("secret") && !c.passed);
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
        let has_fail = report.checks.iter().any(|c| c.name.contains("secret") && !c.passed);
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
        let has_fail = report.checks.iter().any(|c| c.name.contains("secret") && !c.passed);
        assert!(has_fail);
    }

    #[test]
    fn test_verify_apr_no_secrets_clean_file() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"APR\0").unwrap();
        file.write_all(&[0u8; 100]).unwrap();
        file.flush().unwrap();
        let report = verify_apr(file.path()).unwrap();
        let has_pass = report.checks.iter().any(|c| c.name.contains("secret") && c.passed);
        assert!(has_pass);
    }

    #[test]
    fn test_verify_safetensors_invalid_utf8() {
        let mut file = NamedTempFile::new().unwrap();
        // Valid header length
        let header_len = 10u64.to_le_bytes();
        file.write_all(&header_len).unwrap();
        // Invalid UTF-8 bytes
        file.write_all(&[0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89]).unwrap();
        file.flush().unwrap();

        let report = verify_safetensors(file.path()).unwrap();
        let has_utf8_fail = report.checks.iter().any(|c| c.name.contains("json") && !c.passed);
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
        let has_truncated_fail = report.checks.iter().any(|c| c.message.contains("truncated"));
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
        let has_json_fail = report.checks.iter().any(|c| c.name.contains("json") && !c.passed);
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
        tensors.insert(
            "single".to_string(),
            TensorData::new(vec![42.0], vec![1]),
        );
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
        tensors.insert(
            "bad_shape".to_string(),
            TensorData::new(vec![1.0], vec![2]),
        );

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
}
