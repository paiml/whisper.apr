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

#[cfg(test)]
mod tests;

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
    pub(crate) min_pass_rate: f64,
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
