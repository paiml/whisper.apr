//! Validation check and report types

/// Result of a single validation check
#[derive(Debug, Clone)]
pub struct ValidationCheck {
    /// Check ID (1-25)
    pub id: u8,
    /// Check name
    pub name: String,
    /// Whether the check passed
    pub passed: bool,
    /// Detailed message
    pub message: String,
    /// Category (A-E)
    pub category: char,
}

impl ValidationCheck {
    /// Create a passing check
    pub(crate) fn pass(id: u8, category: char, name: &str, message: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            passed: true,
            message: message.to_string(),
            category,
        }
    }

    /// Create a failing check
    pub(crate) fn fail(id: u8, category: char, name: &str, message: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            passed: false,
            message: message.to_string(),
            category,
        }
    }
}

/// Complete validation report
#[derive(Debug)]
pub struct ValidationReport {
    /// Individual check results
    pub checks: Vec<ValidationCheck>,
    /// Total score
    pub score: u8,
    /// Maximum possible score
    pub max_score: u8,
    /// Whether validation passed (23/25 or better, no critical failures)
    pub passed: bool,
    /// Critical failures (automatic rejection)
    pub critical_failures: Vec<String>,
}

impl ValidationReport {
    /// Create new report from checks
    pub(crate) fn from_checks(
        checks: Vec<ValidationCheck>,
        critical_failures: Vec<String>,
    ) -> Self {
        let score = checks.iter().filter(|c| c.passed).count() as u8;
        let max_score = checks.len() as u8;
        let passed = score >= 23 && critical_failures.is_empty();

        Self {
            checks,
            score,
            max_score,
            passed,
            critical_failures,
        }
    }

    /// Get checks by category
    #[must_use]
    pub fn checks_by_category(&self, category: char) -> Vec<&ValidationCheck> {
        self.checks
            .iter()
            .filter(|c| c.category == category)
            .collect()
    }
}
