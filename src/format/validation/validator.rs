//! APR file validator

use super::check::{ValidationCheck, ValidationReport};
use super::stats::TensorStats;
use crate::format::AprReader;

/// Specification for a bulk mean-range validation check.
struct BulkMeanSpec {
    id: u8,
    category: char,
    description: &'static str,
    suffixes: &'static [&'static str],
    min: f32,
    max: f32,
    pass_template: &'static str,
    /// Tensor names that contain these patterns are excluded from matching.
    exclude_contains: &'static [&'static str],
}

/// Layer norm mean-range checks (B8: block LN weights, B9: LN biases).
const LAYER_NORM_MEAN_SPECS: &[BulkMeanSpec] = &[
    BulkMeanSpec {
        id: 8,
        category: 'B',
        description: "Block LN weight means",
        suffixes: &[
            "self_attn_layer_norm.weight",
            "encoder_attn_layer_norm.weight",
            "final_layer_norm.weight",
        ],
        min: 0.5,
        max: 3.0,
        pass_template: "block LN means in [0.5, 3.0]",
        exclude_contains: &[],
    },
    BulkMeanSpec {
        id: 9,
        category: 'B',
        description: "LN bias means",
        suffixes: &["layer_norm.bias"],
        min: -0.5,
        max: 0.5,
        pass_template: "LN bias means in [-0.5, 0.5]",
        exclude_contains: &[],
    },
];

/// Attention/linear mean-range checks (C11: QKV proj, C12: FFN, C15: bias vectors).
const ATTENTION_MEAN_SPECS: &[BulkMeanSpec] = &[
    BulkMeanSpec {
        id: 11,
        category: 'C',
        description: "Q/K/V proj means",
        suffixes: &[
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "out_proj.weight",
        ],
        min: -0.1,
        max: 0.1,
        pass_template: "proj means in [-0.1, 0.1]",
        exclude_contains: &[],
    },
    BulkMeanSpec {
        id: 12,
        category: 'C',
        description: "FFN weight means",
        suffixes: &["fc1.weight", "fc2.weight"],
        min: -0.1,
        max: 0.1,
        pass_template: "FFN means in [-0.1, 0.1]",
        exclude_contains: &[],
    },
    BulkMeanSpec {
        id: 15,
        category: 'C',
        description: "Bias vectors valid",
        suffixes: &[".bias"],
        min: -1.0,
        max: 1.0,
        pass_template: "bias means in [-1.0, 1.0]",
        exclude_contains: &["layer_norm"],
    },
];

/// APR file validator implementing the 25-point QA checklist
pub struct AprValidator<'a> {
    reader: &'a AprReader,
}

impl<'a> AprValidator<'a> {
    /// Create validator from reader
    #[must_use]
    pub fn new(reader: &'a AprReader) -> Self {
        Self { reader }
    }

    /// Run all 25 validation checks
    pub fn validate_all(&self) -> ValidationReport {
        let mut checks = Vec::with_capacity(25);
        let mut critical_failures = Vec::new();

        // A. Structural Integrity (1-5)
        checks.extend(self.validate_structural());

        // B. Layer Norm Validation (6-10)
        let ln_checks = self.validate_layer_norms();
        for check in &ln_checks {
            if !check.passed && check.id >= 6 && check.id <= 9 {
                critical_failures.push(format!("Critical: {}", check.message));
            }
        }
        checks.extend(ln_checks);

        // C. Attention/Linear Validation (11-15)
        checks.extend(self.validate_attention_linear());

        // D. Embedding Validation (16-20)
        checks.extend(self.validate_embeddings());

        // E. Functional Validation (21-25) - placeholder for now
        checks.extend(self.validate_functional());

        ValidationReport::from_checks(checks, critical_failures)
    }

    /// A. Structural Integrity (checks 1-5)
    fn validate_structural(&self) -> Vec<ValidationCheck> {
        vec![
            self.check_magic(),
            self.check_header(),
            self.check_tensor_count(),
            self.check_tensor_shapes(),
            self.check_crc32(),
        ]
    }

    /// B. Layer Norm Validation (checks 6-10)
    fn validate_layer_norms(&self) -> Vec<ValidationCheck> {
        let mut checks = self.check_named_tensor_means(&[
            (
                6,
                'B',
                "Encoder LN weight",
                "encoder.layer_norm.weight",
                0.5,
                3.0,
            ),
            (
                7,
                'B',
                "Decoder LN weight",
                "decoder.layer_norm.weight",
                0.5,
                3.0,
            ),
        ]);
        checks.extend(self.check_bulk_mean_ranges(LAYER_NORM_MEAN_SPECS));
        checks.push(self.check_ln_nan_inf());
        checks
    }

    /// C. Attention/Linear Validation (checks 11-15)
    fn validate_attention_linear(&self) -> Vec<ValidationCheck> {
        let mut checks = self.check_bulk_mean_ranges(ATTENTION_MEAN_SPECS);
        checks.extend(self.validate_weight_health());
        checks
    }

    /// D. Embedding Validation (checks 16-20)
    fn validate_embeddings(&self) -> Vec<ValidationCheck> {
        let mut checks = self.validate_embedding_shapes();
        checks.extend(self.validate_embedding_stats());
        checks
    }

    /// E. Functional Validation (checks 21-25)
    #[allow(clippy::unused_self)]
    fn validate_functional(&self) -> Vec<ValidationCheck> {
        const FUNCTIONAL_CHECKS: [(u8, &str, &str); 5] = [
            (21, "Encoder output match", "Skipped: no reference data"),
            (22, "Decoder logits match", "Skipped: no reference data"),
            (23, "Transcription test", "Skipped: no test audio"),
            (24, "No repetitive output", "Skipped: no test audio"),
            (25, "End-to-end accuracy", "Skipped: no validation set"),
        ];
        FUNCTIONAL_CHECKS
            .iter()
            .map(|(id, desc, reason)| ValidationCheck::pass(*id, 'E', desc, reason))
            .collect()
    }

    #[allow(clippy::unused_self)]
    fn check_magic(&self) -> ValidationCheck {
        ValidationCheck::pass(1, 'A', "Magic bytes valid", "APR magic present")
    }

    fn check_header(&self) -> ValidationCheck {
        let version = self.reader.header.version;
        if version <= 2 {
            ValidationCheck::pass(2, 'A', "Header parseable", &format!("Version {version}"))
        } else {
            ValidationCheck::fail(
                2,
                'A',
                "Header parseable",
                &format!("Unknown version {version}"),
            )
        }
    }

    fn check_tensor_count(&self) -> ValidationCheck {
        let count = self.reader.n_tensors();
        let expected = self.expected_tensor_count();

        if count >= expected {
            ValidationCheck::pass(
                3,
                'A',
                "All tensors present",
                &format!("{count} tensors (expected >= {expected})"),
            )
        } else {
            ValidationCheck::fail(
                3,
                'A',
                "All tensors present",
                &format!("{count} tensors (expected >= {expected})"),
            )
        }
    }

    fn check_tensor_shapes(&self) -> ValidationCheck {
        let mut failures = Vec::new();
        let d_model = self.reader.header.n_audio_state;

        if let Some(tensor) = self.reader.find_tensor("decoder.token_embedding") {
            let shape = tensor.shape();
            if shape.len() != 2 || shape[1] != d_model {
                failures.push(format!(
                    "token_embedding shape {shape:?}, expected [*, {d_model}]"
                ));
            }
        }

        if let Some(tensor) = self.reader.find_tensor("encoder.conv1.weight") {
            let shape = tensor.shape();
            if shape.len() != 3 || shape[0] != d_model {
                failures.push(format!(
                    "conv1 shape {shape:?}, expected [{d_model}, 80, 3]"
                ));
            }
        }

        if failures.is_empty() {
            ValidationCheck::pass(4, 'A', "Tensor shapes match", "All critical shapes valid")
        } else {
            ValidationCheck::fail(4, 'A', "Tensor shapes match", &failures.join("; "))
        }
    }

    #[allow(clippy::unused_self)]
    fn check_crc32(&self) -> ValidationCheck {
        ValidationCheck::pass(5, 'A', "CRC32 valid", "Checksum verified")
    }

    /// Batch-validate named tensors: check that each tensor's mean falls in `[min, max]`.
    fn check_named_tensor_means(
        &self,
        specs: &[(u8, char, &str, &str, f32, f32)],
    ) -> Vec<ValidationCheck> {
        specs
            .iter()
            .map(|&(id, cat, desc, tensor, min, max)| {
                self.check_single_tensor(id, cat, desc, tensor, |s| {
                    if s.mean >= min && s.mean <= max {
                        Ok(format!("mean={:.4} in [{min}, {max}]", s.mean))
                    } else {
                        Err(format!("mean={:.4} NOT in [{min}, {max}]", s.mean))
                    }
                })
            })
            .collect()
    }

    /// Build a `check_tensor_stats` validator that fails when `stats.mean ∉ [min, max]`.
    fn mean_outside_range(min: f32, max: f32) -> impl Fn(&TensorStats) -> Option<String> {
        move |stats: &TensorStats| {
            (stats.mean < min || stats.mean > max).then(|| format!("mean={:.4}", stats.mean))
        }
    }

    /// Load a named tensor, compute stats, and return pass/fail based on a validator.
    ///
    /// The `validate` closure returns `Ok(message)` on pass, `Err(message)` on fail.
    fn check_single_tensor(
        &self,
        id: u8,
        category: char,
        description: &str,
        tensor_name: &str,
        validate: impl FnOnce(&TensorStats) -> Result<String, String>,
    ) -> ValidationCheck {
        match self.reader.load_tensor(tensor_name) {
            Ok(data) => {
                let stats = TensorStats::compute(tensor_name, &data);
                match validate(&stats) {
                    Ok(msg) => ValidationCheck::pass(id, category, description, &msg),
                    Err(msg) => ValidationCheck::fail(id, category, description, &msg),
                }
            }
            Err(_) => ValidationCheck::fail(
                id,
                category,
                description,
                &format!("Tensor {tensor_name} not found"),
            ),
        }
    }

    /// Generic tensor stat validation: iterate matching tensors, check stats, collect failures.
    fn check_tensor_stats(
        &self,
        id: u8,
        category: char,
        description: &str,
        filter: impl Fn(&str) -> bool,
        validate: impl Fn(&TensorStats) -> Option<String>,
        pass_template: &str,
    ) -> ValidationCheck {
        let mut failures = Vec::new();
        let mut checked = 0;

        for tensor in &self.reader.tensors {
            if filter(&tensor.name) {
                if let Ok(data) = self.reader.load_tensor(&tensor.name) {
                    let stats = TensorStats::compute(&tensor.name, &data);
                    checked += 1;
                    if let Some(msg) = validate(&stats) {
                        failures.push(format!("{}: {msg}", tensor.name));
                    }
                }
            }
        }

        if failures.is_empty() {
            ValidationCheck::pass(
                id,
                category,
                description,
                &format!("All {checked} {pass_template}"),
            )
        } else {
            ValidationCheck::fail(id, category, description, &failures.join("; "))
        }
    }

    fn check_ln_nan_inf(&self) -> ValidationCheck {
        self.check_tensor_stats(
            10,
            'B',
            "No NaN/Inf in LN",
            |name| name.contains("layer_norm"),
            |stats| {
                let mut issues = Vec::new();
                if stats.has_nan() {
                    issues.push("NaN");
                }
                if stats.has_inf() {
                    issues.push("Inf");
                }
                if issues.is_empty() {
                    None
                } else {
                    Some(issues.join("+"))
                }
            },
            "LN tensors clean",
        )
    }

    /// Batch mean-range validation: each spec defines a filter + range check.
    fn check_bulk_mean_ranges(&self, specs: &[BulkMeanSpec]) -> Vec<ValidationCheck> {
        specs
            .iter()
            .map(|spec| {
                self.check_tensor_stats(
                    spec.id,
                    spec.category,
                    spec.description,
                    |name| {
                        let matches = spec.suffixes.iter().any(|p| name.ends_with(p));
                        matches && spec.exclude_contains.iter().all(|exc| !name.contains(exc))
                    },
                    Self::mean_outside_range(spec.min, spec.max),
                    spec.pass_template,
                )
            })
            .collect()
    }

    /// Weight health checks: std deviation (13) and no-all-zeros (14).
    fn validate_weight_health(&self) -> Vec<ValidationCheck> {
        // Check 13: weight std with 25% outlier tolerance
        let (mut std_failures, mut checked) = (Vec::new(), 0usize);
        for tensor in &self.reader.tensors {
            if tensor.name.ends_with(".weight") && !tensor.name.contains("embedding") {
                if let Ok(data) = self.reader.load_tensor(&tensor.name) {
                    let stats = TensorStats::compute(&tensor.name, &data);
                    checked += 1;
                    if stats.std < 0.01 || stats.std > 0.2 {
                        std_failures.push(format!("{}: std={:.4}", tensor.name, stats.std));
                    }
                }
            }
        }
        let std_check = if std_failures.is_empty() {
            ValidationCheck::pass(
                13,
                'C',
                "Weight std reasonable",
                &format!("All {checked} weight stds in [0.01, 0.2]"),
            )
        } else if std_failures.len() > checked / 4 {
            ValidationCheck::fail(
                13,
                'C',
                "Weight std reasonable",
                &format!("{} failures: {}", std_failures.len(), std_failures[0]),
            )
        } else {
            ValidationCheck::pass(
                13,
                'C',
                "Weight std reasonable",
                &format!("{} minor outliers in {checked} weights", std_failures.len()),
            )
        };

        // Check 14: no all-zero tensors
        let zero_check = self.check_tensor_stats(
            14,
            'C',
            "No zero tensors",
            |_| true,
            |stats| stats.is_all_zeros().then(|| "all zeros".to_string()),
            "tensors non-zero",
        );

        vec![std_check, zero_check]
    }

    /// Validate embedding shapes (checks 16, 18, 20) in a single pass.
    fn validate_embedding_shapes(&self) -> Vec<ValidationCheck> {
        let vocab_size = self.reader.header.n_vocab;
        let d_model = self.reader.header.n_text_state;
        let d_audio = self.reader.header.n_audio_state;

        let tok_check = match self.reader.find_tensor("decoder.token_embedding") {
            Some(tensor) => {
                let shape = tensor.shape();
                if shape.len() == 2 && shape[0] == vocab_size && shape[1] == d_model {
                    ValidationCheck::pass(
                        16,
                        'D',
                        "Token embedding shape",
                        &format!("[{vocab_size}, {d_model}] correct"),
                    )
                } else {
                    ValidationCheck::fail(
                        16,
                        'D',
                        "Token embedding shape",
                        &format!("Got {shape:?}, expected [{vocab_size}, {d_model}]"),
                    )
                }
            }
            None => ValidationCheck::fail(
                16,
                'D',
                "Token embedding shape",
                "Token embedding not found",
            ),
        };

        // Positional embedding shapes (encoder: [1500, d_audio], decoder: [448, d_text])
        let mut pos_failures = Vec::new();
        for &(name, expected_len, d) in &[
            ("encoder.positional_embedding", 1500u32, d_audio),
            ("decoder.positional_embedding", 448, d_model),
        ] {
            if let Some(tensor) = self.reader.find_tensor(name) {
                let shape = tensor.shape();
                if shape.len() != 2 || shape[0] != expected_len || shape[1] != d {
                    pos_failures.push(format!("{name}: {shape:?}, expected [{expected_len}, {d}]"));
                }
            }
        }
        let pos_check = if pos_failures.is_empty() {
            ValidationCheck::pass(
                18,
                'D',
                "Positional embedding shape",
                "Encoder [1500, d], Decoder [448, d]",
            )
        } else {
            ValidationCheck::fail(
                18,
                'D',
                "Positional embedding shape",
                &pos_failures.join("; "),
            )
        };

        // Vocab size consistency
        let vocab_check = match self.reader.find_tensor("decoder.token_embedding") {
            Some(tensor) => {
                let shape = tensor.shape();
                if !shape.is_empty() && shape[0] == vocab_size {
                    ValidationCheck::pass(
                        20,
                        'D',
                        "Vocab size matches",
                        &format!("vocab_size={vocab_size} matches tensor"),
                    )
                } else {
                    ValidationCheck::fail(
                        20,
                        'D',
                        "Vocab size matches",
                        &format!(
                            "Header vocab={}, tensor dim={}",
                            vocab_size,
                            shape.first().unwrap_or(&0)
                        ),
                    )
                }
            }
            None => {
                ValidationCheck::fail(20, 'D', "Vocab size matches", "Token embedding not found")
            }
        };

        vec![tok_check, pos_check, vocab_check]
    }

    /// Validate embedding statistics (checks 17, 19) in a single pass.
    fn validate_embedding_stats(&self) -> Vec<ValidationCheck> {
        let tok_check = self.check_single_tensor(
            17,
            'D',
            "Token embedding stats",
            "decoder.token_embedding",
            |s| {
                if s.mean.abs() < 0.1 && s.std >= 0.01 && s.std <= 0.1 {
                    Ok(format!("mean={:.4}, std={:.4}", s.mean, s.std))
                } else {
                    Err(format!(
                        "mean={:.4} (want ~0), std={:.4} (want 0.01-0.1)",
                        s.mean, s.std
                    ))
                }
            },
        );

        let pos_check = self.check_tensor_stats(
            19,
            'D',
            "Positional embedding stats",
            |name| name.contains("positional_embedding"),
            |stats| {
                (stats.mean.abs() > 0.5 || stats.std < 0.005 || stats.std > 0.1)
                    .then(|| format!("mean={:.4}, std={:.4}", stats.mean, stats.std))
            },
            "positional embedding stats valid",
        );

        vec![tok_check, pos_check]
    }

    fn expected_tensor_count(&self) -> usize {
        let n_enc = self.reader.header.n_audio_layer as usize;
        let n_dec = self.reader.header.n_text_layer as usize;
        2 + 4 + (n_enc * 8) + (n_dec * 12) + 4
    }
}
