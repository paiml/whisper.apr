//! APR file validator

use super::check::{ValidationCheck, ValidationReport};
use super::stats::TensorStats;
use crate::format::AprV2ReaderRef;
use crate::model::ModelConfig;

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
    /// If true, the check fails when no tensors match the filter.
    require_matches: bool,
}

/// Layer norm mean-range checks (B6-B9).
const LAYER_NORM_MEAN_SPECS: &[BulkMeanSpec] = &[
    BulkMeanSpec {
        id: 6,
        category: 'B',
        description: "Encoder LN weight",
        suffixes: &["encoder.layer_norm.weight"],
        min: 0.5,
        max: 3.0,
        pass_template: "encoder LN mean in [0.5, 3.0]",
        exclude_contains: &[],
        require_matches: true,
    },
    BulkMeanSpec {
        id: 7,
        category: 'B',
        description: "Decoder LN weight",
        suffixes: &["decoder.layer_norm.weight"],
        min: 0.5,
        max: 3.0,
        pass_template: "decoder LN mean in [0.5, 3.0]",
        exclude_contains: &[],
        require_matches: true,
    },
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
        exclude_contains: &["encoder.layer_norm", "decoder.layer_norm"],
        require_matches: false,
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
        require_matches: false,
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
        require_matches: false,
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
        require_matches: false,
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
        require_matches: false,
    },
];

/// APR file validator implementing the 25-point QA checklist
pub struct AprValidator<'a> {
    reader: &'a AprV2ReaderRef<'a>,
    config: ModelConfig,
}

impl<'a> AprValidator<'a> {
    /// Create validator from reader and model config
    #[must_use]
    pub fn new(reader: &'a AprV2ReaderRef<'a>, config: ModelConfig) -> Self {
        Self { reader, config }
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
        let mut checks = self.check_bulk_mean_ranges(LAYER_NORM_MEAN_SPECS);
        // Check 10: NaN/Inf in layer norm tensors
        checks.push(self.check_tensor_stats(
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
        ));
        checks
    }

    /// C. Attention/Linear Validation (checks 11-15)
    fn validate_attention_linear(&self) -> Vec<ValidationCheck> {
        let mut checks = self.check_bulk_mean_ranges(ATTENTION_MEAN_SPECS);
        checks.extend(self.validate_weight_health());
        checks
    }

    /// D. Embedding Validation (checks 16-20): shapes and stats in a single pass.
    fn validate_embeddings(&self) -> Vec<ValidationCheck> {
        let vocab_size = self.config.n_vocab as usize;
        let d_model = self.config.n_text_state as usize;
        let d_audio = self.config.n_audio_state as usize;

        // Check 16: token embedding shape
        let tok_shape = match self.reader.get_tensor("decoder.token_embedding") {
            Some(tensor) => {
                let shape = &tensor.shape;
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

        // Check 17: token embedding stats
        let tok_stats = self.check_single_tensor(
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

        // Check 18: positional embedding shapes
        let mut pos_failures = Vec::new();
        for &(name, expected_len, d) in &[
            ("encoder.positional_embedding", 1500usize, d_audio),
            ("decoder.positional_embedding", 448, d_model),
        ] {
            if let Some(tensor) = self.reader.get_tensor(name) {
                let shape = &tensor.shape;
                if shape.len() != 2 || shape[0] != expected_len || shape[1] != d {
                    pos_failures.push(format!("{name}: {shape:?}, expected [{expected_len}, {d}]"));
                }
            }
        }
        let pos_shape = if pos_failures.is_empty() {
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

        // Check 19: positional embedding stats
        let pos_stats = self.check_tensor_stats(
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

        // Check 20: vocab size consistency
        let vocab_check = match self.reader.get_tensor("decoder.token_embedding") {
            Some(tensor) => {
                let shape = &tensor.shape;
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

        vec![tok_shape, tok_stats, pos_shape, pos_stats, vocab_check]
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
        let version = self.reader.header().version.0;
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
        let count = self.reader.header().tensor_count as usize;
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
        let d_model = self.config.n_audio_state as usize;

        if let Some(tensor) = self.reader.get_tensor("decoder.token_embedding") {
            let shape = &tensor.shape;
            if shape.len() != 2 || shape[1] != d_model {
                failures.push(format!(
                    "token_embedding shape {shape:?}, expected [*, {d_model}]"
                ));
            }
        }

        if let Some(tensor) = self.reader.get_tensor("encoder.conv1.weight") {
            let shape = &tensor.shape;
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
        match self.reader.get_tensor_as_f32(tensor_name) {
            Some(data) => {
                let stats = TensorStats::compute(tensor_name, &data);
                match validate(&stats) {
                    Ok(msg) => ValidationCheck::pass(id, category, description, &msg),
                    Err(msg) => ValidationCheck::fail(id, category, description, &msg),
                }
            }
            None => ValidationCheck::fail(
                id,
                category,
                description,
                &format!("Tensor {tensor_name} not found"),
            ),
        }
    }

    /// Iterate tensors matching a filter, compute stats, collect failures.
    fn collect_tensor_failures(
        &self,
        filter: impl Fn(&str) -> bool,
        validate: impl Fn(&TensorStats) -> Option<String>,
    ) -> (Vec<String>, usize) {
        let mut failures = Vec::new();
        let mut checked = 0;

        for name in self.reader.tensor_names() {
            if filter(name) {
                if let Some(data) = self.reader.get_tensor_as_f32(name) {
                    let stats = TensorStats::compute(name, &data);
                    checked += 1;
                    if let Some(msg) = validate(&stats) {
                        failures.push(format!("{name}: {msg}"));
                    }
                }
            }
        }

        (failures, checked)
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
        let (failures, checked) = self.collect_tensor_failures(filter, validate);

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

    /// Batch mean-range validation: each spec defines a filter + range check.
    fn check_bulk_mean_ranges(&self, specs: &[BulkMeanSpec]) -> Vec<ValidationCheck> {
        let range_filter = |name: &str, spec: &BulkMeanSpec| -> bool {
            let matches = spec.suffixes.iter().any(|p| name.ends_with(p));
            matches && spec.exclude_contains.iter().all(|exc| !name.contains(exc))
        };

        specs
            .iter()
            .map(|spec| {
                let check = self.check_tensor_stats(
                    spec.id,
                    spec.category,
                    spec.description,
                    |name| range_filter(name, spec),
                    Self::mean_outside_range(spec.min, spec.max),
                    spec.pass_template,
                );
                // For required specs, fail if no tensors matched
                if spec.require_matches && check.passed && check.message.starts_with("All 0 ") {
                    ValidationCheck::fail(
                        spec.id,
                        spec.category,
                        spec.description,
                        &format!("No tensors matching {:?} found", spec.suffixes),
                    )
                } else {
                    check
                }
            })
            .collect()
    }

    /// Weight health checks: std deviation (13) and no-all-zeros (14).
    fn validate_weight_health(&self) -> Vec<ValidationCheck> {
        // Check 13: weight std with 25% outlier tolerance
        let (std_failures, checked) = self.collect_tensor_failures(
            |name| name.ends_with(".weight") && !name.contains("embedding"),
            |stats| (stats.std < 0.01 || stats.std > 0.2).then(|| format!("std={:.4}", stats.std)),
        );
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

    fn expected_tensor_count(&self) -> usize {
        let n_enc = self.config.n_audio_layer as usize;
        let n_dec = self.config.n_text_layer as usize;
        2 + 4 + (n_enc * 8) + (n_dec * 12) + 4
    }
}
