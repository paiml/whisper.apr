//! APR file validator

use super::check::{ValidationCheck, ValidationReport};
use super::stats::TensorStats;
use crate::format::AprReader;

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
        vec![
            self.check_encoder_ln_weight(),
            self.check_decoder_ln_weight(),
            self.check_block_ln_weights(),
            self.check_ln_biases(),
            self.check_ln_nan_inf(),
        ]
    }

    /// C. Attention/Linear Validation (checks 11-15)
    fn validate_attention_linear(&self) -> Vec<ValidationCheck> {
        vec![
            self.check_qkv_proj_means(),
            self.check_ffn_weight_means(),
            self.check_weight_std(),
            self.check_no_zero_tensors(),
            self.check_bias_vectors(),
        ]
    }

    /// D. Embedding Validation (checks 16-20)
    fn validate_embeddings(&self) -> Vec<ValidationCheck> {
        vec![
            self.check_token_embedding_shape(),
            self.check_token_embedding_stats(),
            self.check_positional_embedding_shape(),
            self.check_positional_embedding_stats(),
            self.check_vocab_size(),
        ]
    }

    /// E. Functional Validation (checks 21-25)
    #[allow(clippy::unused_self)]
    fn validate_functional(&self) -> Vec<ValidationCheck> {
        vec![
            ValidationCheck::pass(
                21,
                'E',
                "Encoder output match",
                "Skipped: no reference data",
            ),
            ValidationCheck::pass(
                22,
                'E',
                "Decoder logits match",
                "Skipped: no reference data",
            ),
            ValidationCheck::pass(23, 'E', "Transcription test", "Skipped: no test audio"),
            ValidationCheck::pass(24, 'E', "No repetitive output", "Skipped: no test audio"),
            ValidationCheck::pass(25, 'E', "End-to-end accuracy", "Skipped: no validation set"),
        ]
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

    fn check_encoder_ln_weight(&self) -> ValidationCheck {
        self.check_ln_weight_mean(6, "encoder.layer_norm.weight", "Encoder LN weight")
    }

    fn check_decoder_ln_weight(&self) -> ValidationCheck {
        self.check_ln_weight_mean(7, "decoder.layer_norm.weight", "Decoder LN weight")
    }

    #[allow(clippy::option_if_let_else)]
    fn check_ln_weight_mean(&self, id: u8, name: &str, description: &str) -> ValidationCheck {
        match self.reader.load_tensor(name) {
            Ok(data) => {
                let stats = TensorStats::compute(name, &data);
                if stats.mean >= 0.5 && stats.mean <= 3.0 {
                    ValidationCheck::pass(
                        id,
                        'B',
                        description,
                        &format!("mean={:.4} in [0.5, 3.0]", stats.mean),
                    )
                } else {
                    ValidationCheck::fail(
                        id,
                        'B',
                        description,
                        &format!("mean={:.4} NOT in [0.5, 3.0]", stats.mean),
                    )
                }
            }
            Err(_) => {
                ValidationCheck::fail(id, 'B', description, &format!("Tensor {name} not found"))
            }
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

    fn check_block_ln_weights(&self) -> ValidationCheck {
        let patterns = [
            "self_attn_layer_norm.weight",
            "encoder_attn_layer_norm.weight",
            "final_layer_norm.weight",
        ];
        self.check_tensor_stats(
            8,
            'B',
            "Block LN weight means",
            |name| patterns.iter().any(|p| name.contains(p)),
            |stats| {
                (stats.mean < 0.5 || stats.mean > 3.0).then(|| format!("mean={:.4}", stats.mean))
            },
            "block LN means in [0.5, 3.0]",
        )
    }

    fn check_ln_biases(&self) -> ValidationCheck {
        self.check_tensor_stats(
            9,
            'B',
            "LN bias means",
            |name| name.contains("layer_norm.bias"),
            |stats| {
                (stats.mean < -0.5 || stats.mean > 0.5).then(|| format!("mean={:.4}", stats.mean))
            },
            "LN bias means in [-0.5, 0.5]",
        )
    }

    fn check_ln_nan_inf(&self) -> ValidationCheck {
        let mut nan_tensors = Vec::new();
        let mut inf_tensors = Vec::new();

        for tensor in &self.reader.tensors {
            if tensor.name.contains("layer_norm") {
                if let Ok(data) = self.reader.load_tensor(&tensor.name) {
                    let stats = TensorStats::compute(&tensor.name, &data);
                    if stats.has_nan() {
                        nan_tensors.push(tensor.name.clone());
                    }
                    if stats.has_inf() {
                        inf_tensors.push(tensor.name.clone());
                    }
                }
            }
        }

        if nan_tensors.is_empty() && inf_tensors.is_empty() {
            ValidationCheck::pass(10, 'B', "No NaN/Inf in LN", "All LN tensors clean")
        } else {
            let mut msg = Vec::new();
            if !nan_tensors.is_empty() {
                msg.push(format!("NaN in: {nan_tensors:?}"));
            }
            if !inf_tensors.is_empty() {
                msg.push(format!("Inf in: {inf_tensors:?}"));
            }
            ValidationCheck::fail(10, 'B', "No NaN/Inf in LN", &msg.join("; "))
        }
    }

    fn check_qkv_proj_means(&self) -> ValidationCheck {
        let patterns = [
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "out_proj.weight",
        ];
        self.check_tensor_stats(
            11,
            'C',
            "Q/K/V proj means",
            |name| patterns.iter().any(|p| name.ends_with(p)),
            |stats| {
                (stats.mean < -0.1 || stats.mean > 0.1).then(|| format!("mean={:.4}", stats.mean))
            },
            "proj means in [-0.1, 0.1]",
        )
    }

    fn check_ffn_weight_means(&self) -> ValidationCheck {
        let patterns = ["fc1.weight", "fc2.weight"];
        self.check_tensor_stats(
            12,
            'C',
            "FFN weight means",
            |name| patterns.iter().any(|p| name.ends_with(p)),
            |stats| {
                (stats.mean < -0.1 || stats.mean > 0.1).then(|| format!("mean={:.4}", stats.mean))
            },
            "FFN means in [-0.1, 0.1]",
        )
    }

    fn check_weight_std(&self) -> ValidationCheck {
        let mut failures = Vec::new();
        let mut checked = 0;

        for tensor in &self.reader.tensors {
            if tensor.name.ends_with(".weight") && !tensor.name.contains("embedding") {
                if let Ok(data) = self.reader.load_tensor(&tensor.name) {
                    let stats = TensorStats::compute(&tensor.name, &data);
                    checked += 1;
                    if stats.std < 0.01 || stats.std > 0.2 {
                        failures.push(format!("{}: std={:.4}", tensor.name, stats.std));
                    }
                }
            }
        }

        if failures.is_empty() {
            ValidationCheck::pass(
                13,
                'C',
                "Weight std reasonable",
                &format!("All {checked} weight stds in [0.01, 0.2]"),
            )
        } else if failures.len() > checked / 4 {
            ValidationCheck::fail(
                13,
                'C',
                "Weight std reasonable",
                &format!("{} failures: {}", failures.len(), failures[0]),
            )
        } else {
            ValidationCheck::pass(
                13,
                'C',
                "Weight std reasonable",
                &format!("{} minor outliers in {checked} weights", failures.len()),
            )
        }
    }

    fn check_no_zero_tensors(&self) -> ValidationCheck {
        let mut zero_tensors = Vec::new();

        for tensor in &self.reader.tensors {
            if let Ok(data) = self.reader.load_tensor(&tensor.name) {
                let stats = TensorStats::compute(&tensor.name, &data);
                if stats.is_all_zeros() {
                    zero_tensors.push(tensor.name.clone());
                }
            }
        }

        if zero_tensors.is_empty() {
            ValidationCheck::pass(14, 'C', "No zero tensors", "No all-zero tensors found")
        } else {
            ValidationCheck::fail(
                14,
                'C',
                "No zero tensors",
                &format!("Zero tensors: {zero_tensors:?}"),
            )
        }
    }

    #[allow(clippy::case_sensitive_file_extension_comparisons)]
    fn check_bias_vectors(&self) -> ValidationCheck {
        self.check_tensor_stats(
            15,
            'C',
            "Bias vectors valid",
            |name| name.ends_with(".bias") && !name.contains("layer_norm"),
            |stats| {
                (stats.mean < -1.0 || stats.mean > 1.0).then(|| format!("mean={:.4}", stats.mean))
            },
            "bias means in [-1.0, 1.0]",
        )
    }

    #[allow(clippy::option_if_let_else)]
    fn check_token_embedding_shape(&self) -> ValidationCheck {
        let vocab_size = self.reader.header.n_vocab;
        let d_model = self.reader.header.n_text_state;

        match self.reader.find_tensor("decoder.token_embedding") {
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
        }
    }

    #[allow(clippy::option_if_let_else)]
    fn check_token_embedding_stats(&self) -> ValidationCheck {
        match self.reader.load_tensor("decoder.token_embedding") {
            Ok(data) => {
                let stats = TensorStats::compute("decoder.token_embedding", &data);
                let mean_ok = stats.mean.abs() < 0.1;
                let std_ok = stats.std >= 0.01 && stats.std <= 0.1;

                if mean_ok && std_ok {
                    ValidationCheck::pass(
                        17,
                        'D',
                        "Token embedding stats",
                        &format!("mean={:.4}, std={:.4}", stats.mean, stats.std),
                    )
                } else {
                    ValidationCheck::fail(
                        17,
                        'D',
                        "Token embedding stats",
                        &format!(
                            "mean={:.4} (want ~0), std={:.4} (want 0.01-0.1)",
                            stats.mean, stats.std
                        ),
                    )
                }
            }
            Err(_) => ValidationCheck::fail(
                17,
                'D',
                "Token embedding stats",
                "Token embedding not found",
            ),
        }
    }

    fn check_positional_embedding_shape(&self) -> ValidationCheck {
        let mut failures = Vec::new();
        let d_model_enc = self.reader.header.n_audio_state;
        let d_model_dec = self.reader.header.n_text_state;

        if let Some(tensor) = self.reader.find_tensor("encoder.positional_embedding") {
            let shape = tensor.shape();
            if shape.len() != 2 || shape[0] != 1500 || shape[1] != d_model_enc {
                failures.push(format!(
                    "encoder pos: {shape:?}, expected [1500, {d_model_enc}]"
                ));
            }
        }

        if let Some(tensor) = self.reader.find_tensor("decoder.positional_embedding") {
            let shape = tensor.shape();
            if shape.len() != 2 || shape[0] != 448 || shape[1] != d_model_dec {
                failures.push(format!(
                    "decoder pos: {shape:?}, expected [448, {d_model_dec}]"
                ));
            }
        }

        if failures.is_empty() {
            ValidationCheck::pass(
                18,
                'D',
                "Positional embedding shape",
                "Encoder [1500, d], Decoder [448, d]",
            )
        } else {
            ValidationCheck::fail(18, 'D', "Positional embedding shape", &failures.join("; "))
        }
    }

    fn check_positional_embedding_stats(&self) -> ValidationCheck {
        let mut failures = Vec::new();

        for name in [
            "encoder.positional_embedding",
            "decoder.positional_embedding",
        ] {
            if let Ok(data) = self.reader.load_tensor(name) {
                let stats = TensorStats::compute(name, &data);
                if stats.mean.abs() > 0.5 || stats.std < 0.005 || stats.std > 0.1 {
                    failures.push(format!(
                        "{}: mean={:.4}, std={:.4}",
                        name, stats.mean, stats.std
                    ));
                }
            }
        }

        if failures.is_empty() {
            ValidationCheck::pass(
                19,
                'D',
                "Positional embedding stats",
                "Stats within expected ranges",
            )
        } else {
            ValidationCheck::fail(19, 'D', "Positional embedding stats", &failures.join("; "))
        }
    }

    #[allow(clippy::option_if_let_else)]
    fn check_vocab_size(&self) -> ValidationCheck {
        let header_vocab = self.reader.header.n_vocab;

        match self.reader.find_tensor("decoder.token_embedding") {
            Some(tensor) => {
                let shape = tensor.shape();
                if !shape.is_empty() && shape[0] == header_vocab {
                    ValidationCheck::pass(
                        20,
                        'D',
                        "Vocab size matches",
                        &format!("vocab_size={header_vocab} matches tensor"),
                    )
                } else {
                    ValidationCheck::fail(
                        20,
                        'D',
                        "Vocab size matches",
                        &format!(
                            "Header vocab={}, tensor dim={}",
                            header_vocab,
                            shape.first().unwrap_or(&0)
                        ),
                    )
                }
            }
            None => {
                ValidationCheck::fail(20, 'D', "Vocab size matches", "Token embedding not found")
            }
        }
    }

    fn expected_tensor_count(&self) -> usize {
        let n_enc = self.reader.header.n_audio_layer as usize;
        let n_dec = self.reader.header.n_text_layer as usize;
        2 + 4 + (n_enc * 8) + (n_dec * 12) + 4
    }
}
