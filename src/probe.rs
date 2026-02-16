//! Activation probing infrastructure for forward-pass debugging
//!
//! Provides [`ActivationProbe`] for recording activation statistics at named
//! checkpoints throughout the model's forward pass, enabling numerical parity
//! comparison against reference implementations (HuggingFace, whisper.cpp).
//!
//! # Checkpoint Naming Convention
//!
//! ```text
//! conv_stem.conv1_out
//! conv_stem.groupnorm_out
//! conv_stem.conv2_out
//! conv_stem.conv3_out
//! conv_stem.gelu3_out
//! encoder.block_{i}.ln1_out
//! encoder.block_{i}.self_attn_out
//! encoder.block_{i}.residual_1
//! encoder.block_{i}.ln2_out
//! encoder.block_{i}.ffn_out
//! encoder.block_{i}.residual_2
//! encoder.ln_post_out
//! decoder.token_emb
//! decoder.block_{i}.ln1_out
//! decoder.block_{i}.self_attn_out
//! decoder.block_{i}.residual_1
//! decoder.block_{i}.ln_cross_out
//! decoder.block_{i}.cross_attn_out
//! decoder.block_{i}.residual_2
//! decoder.block_{i}.ln2_out
//! decoder.block_{i}.ffn_out
//! decoder.block_{i}.residual_3
//! decoder.ln_post_out
//! decoder.logits
//! ```

#[cfg(feature = "cli")]
use serde::{Deserialize, Serialize};

/// Statistical snapshot of an activation tensor at a named checkpoint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "cli", derive(Serialize, Deserialize))]
pub struct ActivationSnapshot {
    /// Checkpoint name (e.g. "encoder.block_0.self_attn_out")
    pub name: String,
    /// Tensor shape (e.g. [seq_len, d_model])
    pub shape: Vec<usize>,
    /// L2 norm of the full tensor
    pub l2: f32,
    /// Arithmetic mean
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// First N values for visual comparison
    pub first_n: Vec<f32>,
    /// Full tensor data (only captured when `capture_full` is set)
    #[cfg_attr(feature = "cli", serde(skip_serializing_if = "Option::is_none"))]
    pub full_data: Option<Vec<f32>>,
}

/// Activation probe that records snapshots during a forward pass
#[derive(Debug, Clone, Default)]
pub struct ActivationProbe {
    /// Collected activation snapshots
    pub snapshots: Vec<ActivationSnapshot>,
    /// Whether to capture full tensor data
    pub capture_full: bool,
    /// Number of leading values to capture for visual comparison
    pub first_n: usize,
    /// Optional prefix filter (e.g. "conv_stem", "encoder.block_0")
    pub stage_filter: Option<String>,
}

impl ActivationProbe {
    /// Create a new probe with default settings (first 8 values, no full capture)
    #[must_use]
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            capture_full: false,
            first_n: 8,
            stage_filter: None,
        }
    }

    /// Set whether to capture full tensor data
    #[must_use]
    pub fn with_full_capture(mut self, capture: bool) -> Self {
        self.capture_full = capture;
        self
    }

    /// Set number of leading values to capture
    #[must_use]
    pub fn with_first_n(mut self, n: usize) -> Self {
        self.first_n = n;
        self
    }

    /// Set stage prefix filter
    #[must_use]
    pub fn with_stage_filter(mut self, filter: String) -> Self {
        self.stage_filter = Some(filter);
        self
    }

    /// Record an activation tensor at the given checkpoint
    ///
    /// Computes L2 norm, mean, std_dev, min, max, captures first_n values,
    /// and optionally stores the full tensor. Skips recording if `stage_filter`
    /// is set and the checkpoint name doesn't start with the filter prefix.
    pub fn record(&mut self, name: &str, data: &[f32], shape: &[usize]) {
        // Apply stage filter
        if let Some(ref filter) = self.stage_filter {
            if !name.starts_with(filter) {
                return;
            }
        }

        let n = data.len();
        if n == 0 {
            self.snapshots.push(ActivationSnapshot {
                name: name.to_string(),
                shape: shape.to_vec(),
                l2: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                first_n: Vec::new(),
                full_data: None,
            });
            return;
        }

        let sum: f32 = data.iter().sum();
        let mean = sum / n as f32;

        let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let std_dev = variance.sqrt();

        let l2: f32 = data.iter().map(|&x| x * x).sum::<f32>().sqrt();

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &x in data {
            if x < min_val {
                min_val = x;
            }
            if x > max_val {
                max_val = x;
            }
        }

        let first_n_vals: Vec<f32> = data.iter().take(self.first_n).copied().collect();

        let full_data = if self.capture_full {
            Some(data.to_vec())
        } else {
            None
        };

        self.snapshots.push(ActivationSnapshot {
            name: name.to_string(),
            shape: shape.to_vec(),
            l2,
            mean,
            std_dev,
            min: min_val,
            max: max_val,
            first_n: first_n_vals,
            full_data,
        });
    }
}

/// Serializable output container for probe results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "cli", derive(Serialize, Deserialize))]
pub struct ProbeOutput {
    /// Model identifier (e.g. "moonshine-tiny.apr")
    pub model: String,
    /// Audio file path
    pub audio: String,
    /// Model family (e.g. "moonshine", "whisper")
    pub model_family: String,
    /// Activation checkpoints in forward-pass order
    pub checkpoints: Vec<ActivationSnapshot>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_new() {
        let probe = ActivationProbe::new();
        assert!(probe.snapshots.is_empty());
        assert!(!probe.capture_full);
        assert_eq!(probe.first_n, 8);
        assert!(probe.stage_filter.is_none());
    }

    #[test]
    fn test_probe_record_basic() {
        let mut probe = ActivationProbe::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        probe.record("test.checkpoint", &data, &[2, 2]);

        assert_eq!(probe.snapshots.len(), 1);
        let snap = &probe.snapshots[0];
        assert_eq!(snap.name, "test.checkpoint");
        assert_eq!(snap.shape, vec![2, 2]);
        assert!((snap.mean - 2.5).abs() < 1e-5);
        assert!((snap.min - 1.0).abs() < 1e-5);
        assert!((snap.max - 4.0).abs() < 1e-5);
        assert!(snap.l2 > 0.0);
        assert!(snap.std_dev > 0.0);
        assert_eq!(snap.first_n.len(), 4); // all 4 values (< first_n=8)
        assert!(snap.full_data.is_none());
    }

    #[test]
    fn test_probe_record_full_capture() {
        let mut probe = ActivationProbe::new().with_full_capture(true);
        let data = vec![1.0, 2.0, 3.0];
        probe.record("test", &data, &[3]);

        assert!(probe.snapshots[0].full_data.is_some());
        assert_eq!(
            probe.snapshots[0].full_data.as_ref().map(|d| d.len()),
            Some(3)
        );
    }

    #[test]
    fn test_probe_record_first_n() {
        let mut probe = ActivationProbe::new().with_first_n(2);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        probe.record("test", &data, &[5]);

        assert_eq!(probe.snapshots[0].first_n.len(), 2);
        assert!((probe.snapshots[0].first_n[0] - 1.0).abs() < 1e-5);
        assert!((probe.snapshots[0].first_n[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_probe_stage_filter() {
        let mut probe = ActivationProbe::new().with_stage_filter("encoder".to_string());
        probe.record("conv_stem.conv1_out", &[1.0, 2.0], &[2]);
        probe.record("encoder.block_0.ln1_out", &[3.0, 4.0], &[2]);
        probe.record("decoder.token_emb", &[5.0, 6.0], &[2]);

        assert_eq!(probe.snapshots.len(), 1);
        assert_eq!(probe.snapshots[0].name, "encoder.block_0.ln1_out");
    }

    #[test]
    fn test_probe_record_empty() {
        let mut probe = ActivationProbe::new();
        probe.record("empty", &[], &[0]);

        assert_eq!(probe.snapshots.len(), 1);
        assert!((probe.snapshots[0].l2).abs() < 1e-5);
        assert!((probe.snapshots[0].mean).abs() < 1e-5);
    }

    #[test]
    fn test_probe_l2_norm() {
        let mut probe = ActivationProbe::new();
        // L2 of [3, 4] = 5
        probe.record("test", &[3.0, 4.0], &[2]);
        assert!((probe.snapshots[0].l2 - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_probe_std_dev() {
        let mut probe = ActivationProbe::new();
        // std_dev of [1, 1, 1, 1] = 0
        probe.record("const", &[1.0, 1.0, 1.0, 1.0], &[4]);
        assert!((probe.snapshots[0].std_dev).abs() < 1e-5);
    }

    #[test]
    fn test_probe_output_structure() {
        let output = ProbeOutput {
            model: "moonshine-tiny.apr".to_string(),
            audio: "test.wav".to_string(),
            model_family: "moonshine".to_string(),
            checkpoints: vec![],
        };
        assert_eq!(output.model, "moonshine-tiny.apr");
        assert!(output.checkpoints.is_empty());
    }

    #[test]
    fn test_probe_multiple_records() {
        let mut probe = ActivationProbe::new();
        for i in 0..5 {
            probe.record(&format!("layer_{i}"), &[i as f32; 10], &[10]);
        }
        assert_eq!(probe.snapshots.len(), 5);
        assert_eq!(probe.snapshots[3].name, "layer_3");
    }

    #[test]
    fn test_probe_builder_chain() {
        let probe = ActivationProbe::new()
            .with_full_capture(true)
            .with_first_n(16)
            .with_stage_filter("decoder".to_string());

        assert!(probe.capture_full);
        assert_eq!(probe.first_n, 16);
        assert_eq!(probe.stage_filter.as_deref(), Some("decoder"));
    }
}
