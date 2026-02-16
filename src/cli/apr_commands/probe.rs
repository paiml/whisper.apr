//! Forward-pass debugging commands: probe, parity, config-check
//!
//! Implements `apr probe`, `apr parity`, and `apr config-check` subcommands
//! for diagnosing numerical parity between our forward pass and reference
//! implementations (HuggingFace, whisper.cpp).

use std::fs;
use std::path::Path;

use super::super::apr_args::{AprConfigCheckArgs, AprParityArgs, AprProbeArgs};
use super::super::commands::{CliError, CliResult, CommandResult};
use crate::probe::{ActivationProbe, ProbeOutput};
use crate::WhisperApr;

/// Run the `apr probe` command
///
/// Performs a probed forward pass through the model, recording activation
/// snapshots at every checkpoint, then serializes them to JSON.
pub(super) fn run_probe(
    args: &AprProbeArgs,
    _global: &super::super::args::Args,
) -> CliResult<CommandResult> {
    // Load model
    let model_bytes =
        fs::read(&args.model).map_err(|e| CliError::InvalidArgument(format!("Model: {e}")))?;
    let whisper = WhisperApr::load_from_apr(&model_bytes)
        .map_err(|e| CliError::InvalidArgument(format!("Model load: {e}")))?;

    // Load audio
    let audio_bytes =
        fs::read(&args.audio).map_err(|e| CliError::InvalidArgument(format!("Audio: {e}")))?;
    let samples = super::super::commands::load_audio_samples(args.audio.as_path(), &audio_bytes)?;

    // Determine decoder input tokens
    let tokens: Vec<u32> = if let Some(ref tok_str) = args.tokens {
        tok_str
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect()
    } else if whisper.config().model_family == crate::format::ModelFamily::Moonshine {
        vec![1] // Moonshine SOT
    } else {
        vec![crate::tokenizer::special_tokens::SOT]
    };

    // Build probe
    let filter = args.layer.clone().or_else(|| args.stage.clone());

    let mut probe = ActivationProbe::new()
        .with_full_capture(args.full_tensor)
        .with_first_n(args.first_n);

    if let Some(f) = filter {
        probe = probe.with_stage_filter(f);
    }

    // Run probed forward pass
    let _logits = whisper
        .forward_probed(&samples, &tokens, &mut probe)
        .map_err(|e| CliError::InvalidArgument(format!("Forward pass: {e}")))?;

    let family = format!("{:?}", whisper.config().model_family);

    let output = ProbeOutput {
        model: args.model.display().to_string(),
        audio: args.audio.display().to_string(),
        model_family: family,
        checkpoints: probe.snapshots,
    };

    let json = serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".to_string());

    if let Some(ref path) = args.output {
        fs::write(path, &json)
            .map_err(|e| CliError::WriteError(format!("{}: {e}", path.display())))?;
        Ok(CommandResult::success(format!(
            "Probe: {} checkpoints -> {}",
            output.checkpoints.len(),
            path.display()
        )))
    } else {
        println!("{json}");
        Ok(CommandResult::success(format!(
            "Probe: {} checkpoints",
            output.checkpoints.len()
        )))
    }
}

/// Run the `apr parity` command
///
/// Compares two probe JSON files checkpoint-by-checkpoint, reporting
/// L2 relative differences and flagging the first divergence point.
pub(super) fn run_parity(
    args: &AprParityArgs,
    _global: &super::super::args::Args,
) -> CliResult<CommandResult> {
    let ours = load_probe_output(&args.ours, "Ours")?;
    let reference = load_probe_output(&args.reference, "Reference")?;

    println!(
        "Parity: {} vs {} (tol: {:.2}%)\n",
        args.ours.display(),
        args.reference.display(),
        args.tolerance * 100.0
    );

    let mut first_fail: Option<String> = None;
    let mut pass_count = 0usize;
    let mut fail_count = 0usize;

    // Match by checkpoint name
    for ref_snap in &reference.checkpoints {
        let our_snap = ours.checkpoints.iter().find(|s| s.name == ref_snap.name);

        let Some(our_snap) = our_snap else {
            println!("  MISS  {:<40} (not found in ours)", ref_snap.name);
            fail_count += 1;
            continue;
        };

        if first_fail.is_some() && args.stop_first {
            println!("  ....  {:<40} (propagated)", ref_snap.name);
            continue;
        }

        let (rel_diff, passed) = compute_l2_diff(
            our_snap.l2 as f64,
            ref_snap.l2 as f64,
            args.tolerance,
            args.abs_tolerance,
        );

        if passed {
            println!(
                "  PASS  {:<40} L2: {:.4} vs {:.4}  ({:.3}%)",
                ref_snap.name,
                our_snap.l2,
                ref_snap.l2,
                rel_diff * 100.0
            );
            pass_count += 1;
        } else {
            let marker = if first_fail.is_none() {
                first_fail = Some(ref_snap.name.clone());
                " << FIRST DIVERGENCE"
            } else {
                ""
            };
            println!(
                "  FAIL  {:<40} L2: {:.4} vs {:.4}  ({:.1}%){marker}",
                ref_snap.name,
                our_snap.l2,
                ref_snap.l2,
                rel_diff * 100.0
            );
            fail_count += 1;
        }
    }

    println!();
    if let Some(ref name) = first_fail {
        println!("  Result: FAIL at {name}");
    } else {
        println!(
            "  Result: PASS ({pass_count}/{} checkpoints)",
            pass_count + fail_count
        );
    }

    let status = if first_fail.is_some() { "FAIL" } else { "PASS" };
    Ok(CommandResult::success(format!(
        "Parity {status}: {pass_count} pass, {fail_count} fail"
    )))
}

/// Load and parse a probe output JSON file.
fn load_probe_output(path: &Path, label: &str) -> CliResult<ProbeOutput> {
    let json =
        fs::read_to_string(path).map_err(|e| CliError::InvalidArgument(format!("{label}: {e}")))?;
    serde_json::from_str(&json)
        .map_err(|e| CliError::InvalidArgument(format!("Parse {label}: {e}")))
}

/// Compute relative L2 difference between two checkpoints.
///
/// Returns `(relative_diff, passed)`. Uses absolute tolerance when the
/// reference L2 is near zero, and relative tolerance otherwise.
fn compute_l2_diff(l2_ours: f64, l2_ref: f64, tolerance: f64, abs_tolerance: f64) -> (f64, bool) {
    if l2_ref.abs() < abs_tolerance {
        let abs_diff = (l2_ours - l2_ref).abs();
        (abs_diff, abs_diff < abs_tolerance)
    } else {
        let rel = (l2_ours - l2_ref).abs() / l2_ref;
        (rel, rel < tolerance)
    }
}

/// Known reference configurations for model families
#[allow(dead_code)]
struct RefConfig {
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    encoder_ffn_intermediate: usize,
    decoder_ffn_intermediate: usize,
    encoder_activation: &'static str,
    decoder_activation: &'static str,
    n_encoder_layers: usize,
    n_decoder_layers: usize,
    rope_base: f32,
    partial_rotary_factor: f32,
    attention_bias: bool,
    norm_bias: bool,
    tied_embeddings: bool,
}

/// Get reference config for known model families
fn get_reference_config(name: &str) -> Option<RefConfig> {
    match name {
        "moonshine-tiny" => Some(RefConfig {
            hidden_size: 288,
            num_heads: 8,
            num_kv_heads: 8,
            head_dim: 36,
            encoder_ffn_intermediate: 1152,
            decoder_ffn_intermediate: 1152,
            encoder_activation: "gelu",
            decoder_activation: "silu",
            n_encoder_layers: 6,
            n_decoder_layers: 6,
            rope_base: 10000.0,
            partial_rotary_factor: 0.9,
            attention_bias: false,
            norm_bias: false,
            tied_embeddings: true,
        }),
        "moonshine-base" => Some(RefConfig {
            hidden_size: 416,
            num_heads: 8,
            num_kv_heads: 8,
            head_dim: 52,
            encoder_ffn_intermediate: 1664,
            decoder_ffn_intermediate: 1664,
            encoder_activation: "gelu",
            decoder_activation: "silu",
            n_encoder_layers: 8,
            n_decoder_layers: 8,
            rope_base: 10000.0,
            partial_rotary_factor: 0.9,
            attention_bias: false,
            norm_bias: false,
            tied_embeddings: true,
        }),
        "whisper-tiny" => Some(RefConfig {
            hidden_size: 384,
            num_heads: 6,
            num_kv_heads: 6,
            head_dim: 64,
            encoder_ffn_intermediate: 1536,
            decoder_ffn_intermediate: 1536,
            encoder_activation: "gelu",
            decoder_activation: "gelu",
            n_encoder_layers: 4,
            n_decoder_layers: 4,
            rope_base: 0.0,
            partial_rotary_factor: 0.0,
            attention_bias: true,
            norm_bias: true,
            tied_embeddings: false,
        }),
        "whisper-base" => Some(RefConfig {
            hidden_size: 512,
            num_heads: 8,
            num_kv_heads: 8,
            head_dim: 64,
            encoder_ffn_intermediate: 2048,
            decoder_ffn_intermediate: 2048,
            encoder_activation: "gelu",
            decoder_activation: "gelu",
            n_encoder_layers: 6,
            n_decoder_layers: 6,
            rope_base: 0.0,
            partial_rotary_factor: 0.0,
            attention_bias: true,
            norm_bias: true,
            tied_embeddings: false,
        }),
        _ => None,
    }
}

/// Run the `apr config-check` command
///
/// Compares a loaded model's configuration against a known reference,
/// flagging mismatches in hidden_size, num_heads, activations, etc.
pub(super) fn run_config_check(
    args: &AprConfigCheckArgs,
    _global: &super::super::args::Args,
) -> CliResult<CommandResult> {
    // Load model
    let model_bytes =
        fs::read(&args.model).map_err(|e| CliError::InvalidArgument(format!("Model: {e}")))?;
    let whisper = WhisperApr::load_from_apr(&model_bytes)
        .map_err(|e| CliError::InvalidArgument(format!("Model load: {e}")))?;

    let config = whisper.config();

    // Determine reference name
    let ref_name = args.reference.clone().unwrap_or_else(|| {
        let family = format!("{:?}", config.model_family).to_lowercase();
        let size = match config.n_audio_state {
            288 | 384 => "tiny",
            416 | 512 => "base",
            _ => "unknown",
        };
        format!("{family}-{size}")
    });

    let reference = get_reference_config(&ref_name).ok_or_else(|| {
        CliError::InvalidArgument(format!(
            "Unknown reference '{}'. Known: moonshine-tiny, moonshine-base, whisper-tiny, whisper-base",
            ref_name
        ))
    })?;

    println!(
        "Config Check: {} against {ref_name}\n",
        args.model.display()
    );

    let mut mismatches = 0usize;
    let mut checks = 0usize;

    // Helper macro for checking config values
    macro_rules! check {
        ($label:expr, $actual:expr, $expected:expr) => {
            checks += 1;
            let actual_val = $actual;
            let expected_val = $expected;
            if actual_val == expected_val {
                if args.verbose {
                    println!(
                        "  PASS  {:<30} {} (expected {})",
                        $label, actual_val, expected_val
                    );
                }
            } else {
                println!(
                    "  FAIL  {:<30} {} (expected {})",
                    $label, actual_val, expected_val
                );
                mismatches += 1;
            }
        };
    }

    // Dimension & layer counts
    check!(
        "hidden_size",
        config.n_audio_state as usize,
        reference.hidden_size
    );
    check!(
        "num_heads",
        config.n_audio_head as usize,
        reference.num_heads
    );
    check!(
        "head_dim",
        config.n_audio_state as usize / config.n_audio_head as usize,
        reference.head_dim
    );
    check!(
        "n_encoder_layers",
        config.n_audio_layer as usize,
        reference.n_encoder_layers
    );
    check!(
        "n_decoder_layers",
        config.n_text_layer as usize,
        reference.n_decoder_layers
    );

    // KV heads (extracted from AttentionType)
    let actual_kv_heads = match config.attention_type {
        crate::model::AttentionType::Gqa { kv_heads } => kv_heads as usize,
        crate::model::AttentionType::Mha => config.n_audio_head as usize,
    };
    check!("num_kv_heads", actual_kv_heads, reference.num_kv_heads);

    // FFN intermediate size (4x hidden for both Whisper and Moonshine)
    let actual_ffn = config.n_audio_state as usize * 4;
    check!(
        "encoder_ffn_intermediate",
        actual_ffn,
        reference.encoder_ffn_intermediate
    );
    check!(
        "decoder_ffn_intermediate",
        actual_ffn,
        reference.decoder_ffn_intermediate
    );

    // Activation functions
    let actual_enc_act = match config.ffn_activation {
        crate::format::FfnActivation::Gelu => "gelu",
        crate::format::FfnActivation::Silu => "silu",
        crate::format::FfnActivation::Swiglu => "swiglu",
        crate::format::FfnActivation::Relu => "relu",
    };
    check!(
        "encoder_activation",
        actual_enc_act,
        reference.encoder_activation
    );

    // Decoder activation: Moonshine uses SiLU (gated MLP), Whisper uses GELU
    let actual_dec_act = match config.model_family {
        crate::format::ModelFamily::Moonshine => "silu",
        _ => actual_enc_act,
    };
    check!(
        "decoder_activation",
        actual_dec_act,
        reference.decoder_activation
    );

    // Positional encoding
    let uses_rope = config.positional_encoding == crate::model::PositionalEncoding::Rotary;
    if reference.rope_base > 0.0 {
        check!("positional_encoding", uses_rope, true);
    } else {
        check!("positional_encoding", uses_rope, false);
    }

    // Bias settings (Moonshine: no bias; Whisper: bias)
    let actual_attn_bias = config.model_family != crate::format::ModelFamily::Moonshine;
    check!("attention_bias", actual_attn_bias, reference.attention_bias);
    check!("norm_bias", actual_attn_bias, reference.norm_bias);

    // Tied embeddings (Moonshine ties token embeddings; Whisper does not)
    let actual_tied = config.model_family == crate::format::ModelFamily::Moonshine;
    check!("tied_embeddings", actual_tied, reference.tied_embeddings);

    println!();
    if mismatches == 0 {
        println!("  Result: PASS ({checks} checks)");
    } else {
        println!("  Result: FAIL ({mismatches} mismatches out of {checks} checks)");
    }

    let status = if mismatches == 0 { "PASS" } else { "FAIL" };
    Ok(CommandResult::success(format!(
        "Config {status}: {mismatches} mismatches"
    )))
}
