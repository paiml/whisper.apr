//! Debug cross-attention behavior - Testing 5 Hypotheses for Step 20
//!
//! This example tests the following hypotheses for cross-attention divergence:
//!
//! H1: Cross-attention K/V are not correctly connected to encoder output
//! H2: Encoder output shape is wrong (should be [1, 1500, d_model])
//! H3: Cross-attention weights never loaded (silent failure)
//! H4: Attention scaling factor wrong (missing 1/sqrt(d_k))
//! H5: KV cache overwrites encoder context (streaming bug)
//!
//! Run with: cargo run --release --example debug_cross_attn

use std::path::Path;
use whisper_apr::format::{AprV2ReaderRef, metadata_to_model_config};

fn test_h3_weight_loading(reader: &AprV2ReaderRef<'_>) -> (usize, usize) {
    println!("=== H3: CROSS-ATTENTION WEIGHT LOADING ===\n");

    let cross_attn_tensors = [
        "decoder.layers.0.encoder_attn.q_proj.weight",
        "decoder.layers.0.encoder_attn.q_proj.bias",
        "decoder.layers.0.encoder_attn.k_proj.weight",
        "decoder.layers.0.encoder_attn.k_proj.bias",
        "decoder.layers.0.encoder_attn.v_proj.weight",
        "decoder.layers.0.encoder_attn.v_proj.bias",
        "decoder.layers.0.encoder_attn.out_proj.weight",
        "decoder.layers.0.encoder_attn.out_proj.bias",
        "decoder.layers.0.encoder_attn_layer_norm.weight",
        "decoder.layers.0.encoder_attn_layer_norm.bias",
    ];

    let mut total_missing = 0;
    let mut total_zero = 0;

    for name in &cross_attn_tensors {
        match reader.get_tensor_as_f32(name) {
            Some(data) => {
                let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
                let max_abs: f32 = data.iter().map(|x| x.abs()).fold(0.0, f32::max);
                let non_zero = data.iter().filter(|&&v| v.abs() > 1e-8).count();
                let pct = non_zero as f32 / data.len() as f32 * 100.0;

                if non_zero == 0 {
                    println!("  {name} - ALL ZEROS!");
                    total_zero += 1;
                } else if pct < 50.0 {
                    println!("  {name} - sparse ({pct:.1}% non-zero, norm={norm:.4})");
                } else {
                    println!("  {name} - loaded (norm={norm:.4}, max={max_abs:.4}, {pct:.1}% non-zero)");
                }
            }
            None => {
                println!("  {name} - NOT FOUND");
                total_missing += 1;
            }
        }
    }

    check_all_layers(reader);

    if total_missing > 0 || total_zero > 0 {
        println!("\n  H3 FAILED: {total_missing} missing, {total_zero} zero tensors");
    } else {
        println!("\n  H3 PASSED: All cross-attention weights loaded");
    }

    (total_missing, total_zero)
}

fn check_all_layers(reader: &AprV2ReaderRef<'_>) {
    println!("\n  --- All Layers Summary ---");
    for layer in 0..4 {
        let qkv = [
            format!("decoder.layers.{layer}.encoder_attn.q_proj.weight"),
            format!("decoder.layers.{layer}.encoder_attn.k_proj.weight"),
            format!("decoder.layers.{layer}.encoder_attn.v_proj.weight"),
        ];

        let mut status = "OK";
        for name in &qkv {
            match reader.get_tensor_as_f32(name) {
                Some(data) if data.iter().map(|x| x * x).sum::<f32>().sqrt() < 1e-6 => {
                    status = "FAIL (zero norm)";
                    break;
                }
                None => {
                    status = "FAIL (missing)";
                    break;
                }
                _ => {}
            }
        }
        println!("  Layer {layer} cross-attn Q/K/V: {status}");
    }
}

fn test_h2_encoder_shape(model: &whisper_apr::WhisperApr, d_model: usize) {
    println!("\n=== H2: ENCODER OUTPUT SHAPE ===\n");

    let audio_path = Path::new("demos/test-audio/test-speech-1.5s.wav");
    if !audio_path.exists() {
        println!("  Test audio not found, skipping shape test");
        return;
    }

    let Ok(audio_bytes) = std::fs::read(audio_path) else { return };
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let s = i16::from_le_bytes([chunk[0], chunk[1]]);
            s as f32 / 32768.0
        })
        .collect();

    let Ok(mel) = model.compute_mel(&samples) else { return };
    let Ok(encoded) = model.encode(&mel) else { return };

    let expected = 1500 * d_model;
    println!("  Expected: [1, 1500, {d_model}] = {expected} elements");
    println!("  Actual:   {} elements", encoded.len());

    if encoded.len() == expected {
        println!("  H2 PASSED: Encoder output shape correct");
    } else {
        println!("  H2 FAILED: Shape mismatch!");
    }

    print_encoder_stats(&encoded);
}

fn print_encoder_stats(encoded: &[f32]) {
    let mean: f32 = encoded.iter().sum::<f32>() / encoded.len() as f32;
    let var: f32 = encoded.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / encoded.len() as f32;
    let std = var.sqrt();
    let max: f32 = encoded.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min: f32 = encoded.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    println!("\n  Encoder output statistics:");
    println!("    Mean: {mean:.6}");
    println!("    Std:  {std:.6}");
    println!("    Min:  {min:.6}");
    println!("    Max:  {max:.6}");

    if std.abs() < 0.01 {
        println!("    Very low variance - encoder may not be processing correctly");
    } else if max.abs() > 100.0 {
        println!("    Very large values - possible numerical issue");
    } else {
        println!("    Statistics look reasonable");
    }
}

fn test_h1_input(model: &whisper_apr::WhisperApr) {
    println!("\n=== H1: CROSS-ATTENTION INPUT VERIFICATION ===\n");
    println!("  Running diagnostic transcriptions...\n");

    let audio_path = Path::new("demos/test-audio/test-speech-1.5s.wav");
    if !audio_path.exists() {
        return;
    }

    let Ok(audio_bytes) = std::fs::read(audio_path) else { return };
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let s = i16::from_le_bytes([chunk[0], chunk[1]]);
            s as f32 / 32768.0
        })
        .collect();

    let opts = whisper_apr::TranscribeOptions::default();
    let Ok(result) = model.transcribe(&samples, opts.clone()) else { return };
    println!("  Real audio: {:?}", &result.text[..result.text.len().min(100)]);

    let silence = vec![0.0; samples.len()];
    let Ok(silence_result) = model.transcribe(&silence, opts.clone()) else { return };
    println!("  Silence:    {:?}", &silence_result.text[..silence_result.text.len().min(100)]);

    let noise: Vec<f32> = (0..samples.len()).map(|i| (i as f32 * 0.1).sin() * 0.1).collect();
    let Ok(noise_result) = model.transcribe(&noise, opts) else { return };
    println!("  Noise:      {:?}", &noise_result.text[..noise_result.text.len().min(100)]);

    if result.text == silence_result.text && result.text == noise_result.text {
        println!("\n  H1 LIKELY FAILED: Same output for different inputs!");
    } else if result.text != silence_result.text {
        println!("\n  H1 LIKELY PASSED: Different inputs produce different outputs");
    } else {
        println!("\n  H1 INCONCLUSIVE: Need more analysis");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CROSS-ATTENTION DEBUG (Step 20 Hypotheses) ===\n");

    let model_path = Path::new("models/whisper-tiny.apr");
    if !model_path.exists() {
        eprintln!("Model not found: {}", model_path.display());
        return Ok(());
    }

    let model_bytes = std::fs::read(model_path)?;
    let reader = AprV2ReaderRef::from_bytes(&model_bytes)?;
    let config = metadata_to_model_config(reader.metadata());
    println!("Model: {:?}", config.model_type);

    let (total_missing, total_zero) = test_h3_weight_loading(&reader);

    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;
    test_h2_encoder_shape(&model, 384);

    println!("\n=== H4: ATTENTION SCALING FACTOR ===\n");
    let d_head = 384 / 6;
    let scale = 1.0 / (d_head as f32).sqrt();
    println!("  d_head={d_head}, expected scale=1/sqrt({d_head})={scale:.6}");

    test_h1_input(&model);

    println!("\n=== H5: KV CACHE ISOLATION CHECK ===\n");
    println!("  H5 requires streaming tests or code review");

    println!("\n{}", "=".repeat(60));
    println!("HYPOTHESIS TEST SUMMARY");
    println!("{}", "=".repeat(60));
    println!("H1 (Cross-attn input):     See transcription comparison above");
    println!("H2 (Encoder shape):        See shape analysis above");
    println!("H3 (Weights loaded):       {total_missing} missing, {total_zero} zero");
    println!("H4 (Scaling factor):       Manual code review required");
    println!("H5 (KV cache):             Streaming test required");

    Ok(())
}
