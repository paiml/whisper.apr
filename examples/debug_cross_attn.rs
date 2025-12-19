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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CROSS-ATTENTION DEBUG (Step 20 Hypotheses) ===\n");

    let model_path = Path::new("models/whisper-tiny.apr");
    if !model_path.exists() {
        eprintln!("Model not found: {}", model_path.display());
        eprintln!(
            "Please convert with: cargo run --bin whisper-convert --features converter -- tiny"
        );
        return Ok(());
    }

    let model_bytes = std::fs::read(model_path)?;
    let reader = whisper_apr::format::AprReader::new(model_bytes.clone())?;

    println!("Model: {:?}", reader.header.model_type);
    println!("Quantization: {:?}\n", reader.header.quantization);

    // =========================================================================
    // H3: Check if cross-attention weights are loaded (HIGHEST PRIORITY)
    // =========================================================================
    println!("=== H3: CROSS-ATTENTION WEIGHT LOADING ===\n");

    let cross_attn_tensors = [
        // Layer 0
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
        match reader.load_tensor(name) {
            Ok(data) => {
                let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
                let max_abs: f32 = data.iter().map(|x| x.abs()).fold(0.0, f32::max);
                let non_zero = data.iter().filter(|&&v| v.abs() > 1e-8).count();
                let pct_nonzero = non_zero as f32 / data.len() as f32 * 100.0;

                if non_zero == 0 {
                    println!("  ❌ {} - ALL ZEROS!", name);
                    total_zero += 1;
                } else if pct_nonzero < 50.0 {
                    println!(
                        "  ⚠️  {} - sparse ({:.1}% non-zero, norm={:.4})",
                        name, pct_nonzero, norm
                    );
                } else {
                    println!(
                        "  ✅ {} - loaded (norm={:.4}, max={:.4}, {:.1}% non-zero)",
                        name, norm, max_abs, pct_nonzero
                    );
                }
            }
            Err(e) => {
                println!("  ❌ {} - NOT FOUND: {}", name, e);
                total_missing += 1;
            }
        }
    }

    // Check all layers
    println!("\n  --- All Layers Summary ---");
    let num_layers = 4; // tiny model has 4 layers
    let d_model = 384; // tiny model dimension

    for layer in 0..num_layers {
        let qkv_names = [
            format!("decoder.layers.{}.encoder_attn.q_proj.weight", layer),
            format!("decoder.layers.{}.encoder_attn.k_proj.weight", layer),
            format!("decoder.layers.{}.encoder_attn.v_proj.weight", layer),
        ];

        let mut layer_status = "✅";
        for name in &qkv_names {
            match reader.load_tensor(name) {
                Ok(data) => {
                    let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm < 1e-6 {
                        layer_status = "❌ (zero norm)";
                        break;
                    }
                }
                Err(_) => {
                    layer_status = "❌ (missing)";
                    break;
                }
            }
        }
        println!("  Layer {} cross-attn Q/K/V: {}", layer, layer_status);
    }

    if total_missing > 0 || total_zero > 0 {
        println!(
            "\n  🔴 H3 FAILED: {} missing, {} zero tensors",
            total_missing, total_zero
        );
    } else {
        println!("\n  🟢 H3 PASSED: All cross-attention weights loaded");
    }

    // =========================================================================
    // H2: Check encoder output shape
    // =========================================================================
    println!("\n=== H2: ENCODER OUTPUT SHAPE ===\n");

    // Load model and run encoder
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    // Load test audio
    let audio_path = Path::new("demos/test-audio/test-speech-1.5s.wav");
    if !audio_path.exists() {
        println!("  ⚠️  Test audio not found, skipping shape test");
    } else {
        let audio_bytes = std::fs::read(audio_path)?;
        let samples: Vec<f32> = audio_bytes[44..]
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            })
            .collect();

        let mel = model.compute_mel(&samples)?;
        let encoded = model.encode(&mel)?;

        // Expected: [1, 1500, d_model] flattened = 1500 * d_model = 576000
        let expected_elements = 1500 * d_model;
        let actual_elements = encoded.len();

        println!(
            "  Expected encoder output: [1, 1500, {}] = {} elements",
            d_model, expected_elements
        );
        println!("  Actual encoder output: {} elements", actual_elements);

        if actual_elements == expected_elements {
            println!("  🟢 H2 PASSED: Encoder output shape correct");
        } else {
            println!(
                "  🔴 H2 FAILED: Shape mismatch! Got {} instead of {}",
                actual_elements, expected_elements
            );

            // Additional diagnostics
            if actual_elements == 1500 * d_model / 2 {
                println!("     → Possible cause: Only half the frames processed");
            } else if actual_elements == d_model {
                println!("     → Possible cause: Only first frame output");
            } else if actual_elements % 1500 == 0 {
                println!(
                    "     → Possible cause: Wrong d_model ({} instead of {})",
                    actual_elements / 1500,
                    d_model
                );
            }
        }

        // Check encoder output statistics
        println!("\n  Encoder output statistics:");
        let enc_mean: f32 = encoded.iter().sum::<f32>() / encoded.len() as f32;
        let enc_var: f32 =
            encoded.iter().map(|&x| (x - enc_mean).powi(2)).sum::<f32>() / encoded.len() as f32;
        let enc_std = enc_var.sqrt();
        let enc_max: f32 = encoded.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let enc_min: f32 = encoded.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        println!("    Mean: {:.6}", enc_mean);
        println!("    Std:  {:.6}", enc_std);
        println!("    Min:  {:.6}", enc_min);
        println!("    Max:  {:.6}", enc_max);

        // Check if normalized (Whisper uses LayerNorm, so values should be reasonable)
        if enc_std.abs() < 0.01 {
            println!("    ⚠️  Very low variance - encoder may not be processing correctly");
        } else if enc_max.abs() > 100.0 {
            println!("    ⚠️  Very large values - possible numerical issue");
        } else {
            println!("    ✅ Statistics look reasonable");
        }
    }

    // =========================================================================
    // H4: Check attention scaling factor
    // =========================================================================
    println!("\n=== H4: ATTENTION SCALING FACTOR ===\n");

    // For Whisper tiny: d_model=384, n_heads=6, d_head=64
    // Scaling factor should be 1/sqrt(64) = 0.125
    let n_heads = 6;
    let d_head = d_model / n_heads;
    let expected_scale = 1.0 / (d_head as f32).sqrt();

    println!("  Model config:");
    println!("    d_model = {}", d_model);
    println!("    n_heads = {}", n_heads);
    println!("    d_head  = {} (= d_model / n_heads)", d_head);
    println!(
        "    Expected attention scale = 1/sqrt({}) = {:.6}",
        d_head, expected_scale
    );

    // We can't directly inspect the scale factor at runtime without modifying code,
    // but we can verify the math is correct
    println!("\n  ℹ️  To verify H4: Check src/model/attention.rs for scaling");
    println!("      Look for: let scale = 1.0 / (d_head as f32).sqrt()");
    println!("      Or equivalent scaling in attention score calculation");

    // =========================================================================
    // H1: Verify cross-attention uses encoder output
    // =========================================================================
    println!("\n=== H1: CROSS-ATTENTION INPUT VERIFICATION ===\n");

    // This requires runtime instrumentation. We can:
    // 1. Check that encoder output is non-trivial
    // 2. Run transcription and see if output changes with different audio

    println!("  Running diagnostic transcriptions...\n");

    // Test with actual audio
    if Path::new("demos/test-audio/test-speech-1.5s.wav").exists() {
        let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
        let samples: Vec<f32> = audio_bytes[44..]
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            })
            .collect();

        let result = model.transcribe(&samples, whisper_apr::TranscribeOptions::default())?;
        println!(
            "  Real audio transcription: {:?}",
            &result.text[..result.text.len().min(100)]
        );

        // Test with silence (zeros)
        let silence: Vec<f32> = vec![0.0; samples.len()];
        let silence_result =
            model.transcribe(&silence, whisper_apr::TranscribeOptions::default())?;
        println!(
            "  Silence transcription:    {:?}",
            &silence_result.text[..silence_result.text.len().min(100)]
        );

        // Test with noise
        let noise: Vec<f32> = (0..samples.len())
            .map(|i| (i as f32 * 0.1).sin() * 0.1)
            .collect();
        let noise_result = model.transcribe(&noise, whisper_apr::TranscribeOptions::default())?;
        println!(
            "  Noise transcription:      {:?}",
            &noise_result.text[..noise_result.text.len().min(100)]
        );

        // Analysis
        if result.text == silence_result.text && result.text == noise_result.text {
            println!("\n  🔴 H1 LIKELY FAILED: Same output for different inputs!");
            println!("     → Cross-attention may not be using encoder output");
        } else if result.text != silence_result.text {
            println!("\n  🟢 H1 LIKELY PASSED: Different inputs produce different outputs");
        } else {
            println!("\n  ⚠️  H1 INCONCLUSIVE: Need more analysis");
        }
    }

    // =========================================================================
    // H5: KV Cache check
    // =========================================================================
    println!("\n=== H5: KV CACHE ISOLATION CHECK ===\n");
    println!("  ℹ️  H5 requires streaming tests or code review");
    println!("      Check src/model/decoder.rs for KV cache implementation");
    println!("      Verify encoder KV cache is separate from decoder self-attn cache");

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n{}", "=".repeat(60));
    println!("HYPOTHESIS TEST SUMMARY");
    println!("{}", "=".repeat(60));
    println!("H1 (Cross-attn input):     See transcription comparison above");
    println!("H2 (Encoder shape):        See shape analysis above");
    println!(
        "H3 (Weights loaded):       {} missing, {} zero",
        total_missing, total_zero
    );
    println!("H4 (Scaling factor):       Manual code review required");
    println!("H5 (KV cache):             Streaming test required");
    println!("{}", "=".repeat(60));

    Ok(())
}
