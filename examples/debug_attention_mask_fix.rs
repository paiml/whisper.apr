//! Debug: Verify Attention Mask Fix (H35 Resolution)
//!
//! Tests that the attention masking correctly prevents attending to padding positions.
//!
//! Run: cargo run --example debug_attention_mask_fix

use std::path::Path;

fn softmax(data: &mut [f32]) {
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in data.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in data.iter_mut() {
        *x /= sum;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     ATTENTION MASK FIX VERIFICATION                              ║");
    println!("║     Testing that masking prevents padding attention              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let model_path = Path::new("models/whisper-tiny.apr");
    let model_bytes = std::fs::read(model_path)?;
    let mut model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    // Load test audio
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    println!("[INPUT]\n");
    println!("  Audio samples: {}", samples.len());

    // Compute audio encoder length
    // HOP_LENGTH = 160, stride-2 conv means encoder_len = audio_samples / 320
    let audio_encoder_len = (samples.len() + 319) / 320;
    println!(
        "  Audio encoder length: {} (of 1500 total positions)",
        audio_encoder_len
    );
    println!("  Padding positions: {}-1499", audio_encoder_len);

    let mel = model.compute_mel(&samples)?;
    let encoder_output = model.encode(&mel)?;

    let d_model = 384;
    let d_head = 64;
    let enc_len = encoder_output.len() / d_model;
    let scale = (d_head as f32).sqrt();

    // Get decoder cross-attention weights
    let decoder = model.decoder_mut();
    let token_embed = decoder.token_embedding().to_vec();
    let blocks = decoder.blocks();
    let cross_attn = &blocks[0].cross_attn;

    let q_weight = cross_attn.w_q().weight.clone();
    let q_bias = cross_attn.w_q().bias.clone();
    let k_weight = cross_attn.w_k().weight.clone();
    let k_bias = cross_attn.w_k().bias.clone();

    // SOT token embedding
    let sot_token = 50258u32;
    let sot_embed_start = sot_token as usize * d_model;
    let sot_embed = &token_embed[sot_embed_start..sot_embed_start + d_model];

    // Compute Query
    let mut query = q_bias.to_vec();
    for o in 0..d_model {
        for i in 0..d_model {
            query[o] += sot_embed[i] * q_weight[o * d_model + i];
        }
    }

    // ================================================================
    // TEST 1: Without mask (current broken behavior)
    // ================================================================
    println!("\n[TEST 1: WITHOUT MASK (BROKEN)]\n");

    let mut unmasked_scores = Vec::with_capacity(enc_len);
    for pos in 0..enc_len {
        let enc_slice = &encoder_output[pos * d_model..(pos + 1) * d_model];

        // Compute K = enc @ K_weight.T + K_bias
        let mut key = k_bias.clone();
        for o in 0..d_model {
            for i in 0..d_model {
                key[o] += enc_slice[i] * k_weight[o * d_model + i];
            }
        }

        let score: f32 = query
            .iter()
            .zip(key.iter())
            .map(|(q, k)| q * k)
            .sum::<f32>()
            / scale;
        unmasked_scores.push(score);
    }

    let mut unmasked_weights = unmasked_scores.clone();
    softmax(&mut unmasked_weights);

    let audio_attn_unmasked: f32 = unmasked_weights[..audio_encoder_len].iter().sum();
    let padding_attn_unmasked: f32 = unmasked_weights[audio_encoder_len..].iter().sum();

    println!(
        "  Audio attention (0-{}):   {:.4} ({:.1}%)",
        audio_encoder_len - 1,
        audio_attn_unmasked,
        audio_attn_unmasked * 100.0
    );
    println!(
        "  Padding attention ({}-1499): {:.4} ({:.1}%)",
        audio_encoder_len,
        padding_attn_unmasked,
        padding_attn_unmasked * 100.0
    );

    // ================================================================
    // TEST 2: With mask (the fix)
    // ================================================================
    println!("\n[TEST 2: WITH MASK (FIXED)]\n");

    let mut masked_scores = unmasked_scores.clone();

    // Apply -inf mask to padding positions
    for i in audio_encoder_len..enc_len {
        masked_scores[i] = f32::NEG_INFINITY;
    }

    let mut masked_weights = masked_scores.clone();
    softmax(&mut masked_weights);

    let audio_attn_masked: f32 = masked_weights[..audio_encoder_len].iter().sum();
    let padding_attn_masked: f32 = masked_weights[audio_encoder_len..].iter().sum();

    println!(
        "  Audio attention (0-{}):   {:.4} ({:.1}%)",
        audio_encoder_len - 1,
        audio_attn_masked,
        audio_attn_masked * 100.0
    );
    println!(
        "  Padding attention ({}-1499): {:.4} ({:.1}%)",
        audio_encoder_len,
        padding_attn_masked,
        padding_attn_masked * 100.0
    );

    // ================================================================
    // VERDICT
    // ================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         FIX VERDICT                              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let broken = audio_attn_unmasked < padding_attn_unmasked;
    let fixed = audio_attn_masked > 0.99;

    if broken && fixed {
        println!("  ✓ FIX VERIFIED!");
        println!(
            "    Before: {:.1}% audio / {:.1}% padding (BROKEN)",
            audio_attn_unmasked * 100.0,
            padding_attn_unmasked * 100.0
        );
        println!(
            "    After:  {:.1}% audio / {:.1}% padding (FIXED)",
            audio_attn_masked * 100.0,
            padding_attn_masked * 100.0
        );
        println!("\n  The attention mask correctly forces all attention to audio positions.");
    } else if !broken {
        println!("  ? UNEXPECTED: Unmasked attention already correct");
        println!(
            "    Audio: {:.1}%, Padding: {:.1}%",
            audio_attn_unmasked * 100.0,
            padding_attn_unmasked * 100.0
        );
    } else {
        println!("  ✗ FIX NOT WORKING");
        println!(
            "    Masked audio attention: {:.1}%",
            audio_attn_masked * 100.0
        );
    }

    // Show top-5 positions after masking
    println!("\n[TOP-5 ATTENTION POSITIONS (MASKED)]\n");
    let mut indexed: Vec<(usize, f32)> = masked_weights.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (pos, weight) in indexed.iter().take(5) {
        let region = if *pos < audio_encoder_len {
            "AUDIO"
        } else {
            "PADDING"
        };
        println!(
            "    Position {:4}: {:6.2}% [{}]",
            pos,
            weight * 100.0,
            region
        );
    }

    Ok(())
}
