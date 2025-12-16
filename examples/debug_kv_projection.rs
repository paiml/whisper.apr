//! H8 Falsification: Check if K = encoder_output @ W_k produces degenerate K vectors
//!
//! If K vectors at different timesteps are identical: Projection bug
//! If K vectors are differentiated: Projection OK, bug in attention math

use std::path::Path;

fn stats(data: &[f32]) -> (f32, f32, f32, f32) {
    let n = data.len() as f32;
    let sum: f32 = data.iter().sum();
    let mean = sum / n;
    let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (mean, std, min, max)
}

fn l1_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== H8 FALSIFICATION: K/V PROJECTION ===\n");

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

    // Get encoder output
    let mel = model.compute_mel(&samples)?;
    let encoded = model.encode(&mel)?;

    println!("Encoder output: {} values", encoded.len());
    let d_model = 384;
    let n_heads = 6;
    let d_head = d_model / n_heads; // 64
    let seq_len = encoded.len() / d_model;
    println!("  {} timesteps x {} dims ({} heads x {} per head)\n", seq_len, d_model, n_heads, d_head);

    // Get cross-attention K projection weight from decoder layer 0
    let decoder = model.decoder_mut();
    let blocks = decoder.blocks();
    let block0 = &blocks[0];
    let w_k = &block0.cross_attn.w_k().weight;

    println!("[W_k WEIGHT]");
    let (wk_mean, wk_std, wk_min, wk_max) = stats(w_k);
    println!("  shape: {} values ({}x{})", w_k.len(), d_model, d_model);
    println!("  mean={:.6}  std={:.6}  min={:.6}  max={:.6}", wk_mean, wk_std, wk_min, wk_max);

    // Manually compute K = encoded @ W_k for a few timesteps
    println!("\n[K PROJECTION: K = encoder_output @ W_k]");

    // Extract encoder output for timestep 0 and 100
    let enc_t0: Vec<f32> = encoded[0..d_model].to_vec();
    let enc_t100: Vec<f32> = if seq_len > 100 {
        encoded[100*d_model..(100+1)*d_model].to_vec()
    } else {
        encoded[(seq_len-1)*d_model..seq_len*d_model].to_vec()
    };

    // Manual matmul: K[i] = sum_j(enc[j] * W_k[j, i])
    // W_k is stored as [out_dim, in_dim] = [384, 384]
    let mut k_t0 = vec![0.0f32; d_model];
    let mut k_t100 = vec![0.0f32; d_model];

    for out_idx in 0..d_model {
        for in_idx in 0..d_model {
            let w = w_k[out_idx * d_model + in_idx];
            k_t0[out_idx] += enc_t0[in_idx] * w;
            k_t100[out_idx] += enc_t100[in_idx] * w;
        }
    }

    let (k0_mean, k0_std, k0_min, k0_max) = stats(&k_t0);
    let (k100_mean, k100_std, k100_min, k100_max) = stats(&k_t100);

    println!("  K[t=0]:   mean={:+.4}  std={:.4}  min={:.4}  max={:.4}", k0_mean, k0_std, k0_min, k0_max);
    println!("  K[t=100]: mean={:+.4}  std={:.4}  min={:.4}  max={:.4}", k100_mean, k100_std, k100_min, k100_max);

    let k_dist = l1_distance(&k_t0, &k_t100);
    println!("\n  L1 distance K[t=0] vs K[t=100]: {:.6}", k_dist);

    // H8 Verdict
    println!("\n=== H8 VERDICT ===");
    if k_dist < 0.01 {
        println!("CONFIRMED: K vectors are IDENTICAL (dist={:.6} < 0.01)", k_dist);
        println!("  -> K projection is collapsing encoder signal");
        println!("  -> Check W_k weight initialization or loading");
    } else if k_dist > 0.1 {
        println!("FALSIFIED: K vectors are DIFFERENTIATED (dist={:.6} > 0.1)", k_dist);
        println!("  -> K projection is working correctly");
        println!("  -> Proceed to H9 (matmul dimensions) or H10 (scale factor)");
    } else {
        println!("INCONCLUSIVE: K distance={:.6} (borderline)", k_dist);
    }

    // Additional: Check Q projection for decoder
    println!("\n[Q PROJECTION CHECK]");
    let w_q = &block0.cross_attn.w_q().weight;
    let (wq_mean, wq_std, _, _) = stats(w_q);
    println!("  W_q: mean={:.6}  std={:.6}", wq_mean, wq_std);

    // Check V projection
    let w_v = &block0.cross_attn.w_v().weight;
    let (wv_mean, wv_std, _, _) = stats(w_v);
    println!("  W_v: mean={:.6}  std={:.6}", wv_mean, wv_std);

    Ok(())
}
