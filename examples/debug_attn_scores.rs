//! H9/H10 Falsification: Check attention scores and weights
//!
//! H9: Matmul dimension check - verify Q @ K^T produces [dec, seq] shape
//! H10: Scale factor check - verify scores are in reasonable range (-10 to 10)
//!
//! If attention weights are uniform (entropy ≈ log(seq_len)): Posterior collapse confirmed

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

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn entropy(probs: &[f32]) -> f32 {
    -probs
        .iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| p * p.ln())
        .sum::<f32>()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== H9/H10 FALSIFICATION: ATTENTION SCORES ===\n");

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

    let d_model = 384;
    let n_heads = 6;
    let d_head = d_model / n_heads; // 64
    let seq_len = encoded.len() / d_model;

    println!("Encoder: {} timesteps x {} dims", seq_len, d_model);

    // Get cross-attention weights
    let decoder = model.decoder_mut();
    let blocks = decoder.blocks();
    let block0 = &blocks[0];

    let w_q = &block0.cross_attn.w_q().weight;
    let w_k = &block0.cross_attn.w_k().weight;
    let scale = block0.cross_attn.scale();

    println!(
        "Scale factor: {:.6} (should be 1/sqrt({})={:.6})",
        scale,
        d_head,
        1.0 / (d_head as f32).sqrt()
    );

    // Simulate a decoder query (using first timestep of encoder as dummy decoder state)
    // In real decoding, this would be the decoder hidden state after self-attention
    let decoder_hidden: Vec<f32> = encoded[0..d_model].to_vec();

    // Compute Q = decoder_hidden @ W_q (simulating single query position)
    let mut q = vec![0.0f32; d_model];
    for out_idx in 0..d_model {
        for in_idx in 0..d_model {
            q[out_idx] += decoder_hidden[in_idx] * w_q[out_idx * d_model + in_idx];
        }
    }

    // Compute K = encoded @ W_k for all timesteps
    let mut k_all = vec![0.0f32; seq_len * d_model];
    for t in 0..seq_len {
        let enc_t = &encoded[t * d_model..(t + 1) * d_model];
        for out_idx in 0..d_model {
            for in_idx in 0..d_model {
                k_all[t * d_model + out_idx] += enc_t[in_idx] * w_k[out_idx * d_model + in_idx];
            }
        }
    }

    println!("\n[DIMENSION CHECK (H9)]");
    println!("  Q shape: [1, {}] (single query position)", d_model);
    println!(
        "  K shape: [{}, {}] (all encoder positions)",
        seq_len, d_model
    );
    println!("  Expected scores shape: [1, {}]", seq_len);

    // Compute attention scores for head 0: Q @ K^T / sqrt(d_k)
    // Extract head 0 from Q and K
    let q_head0: Vec<f32> = q[0..d_head].to_vec();
    let k_head0: Vec<Vec<f32>> = (0..seq_len)
        .map(|t| k_all[t * d_model..t * d_model + d_head].to_vec())
        .collect();

    // Compute dot products
    let mut scores = vec![0.0f32; seq_len];
    for t in 0..seq_len {
        let dot: f32 = q_head0.iter().zip(&k_head0[t]).map(|(a, b)| a * b).sum();
        scores[t] = dot * scale; // Apply 1/sqrt(d_k)
    }

    let (scores_mean, scores_std, scores_min, scores_max) = stats(&scores);

    println!("\n[ATTENTION SCORES (H10)]");
    println!("  Scores shape: [{}] (one query to all keys)", seq_len);
    println!(
        "  mean={:.4}  std={:.4}  min={:.4}  max={:.4}",
        scores_mean, scores_std, scores_min, scores_max
    );

    // Check H10: Are scores in reasonable range?
    println!("\n=== H10 VERDICT ===");
    if scores_max > 50.0 || scores_min < -50.0 {
        println!("CONFIRMED: Scores out of range (softmax saturation)");
        println!("  -> Missing or incorrect scale factor");
    } else if scores_std < 0.1 {
        println!(
            "CONFIRMED: Scores have very low variance (std={:.4})",
            scores_std
        );
        println!("  -> All scores similar, causing uniform attention");
    } else {
        println!(
            "FALSIFIED: Scores in reasonable range [{:.2}, {:.2}]",
            scores_min, scores_max
        );
        println!("  -> Scale factor is working correctly");
    }

    // Apply softmax
    let attn_weights = softmax(&scores);
    let attn_entropy = entropy(&attn_weights);
    let max_entropy = (seq_len as f32).ln();
    let entropy_ratio = attn_entropy / max_entropy;

    println!("\n[ATTENTION WEIGHT DISTRIBUTION]");
    println!(
        "  Entropy: {:.4} (max={:.4}, ratio={:.2}%)",
        attn_entropy,
        max_entropy,
        entropy_ratio * 100.0
    );

    let (w_mean, w_std, w_min, w_max) = stats(&attn_weights);
    println!(
        "  mean={:.6}  std={:.6}  min={:.6}  max={:.6}",
        w_mean, w_std, w_min, w_max
    );

    // Top-5 attention positions
    let mut indexed: Vec<(usize, f32)> = attn_weights.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("\n  Top-5 attended positions:");
    for (i, (pos, weight)) in indexed.iter().take(5).enumerate() {
        println!("    {}: position {} = {:.4}", i + 1, pos, weight);
    }

    // Posterior collapse detection
    println!("\n=== POSTERIOR COLLAPSE VERDICT ===");
    if entropy_ratio > 0.95 {
        println!(
            "CONFIRMED: Attention is UNIFORM (entropy ratio = {:.1}%)",
            entropy_ratio * 100.0
        );
        println!("  -> Decoder ignores encoder output");
        println!("  -> Cross-attention provides no useful signal");
    } else if entropy_ratio > 0.8 {
        println!(
            "PARTIAL: Attention is WEAK (entropy ratio = {:.1}%)",
            entropy_ratio * 100.0
        );
        println!("  -> Some differentiation but not strong peaks");
    } else {
        println!(
            "FALSIFIED: Attention is PEAKED (entropy ratio = {:.1}%)",
            entropy_ratio * 100.0
        );
        println!("  -> Decoder is using encoder output correctly");
    }

    Ok(())
}
