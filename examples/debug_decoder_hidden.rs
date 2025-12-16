//! H12/H13/H14: Trace decoder hidden state through self-attention and LayerNorm
//!
//! Find where variance is lost in the decoder path:
//! Token Embed → Self-Attn → LN1 → Cross-Attn (Q input)

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== H12/H13/H14: DECODER HIDDEN STATE TRACE ===\n");

    let model_path = Path::new("models/whisper-tiny.apr");
    let model_bytes = std::fs::read(model_path)?;
    let mut model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    let d_model = 384;

    // Check H13: Token embedding scale
    println!("[H13: TOKEN EMBEDDING WEIGHTS]");
    let decoder = model.decoder_mut();
    let token_embed = decoder.token_embedding();
    let vocab_size = 51865;

    let (te_mean, te_std, te_min, te_max) = stats(token_embed);
    println!("  Total shape: {} values ({}×{})", token_embed.len(), vocab_size, d_model);
    println!("  mean={:.6}  std={:.6}  min={:.6}  max={:.6}", te_mean, te_std, te_min, te_max);

    // Check specific token embeddings
    println!("\n  Per-token embedding stats:");
    use whisper_apr::tokenizer::special_tokens;
    let tokens = [
        (special_tokens::SOT, "SOT"),
        (special_tokens::LANG_BASE, "EN"),
        (special_tokens::TRANSCRIBE, "TRANSCRIBE"),
        (special_tokens::NO_TIMESTAMPS, "NO_TS"),
        (220, "' '(space)"),
        (464, "'The'"),
    ];

    for (tok, name) in &tokens {
        let start = (*tok as usize) * d_model;
        let end = start + d_model;
        if end <= token_embed.len() {
            let emb = &token_embed[start..end];
            let (m, s, mn, mx) = stats(emb);
            println!("    {:5} {:<12}: mean={:+.4}  std={:.4}  range=[{:.3}, {:.3}]",
                     tok, name, m, s, mn, mx);
        }
    }

    // Check H13 verdict
    println!("\n=== H13 VERDICT ===");
    if te_std < 0.001 {
        println!("CONFIRMED: Token embeddings have near-zero variance (std={:.6})", te_std);
        println!("  -> Embeddings not properly initialized or loaded");
    } else if te_std > 0.1 {
        println!("FALSIFIED: Token embeddings have healthy variance (std={:.6})", te_std);
        println!("  -> Problem is NOT in token embedding weights");
    } else {
        println!("INCONCLUSIVE: Token embedding std={:.6} (borderline)", te_std);
    }

    // Check positional embedding
    println!("\n[POSITIONAL EMBEDDING]");
    let pos_embed = decoder.positional_embedding();
    let max_pos = 448; // Whisper tiny max positions
    let (pe_mean, pe_std, pe_min, pe_max) = stats(pos_embed);
    println!("  shape: {} values ({}×{})", pos_embed.len(), max_pos, d_model);
    println!("  mean={:.6}  std={:.6}  min={:.6}  max={:.6}", pe_mean, pe_std, pe_min, pe_max);

    // Check LayerNorm weights
    println!("\n[H14: LAYERNORM WEIGHTS]");
    let blocks = decoder.blocks();
    let block0 = &blocks[0];

    // Pre-self-attention LayerNorm (ln1)
    let ln1_gamma = &block0.ln1.weight;
    let ln1_beta = &block0.ln1.bias;
    let (g1_mean, g1_std, g1_min, g1_max) = stats(ln1_gamma);
    let (b1_mean, b1_std, b1_min, b1_max) = stats(ln1_beta);
    println!("  ln1 (pre-self-attn):");
    println!("    gamma: mean={:.4}  std={:.4}  range=[{:.4}, {:.4}]", g1_mean, g1_std, g1_min, g1_max);
    println!("    beta:  mean={:.4}  std={:.4}  range=[{:.4}, {:.4}]", b1_mean, b1_std, b1_min, b1_max);

    // Pre-cross-attention LayerNorm (ln2)
    let ln2_gamma = &block0.ln2.weight;
    let ln2_beta = &block0.ln2.bias;
    let (g2_mean, g2_std, g2_min, g2_max) = stats(ln2_gamma);
    let (b2_mean, b2_std, b2_min, b2_max) = stats(ln2_beta);
    println!("  ln2 (pre-cross-attn):");
    println!("    gamma: mean={:.4}  std={:.4}  range=[{:.4}, {:.4}]", g2_mean, g2_std, g2_min, g2_max);
    println!("    beta:  mean={:.4}  std={:.4}  range=[{:.4}, {:.4}]", b2_mean, b2_std, b2_min, b2_max);

    // Manually trace through the decoder for one token
    println!("\n[STEP-BY-STEP TRACE FOR TOKEN 'SOT']");

    // Step 1: Token embedding lookup
    let sot_id = special_tokens::SOT as usize;
    let tok_emb: Vec<f32> = token_embed[sot_id * d_model..(sot_id + 1) * d_model].to_vec();
    let (s1_mean, s1_std, _, _) = stats(&tok_emb);
    println!("  1. Token embed:     mean={:+.6}  std={:.6}", s1_mean, s1_std);

    // Step 2: Add positional embedding (position 0)
    let mut hidden: Vec<f32> = tok_emb.iter()
        .zip(&pos_embed[0..d_model])
        .map(|(t, p)| t + p)
        .collect();
    let (s2_mean, s2_std, _, _) = stats(&hidden);
    println!("  2. + Positional:    mean={:+.6}  std={:.6}", s2_mean, s2_std);

    // Step 3: LayerNorm before self-attention
    // LN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
    let eps = 1e-5f32;
    let h_mean: f32 = hidden.iter().sum::<f32>() / d_model as f32;
    let h_var: f32 = hidden.iter().map(|&x| (x - h_mean).powi(2)).sum::<f32>() / d_model as f32;
    let h_std_ln = (h_var + eps).sqrt();

    let normed: Vec<f32> = hidden.iter()
        .zip(ln1_gamma.iter().zip(ln1_beta.iter()))
        .map(|(&x, (&g, &b))| g * (x - h_mean) / h_std_ln + b)
        .collect();
    let (s3_mean, s3_std, _, _) = stats(&normed);
    println!("  3. LayerNorm1:      mean={:+.6}  std={:.6}", s3_mean, s3_std);

    // Step 4: Self-attention (simplified - using just Q projection for now)
    let w_q_self = &block0.self_attn.w_q().weight;
    let mut q_self = vec![0.0f32; d_model];
    for out_idx in 0..d_model {
        for in_idx in 0..d_model {
            q_self[out_idx] += normed[in_idx] * w_q_self[out_idx * d_model + in_idx];
        }
    }
    let (s4_mean, s4_std, _, _) = stats(&q_self);
    println!("  4. Self-attn Q:     mean={:+.6}  std={:.6}", s4_mean, s4_std);

    // Step 5: After self-attention (approximation: just use q_self as output)
    // In reality, we'd compute full attention, but let's check if Q itself is reasonable

    // Step 6: LayerNorm before cross-attention
    let q_mean: f32 = q_self.iter().sum::<f32>() / d_model as f32;
    let q_var: f32 = q_self.iter().map(|&x| (x - q_mean).powi(2)).sum::<f32>() / d_model as f32;
    let q_std_ln = (q_var + eps).sqrt();

    let normed2: Vec<f32> = q_self.iter()
        .zip(ln2_gamma.iter().zip(ln2_beta.iter()))
        .map(|(&x, (&g, &b))| g * (x - q_mean) / q_std_ln + b)
        .collect();
    let (s5_mean, s5_std, _, _) = stats(&normed2);
    println!("  5. LayerNorm2:      mean={:+.6}  std={:.6}", s5_mean, s5_std);

    // Step 7: Cross-attention Q projection
    let w_q_cross = &block0.cross_attn.w_q().weight;
    let mut q_cross = vec![0.0f32; d_model];
    for out_idx in 0..d_model {
        for in_idx in 0..d_model {
            q_cross[out_idx] += normed2[in_idx] * w_q_cross[out_idx * d_model + in_idx];
        }
    }
    let (s6_mean, s6_std, _, _) = stats(&q_cross);
    println!("  6. Cross-attn Q:    mean={:+.6}  std={:.6}", s6_mean, s6_std);

    // Summary
    println!("\n=== VARIANCE TRACE SUMMARY ===");
    println!("  Token Embed  → std={:.6}", s1_std);
    println!("  +Positional  → std={:.6}", s2_std);
    println!("  LayerNorm1   → std={:.6}", s3_std);
    println!("  Self-attn Q  → std={:.6}", s4_std);
    println!("  LayerNorm2   → std={:.6}", s5_std);
    println!("  Cross-attn Q → std={:.6}", s6_std);

    // Identify the drop
    let stds = [s1_std, s2_std, s3_std, s4_std, s5_std, s6_std];
    let names = ["Token Embed", "+Positional", "LayerNorm1", "Self-attn Q", "LayerNorm2", "Cross-attn Q"];

    println!("\n=== DIAGNOSIS ===");
    for i in 1..stds.len() {
        let ratio = stds[i] / stds[i-1];
        if ratio < 0.1 {
            println!("  ⚠️  {} → {}: std dropped by {:.1}x",
                     names[i-1], names[i], 1.0/ratio);
        }
    }

    if s6_std < 0.1 {
        println!("\n  CONFIRMED: Cross-attention Q has low variance (std={:.6})", s6_std);

        if s3_std < 0.1 {
            println!("  → Problem starts at LayerNorm1 (std={:.6})", s3_std);
        } else if s4_std < 0.1 {
            println!("  → Problem starts at Self-attention (std={:.6})", s4_std);
        } else if s5_std < 0.1 {
            println!("  → Problem starts at LayerNorm2 (std={:.6})", s5_std);
        }
    } else {
        println!("\n  Cross-attention Q has healthy variance (std={:.6})", s6_std);
        println!("  → Bug may be in actual forward pass, not weights");
    }

    Ok(())
}
