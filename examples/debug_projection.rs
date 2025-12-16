//! Debug the vocabulary projection (logits computation)
//!
//! Traces the hidden state before projection and analyzes why logits are shifted.

use std::cell::RefCell;
use whisper_apr::model::DecoderKVCache;
use whisper_apr::tokenizer::special_tokens;
use whisper_apr::WhisperApr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DEBUG: Vocabulary Projection ===\n");

    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let mut model = WhisperApr::load_from_apr(&model_bytes)?;

    // Load audio and compute encoder output
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let mel = model.compute_mel(&samples)?;
    let encoded = model.encode(&mel)?;

    let n_vocab = 51865;
    let max_tokens = 448;
    let d_model = 384;
    let n_layers = 4;

    // Process initial tokens
    let initial_tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    let cache = RefCell::new(DecoderKVCache::new(n_layers, d_model, max_tokens));

    // Process all initial tokens
    for &token in &initial_tokens {
        let _ = model
            .decoder_mut()
            .forward_one(token, &encoded, &mut cache.borrow_mut())?;
    }

    // Get the decoder's token embeddings
    let decoder = model.decoder_mut();
    let token_emb = decoder.token_embedding();

    // Analyze token embedding matrix structure
    println!("=== Token Embedding Analysis ===\n");
    println!("Shape: {} x {}", n_vocab, d_model);

    // Compute column means (each column is an embedding dimension)
    let mut col_means = vec![0.0f64; d_model];
    let mut col_vars = vec![0.0f64; d_model];

    for v in 0..n_vocab {
        for d in 0..d_model {
            col_means[d] += token_emb[v * d_model + d] as f64;
        }
    }

    for d in 0..d_model {
        col_means[d] /= n_vocab as f64;
    }

    for v in 0..n_vocab {
        for d in 0..d_model {
            let diff = token_emb[v * d_model + d] as f64 - col_means[d];
            col_vars[d] += diff * diff;
        }
    }

    for d in 0..d_model {
        col_vars[d] = (col_vars[d] / n_vocab as f64).sqrt();
    }

    let col_mean_mean: f64 = col_means.iter().sum::<f64>() / d_model as f64;
    let col_mean_min = col_means.iter().cloned().fold(f64::INFINITY, f64::min);
    let col_mean_max = col_means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("Column (dimension) mean statistics:");
    println!("  Mean of means: {:.6}", col_mean_mean);
    println!("  Min col mean:  {:.6}", col_mean_min);
    println!("  Max col mean:  {:.6}", col_mean_max);

    // Compute row sums (each row is a token's embedding)
    let mut row_sums = vec![0.0f64; n_vocab];
    for v in 0..n_vocab {
        for d in 0..d_model {
            row_sums[v] += token_emb[v * d_model + d] as f64;
        }
    }

    let row_sum_mean: f64 = row_sums.iter().sum::<f64>() / n_vocab as f64;
    let row_sum_min = row_sums.iter().cloned().fold(f64::INFINITY, f64::min);
    let row_sum_max = row_sums.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\nRow (token) sum statistics:");
    println!("  Mean of sums:  {:.6}", row_sum_mean);
    println!("  Min row sum:   {:.6}", row_sum_min);
    println!("  Max row sum:   {:.6}", row_sum_max);

    // Now let's trace what happens if we project a zero vector
    println!("\n=== Zero Vector Projection Test ===\n");

    let zero_vec = vec![0.0f32; d_model];
    let mut zero_logits = vec![0.0f32; n_vocab];

    // Manual matmul: logits[v] = sum_d (x[d] * W[v,d])
    // Since x is zero, logits should all be zero
    // But let's check...
    for v in 0..n_vocab {
        for d in 0..d_model {
            zero_logits[v] += zero_vec[d] * token_emb[v * d_model + d];
        }
    }

    let zero_logits_mean: f32 = zero_logits.iter().sum::<f32>() / n_vocab as f32;
    println!("Zero vector -> logits mean: {:.10} (should be 0)", zero_logits_mean);

    // Now let's try a constant vector
    println!("\n=== Constant Vector Projection Test ===\n");

    let const_vec: Vec<f32> = (0..d_model).map(|_| 1.0f32).collect();
    let mut const_logits = vec![0.0f32; n_vocab];

    for v in 0..n_vocab {
        for d in 0..d_model {
            const_logits[v] += const_vec[d] * token_emb[v * d_model + d];
        }
    }

    let const_logits_mean: f32 = const_logits.iter().sum::<f32>() / n_vocab as f32;
    let const_logits_min = const_logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let const_logits_max = const_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("Constant(1.0) vector -> logits:");
    println!("  Mean: {:.4}", const_logits_mean);
    println!("  Min:  {:.4}", const_logits_min);
    println!("  Max:  {:.4}", const_logits_max);
    println!("  (This equals the row sums of the embedding matrix)");

    // What hidden state would produce the observed logit shift?
    println!("\n=== Hidden State Hypothesis ===\n");

    // If logits = x @ W^T, and we observe logits with mean ~23
    // Then we need to find what x produces this
    //
    // For a centered embedding (row means ≈ 0), a large positive x would
    // not shift logits. But if embedding has consistent positive structure...

    // Let's check: what is the inner product of the encoder output mean with embeddings?
    let enc_mean: Vec<f32> = {
        let mut m = vec![0.0f32; d_model];
        let n_frames = encoded.len() / d_model;
        for f in 0..n_frames {
            for d in 0..d_model {
                m[d] += encoded[f * d_model + d];
            }
        }
        m.iter_mut().for_each(|v| *v /= n_frames as f32);
        m
    };

    println!("Encoder output mean (per dimension):");
    let enc_mean_mean: f32 = enc_mean.iter().sum::<f32>() / d_model as f32;
    println!("  Mean of means: {:.6}", enc_mean_mean);
    println!("  First 10: {:?}", &enc_mean[..10].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());

    // Project encoder mean through embeddings
    let mut enc_mean_logits = vec![0.0f32; n_vocab];
    for v in 0..n_vocab {
        for d in 0..d_model {
            enc_mean_logits[v] += enc_mean[d] * token_emb[v * d_model + d];
        }
    }

    let enc_mean_logits_mean: f32 = enc_mean_logits.iter().sum::<f32>() / n_vocab as f32;
    println!("\nEncoder mean -> logits mean: {:.4}", enc_mean_logits_mean);

    // What if the decoder hidden state has a large constant component?
    // Let's see what constant would produce mean logit of 23
    let target_logit_mean = 23.0f32;
    let implied_constant = target_logit_mean / const_logits_mean;

    println!("\nTo get logit mean of {:.1}:", target_logit_mean);
    println!("  Need hidden state with constant component: {:.4}", implied_constant);
    println!("  (if every hidden dim had this value)");

    // The key question: is there a bias term we're missing?
    println!("\n=== Possible Root Causes ===\n");
    println!("1. Missing output bias: Whisper might have a bias term in logit projection");
    println!("2. LayerNorm shift: Final layer norm might be shifting values");
    println!("3. Cross-attention issue: Cross-attn might be adding constant offset");
    println!("4. Residual accumulation: Residual connections accumulating positive values");

    Ok(())
}
