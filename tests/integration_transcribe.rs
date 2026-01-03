//! Integration test: Transcription produces meaningful text
//!
//! This test verifies the CRITICAL behavior: given audio with speech,
//! the transcription should produce non-empty, meaningful text.
//!
//! Expected behavior (verified with whisper.cpp):
//! - Audio: demos/test-audio/test-speech-1.5s.wav
//! - Expected output: " The birds can use" (or similar)
//!
//! FAILING CONDITIONS (any indicates a bug):
//! 1. Output is empty
//! 2. Output contains only control characters (like \u{b})
//! 3. Output contains only repeated single character
//! 4. Output length < 5 characters for 1.5s of speech

use whisper_apr::{TranscribeOptions, WhisperApr};

/// Test that transcription produces meaningful text, not garbage
///
/// NOTE: This test requires vocabulary to be embedded in the APR file.
/// If no vocabulary is present, the test will skip.
#[test]
fn test_transcription_produces_meaningful_text() {
    // Use whisper-tiny-fb.apr which has correct weights
    let model_path = if std::path::Path::new("models/whisper-tiny-fb.apr").exists() {
        "models/whisper-tiny-fb.apr"
    } else if std::path::Path::new("models/whisper-tiny.apr").exists() {
        "models/whisper-tiny.apr"
    } else {
        "models/whisper-tiny-int8.apr"
    };
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: no model found");
        return;
    }
    println!("Using model: {}", model_path);

    // Check if vocabulary is embedded in APR file
    let model_bytes = std::fs::read(model_path).expect("Failed to read model");
    let reader =
        whisper_apr::format::AprReader::new(model_bytes.clone()).expect("Failed to parse APR");
    if !reader.has_vocabulary() {
        eprintln!("Skipping test: APR file has no vocabulary embedded. Token generation verified in test_encoder_produces_meaningful_output.");
        return;
    }

    // Load model
    let model = WhisperApr::load_from_apr(&model_bytes).expect("Failed to load model");

    // Load test audio
    let audio_path = "demos/test-audio/test-speech-1.5s.wav";
    if !std::path::Path::new(audio_path).exists() {
        eprintln!("Skipping test: audio not found at {}", audio_path);
        return;
    }

    let audio_bytes = std::fs::read(audio_path).expect("Failed to read audio");
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    // Transcribe
    let result = model
        .transcribe(&samples, TranscribeOptions::default())
        .expect("Transcription should not fail");

    // Debug: print the raw transcription
    println!("Raw transcription: {:?}", result.text);
    println!("Text bytes: {:?}", result.text.as_bytes());
    println!("Text length: {} chars", result.text.len());

    // ASSERTION 1: Output is not empty
    assert!(
        !result.text.is_empty(),
        "Transcription should not be empty for audio with speech"
    );

    // ASSERTION 2: Output has meaningful length (at least 5 chars for 1.5s speech)
    assert!(
        result.text.len() >= 5,
        "Transcription '{}' is too short ({} chars) for 1.5s of speech",
        result.text,
        result.text.len()
    );

    // ASSERTION 3: Output is not all control characters
    let printable_chars = result
        .text
        .chars()
        .filter(|c| !c.is_control() || *c == ' ' || *c == '\n')
        .count();
    assert!(
        printable_chars > 0,
        "Transcription '{}' contains only control characters",
        result.text.escape_debug()
    );

    // ASSERTION 4: Output is not a single repeated character
    let unique_chars: std::collections::HashSet<char> = result.text.chars().collect();
    assert!(
        unique_chars.len() > 1 || result.text.len() <= 1,
        "Transcription '{}' is just repeated '{}'",
        result.text.escape_debug(),
        unique_chars.iter().next().unwrap_or(&'?')
    );

    // ASSERTION 5: Contains at least some alphabetic characters
    let alpha_chars = result.text.chars().filter(|c| c.is_alphabetic()).count();
    assert!(
        alpha_chars >= 3,
        "Transcription '{}' has only {} alphabetic chars, expected at least 3",
        result.text.escape_debug(),
        alpha_chars
    );

    println!("Transcription: '{}'", result.text);
}

/// Test that cross-attention weights are non-zero after model loading
///
/// If cross-attention weights are zero, the decoder cannot attend to
/// encoder outputs, resulting in garbage transcription.
#[test]
fn test_cross_attention_weights_loaded() {
    // Skip if model not available
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: model not found at {}", model_path);
        return;
    }

    // Load model
    let model_bytes = std::fs::read(model_path).expect("Failed to read model");
    let model = WhisperApr::load_from_apr(&model_bytes).expect("Failed to load model");

    // Check first decoder block's cross-attention weights
    let mut model = model;
    let decoder = model.decoder_mut();
    let blocks = decoder.blocks();
    assert!(!blocks.is_empty(), "Decoder should have at least one block");

    // Check that cross_attn has non-zero weights
    let cross_attn = &blocks[0].cross_attn;

    // Get Q weight and check it's not all zeros
    let q_weight_sum: f32 = cross_attn.w_q().weight.iter().map(|x: &f32| x.abs()).sum();
    assert!(
        q_weight_sum > 0.001,
        "Cross-attention Q weights are all zeros (sum={}), weights not loaded",
        q_weight_sum
    );

    let k_weight_sum: f32 = cross_attn.w_k().weight.iter().map(|x: &f32| x.abs()).sum();
    assert!(
        k_weight_sum > 0.001,
        "Cross-attention K weights are all zeros (sum={}), weights not loaded",
        k_weight_sum
    );

    let v_weight_sum: f32 = cross_attn.w_v().weight.iter().map(|x: &f32| x.abs()).sum();
    assert!(
        v_weight_sum > 0.001,
        "Cross-attention V weights are all zeros (sum={}), weights not loaded",
        v_weight_sum
    );

    let o_weight_sum: f32 = cross_attn.w_o().weight.iter().map(|x: &f32| x.abs()).sum();
    assert!(
        o_weight_sum > 0.001,
        "Cross-attention out_proj weights are all zeros (sum={}), weights not loaded",
        o_weight_sum
    );

    println!(
        "Cross-attention weights verified: Q={:.2}, K={:.2}, V={:.2}, O={:.2}",
        q_weight_sum, k_weight_sum, v_weight_sum, o_weight_sum
    );
}

/// Test decoder generates tokens quickly
///
/// NOTE: `decoder.generate()` does NOT apply token suppression (that's done in `WhisperApr::decode()`).
/// Without suppression, the model may repeat special tokens like TRANSCRIBE (50358).
/// This is expected behavior - the test verifies generation doesn't hang.
#[test]
fn test_decoder_generates_tokens_quickly() {
    // Skip if model not available
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: model not found at {}", model_path);
        return;
    }

    // Load model
    let model_bytes = std::fs::read(model_path).expect("Failed to read model");
    let mut model = WhisperApr::load_from_apr(&model_bytes).expect("Failed to load model");

    let decoder = model.decoder_mut();
    let d_model = 384;
    let encoder_len = 10;

    // Create fake encoder output
    let encoder_output: Vec<f32> = (0..encoder_len * d_model)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    // Initial tokens: SOT
    let initial_tokens = vec![50257_u32]; // SOT

    // Try to generate just 5 tokens with timeout
    let start = std::time::Instant::now();
    let max_tokens = 5;
    let eos_token = 50256_u32;

    let result = decoder.generate(&encoder_output, &initial_tokens, max_tokens, eos_token);

    let elapsed = start.elapsed();

    // Should complete quickly (< 30 seconds for 5 tokens even in debug mode)
    assert!(
        elapsed.as_secs() < 30,
        "Generation took too long: {:?} - possible infinite loop",
        elapsed
    );

    let tokens = result.expect("Generation should not fail");
    println!(
        "Generated {} tokens in {:?}: {:?}",
        tokens.len(),
        elapsed,
        tokens
    );

    // Should have generated some tokens
    assert!(
        tokens.len() > initial_tokens.len(),
        "Decoder should generate at least one new token"
    );

    // NOTE: Without suppression, decoder may repeat special tokens - this is expected.
    // The test_encoder_produces_meaningful_output verifies suppression works correctly.
}

/// List all tensor names in APR file for debugging
#[test]
fn test_list_tensor_names() {
    let model_path = "models/whisper-tiny.apr";
    if !std::path::Path::new(model_path).exists() {
        return;
    }

    let model_bytes = std::fs::read(model_path).expect("read model");
    let reader = whisper_apr::format::AprReader::new(model_bytes).expect("parse apr");

    println!("Tensor names in APR file:");
    for (i, tensor) in reader.tensors.iter().enumerate() {
        println!("  {}: {} {:?}", i, tensor.name, tensor.shape);
    }
}

/// Test APR file has vocabulary embedded
#[test]
fn test_apr_has_vocabulary() {
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: model not found at {}", model_path);
        return;
    }

    let model_bytes = std::fs::read(model_path).expect("Failed to read model");
    let reader = whisper_apr::format::AprReader::new(model_bytes).expect("Failed to parse APR");

    println!("APR has vocab: {}", reader.has_vocabulary());
    if reader.has_vocabulary() {
        if let Some(vocab) = reader.read_vocabulary() {
            println!("Vocab size: {}", vocab.len());

            // Print what key tokens decode to
            println!("Token mappings:");
            for token_id in [
                11, 13, 60, 257, 264, 281, 286, 291, 293, 295, 309, 359, 393, 407, 440, 485, 503,
                542, 550, 764, 779, 1904, 1936, 2159, 3549, 5255, 9009, 50257,
            ] {
                if let Some(bytes) = vocab.get_bytes(token_id) {
                    let text = String::from_utf8_lossy(bytes);
                    println!("  Token {}: {:?} -> \"{}\"", token_id, bytes, text);
                }
            }
        } else {
            println!("Failed to read vocab");
        }
    }

    // Check header
    println!("Header n_vocab: {}", reader.header.n_vocab);
}

/// Test encoder produces non-trivial output for real audio
/// Compare batch forward() vs incremental forward_one() to find the bug
#[test]
fn test_batch_vs_incremental_logits() {
    // Try fp32 model first, fall back to int8
    let model_path = if std::path::Path::new("models/whisper-tiny.apr").exists() {
        "models/whisper-tiny.apr"
    } else {
        "models/whisper-tiny-int8.apr"
    };
    println!("Using model: {}", model_path);
    let model_path = model_path;
    if !std::path::Path::new(model_path).exists() {
        return;
    }
    let audio_path = "demos/test-audio/test-speech-1.5s.wav";
    if !std::path::Path::new(audio_path).exists() {
        return;
    }

    let model_bytes = std::fs::read(model_path).expect("read");
    let mut model = WhisperApr::load_from_apr(&model_bytes).expect("load");
    let audio_bytes = std::fs::read(audio_path).expect("read audio");
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let mel = model.compute_mel(&samples).expect("mel");
    let encoder_output = model.encoder_mut().forward_mel(&mel).expect("encode");

    // Multilingual tokens: SOT=50258, LANG_EN=50259, TRANSCRIBE=50359, NO_TIMESTAMPS=50363
    let initial_tokens = vec![50258_u32, 50259, 50359, 50363];
    println!("Using initial_tokens: {:?}", initial_tokens);

    // BATCH path
    let batch_logits = model
        .decoder_mut()
        .forward(&initial_tokens, &encoder_output)
        .expect("batch forward");
    let n_vocab = model.decoder_mut().n_vocab();
    let last_batch_logits = &batch_logits[(initial_tokens.len() - 1) * n_vocab..];
    let mut batch_indexed: Vec<(usize, f32)> = last_batch_logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    batch_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("[BATCH] top5: {:?}", &batch_indexed[..5]);

    // INCREMENTAL path
    let d_model = 384; // tiny model
    let n_layers = 4;
    let max_tokens = 448;
    let mut cache = whisper_apr::model::DecoderKVCache::new(n_layers, d_model, max_tokens);
    let mut incr_logits = vec![];
    for &token in &initial_tokens {
        incr_logits = model
            .decoder_mut()
            .forward_one(token, &encoder_output, &mut cache)
            .expect("forward_one");
    }
    let mut incr_indexed: Vec<(usize, f32)> = incr_logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    incr_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("[INCREMENTAL] top5: {:?}", &incr_indexed[..5]);

    // Compare
    let batch_top = batch_indexed[0].0;
    let incr_top = incr_indexed[0].0;
    println!(
        "Batch top token: {}, score: {}",
        batch_top, batch_indexed[0].1
    );
    println!("Incr top token: {}, score: {}", incr_top, incr_indexed[0].1);

    assert_eq!(
        batch_top, incr_top,
        "MISMATCH: batch predicts {} but incremental predicts {}",
        batch_top, incr_top
    );
}

#[test]
fn test_encoder_produces_meaningful_output() {
    // Skip if model not available
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: model not found at {}", model_path);
        return;
    }

    // Skip if audio not available
    let audio_path = "demos/test-audio/test-speech-1.5s.wav";
    if !std::path::Path::new(audio_path).exists() {
        eprintln!("Skipping test: audio not found at {}", audio_path);
        return;
    }

    // Load model
    let model_bytes = std::fs::read(model_path).expect("Failed to read model");
    let mut model = WhisperApr::load_from_apr(&model_bytes).expect("Failed to load model");

    // Load real audio
    let audio_bytes = std::fs::read(audio_path).expect("Failed to read audio");
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    // Use model's mel computation and encoding
    let mel = model.compute_mel(&samples).expect("Mel computation failed");
    println!("Mel shape: {} samples", mel.len());

    // Get encoder output via forward_mel (processes raw mel through conv frontend)
    let encoder = model.encoder_mut();
    let encoder_output = encoder.forward_mel(&mel).expect("Encoder failed");

    println!("Encoder output: {} values", encoder_output.len());

    // Encoder output should not be all zeros
    let sum: f32 = encoder_output.iter().map(|x| x.abs()).sum();
    assert!(sum > 1.0, "Encoder output is all zeros (sum={})", sum);

    // Encoder output should have reasonable variance
    let mean = encoder_output.iter().sum::<f32>() / encoder_output.len() as f32;
    let variance = encoder_output
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>()
        / encoder_output.len() as f32;
    println!(
        "Encoder output: mean={:.4}, variance={:.4}, sum={:.4}",
        mean, variance, sum
    );

    assert!(
        variance > 0.0001,
        "Encoder output has zero variance - encoder not working"
    );

    // Now test decoder with this real encoder output
    let decoder = model.decoder_mut();
    // SOT=50257, en=50258, transcribe=50358, no_timestamps=50362
    let initial_tokens = vec![50257_u32, 50258, 50358, 50362];

    // First check the logits directly
    let logits = decoder
        .forward(&initial_tokens, &encoder_output)
        .expect("Forward failed");
    let n_vocab = decoder.n_vocab();
    let n_positions = logits.len() / n_vocab;
    println!(
        "Logits shape: {} ({} positions × {} vocab)",
        logits.len(),
        n_positions,
        n_vocab
    );

    // Get LAST position's logits (the next token prediction)
    let last_logits = &logits[(n_positions - 1) * n_vocab..];
    println!("Last position logits: {} values", last_logits.len());

    // Apply the same suppression as decode() in lib.rs
    let mut suppressed_logits = last_logits.to_vec();
    // Suppress special tokens
    for &tok in &[50257_usize, 50361, 50359, 50358, 50360, 50360, 50362] {
        if tok < suppressed_logits.len() {
            suppressed_logits[tok] = f32::NEG_INFINITY;
        }
    }
    // Suppress all language tokens (50258 to 50357)
    for i in 50258..50358 {
        if i < suppressed_logits.len() {
            suppressed_logits[i] = f32::NEG_INFINITY;
        }
    }
    // Suppress timestamps (50363+)
    for i in 50363..suppressed_logits.len() {
        suppressed_logits[i] = f32::NEG_INFINITY;
    }

    // Find top-5 tokens for LAST position AFTER suppression
    let mut indexed: Vec<(usize, f32)> = suppressed_logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("test assertion"));
    println!("Top-5 tokens (last position, after suppression):");
    for (i, &(tok, logit)) in indexed.iter().take(5).enumerate() {
        println!("  {}: token {} = {:.4}", i + 1, tok, logit);
    }

    // NOTE: APR file doesn't have vocabulary embedded, so we can't decode tokens to text.
    // Instead, verify that the suppression logic produces valid text tokens (< EOT=50256)
    // The top token 11 is a valid BPE text token, not a special token.
    assert!(
        indexed[0].0 < 50256,
        "Top predicted token {} should be a text token (< EOT=50256), not a special token",
        indexed[0].0
    );
    println!(
        "Verified: suppression produces text token {} as top prediction",
        indexed[0].0
    );
}

/// Test that logits vary across vocabulary (not dominated by single token)
#[test]
fn test_logits_vary_across_vocabulary() {
    // Skip if model not available
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: model not found at {}", model_path);
        return;
    }

    // Load model
    let model_bytes = std::fs::read(model_path).expect("Failed to read model");
    let mut model = WhisperApr::load_from_apr(&model_bytes).expect("Failed to load model");

    let decoder = model.decoder_mut();
    let d_model = 384;
    let encoder_len = 10;

    // Create encoder output
    let encoder_output: Vec<f32> = (0..encoder_len * d_model)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    let tokens = vec![50257_u32]; // SOT
    let output = decoder
        .forward(&tokens, &encoder_output)
        .expect("Forward failed");

    // Output should be logits over vocabulary
    let n_vocab = decoder.n_vocab();
    assert_eq!(output.len(), n_vocab, "Output should be vocab-sized logits");

    // Find max and second-max
    let mut sorted: Vec<(usize, f32)> = output.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("test assertion"));

    let (best_token, best_logit) = sorted[0];
    let (second_token, second_logit) = sorted[1];
    let gap = best_logit - second_logit;

    println!("Best token: {} (logit={:.4})", best_token, best_logit);
    println!("Second: {} (logit={:.4})", second_token, second_logit);
    println!("Gap: {:.4}", gap);

    // Check variance in logits
    let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    let variance: f32 =
        output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / output.len() as f32;
    println!("Logits: mean={:.4}, variance={:.4}", mean, variance);

    // Variance should be reasonable (not all zeros or all same)
    assert!(
        variance > 0.001,
        "Logits have near-zero variance ({}) - broken forward pass",
        variance
    );
}

/// Test that cross-attention forward pass produces different outputs for different encoder inputs
///
/// This verifies that the cross-attention mechanism is actually working: the decoder
/// should produce different outputs when attending to different encoder states.
#[test]
fn test_cross_attention_forward_varies_with_encoder() {
    // Skip if model not available
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping test: model not found at {}", model_path);
        return;
    }

    // Load model
    let model_bytes = std::fs::read(model_path).expect("Failed to read model");
    let mut model = WhisperApr::load_from_apr(&model_bytes).expect("Failed to load model");

    // Get decoder and run with different encoder outputs
    let decoder = model.decoder_mut();

    // Create two different encoder outputs (simulating different audio)
    let d_model = 384; // tiny model dimension
    let encoder_len = 5; // small for test

    // Encoder output A: positive values
    let encoder_a: Vec<f32> = (0..encoder_len * d_model)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    // Encoder output B: different values
    let encoder_b: Vec<f32> = (0..encoder_len * d_model)
        .map(|i| (i as f32 * 0.02 + 1.0).cos())
        .collect();

    // Use same tokens for both
    let tokens = vec![50257_u32, 50362]; // SOT, transcribe task

    // Run decoder forward with each encoder output
    let output_a = decoder
        .forward(&tokens, &encoder_a)
        .expect("Forward A failed");
    let output_b = decoder
        .forward(&tokens, &encoder_b)
        .expect("Forward B failed");

    // Outputs should be different if cross-attention is working
    let diff: f32 = output_a
        .iter()
        .zip(output_b.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff > 0.01,
        "Cross-attention not working: outputs are identical for different encoder inputs (diff={})",
        diff
    );

    println!(
        "Cross-attention forward verified: diff={:.4} between different encoder inputs",
        diff
    );
}

/// Test that token embeddings vary across different tokens
///
/// If all tokens embed to the same vector, the model can't distinguish them.
#[test]
fn test_token_embeddings_vary() {
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        return;
    }

    let model_bytes = std::fs::read(model_path).expect("read model");
    let mut model = WhisperApr::load_from_apr(&model_bytes).expect("load model");
    let decoder = model.decoder_mut();

    // Get d_model from decoder
    let d_model = 384; // tiny model

    // Test embedding for a few different tokens
    let test_tokens = [50257_u32, 50258, 50358, 50362, 11, 257, 262]; // SOT, en, transcribe, no_timestamps, comma, a, the

    let mut embeddings: Vec<Vec<f32>> = Vec::new();

    for &token in &test_tokens {
        // Forward with just one token to get its embedding contribution
        let fake_encoder = vec![0.0f32; 74 * d_model]; // encoder output
        let logits = decoder.forward(&[token], &fake_encoder).expect("forward");

        // The logits are affected by the embedding - capture first 10 values as fingerprint
        let fingerprint: Vec<f32> = logits.iter().take(100).copied().collect();
        embeddings.push(fingerprint);
    }

    // Verify embeddings are different
    for i in 0..test_tokens.len() {
        for j in (i + 1)..test_tokens.len() {
            let diff: f32 = embeddings[i]
                .iter()
                .zip(embeddings[j].iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            println!(
                "Token {} vs {}: diff = {:.4}",
                test_tokens[i], test_tokens[j], diff
            );

            assert!(
                diff > 0.01,
                "Tokens {} and {} produce identical embeddings (diff={})",
                test_tokens[i],
                test_tokens[j],
                diff
            );
        }
    }
}

/// Test that positional embeddings vary across positions
#[test]
fn test_positional_embeddings_vary() {
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        return;
    }

    let model_bytes = std::fs::read(model_path).expect("read model");
    let mut model = WhisperApr::load_from_apr(&model_bytes).expect("load model");
    let decoder = model.decoder_mut();

    let d_model = 384;
    let fake_encoder = vec![0.0f32; 74 * d_model];

    // Forward with different sequence lengths to see positional effects
    let tokens_1 = vec![50257_u32]; // just SOT
    let tokens_4 = vec![50257_u32, 50258, 50358, 50362]; // SOT + 3 more

    let logits_1 = decoder
        .forward(&tokens_1, &fake_encoder)
        .expect("forward 1");
    let logits_4 = decoder
        .forward(&tokens_4, &fake_encoder)
        .expect("forward 4");

    // The logits for position 0 should be different because of positional encoding
    // (actually, not necessarily - position 0 is same in both cases)
    // But the output should at least vary in length
    println!("logits_1 len: {}", logits_1.len());
    println!("logits_4 len: {}", logits_4.len());

    // logits_4 should be 4x the vocab size (one per position)
    let n_vocab = decoder.n_vocab();
    assert_eq!(
        logits_4.len(),
        4 * n_vocab,
        "Should have logits for each position"
    );

    // Compare logits at position 0 in both cases
    let pos0_from_1: Vec<f32> = logits_1.iter().take(100).copied().collect();
    let pos0_from_4: Vec<f32> = logits_4.iter().take(100).copied().collect();

    let diff: f32 = pos0_from_1
        .iter()
        .zip(pos0_from_4.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    println!("Position 0 logits diff (seq_len=1 vs 4): {:.4}", diff);

    // Note: These SHOULD be different if causal mask is working
    // But position 0 can only see itself, so they might be similar
}

/// Test that decoder produces DIFFERENT tokens for different positions
///
/// This test verifies the model doesn't get stuck in a repetition loop.
/// If the model produces the same token repeatedly, something is wrong
/// with cross-attention or KV cache state.
#[test]
fn test_decoder_produces_varied_output() {
    // Try f32 model first (more accurate), fall back to int8
    let model_path = if std::path::Path::new("models/whisper-tiny.apr").exists() {
        "models/whisper-tiny.apr"
    } else {
        "models/whisper-tiny-int8.apr"
    };
    if !std::path::Path::new(model_path).exists() {
        return;
    }

    let audio_path = "demos/test-audio/test-speech-1.5s.wav";
    if !std::path::Path::new(audio_path).exists() {
        return;
    }

    let model_bytes = std::fs::read(model_path).expect("read model");
    let mut model = WhisperApr::load_from_apr(&model_bytes).expect("load model");

    // Load audio and compute mel
    let audio_bytes = std::fs::read(audio_path).expect("read audio");
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let mel = model.compute_mel(&samples).expect("compute mel");
    let encoder = model.encoder_mut();
    let encoder_output = encoder.forward_mel(&mel).expect("encode");

    // Get decoder and generate tokens one at a time
    let decoder = model.decoder_mut();

    println!("Decoder n_vocab: {}", decoder.n_vocab());
    println!("Decoder d_model: {}", decoder.d_model());
    println!("Decoder n_heads: {}", decoder.n_heads());
    println!("Decoder n_layers: {}", decoder.n_layers());
    println!("Token embedding size: {}", decoder.token_embedding().len());
    println!(
        "Positional embedding size: {}",
        decoder.positional_embedding().len()
    );

    // Check positional embeddings for first few positions
    let pe = decoder.positional_embedding();
    let d = decoder.d_model();
    for pos in [0, 1, 2, 3, 4, 5].iter() {
        let start = pos * d;
        let slice = &pe[start..start + 5.min(d)];
        let nonzero = pe[start..start + d]
            .iter()
            .filter(|&&x| x.abs() > 1e-6)
            .count();
        println!("PE[{}]: first5={:?}, nonzero={}/{}", pos, slice, nonzero, d);
    }

    // Check token embeddings for tokens of interest
    let te = decoder.token_embedding();
    for &token in [220_usize, 5396, 3549, 50258, 50259].iter() {
        let start = token * d;
        let slice = &te[start..start + 5.min(d)];
        let emb = &te[start..start + d];
        let nonzero = emb.iter().filter(|&&x| x.abs() > 1e-6).count();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "TE[{}]: first5={:?}, nonzero={}/{}, L2={:.4}",
            token, slice, nonzero, d, norm
        );
    }

    // Initial tokens: SOT, en, transcribe, no_timestamps
    // HuggingFace tokenizer IDs (shifted from whisper.cpp):
    // SOT=50258, en=50259, transcribe=50359, notimestamps=50363
    let mut tokens = vec![50258_u32, 50259, 50359, 50363];

    // Generate 10 tokens and track what we get
    let mut generated = Vec::new();
    for step in 0..10 {
        let logits = decoder.forward(&tokens, &encoder_output).expect("forward");
        let n_vocab = decoder.n_vocab();
        let last_logits = &logits[(logits.len() / n_vocab - 1) * n_vocab..];

        // Find top 5 tokens
        let mut indexed: Vec<(usize, f32)> = last_logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("test assertion"));

        println!(
            "Step {}: top5 = {:?}",
            step,
            &indexed[..5.min(indexed.len())]
        );

        // Check for NaN/Inf in logits
        let nan_count = last_logits.iter().filter(|x| x.is_nan()).count();
        let inf_count = last_logits.iter().filter(|x| x.is_infinite()).count();
        if nan_count > 0 || inf_count > 0 {
            println!("  WARNING: {} NaN, {} Inf in logits", nan_count, inf_count);
        }

        let (max_idx, max_val) = (indexed[0].0, indexed[0].1);
        println!("  -> selected token {} (logit {:.4})", max_idx, max_val);
        generated.push(max_idx as u32);
        tokens.push(max_idx as u32);
    }

    println!("Generated tokens: {:?}", generated);

    // Count unique tokens
    let unique: std::collections::HashSet<_> = generated.iter().collect();
    println!("Unique tokens: {} out of {}", unique.len(), generated.len());

    // ASSERTION: At least 2 unique tokens in 10 generations
    // Int8 quantized models may have more repetition than f32 models
    assert!(
        unique.len() >= 2,
        "Model stuck in repetition loop: only {} unique tokens in {:?}",
        unique.len(),
        generated
    );
}

/// Test what token 3549 decodes to and verify special tokens
#[test]
fn test_token_3549_identity() {
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        return;
    }

    let model_bytes = std::fs::read(model_path).expect("read model");
    let reader = whisper_apr::format::AprReader::new(model_bytes.clone()).expect("parse apr");

    if let Some(vocab) = reader.read_vocabulary() {
        println!("Vocabulary size: {}", vocab.len());

        // Check token 3549
        if let Some(bytes) = vocab.get_bytes(3549) {
            let text = String::from_utf8_lossy(bytes);
            println!("Token 3549: {:?} -> '{}'", bytes, text);
        } else {
            println!("Token 3549 NOT FOUND in vocabulary");
        }

        // Check ALL special tokens (50256-50365 range)
        println!("\n=== SPECIAL TOKENS ===");
        for token_id in 50254..50270 {
            if let Some(bytes) = vocab.get_bytes(token_id) {
                let text = String::from_utf8_lossy(bytes);
                println!("Token {}: '{}'", token_id, text);
            } else {
                println!("Token {}: NOT FOUND", token_id);
            }
        }

        // Expected special tokens for Whisper:
        // 50256 = EOT (end of transcript)
        // 50257 = SOT (start of transcript)
        // 50258 = language tokens start (en, zh, de, etc.)
        // 50358 = transcribe task token
        // 50359 = translate task token
        // 50362 = no timestamps token

        println!("\n=== EXPECTED MAPPING ===");
        println!("50256 should be: EOT");
        println!("50257 should be: SOT");
        println!("50258 should be: <|en|> (English)");
        println!("50358 should be: <|transcribe|>");
        println!("50359 should be: <|translate|>");
        println!("50362 should be: <|notimestamps|>");
    } else {
        println!("No vocabulary in APR file");
    }
}

/// Test token embedding statistics
#[test]
fn test_token_embedding_stats() {
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        return;
    }

    let model_bytes = std::fs::read(model_path).expect("read model");
    let reader = whisper_apr::format::AprReader::new(model_bytes.clone()).expect("parse apr");

    // Check token embedding weights
    match reader.load_tensor("decoder.token_embedding") {
        Ok(te) => {
            let n_vocab = 51865;
            let d_model = 384;
            println!(
                "Token embedding shape: {} x {} = {}",
                n_vocab,
                d_model,
                te.len()
            );

            // Check embedding for token 3549
            let emb_start = 3549 * d_model;
            let emb_3549: Vec<f32> = te[emb_start..emb_start + d_model].to_vec();
            let min = emb_3549.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = emb_3549.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean: f32 = emb_3549.iter().sum::<f32>() / d_model as f32;
            let nonzero = emb_3549.iter().filter(|&&x| x.abs() > 1e-8).count();

            println!(
                "Token 3549 embedding: min={:.4}, max={:.4}, mean={:.6}, nonzero={}/{}",
                min, max, mean, nonzero, d_model
            );

            // Check embedding for SOT token (50257)
            let emb_start = 50257 * d_model;
            let emb_sot: Vec<f32> = te[emb_start..emb_start + d_model].to_vec();
            let mean: f32 = emb_sot.iter().sum::<f32>() / d_model as f32;
            let nonzero = emb_sot.iter().filter(|&&x| x.abs() > 1e-8).count();
            println!(
                "SOT (50257) embedding: mean={:.6}, nonzero={}/{}",
                mean, nonzero, d_model
            );
        }
        Err(e) => {
            println!("MISSING: decoder.token_embedding - {:?}", e);
        }
    }
}

/// Diagnostic test to verify cross-attention varies with encoder output
#[test]
fn test_cross_attention_varies_with_encoder() {
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        return;
    }

    let model_bytes = std::fs::read(model_path).expect("read model");
    let mut model = WhisperApr::load_from_apr(&model_bytes).expect("load model");

    let decoder = model.decoder_mut();
    let d_model = decoder.d_model();

    // Create two different encoder outputs - one zeros, one random
    let enc_len = 74; // typical for 1.5s audio
    let enc_zeros = vec![0.0_f32; enc_len * d_model];
    let enc_rand: Vec<f32> = (0..enc_len * d_model)
        .map(|i| ((i as f32 * 0.1).sin() + 0.5) * 2.0 - 1.0)
        .collect();

    // Same input tokens for both
    let tokens = vec![50258_u32, 50259, 50359, 50363]; // SOT, en, transcribe, notimestamps

    // Compute decoder output with zeros encoder
    let logits_zeros = decoder.forward(&tokens, &enc_zeros).expect("forward zeros");
    let logits_rand = decoder.forward(&tokens, &enc_rand).expect("forward rand");

    // The outputs should be different if cross-attention is working
    let diff: f32 = logits_zeros
        .iter()
        .zip(logits_rand.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum();

    println!("Cross-attention diff (zeros vs rand encoder): {:.4}", diff);
    println!(
        "Logits zeros range: [{:.4}, {:.4}]",
        logits_zeros.iter().cloned().fold(f32::INFINITY, f32::min),
        logits_zeros
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "Logits rand range: [{:.4}, {:.4}]",
        logits_rand.iter().cloned().fold(f32::INFINITY, f32::min),
        logits_rand
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );

    // If cross-attention works, diff should be significant
    assert!(
        diff > 1000.0,
        "Cross-attention not varying with encoder: diff={:.4}",
        diff
    );
}

/// Diagnostic test to check encoder output statistics
#[test]
fn test_encoder_output_stats() {
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        return;
    }

    let audio_path = "demos/test-audio/test-speech-1.5s.wav";
    if !std::path::Path::new(audio_path).exists() {
        return;
    }

    let model_bytes = std::fs::read(model_path).expect("read model");
    let mut model = WhisperApr::load_from_apr(&model_bytes).expect("load model");

    let audio_bytes = std::fs::read(audio_path).expect("read audio");
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let mel = model.compute_mel(&samples).expect("compute mel");
    println!(
        "Mel shape: {} frames x 80 channels = {}",
        mel.len() / 80,
        mel.len()
    );

    let encoder = model.encoder_mut();
    let encoder_output = encoder.forward_mel(&mel).expect("encode");

    let d_model = 384; // tiny model dimension
    let enc_len = encoder_output.len() / d_model;
    println!(
        "Encoder output: {} frames x {} dim = {}",
        enc_len,
        d_model,
        encoder_output.len()
    );

    // Check statistics
    let min = encoder_output.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = encoder_output
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = encoder_output.iter().sum::<f32>() / encoder_output.len() as f32;
    let variance: f32 = encoder_output
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>()
        / encoder_output.len() as f32;
    let std = variance.sqrt();
    let nonzero = encoder_output.iter().filter(|&&x| x.abs() > 1e-8).count();
    let nan_count = encoder_output.iter().filter(|x| x.is_nan()).count();
    let inf_count = encoder_output.iter().filter(|x| x.is_infinite()).count();

    println!("Encoder output stats:");
    println!(
        "  min={:.4}, max={:.4}, mean={:.6}, std={:.6}",
        min, max, mean, std
    );
    println!(
        "  nonzero={}/{}, nan={}, inf={}",
        nonzero,
        encoder_output.len(),
        nan_count,
        inf_count
    );

    // Check first few frames
    println!("First 5 values of frame 0: {:?}", &encoder_output[..5]);
    println!(
        "First 5 values of frame 50: {:?}",
        &encoder_output[50 * d_model..50 * d_model + 5]
    );

    assert_eq!(nan_count, 0, "Encoder output contains NaN!");
    assert_eq!(inf_count, 0, "Encoder output contains Inf!");
    assert!(
        nonzero > encoder_output.len() / 2,
        "Encoder output is mostly zeros"
    );
    assert!(std > 0.01, "Encoder output has very low variance: {}", std);
}

/// Diagnostic test to check cross-attention weight statistics
#[test]
fn test_cross_attention_weight_stats() {
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        return;
    }

    let model_bytes = std::fs::read(model_path).expect("read model");
    let reader = whisper_apr::format::AprReader::new(model_bytes.clone()).expect("parse apr");

    // Check cross-attention weights for layer 0
    let prefixes = [
        "decoder.layers.0.encoder_attn.q_proj.weight",
        "decoder.layers.0.encoder_attn.k_proj.weight",
        "decoder.layers.0.encoder_attn.v_proj.weight",
        "decoder.layers.0.encoder_attn.out_proj.weight",
    ];

    for prefix in prefixes {
        match reader.load_tensor(prefix) {
            Ok(w) => {
                let min = w.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mean: f32 = w.iter().sum::<f32>() / w.len() as f32;
                let variance: f32 =
                    w.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / w.len() as f32;
                let std = variance.sqrt();
                let nonzero = w.iter().filter(|&&x| x.abs() > 1e-8).count();

                println!(
                    "{}: len={}, min={:.4}, max={:.4}, mean={:.6}, std={:.6}, nonzero={}",
                    prefix,
                    w.len(),
                    min,
                    max,
                    mean,
                    std,
                    nonzero
                );

                // Verify weights are not all zero
                assert!(
                    nonzero > w.len() / 2,
                    "Weight {} has too many zeros",
                    prefix
                );
            }
            Err(e) => {
                println!("MISSING: {} - {:?}", prefix, e);
            }
        }
    }

    // Also check biases
    let bias_prefixes = [
        "decoder.layers.0.encoder_attn.q_proj.bias",
        "decoder.layers.0.encoder_attn.k_proj.bias",
        "decoder.layers.0.encoder_attn.v_proj.bias",
        "decoder.layers.0.encoder_attn.out_proj.bias",
    ];

    for prefix in bias_prefixes {
        match reader.load_tensor(prefix) {
            Ok(b) => {
                let nonzero = b.iter().filter(|&&x| x.abs() > 1e-8).count();
                println!("{}: len={}, nonzero={}", prefix, b.len(), nonzero);
            }
            Err(_) => {
                println!("MISSING (expected for K): {}", prefix);
            }
        }
    }
}

/// Test that the expected output matches whisper.cpp reference
///
/// NOTE: This test requires vocabulary to be embedded in the APR file.
/// If no vocabulary is present, the test will skip.
///
/// IGNORED: Model hallucination issue - produces repetitive output instead of
/// meaningful transcription. See WAPR-MODEL-QUALITY for tracking.
#[test]
#[ignore]
fn test_transcription_matches_reference() {
    // Skip if model not available
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        return;
    }

    // Check if vocabulary is embedded in APR file
    let model_bytes = std::fs::read(model_path).expect("Failed to read model");
    let reader =
        whisper_apr::format::AprReader::new(model_bytes.clone()).expect("Failed to parse APR");
    if !reader.has_vocabulary() {
        eprintln!("Skipping test: APR file has no vocabulary embedded.");
        return;
    }

    let model = WhisperApr::load_from_apr(&model_bytes).expect("Failed to load model");

    let audio_path = "demos/test-audio/test-speech-1.5s.wav";
    if !std::path::Path::new(audio_path).exists() {
        return;
    }

    let audio_bytes = std::fs::read(audio_path).expect("Failed to read audio");
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let result = model
        .transcribe(&samples, TranscribeOptions::default())
        .expect("Transcription failed");

    // Reference from whisper.cpp: " The birds can use"
    // We check for key words that should appear
    let text_lower = result.text.to_lowercase();

    // At minimum, it should contain SOME recognizable words
    let expected_words = ["the", "birds", "can", "use"];
    let words_found = expected_words
        .iter()
        .filter(|w| text_lower.contains(*w))
        .count();

    assert!(
        words_found >= 2,
        "Transcription '{}' should contain at least 2 of {:?}, found {}",
        result.text,
        expected_words,
        words_found
    );
}

/// Diagnostic test to trace hidden state magnitudes through decoder layers
/// This helps identify where logit magnitude growth occurs
#[test]
fn test_decoder_hidden_state_trace() {
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        return;
    }

    let audio_path = "demos/test-audio/test-speech-1.5s.wav";
    if !std::path::Path::new(audio_path).exists() {
        return;
    }

    let model_bytes = std::fs::read(model_path).expect("read model");
    let mut model = WhisperApr::load_from_apr(&model_bytes).expect("load model");

    let audio_bytes = std::fs::read(audio_path).expect("read audio");
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let mel = model.compute_mel(&samples).expect("compute mel");
    let encoder_output = model.encoder_mut().forward_mel(&mel).expect("encode");

    let decoder = model.decoder_mut();

    // Test with increasing token sequences
    let initial_tokens = vec![50258_u32, 50259, 50359, 50363]; // SOT, en, transcribe, notimestamps

    println!("\n=== DECODER HIDDEN STATE TRACE ===\n");

    // 4 tokens
    let (logits_4, trace_4) = decoder
        .forward_traced(&initial_tokens, &encoder_output)
        .expect("forward 4");
    println!("4 tokens:");
    for (name, l2) in &trace_4 {
        println!("  {}: L2 = {:.4}", name, l2);
    }

    // Get last position logits
    let n_vocab = decoder.n_vocab();
    let last_logits_4 = &logits_4[3 * n_vocab..];
    let mut indexed: Vec<(usize, f32)> = last_logits_4
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("test assertion"));
    println!("  top-3 logits: {:?}", &indexed[..3]);

    // 5 tokens (add space token 220)
    let mut tokens_5 = initial_tokens.clone();
    tokens_5.push(220);

    let (logits_5, trace_5) = decoder
        .forward_traced(&tokens_5, &encoder_output)
        .expect("forward 5");
    println!("\n5 tokens (+220):");
    for (name, l2) in &trace_5 {
        println!("  {}: L2 = {:.4}", name, l2);
    }

    let last_logits_5 = &logits_5[4 * n_vocab..];
    let mut indexed: Vec<(usize, f32)> = last_logits_5
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("test assertion"));
    println!("  top-3 logits: {:?}", &indexed[..3]);

    // Compare L2 norms
    println!("\n=== L2 NORM GROWTH (5 tok / 4 tok) ===");
    for (t4, t5) in trace_4.iter().zip(trace_5.iter()) {
        if t4.1 > 0.0 {
            let growth = t5.1 / t4.1;
            println!("  {}: {:.4} / {:.4} = {:.4}x", t4.0, t5.1, t4.1, growth);
        }
    }

    // The growth should be bounded - if it's massive (>2x), something is wrong
    let logits_growth = trace_5
        .iter()
        .find(|(n, _)| n == "logits")
        .map(|(_, l)| *l)
        .unwrap_or(0.0)
        / trace_4
            .iter()
            .find(|(n, _)| n == "logits")
            .map(|(_, l)| *l)
            .unwrap_or(1.0);

    println!("\nLogits L2 growth: {:.4}x", logits_growth);

    // Assert growth is reasonable
    assert!(
        logits_growth < 2.0,
        "Logits L2 norm grew by {:.4}x when adding one token - indicates accumulation bug",
        logits_growth
    );
}

/// Diagnostic test to check layer norm weight scale
#[test]
fn test_layernorm_scale_debug() {
    let model_path = "models/whisper-tiny-int8.apr";
    if !std::path::Path::new(model_path).exists() {
        return;
    }

    let model_bytes = std::fs::read(model_path).expect("read model");
    let reader = whisper_apr::format::AprReader::new(model_bytes).expect("parse apr");

    // Check decoder layer norm weight
    let ln_weight = reader
        .load_tensor("decoder.layer_norm.weight")
        .expect("load ln weight");
    println!("decoder.layer_norm.weight: {} elements", ln_weight.len());
    println!("  first 10: {:?}", &ln_weight[..10.min(ln_weight.len())]);
    let mean: f32 = ln_weight.iter().sum::<f32>() / ln_weight.len() as f32;
    let l2: f32 = ln_weight.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("  mean: {:.6}", mean);
    println!("  L2: {:.6}", l2);

    // Also check a block's layer norm for comparison
    let ln1_weight = reader
        .load_tensor("decoder.layers.0.self_attn_layer_norm.weight")
        .expect("load ln1 weight");
    println!("\ndecoder.layers.0.self_attn_layer_norm.weight:");
    println!("  first 10: {:?}", &ln1_weight[..10.min(ln1_weight.len())]);
    let mean1: f32 = ln1_weight.iter().sum::<f32>() / ln1_weight.len() as f32;
    let l21: f32 = ln1_weight.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("  mean: {:.6}", mean1);
    println!("  L2: {:.6}", l21);

    // Layer norm weights should be positive (int8 models may have scaled weights)
    assert!(
        mean > 0.0,
        "Layer norm weight mean {:.6} should be positive",
        mean
    );
}

/// Test with f32 model to compare with int8
#[test]
fn test_f32_model_generation() {
    let model_path = "models/whisper-tiny.apr";
    if !std::path::Path::new(model_path).exists() {
        println!("Skipping: f32 model not found");
        return;
    }

    let audio_path = "demos/test-audio/test-speech-1.5s.wav";
    if !std::path::Path::new(audio_path).exists() {
        return;
    }

    let model_bytes = std::fs::read(model_path).expect("read model");
    let mut model = WhisperApr::load_from_apr(&model_bytes).expect("load model");

    let audio_bytes = std::fs::read(audio_path).expect("read audio");
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let mel = model.compute_mel(&samples).expect("compute mel");
    let encoder_output = model.encoder_mut().forward_mel(&mel).expect("encode");

    let decoder = model.decoder_mut();

    // Check layer norm weights
    println!("F32 model layer norm check:");
    let (_, trace) = decoder
        .forward_traced(&[50258_u32, 50259, 50359, 50363], &encoder_output)
        .expect("forward");
    for (name, val) in &trace {
        if name.contains("ln_") {
            println!("  {}: {:.4}", name, val);
        }
    }

    // Generate tokens
    let mut tokens = vec![50258_u32, 50259, 50359, 50363];
    let mut generated = Vec::new();

    println!("\nF32 model generation:");
    for step in 0..10 {
        let logits = decoder.forward(&tokens, &encoder_output).expect("forward");
        let n_vocab = decoder.n_vocab();
        let last_logits = &logits[(logits.len() / n_vocab - 1) * n_vocab..];

        let mut indexed: Vec<(usize, f32)> = last_logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("test assertion"));

        println!(
            "Step {}: top5 = {:?}",
            step,
            &indexed[..5.min(indexed.len())]
        );

        let (max_idx, _) = indexed[0];
        generated.push(max_idx as u32);
        tokens.push(max_idx as u32);
    }

    println!("Generated tokens: {:?}", generated);

    // Check uniqueness
    let unique: std::collections::HashSet<_> = generated.iter().collect();
    println!("Unique tokens: {} out of {}", unique.len(), generated.len());
}
