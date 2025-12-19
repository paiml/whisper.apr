//! Trace encoder layer by layer to find divergence point
//!
//! Compare hidden states at each layer with HuggingFace

use std::fs::File;
use std::io::Read;
use whisper_apr::WhisperApr;

fn load_npy_f32(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    if &buf[0..6] != b"\x93NUMPY" {
        return Err("Not a numpy file".into());
    }

    let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let data_start = 10 + header_len;
    let header = std::str::from_utf8(&buf[10..data_start])?;

    let is_f64 = header.contains("float64") || header.contains("<f8");
    let is_f32 = header.contains("float32") || header.contains("<f4");

    let data = &buf[data_start..];

    if is_f64 {
        let f64_values: Vec<f64> = data
            .chunks(8)
            .map(|chunk| {
                f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();
        Ok(f64_values.iter().map(|&x| x as f32).collect())
    } else if is_f32 {
        Ok(data
            .chunks(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    } else {
        Err(format!("Unknown dtype: {}", header).into())
    }
}

fn stats(x: &[f32]) -> (f32, f32, f32, f32) {
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let variance: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
    let std = variance.sqrt();
    let l2: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    (
        mean,
        std,
        l2,
        x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs())),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ENCODER LAYER-BY-LAYER TRACE ===\n");

    // Load HuggingFace outputs
    let hf_encoder = load_npy_f32("/tmp/hf_encoder_output.npy")?;
    println!("HF encoder output: {} values", hf_encoder.len());

    // Load our model
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let mut model = WhisperApr::load_from_apr(&model_bytes)?;

    // Check encoder configuration
    let encoder = model.encoder_mut();
    println!("\nEncoder config:");
    println!("  n_layers: {}", encoder.n_layers());
    println!("  d_model:  {}", encoder.d_model());
    println!("  n_heads:  {}", encoder.n_heads());
    println!("  max_len:  {}", encoder.max_len());
    println!("  n_mels:   {}", encoder.n_mels());

    // Check positional embeddings
    let pe = encoder.positional_embedding();
    let (pe_mean, pe_std, pe_l2, pe_max) = stats(pe);
    println!("\nPositional embeddings ({} values):", pe.len());
    println!(
        "  mean: {:.4}, std: {:.4}, L2: {:.4}, max_abs: {:.4}",
        pe_mean, pe_std, pe_l2, pe_max
    );

    // Check conv frontend weights
    let conv1_weight = &encoder.conv_frontend().conv1.weight;
    let conv1_bias = &encoder.conv_frontend().conv1.bias;
    let conv2_weight = &encoder.conv_frontend().conv2.weight;
    let conv2_bias = &encoder.conv_frontend().conv2.bias;

    println!("\nConv1 weights ({} values):", conv1_weight.len());
    let (w1_mean, w1_std, w1_l2, w1_max) = stats(conv1_weight);
    println!(
        "  mean: {:.6}, std: {:.4}, L2: {:.4}, max_abs: {:.4}",
        w1_mean, w1_std, w1_l2, w1_max
    );
    let (b1_mean, b1_std, b1_l2, b1_max) = stats(conv1_bias);
    println!(
        "Conv1 bias: mean: {:.6}, std: {:.4}, L2: {:.4}, max_abs: {:.4}",
        b1_mean, b1_std, b1_l2, b1_max
    );

    println!("\nConv2 weights ({} values):", conv2_weight.len());
    let (w2_mean, w2_std, w2_l2, w2_max) = stats(conv2_weight);
    println!(
        "  mean: {:.6}, std: {:.4}, L2: {:.4}, max_abs: {:.4}",
        w2_mean, w2_std, w2_l2, w2_max
    );
    let (b2_mean, b2_std, b2_l2, b2_max) = stats(conv2_bias);
    println!(
        "Conv2 bias: mean: {:.6}, std: {:.4}, L2: {:.4}, max_abs: {:.4}",
        b2_mean, b2_std, b2_l2, b2_max
    );

    // Check encoder layer norms
    println!("\nEncoder layer norms (block.ln1 and block.ln2):");
    for (i, block) in encoder.blocks().iter().enumerate() {
        let ln1_w = &block.ln1.weight;
        let ln1_b = &block.ln1.bias;
        let ln2_w = &block.ln2.weight;
        let ln2_b = &block.ln2.bias;

        let (ln1w_mean, _, _, ln1w_max) = stats(ln1_w);
        let (ln1b_mean, _, _, ln1b_max) = stats(ln1_b);
        let (ln2w_mean, _, _, ln2w_max) = stats(ln2_w);
        let (ln2b_mean, _, _, ln2b_max) = stats(ln2_b);

        println!(
            "  Block {}: ln1.w mean={:.4} max={:.4}, ln1.b mean={:.4} max={:.4}",
            i, ln1w_mean, ln1w_max, ln1b_mean, ln1b_max
        );
        println!(
            "           ln2.w mean={:.4} max={:.4}, ln2.b mean={:.4} max={:.4}",
            ln2w_mean, ln2w_max, ln2b_mean, ln2b_max
        );
    }

    // Check ln_post
    let ln_post_w = &encoder.ln_post().weight;
    let ln_post_b = &encoder.ln_post().bias;
    let (lnpw_mean, lnpw_std, lnpw_l2, lnpw_max) = stats(ln_post_w);
    let (lnpb_mean, lnpb_std, lnpb_l2, lnpb_max) = stats(ln_post_b);
    println!("\nEncoder ln_post:");
    println!(
        "  weight: mean={:.4}, std={:.4}, L2={:.4}, max={:.4}",
        lnpw_mean, lnpw_std, lnpw_l2, lnpw_max
    );
    println!(
        "  bias:   mean={:.4}, std={:.4}, L2={:.4}, max={:.4}",
        lnpb_mean, lnpb_std, lnpb_l2, lnpb_max
    );

    // Now check HF encoder output stats
    let (hf_mean, hf_std, hf_l2, hf_max) = stats(&hf_encoder);
    println!("\n=== HF Encoder Output Stats ===");
    println!("  shape:   {} positions × 384", hf_encoder.len() / 384);
    println!("  mean:    {:.6}", hf_mean);
    println!("  std:     {:.4}", hf_std);
    println!("  L2:      {:.4}", hf_l2);
    println!("  max_abs: {:.4}", hf_max);

    // Run our encoder and compare
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let mel = model.compute_mel(&samples)?;
    println!("\nMel spectrogram: {} frames × 80", mel.len() / 80);
    let (mel_mean, mel_std, mel_l2, mel_max) = stats(&mel);
    println!(
        "  mean={:.6}, std={:.4}, L2={:.4}, max={:.4}",
        mel_mean, mel_std, mel_l2, mel_max
    );

    let our_encoder = model.encode(&mel)?;
    let (our_mean, our_std, our_l2, our_max) = stats(&our_encoder);
    println!("\n=== Our Encoder Output Stats ===");
    println!("  shape:   {} positions × 384", our_encoder.len() / 384);
    println!("  mean:    {:.6}", our_mean);
    println!("  std:     {:.4}", our_std);
    println!("  L2:      {:.4}", our_l2);
    println!("  max_abs: {:.4}", our_max);

    // Compare first position (both should have same processing for first few positions)
    println!("\n=== First Position Comparison (first 10 dims) ===");
    println!(
        "HF:  {:?}",
        &hf_encoder[0..10]
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "Our: {:?}",
        &our_encoder[0..10]
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    // Compute per-position L2 norms
    println!("\n=== Per-Position L2 Norms ===");
    let hf_positions = hf_encoder.len() / 384;
    let our_positions = our_encoder.len() / 384;

    println!("HF (first 5 positions):");
    for pos in 0..5.min(hf_positions) {
        let start = pos * 384;
        let l2: f32 = hf_encoder[start..start + 384]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        println!("  pos {}: L2 = {:.4}", pos, l2);
    }

    println!("Our (first 5 positions):");
    for pos in 0..5.min(our_positions) {
        let start = pos * 384;
        let l2: f32 = our_encoder[start..start + 384]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        println!("  pos {}: L2 = {:.4}", pos, l2);
    }

    Ok(())
}
