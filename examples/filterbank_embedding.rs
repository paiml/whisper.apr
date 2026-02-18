//! Filterbank Embedding Example
//!
//! Demonstrates how to read embedded mel filterbank from .apr files and
//! compare with computed filterbank to verify slaney normalization.
//!
//! Run with: `cargo run --example filterbank_embedding`
//!
//! With model file: `cargo run --example filterbank_embedding -- models/whisper-tiny.apr`

use whisper_apr::audio::MelFilterbank;
use whisper_apr::format::{
    AprV2ReaderRef, AprV2Writer, MelFilterbankData, build_whisper_metadata, metadata_to_model_config,
};
use whisper_apr::model::ModelConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   MEL FILTERBANK EMBEDDING EXAMPLE                         ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Check if model file provided
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        // Read existing .apr file
        inspect_apr_file(&args[1])?;
    } else {
        // Demo mode: create and test filterbank embedding
        demo_filterbank_embedding()?;
    }

    Ok(())
}

fn inspect_apr_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Inspecting: {} ===\n", path);

    let data = std::fs::read(path)?;
    let reader = AprV2ReaderRef::from_bytes(&data)?;
    let config = metadata_to_model_config(reader.metadata());

    println!("Model Information:");
    println!(
        "  Type: {:?} ({})",
        config.model_type,
        model_type_name(config.model_type)
    );
    println!("  Tensors: {}", reader.header().tensor_count);
    println!("  n_mels: {}", config.n_mels);
    println!();

    println!("Embedded Data:");
    println!("  Has vocabulary: {}", reader.get_tensor("__vocab__").is_some());
    println!("  Has filterbank: {}", reader.get_tensor("__mel_filters__").is_some());

    if let Some(vocab_data) = reader.get_tensor_data("__vocab__") {
        if let Some(vocab) = whisper_apr::tokenizer::Vocabulary::from_bytes(vocab_data) {
            println!("  Vocabulary size: {} tokens", vocab.len());
        }
    }

    if let Some(fb_data) = reader.get_tensor_data("__mel_filters__") {
        if let Ok(fb) = MelFilterbankData::from_bytes(fb_data) {
            println!("\n=== Filterbank Analysis ===");
            println!("  Dimensions: {} x {}", fb.n_mels, fb.n_freqs);
            println!("  Total values: {}", fb.data.len());

            // Check slaney normalization (row sums should be ~0.025)
            let n_freqs = fb.n_freqs as usize;
            println!("\n  Row Sums (should be ~0.025 for slaney normalization):");
            for row in 0..5.min(fb.n_mels as usize) {
                let row_sum: f32 = fb.data[row * n_freqs..(row + 1) * n_freqs].iter().sum();
                println!("    Row {}: {:.6}", row, row_sum);
            }

            // Compare with computed filterbank
            println!("\n=== Comparison with Computed Filterbank ===");
            let computed = MelFilterbank::new(fb.n_mels as usize, 400, 16000);
            let cosine_sim = cosine_similarity(&fb.data, computed.filters());
            println!("  Cosine similarity: {:.6}", cosine_sim);

            if cosine_sim > 0.99 {
                println!("  Status: MATCH - Filterbanks are nearly identical");
            } else {
                println!("  Status: DIFFERENT - Using embedded filterbank is critical!");
                println!("  (This is expected - OpenAI's filterbank uses slaney normalization)");
            }

            // Create MelFilterbank from embedded data
            let mel = MelFilterbank::from_apr_data(fb, 16000);
            println!(
                "\n  Created MelFilterbank: {} mels, {} FFT, {} Hz",
                mel.n_mels(),
                mel.n_fft(),
                mel.sample_rate()
            );
        }
    } else {
        println!("\n  No filterbank embedded in this model");
        println!("      Consider reconverting with the latest converter.");
    }

    Ok(())
}

fn demo_filterbank_embedding() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Demo: Filterbank Embedding ===\n");
    println!("(No model file provided, running demo mode)\n");

    // Create a synthetic "OpenAI-like" filterbank with slaney normalization
    let n_mels = 80usize;
    let n_freqs = 201usize;

    println!("Creating slaney-normalized filterbank (80x201)...");
    let filterbank_data = create_slaney_filterbank(n_mels, n_freqs);

    // Verify normalization
    let row0_sum: f32 = filterbank_data[0..n_freqs].iter().sum();
    println!("  Row 0 sum: {:.6} (target: ~0.025)", row0_sum);

    // Create MelFilterbankData
    let fb_data = MelFilterbankData::mel_80(filterbank_data.clone());
    println!(
        "  Created MelFilterbankData: {}x{}",
        fb_data.n_mels, fb_data.n_freqs
    );

    // Create a minimal .apr file with filterbank
    println!("\nCreating .apr file with embedded filterbank...");
    let config = ModelConfig::tiny();
    let meta = build_whisper_metadata(&config, "test");
    let mut writer = AprV2Writer::new(meta);

    // Add a dummy tensor
    writer.add_f32_tensor("dummy.weight", vec![10, 10], &[0.0; 100]);

    // Embed the filterbank as a tensor
    let fb_bytes = fb_data.to_bytes();
    writer.add_tensor(
        "__mel_filters__",
        whisper_apr::format::TensorDType::F32,
        vec![n_mels, n_freqs],
        fb_bytes,
    );

    let bytes = writer.write()?;
    println!("  File size: {} bytes", bytes.len());

    // Read it back
    println!("\nReading back filterbank...");
    let reader = AprV2ReaderRef::from_bytes(&bytes)?;

    if let Some(raw_data) = reader.get_tensor_data("__mel_filters__") {
        if let Ok(fb) = MelFilterbankData::from_bytes(raw_data) {
            println!("  Recovered: {}x{}", fb.n_mels, fb.n_freqs);

            // Verify data integrity
            let orig_sum: f32 = filterbank_data.iter().sum();
            let read_sum: f32 = fb.data.iter().sum();
            println!("  Original sum: {:.6}", orig_sum);
            println!("  Read sum:     {:.6}", read_sum);
            println!("  Match: {}", (orig_sum - read_sum).abs() < 1e-6);

            // Create MelFilterbank for use
            let mel = MelFilterbank::from_apr_data(fb, 16000);
            println!(
                "\n  Ready to use: MelFilterbank with {} mel bands",
                mel.n_mels()
            );

            // Compare with standard computed filterbank
            let computed = MelFilterbank::new(n_mels, 400, 16000);
            let cosine_sim = cosine_similarity(mel.filters(), computed.filters());
            println!("\n=== Filterbank Comparison ===");
            println!(
                "  Embedded (slaney) vs Computed: {:.4} cosine similarity",
                cosine_sim
            );

            if cosine_sim < 0.99 {
                println!("  DIFFERENT! This is why embedded filterbank matters.");
                println!("     The model was trained with slaney-normalized filterbank.");
                println!("     Computing from scratch produces different values.");
            }
        }
    }

    println!("\n=== Usage Example ===");
    println!("```rust");
    println!("let reader = AprV2ReaderRef::from_bytes(&model_bytes)?;");
    println!("let mel = reader.get_tensor_data(\"__mel_filters__\")");
    println!("    .and_then(|d| MelFilterbankData::from_bytes(d).ok())");
    println!("    .map(|fb| MelFilterbank::from_apr_data(fb, 16000))");
    println!("    .unwrap_or_else(|| MelFilterbank::new(80, 400, 16000));");
    println!("```");

    Ok(())
}

/// Create a slaney-normalized filterbank (simplified version)
fn create_slaney_filterbank(n_mels: usize, n_freqs: usize) -> Vec<f32> {
    let mut filters = vec![0.0f32; n_mels * n_freqs];
    let sample_rate = 16000.0f32;
    let n_fft = 400usize;

    // Mel scale boundaries
    let f_min = 0.0f32;
    let f_max = sample_rate / 2.0;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Create mel points
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32))
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&f| ((n_fft as f32 + 1.0) * f / sample_rate).floor() as usize)
        .collect();

    // Create triangular filters with slaney normalization
    for m in 0..n_mels {
        let f_m_minus = bin_points[m];
        let f_m = bin_points[m + 1];
        let f_m_plus = bin_points[m + 2];

        // Slaney normalization: scale by 2 / (f_high - f_low)
        let bandwidth = hz_points[m + 2] - hz_points[m];
        let norm = if bandwidth > 0.0 {
            2.0 / bandwidth
        } else {
            1.0
        };

        // Rising slope
        for k in f_m_minus..f_m {
            if k < n_freqs && f_m > f_m_minus {
                let slope = (k - f_m_minus) as f32 / (f_m - f_m_minus) as f32;
                filters[m * n_freqs + k] = slope * norm;
            }
        }

        // Falling slope
        for k in f_m..f_m_plus {
            if k < n_freqs && f_m_plus > f_m {
                let slope = (f_m_plus - k) as f32 / (f_m_plus - f_m) as f32;
                filters[m * n_freqs + k] = slope * norm;
            }
        }
    }

    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        norm_a += (*x as f64).powi(2);
        norm_b += (*y as f64).powi(2);
    }

    (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
}

fn model_type_name(t: whisper_apr::ModelType) -> &'static str {
    match t {
        whisper_apr::ModelType::Tiny => "tiny",
        whisper_apr::ModelType::TinyEn => "tiny.en",
        whisper_apr::ModelType::Base => "base",
        whisper_apr::ModelType::BaseEn => "base.en",
        whisper_apr::ModelType::Small => "small",
        whisper_apr::ModelType::SmallEn => "small.en",
        whisper_apr::ModelType::Medium => "medium",
        whisper_apr::ModelType::MediumEn => "medium.en",
        whisper_apr::ModelType::Large => "large",
        whisper_apr::ModelType::LargeV1 => "large-v1",
        whisper_apr::ModelType::LargeV2 => "large-v2",
        whisper_apr::ModelType::LargeV3 => "large-v3",
        whisper_apr::ModelType::LargeV3Turbo => "large-v3-turbo",
    }
}
