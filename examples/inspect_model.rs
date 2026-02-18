//! Model Inspection Tool
//!
//! Inspect whisper.apr model files and compare with whisper.cpp ground truth.
//!
//! Usage:
//!   cargo run --example inspect_model -- models/whisper-tiny-fb.apr
//!   cargo run --example inspect_model -- models/whisper-tiny-fb.apr --compare-wcpp

use std::path::Path;
use whisper_apr::format::{AprV2ReaderRef, MelFilterbankData, metadata_to_model_config};
use whisper_apr::model::ModelConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("Usage: {} <model.apr> [--compare-wcpp] [--json]", args[0]);
        println!("\nOptions:");
        println!("  --compare-wcpp  Compare with whisper.cpp ground truth");
        println!("  --json          Output in JSON format");
        return Ok(());
    }

    let model_path = &args[1];
    let compare_wcpp = args.iter().any(|a| a == "--compare-wcpp");
    let json_output = args.iter().any(|a| a == "--json");

    let model_bytes = std::fs::read(model_path)?;
    let reader = AprV2ReaderRef::from_bytes(&model_bytes)?;

    if json_output {
        print_json(&reader)?;
    } else {
        print_human(&reader, model_path)?;
    }

    if compare_wcpp {
        compare_with_whisper_cpp(&reader)?;
    }

    Ok(())
}

fn print_human(reader: &AprV2ReaderRef<'_>, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let header = reader.header();
    let config = metadata_to_model_config(reader.metadata());

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   WHISPER.APR MODEL INSPECTION                             ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("=== File Information ===");
    println!("  Path:           {}", path);
    println!(
        "  Size:           {} bytes ({:.2} MB)",
        std::fs::metadata(path)?.len(),
        std::fs::metadata(path)?.len() as f64 / 1_000_000.0
    );

    println!("\n=== Header ===");
    println!("  Version:        {}.{}", header.version.0, header.version.1);
    println!(
        "  Model Type:     {:?} ({})",
        config.model_type,
        model_type_name(&config)
    );
    println!("  Tensors:        {}", header.tensor_count);

    let has_vocab = reader.get_tensor("__vocab__").is_some();
    let has_filterbank = reader.get_tensor("__mel_filters__").is_some();

    println!("\n=== Embedded Data ===");
    println!("  Has Vocabulary: {}", has_vocab);
    println!("  Has Filterbank: {}", has_filterbank);

    if has_vocab {
        if let Some(vocab_data) = reader.get_tensor_data("__vocab__") {
            if let Some(vocab) = whisper_apr::tokenizer::Vocabulary::from_bytes(vocab_data) {
                println!("  Vocabulary Size: {} tokens", vocab.len());
            }
        }
    }

    if has_filterbank {
        if let Some(fb_data) = reader.get_tensor_data("__mel_filters__") {
            if let Ok(fb) = MelFilterbankData::from_bytes(fb_data) {
                println!(
                    "  Filterbank:     {}x{} ({} values)",
                    fb.n_mels,
                    fb.n_freqs,
                    fb.data.len()
                );
                let row_sum: f32 = fb.data[0..fb.n_freqs as usize].iter().sum();
                println!("  Row 0 Sum:      {:.6} (slaney: ~0.025)", row_sum);
            }
        }
    }

    println!("\n=== Model Architecture ===");
    println!("  n_vocab:        {}", config.n_vocab);
    println!("  n_mels:         {}", config.n_mels);
    println!("  n_audio_ctx:    {}", config.n_audio_ctx);
    println!("  n_audio_state:  {}", config.n_audio_state);
    println!("  n_audio_head:   {}", config.n_audio_head);
    println!("  n_audio_layer:  {}", config.n_audio_layer);
    println!("  n_text_ctx:     {}", config.n_text_ctx);
    println!("  n_text_state:   {}", config.n_text_state);
    println!("  n_text_head:    {}", config.n_text_head);
    println!("  n_text_layer:   {}", config.n_text_layer);

    println!("\n=== Tensor Summary ===");
    let mut total_params = 0usize;
    let mut encoder_params = 0usize;
    let mut decoder_params = 0usize;

    let tensor_names = reader.tensor_names();
    for name in &tensor_names {
        let entry = reader.get_tensor(name).unwrap();
        let params: usize = entry.shape.iter().product();
        total_params += params;
        if name.starts_with("encoder") {
            encoder_params += params;
        } else if name.starts_with("decoder") {
            decoder_params += params;
        }
    }

    println!("  Total Tensors:  {}", tensor_names.len());
    println!(
        "  Total Params:   {} ({:.2}M)",
        total_params,
        total_params as f64 / 1_000_000.0
    );
    println!(
        "  Encoder Params: {} ({:.2}M)",
        encoder_params,
        encoder_params as f64 / 1_000_000.0
    );
    println!(
        "  Decoder Params: {} ({:.2}M)",
        decoder_params,
        decoder_params as f64 / 1_000_000.0
    );

    // Show first few tensors
    println!("\n=== First 10 Tensors ===");
    for (i, name) in tensor_names.iter().take(10).enumerate() {
        let entry = reader.get_tensor(name).unwrap();
        println!("  {:2}. {} {:?}", i, name, entry.shape);
    }

    Ok(())
}

fn print_json(reader: &AprV2ReaderRef<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let header = reader.header();
    let config = metadata_to_model_config(reader.metadata());

    println!("{{");
    println!("  \"version\": \"{}.{}\",", header.version.0, header.version.1);
    println!("  \"model_type\": \"{:?}\",", config.model_type);
    println!(
        "  \"model_type_name\": \"{}\",",
        model_type_name(&config)
    );
    println!("  \"n_tensors\": {},", header.tensor_count);
    println!("  \"has_vocab\": {},", reader.get_tensor("__vocab__").is_some());
    println!("  \"has_filterbank\": {},", reader.get_tensor("__mel_filters__").is_some());
    println!("  \"n_vocab\": {},", config.n_vocab);
    println!("  \"n_mels\": {},", config.n_mels);
    println!("  \"n_audio_state\": {},", config.n_audio_state);
    println!("  \"n_audio_layer\": {},", config.n_audio_layer);
    println!("  \"n_text_state\": {},", config.n_text_state);
    println!("  \"n_text_layer\": {}", config.n_text_layer);
    println!("}}");

    Ok(())
}

fn compare_with_whisper_cpp(reader: &AprV2ReaderRef<'_>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║   COMPARISON WITH WHISPER.CPP                              ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Compare filterbank
    let wcpp_fb_path = "/tmp/whisper_cpp_filterbank.bin";
    if !Path::new(wcpp_fb_path).exists() {
        println!("whisper.cpp filterbank not found at {}", wcpp_fb_path);
        println!("   Run: python3 tools/extract_filterbank.py ../whisper.cpp/models/ggml-tiny.bin");
        return Ok(());
    }

    let wcpp_fb_bytes = std::fs::read(wcpp_fb_path)?;
    let wcpp_fb: Vec<f32> = wcpp_fb_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    println!("=== Filterbank Comparison ===");

    if let Some(fb_data) = reader.get_tensor_data("__mel_filters__") {
        if let Ok(our_fb) = MelFilterbankData::from_bytes(fb_data) {
            let cosine = cosine_similarity(&wcpp_fb, &our_fb.data);
            let status = if cosine > 0.9999 {
                "MATCH"
            } else {
                "DIFFER"
            };

            println!("  whisper.cpp: {} values", wcpp_fb.len());
            println!("  ours:        {} values", our_fb.data.len());
            println!("  Cosine Sim:  {:.10}", cosine);
            println!("  Status:      {}", status);
        }
    } else {
        println!("  No filterbank embedded in model");
    }

    // Compare expected architecture
    println!("\n=== Architecture Comparison (tiny model) ===");
    let config = metadata_to_model_config(reader.metadata());
    let checks = [
        ("n_vocab", config.n_vocab, 51865),
        ("n_mels", config.n_mels, 80),
        ("n_audio_ctx", config.n_audio_ctx, 1500),
        ("n_audio_state", config.n_audio_state, 384),
        ("n_audio_head", config.n_audio_head, 6),
        ("n_audio_layer", config.n_audio_layer, 4),
        ("n_text_ctx", config.n_text_ctx, 448),
        ("n_text_state", config.n_text_state, 384),
        ("n_text_head", config.n_text_head, 6),
        ("n_text_layer", config.n_text_layer, 4),
    ];

    let mut all_match = true;
    for (name, actual, expected) in checks {
        let status = if actual == expected {
            "OK"
        } else {
            all_match = false;
            "MISMATCH"
        };
        println!("  {} {}: {} (expected {})", status, name, actual, expected);
    }

    if all_match {
        println!("\n  Architecture matches whisper-tiny");
    } else {
        println!("\n  Architecture mismatch!");
    }

    Ok(())
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

fn model_type_name(config: &ModelConfig) -> &'static str {
    match config.model_type {
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
