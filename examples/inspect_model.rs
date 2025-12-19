//! Model Inspection Tool
//!
//! Inspect whisper.apr model files and compare with whisper.cpp ground truth.
//!
//! Usage:
//!   cargo run --example inspect_model -- models/whisper-tiny-fb.apr
//!   cargo run --example inspect_model -- models/whisper-tiny-fb.apr --compare-wcpp

use std::path::Path;
use whisper_apr::format::AprReader;

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
    let reader = AprReader::new(model_bytes.clone())?;

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

fn print_human(reader: &AprReader, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let header = &reader.header;

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
    println!("  Version:        {}", header.version);
    println!(
        "  Model Type:     {} ({})",
        header.model_type,
        model_type_name(header.model_type)
    );
    println!("  Quantization:   {:?}", header.quantization);
    println!("  Compressed:     {}", header.compressed);
    println!("  Tensors:        {}", header.n_tensors);

    println!("\n=== Embedded Data ===");
    println!("  Has Vocabulary: {}", header.has_vocab);
    println!("  Has Filterbank: {}", header.has_filterbank);

    if header.has_vocab {
        if let Some(vocab) = reader.read_vocabulary() {
            println!("  Vocabulary Size: {} tokens", vocab.len());
        }
    }

    if header.has_filterbank {
        if let Some(fb) = reader.read_mel_filterbank() {
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

    println!("\n=== Model Architecture ===");
    println!("  n_vocab:        {}", header.n_vocab);
    println!("  n_mels:         {}", header.n_mels);
    println!("  n_audio_ctx:    {}", header.n_audio_ctx);
    println!("  n_audio_state:  {}", header.n_audio_state);
    println!("  n_audio_head:   {}", header.n_audio_head);
    println!("  n_audio_layer:  {}", header.n_audio_layer);
    println!("  n_text_ctx:     {}", header.n_text_ctx);
    println!("  n_text_state:   {}", header.n_text_state);
    println!("  n_text_head:    {}", header.n_text_head);
    println!("  n_text_layer:   {}", header.n_text_layer);

    println!("\n=== Tensor Summary ===");
    let mut total_params = 0usize;
    let mut encoder_params = 0usize;
    let mut decoder_params = 0usize;

    for tensor in &reader.tensors {
        let params: usize = tensor.shape.iter().map(|&d| d as usize).product();
        total_params += params;
        if tensor.name.starts_with("encoder") {
            encoder_params += params;
        } else if tensor.name.starts_with("decoder") {
            decoder_params += params;
        }
    }

    println!("  Total Tensors:  {}", reader.tensors.len());
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
    for (i, tensor) in reader.tensors.iter().take(10).enumerate() {
        println!("  {:2}. {} {:?}", i, tensor.name, tensor.shape);
    }

    Ok(())
}

fn print_json(reader: &AprReader) -> Result<(), Box<dyn std::error::Error>> {
    let header = &reader.header;

    println!("{{");
    println!("  \"version\": {},", header.version);
    println!("  \"model_type\": {},", header.model_type);
    println!(
        "  \"model_type_name\": \"{}\",",
        model_type_name(header.model_type)
    );
    println!("  \"quantization\": \"{:?}\",", header.quantization);
    println!("  \"n_tensors\": {},", header.n_tensors);
    println!("  \"has_vocab\": {},", header.has_vocab);
    println!("  \"has_filterbank\": {},", header.has_filterbank);
    println!("  \"n_vocab\": {},", header.n_vocab);
    println!("  \"n_mels\": {},", header.n_mels);
    println!("  \"n_audio_state\": {},", header.n_audio_state);
    println!("  \"n_audio_layer\": {},", header.n_audio_layer);
    println!("  \"n_text_state\": {},", header.n_text_state);
    println!("  \"n_text_layer\": {}", header.n_text_layer);
    println!("}}");

    Ok(())
}

fn compare_with_whisper_cpp(reader: &AprReader) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║   COMPARISON WITH WHISPER.CPP                              ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Compare filterbank
    let wcpp_fb_path = "/tmp/whisper_cpp_filterbank.bin";
    if !Path::new(wcpp_fb_path).exists() {
        println!("⚠️  whisper.cpp filterbank not found at {}", wcpp_fb_path);
        println!("   Run: python3 tools/extract_filterbank.py ../whisper.cpp/models/ggml-tiny.bin");
        return Ok(());
    }

    let wcpp_fb_bytes = std::fs::read(wcpp_fb_path)?;
    let wcpp_fb: Vec<f32> = wcpp_fb_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    println!("=== Filterbank Comparison ===");

    if let Some(our_fb) = reader.read_mel_filterbank() {
        let cosine = cosine_similarity(&wcpp_fb, &our_fb.data);
        let status = if cosine > 0.9999 {
            "✓ MATCH"
        } else {
            "✗ DIFFER"
        };

        println!("  whisper.cpp: {} values", wcpp_fb.len());
        println!("  ours:        {} values", our_fb.data.len());
        println!("  Cosine Sim:  {:.10}", cosine);
        println!("  Status:      {}", status);
    } else {
        println!("  ⚠️  No filterbank embedded in model");
    }

    // Compare expected architecture
    println!("\n=== Architecture Comparison (tiny model) ===");
    let header = &reader.header;
    let checks = [
        ("n_vocab", header.n_vocab, 51865),
        ("n_mels", header.n_mels, 80),
        ("n_audio_ctx", header.n_audio_ctx, 1500),
        ("n_audio_state", header.n_audio_state, 384),
        ("n_audio_head", header.n_audio_head, 6),
        ("n_audio_layer", header.n_audio_layer, 4),
        ("n_text_ctx", header.n_text_ctx, 448),
        ("n_text_state", header.n_text_state, 384),
        ("n_text_head", header.n_text_head, 6),
        ("n_text_layer", header.n_text_layer, 4),
    ];

    let mut all_match = true;
    for (name, actual, expected) in checks {
        let status = if actual == expected {
            "✓"
        } else {
            all_match = false;
            "✗"
        };
        println!("  {} {}: {} (expected {})", status, name, actual, expected);
    }

    if all_match {
        println!("\n  ✓ Architecture matches whisper-tiny");
    } else {
        println!("\n  ✗ Architecture mismatch!");
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

fn model_type_name(t: u8) -> &'static str {
    match t {
        0 => "tiny",
        1 => "tiny.en",
        2 => "base",
        3 => "base.en",
        4 => "small",
        5 => "small.en",
        6 => "medium",
        7 => "medium.en",
        8 => "large",
        9 => "large-v1",
        10 => "large-v2",
        11 => "large-v3",
        _ => "unknown",
    }
}
