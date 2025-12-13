//! Language detection example
//!
//! Demonstrates the language detection API.
//!
//! Run with: `cargo run --example language_detection`

use whisper_apr::detection::{is_supported, language_name, LanguageDetector, LanguageProbs};

fn main() {
    println!("=== Whisper.apr Language Detection Example ===\n");

    // Show supported languages
    println!("Sample supported languages:");
    for code in ["en", "es", "fr", "de", "zh", "ja", "ko", "ru", "ar", "hi"] {
        if is_supported(code) {
            let name = language_name(code).unwrap_or("Unknown");
            println!("  {} - {}", code, name);
        }
    }
    println!();

    // Create language detector
    let detector = LanguageDetector::default();
    println!(
        "Detector confidence threshold: {:.0}%",
        detector.confidence_threshold() * 100.0
    );
    println!();

    // Simulate language detection with mock logits
    // In real usage, these come from the decoder's output over vocabulary
    println!("=== Simulated Detection Results ===\n");

    // Scenario 1: Clear English detection
    let english_logits = create_mock_logits("en", 0.92);
    demonstrate_detection("English-dominant signal", &english_logits);

    // Scenario 2: Clear Spanish detection
    let spanish_logits = create_mock_logits("es", 0.88);
    demonstrate_detection("Spanish-dominant signal", &spanish_logits);

    // Scenario 3: Ambiguous (mixed languages)
    let ambiguous_logits = create_ambiguous_logits();
    demonstrate_detection("Ambiguous signal", &ambiguous_logits);

    // Scenario 4: Low confidence
    let low_conf_logits = create_mock_logits("fr", 0.45);
    demonstrate_detection("Low confidence signal", &low_conf_logits);

    println!("=== Example Complete ===");
}

fn demonstrate_detection(scenario: &str, logits: &[f32]) {
    println!("{}:", scenario);

    let probs = LanguageProbs::from_logits(logits);

    // Get top 3 languages
    let top = probs.top_n(3);
    println!("  Top predictions:");
    for (code, prob) in &top {
        let name = language_name(code).unwrap_or("Unknown");
        let bar_len = (prob * 30.0) as usize;
        let bar: String = (0..bar_len).map(|_| '#').collect();
        println!(
            "    {} ({:>10}): {:>5.1}% {}",
            code,
            name,
            prob * 100.0,
            bar
        );
    }

    // Check confidence
    if probs.is_confident(0.7) {
        println!("  Status: Confident detection");
    } else {
        println!("  Status: Low confidence - may need more audio");
    }
    println!();
}

fn create_mock_logits(dominant_lang: &str, confidence: f32) -> Vec<f32> {
    // Create logits array for vocabulary
    // Language tokens start at special_tokens::LANG_BASE (50259)
    let vocab_size = 51865; // Whisper vocab size
    let mut logits = vec![-10.0; vocab_size];

    // Language token base is 50259
    let lang_base = 50259usize;

    // Map of language codes to offsets
    let lang_offsets = [
        ("en", 0),
        ("zh", 1),
        ("de", 2),
        ("es", 3),
        ("ru", 4),
        ("ko", 5),
        ("fr", 6),
        ("ja", 7),
        ("pt", 8),
        ("tr", 9),
        ("pl", 10),
        ("ca", 11),
        ("nl", 12),
        ("ar", 13),
        ("sv", 14),
        ("it", 15),
        ("id", 16),
        ("hi", 17),
    ];

    // Set language logits
    for (code, offset) in &lang_offsets {
        let token_id = lang_base + *offset;
        if token_id < vocab_size {
            if *code == dominant_lang {
                // Convert confidence to logit (inverse softmax approximation)
                logits[token_id] = (confidence / (1.0 - confidence + 0.01))
                    .ln()
                    .max(-10.0)
                    .min(10.0);
            } else {
                // Small random variation for other languages
                logits[token_id] = -3.0 + (*offset as f32 * 0.1).sin() * 0.5;
            }
        }
    }

    logits
}

fn create_ambiguous_logits() -> Vec<f32> {
    let vocab_size = 51865;
    let mut logits = vec![-10.0; vocab_size];
    let lang_base = 50259usize;

    // Multiple languages with similar probabilities
    logits[lang_base] = 1.2; // en
    logits[lang_base + 3] = 1.0; // es
    logits[lang_base + 6] = 0.8; // fr

    logits
}
