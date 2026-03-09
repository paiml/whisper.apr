//! Design by Contract demonstration for whisper.apr
//!
//! Validates mel spectrogram shape contracts, encoder/decoder dimension
//! consistency, and architecture-specific configuration invariants.
//!
//! Run: `cargo run --example design_by_contract`

use whisper_apr::audio::MelConfig;
use whisper_apr::model::ModelConfig;
use whisper_apr::WhisperApr;

/// Verify mel spectrogram dimension contracts for a given config.
fn check_mel_contracts(config: &MelConfig) {
    assert_eq!(
        config.n_fft / 2 + 1,
        config.n_freqs(),
        "n_freqs = n_fft/2 + 1"
    );
    assert!(config.n_mels > 0, "n_mels must be positive");
    assert!(config.hop_length > 0, "hop_length must be positive");
    assert_eq!(config.sample_rate, 16000, "Whisper requires 16kHz audio");
}

/// Verify encoder/decoder dimension consistency for a model config.
fn check_architecture_contracts(config: &ModelConfig) {
    // Cross-attention requires matching encoder and decoder hidden dims
    assert_eq!(
        config.n_audio_state, config.n_text_state,
        "encoder d_model ({}) != decoder d_model ({})",
        config.n_audio_state, config.n_text_state,
    );

    // Attention head dimension must divide evenly
    assert_eq!(
        config.n_audio_state % config.n_audio_head,
        0,
        "d_model ({}) not divisible by n_audio_head ({})",
        config.n_audio_state,
        config.n_audio_head,
    );
    assert_eq!(
        config.n_text_state % config.n_text_head,
        0,
        "d_model ({}) not divisible by n_text_head ({})",
        config.n_text_state,
        config.n_text_head,
    );

    // At least one layer in both encoder and decoder
    assert!(config.n_audio_layer > 0, "encoder must have >= 1 layer");
    assert!(config.n_text_layer > 0, "decoder must have >= 1 layer");
}

fn main() {
    println!("=== Mel Spectrogram Shape Contracts ===\n");

    let mel_80 = MelConfig::default();
    check_mel_contracts(&mel_80);
    println!(
        "  80-mel config: n_mels={}, n_fft={}, hop_length={}, n_freqs={}",
        mel_80.n_mels,
        mel_80.n_fft,
        mel_80.hop_length,
        mel_80.n_freqs()
    );

    let mel_128 = MelConfig {
        n_mels: 128,
        ..MelConfig::whisper()
    };
    check_mel_contracts(&mel_128);
    println!(
        "  128-mel config: n_mels={}, n_fft={}, hop_length={}, n_freqs={}",
        mel_128.n_mels,
        mel_128.n_fft,
        mel_128.hop_length,
        mel_128.n_freqs()
    );

    println!("\n=== Architecture Dimension Contracts ===\n");

    let variants: &[(&str, ModelConfig)] = &[
        ("tiny", ModelConfig::tiny()),
        ("base", ModelConfig::base()),
        ("small", ModelConfig::small()),
        ("medium", ModelConfig::medium()),
        ("large", ModelConfig::large()),
        ("large_v3_turbo", ModelConfig::large_v3_turbo()),
    ];

    for (name, config) in variants {
        check_architecture_contracts(config);
        println!(
            "  {:<16} d_model={:<5} enc={:<3} dec={:<3} heads={:<3} mels={}",
            name,
            config.n_audio_state,
            config.n_audio_layer,
            config.n_text_layer,
            config.n_audio_head,
            config.n_mels
        );
    }

    println!("\n=== WhisperApr Construction Contracts ===\n");

    let tiny = WhisperApr::tiny();
    assert_eq!(tiny.encoder().n_layers(), 4);
    assert_eq!(tiny.decoder().n_layers(), 4);
    println!(
        "  tiny: encoder={} layers, decoder={} layers",
        tiny.encoder().n_layers(),
        tiny.decoder().n_layers()
    );

    let base = WhisperApr::base();
    assert_eq!(base.encoder().n_layers(), 6);
    assert_eq!(base.decoder().n_layers(), 6);
    assert!(
        base.memory_size() > tiny.memory_size(),
        "base must use more memory than tiny"
    );
    println!(
        "  base: encoder={} layers, decoder={} layers, memory > tiny",
        base.encoder().n_layers(),
        base.decoder().n_layers()
    );

    println!("\nAll contracts passed.");
}
