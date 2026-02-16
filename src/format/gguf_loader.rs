//! GGUF model loading for Whisper
//!
//! Loads pre-quantized Whisper GGUF models (e.g. from HuggingFace) and converts
//! them to in-memory APR format for use with the existing `WhisperApr::load_from_apr` pipeline.
//!
//! # Tensor Name Mapping
//!
//! GGUF files from whisper.cpp use tensor names like `encoder.blocks.0.attn.query.weight`,
//! which must be remapped to the HuggingFace-style names that `load_tensor()` expects
//! (e.g. `encoder.layers.0.self_attn.q_proj.weight`).

use std::path::Path;

use crate::error::{WhisperError, WhisperResult};
use crate::format::{AprWriter, MelFilterbankData};
use crate::model::ModelConfig;
use crate::tokenizer::Vocabulary;

/// Map a whisper.cpp GGUF tensor name to the internal name expected by whisper.apr.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(
///     map_gguf_whisper_tensor_name("encoder.blocks.0.attn.query.weight"),
///     "encoder.layers.0.self_attn.q_proj.weight"
/// );
/// ```
#[must_use]
pub fn map_gguf_whisper_tensor_name(name: &str) -> String {
    // Encoder post-layernorm
    if let Some(suffix) = name.strip_prefix("encoder.ln_post.") {
        return format!("encoder.layer_norm.{suffix}");
    }

    // Decoder final layernorm
    if let Some(suffix) = name.strip_prefix("decoder.ln.") {
        return format!("decoder.layer_norm.{suffix}");
    }

    // Encoder block patterns: encoder.blocks.{n}.*
    if let Some(rest) = name.strip_prefix("encoder.blocks.") {
        if let Some((layer_str, component)) = rest.split_once('.') {
            return format!(
                "encoder.layers.{layer_str}.{}",
                map_block_component(component)
            );
        }
    }

    // Decoder block patterns: decoder.blocks.{n}.*
    if let Some(rest) = name.strip_prefix("decoder.blocks.") {
        if let Some((layer_str, component)) = rest.split_once('.') {
            return format!(
                "decoder.layers.{layer_str}.{}",
                map_decoder_block_component(component)
            );
        }
    }

    // Names that pass through unchanged:
    // encoder.conv1.weight/bias, encoder.conv2.weight/bias,
    // encoder.positional_embedding, decoder.token_embedding.weight,
    // decoder.positional_embedding
    name.to_string()
}

/// Map encoder block component from GGUF naming to internal naming.
fn map_block_component(component: &str) -> String {
    // Self-attention layernorm
    if let Some(suffix) = component.strip_prefix("attn_ln.") {
        return format!("self_attn_layer_norm.{suffix}");
    }

    // Self-attention projections
    if let Some(suffix) = component.strip_prefix("attn.query.") {
        return format!("self_attn.q_proj.{suffix}");
    }
    if let Some(suffix) = component.strip_prefix("attn.key.") {
        return format!("self_attn.k_proj.{suffix}");
    }
    if let Some(suffix) = component.strip_prefix("attn.value.") {
        return format!("self_attn.v_proj.{suffix}");
    }
    if let Some(suffix) = component.strip_prefix("attn.out.") {
        return format!("self_attn.out_proj.{suffix}");
    }

    // FFN layernorm
    if let Some(suffix) = component.strip_prefix("mlp_ln.") {
        return format!("final_layer_norm.{suffix}");
    }

    // FFN layers
    if let Some(suffix) = component.strip_prefix("mlp.0.") {
        return format!("fc1.{suffix}");
    }
    if let Some(suffix) = component.strip_prefix("mlp.2.") {
        return format!("fc2.{suffix}");
    }

    // Unknown: pass through
    component.to_string()
}

/// Map decoder block component from GGUF naming to internal naming.
fn map_decoder_block_component(component: &str) -> String {
    // Self-attention layernorm
    if let Some(suffix) = component.strip_prefix("attn_ln.") {
        return format!("self_attn_layer_norm.{suffix}");
    }

    // Self-attention projections
    if let Some(suffix) = component.strip_prefix("attn.query.") {
        return format!("self_attn.q_proj.{suffix}");
    }
    if let Some(suffix) = component.strip_prefix("attn.key.") {
        return format!("self_attn.k_proj.{suffix}");
    }
    if let Some(suffix) = component.strip_prefix("attn.value.") {
        return format!("self_attn.v_proj.{suffix}");
    }
    if let Some(suffix) = component.strip_prefix("attn.out.") {
        return format!("self_attn.out_proj.{suffix}");
    }

    // Cross-attention layernorm
    if let Some(suffix) = component.strip_prefix("cross_attn_ln.") {
        return format!("encoder_attn_layer_norm.{suffix}");
    }

    // Cross-attention projections
    if let Some(suffix) = component.strip_prefix("cross_attn.query.") {
        return format!("encoder_attn.q_proj.{suffix}");
    }
    if let Some(suffix) = component.strip_prefix("cross_attn.key.") {
        return format!("encoder_attn.k_proj.{suffix}");
    }
    if let Some(suffix) = component.strip_prefix("cross_attn.value.") {
        return format!("encoder_attn.v_proj.{suffix}");
    }
    if let Some(suffix) = component.strip_prefix("cross_attn.out.") {
        return format!("encoder_attn.out_proj.{suffix}");
    }

    // FFN layernorm
    if let Some(suffix) = component.strip_prefix("mlp_ln.") {
        return format!("final_layer_norm.{suffix}");
    }

    // FFN layers
    if let Some(suffix) = component.strip_prefix("mlp.0.") {
        return format!("fc1.{suffix}");
    }
    if let Some(suffix) = component.strip_prefix("mlp.2.") {
        return format!("fc2.{suffix}");
    }

    // Unknown: pass through
    component.to_string()
}

/// Detect Whisper model configuration from GGUF tensor shapes.
///
/// Inspects tensor dimensions to infer `d_model`, `n_mels`, `n_encoder_layers`,
/// and `n_decoder_layers`, then matches to a known configuration.
pub fn detect_whisper_config(
    tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> WhisperResult<ModelConfig> {
    // Infer d_model from encoder.conv1.weight shape[0]
    let d_model = tensors
        .iter()
        .find(|(name, _)| name.contains("conv1.weight"))
        .map(|(_, (_, shape))| shape[0])
        .ok_or_else(|| WhisperError::Format("No conv1.weight tensor found in GGUF".into()))?;

    // Infer n_mels from encoder.conv1.weight shape[1]
    let n_mels = tensors
        .iter()
        .find(|(name, _)| name.contains("conv1.weight"))
        .map_or(80, |(_, (_, shape))| shape[1]);

    // Count encoder layers (highest encoder.blocks.{n} index + 1)
    let n_encoder_layers = count_layers(tensors, "encoder.blocks.");

    // Count decoder layers (highest decoder.blocks.{n} index + 1)
    let n_decoder_layers = count_layers(tensors, "decoder.blocks.");

    match (d_model, n_encoder_layers, n_decoder_layers, n_mels) {
        (384, 4, 4, 80) => Ok(ModelConfig::tiny()),
        (512, 6, 6, 80) => Ok(ModelConfig::base()),
        (768, 12, 12, 80) => Ok(ModelConfig::small()),
        (1024, 24, 24, 80) => Ok(ModelConfig::medium()),
        (1280, 32, 32, 80 | 128) => Ok(ModelConfig::large()),
        (1280, 32, 4, _) => Ok(ModelConfig::large_v3_turbo()),
        _ => Err(WhisperError::Format(format!(
            "Unknown Whisper GGUF config: d_model={d_model}, enc={n_encoder_layers}, dec={n_decoder_layers}, mels={n_mels}"
        ))),
    }
}

/// Count the number of layers by finding the highest block index.
fn count_layers(
    tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
) -> usize {
    let mut max_idx: Option<usize> = None;
    for name in tensors.keys() {
        if let Some(rest) = name.strip_prefix(prefix) {
            if let Some(idx_str) = rest.split('.').next() {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    max_idx = Some(max_idx.map_or(idx, |prev: usize| prev.max(idx)));
                }
            }
        }
    }
    max_idx.map_or(0, |m| m + 1)
}

/// Load a Whisper GGUF file and return APR bytes.
///
/// This is the main entry point for GGUF loading. It:
/// 1. Loads all tensors from the GGUF file via aprender
/// 2. Remaps tensor names from whisper.cpp convention to internal convention
/// 3. Detects the model configuration from tensor shapes
/// 4. Builds APR bytes with embedded vocabulary and mel filterbank
/// 5. Returns bytes suitable for `WhisperApr::load_from_apr()`
///
/// # Errors
///
/// Returns error if the GGUF file cannot be read, contains no recognizable
/// Whisper tensors, or if APR serialization fails.
pub fn load_gguf_whisper(path: &Path) -> WhisperResult<Vec<u8>> {
    let result = aprender::format::gguf::load_gguf_with_tokenizer(path)
        .map_err(|e| WhisperError::Format(format!("Failed to load GGUF: {e}")))?;

    // Detect config from original tensor names (before remapping)
    let config = detect_whisper_config(&result.tensors)?;

    // Build APR writer
    let mut writer = AprWriter::from_config(&config);

    // Remap tensor names and add to writer
    for (gguf_name, (data, shape)) in &result.tensors {
        let apr_name = map_gguf_whisper_tensor_name(gguf_name);
        writer.add(apr_name, shape.clone(), data.clone());
    }

    // Embed vocabulary from GGUF tokenizer
    if result.tokenizer.has_vocabulary() {
        let vocab = build_vocabulary_from_gguf(&result.tokenizer);
        writer.set_vocabulary(vocab);
    }

    // Generate and embed mel filterbank
    let n_mels = config.n_mels;
    let n_freqs = 201u32; // Whisper uses n_fft=400, so n_fft/2+1 = 201
    let filterbank_data = generate_mel_filterbank_data(n_mels, n_freqs);
    writer.set_mel_filterbank(MelFilterbankData::new(n_mels, n_freqs, filterbank_data));

    writer
        .to_bytes()
        .map_err(|e| WhisperError::Format(format!("Failed to write APR bytes: {e}")))
}

/// Build a `Vocabulary` from GGUF tokenizer data.
fn build_vocabulary_from_gguf(tokenizer: &aprender::format::gguf::GgufTokenizer) -> Vocabulary {
    let mut vocab = Vocabulary::new();
    for token_str in &tokenizer.vocabulary {
        vocab.add_token(token_str.as_bytes().to_vec());
    }
    vocab
}

/// Generate mel filterbank weights algorithmically (Slaney normalization).
///
/// Produces `n_mels * n_freqs` f32 values in row-major order.
fn generate_mel_filterbank_data(n_mels: u32, n_freqs: u32) -> Vec<f32> {
    let sample_rate = 16000.0_f64;
    let n_fft = (n_freqs as usize - 1) * 2; // 400 for n_freqs=201

    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(sample_rate / 2.0);

    // n_mels + 2 points (edges of all triangular filters)
    let n_points = n_mels as usize + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_low + (mel_high - mel_low) * i as f64 / (n_points - 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin indices
    let bin_points: Vec<f64> = hz_points
        .iter()
        .map(|&hz| hz * n_fft as f64 / sample_rate)
        .collect();

    let mut data = vec![0.0f32; n_mels as usize * n_freqs as usize];

    for m in 0..n_mels as usize {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        // Slaney normalization factor
        let enorm = 2.0 / (hz_points[m + 2] - hz_points[m]);

        for k in 0..n_freqs as usize {
            let kf = k as f64;
            let val = if kf >= left && kf < center && center > left {
                enorm * (kf - left) / (center - left)
            } else if kf >= center && kf <= right && right > center {
                enorm * (right - kf) / (right - center)
            } else {
                0.0
            };
            data[m * n_freqs as usize + k] = val as f32;
        }
    }

    data
}

/// Convert frequency in Hz to mel scale.
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale to Hz.
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Tensor name mapping tests — encoder
    // =========================================================================

    #[test]
    fn test_map_encoder_conv_unchanged() {
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.conv1.weight"),
            "encoder.conv1.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.conv1.bias"),
            "encoder.conv1.bias"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.conv2.weight"),
            "encoder.conv2.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.conv2.bias"),
            "encoder.conv2.bias"
        );
    }

    #[test]
    fn test_map_encoder_positional_embedding() {
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.positional_embedding"),
            "encoder.positional_embedding"
        );
    }

    #[test]
    fn test_map_encoder_ln_post() {
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.ln_post.weight"),
            "encoder.layer_norm.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.ln_post.bias"),
            "encoder.layer_norm.bias"
        );
    }

    #[test]
    fn test_map_encoder_block_attn_ln() {
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.0.attn_ln.weight"),
            "encoder.layers.0.self_attn_layer_norm.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.5.attn_ln.bias"),
            "encoder.layers.5.self_attn_layer_norm.bias"
        );
    }

    #[test]
    fn test_map_encoder_block_self_attn() {
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.0.attn.query.weight"),
            "encoder.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.0.attn.key.weight"),
            "encoder.layers.0.self_attn.k_proj.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.0.attn.value.weight"),
            "encoder.layers.0.self_attn.v_proj.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.0.attn.out.weight"),
            "encoder.layers.0.self_attn.out_proj.weight"
        );
    }

    #[test]
    fn test_map_encoder_block_self_attn_bias() {
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.3.attn.query.bias"),
            "encoder.layers.3.self_attn.q_proj.bias"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.3.attn.key.bias"),
            "encoder.layers.3.self_attn.k_proj.bias"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.3.attn.value.bias"),
            "encoder.layers.3.self_attn.v_proj.bias"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.3.attn.out.bias"),
            "encoder.layers.3.self_attn.out_proj.bias"
        );
    }

    #[test]
    fn test_map_encoder_block_mlp_ln() {
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.0.mlp_ln.weight"),
            "encoder.layers.0.final_layer_norm.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.0.mlp_ln.bias"),
            "encoder.layers.0.final_layer_norm.bias"
        );
    }

    #[test]
    fn test_map_encoder_block_ffn() {
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.0.mlp.0.weight"),
            "encoder.layers.0.fc1.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.0.mlp.0.bias"),
            "encoder.layers.0.fc1.bias"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.0.mlp.2.weight"),
            "encoder.layers.0.fc2.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.0.mlp.2.bias"),
            "encoder.layers.0.fc2.bias"
        );
    }

    // =========================================================================
    // Tensor name mapping tests — decoder
    // =========================================================================

    #[test]
    fn test_map_decoder_token_embedding() {
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.token_embedding.weight"),
            "decoder.token_embedding.weight"
        );
    }

    #[test]
    fn test_map_decoder_positional_embedding() {
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.positional_embedding"),
            "decoder.positional_embedding"
        );
    }

    #[test]
    fn test_map_decoder_ln() {
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.ln.weight"),
            "decoder.layer_norm.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.ln.bias"),
            "decoder.layer_norm.bias"
        );
    }

    #[test]
    fn test_map_decoder_block_self_attn_ln() {
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.0.attn_ln.weight"),
            "decoder.layers.0.self_attn_layer_norm.weight"
        );
    }

    #[test]
    fn test_map_decoder_block_self_attn() {
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.0.attn.query.weight"),
            "decoder.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.0.attn.key.weight"),
            "decoder.layers.0.self_attn.k_proj.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.0.attn.value.weight"),
            "decoder.layers.0.self_attn.v_proj.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.0.attn.out.weight"),
            "decoder.layers.0.self_attn.out_proj.weight"
        );
    }

    #[test]
    fn test_map_decoder_block_cross_attn_ln() {
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.0.cross_attn_ln.weight"),
            "decoder.layers.0.encoder_attn_layer_norm.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.0.cross_attn_ln.bias"),
            "decoder.layers.0.encoder_attn_layer_norm.bias"
        );
    }

    #[test]
    fn test_map_decoder_block_cross_attn() {
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.0.cross_attn.query.weight"),
            "decoder.layers.0.encoder_attn.q_proj.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.0.cross_attn.key.weight"),
            "decoder.layers.0.encoder_attn.k_proj.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.0.cross_attn.value.weight"),
            "decoder.layers.0.encoder_attn.v_proj.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.0.cross_attn.out.weight"),
            "decoder.layers.0.encoder_attn.out_proj.weight"
        );
    }

    #[test]
    fn test_map_decoder_block_cross_attn_bias() {
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.2.cross_attn.query.bias"),
            "decoder.layers.2.encoder_attn.q_proj.bias"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.2.cross_attn.key.bias"),
            "decoder.layers.2.encoder_attn.k_proj.bias"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.2.cross_attn.value.bias"),
            "decoder.layers.2.encoder_attn.v_proj.bias"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.2.cross_attn.out.bias"),
            "decoder.layers.2.encoder_attn.out_proj.bias"
        );
    }

    #[test]
    fn test_map_decoder_block_mlp_ln() {
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.0.mlp_ln.weight"),
            "decoder.layers.0.final_layer_norm.weight"
        );
    }

    #[test]
    fn test_map_decoder_block_ffn() {
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.0.mlp.0.weight"),
            "decoder.layers.0.fc1.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.0.mlp.2.weight"),
            "decoder.layers.0.fc2.weight"
        );
    }

    // =========================================================================
    // Multi-digit layer index tests
    // =========================================================================

    #[test]
    fn test_map_high_layer_index() {
        assert_eq!(
            map_gguf_whisper_tensor_name("encoder.blocks.31.attn.query.weight"),
            "encoder.layers.31.self_attn.q_proj.weight"
        );
        assert_eq!(
            map_gguf_whisper_tensor_name("decoder.blocks.3.cross_attn.value.bias"),
            "decoder.layers.3.encoder_attn.v_proj.bias"
        );
    }

    // =========================================================================
    // Config detection tests
    // =========================================================================

    #[test]
    fn test_detect_tiny_config() {
        let tensors = make_fake_tensors(384, 80, 4, 4);
        let config = detect_whisper_config(&tensors).expect("detect tiny");
        assert_eq!(config.n_audio_state, 384);
        assert_eq!(config.n_audio_layer, 4);
        assert_eq!(config.n_text_layer, 4);
    }

    #[test]
    fn test_detect_base_config() {
        let tensors = make_fake_tensors(512, 80, 6, 6);
        let config = detect_whisper_config(&tensors).expect("detect base");
        assert_eq!(config.n_audio_state, 512);
        assert_eq!(config.n_audio_layer, 6);
        assert_eq!(config.n_text_layer, 6);
    }

    #[test]
    fn test_detect_small_config() {
        let tensors = make_fake_tensors(768, 80, 12, 12);
        let config = detect_whisper_config(&tensors).expect("detect small");
        assert_eq!(config.n_audio_state, 768);
    }

    #[test]
    fn test_detect_medium_config() {
        let tensors = make_fake_tensors(1024, 80, 24, 24);
        let config = detect_whisper_config(&tensors).expect("detect medium");
        assert_eq!(config.n_audio_state, 1024);
    }

    #[test]
    fn test_detect_large_config() {
        let tensors = make_fake_tensors(1280, 80, 32, 32);
        let config = detect_whisper_config(&tensors).expect("detect large");
        assert_eq!(config.n_audio_state, 1280);
        assert_eq!(config.n_audio_layer, 32);
        assert_eq!(config.n_text_layer, 32);
    }

    #[test]
    fn test_detect_large_v3_turbo_config() {
        let tensors = make_fake_tensors(1280, 128, 32, 4);
        let config = detect_whisper_config(&tensors).expect("detect turbo");
        assert_eq!(config.n_audio_state, 1280);
        assert_eq!(config.n_audio_layer, 32);
        assert_eq!(config.n_text_layer, 4);
        assert_eq!(config.n_mels, 128);
    }

    #[test]
    fn test_detect_unknown_config_errors() {
        let tensors = make_fake_tensors(999, 80, 7, 7);
        assert!(detect_whisper_config(&tensors).is_err());
    }

    // =========================================================================
    // Mel filterbank generation tests
    // =========================================================================

    #[test]
    fn test_generate_mel_filterbank_80() {
        let data = generate_mel_filterbank_data(80, 201);
        assert_eq!(data.len(), 80 * 201);
        // All values should be non-negative
        assert!(data.iter().all(|&v| v >= 0.0));
        // First row should have some nonzero values
        let first_row_sum: f32 = data[..201].iter().sum();
        assert!(first_row_sum > 0.0, "First mel band should be nonzero");
    }

    #[test]
    fn test_generate_mel_filterbank_128() {
        let data = generate_mel_filterbank_data(128, 201);
        assert_eq!(data.len(), 128 * 201);
        assert!(data.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_hz_to_mel_roundtrip() {
        for &hz in &[0.0, 100.0, 1000.0, 8000.0] {
            let mel = hz_to_mel(hz);
            let recovered = mel_to_hz(mel);
            assert!(
                (hz - recovered).abs() < 0.01,
                "hz_to_mel roundtrip failed: {hz} -> {mel} -> {recovered}"
            );
        }
    }

    // =========================================================================
    // Vocabulary building test
    // =========================================================================

    #[test]
    fn test_build_vocabulary_from_gguf() {
        let tokenizer = aprender::format::gguf::GgufTokenizer {
            vocabulary: vec![
                "hello".to_string(),
                "world".to_string(),
                "<|startoftranscript|>".to_string(),
            ],
            ..Default::default()
        };
        let vocab = build_vocabulary_from_gguf(&tokenizer);
        assert_eq!(vocab.len(), 3);
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Build a fake tensor map with the right shape for config detection.
    fn make_fake_tensors(
        d_model: usize,
        n_mels: usize,
        n_enc_layers: usize,
        n_dec_layers: usize,
    ) -> std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        let mut map = std::collections::BTreeMap::new();
        // conv1.weight: [d_model, n_mels, 3]
        map.insert(
            "encoder.conv1.weight".to_string(),
            (vec![0.0; d_model * n_mels * 3], vec![d_model, n_mels, 3]),
        );
        // Add encoder block tensors for layer counting
        for i in 0..n_enc_layers {
            map.insert(
                format!("encoder.blocks.{i}.attn.query.weight"),
                (vec![0.0; d_model * d_model], vec![d_model, d_model]),
            );
        }
        // Add decoder block tensors for layer counting
        for i in 0..n_dec_layers {
            map.insert(
                format!("decoder.blocks.{i}.attn.query.weight"),
                (vec![0.0; d_model * d_model], vec![d_model, d_model]),
            );
        }
        map
    }
}
