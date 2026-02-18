//! Whisper-specific metadata mapping for APR v2 format
//!
//! Converts between `ModelConfig` and `AprV2Metadata` for Whisper and Moonshine models.

use crate::format::{FfnActivation, ModelFamily};
use crate::model::{AttentionType, AudioFrontend, ModelConfig, PositionalEncoding};
use crate::ModelType;
use aprender::format::v2::AprV2Metadata;
use serde_json::json;

/// Build `AprV2Metadata` from a Whisper/Moonshine `ModelConfig`.
///
/// Audio-specific fields that don't have direct `AprV2Metadata` counterparts
/// are stored in `custom` as JSON values.
#[must_use]
pub fn build_whisper_metadata(config: &ModelConfig, source: &str) -> AprV2Metadata {
    let mut meta = AprV2Metadata::new("speech-recognition");

    let arch = match config.model_family {
        ModelFamily::Moonshine => "moonshine",
        _ => "whisper",
    };
    meta.architecture = Some(arch.to_string());
    meta.vocab_size = Some(config.n_vocab as usize);
    meta.hidden_size = Some(config.n_text_state as usize);
    meta.num_layers = Some(config.n_text_layer as usize);
    meta.num_heads = Some(config.n_text_head as usize);
    meta.source = Some(source.to_string());
    meta.original_format = Some("safetensors".to_string());

    // Audio-specific in custom
    meta.custom
        .insert("n_audio_state".into(), json!(config.n_audio_state));
    meta.custom
        .insert("n_audio_layer".into(), json!(config.n_audio_layer));
    meta.custom
        .insert("n_audio_head".into(), json!(config.n_audio_head));
    meta.custom
        .insert("n_audio_ctx".into(), json!(config.n_audio_ctx));
    meta.custom
        .insert("n_text_ctx".into(), json!(config.n_text_ctx));
    meta.custom
        .insert("n_mels".into(), json!(config.n_mels));
    meta.custom
        .insert("model_type_id".into(), json!(model_type_to_u8(config.model_type)));

    // Audio frontend and encoding types
    let frontend_str = match config.audio_frontend {
        AudioFrontend::MelFilterbank => "mel_filterbank",
        AudioFrontend::LearnedConv => "learned_conv",
    };
    meta.custom
        .insert("audio_frontend".into(), json!(frontend_str));

    let encoding_str = match config.positional_encoding {
        PositionalEncoding::Sinusoidal => "sinusoidal",
        PositionalEncoding::Rotary => "rotary",
    };
    meta.custom
        .insert("positional_encoding".into(), json!(encoding_str));

    let family_str = match config.model_family {
        ModelFamily::Whisper => "whisper",
        ModelFamily::Moonshine => "moonshine",
        ModelFamily::Lfm2 => "lfm2",
        ModelFamily::Llama => "llama",
        ModelFamily::Generic => "generic",
    };
    meta.custom
        .insert("model_family".into(), json!(family_str));

    meta
}

/// Reconstruct `ModelConfig` from `AprV2Metadata`.
///
/// Extracts Whisper/Moonshine-specific fields from `custom` HashMap.
#[must_use]
pub fn metadata_to_model_config(meta: &AprV2Metadata) -> ModelConfig {
    let get_u32 = |key: &str, default: u32| -> u32 {
        meta.custom
            .get(key)
            .and_then(|v| v.as_u64())
            .map_or(default, |v| v as u32)
    };

    let n_audio_state = get_u32("n_audio_state", 384);
    let n_audio_layer = get_u32("n_audio_layer", 4);
    let n_audio_head = get_u32("n_audio_head", 6);
    let n_audio_ctx = get_u32("n_audio_ctx", 1500);
    let n_text_ctx = get_u32("n_text_ctx", 448);
    let n_mels = get_u32("n_mels", 80);
    let model_type_id = get_u32("model_type_id", 0) as u8;

    let n_vocab = meta.vocab_size.unwrap_or(51865) as u32;
    let n_text_state = meta.hidden_size.unwrap_or(384) as u32;
    let n_text_layer = meta.num_layers.unwrap_or(4) as u32;
    let n_text_head = meta.num_heads.unwrap_or(6) as u32;

    let model_type = u8_to_model_type(model_type_id);

    let audio_frontend = match meta
        .custom
        .get("audio_frontend")
        .and_then(|v| v.as_str())
    {
        Some("learned_conv") => AudioFrontend::LearnedConv,
        _ => AudioFrontend::MelFilterbank,
    };

    let positional_encoding = match meta
        .custom
        .get("positional_encoding")
        .and_then(|v| v.as_str())
    {
        Some("rotary") => PositionalEncoding::Rotary,
        _ => PositionalEncoding::Sinusoidal,
    };

    let model_family = match meta.custom.get("model_family").and_then(|v| v.as_str()) {
        Some("moonshine") => ModelFamily::Moonshine,
        Some("lfm2") => ModelFamily::Lfm2,
        Some("llama") => ModelFamily::Llama,
        Some("generic") => ModelFamily::Generic,
        _ => ModelFamily::Whisper,
    };

    let attention_type = match meta.architecture.as_deref() {
        Some("moonshine") => {
            let kv_heads = meta.num_kv_heads.unwrap_or(n_text_head as usize) as u32;
            if kv_heads < n_text_head {
                AttentionType::Gqa { kv_heads }
            } else {
                AttentionType::Mha
            }
        }
        _ => AttentionType::Mha,
    };

    ModelConfig {
        model_type,
        n_vocab,
        n_audio_ctx,
        n_audio_state,
        n_audio_head,
        n_audio_layer,
        n_text_ctx,
        n_text_state,
        n_text_head,
        n_text_layer,
        n_mels,
        audio_frontend,
        positional_encoding,
        ffn_activation: FfnActivation::Gelu,
        attention_type,
        model_family,
    }
}

fn model_type_to_u8(mt: ModelType) -> u8 {
    match mt {
        ModelType::Tiny => 0,
        ModelType::TinyEn => 1,
        ModelType::Base => 2,
        ModelType::BaseEn => 3,
        ModelType::Small => 4,
        ModelType::SmallEn => 5,
        ModelType::Medium => 6,
        ModelType::MediumEn => 7,
        ModelType::Large => 8,
        ModelType::LargeV1 => 9,
        ModelType::LargeV2 => 10,
        ModelType::LargeV3 => 11,
        ModelType::LargeV3Turbo => 12,
    }
}

fn u8_to_model_type(id: u8) -> ModelType {
    match id {
        1 => ModelType::TinyEn,
        2 => ModelType::Base,
        3 => ModelType::BaseEn,
        4 => ModelType::Small,
        5 => ModelType::SmallEn,
        6 => ModelType::Medium,
        7 => ModelType::MediumEn,
        8 => ModelType::Large,
        9 => ModelType::LargeV1,
        10 => ModelType::LargeV2,
        11 => ModelType::LargeV3,
        12 => ModelType::LargeV3Turbo,
        _ => ModelType::Tiny,
    }
}

/// Mel filterbank data for embedding in .apr files
#[derive(Debug, Clone)]
pub struct MelFilterbankData {
    /// Number of mel bands (80 or 128)
    pub n_mels: u32,
    /// Number of frequency bins (typically 201)
    pub n_freqs: u32,
    /// Raw f32 filterbank data (row-major: n_mels x n_freqs)
    pub data: Vec<f32>,
}

impl MelFilterbankData {
    /// Create new mel filterbank data
    #[must_use]
    pub fn new(n_mels: u32, n_freqs: u32, data: Vec<f32>) -> Self {
        let expected = (n_mels * n_freqs) as usize;
        assert_eq!(
            data.len(),
            expected,
            "filterbank data length {} doesn't match {}x{}={}",
            data.len(),
            n_mels,
            n_freqs,
            expected
        );
        Self {
            n_mels,
            n_freqs,
            data,
        }
    }

    /// Create mel_80 filterbank (80 bands x 201 freqs)
    #[must_use]
    pub fn mel_80(data: Vec<f32>) -> Self {
        Self::new(80, 201, data)
    }

    /// Create mel_128 filterbank (128 bands x 201 freqs)
    #[must_use]
    pub fn mel_128(data: Vec<f32>) -> Self {
        Self::new(128, 201, data)
    }

    /// Byte size of filterbank section: header(8 bytes) + data
    #[must_use]
    pub fn byte_size(&self) -> usize {
        8 + self.data.len() * 4
    }

    /// Serialize to bytes
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.byte_size());
        bytes.extend_from_slice(&self.n_mels.to_le_bytes());
        bytes.extend_from_slice(&self.n_freqs.to_le_bytes());
        for &v in &self.data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    /// Parse from bytes
    pub fn from_bytes(data: &[u8]) -> crate::error::WhisperResult<Self> {
        use crate::error::WhisperError;

        if data.len() < 8 {
            return Err(WhisperError::Format("filterbank header too short".into()));
        }

        let n_mels = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let n_freqs = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let expected_bytes = (n_mels * n_freqs) as usize * 4;

        if data.len() < 8 + expected_bytes {
            return Err(WhisperError::Format(format!(
                "filterbank data too short: expected {} bytes, got {}",
                8 + expected_bytes,
                data.len()
            )));
        }

        let f32_data: Vec<f32> = data[8..8 + expected_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(Self {
            n_mels,
            n_freqs,
            data: f32_data,
        })
    }
}

/// Create a minimal valid .apr file for testing using AprV2Writer
#[must_use]
pub fn create_test_apr() -> Vec<u8> {
    use aprender::format::v2::AprV2Writer;

    let config = ModelConfig::tiny();
    let meta = build_whisper_metadata(&config, "test");
    let mut writer = AprV2Writer::new(meta);

    // Add a small dummy tensor so the file is valid
    writer.add_f32_tensor("test.weight", vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);

    // SAFETY: test helper — writer.write() only fails on I/O errors, not in-memory
    #[allow(clippy::expect_used)]
    writer.write().expect("test apr write should succeed")
}
