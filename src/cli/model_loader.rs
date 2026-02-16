//! Model loading and caching for whisper-apr CLI
//!
//! This module handles automatic downloading of Whisper and Moonshine models
//! from HuggingFace when no local model path is provided.
//!
//! # Cache Location
//!
//! Models are cached in `~/.cache/whisper-apr/models/` following XDG conventions.
//!
//! # Supported Models
//!
//! - `openai/whisper-tiny` → tiny.apr
//! - `openai/whisper-base` → base.apr
//! - `openai/whisper-small` → small.apr
//! - `openai/whisper-medium` → medium.apr
//! - `openai/whisper-large-v3` → large.apr
//! - `usefulsensors/moonshine-tiny` → moonshine-tiny.apr
//! - `usefulsensors/moonshine-base` → moonshine-base.apr

use std::fs;
use std::path::PathBuf;

use super::args::ModelSize;
use crate::tokenizer::Vocabulary;
use crate::WhisperApr;

/// Model loader error
#[derive(Debug, thiserror::Error)]
pub enum ModelLoaderError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Model load error
    #[error("Model load error: {0}")]
    ModelLoad(#[from] crate::WhisperError),

    /// Download error
    #[error("Download error: {0}")]
    Download(String),

    /// Cache error
    #[error("Cache error: {0}")]
    Cache(String),
}

/// Result type for model loader operations
pub type ModelLoaderResult<T> = Result<T, ModelLoaderError>;

/// HuggingFace repository IDs for Whisper models
fn get_hf_repo_id(size: ModelSize) -> &'static str {
    match size {
        ModelSize::Tiny => "openai/whisper-tiny",
        ModelSize::Base => "openai/whisper-base",
        ModelSize::Small => "openai/whisper-small",
        ModelSize::Medium => "openai/whisper-medium",
        ModelSize::Large => "openai/whisper-large-v3",
        ModelSize::MoonshineTiny => "usefulsensors/moonshine-tiny",
        ModelSize::MoonshineBase => "usefulsensors/moonshine-base",
    }
}

/// Get the filename for a model size
fn get_model_filename(size: ModelSize) -> &'static str {
    match size {
        ModelSize::Tiny => "tiny.apr",
        ModelSize::Base => "base.apr",
        ModelSize::Small => "small.apr",
        ModelSize::Medium => "medium.apr",
        ModelSize::Large => "large.apr",
        ModelSize::MoonshineTiny => "moonshine-tiny.apr",
        ModelSize::MoonshineBase => "moonshine-base.apr",
    }
}

/// Get the cache directory for whisper-apr models
pub fn get_cache_dir() -> PathBuf {
    // Follow XDG conventions
    if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(xdg_cache).join("whisper-apr").join("models")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home)
            .join(".cache")
            .join("whisper-apr")
            .join("models")
    } else {
        // Fallback to current directory
        PathBuf::from(".cache").join("whisper-apr").join("models")
    }
}

/// Get the cache path for a specific model size
pub fn get_model_cache_path(size: ModelSize) -> PathBuf {
    get_cache_dir().join(get_model_filename(size))
}

/// Check if a model is already cached
pub fn is_model_cached(size: ModelSize) -> bool {
    let path = get_model_cache_path(size);
    path.exists() && path.metadata().map(|m| m.len() > 0).unwrap_or(false)
}

/// Download a model from HuggingFace Hub
///
/// Downloads SafeTensors weights and converts them to .apr format.
/// Whisper models include mel filterbank and vocabulary embedding.
/// Moonshine models use their own tensor name mapping (no mel/vocab).
fn download_model(size: ModelSize, verbose: bool) -> ModelLoaderResult<PathBuf> {
    use hf_hub::api::sync::Api;

    let repo_id = get_hf_repo_id(size);
    let cache_path = get_model_cache_path(size);

    if verbose {
        eprintln!("[INFO] Downloading model from HuggingFace: {repo_id}");
    }

    // Ensure cache directory exists
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Use hf_hub to download
    let api = Api::new().map_err(|e| ModelLoaderError::Download(e.to_string()))?;
    let repo = api.model(repo_id.to_string());

    // Download model.safetensors
    let safetensors_path = repo
        .get("model.safetensors")
        .map_err(|e| ModelLoaderError::Download(format!("Failed to download model: {e}")))?;

    if verbose {
        eprintln!(
            "[INFO] Downloaded safetensors to: {}",
            safetensors_path.display()
        );
    }

    // Moonshine uses SafeTensors with different tensor naming and no mel/vocab
    if size.is_moonshine() {
        convert_moonshine_safetensors_to_apr(&safetensors_path, &cache_path, size, verbose)?;
        return Ok(cache_path);
    }

    // Download vocab.json for tokenizer
    let vocab_path = repo
        .get("vocab.json")
        .map_err(|e| ModelLoaderError::Download(format!("Failed to download vocab: {e}")))?;

    if verbose {
        eprintln!("[INFO] Downloaded vocab.json");
    }

    // Download preprocessor_config.json for mel filters
    let preprocessor_path = repo.get("preprocessor_config.json").map_err(|e| {
        ModelLoaderError::Download(format!("Failed to download preprocessor_config: {e}"))
    })?;

    if verbose {
        eprintln!("[INFO] Downloaded preprocessor_config.json");
    }

    // Convert safetensors to .apr format with vocabulary and mel filters
    convert_safetensors_to_apr(
        &safetensors_path,
        &vocab_path,
        &preprocessor_path,
        &cache_path,
        size,
        verbose,
    )?;

    Ok(cache_path)
}

/// Convert a safetensors tensor view to f32 data, returning None for unsupported dtypes
fn convert_tensor_to_f32(tensor: &safetensors::tensor::TensorView<'_>) -> Option<Vec<f32>> {
    match tensor.dtype() {
        safetensors::Dtype::F32 => Some(
            tensor
                .data()
                .chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        ),
        safetensors::Dtype::F16 => Some(
            tensor
                .data()
                .chunks(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect(),
        ),
        safetensors::Dtype::BF16 => Some(
            tensor
                .data()
                .chunks(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect(),
        ),
        _ => None,
    }
}

/// Convert safetensors to .apr format
fn convert_safetensors_to_apr(
    safetensors_path: &std::path::Path,
    vocab_path: &std::path::Path,
    preprocessor_path: &std::path::Path,
    apr_path: &std::path::Path,
    size: ModelSize,
    verbose: bool,
) -> ModelLoaderResult<()> {
    use crate::format::AprWriter;
    use crate::model::ModelConfig;
    use safetensors::SafeTensors;

    if verbose {
        eprintln!("[INFO] Converting to .apr format...");
    }

    // Read safetensors file
    let data = fs::read(safetensors_path)?;
    let tensors =
        SafeTensors::deserialize(&data).map_err(|e| ModelLoaderError::Download(e.to_string()))?;

    // Get model config
    let config = match size {
        ModelSize::Tiny => ModelConfig::tiny(),
        ModelSize::Base => ModelConfig::base(),
        ModelSize::Small => ModelConfig::small(),
        ModelSize::Medium => ModelConfig::medium(),
        ModelSize::Large => ModelConfig::large(),
        ModelSize::MoonshineTiny => ModelConfig::moonshine_tiny(),
        ModelSize::MoonshineBase => ModelConfig::moonshine_base(),
    };

    // Create APR writer
    let mut writer = AprWriter::from_config(&config);

    // Load mel filters from preprocessor_config.json
    if let Ok(mel_filterbank) = load_mel_filters_from_preprocessor(preprocessor_path, verbose) {
        if verbose {
            eprintln!(
                "[INFO] Embedding mel filterbank: {} x {} = {} values",
                mel_filterbank.n_mels,
                mel_filterbank.n_freqs,
                mel_filterbank.data.len()
            );
        }
        writer.set_mel_filterbank(mel_filterbank);
    }

    // Load and embed vocabulary from vocab.json
    if let Ok(vocab) = load_vocabulary_from_json(vocab_path, verbose) {
        if verbose {
            eprintln!("[INFO] Embedding vocabulary with {} tokens", vocab.len());
        }
        writer.set_vocabulary(vocab);
    } else if verbose {
        eprintln!("[WARN] Failed to load vocabulary, using base tokens");
    }

    // Map tensor names from HuggingFace format to our format and write
    for (name, tensor) in tensors.tensors() {
        let Some(f32_data) = convert_tensor_to_f32(&tensor) else {
            if verbose {
                eprintln!("[WARN] Skipping tensor {name} with unsupported dtype");
            }
            continue;
        };

        let our_name = map_tensor_name(&name);
        let shape: Vec<usize> = tensor.shape().to_vec();
        writer.add(our_name, shape, f32_data);
    }

    // Write to file
    let apr_data = writer
        .to_bytes()
        .map_err(|e| ModelLoaderError::Download(format!("Failed to write APR: {e}")))?;
    fs::write(apr_path, apr_data)?;

    if verbose {
        eprintln!("[INFO] Saved model to: {}", apr_path.display());
    }

    Ok(())
}

/// Convert Moonshine SafeTensors to .apr format
///
/// Moonshine models use different tensor naming, no mel filterbank,
/// and SentencePiece tokenizer (not GPT-2 BPE). Tied embeddings
/// (proj_out = embed_tokens) are handled by cloning.
fn convert_moonshine_safetensors_to_apr(
    safetensors_path: &std::path::Path,
    apr_path: &std::path::Path,
    size: ModelSize,
    verbose: bool,
) -> ModelLoaderResult<()> {
    use crate::format::{map_moonshine_tensor_name, AprWriter};
    use crate::model::ModelConfig;
    use safetensors::SafeTensors;

    if verbose {
        eprintln!("[INFO] Converting Moonshine SafeTensors to .apr format...");
    }

    let data = fs::read(safetensors_path)?;
    let tensors =
        SafeTensors::deserialize(&data).map_err(|e| ModelLoaderError::Download(e.to_string()))?;

    let config = match size {
        ModelSize::MoonshineTiny => ModelConfig::moonshine_tiny(),
        ModelSize::MoonshineBase => ModelConfig::moonshine_base(),
        _ => unreachable!("only called for Moonshine models"),
    };

    let mut writer = AprWriter::from_config(&config);

    // Track embed_tokens for tied embeddings
    let mut embed_tokens_data: Option<(Vec<usize>, Vec<f32>)> = None;
    let mut has_proj_out = false;

    for (name, tensor) in tensors.tensors() {
        let Some(f32_data) = convert_tensor_to_f32(&tensor) else {
            if verbose {
                eprintln!("[WARN] Skipping tensor {name} with unsupported dtype");
            }
            continue;
        };

        let our_name = map_moonshine_tensor_name(&name);
        let shape: Vec<usize> = tensor.shape().to_vec();

        if our_name == "decoder.proj_out.weight" {
            has_proj_out = true;
        }
        if our_name == "decoder.token_embedding.weight" {
            embed_tokens_data = Some((shape.clone(), f32_data.clone()));
        }

        writer.add(our_name, shape, f32_data);
    }

    // Handle tied embeddings: clone embed_tokens → proj_out if missing
    if !has_proj_out {
        if let Some((shape, data)) = embed_tokens_data {
            if verbose {
                eprintln!("[INFO] Tied embeddings: cloning embed_tokens → proj_out");
            }
            writer.add("decoder.proj_out.weight", shape, data);
        }
    }

    let apr_data = writer
        .to_bytes()
        .map_err(|e| ModelLoaderError::Download(format!("Failed to write APR: {e}")))?;
    fs::write(apr_path, apr_data)?;

    if verbose {
        eprintln!("[INFO] Saved Moonshine model to: {}", apr_path.display());
    }

    Ok(())
}

/// Map HuggingFace tensor names to our internal format
///
/// HuggingFace Whisper uses names like:
/// - `model.encoder.conv1.weight`
/// - `model.decoder.layers.0.self_attn.k_proj.weight`
///
/// We strip the `model.` prefix to:
/// 1. Keep tensor names under 48 bytes (APR format limit)
/// 2. Match our loading code which expects `encoder.` / `decoder.` prefixes
///
/// The `find_tensor` function in format/mod.rs handles the reverse mapping
/// by trying both with and without `model.` prefix when loading.
fn map_tensor_name(hf_name: &str) -> String {
    // Strip "model." prefix if present (HuggingFace Whisper uses this)
    if let Some(stripped) = hf_name.strip_prefix("model.") {
        stripped.to_string()
    } else {
        hf_name.to_string()
    }
}

/// Load vocabulary from HuggingFace vocab.json file
///
/// GPT-2 style tokenizers use a special Unicode encoding where each byte
/// is mapped to a printable Unicode character. This function decodes those
/// tokens back to their raw byte sequences.
///
/// After loading the base vocabulary, this adds Whisper special tokens:
/// - SOT, language tokens (99 languages), TRANSLATE, TRANSCRIBE
/// - SPEAKER_TURN, PREV, NO_SPEECH, NO_TIMESTAMPS
/// - Timestamp tokens (1501 tokens for 30 seconds at 0.02s resolution)
fn load_vocabulary_from_json(
    vocab_path: &std::path::Path,
    verbose: bool,
) -> ModelLoaderResult<Vocabulary> {
    use crate::tokenizer::special_tokens;
    use std::collections::HashMap;

    // Read and parse vocab.json
    let vocab_json = fs::read_to_string(vocab_path)?;
    let token_map: HashMap<String, u32> =
        serde_json::from_str(&vocab_json).map_err(|e| ModelLoaderError::Download(e.to_string()))?;

    if verbose {
        eprintln!("[INFO] Loaded vocab.json with {} tokens", token_map.len());
    }

    // Sort tokens by ID to ensure correct ordering
    let mut tokens: Vec<(String, u32)> = token_map.into_iter().collect();
    tokens.sort_by_key(|(_, id)| *id);

    // Build the GPT-2 byte decoder (Unicode char -> byte)
    let byte_decoder = build_gpt2_byte_decoder();

    // Create vocabulary and add tokens in order
    let mut vocab = Vocabulary::new();

    for (token_str, expected_id) in tokens {
        // Decode GPT-2 Unicode string to bytes
        let bytes = decode_gpt2_token(&token_str, &byte_decoder);
        let actual_id = vocab.add_token(bytes);

        // Sanity check - IDs should match
        if actual_id != expected_id && verbose {
            eprintln!(
                "[WARN] Token ID mismatch for '{}': expected {}, got {}",
                token_str, expected_id, actual_id
            );
        }
    }

    // Add Whisper special tokens (multilingual model format)
    // vocab.json has 50258 tokens (0-50257), we need to add the rest up to 51865
    let current_size = vocab.len();

    // Add remaining tokens as placeholders for special tokens
    // SOT is at 50258, language tokens at 50259-50357, task tokens at 50358-50363
    // Timestamp tokens start at 50364

    // First, ensure we have SOT at 50258 (if not already present)
    while vocab.len() < special_tokens::SOT as usize {
        vocab.add_token(vec![0]); // placeholder
    }

    // Add SOT (50258) - "<|startoftranscript|>"
    vocab.add_token(b"<|startoftranscript|>".to_vec());

    // Add 99 language tokens (50259-50357)
    let languages = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv",
        "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr",
        "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
        "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
    ];
    for lang in languages {
        vocab.add_token(format!("<|{lang}|>").into_bytes());
    }

    // Add task tokens (50358-50363)
    vocab.add_token(b"<|translate|>".to_vec()); // 50358
    vocab.add_token(b"<|transcribe|>".to_vec()); // 50359
    vocab.add_token(b"<|startoflm|>".to_vec()); // 50360 - speaker turn/startoflm
    vocab.add_token(b"<|startofprev|>".to_vec()); // 50361 - prev
    vocab.add_token(b"<|nospeech|>".to_vec()); // 50362
    vocab.add_token(b"<|notimestamps|>".to_vec()); // 50363

    // Add timestamp tokens (50364-51864 = 1501 tokens for 30 seconds)
    // Each timestamp represents 0.02 seconds
    for i in 0..1501 {
        let seconds = i as f32 * 0.02;
        vocab.add_token(format!("<|{seconds:.2}|>").into_bytes());
    }

    if verbose {
        eprintln!(
            "[INFO] Added {} special tokens (total: {})",
            vocab.len() - current_size,
            vocab.len()
        );
    }

    Ok(vocab)
}

/// Build the GPT-2 byte decoder mapping (Unicode char -> byte value)
///
/// GPT-2 uses a reversible mapping from bytes to printable Unicode characters.
/// This builds the reverse mapping for decoding vocab.json tokens.
fn build_gpt2_byte_decoder() -> std::collections::HashMap<char, u8> {
    use std::collections::HashMap;

    let mut decoder = HashMap::new();
    let mut n = 0u32;

    // Printable ASCII characters map to themselves
    for b in b'!'..=b'~' {
        decoder.insert(char::from(b), b);
    }
    // Extended characters that map to themselves
    for b in 0xa1u8..=0xac {
        decoder.insert(char::from(b), b);
    }
    for b in 0xaeu8..=0xff {
        decoder.insert(char::from(b), b);
    }

    // Non-printable bytes get mapped to Unicode starting at U+0100
    for b in 0u8..=255 {
        if !decoder.values().any(|&v| v == b) {
            // This byte wasn't mapped yet, so it uses offset encoding
            let unicode_char = char::from_u32(256 + n).unwrap_or('?');
            decoder.insert(unicode_char, b);
            n += 1;
        }
    }

    decoder
}

/// Decode a GPT-2 style token string to bytes
fn decode_gpt2_token(token: &str, decoder: &std::collections::HashMap<char, u8>) -> Vec<u8> {
    token
        .chars()
        .filter_map(|c| decoder.get(&c).copied())
        .collect()
}

/// Load mel filters from HuggingFace preprocessor_config.json
///
/// The mel_filters field contains a 2D array [n_mels][n_freqs] with Slaney-normalized
/// triangular filterbank weights. This is crucial for matching HuggingFace's mel
/// spectrogram output exactly.
fn load_mel_filters_from_preprocessor(
    preprocessor_path: &std::path::Path,
    verbose: bool,
) -> ModelLoaderResult<crate::format::MelFilterbankData> {
    // Read and parse preprocessor_config.json
    let json_str = fs::read_to_string(preprocessor_path)?;
    let config: serde_json::Value =
        serde_json::from_str(&json_str).map_err(|e| ModelLoaderError::Download(e.to_string()))?;

    // Extract mel_filters field
    let mel_filters_value = config
        .get("mel_filters")
        .ok_or_else(|| ModelLoaderError::Download("mel_filters not found in config".to_string()))?;

    // Parse as 2D array and flatten to row-major order
    let mel_filters_2d: Vec<Vec<f64>> = serde_json::from_value(mel_filters_value.clone())
        .map_err(|e| ModelLoaderError::Download(format!("Failed to parse mel_filters: {e}")))?;

    let n_mels = mel_filters_2d.len();
    let n_freqs = mel_filters_2d.first().map_or(0, Vec::len);

    if verbose {
        eprintln!(
            "[INFO] Loaded mel filters from preprocessor_config.json: {} x {}",
            n_mels, n_freqs
        );
    }

    // Flatten to row-major Vec<f32>
    let data: Vec<f32> = mel_filters_2d
        .into_iter()
        .flat_map(|row| row.into_iter().map(|v| v as f32))
        .collect();

    Ok(crate::format::MelFilterbankData {
        n_mels: n_mels as u32,
        n_freqs: n_freqs as u32,
        data,
    })
}

/// Load a model from a file path
fn load_model_from_path(path: &std::path::Path) -> ModelLoaderResult<WhisperApr> {
    let bytes = fs::read(path)?;
    WhisperApr::load_from_apr(&bytes).map_err(ModelLoaderError::from)
}

/// Load a model, downloading from HuggingFace if not cached
///
/// # Arguments
///
/// * `size` - Model size to load
/// * `model_path` - Optional explicit path to .apr file
/// * `verbose` - Whether to print progress messages
///
/// # Returns
///
/// Loaded WhisperApr model with weights
pub fn load_or_download_model(
    size: ModelSize,
    model_path: Option<&std::path::Path>,
    verbose: bool,
) -> ModelLoaderResult<WhisperApr> {
    if let Some(path) = model_path {
        if verbose {
            eprintln!("[INFO] Loading model from: {}", path.display());
        }
        return load_model_from_path(path);
    }

    if is_model_cached(size) {
        let cache_path = get_model_cache_path(size);
        if verbose {
            eprintln!("[INFO] Loading cached model: {}", cache_path.display());
        }
        return load_model_from_path(&cache_path);
    }

    if verbose {
        eprintln!("[INFO] Model not cached, downloading...");
    }
    let downloaded_path = download_model(size, verbose)?;
    load_model_from_path(&downloaded_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_cache_dir() {
        let cache_dir = get_cache_dir();
        let path_str = cache_dir.to_string_lossy();
        assert!(
            path_str.contains("whisper-apr"),
            "Cache dir should contain whisper-apr: {}",
            path_str
        );
    }

    #[test]
    fn test_get_model_cache_path() {
        let path = get_model_cache_path(ModelSize::Tiny);
        assert!(path.ends_with("tiny.apr"), "Path should end with tiny.apr");
    }

    #[test]
    fn test_get_hf_repo_id() {
        assert_eq!(get_hf_repo_id(ModelSize::Tiny), "openai/whisper-tiny");
        assert_eq!(get_hf_repo_id(ModelSize::Base), "openai/whisper-base");
        assert_eq!(get_hf_repo_id(ModelSize::Large), "openai/whisper-large-v3");
    }

    #[test]
    fn test_get_model_filename() {
        assert_eq!(get_model_filename(ModelSize::Tiny), "tiny.apr");
        assert_eq!(get_model_filename(ModelSize::Medium), "medium.apr");
    }
}
