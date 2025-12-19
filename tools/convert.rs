//! Whisper model converter: HuggingFace SafeTensors → .apr format
//!
//! Downloads Whisper models from HuggingFace and converts them to the .apr format
//! optimized for WASM deployment.
//!
//! # Usage
//!
//! ```bash
//! # Convert tiny model
//! whisper-convert tiny --output whisper-tiny.apr
//!
//! # Convert with int8 quantization (4x smaller)
//! whisper-convert tiny --quantize int8 --output whisper-tiny-int8.apr
//!
//! # Convert with cache directory
//! whisper-convert base --cache ~/.cache/whisper --output whisper-base.apr
//! ```

use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use safetensors::SafeTensors;
use serde_json::Value;
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::PathBuf;
use whisper_apr::format::{AprWriter, AprWriterInt8, MelFilterbankData, TensorData};
use whisper_apr::model::ModelConfig;
use whisper_apr::tokenizer::Vocabulary;

/// HuggingFace model repository for Whisper
const HF_REPO_BASE: &str = "https://huggingface.co/openai";

/// OpenAI Whisper repository for assets (mel filters)
const OPENAI_WHISPER_ASSETS: &str =
    "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets";

/// Model configurations
#[derive(Debug, Clone, Copy)]
enum ModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}

impl ModelSize {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "tiny" => Some(Self::Tiny),
            "base" => Some(Self::Base),
            "small" => Some(Self::Small),
            "medium" => Some(Self::Medium),
            "large" | "large-v3" => Some(Self::Large),
            _ => None,
        }
    }

    fn hf_repo_name(&self) -> &'static str {
        match self {
            Self::Tiny => "whisper-tiny",
            Self::Base => "whisper-base",
            Self::Small => "whisper-small",
            Self::Medium => "whisper-medium",
            Self::Large => "whisper-large-v3",
        }
    }

    fn to_model_config(&self) -> ModelConfig {
        match self {
            Self::Tiny => ModelConfig::tiny(),
            Self::Base => ModelConfig::base(),
            Self::Small => ModelConfig::small(),
            Self::Medium => ModelConfig::medium(),
            Self::Large => ModelConfig::large(),
        }
    }
}

/// Quantization type for output
#[derive(Debug, Clone, Copy, PartialEq)]
enum QuantizeType {
    None,
    Int8,
}

impl QuantizeType {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" | "f32" => Some(Self::None),
            "int8" | "i8" => Some(Self::Int8),
            _ => None,
        }
    }
}

/// Tensor name mapping from HuggingFace to .apr format
struct TensorMapper {
    mappings: HashMap<String, String>,
}

impl TensorMapper {
    fn new() -> Self {
        let mut mappings = HashMap::new();

        // Encoder mappings
        mappings.insert(
            "encoder.conv1.weight".to_string(),
            "encoder.conv1.weight".to_string(),
        );
        mappings.insert(
            "encoder.conv1.bias".to_string(),
            "encoder.conv1.bias".to_string(),
        );
        mappings.insert(
            "encoder.conv2.weight".to_string(),
            "encoder.conv2.weight".to_string(),
        );
        mappings.insert(
            "encoder.conv2.bias".to_string(),
            "encoder.conv2.bias".to_string(),
        );
        mappings.insert(
            "encoder.embed_positions.weight".to_string(),
            "encoder.positional_embedding".to_string(),
        );

        // Decoder mappings
        mappings.insert(
            "decoder.embed_tokens.weight".to_string(),
            "decoder.token_embedding".to_string(),
        );
        mappings.insert(
            "decoder.embed_positions.weight".to_string(),
            "decoder.positional_embedding".to_string(),
        );

        Self { mappings }
    }

    fn map_name(&self, hf_name: &str) -> String {
        // Strip "model." prefix if present (HuggingFace models use this)
        let name = hf_name.strip_prefix("model.").unwrap_or(hf_name);

        // Check direct mapping (without model. prefix)
        if let Some(apr_name) = self.mappings.get(name) {
            return apr_name.clone();
        }

        // Handle layer patterns
        // encoder.blocks.X.* -> encoder.blocks.X.*
        // decoder.blocks.X.* -> decoder.blocks.X.*
        // (keep blocks naming for now)

        // Default: keep name without model. prefix
        name.to_string()
    }
}

/// Download model file from HuggingFace
async fn download_model(
    client: &Client,
    model_size: ModelSize,
    cache_dir: &PathBuf,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let repo_name = model_size.hf_repo_name();
    let url = format!("{HF_REPO_BASE}/{repo_name}/resolve/main/model.safetensors");

    let cache_file = cache_dir.join(format!("{repo_name}.safetensors"));

    // Check cache
    if cache_file.exists() {
        println!("Using cached model: {}", cache_file.display());
        return Ok(cache_file);
    }

    println!("Downloading {} from HuggingFace...", repo_name);

    let response = client.get(&url).send().await?;
    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
            .progress_chars("#>-"),
    );

    std::fs::create_dir_all(cache_dir)?;
    let bytes = response.bytes().await?;
    pb.finish_with_message("Download complete");

    std::fs::write(&cache_file, &bytes)?;
    println!("Cached to: {}", cache_file.display());

    Ok(cache_file)
}

/// Download vocabulary from HuggingFace
async fn download_vocab(
    client: &Client,
    model_size: ModelSize,
    cache_dir: &PathBuf,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let repo_name = model_size.hf_repo_name();
    let url = format!("{HF_REPO_BASE}/{repo_name}/resolve/main/vocab.json");

    let cache_file = cache_dir.join(format!("{repo_name}.vocab.json"));

    // Check cache
    if cache_file.exists() {
        println!("Using cached vocab: {}", cache_file.display());
        return Ok(cache_file);
    }

    println!("Downloading vocabulary...");
    let response = client.get(&url).send().await?;
    let bytes = response.bytes().await?;

    std::fs::write(&cache_file, &bytes)?;
    println!("Cached vocab to: {}", cache_file.display());

    Ok(cache_file)
}

/// Download merges.txt from HuggingFace
async fn download_merges(
    client: &Client,
    model_size: ModelSize,
    cache_dir: &PathBuf,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let repo_name = model_size.hf_repo_name();
    let url = format!("{HF_REPO_BASE}/{repo_name}/resolve/main/merges.txt");

    let cache_file = cache_dir.join(format!("{repo_name}.merges.txt"));

    // Check cache
    if cache_file.exists() {
        println!("Using cached merges: {}", cache_file.display());
        return Ok(cache_file);
    }

    println!("Downloading merges...");
    let response = client.get(&url).send().await?;
    let bytes = response.bytes().await?;

    std::fs::write(&cache_file, &bytes)?;
    println!("Cached merges to: {}", cache_file.display());

    Ok(cache_file)
}

/// Download added_tokens.json from HuggingFace (contains Whisper special tokens)
async fn download_added_tokens(
    client: &Client,
    model_size: ModelSize,
    cache_dir: &PathBuf,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let repo_name = model_size.hf_repo_name();
    let url = format!("{HF_REPO_BASE}/{repo_name}/resolve/main/added_tokens.json");

    let cache_file = cache_dir.join(format!("{repo_name}.added_tokens.json"));

    // Check cache
    if cache_file.exists() {
        println!("Using cached added_tokens: {}", cache_file.display());
        return Ok(cache_file);
    }

    println!("Downloading added_tokens...");
    let response = client.get(&url).send().await?;
    let bytes = response.bytes().await?;

    std::fs::write(&cache_file, &bytes)?;
    println!("Cached added_tokens to: {}", cache_file.display());

    Ok(cache_file)
}

/// Download mel_filters.npz from OpenAI Whisper repository
async fn download_mel_filters(
    client: &Client,
    cache_dir: &PathBuf,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let url = format!("{OPENAI_WHISPER_ASSETS}/mel_filters.npz");
    let cache_file = cache_dir.join("mel_filters.npz");

    // Check cache
    if cache_file.exists() {
        println!("Using cached mel_filters: {}", cache_file.display());
        return Ok(cache_file);
    }

    println!("Downloading mel_filters.npz from OpenAI...");
    let response = client.get(&url).send().await?;
    let bytes = response.bytes().await?;

    std::fs::write(&cache_file, &bytes)?;
    println!("Cached mel_filters to: {}", cache_file.display());

    Ok(cache_file)
}

/// Parse mel filterbank from NPZ file
///
/// NPZ is a ZIP archive containing .npy files.
/// NPY format: magic(6) + version(2) + header_len(2) + header + raw_data
fn parse_mel_filterbank(
    npz_path: &PathBuf,
    n_mels: u32,
) -> Result<MelFilterbankData, Box<dyn std::error::Error>> {
    let npy_name = format!("mel_{n_mels}.npy");
    println!("Extracting {npy_name} from NPZ...");

    let npz_data = std::fs::read(npz_path)?;
    let reader = Cursor::new(&npz_data);
    let mut archive = zip::ZipArchive::new(reader)?;

    let mut npy_file = archive.by_name(&npy_name)?;
    let mut npy_data = Vec::new();
    npy_file.read_to_end(&mut npy_data)?;

    // Parse NPY format
    // Magic: \x93NUMPY (6 bytes)
    if npy_data.len() < 10 || &npy_data[0..6] != b"\x93NUMPY" {
        return Err("Invalid NPY magic".into());
    }

    let version_major = npy_data[6];
    let version_minor = npy_data[7];
    println!("  NPY version: {version_major}.{version_minor}");

    // Header length (depends on version)
    let (header_len, data_offset) = if version_major == 1 {
        let len = u16::from_le_bytes([npy_data[8], npy_data[9]]) as usize;
        (len, 10 + len)
    } else {
        // Version 2+ uses u32
        let len =
            u32::from_le_bytes([npy_data[8], npy_data[9], npy_data[10], npy_data[11]]) as usize;
        (len, 12 + len)
    };

    println!("  Header length: {header_len}, data offset: {data_offset}");

    // Raw f32 data starts at data_offset
    let raw_data = &npy_data[data_offset..];
    let n_freqs = 201u32; // Whisper uses n_fft=400, so n_fft/2+1 = 201
    let expected_bytes = (n_mels * n_freqs) as usize * 4;

    if raw_data.len() < expected_bytes {
        return Err(format!(
            "NPY data too short: expected {expected_bytes} bytes, got {}",
            raw_data.len()
        )
        .into());
    }

    // Parse f32 values (little-endian)
    let data: Vec<f32> = raw_data[..expected_bytes]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    println!("  Parsed {} floats ({}×{})", data.len(), n_mels, n_freqs);

    // Verify slaney normalization (each row should sum to ~0.025)
    let row_sum: f32 = data[0..n_freqs as usize].iter().sum();
    println!("  First row sum: {row_sum:.6} (expected ~0.025 for slaney normalization)");

    Ok(MelFilterbankData::new(n_mels, n_freqs, data))
}

/// Parse vocab.json and added_tokens.json to create Vocabulary
fn parse_vocabulary(
    vocab_path: &PathBuf,
    merges_path: &PathBuf,
    added_tokens_path: &PathBuf,
) -> Result<Vocabulary, Box<dyn std::error::Error>> {
    println!("Parsing vocabulary...");

    // Read vocab.json (base GPT-2 vocabulary)
    let vocab_data = std::fs::read_to_string(vocab_path)?;
    let vocab_json: Value = serde_json::from_str(&vocab_data)?;

    // Get token-to-id mapping from base vocab
    let vocab_map = vocab_json
        .as_object()
        .ok_or("vocab.json should be an object")?;

    // Create sorted list of (id, token_string) pairs
    let mut tokens: Vec<(u32, String)> = vocab_map
        .iter()
        .filter_map(|(token_str, id_val)| {
            let id = id_val.as_u64()? as u32;
            Some((id, token_str.clone()))
        })
        .collect();

    // Read added_tokens.json (Whisper special tokens: SOT, language tokens, etc.)
    let added_data = std::fs::read_to_string(added_tokens_path)?;
    let added_json: Value = serde_json::from_str(&added_data)?;

    // added_tokens.json format: {"<|token|>": id, ...}
    if let Some(added_map) = added_json.as_object() {
        let added_count = added_map.len();
        for (token_str, id_val) in added_map {
            if let Some(id) = id_val.as_u64() {
                tokens.push((id as u32, token_str.clone()));
            }
        }
        println!("  Found {} added special tokens", added_count);
    }

    // Sort by ID to ensure correct ordering
    tokens.sort_by_key(|(id, _)| *id);

    println!("  Total tokens (base + added): {}", tokens.len());

    // Create vocabulary with tokens in ID order
    let mut vocab = Vocabulary::new();

    for (expected_id, token_str) in &tokens {
        // Convert token string to bytes
        // For special tokens like <|startoftranscript|>, keep as-is
        // For regular tokens, use GPT-2 byte decoding
        let token_bytes = if token_str.starts_with("<|") && token_str.ends_with("|>") {
            // Special token - store as UTF-8 bytes directly
            token_str.as_bytes().to_vec()
        } else {
            // Regular token - use GPT-2 byte decoding
            gpt2_decode_token(token_str)
        };

        let actual_id = vocab.add_token(token_bytes);

        // Verify ordering
        if actual_id != *expected_id {
            return Err(format!(
                "Token ordering mismatch: expected id={}, got id={} for token '{}'",
                expected_id, actual_id, token_str
            )
            .into());
        }
    }

    // Read merges.txt
    let merges_data = std::fs::read_to_string(merges_path)?;
    let mut merge_count = 0;

    for line in merges_data.lines() {
        // Skip header/comments
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        // Format: "first second"
        let parts: Vec<&str> = line.split(' ').collect();
        if parts.len() != 2 {
            continue;
        }

        let first_bytes = gpt2_decode_token(parts[0]);
        let second_bytes = gpt2_decode_token(parts[1]);
        vocab.add_merge(first_bytes, second_bytes);
        merge_count += 1;
    }

    println!("  Added {} merge rules", merge_count);
    println!("  Final vocabulary size: {}", vocab.len());

    Ok(vocab)
}

/// Decode GPT-2 style token string to bytes
///
/// GPT-2/Whisper uses a special encoding where:
/// - Regular printable ASCII chars are themselves
/// - Ġ (U+0120) represents a leading space
/// - Special bytes 0-255 are encoded as Unicode chars in range U+0100-U+01FF (minus some)
fn gpt2_decode_token(token: &str) -> Vec<u8> {
    // Build GPT-2 byte decoder mapping
    // This mapping converts Unicode codepoints back to bytes
    let mut bytes = Vec::with_capacity(token.len());

    for c in token.chars() {
        let codepoint = c as u32;

        // GPT-2 uses specific Unicode codepoints to represent bytes
        // The mapping is complex, so we handle the common cases:
        match codepoint {
            // Ġ (U+0120) = space (0x20)
            0x0120 => bytes.push(0x20),
            // Ċ (U+010A) = newline (0x0A)
            0x010A => bytes.push(0x0A),
            // Ģ (U+0122) = " (0x22)
            0x0122 => bytes.push(0x22),
            // Standard printable ASCII (0x21-0x7E, excluding some)
            0x21..=0x7E => bytes.push(codepoint as u8),
            // Bytes 0x80-0xFF map to U+0100-U+017F (Latin Extended-A range)
            0x00A1..=0x00AC => bytes.push((codepoint - 0x00A1 + 0x21) as u8),
            0x00AE..=0x00FF => bytes.push((codepoint - 0x00AE + 0x2D) as u8),
            // U+0100-U+013F range maps to various bytes
            0x0100..=0x013F => {
                // Complex GPT-2 mapping - decode based on known patterns
                let byte = gpt2_codepoint_to_byte(codepoint);
                bytes.push(byte);
            }
            // Other characters - use UTF-8 encoding
            _ => {
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                bytes.extend_from_slice(s.as_bytes());
            }
        }
    }

    bytes
}

/// Convert GPT-2 codepoint to byte
fn gpt2_codepoint_to_byte(codepoint: u32) -> u8 {
    // GPT-2 byte encoding table (reverse of encoding table)
    // The encoding maps bytes 0-255 to specific Unicode codepoints
    // This function reverses that mapping
    match codepoint {
        0x0100 => 0x00, // NUL
        0x0101 => 0x01,
        0x0102 => 0x02,
        0x0103 => 0x03,
        0x0104 => 0x04,
        0x0105 => 0x05,
        0x0106 => 0x06,
        0x0107 => 0x07,
        0x0108 => 0x08,
        0x0109 => 0x09, // TAB
        0x010A => 0x0A, // LF (Ċ)
        0x010B => 0x0B,
        0x010C => 0x0C,
        0x010D => 0x0D, // CR
        0x010E => 0x0E,
        0x010F => 0x0F,
        0x0110 => 0x10,
        0x0111 => 0x11,
        0x0112 => 0x12,
        0x0113 => 0x13,
        0x0114 => 0x14,
        0x0115 => 0x15,
        0x0116 => 0x16,
        0x0117 => 0x17,
        0x0118 => 0x18,
        0x0119 => 0x19,
        0x011A => 0x1A,
        0x011B => 0x1B,
        0x011C => 0x1C,
        0x011D => 0x1D,
        0x011E => 0x1E,
        0x011F => 0x1F,
        0x0120 => 0x20, // Space (Ġ)
        0x0121 => 0x7F, // DEL
        0x0122 => 0x80,
        0x0123 => 0x81,
        0x0124 => 0x82,
        0x0125 => 0x83,
        0x0126 => 0x84,
        0x0127 => 0x85,
        0x0128 => 0x86,
        0x0129 => 0x87,
        0x012A => 0x88,
        0x012B => 0x89,
        0x012C => 0x8A,
        0x012D => 0x8B,
        0x012E => 0x8C,
        0x012F => 0x8D,
        0x0130 => 0x8E,
        0x0131 => 0x8F,
        0x0132 => 0x90,
        0x0133 => 0x91,
        0x0134 => 0x92,
        0x0135 => 0x93,
        0x0136 => 0x94,
        0x0137 => 0x95,
        0x0138 => 0x96,
        0x0139 => 0x97,
        0x013A => 0x98,
        0x013B => 0x99,
        0x013C => 0x9A,
        0x013D => 0x9B,
        0x013E => 0x9C,
        0x013F => 0x9D,
        _ => codepoint as u8, // Fallback
    }
}

/// Parsed tensor from SafeTensors
struct ParsedTensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// Load and parse all tensors from SafeTensors file
fn load_tensors(
    safetensors_path: &PathBuf,
) -> Result<Vec<ParsedTensor>, Box<dyn std::error::Error>> {
    println!("Loading SafeTensors...");
    let data = std::fs::read(safetensors_path)?;
    let tensors = SafeTensors::deserialize(&data)?;
    let mapper = TensorMapper::new();

    println!("Parsing {} tensors...", tensors.names().len());
    let pb = ProgressBar::new(tensors.names().len() as u64);

    let mut result = Vec::new();

    for name in tensors.names() {
        let tensor = tensors.tensor(name)?;
        let apr_name = mapper.map_name(name);

        // Convert tensor data to f32
        let f32_data: Vec<f32> = match tensor.dtype() {
            safetensors::Dtype::F32 => tensor
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            safetensors::Dtype::F16 => {
                // Convert f16 to f32
                tensor
                    .data()
                    .chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        half_to_f32(bits)
                    })
                    .collect()
            }
            safetensors::Dtype::BF16 => {
                // Convert bf16 to f32
                tensor
                    .data()
                    .chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        bf16_to_f32(bits)
                    })
                    .collect()
            }
            dtype => {
                eprintln!(
                    "Warning: Unsupported dtype {:?} for {}, skipping",
                    dtype, name
                );
                pb.inc(1);
                continue;
            }
        };

        let shape: Vec<usize> = tensor.shape().iter().map(|&d| d).collect();
        result.push(ParsedTensor {
            name: apr_name,
            shape,
            data: f32_data,
        });
        pb.inc(1);
    }

    pb.finish_with_message("Parsing complete");
    Ok(result)
}

/// Convert SafeTensors to .apr format (f32)
fn convert_to_apr_f32(
    tensors: Vec<ParsedTensor>,
    model_size: ModelSize,
    vocab: Option<Vocabulary>,
    filterbank: Option<MelFilterbankData>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let config = model_size.to_model_config();
    let mut writer = AprWriter::from_config(&config);

    println!("Converting {} tensors to f32...", tensors.len());
    let pb = ProgressBar::new(tensors.len() as u64);

    for tensor in tensors {
        writer.add_tensor(TensorData::new(tensor.name, tensor.shape, tensor.data));
        pb.inc(1);
    }

    pb.finish_with_message("Conversion complete");

    // Embed vocabulary if provided
    if let Some(v) = vocab {
        println!("Embedding vocabulary ({} tokens)...", v.len());
        writer.set_vocabulary(v);
    }

    // Embed mel filterbank if provided
    if let Some(fb) = filterbank {
        println!(
            "Embedding mel filterbank ({}×{} = {} floats)...",
            fb.n_mels,
            fb.n_freqs,
            fb.data.len()
        );
        writer.set_mel_filterbank(fb);
    }

    println!("Serializing to .apr format (f32)...");
    let bytes = writer.to_bytes()?;
    println!(
        "Output size: {:.2} MB ({} tensors, vocab={}, filterbank={})",
        bytes.len() as f64 / 1_000_000.0,
        writer.n_tensors(),
        writer.has_vocabulary(),
        writer.has_mel_filterbank()
    );

    Ok(bytes)
}

/// Convert SafeTensors to .apr format (int8 quantized)
fn convert_to_apr_int8(
    tensors: Vec<ParsedTensor>,
    model_size: ModelSize,
    vocab: Option<Vocabulary>,
    filterbank: Option<MelFilterbankData>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let config = model_size.to_model_config();
    let mut writer = AprWriterInt8::from_config(&config);

    println!("Quantizing {} tensors to int8...", tensors.len());
    let pb = ProgressBar::new(tensors.len() as u64);

    for tensor in tensors {
        writer.add_tensor_f32(tensor.name, tensor.shape, &tensor.data);
        pb.inc(1);
    }

    pb.finish_with_message("Quantization complete");

    // Embed vocabulary if provided
    if let Some(v) = vocab {
        println!("Embedding vocabulary ({} tokens)...", v.len());
        writer.set_vocabulary(v);
    }

    // Embed mel filterbank if provided
    if let Some(fb) = filterbank {
        println!(
            "Embedding mel filterbank ({}×{} = {} floats)...",
            fb.n_mels,
            fb.n_freqs,
            fb.data.len()
        );
        writer.set_mel_filterbank(fb);
    }

    println!("Serializing to .apr format (int8)...");
    let bytes = writer.to_bytes()?;
    println!(
        "Output size: {:.2} MB ({} tensors, vocab={}, filterbank={})",
        bytes.len() as f64 / 1_000_000.0,
        writer.n_tensors(),
        writer.has_vocabulary(),
        writer.has_mel_filterbank()
    );

    Ok(bytes)
}

/// Convert f16 bits to f32
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let mut e = 1u32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let exp32 = 127 - 15 - e + 1;
        let mant32 = (m & 0x3FF) << 13;
        return f32::from_bits((sign << 31) | (exp32 << 23) | mant32);
    }

    if exp == 31 {
        if mant == 0 {
            return f32::from_bits((sign << 31) | (0xFF << 23));
        }
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }

    let exp32 = exp + 127 - 15;
    let mant32 = mant << 13;
    f32::from_bits((sign << 31) | (exp32 << 23) | mant32)
}

/// Convert bf16 bits to f32
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Parsed command-line arguments
struct CliArgs {
    model_size: ModelSize,
    model_str: String,
    output_path: PathBuf,
    cache_dir: PathBuf,
    quantize: QuantizeType,
}

/// Print usage information
fn print_usage() {
    eprintln!(
        r#"whisper-convert: Convert Whisper models to .apr format

USAGE:
    whisper-convert <MODEL> [OPTIONS]

MODELS:
    tiny        39M parameters (fastest, lowest quality)
    base        74M parameters
    small       244M parameters
    medium      769M parameters
    large       1.5B parameters (slowest, highest quality)

OPTIONS:
    --output, -o <PATH>     Output .apr file path (default: whisper-<model>.apr)
    --quantize, -q <TYPE>   Quantization type: none, int8 (default: none)
    --cache <DIR>           Cache directory for downloaded models
                            (default: ~/.cache/whisper-apr)
    --help, -h              Print this help message

QUANTIZATION:
    none/f32    Full precision (default) - largest size, best quality
    int8/i8     8-bit integer - ~4x smaller, minor quality loss

EXAMPLES:
    whisper-convert tiny
    whisper-convert tiny --quantize int8 -o whisper-tiny-int8.apr
    whisper-convert base --output my-model.apr
    whisper-convert small --cache /tmp/models
"#
    );
}

/// Parse a single argument and update state
fn parse_arg(
    arg: &str,
    args: &[String],
    idx: &mut usize,
    model_str: &mut Option<String>,
    output_path: &mut Option<PathBuf>,
    cache_dir: &mut Option<PathBuf>,
    quantize: &mut QuantizeType,
) -> Result<bool, Box<dyn std::error::Error>> {
    match arg {
        "-h" | "--help" => {
            print_usage();
            std::process::exit(0);
        }
        "-o" | "--output" => {
            *idx += 1;
            if *idx >= args.len() {
                return Err("--output requires a path argument".into());
            }
            *output_path = Some(PathBuf::from(&args[*idx]));
        }
        "-q" | "--quantize" => {
            *idx += 1;
            if *idx >= args.len() {
                return Err("--quantize requires a type argument (none, int8)".into());
            }
            *quantize = QuantizeType::from_str(&args[*idx])
                .ok_or_else(|| format!("Unknown quantization type: {}", args[*idx]))?;
        }
        "--cache" => {
            *idx += 1;
            if *idx >= args.len() {
                return Err("--cache requires a directory argument".into());
            }
            *cache_dir = Some(PathBuf::from(&args[*idx]));
        }
        a if !a.starts_with('-') => {
            *model_str = Some(a.to_string());
        }
        a => {
            return Err(format!("Unknown option: {}", a).into());
        }
    }
    Ok(true)
}

/// Parse command-line arguments
fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let mut model_str: Option<String> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut cache_dir: Option<PathBuf> = None;
    let mut quantize = QuantizeType::None;

    let mut i = 1;
    while i < args.len() {
        parse_arg(
            &args[i],
            &args,
            &mut i,
            &mut model_str,
            &mut output_path,
            &mut cache_dir,
            &mut quantize,
        )?;
        i += 1;
    }

    let model_str = model_str.ok_or("No model specified")?;
    let model_size =
        ModelSize::from_str(&model_str).ok_or_else(|| format!("Unknown model: {}", model_str))?;

    let cache_dir = cache_dir.unwrap_or_else(|| {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("whisper-apr")
    });

    let output_path = output_path.unwrap_or_else(|| {
        let suffix = match quantize {
            QuantizeType::None => "",
            QuantizeType::Int8 => "-int8",
        };
        PathBuf::from(format!(
            "whisper-{}{}.apr",
            model_str.to_lowercase(),
            suffix
        ))
    });

    Ok(CliArgs {
        model_size,
        model_str,
        output_path,
        cache_dir,
        quantize,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;

    let quant_str = match args.quantize {
        QuantizeType::None => "f32 (full precision)",
        QuantizeType::Int8 => "int8 (quantized)",
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Whisper Model Converter (HuggingFace → .apr)       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Model:      {:48} ║", args.model_str);
    println!("║  Quantize:   {:48} ║", quant_str);
    println!("║  Output:     {:48} ║", args.output_path.display());
    println!("║  Cache:      {:48} ║", args.cache_dir.display());
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let client = Client::new();

    // Download model
    let safetensors_path = download_model(&client, args.model_size, &args.cache_dir).await?;

    // Download vocabulary files (base vocab, merges, and Whisper special tokens)
    let vocab_path = download_vocab(&client, args.model_size, &args.cache_dir).await?;
    let merges_path = download_merges(&client, args.model_size, &args.cache_dir).await?;
    let added_tokens_path =
        download_added_tokens(&client, args.model_size, &args.cache_dir).await?;

    // Download mel filterbank from OpenAI (slaney-normalized for exact match)
    let mel_filters_path = download_mel_filters(&client, &args.cache_dir).await?;

    // Parse vocabulary (includes all Whisper special tokens)
    let vocab = parse_vocabulary(&vocab_path, &merges_path, &added_tokens_path)?;

    // Parse mel filterbank (use n_mels from model config)
    // Large-v3 uses 128 mel bands, all others use 80
    let n_mels = match args.model_size {
        ModelSize::Large => 128,
        _ => 80,
    };
    let filterbank = parse_mel_filterbank(&mel_filters_path, n_mels)?;

    // Load tensors
    let tensors = load_tensors(&safetensors_path)?;

    // Convert to .apr with appropriate quantization (including vocabulary and filterbank)
    let apr_bytes = match args.quantize {
        QuantizeType::None => {
            convert_to_apr_f32(tensors, args.model_size, Some(vocab), Some(filterbank))?
        }
        QuantizeType::Int8 => {
            convert_to_apr_int8(tensors, args.model_size, Some(vocab), Some(filterbank))?
        }
    };

    // Write output
    std::fs::write(&args.output_path, &apr_bytes)?;
    println!("\n✅ Successfully wrote: {}", args.output_path.display());
    println!("   Size: {:.2} MB", apr_bytes.len() as f64 / 1_000_000.0);

    Ok(())
}
