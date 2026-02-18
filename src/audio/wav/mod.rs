//! WAV file parsing module
//!
//! Provides functions for parsing WAV audio files into f32 samples.

use crate::error::{WhisperError, WhisperResult};

#[cfg(test)]
mod tests;

// WAVE format codes
const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// WAV parsing error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WavError {
    /// File is too small to contain a valid WAV header
    TooSmall,
    /// Missing RIFF header at start of file
    MissingRiff,
    /// Missing WAVE marker in header
    MissingWave,
    /// fmt chunk is truncated
    FmtTruncated,
    /// Unsupported audio format or bit depth
    UnsupportedFormat {
        /// Audio format code (1=PCM, 3=float)
        format: u16,
        /// Bits per sample
        bits: u16,
    },
    /// Unsupported number of audio channels
    UnsupportedChannels(u16),
    /// No data chunk found in file
    NoDataChunk,
}

impl std::fmt::Display for WavError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooSmall => write!(f, "WAV file too small"),
            Self::MissingRiff => write!(f, "Invalid WAV: missing RIFF header"),
            Self::MissingWave => write!(f, "Invalid WAV: missing WAVE marker"),
            Self::FmtTruncated => write!(f, "Invalid WAV: fmt chunk truncated"),
            Self::UnsupportedFormat { format, bits } => {
                write!(f, "Unsupported format: {bits} bits, format {format}")
            }
            Self::UnsupportedChannels(ch) => write!(f, "Unsupported channel count: {ch}"),
            Self::NoDataChunk => write!(f, "Invalid WAV: no data chunk found"),
        }
    }
}

impl std::error::Error for WavError {}

impl From<WavError> for WhisperError {
    fn from(e: WavError) -> Self {
        Self::Audio(e.to_string())
    }
}

/// Result of parsing a WAV file
#[derive(Debug, Clone, PartialEq)]
pub struct WavData {
    /// Audio samples normalized to [-1, 1]
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Original number of channels (before mono conversion)
    pub original_channels: u16,
    /// Original bits per sample
    pub bits_per_sample: u16,
}

/// Parsed fmt chunk data
struct FmtChunk {
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    sub_format: u16,
}

/// Parse fmt chunk from WAV data
fn parse_fmt_chunk(data: &[u8], pos: usize, chunk_size: usize) -> Result<FmtChunk, WavError> {
    if pos + 8 + chunk_size > data.len() {
        return Err(WavError::FmtTruncated);
    }

    let audio_format = u16::from_le_bytes([data[pos + 8], data[pos + 9]]);
    let channels = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
    let sample_rate = u32::from_le_bytes([
        data[pos + 12],
        data[pos + 13],
        data[pos + 14],
        data[pos + 15],
    ]);
    let bits_per_sample = u16::from_le_bytes([data[pos + 22], data[pos + 23]]);

    // Handle WAVE_FORMAT_EXTENSIBLE (0xFFFE)
    let sub_format = if audio_format == WAVE_FORMAT_EXTENSIBLE && chunk_size >= 40 {
        let sub_format_offset = pos + 8 + 24;
        if sub_format_offset + 2 <= data.len() {
            u16::from_le_bytes([data[sub_format_offset], data[sub_format_offset + 1]])
        } else {
            0
        }
    } else {
        0
    };

    Ok(FmtChunk {
        audio_format,
        channels,
        sample_rate,
        bits_per_sample,
        sub_format,
    })
}

/// Convert audio data to f32 samples based on format
fn convert_audio_samples(audio_data: &[u8], fmt: &FmtChunk) -> Result<Vec<f32>, WavError> {
    let effective_format = if fmt.audio_format == WAVE_FORMAT_EXTENSIBLE {
        fmt.sub_format
    } else {
        fmt.audio_format
    };

    match (effective_format, fmt.bits_per_sample) {
        (WAVE_FORMAT_PCM, 16) => Ok(convert_16bit_pcm(audio_data)),
        (WAVE_FORMAT_PCM, 8) => Ok(convert_8bit_pcm(audio_data)),
        (WAVE_FORMAT_PCM, 24) => Ok(convert_24bit_pcm(audio_data)),
        (WAVE_FORMAT_PCM, 32) => Ok(convert_32bit_pcm(audio_data)),
        (WAVE_FORMAT_IEEE_FLOAT, 32) => Ok(convert_32bit_float(audio_data)),
        _ => Err(WavError::UnsupportedFormat {
            format: fmt.audio_format,
            bits: fmt.bits_per_sample,
        }),
    }
}

/// Validate WAV header (RIFF + WAVE markers)
fn validate_wav_header(data: &[u8]) -> Result<(), WavError> {
    if data.len() < 44 {
        return Err(WavError::TooSmall);
    }
    match (data.get(0..4), data.get(8..12)) {
        (Some(b"RIFF"), Some(b"WAVE")) => Ok(()),
        (Some(b"RIFF"), _) => Err(WavError::MissingWave),
        _ => Err(WavError::MissingRiff),
    }
}

/// Process data chunk and return WavData
fn process_data_chunk(
    data: &[u8],
    pos: usize,
    chunk_size: usize,
    fmt: &FmtChunk,
) -> Result<WavData, WavError> {
    let data_start = pos + 8;
    let data_end = (data_start + chunk_size).min(data.len());
    let audio_data = &data[data_start..data_end];

    let samples = convert_audio_samples(audio_data, fmt)?;
    let mono_samples = convert_to_mono(samples, fmt.channels)?;

    Ok(WavData {
        samples: mono_samples,
        sample_rate: fmt.sample_rate,
        original_channels: fmt.channels,
        bits_per_sample: fmt.bits_per_sample,
    })
}

/// Parse a WAV file and return f32 samples normalized to [-1, 1]
///
/// Supports 8/16/24/32-bit PCM and 32-bit float, mono and stereo.
#[cfg_attr(feature = "tracing", tracing::instrument(level = "info", skip(data), fields(data_len = data.len())))]
pub fn parse_wav(data: &[u8]) -> Result<WavData, WavError> {
    validate_wav_header(data)?;

    let mut pos = 12;
    let mut fmt_chunk: Option<FmtChunk> = None;

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                as usize;

        if chunk_id == b"fmt " {
            fmt_chunk = Some(parse_fmt_chunk(data, pos, chunk_size)?);
            pos += 8 + chunk_size;
        } else if chunk_id == b"data" {
            let fmt = fmt_chunk.as_ref().ok_or(WavError::NoDataChunk)?;
            return process_data_chunk(data, pos, chunk_size, fmt);
        } else {
            pos += 8 + chunk_size + (chunk_size % 2);
        }
    }

    Err(WavError::NoDataChunk)
}

/// Parse WAV file and return WhisperResult for easier integration
pub fn parse_wav_file(data: &[u8]) -> WhisperResult<WavData> {
    parse_wav(data).map_err(WhisperError::from)
}

/// Convert 16-bit signed PCM to f32
#[inline]
pub(crate) fn convert_16bit_pcm(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect()
}

/// Convert 8-bit unsigned PCM to f32
#[inline]
pub(crate) fn convert_8bit_pcm(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect()
}

/// Convert 24-bit signed PCM to f32
#[inline]
pub(crate) fn convert_24bit_pcm(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(3)
        .map(|chunk| {
            let sign_extend = if chunk[2] & 0x80 != 0 { 0xFF } else { 0x00 };
            let sample = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], sign_extend]);
            sample as f32 / 8_388_608.0
        })
        .collect()
}

/// Convert 32-bit signed PCM to f32
#[inline]
pub(crate) fn convert_32bit_pcm(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| {
            let sample = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            sample as f32 / 2_147_483_648.0
        })
        .collect()
}

/// Convert 32-bit float to f32
#[inline]
fn convert_32bit_float(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Convert multi-channel audio to mono
///
/// Delegates to `aprender::audio::stereo_to_mono` for 2-channel input.
#[inline]
pub(crate) fn convert_to_mono(samples: Vec<f32>, channels: u16) -> Result<Vec<f32>, WavError> {
    match channels {
        1 => Ok(samples),
        2 => Ok(aprender::audio::stereo_to_mono(&samples)),
        _ => Err(WavError::UnsupportedChannels(channels)),
    }
}

/// Resample audio from source rate to target rate using linear interpolation
///
/// # Arguments
/// * `samples` - Input audio samples
/// * `source_rate` - Source sample rate in Hz
/// * `target_rate` - Target sample rate in Hz
///
/// # Returns
/// Resampled audio at target sample rate
#[cfg_attr(feature = "tracing", tracing::instrument(level = "info", skip(samples), fields(samples_len = samples.len())))]
pub fn resample(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = source_rate as f64 / target_rate as f64;
    let output_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = (src_pos - src_idx as f64) as f32;

        let s0 = samples.get(src_idx).copied().unwrap_or(0.0);
        let s1 = samples.get(src_idx + 1).copied().unwrap_or(s0);
        output.push(s0.mul_add(1.0 - frac, s1 * frac));
    }

    output
}
