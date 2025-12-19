//! WAV file parsing module
//!
//! Provides functions for parsing WAV audio files into f32 samples.

use crate::error::{WhisperError, WhisperResult};

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

/// Parse a WAV file and return f32 samples normalized to [-1, 1]
///
/// Supports:
/// - 8-bit unsigned PCM
/// - 16-bit signed PCM
/// - 24-bit signed PCM
/// - 32-bit signed PCM
/// - 32-bit float
/// - Mono and stereo (stereo is converted to mono)
///
/// # Arguments
/// * `data` - Raw bytes of the WAV file
///
/// # Returns
/// * `Ok(WavData)` - Parsed audio data with samples and metadata
/// * `Err(WavError)` - If the file cannot be parsed
///
/// # Example
/// ```ignore
/// use whisper_apr::audio::wav::parse_wav;
///
/// let wav_bytes = std::fs::read("audio.wav")?;
/// let wav_data = parse_wav(&wav_bytes)?;
/// println!("Sample rate: {}Hz, {} samples", wav_data.sample_rate, wav_data.samples.len());
/// ```
#[cfg_attr(feature = "tracing", tracing::instrument(level = "info", skip(data), fields(data_len = data.len())))]
pub fn parse_wav(data: &[u8]) -> Result<WavData, WavError> {
    // Check minimum size for header
    if data.len() < 44 {
        return Err(WavError::TooSmall);
    }

    // Check RIFF header
    if &data[0..4] != b"RIFF" {
        return Err(WavError::MissingRiff);
    }

    if &data[8..12] != b"WAVE" {
        return Err(WavError::MissingWave);
    }

    // Find fmt chunk
    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut audio_format = 0u16;
    let mut sub_format = 0u16; // For WAVE_FORMAT_EXTENSIBLE

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                as usize;

        if chunk_id == b"fmt " {
            if pos + 8 + chunk_size > data.len() {
                return Err(WavError::FmtTruncated);
            }
            audio_format = u16::from_le_bytes([data[pos + 8], data[pos + 9]]);
            channels = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
            sample_rate = u32::from_le_bytes([
                data[pos + 12],
                data[pos + 13],
                data[pos + 14],
                data[pos + 15],
            ]);
            bits_per_sample = u16::from_le_bytes([data[pos + 22], data[pos + 23]]);

            // Handle WAVE_FORMAT_EXTENSIBLE (0xFFFE)
            // Extension starts at offset 24 from fmt chunk data:
            // - cbSize (2 bytes) at offset 24
            // - wValidBitsPerSample (2 bytes) at offset 26
            // - dwChannelMask (4 bytes) at offset 28
            // - SubFormat GUID (16 bytes) at offset 32, first 2 bytes are the actual format
            if audio_format == WAVE_FORMAT_EXTENSIBLE && chunk_size >= 40 {
                // SubFormat is at fmt_data[32:34] which is pos + 8 + 32
                let sub_format_offset = pos + 8 + 24;
                if sub_format_offset + 2 <= data.len() {
                    sub_format =
                        u16::from_le_bytes([data[sub_format_offset], data[sub_format_offset + 1]]);
                }
            }

            pos += 8 + chunk_size;
        } else if chunk_id == b"data" {
            // Found data chunk
            let data_start = pos + 8;
            let data_end = (data_start + chunk_size).min(data.len());
            let audio_data = &data[data_start..data_end];

            // Determine effective format for conversion
            // For WAVE_FORMAT_EXTENSIBLE, use the sub-format from the GUID
            let effective_format = if audio_format == WAVE_FORMAT_EXTENSIBLE {
                sub_format
            } else {
                audio_format
            };

            // Validate format - only PCM (1) and float (3) supported
            if effective_format != WAVE_FORMAT_PCM && effective_format != WAVE_FORMAT_IEEE_FLOAT {
                return Err(WavError::UnsupportedFormat {
                    format: audio_format,
                    bits: bits_per_sample,
                });
            }

            // Convert to f32 based on format
            let samples: Vec<f32> = match (effective_format, bits_per_sample) {
                (WAVE_FORMAT_PCM, 16) => convert_16bit_pcm(audio_data),
                (WAVE_FORMAT_PCM, 8) => convert_8bit_pcm(audio_data),
                (WAVE_FORMAT_PCM, 24) => convert_24bit_pcm(audio_data),
                (WAVE_FORMAT_PCM, 32) => convert_32bit_pcm(audio_data),
                (WAVE_FORMAT_IEEE_FLOAT, 32) => convert_32bit_float(audio_data),
                _ => {
                    return Err(WavError::UnsupportedFormat {
                        format: audio_format,
                        bits: bits_per_sample,
                    });
                }
            };

            // Convert stereo to mono if needed
            let mono_samples = convert_to_mono(samples, channels)?;

            return Ok(WavData {
                samples: mono_samples,
                sample_rate,
                original_channels: channels,
                bits_per_sample,
            });
        } else {
            // Skip unknown chunk
            pos += 8 + chunk_size;
            // Align to even boundary (WAV spec)
            if chunk_size % 2 != 0 {
                pos += 1;
            }
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
fn convert_16bit_pcm(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect()
}

/// Convert 8-bit unsigned PCM to f32
#[inline]
fn convert_8bit_pcm(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect()
}

/// Convert 24-bit signed PCM to f32
#[inline]
fn convert_24bit_pcm(data: &[u8]) -> Vec<f32> {
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
fn convert_32bit_pcm(data: &[u8]) -> Vec<f32> {
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
#[inline]
fn convert_to_mono(samples: Vec<f32>, channels: u16) -> Result<Vec<f32>, WavError> {
    match channels {
        1 => Ok(samples),
        2 => Ok(samples
            .chunks_exact(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect()),
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
    if source_rate == target_rate {
        return samples.to_vec();
    }

    if samples.is_empty() {
        return Vec::new();
    }

    let ratio = source_rate as f64 / target_rate as f64;
    let output_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = (src_pos - src_idx as f64) as f32;

        let sample = if src_idx + 1 < samples.len() {
            samples[src_idx].mul_add(1.0 - frac, samples[src_idx + 1] * frac)
        } else if src_idx < samples.len() {
            samples[src_idx]
        } else {
            0.0
        };
        output.push(sample);
    }

    output
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Test Helpers
    // =========================================================================

    /// Create a test WAV file with 16-bit mono PCM samples
    fn create_test_wav_16bit_mono(samples: &[i16], sample_rate: u32) -> Vec<u8> {
        let num_samples = samples.len();
        let data_size = (num_samples * 2) as u32;
        let file_size = 36 + data_size;

        let mut wav = Vec::with_capacity(44 + num_samples * 2);

        // RIFF header
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");

        // fmt chunk
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());

        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }

        wav
    }

    /// Create a test WAV file with 16-bit stereo PCM samples
    fn create_test_wav_16bit_stereo(samples: &[i16], sample_rate: u32) -> Vec<u8> {
        let num_samples = samples.len();
        let data_size = (num_samples * 2) as u32;
        let file_size = 36 + data_size;

        let mut wav = Vec::with_capacity(44 + num_samples * 2);

        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes()); // stereo
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 4).to_le_bytes());
        wav.extend_from_slice(&4u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }

        wav
    }

    /// Create a test WAV file with 8-bit unsigned PCM samples
    fn create_test_wav_8bit(samples: &[u8], sample_rate: u32) -> Vec<u8> {
        let num_samples = samples.len();
        let data_size = num_samples as u32;
        let file_size = 36 + data_size;

        let mut wav = Vec::with_capacity(44 + num_samples);

        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&8u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        wav.extend_from_slice(samples);

        wav
    }

    /// Create a test WAV file with 24-bit signed PCM samples
    fn create_test_wav_24bit(samples: &[[u8; 3]], sample_rate: u32) -> Vec<u8> {
        let num_samples = samples.len();
        let data_size = (num_samples * 3) as u32;
        let file_size = 36 + data_size;

        let mut wav = Vec::with_capacity(44 + num_samples * 3);

        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 3).to_le_bytes());
        wav.extend_from_slice(&3u16.to_le_bytes());
        wav.extend_from_slice(&24u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(sample);
        }

        wav
    }

    /// Create a test WAV file with 32-bit float samples
    fn create_test_wav_32bit_float(samples: &[f32], sample_rate: u32) -> Vec<u8> {
        let num_samples = samples.len();
        let data_size = (num_samples * 4) as u32;
        let file_size = 36 + data_size;

        let mut wav = Vec::with_capacity(44 + num_samples * 4);

        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&3u16.to_le_bytes()); // float
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 4).to_le_bytes());
        wav.extend_from_slice(&4u16.to_le_bytes());
        wav.extend_from_slice(&32u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }

        wav
    }

    // =========================================================================
    // WAV Parsing Tests
    // =========================================================================

    #[test]
    fn test_parse_wav_16bit_mono() {
        let samples = vec![0i16, 16384, -16384, 32767, -32768];
        let wav = create_test_wav_16bit_mono(&samples, 16000);
        let result = parse_wav(&wav);
        assert!(result.is_ok());
        let data = result.expect("parse should succeed");
        assert_eq!(data.sample_rate, 16000);
        assert_eq!(data.samples.len(), 5);
        assert!((data.samples[0] - 0.0).abs() < 0.001);
        assert!((data.samples[1] - 0.5).abs() < 0.001);
        assert!((data.samples[2] - (-0.5)).abs() < 0.001);
    }

    #[test]
    fn test_parse_wav_16bit_stereo() {
        let samples = vec![16384i16, -16384, 0, 0, 32767, -32767];
        let wav = create_test_wav_16bit_stereo(&samples, 44100);
        let result = parse_wav(&wav);
        assert!(result.is_ok());
        let data = result.expect("parse should succeed");
        assert_eq!(data.sample_rate, 44100);
        assert_eq!(data.samples.len(), 3);
        assert!((data.samples[0] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_wav_8bit() {
        let samples = vec![128u8, 255, 0, 192, 64];
        let wav = create_test_wav_8bit(&samples, 8000);
        let result = parse_wav(&wav);
        assert!(result.is_ok());
        let data = result.expect("parse should succeed");
        assert_eq!(data.sample_rate, 8000);
        assert_eq!(data.samples.len(), 5);
        assert!((data.samples[0] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_wav_24bit() {
        let samples = [[0, 0, 0], [0xFF, 0xFF, 0x7F], [0, 0, 0x80]];
        let wav = create_test_wav_24bit(&samples, 48000);
        let result = parse_wav(&wav);
        assert!(result.is_ok());
        let data = result.expect("parse should succeed");
        assert_eq!(data.sample_rate, 48000);
        assert_eq!(data.samples.len(), 3);
    }

    #[test]
    fn test_parse_wav_32bit_float() {
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let wav = create_test_wav_32bit_float(&samples, 16000);
        let result = parse_wav(&wav);
        assert!(result.is_ok());
        let data = result.expect("parse should succeed");
        assert_eq!(data.sample_rate, 16000);
        for (i, &expected) in samples.iter().enumerate() {
            assert!((data.samples[i] - expected).abs() < 0.0001);
        }
    }

    #[test]
    fn test_parse_wav_invalid_too_small() {
        let result = parse_wav(b"RIFF");
        assert_eq!(result, Err(WavError::TooSmall));
    }

    #[test]
    fn test_parse_wav_invalid_missing_riff() {
        let mut wav = vec![0u8; 44];
        wav[0..4].copy_from_slice(b"XXXX");
        let result = parse_wav(&wav);
        assert_eq!(result, Err(WavError::MissingRiff));
    }

    #[test]
    fn test_parse_wav_invalid_missing_wave() {
        let mut wav = create_test_wav_16bit_mono(&[0i16; 10], 16000);
        wav[8..12].copy_from_slice(b"XXXX");
        let result = parse_wav(&wav);
        assert_eq!(result, Err(WavError::MissingWave));
    }

    #[test]
    fn test_parse_wav_no_data_chunk() {
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&100u32.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&32000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"JUNK");
        wav.extend_from_slice(&4u32.to_le_bytes());
        wav.extend_from_slice(&[0u8; 4]);
        let result = parse_wav(&wav);
        assert_eq!(result, Err(WavError::NoDataChunk));
    }

    #[test]
    fn test_parse_wav_unsupported_channels() {
        let mut wav = create_test_wav_16bit_mono(&[0i16; 30], 16000);
        wav[22] = 6;
        wav[23] = 0;
        let result = parse_wav(&wav);
        assert_eq!(result, Err(WavError::UnsupportedChannels(6)));
    }

    // =========================================================================
    // Resampling Tests
    // =========================================================================

    #[test]
    fn test_resample_no_change() {
        let samples = vec![0.5, -0.5, 0.25, -0.25, 0.0];
        let resampled = resample(&samples, 16000, 16000);
        assert_eq!(resampled, samples);
    }

    #[test]
    fn test_resample_downsample_48k_to_16k() {
        let samples: Vec<f32> = (0..4800).map(|i| i as f32 / 4800.0).collect();
        let resampled = resample(&samples, 48000, 16000);
        assert_eq!(resampled.len(), 1600);
    }

    #[test]
    fn test_resample_upsample_8k_to_16k() {
        let samples: Vec<f32> = (0..800).map(|i| (i as f32 / 800.0) * 2.0 - 1.0).collect();
        let resampled = resample(&samples, 8000, 16000);
        assert_eq!(resampled.len(), 1600);
    }

    #[test]
    fn test_resample_empty() {
        let samples: Vec<f32> = Vec::new();
        let resampled = resample(&samples, 48000, 16000);
        assert!(resampled.is_empty());
    }

    #[test]
    fn test_wav_error_display() {
        assert_eq!(WavError::TooSmall.to_string(), "WAV file too small");
        assert_eq!(
            WavError::MissingRiff.to_string(),
            "Invalid WAV: missing RIFF header"
        );
        assert_eq!(
            WavError::UnsupportedChannels(6).to_string(),
            "Unsupported channel count: 6"
        );
    }

    // =========================================================================
    // Conversion Tests
    // =========================================================================

    #[test]
    fn test_convert_16bit_pcm() {
        let data = [0u8, 0, 0, 128, 255, 127];
        let samples = convert_16bit_pcm(&data);
        assert_eq!(samples.len(), 3);
        assert!((samples[0] - 0.0).abs() < 0.001);
        assert!((samples[1] - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_convert_8bit_pcm() {
        let data = [128u8, 0, 255];
        let samples = convert_8bit_pcm(&data);
        assert_eq!(samples.len(), 3);
        assert!((samples[0] - 0.0).abs() < 0.01);
        assert!((samples[1] - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_convert_to_mono_stereo() {
        let samples = vec![0.5, -0.5, 1.0, 0.0, -1.0, 1.0];
        let mono = convert_to_mono(samples, 2).expect("convert");
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.0).abs() < 0.001);
    }
}

// =============================================================================
// PROPERTY-BASED TESTS
// =============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// WAV parsing never panics on arbitrary input
        #[test]
        fn fuzz_wav_parsing(data: Vec<u8>) {
            let _ = parse_wav(&data);
        }

        /// Resampling preserves approximate duration
        #[test]
        fn prop_resample_duration(
            samples in prop::collection::vec(any::<f32>(), 100..5000),
            src_rate in 8000u32..96000,
            dst_rate in 8000u32..96000,
        ) {
            let resampled = resample(&samples, src_rate, dst_rate);
            let expected_len = ((samples.len() as f64 * dst_rate as f64) / src_rate as f64).ceil() as i64;
            let actual_len = resampled.len() as i64;
            prop_assert!((actual_len - expected_len).abs() <= 2);
        }

        /// Resampling output stays bounded for bounded input
        #[test]
        fn prop_resample_bounded(
            samples in prop::collection::vec(-1.0f32..1.0, 100..1000),
        ) {
            let resampled = resample(&samples, 48000, 16000);
            for &s in &resampled {
                prop_assert!(s >= -1.5 && s <= 1.5);
            }
        }

        /// 16-bit PCM conversion is bounded
        #[test]
        fn prop_16bit_pcm_bounded(data in prop::collection::vec(any::<u8>(), 0..1000)) {
            let samples = convert_16bit_pcm(&data);
            for &s in &samples {
                prop_assert!(s >= -1.0 && s <= 1.0);
            }
        }

        /// 8-bit PCM conversion is bounded
        #[test]
        fn prop_8bit_pcm_bounded(data in prop::collection::vec(any::<u8>(), 0..1000)) {
            let samples = convert_8bit_pcm(&data);
            for &s in &samples {
                prop_assert!(s >= -1.0 && s <= 1.0);
            }
        }
    }
}

// =============================================================================
// WAVE_FORMAT_EXTENSIBLE TESTS (WAPR-AUDIO-001)
// =============================================================================

#[cfg(test)]
mod extensible_tests {
    use super::*;

    /// Create a WAVE_FORMAT_EXTENSIBLE 24-bit PCM WAV file
    fn create_extensible_24bit_wav(samples: &[[u8; 3]], sample_rate: u32) -> Vec<u8> {
        let num_samples = samples.len();
        let data_size = (num_samples * 3) as u32;
        // Header: RIFF(4) + size(4) + WAVE(4) + fmt(4) + size(4) + fmt_data(40) + data(4) + size(4) = 68 bytes
        let file_size = 60 + data_size;

        let mut wav = Vec::with_capacity(68 + num_samples * 3);

        // RIFF header
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");

        // fmt chunk - WAVE_FORMAT_EXTENSIBLE
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&40u32.to_le_bytes()); // fmt chunk size
        wav.extend_from_slice(&0xFFFEu16.to_le_bytes()); // WAVE_FORMAT_EXTENSIBLE
        wav.extend_from_slice(&1u16.to_le_bytes()); // channels
        wav.extend_from_slice(&sample_rate.to_le_bytes()); // sample rate
        wav.extend_from_slice(&(sample_rate * 3).to_le_bytes()); // byte rate
        wav.extend_from_slice(&3u16.to_le_bytes()); // block align
        wav.extend_from_slice(&24u16.to_le_bytes()); // bits per sample

        // Extension
        wav.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        wav.extend_from_slice(&24u16.to_le_bytes()); // valid bits per sample
        wav.extend_from_slice(&4u32.to_le_bytes()); // channel mask (front center)

        // SubFormat GUID for PCM: 00000001-0000-0010-8000-00aa00389b71
        wav.extend_from_slice(&[
            0x01, 0x00, 0x00, 0x00, // format type (1 = PCM)
            0x00, 0x00, 0x10, 0x00, // -
            0x80, 0x00, 0x00, 0xAA, // - GUID parts
            0x00, 0x38, 0x9B, 0x71, // -
        ]);

        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(sample);
        }

        wav
    }

    /// Create a WAVE_FORMAT_EXTENSIBLE 32-bit float WAV file
    fn create_extensible_32bit_float_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
        let num_samples = samples.len();
        let data_size = (num_samples * 4) as u32;
        let file_size = 60 + data_size;

        let mut wav = Vec::with_capacity(68 + num_samples * 4);

        // RIFF header
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");

        // fmt chunk - WAVE_FORMAT_EXTENSIBLE
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&40u32.to_le_bytes());
        wav.extend_from_slice(&0xFFFEu16.to_le_bytes()); // WAVE_FORMAT_EXTENSIBLE
        wav.extend_from_slice(&1u16.to_le_bytes()); // channels
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 4).to_le_bytes()); // byte rate
        wav.extend_from_slice(&4u16.to_le_bytes()); // block align
        wav.extend_from_slice(&32u16.to_le_bytes()); // bits per sample

        // Extension
        wav.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        wav.extend_from_slice(&32u16.to_le_bytes()); // valid bits per sample
        wav.extend_from_slice(&4u32.to_le_bytes()); // channel mask

        // SubFormat GUID for IEEE Float: 00000003-0000-0010-8000-00aa00389b71
        wav.extend_from_slice(&[
            0x03, 0x00, 0x00, 0x00, // format type (3 = IEEE_FLOAT)
            0x00, 0x00, 0x10, 0x00,
            0x80, 0x00, 0x00, 0xAA,
            0x00, 0x38, 0x9B, 0x71,
        ]);

        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }

        wav
    }

    /// Create a WAVE_FORMAT_EXTENSIBLE 32-bit PCM WAV file
    fn create_extensible_32bit_pcm_wav(samples: &[i32], sample_rate: u32) -> Vec<u8> {
        let num_samples = samples.len();
        let data_size = (num_samples * 4) as u32;
        let file_size = 60 + data_size;

        let mut wav = Vec::with_capacity(68 + num_samples * 4);

        // RIFF header
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");

        // fmt chunk - WAVE_FORMAT_EXTENSIBLE
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&40u32.to_le_bytes());
        wav.extend_from_slice(&0xFFFEu16.to_le_bytes()); // WAVE_FORMAT_EXTENSIBLE
        wav.extend_from_slice(&1u16.to_le_bytes()); // channels
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 4).to_le_bytes()); // byte rate
        wav.extend_from_slice(&4u16.to_le_bytes()); // block align
        wav.extend_from_slice(&32u16.to_le_bytes()); // bits per sample

        // Extension
        wav.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        wav.extend_from_slice(&32u16.to_le_bytes()); // valid bits per sample
        wav.extend_from_slice(&4u32.to_le_bytes()); // channel mask

        // SubFormat GUID for PCM: 00000001-0000-0010-8000-00aa00389b71
        wav.extend_from_slice(&[
            0x01, 0x00, 0x00, 0x00, // format type (1 = PCM)
            0x00, 0x00, 0x10, 0x00,
            0x80, 0x00, 0x00, 0xAA,
            0x00, 0x38, 0x9B, 0x71,
        ]);

        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }

        wav
    }

    // =========================================================================
    // RED PHASE: These tests MUST fail before implementation
    // =========================================================================

    #[test]
    fn test_extensible_24bit_pcm_parses() {
        // 24-bit samples: silence, max positive, max negative
        let samples = [[0, 0, 0], [0xFF, 0xFF, 0x7F], [0x00, 0x00, 0x80]];
        let wav = create_extensible_24bit_wav(&samples, 16000);

        let result = parse_wav(&wav);

        // This SHOULD succeed - currently fails with "Unsupported format: 24 bits, format 65534"
        assert!(
            result.is_ok(),
            "WAVE_FORMAT_EXTENSIBLE 24-bit PCM should parse: {:?}",
            result
        );

        let data = result.expect("should parse");
        assert_eq!(data.sample_rate, 16000);
        assert_eq!(data.bits_per_sample, 24);
        assert_eq!(data.samples.len(), 3);
    }

    #[test]
    fn test_extensible_32bit_float_parses() {
        let samples = [0.0f32, 0.5, -0.5, 1.0, -1.0];
        let wav = create_extensible_32bit_float_wav(&samples, 16000);

        let result = parse_wav(&wav);

        // This SHOULD succeed - currently fails with "Unsupported format: 32 bits, format 65534"
        assert!(
            result.is_ok(),
            "WAVE_FORMAT_EXTENSIBLE 32-bit float should parse: {:?}",
            result
        );

        let data = result.expect("should parse");
        assert_eq!(data.sample_rate, 16000);
        assert_eq!(data.bits_per_sample, 32);
        for (i, &expected) in samples.iter().enumerate() {
            assert!(
                (data.samples[i] - expected).abs() < 0.0001,
                "Sample {} mismatch: {} vs {}",
                i,
                data.samples[i],
                expected
            );
        }
    }

    #[test]
    fn test_extensible_32bit_pcm_parses() {
        let samples = [0i32, 1_073_741_824, -1_073_741_824]; // 0, ~0.5, ~-0.5
        let wav = create_extensible_32bit_pcm_wav(&samples, 16000);

        let result = parse_wav(&wav);

        // This SHOULD succeed - currently fails with "Unsupported format: 32 bits, format 65534"
        assert!(
            result.is_ok(),
            "WAVE_FORMAT_EXTENSIBLE 32-bit PCM should parse: {:?}",
            result
        );

        let data = result.expect("should parse");
        assert_eq!(data.sample_rate, 16000);
        assert_eq!(data.bits_per_sample, 32);
        assert_eq!(data.samples.len(), 3);
    }

    #[test]
    fn test_real_24bit_file_parses() {
        // Test with actual 24-bit test file if it exists
        let path = "demos/test-audio/test-24bit.wav";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping test: {} not found", path);
            return;
        }

        let data = std::fs::read(path).expect("Should read file");
        let result = parse_wav(&data);

        assert!(
            result.is_ok(),
            "Real 24-bit file should parse: {:?}",
            result
        );
    }

    #[test]
    fn test_real_32bit_float_file_parses() {
        // Test with actual 32-bit float test file if it exists
        let path = "demos/test-audio/test-32f.wav";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping test: {} not found", path);
            return;
        }

        let data = std::fs::read(path).expect("Should read file");
        let result = parse_wav(&data);

        assert!(
            result.is_ok(),
            "Real 32-bit float file should parse: {:?}",
            result
        );
    }
}
