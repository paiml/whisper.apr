//! Tests for WAV file parsing module

use super::*;
use crate::error::WhisperError;

// =========================================================================
// Test Helpers
// =========================================================================

/// Create a test WAV file with 16-bit mono PCM samples
pub(crate) fn create_test_wav_16bit_mono(samples: &[i16], sample_rate: u32) -> Vec<u8> {
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

// =========================================================================
// Property-Based Tests
// =========================================================================

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

// =========================================================================
// WAVE_FORMAT_EXTENSIBLE Tests (WAPR-AUDIO-001)
// =========================================================================

mod extensible_tests {
    use super::*;

    /// Create a WAVE_FORMAT_EXTENSIBLE 24-bit PCM WAV file
    fn create_extensible_24bit_wav(samples: &[[u8; 3]], sample_rate: u32) -> Vec<u8> {
        let num_samples = samples.len();
        let data_size = (num_samples * 3) as u32;
        let file_size = 60 + data_size;

        let mut wav = Vec::with_capacity(68 + num_samples * 3);

        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");

        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&40u32.to_le_bytes());
        wav.extend_from_slice(&0xFFFEu16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 3).to_le_bytes());
        wav.extend_from_slice(&3u16.to_le_bytes());
        wav.extend_from_slice(&24u16.to_le_bytes());

        wav.extend_from_slice(&22u16.to_le_bytes());
        wav.extend_from_slice(&24u16.to_le_bytes());
        wav.extend_from_slice(&4u32.to_le_bytes());

        wav.extend_from_slice(&[
            0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x10, 0x00,
            0x80, 0x00, 0x00, 0xAA,
            0x00, 0x38, 0x9B, 0x71,
        ]);

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

        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");

        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&40u32.to_le_bytes());
        wav.extend_from_slice(&0xFFFEu16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 4).to_le_bytes());
        wav.extend_from_slice(&4u16.to_le_bytes());
        wav.extend_from_slice(&32u16.to_le_bytes());

        wav.extend_from_slice(&22u16.to_le_bytes());
        wav.extend_from_slice(&32u16.to_le_bytes());
        wav.extend_from_slice(&4u32.to_le_bytes());

        wav.extend_from_slice(&[
            0x03, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
        ]);

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

        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");

        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&40u32.to_le_bytes());
        wav.extend_from_slice(&0xFFFEu16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 4).to_le_bytes());
        wav.extend_from_slice(&4u16.to_le_bytes());
        wav.extend_from_slice(&32u16.to_le_bytes());

        wav.extend_from_slice(&22u16.to_le_bytes());
        wav.extend_from_slice(&32u16.to_le_bytes());
        wav.extend_from_slice(&4u32.to_le_bytes());

        wav.extend_from_slice(&[
            0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
        ]);

        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }

        wav
    }

    #[test]
    fn test_extensible_24bit_pcm_parses() {
        let samples = [[0, 0, 0], [0xFF, 0xFF, 0x7F], [0x00, 0x00, 0x80]];
        let wav = create_extensible_24bit_wav(&samples, 16000);

        let result = parse_wav(&wav);

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
        let samples = [0i32, 1_073_741_824, -1_073_741_824];
        let wav = create_extensible_32bit_pcm_wav(&samples, 16000);

        let result = parse_wav(&wav);

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

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_parse_wav_file_wrapper() {
    let samples = vec![0i16, 1000, -1000];
    let wav = create_test_wav_16bit_mono(&samples, 16000);
    let result = parse_wav_file(&wav);
    assert!(result.is_ok());
}

#[test]
fn test_parse_wav_file_error() {
    let result = parse_wav_file(b"invalid");
    assert!(result.is_err());
    match result {
        Err(WhisperError::Audio(msg)) => {
            assert!(msg.contains("small") || msg.contains("WAV"));
        }
        _ => panic!("Expected WhisperError::Audio"),
    }
}

#[test]
fn test_wav_error_display_fmt_truncated() {
    let err = WavError::FmtTruncated;
    assert_eq!(err.to_string(), "Invalid WAV: fmt chunk truncated");
}

#[test]
fn test_wav_error_display_unsupported_format() {
    let err = WavError::UnsupportedFormat {
        format: 99,
        bits: 64,
    };
    assert_eq!(err.to_string(), "Unsupported format: 64 bits, format 99");
}

#[test]
fn test_wav_error_display_no_data_chunk() {
    let err = WavError::NoDataChunk;
    assert_eq!(err.to_string(), "Invalid WAV: no data chunk found");
}

#[test]
fn test_resample_edge_last_sample() {
    let samples = vec![0.5, 0.6, 0.7];
    let resampled = resample(&samples, 48000, 16000);
    assert!(!resampled.is_empty());
}

#[test]
fn test_resample_out_of_bounds() {
    let samples = vec![0.5];
    let resampled = resample(&samples, 8000, 48000);
    assert!(!resampled.is_empty());
    for &s in &resampled {
        assert!(s >= -1.0 && s <= 1.0);
    }
}

#[test]
fn test_convert_24bit_pcm_negative() {
    let data = [0x00, 0x00, 0x80];
    let samples = convert_24bit_pcm(&data);
    assert_eq!(samples.len(), 1);
    assert!(samples[0] < 0.0);
}

#[test]
fn test_convert_32bit_pcm() {
    let data = [0u8, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0x7F];
    let samples = convert_32bit_pcm(&data);
    assert_eq!(samples.len(), 2);
    assert!((samples[0] - 0.0).abs() < 0.001);
    assert!(samples[1] > 0.99);
}

#[test]
fn test_convert_to_mono_unsupported() {
    let samples = vec![0.5; 6];
    let result = convert_to_mono(samples, 3);
    assert_eq!(result, Err(WavError::UnsupportedChannels(3)));
}

#[test]
fn test_parse_fmt_chunk_truncated() {
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&100u32.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&1000u32.to_le_bytes());
    wav.extend_from_slice(&[0u8; 28]);

    let result = parse_wav(&wav);
    assert_eq!(result, Err(WavError::FmtTruncated));
}

#[test]
fn test_parse_wav_unknown_chunk_padding() {
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&200u32.to_le_bytes());
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
    wav.extend_from_slice(&3u32.to_le_bytes());
    wav.extend_from_slice(&[0u8; 4]);

    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&4u32.to_le_bytes());
    wav.extend_from_slice(&[0u8; 4]);

    let result = parse_wav(&wav);
    assert!(result.is_ok());
}

#[test]
fn test_parse_wav_data_before_fmt() {
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&100u32.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&4u32.to_le_bytes());
    wav.extend_from_slice(&[0u8; 24]);

    let result = parse_wav(&wav);
    assert_eq!(result, Err(WavError::NoDataChunk));
}

#[test]
fn test_extensible_format_small_chunk() {
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&100u32.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&0xFFFEu16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&16000u32.to_le_bytes());
    wav.extend_from_slice(&32000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());

    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&4u32.to_le_bytes());
    wav.extend_from_slice(&[0u8; 4]);

    let result = parse_wav(&wav);
    assert!(matches!(result, Err(WavError::UnsupportedFormat { .. })));
}

#[test]
fn test_unsupported_format_bits() {
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
    wav.extend_from_slice(&12u16.to_le_bytes());

    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&4u32.to_le_bytes());
    wav.extend_from_slice(&[0u8; 4]);

    let result = parse_wav(&wav);
    assert!(matches!(
        result,
        Err(WavError::UnsupportedFormat { format: 1, bits: 12 })
    ));
}
