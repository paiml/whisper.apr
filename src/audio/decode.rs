//! Multi-format audio decoding for whisper-apr.
//!
//! Provides a unified audio loading API that supports:
//! - WAV natively (always available)
//! - MP3, FLAC, OGG, M4A, AAC, MP4, MOV, WebM, MKV, AVI, Opus via symphonia
//!   (requires `symphonia` feature)
//!
//! All output is mono f32 samples at 16kHz, ready for Whisper inference.

use crate::audio::wav::{parse_wav_file, resample};
use std::path::Path;

/// Supported audio/media file extensions.
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "flac", "ogg", "m4a", "aac", "mp4", "mov", "webm", "mkv", "avi", "opus",
];

/// Check if a file extension is supported for audio decoding.
#[must_use]
pub fn is_supported_extension(ext: &str) -> bool {
    SUPPORTED_EXTENSIONS.contains(&ext.to_lowercase().as_str())
}

/// Load audio from a file path, returning mono f32 samples at 16kHz.
///
/// Supports WAV natively. With the `symphonia` feature enabled, also
/// supports MP3, FLAC, OGG, M4A, AAC, MP4, WebM, MKV, AVI, and Opus.
///
/// # Errors
///
/// Returns an error if the file cannot be read, the format is unsupported,
/// or audio decoding fails.
pub fn load_audio_file(path: &Path) -> Result<Vec<f32>, AudioDecodeError> {
    let data = std::fs::read(path).map_err(AudioDecodeError::Io)?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    load_audio_samples(&data, &ext)
}

/// Load audio samples from raw bytes with a known format extension.
///
/// Returns mono f32 samples at 16kHz.
pub fn load_audio_samples(data: &[u8], ext: &str) -> Result<Vec<f32>, AudioDecodeError> {
    let samples = match ext {
        "wav" => {
            let wav = parse_wav_file(data)
                .map_err(|e| AudioDecodeError::Format(format!("WAV parse failed: {e}")))?;
            if wav.sample_rate == 16000 {
                wav.samples
            } else {
                resample(&wav.samples, wav.sample_rate, 16000)
            }
        }
        #[cfg(feature = "symphonia")]
        "mp3" | "flac" | "ogg" | "m4a" | "aac" | "mp4" | "mov" | "webm" | "mkv" | "avi"
        | "opus" => decode_with_symphonia(data, ext)?,
        #[cfg(not(feature = "symphonia"))]
        "mp3" | "flac" | "ogg" | "m4a" | "aac" | "mp4" | "mov" | "webm" | "mkv" | "avi"
        | "opus" => {
            return Err(AudioDecodeError::FeatureRequired(format!(
                "{ext} format requires the 'symphonia' feature. \
                     Build with: cargo build --features symphonia"
            )));
        }
        _ => return Err(AudioDecodeError::UnsupportedFormat(ext.to_string())),
    };

    // Validate decoded audio for NaN and Inf (aprender A14-A15 defects)
    // Note: clipping check omitted here — lossy codecs may produce values slightly
    // outside ±1.0; the mel filterbank normalizes anyway.
    if aprender::audio::has_nan(&samples) {
        return Err(AudioDecodeError::Validation(
            "Decoded audio contains NaN values".to_string(),
        ));
    }
    if aprender::audio::has_inf(&samples) {
        return Err(AudioDecodeError::Validation(
            "Decoded audio contains Infinity values".to_string(),
        ));
    }

    Ok(samples)
}

/// Audio decoding error type.
#[derive(Debug)]
pub enum AudioDecodeError {
    /// I/O error reading the file.
    Io(std::io::Error),
    /// Audio format parsing/decoding error.
    Format(String),
    /// Feature not enabled for this format.
    FeatureRequired(String),
    /// Unsupported file extension.
    UnsupportedFormat(String),
    /// Audio validation failed (NaN, Inf, or clipping detected).
    Validation(String),
}

impl std::fmt::Display for AudioDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Format(msg) => write!(f, "Audio format error: {msg}"),
            Self::FeatureRequired(msg) => write!(f, "{msg}"),
            Self::UnsupportedFormat(ext) => {
                write!(f, "Unsupported audio format: .{ext}")
            }
            Self::Validation(msg) => write!(f, "Audio validation failed: {msg}"),
        }
    }
}

impl std::error::Error for AudioDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// Decode audio using symphonia (multi-format decoder).
///
/// Returns mono f32 samples at 16kHz.
#[cfg(feature = "symphonia")]
fn decode_with_symphonia(data: &[u8], ext: &str) -> Result<Vec<f32>, AudioDecodeError> {
    use std::io::Cursor;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let cursor = Cursor::new(data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let mut hint = Hint::new();
    hint.with_extension(ext);

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| AudioDecodeError::Format(format!("Failed to probe {ext}: {e}")))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| AudioDecodeError::Format("No audio track found".into()))?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| AudioDecodeError::Format("Unknown sample rate".into()))?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioDecodeError::Format(format!("Decoder creation failed: {e}")))?;

    let samples = decode_all_packets(&mut format, &mut *decoder, track_id)?;

    if sample_rate == 16000 {
        Ok(samples)
    } else {
        Ok(resample(&samples, sample_rate, 16000))
    }
}

/// Read the next packet for a specific track, returning None at EOF.
#[cfg(feature = "symphonia")]
fn next_packet_for_track(
    format: &mut Box<dyn symphonia::core::formats::FormatReader>,
    track_id: u32,
) -> Result<Option<symphonia::core::formats::Packet>, AudioDecodeError> {
    loop {
        match format.next_packet() {
            Ok(p) if p.track_id() == track_id => return Ok(Some(p)),
            Ok(_) => continue,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                return Ok(None);
            }
            Err(e) => {
                return Err(AudioDecodeError::Format(format!(
                    "Failed to read packet: {e}"
                )));
            }
        }
    }
}

/// Read and decode the next audio packet, returning samples and channel count.
#[cfg(feature = "symphonia")]
fn read_next_audio_packet(
    format: &mut Box<dyn symphonia::core::formats::FormatReader>,
    decoder: &mut dyn symphonia::core::codecs::Decoder,
    track_id: u32,
) -> Result<Option<(Vec<f32>, usize)>, AudioDecodeError> {
    use symphonia::core::audio::SampleBuffer;

    loop {
        let packet = match next_packet_for_track(format, track_id)? {
            Some(p) => p,
            None => return Ok(None),
        };

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                buf.copy_interleaved_ref(decoded);
                return Ok(Some((buf.samples().to_vec(), spec.channels.count())));
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => {
                return Err(AudioDecodeError::Format(format!("Decode error: {e}")));
            }
        }
    }
}

/// Decode all audio packets, mixing to mono.
#[cfg(feature = "symphonia")]
fn decode_all_packets(
    format: &mut Box<dyn symphonia::core::formats::FormatReader>,
    decoder: &mut dyn symphonia::core::codecs::Decoder,
    track_id: u32,
) -> Result<Vec<f32>, AudioDecodeError> {
    let mut samples: Vec<f32> = Vec::new();
    loop {
        match read_next_audio_packet(format, decoder, track_id)? {
            Some((interleaved, channels)) => {
                mix_to_mono(&interleaved, channels, &mut samples);
            }
            None => break,
        }
    }
    Ok(samples)
}

/// Mix interleaved multi-channel audio to mono.
///
/// Delegates to `aprender::audio::stereo_to_mono` for 2-channel input.
#[cfg(feature = "symphonia")]
pub fn mix_to_mono(interleaved: &[f32], channels: usize, output: &mut Vec<f32>) {
    match channels {
        1 => output.extend_from_slice(interleaved),
        2 => output.extend(aprender::audio::stereo_to_mono(interleaved)),
        n => {
            for chunk in interleaved.chunks(n) {
                let sum: f32 = chunk.iter().sum();
                output.push(sum / n as f32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_extensions() {
        assert!(is_supported_extension("wav"));
        assert!(is_supported_extension("mp4"));
        assert!(is_supported_extension("MP3"));
        assert!(is_supported_extension("mov"));
        assert!(is_supported_extension("MOV"));
        assert!(!is_supported_extension("txt"));
        assert!(!is_supported_extension("pdf"));
    }

    #[test]
    fn test_mov_dispatches_to_symphonia() {
        // .mov should not return UnsupportedFormat — it should attempt decoding
        let result = load_audio_samples(b"not-real-mov-data", "mov");
        assert!(result.is_err());
        let err_msg = result.expect_err("expected error for fake mov data").to_string();
        // Should get a format/probe error, NOT an "Unsupported audio format" error
        assert!(
            !err_msg.contains("Unsupported audio format"),
            "mov should be recognized, got: {err_msg}"
        );
    }

    #[test]
    fn test_unsupported_format_error() {
        let result = load_audio_samples(b"fake", "xyz");
        assert!(result.is_err());
        let err = result.expect_err("expected error for unsupported format");
        assert!(err.to_string().contains("Unsupported"));
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = load_audio_file(Path::new("/tmp/nonexistent_audio.wav"));
        assert!(result.is_err());
    }

    #[test]
    fn test_error_display() {
        let io_err = AudioDecodeError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "not found",
        ));
        assert!(io_err.to_string().contains("I/O"));

        let fmt_err = AudioDecodeError::Format("bad header".into());
        assert!(fmt_err.to_string().contains("bad header"));

        let feat_err = AudioDecodeError::FeatureRequired("need symphonia".into());
        assert!(feat_err.to_string().contains("symphonia"));

        let unsup_err = AudioDecodeError::UnsupportedFormat("xyz".into());
        assert!(unsup_err.to_string().contains("xyz"));
    }

    #[test]
    fn test_error_source() {
        let io_err = AudioDecodeError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "not found",
        ));
        assert!(std::error::Error::source(&io_err).is_some());

        let fmt_err = AudioDecodeError::Format("test".into());
        assert!(std::error::Error::source(&fmt_err).is_none());
    }
}
