//! Silence gate: audio with no signal is refused, never captioned.
//!
//! Measured on a GX10 (large-v3-turbo, d46f381 + #109..#111): a 60 s WAV of
//! digital zeros decoded to a single cue `Thank you.` spanning 0 -> 29.98 s,
//! exit 0, after 130 s of CPU. The decoder always emits *something*; a window
//! with nothing in it must never reach the decoder.
//!
//! Contract `speech-gate-v1`, FALSIFY-SPEECHGATE-001..004.

use crate::audio::SAMPLE_RATE;

/// Frame length the energy is measured over: 20 ms.
pub const FRAME_SAMPLES: usize = SAMPLE_RATE as usize / 50;

/// A window whose loudest 20 ms frame has an RMS below this is silent.
/// 1e-3 is -60 dBFS: below the noise floor of any recording that carries a voice.
pub const SILENCE_RMS: f32 = 1.0e-3;

/// Loudest 20 ms frame RMS of `audio` (0.0 for empty input).
#[must_use]
#[provable_contracts_macros::contract("speech-gate-v1", equation = "peak_frame_rms")]
pub fn peak_frame_rms(audio: &[f32]) -> f32 {
    audio
        .chunks(FRAME_SAMPLES)
        .map(|frame| {
            let sum: f32 = frame.iter().map(|s| s * s).sum();
            (sum / frame.len() as f32).sqrt()
        })
        .fold(0.0_f32, f32::max)
}

/// True when `audio` carries no signal a decoder could transcribe.
/// Non-finite samples count as silence: they are not audio.
#[must_use]
#[provable_contracts_macros::contract("speech-gate-v1", equation = "is_silent")]
pub fn is_silent(audio: &[f32]) -> bool {
    let peak = peak_frame_rms(audio);
    !(peak.is_finite() && peak >= SILENCE_RMS)
}

use crate::error::{WhisperError, WhisperResult};
use crate::{TranscribeOptions, TranscriptionResult};

/// Refuse silent input: the decoder always emits something (60 s of zeros ->
/// "Thank you."), so silence must never reach it.
///
/// # Errors
/// `WhisperError::Audio` when `audio` is silent.
#[must_use = "a refusal that is not checked refuses nothing"]
pub fn refuse_silent(audio: &[f32]) -> WhisperResult<()> {
    if is_silent(audio) {
        return Err(WhisperError::Audio(
            "no speech detected: the audio is silent (peak 20 ms RMS below -60 dBFS)".into(),
        ));
    }
    Ok(())
}

/// The result of a window that is not decoded because it is silent: no
/// segments, closed (nothing was cut), `None` when the window has signal.
#[must_use]
pub fn silent_window(
    audio: &[f32],
    options: &TranscribeOptions,
) -> Option<(TranscriptionResult, bool)> {
    is_silent(audio).then(|| {
        (
            TranscriptionResult {
                text: String::new(),
                language: options.language.clone().unwrap_or_else(|| "en".to_string()),
                segments: Vec::new(),
                profiling: None,
            },
            true,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(secs: f32, amp: f32) -> Vec<f32> {
        (0..(secs * SAMPLE_RATE as f32) as usize)
            .map(|i| amp * (i as f32 * 0.05).sin())
            .collect()
    }

    /// FALSIFY-SPEECHGATE-001
    #[test]
    fn empty_and_zero_audio_are_silent() {
        assert!(is_silent(&[]));
        assert!(is_silent(&vec![0.0; 60 * SAMPLE_RATE as usize]));
    }

    /// FALSIFY-SPEECHGATE-002 — the gate must also be able to say "not silent".
    #[test]
    fn audible_audio_is_not_silent() {
        assert!(!is_silent(&tone(2.0, 0.05)));
    }

    /// FALSIFY-SPEECHGATE-003 — one loud second in a minute of zeros is kept.
    #[test]
    fn a_single_audible_burst_in_silence_is_not_silent() {
        let mut a = vec![0.0; 60 * SAMPLE_RATE as usize];
        a[30 * SAMPLE_RATE as usize..31 * SAMPLE_RATE as usize].copy_from_slice(&tone(1.0, 0.05));
        assert!(!is_silent(&a));
    }

    /// FALSIFY-SPEECHGATE-004 — sub-threshold hiss and NaN are not audio.
    /// FALSIFY-SPEECHGATE-005 - silent input is refused, long or short.
    #[test]
    fn transcribe_refuses_silent_audio() {
        let whisper = crate::WhisperApr::tiny();
        for secs in [1usize, 45, 400] {
            let silence = vec![0.0_f32; secs * 16_000];
            let r = whisper.transcribe(&silence, TranscribeOptions::default());
            assert!(
                matches!(r, Err(WhisperError::Audio(_))),
                "{secs}s of zeros must be refused, got {r:?}"
            );
        }
    }

    #[test]
    fn hiss_below_the_floor_and_nonfinite_samples_are_silent() {
        assert!(is_silent(&tone(5.0, 5.0e-4)));
        assert!(is_silent(&vec![f32::NAN; 16_000]));
    }
}
