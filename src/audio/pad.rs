//! Whisper's `pad_or_trim` (#93).
//!
//! Whisper was trained on 30 s windows. A shorter clip must be zero-padded in
//! the WAVEFORM before the mel spectrogram, so the encoder sees trailing
//! silence and the log-mel clamp is taken over the whole window. Without it a
//! 1.5 s clip gave 150 mel frames instead of 3000 and the decoder kept
//! generating after the speech ended (54 words where whisper.cpp says 4).

use std::borrow::Cow;

/// Zero-pad `audio` to exactly `n` samples, or trim it to the first `n`.
///
/// Borrows when `audio` is already exactly `n` samples or longer.
#[must_use]
pub fn pad_or_trim(audio: &[f32], n: usize) -> Cow<'_, [f32]> {
    if audio.len() >= n {
        Cow::Borrowed(&audio[..n])
    } else {
        let mut v = Vec::with_capacity(n);
        v.extend_from_slice(audio);
        v.resize(n, 0.0);
        Cow::Owned(v)
    }
}

#[cfg(test)]
mod tests {
    use super::pad_or_trim;

    #[test]
    fn short_audio_is_zero_padded() {
        let out = pad_or_trim(&[0.5; 3], 5);
        assert_eq!(&*out, &[0.5, 0.5, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn long_audio_is_trimmed_and_borrowed() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let out = pad_or_trim(&a, 2);
        assert_eq!(&*out, &[1.0, 2.0]);
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn exact_length_is_unchanged() {
        let a = [1.0, 2.0];
        assert_eq!(&*pad_or_trim(&a, 2), &a);
    }
}
