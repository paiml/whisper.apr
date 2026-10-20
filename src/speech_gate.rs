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
        .filter(|rms| rms.is_finite())
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
use crate::{Segment, TranscribeOptions, TranscriptionResult};

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

/// Refuse silent audio, transcribe (chunked above 30 s), then snap cues to the signal.
///
/// # Errors
/// `WhisperError::Audio` when the audio is silent, or the decoder fails.
pub fn transcribe_gated(
    model: &crate::WhisperApr,
    audio: &[f32],
    options: TranscribeOptions,
) -> WhisperResult<TranscriptionResult> {
    refuse_silent(audio)?;
    let result = if audio.len() > crate::WhisperApr::CHUNK_SAMPLES {
        model.transcribe_chunked(audio, options)
    } else {
        model.transcribe_single_chunk(audio, options)
    }?;
    Ok(snap_to_signal(result, audio))
}

/// A frame at or above this RMS (-50 dBFS) carries signal for cue snapping.
pub const SIGNAL_RMS: f32 = 3.16e-3;

/// A silence this long (3 s of frames) inside a cue ends one run of speech.
const RUN_GAP_FRAMES: usize = 150;

/// The longest run of signal frames, runs split by a gap of `RUN_GAP_FRAMES`:
/// a cue timed across a 20 s silence belongs to the speech, not the music before it.
fn dominant_run(loud: impl Iterator<Item = usize>) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    let mut cur: Option<(usize, usize)> = None;
    for i in loud {
        cur = match cur {
            Some((s, e)) if i - e < RUN_GAP_FRAMES => Some((s, i)),
            _ => Some((i, i)),
        };
        let c = cur?;
        if best.map_or(true, |(s, e)| c.1 - c.0 > e - s) {
            best = Some(c);
        }
    }
    best
}

/// Snap one cue to the signal inside it: start moves to the first 20 ms frame
/// with signal, end to the end of the last. `None` when the span is silent.
#[must_use]
#[provable_contracts_macros::contract("speech-gate-v1", equation = "snap_span")]
pub fn snap_span(audio: &[f32], start: f32, end: f32) -> Option<(f32, f32)> {
    let sr = SAMPLE_RATE as f32;
    let lo = ((start.max(0.0) * sr) as usize).min(audio.len());
    let hi = ((end.max(0.0) * sr) as usize).min(audio.len());
    let frames: Vec<_> = audio[lo..hi.max(lo)].chunks(FRAME_SAMPLES).collect();
    let loud = |f: &&[f32]| peak_frame_rms(f) >= SIGNAL_RMS;
    let (first, last) = dominant_run(
        frames
            .iter()
            .enumerate()
            .filter(|(_, f)| loud(f))
            .map(|(i, _)| i),
    )?;
    let s = (lo + first * FRAME_SAMPLES) as f32 / sr;
    let e = (lo + ((last + 1) * FRAME_SAMPLES).min(hi - lo)) as f32 / sr;
    Some((s, e))
}

/// Slower than this (words per second) is not speech, once the cue is long enough to judge.
const MIN_WORDS_PER_SEC: f32 = 0.8;
/// `slow` only applies to cues at least this long: a short cue has no rate.
const SLOW_MIN_SECS: f32 = 2.0;
/// A short last cue this long after the previous one is closing-music noise.
const TAIL_GAP_SECS: f32 = 2.0;
/// A closing filler is popped only when it stands this far after the cue before it.
const FILLER_GAP_SECS: f32 = 0.5;

/// A bracketed sound tag: `[MUSIC]`, `(applause)`, `*music*`. The opening and
/// closing marks must match, so "[1, 2, 3] is a list" is speech, not a tag.
fn is_sound_tag(text: &str) -> bool {
    let mut chars = text.chars();
    let (Some(first), Some(last)) = (chars.next(), chars.next_back()) else {
        return false;
    };
    matches!((first, last), ('*', '*') | ('[', ']') | ('(', ')'))
}

/// The decoder's stock closing lines over music: "Thank you.", "Oh, oh.", "you".
fn is_closing_filler(text: &str) -> bool {
    let t: String = text
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphabetic() || c.is_whitespace())
        .collect();
    matches!(
        t.split_whitespace().collect::<Vec<_>>().join(" ").as_str(),
        "thank you"
            | "thanks"
            | "you"
            | "oh"
            | "oh oh"
            | "bye"
            | "god bless you"
            | "thanks for watching"
    )
}

/// Drop the decoder's fillers over music and silence: bracketed sound tags,
/// one-word flickers, cues far too slow to be speech, and a short last cue
/// stranded after a gap ("Thank you." over the closing music).
fn drop_noise_cues(cues: Vec<Segment>) -> Vec<Segment> {
    let n = cues.len();
    let mut prev_end: Option<f32> = None;
    let mut kept = Vec::with_capacity(n);
    for (i, cue) in cues.into_iter().enumerate() {
        let text = cue.text.trim();
        let words = text.split_whitespace().count() as f32;
        let secs = (cue.end - cue.start).max(1.0e-3);
        let gap = prev_end.map_or(0.0, |p| cue.start - p);
        let tag = words <= 4.0 && is_sound_tag(text);
        let flicker =
            secs < 0.3 || (words <= 1.0 && secs <= 1.0 && prev_end.is_some() && gap >= 1.0);
        let slow = secs >= SLOW_MIN_SECS && words / secs < MIN_WORDS_PER_SEC;
        let stranded = i > 0 && i + 1 == n && words <= 4.0 && gap >= TAIL_GAP_SECS;
        prev_end = Some(cue.end);
        if !(tag || flicker || slow || stranded) {
            kept.push(cue);
        }
    }
    // Pop closing fillers only when they stand apart from the speech before them:
    // a real "Thank you." that follows its sentence directly is a sign-off.
    while let [.., prev, last] = kept.as_slice() {
        if is_closing_filler(&last.text) && last.start - prev.end >= FILLER_GAP_SECS {
            kept.pop();
        } else {
            break;
        }
    }
    kept
}

/// Snap every cue of `result` to the signal under it and drop cues over pure
/// silence (decoder timestamps drift into silence: a cue read 0.5->26.3 s over
/// audio that is silent until 20.9 s).
#[must_use]
pub fn snap_to_signal(mut result: TranscriptionResult, audio: &[f32]) -> TranscriptionResult {
    let snapped: Vec<Segment> = result
        .segments
        .into_iter()
        .filter_map(|mut seg| {
            let (s, e) = snap_span(audio, seg.start, seg.end)?;
            seg.start = s;
            seg.end = e;
            Some(seg)
        })
        .collect();
    let snapped = drop_noise_cues(snapped);
    result.text = snapped
        .iter()
        .map(|s| s.text.trim())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    result.segments = snapped;
    result
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

    /// FALSIFY-SPEECHGATE-006 - a cue is trimmed to the signal under it, and a cue over silence is None.
    /// FALSIFY-SPEECHGATE-007 - decoder fillers over closing music are dropped; real speech stays.
    #[test]
    fn noise_cues_over_music_are_dropped() {
        let cue = |s: f32, e: f32, t: &str| Segment {
            start: s,
            end: e,
            text: t.into(),
            tokens: vec![],
        };
        let kept = drop_noise_cues(vec![
            cue(
                0.0,
                6.0,
                "these are five key points to discover when you are thinking",
            ),
            cue(7.5, 7.9, "you"),
            cue(8.0, 9.0, "*music*"),
            cue(9.0, 15.0, "Thank you."),
            cue(
                16.0,
                20.0,
                "and they can be deployed either standalone or distributed",
            ),
            cue(23.0, 25.0, "God bless you."),
            cue(26.0, 27.0, "Oh, oh."),
            cue(27.5, 28.5, "Thank you."),
        ]);
        let texts: Vec<_> = kept.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            texts,
            [
                "these are five key points to discover when you are thinking",
                "and they can be deployed either standalone or distributed"
            ]
        );
    }

    #[test]
    fn cue_edges_snap_to_signal_and_silent_cues_vanish() {
        let sr = SAMPLE_RATE as usize;
        let mut a = vec![0.0; 30 * sr];
        a[20 * sr..27 * sr].copy_from_slice(&tone(7.0, 0.05));
        let (s, e) = snap_span(&a, 0.5, 26.34).expect("26.34 > 20");
        assert!((s - 20.0).abs() < 0.03, "start {s}");
        assert!((e - 26.34).abs() < 0.03, "end {e}");
        assert_eq!(snap_span(&a, 1.0, 19.0), None);
        // music at 0.5-3.5 s, then 17 s of silence, then speech: the speech wins
        let mut b = a.clone();
        b[sr / 2..7 * sr / 2].copy_from_slice(&tone(3.0, 0.05));
        let (s2, _) = snap_span(&b, 0.5, 26.34).expect("speech");
        assert!((s2 - 20.0).abs() < 0.03, "start {s2}");
        assert_eq!(snap_span(&a, 40.0, 50.0), None);
    }

    #[test]
    fn hiss_below_the_floor_and_nonfinite_samples_are_silent() {
        assert!(is_silent(&tone(5.0, 5.0e-4)));
        assert!(is_silent(&vec![f32::NAN; 16_000]));
        assert!(is_silent(&vec![f32::INFINITY; 16_000]));
        let mut speech = tone(1.0, 0.05);
        speech[10] = f32::INFINITY;
        assert!(
            !is_silent(&speech),
            "one inf sample must not refuse real audio"
        );
    }

    fn cue(s: f32, e: f32, t: &str) -> Segment {
        Segment {
            start: s,
            end: e,
            text: t.into(),
            tokens: vec![],
        }
    }

    /// FALSIFY-SPEECHGATE-008 - genuine short speech survives the noise filter.
    #[test]
    fn short_genuine_cues_survive() {
        assert_eq!(
            drop_noise_cues(vec![cue(0.0, 1.5, "Hello there.")]).len(),
            1
        );
        assert_eq!(drop_noise_cues(vec![cue(2.0, 2.8, "Thank you.")]).len(), 1);
        let kept = drop_noise_cues(vec![
            cue(0.0, 4.0, "Is this a list or a tuple in Rust?"),
            cue(4.2, 4.9, "Yes."),
            cue(5.0, 8.0, "[1, 2, 3] is a list of three integers."),
            cue(8.1, 9.0, "Thank you."),
        ]);
        assert_eq!(kept.len(), 4, "{kept:?}");
        let tag = drop_noise_cues(vec![
            cue(0.0, 4.0, "hello world and welcome to the course"),
            cue(4.0, 6.0, "[MUSIC]"),
        ]);
        assert_eq!(tag.len(), 1);
    }
}
