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

/// Frames (20 ms) of `audio` whose RMS reaches [`SILENCE_RMS`]; non-finite frames never count.
pub fn signal_frames(audio: &[f32]) -> usize {
    audio
        .chunks(FRAME_SAMPLES)
        .filter(|frame| {
            let sum: f32 = frame.iter().map(|s| s * s).sum();
            let rms = (sum / frame.len() as f32).sqrt();
            rms.is_finite() && rms >= SILENCE_RMS
        })
        .count()
}

/// Fewest signal frames (10 x 20 ms = 200 ms) that make audio worth decoding: a
/// single click or pop is one frame, and the decoder would caption it.
pub const MIN_SIGNAL_FRAMES: usize = 10;

/// True when `audio` carries no signal a decoder could transcribe.
/// Non-finite samples count as silence: they are not audio, and neither is a
/// lone click (fewer than [`MIN_SIGNAL_FRAMES`] frames of signal).
#[must_use]
#[provable_contracts_macros::contract("speech-gate-v1", equation = "is_silent")]
pub fn is_silent(audio: &[f32]) -> bool {
    signal_frames(audio) < MIN_SIGNAL_FRAMES
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
    if peak_frame_rms(audio) < SIGNAL_RMS {
        return Err(WhisperError::Audio(
            "no speech detected: the audio is below -50 dBFS, too quiet to caption".into(),
        ));
    }
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
const MIN_WORDS_PER_SEC: f32 = 0.5;
/// `slow` only applies to cues at least this long: a short cue has no rate.
const SLOW_MIN_SECS: f32 = 5.0;
/// A short last cue this long after the previous one is closing-music noise.
const TAIL_GAP_SECS: f32 = 2.0;
/// A closing filler is popped only when it stands this far after the cue before it.
const FILLER_GAP_SECS: f32 = 1.5;
/// A one-word or sub-300 ms cue is a flicker only this far after the cue before it.
const FLICKER_GAP_SECS: f32 = 2.0;

/// A bracketed sound tag: `[MUSIC]`, `(applause)`, `*music*`. The opening and
/// closing marks must match, so "[1, 2, 3] is a list" is speech, not a tag.
fn is_sound_tag(text: &str) -> bool {
    let mut chars = text.chars();
    let (Some(first), Some(last)) = (chars.next(), chars.next_back()) else {
        return false;
    };
    let inner = &text[first.len_utf8()..text.len() - last.len_utf8()];
    let sound = inner
        .chars()
        .all(|c| c.is_alphabetic() || c == ' ' || c == '_' || c == '-');
    sound && matches!((first, last), ('*', '*') | ('[', ']') | ('(', ')'))
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
        let isolated = prev_end.is_some() && gap >= FLICKER_GAP_SECS;
        let flicker =
            isolated && (secs < 0.3 || (words <= 1.0 && secs <= 1.0 && is_closing_filler(text)));
        let slow = secs >= SLOW_MIN_SECS && words / secs < MIN_WORDS_PER_SEC;
        let stranded =
            i > 0 && i + 1 == n && words <= 4.0 && gap >= TAIL_GAP_SECS && is_closing_filler(text);
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
#[path = "speech_gate_tests.rs"]
mod tests;
