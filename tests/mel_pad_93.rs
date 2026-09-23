//! #93: `compute_mel` pads short audio to Whisper's 30 s window.
//!
//! Before the fix a 1.5 s clip gave 150 mel frames (12,000 values) instead of
//! 3000 (240,000), the decoder saw no trailing silence, and it kept generating.

use whisper_apr::WhisperApr;

const N_MELS: usize = 80;
const FRAMES: usize = 3000;

#[test]
fn compute_mel_is_always_the_30s_window() {
    let w = WhisperApr::tiny();
    for (samples, what) in [(24_000, "1.5 s"), (480_000, "30 s"), (640_000, "40 s")] {
        let mel = w.compute_mel(&vec![0.1_f32; samples]).expect("mel");
        assert_eq!(
            mel.len(),
            FRAMES * N_MELS,
            "{what} must give {FRAMES} frames"
        );
    }
}

#[test]
fn padded_tail_is_the_silence_floor() {
    let w = WhisperApr::tiny();
    let audio: Vec<f32> = (0..24_000).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
    let mel = w.compute_mel(&audio).expect("mel");
    let last = &mel[(FRAMES - 1) * N_MELS..];
    assert_eq!(
        last,
        &mel[2000 * N_MELS..2001 * N_MELS],
        "every padded frame is identical"
    );
    let floor = last[0];
    assert!(
        last.iter().all(|&v| (v - floor).abs() < 1e-6),
        "silence is flat across bins"
    );
    let speech_max = mel[..150 * N_MELS].iter().copied().fold(f32::MIN, f32::max);
    assert!(
        speech_max > floor + 0.5,
        "speech sits well above the silence floor"
    );
}
