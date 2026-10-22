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
        cue(8.5, 8.9, "you"),
        cue(9.0, 10.0, "*music*"),
        cue(10.0, 16.0, "Thank you."),
        cue(
            16.0,
            20.0,
            "and they can be deployed either standalone or distributed",
        ),
        cue(23.0, 25.0, "God bless you."),
        cue(27.0, 28.0, "Oh, oh."),
        cue(29.5, 30.5, "Thank you."),
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
    // between -60 and -50 dBFS: passes the silence gate, still refused before decode
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

/// FALSIFY-SPEECHGATE-009 - one click is not speech: it is refused, a 200 ms burst is not.
#[test]
fn a_single_click_is_silent_but_a_short_burst_is_not() {
    let sr = SAMPLE_RATE as usize;
    let mut click = vec![0.0; 10 * sr];
    click[5 * sr..5 * sr + 20].fill(0.9);
    assert!(is_silent(&click), "a 1.25 ms click must not be captioned");
    let mut burst = vec![0.0; 10 * sr];
    burst[5 * sr..5 * sr + sr / 4].copy_from_slice(&tone(0.25, 0.05));
    assert!(!is_silent(&burst), "250 ms of tone is signal");
}

/// FALSIFY-SPEECHGATE-010 - long audio: a cue over a 400 s recording snaps to
/// the one voiced stretch, and a cue over the silence around it vanishes.
#[test]
fn long_audio_cues_snap_to_the_voice_and_silent_cues_vanish() {
    let sr = SAMPLE_RATE as usize;
    let mut a = vec![0.0; 400 * sr];
    a[300 * sr..310 * sr].copy_from_slice(&tone(10.0, 0.05));
    let result = TranscriptionResult {
        text: String::new(),
        language: "en".into(),
        segments: vec![
            cue(0.0, 29.9, "Thank you."),
            cue(298.0, 311.0, "so today we walk through how the build cache keys are derived from the lockfile and why a cold run is slower"),
            cue(390.0, 399.9, "Thanks for watching."),
        ],
        profiling: None,
    };
    let out = snap_to_signal(result, &a);
    assert_eq!(out.segments.len(), 1, "{:?}", out.segments);
    let c = &out.segments[0];
    assert!((c.start - 300.0).abs() < 0.05 && (c.end - 310.0).abs() < 0.05);
    assert_eq!(out.text, "so today we walk through how the build cache keys are derived from the lockfile and why a cold run is slower");
}

/// FALSIFY-SPEECHGATE-008 - genuine short speech survives the noise filter.
#[test]
fn replies_after_pauses_and_quiet_audio_are_classified() {
    let keep = |cues: Vec<Segment>| drop_noise_cues(cues).len();
    assert_eq!(
        keep(vec![
            cue(0.0, 3.0, "Is this a list or a tuple?"),
            cue(5.5, 6.1, "Yes."),
            cue(6.5, 9.0, "It is a tuple here.")
        ]),
        3
    );
    assert_eq!(
        keep(vec![
            cue(0.0, 5.0, "That is all for this lesson."),
            cue(7.1, 8.5, "See you next week.")
        ]),
        2
    );
    // -57 dBFS tone: not silent (>-60), but under the -50 dBFS caption floor
    let quiet = tone(5.0, 2.0e-3);
    assert!(!is_silent(&quiet));
    assert!(peak_frame_rms(&quiet) < SIGNAL_RMS);
}

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
    // reviewer cases: short replies, pauses, breaths, spoken literals, slow talkers
    let keep = |cues: Vec<Segment>| drop_noise_cues(cues).len();
    assert_eq!(
        keep(vec![
            cue(0.0, 4.0, "Is it a list or a tuple?"),
            cue(4.2, 4.45, "No."),
            cue(4.6, 8.0, "It is a tuple in this case.")
        ]),
        3
    );
    assert_eq!(
        keep(vec![
            cue(0.0, 3.0, "Is this a list or a tuple?"),
            cue(4.0, 4.6, "Yes."),
            cue(5.0, 8.0, "[1, 2, 3] is a list of three integers.")
        ]),
        3
    );
    assert_eq!(
        keep(vec![
            cue(0.0, 4.0, "That is all for this lesson."),
            cue(4.6, 5.3, "Thank you.")
        ]),
        2
    );
    assert_eq!(
        keep(vec![
            cue(0.0, 4.0, "Here is a list literal."),
            cue(4.2, 5.5, "[1, 2, 3]")
        ]),
        2
    );
    assert_eq!(keep(vec![cue(0.0, 4.0, "Okay, let's go.")]), 1);
}
