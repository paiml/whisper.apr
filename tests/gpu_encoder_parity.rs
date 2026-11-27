//! The CUDA encoder must produce what the CPU encoder produces from the same weights
//! (whisper.apr#103). Needs a CUDA device and a Whisper `.apr` model:
//! `TRUENO_FORCE_FP32_GEMM=1 WAPR_MODEL=<model.apr> cargo test --release --features realizar-gpu
//! --test gpu_encoder_parity -- --ignored --nocapture --test-threads=1`.
#![cfg(feature = "realizar-gpu")]

use whisper_apr::{TranscribeOptions, WhisperApr};

/// A swept tone with a syllable-like envelope, so every mel band moves.
fn signal(samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            let envelope = (t * 4.0 * std::f32::consts::PI).sin().abs();
            0.3 * envelope * (t * (200.0 + 110.0 * t) * std::f32::consts::TAU).sin()
        })
        .collect()
}

fn model() -> WhisperApr {
    assert!(
        std::env::var("TRUENO_FORCE_FP32_GEMM").is_ok(),
        "run with TRUENO_FORCE_FP32_GEMM=1"
    );
    let path = std::env::var("WAPR_MODEL").expect("WAPR_MODEL");
    let bytes = std::fs::read(path).expect("model file");
    WhisperApr::load_from_apr(&bytes).expect("model")
}

/// Root-mean-square of the difference, relative to that of `want`.
fn relative_error(got: &[f32], want: &[f32]) -> f32 {
    let diff: f32 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
    let norm: f32 = want.iter().map(|w| w * w).sum();
    (diff / norm).sqrt()
}

#[test]
#[ignore = "needs a CUDA device and WAPR_MODEL"]
fn gpu_encoder_matches_cpu_encoder() {
    let whisper = model();
    let mel = whisper.compute_mel(&signal(480_000)).expect("mel");
    let want = whisper.encode(&mel).expect("CPU encoder");

    let mut cuda = whisper.into_cuda(0).expect("CUDA model");
    let got = cuda.encode_gpu_total_offload(&mel).expect("GPU encoder");

    assert_eq!(got.len(), want.len());
    let error = relative_error(&got, &want);
    println!("encoder relative error: {error}");
    // measured 1e-4
    assert!(
        error < 1e-3,
        "GPU encoder differs from CPU: relative error {error}"
    );
}

/// The encoder holds 30 s: longer audio has to be split before it gets there.
#[test]
#[ignore = "needs a CUDA device and WAPR_MODEL"]
fn gpu_transcribes_audio_longer_than_the_window() {
    let mut cuda = model().into_cuda(0).expect("CUDA model");
    let audio = signal(31 * 16_000);
    cuda.transcribe_gpu(&audio, TranscribeOptions::default())
        .expect("31 s transcription");
}
