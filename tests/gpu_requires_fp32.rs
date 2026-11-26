//! A library constructor must not change the process environment, and must not build a
//! model that would multiply with the nondeterministic FP16 kernel (whisper.apr#103).
//! Needs a CUDA device and `WAPR_MODEL`. One test per binary: it reads the environment.
#![cfg(feature = "realizar-gpu")]

use whisper_apr::WhisperApr;

#[test]
#[ignore = "needs a CUDA device and WAPR_MODEL"]
fn model_construction_fails_without_the_fp32_switch() {
    const SWITCH: &str = "TRUENO_FORCE_FP32_GEMM";
    assert!(std::env::var_os(SWITCH).is_none(), "run without {SWITCH}");
    let path = std::env::var("WAPR_MODEL").expect("WAPR_MODEL");
    let bytes = std::fs::read(path).expect("model file");
    let whisper = WhisperApr::load_from_apr(&bytes).expect("model");

    let error = whisper.into_cuda(0).err().expect("construction must fail");
    assert!(error.to_string().contains(SWITCH), "{error}");
    assert!(
        std::env::var_os(SWITCH).is_none(),
        "environment was changed"
    );
}
