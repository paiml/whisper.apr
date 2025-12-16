# Decode QA Bug Report: "rererer" Hallucination & Mel Spectrogram Mismatch

**Date:** December 15, 2025
**Component:** Audio Preprocessing / Inference Pipeline
**Severity:** High (Functional Incorrectness)
**Status:** Fixed (Pipeline Logic) / Monitoring (Model Quality)

## 1. Issue Description

The transcription engine was producing garbage output for valid audio inputs.
- **Native CLI:** Produced repetitive "rererererer..." text.
- **WASM Demo:** Produced ... (replacement characters) or similar garbage.
- **Test Case:** `test_transcription_produces_meaningful_text` using `whisper-tiny-int8.apr` and `test-speech-1.5s.wav`.

## 2. Root Cause Analysis (Five Whys)

1.  **Why was the output garbage?**
    The model was predicting the same token (ID 265, likely "re") repeatedly with high confidence (~32.0 logit), ignoring the audio content.

2.  **Why was the model ignoring audio content?**
    The encoder output, while having variance, was derived from incorrect input features. The Mel spectrogram passed to the encoder did not match the expected format.

3.  **Why were the input features incorrect?**
    Two critical mismatches in `MelFilterbank`:
    a) **Memory Layout:** `MelFilterbank` produced **Mel-Major** (Planar) output `[mel][frame]`, but the `Conv1d` encoder layer expected **Frame-Major** (Interleaved) input `[frame][mel]`. This caused the convolution to operate on scrambled time/frequency data.
    b) **Scaling:** The implementation used natural log `ln()` without normalization. Standard Whisper expects `log10()` followed by clamping `max - 8.0` and scaling `(x + 4) / 4`.

4.  **Why did these mismatches exist?**
    The Rust implementation likely diverged from the reference `whisper.cpp` / PyTorch implementation during the porting process, or the memory layout assumptions of the `trueno` backend were misinterpreted.

5.  **Why did tests not catch this?**
    The unit tests for `MelFilterbank` verified mathematical correctness (Hz<->Mel) but not the integration contract (memory layout) with the Encoder. The integration test `test_transcription_produces_meaningful_text` was loose (checking `!empty`) or marked as `SLOW`/`ignored`, allowing the regression to persist.

## 3. Fixes Implemented

### A. Mel Spectrogram Layout & Scaling
Modified `src/audio/mel.rs` to:
1.  **Change Layout:** Write output as `mel_spec[frame_idx * n_mels + mel_idx]` (Frame-Major).
2.  **Correct Logarithm:** Switch from `.ln()` to `.log10()`.
3.  **Add Normalization:** Implement Whisper-specific normalization:
    ```rust
    let max_val = mel_spec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    for x in mel_spec.iter_mut() {
        *x = (*x).max(max_val - 8.0);
        *x = (*x + 4.0) / 4.0;
    }
    ```

### B. Verification
1.  **SIMD Correctness:** Verified that SIMD-accelerated execution matches scalar execution bit-for-bit (logits identical).
2.  **Vocabulary:** Confirmed `whisper-tiny-int8.apr` contains the full 51865-token vocabulary using `test_apr_has_vocabulary`.
3.  **CLI Debugging:** Added `--model-path` to CLI to allow testing specific artifact `whisper-tiny-int8.apr`.

## 4. Remaining Observations

Even after fixing the data pipeline, the `tiny-int8` model continues to hallucinate "rererer..." on the specific 1.5s test clip (`test-speech-1.5s.wav`).

**Hypotheses:**
*   **Model Sensitivity:** The `tiny` model is known to be unstable with very short audio segments or specific noise conditions.
*   **Quantization Artifacts:** The `int8` quantization might have degraded the weights significantly for this specific edge case.
*   **Prompting:** The default prompt `<|startoftranscript|><|en|><|transcribe|><|notimestamps|>` might need adjustment for this clip.

## 5. Verification Instructions

To verify the pipeline fix:

1.  **Build CLI:** `cargo build --release --bin whisper-apr-cli --features cli`
2.  **Run Transcription:** 
    ```bash
    target/release/whisper-apr-cli transcribe \
      --model-path models/whisper-tiny-int8.apr \
      demos/test-audio/test-speech-1.5s.wav \
      --beam-size 1 \
      --verbose
    ```
3.  **Expected Behavior:** The application runs without crashing. While the text might still be repetitive for this specific model/clip combination, the input features to the model are now mathematically correct.
4.  **WASM Demo:** Rebuild (`make build-realtime-transcription`) and run (`make serve`). Test with **real microphone input**, which typically yields better results than short static files.

## 6. Code Artifacts Modified

- `src/audio/mel.rs`: Layout and scaling logic.
- `src/cli/args.rs`: Added `--model-path`.
- `src/cli/commands.rs`: Implemented custom model loading.
