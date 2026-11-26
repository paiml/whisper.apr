//! whisper.apr#103: the CUDA encoder is fed the full 30 s window Whisper is trained on.
//! Unpadded chunks dropped most of a 20 s take's words, and odd encoder lengths faulted
//! the WMMA kernel (CUDA error 716).

use whisper_apr::audio::{gpu_chunk_len, GPU_CHUNK_MAX};

#[test]
fn every_chunk_is_padded_to_thirty_seconds() {
    for samples in [1, 99_520, 320_000, 479_999, GPU_CHUNK_MAX] {
        assert_eq!(gpu_chunk_len(samples), GPU_CHUNK_MAX, "{samples} samples");
    }
}

#[test]
fn a_longer_buffer_is_never_truncated() {
    assert_eq!(gpu_chunk_len(GPU_CHUNK_MAX + 1), GPU_CHUNK_MAX + 1);
}
