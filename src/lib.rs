//! # Whisper.apr
//!
//! WASM-first automatic speech recognition engine implementing OpenAI's Whisper architecture.
//!
//! ## Overview
//!
//! Whisper.apr is designed from inception for WASM deployment via `wasm32-unknown-unknown`,
//! leveraging Rust's superior WASM toolchain for:
//! - 30-40% smaller binary sizes through tree-shaking
//! - Native WASM SIMD 128-bit intrinsics without Emscripten overhead
//! - Zero-copy audio buffer handling via shared memory
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use whisper_apr::{WhisperApr, TranscribeOptions};
//!
//! let whisper = WhisperApr::load("base.apr")?;
//! let result = whisper.transcribe(&audio_samples, TranscribeOptions::default())?;
//! println!("{}", result.text);
//! ```
//!
//! ## Features
//!
//! - `std` (default): Standard library support
//! - `wasm`: WASM bindings via wasm-bindgen
//! - `simd`: SIMD acceleration via trueno
//! - `tracing`: Performance tracing via renacer

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![deny(clippy::unwrap_used)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(feature = "dhat-profiler")]
#[doc(hidden)]
pub use dhat;

pub mod audio;
pub mod detection;
pub mod error;
pub mod format;
pub mod inference;
pub mod memory;
pub mod model;
/// Unified parallelism abstraction for CLI and WASM (§11.3.2)
pub mod parallel;
pub mod progress;
pub mod simd;
pub mod timestamps;
pub mod tokenizer;
#[macro_use]
pub mod trace;
pub mod vad;

/// Speaker diarization module (who spoke when)
pub mod diarization;

/// WebGPU compute backend for accelerated inference
pub mod gpu;

/// Backend abstraction and automatic selection
pub mod backend;

/// Vocabulary and hotword customization
pub mod vocabulary;

/// Activation probing for forward-pass debugging (WAPR-MOONSHINE-013)
pub mod probe;

/// HuggingFace Hub publishing (WAPR-PUB-001)
pub mod publish;

/// Pre-publish verification (WAPR-PUB-001)
pub mod verify;

/// Benchmark infrastructure for multi-backend comparison
#[allow(clippy::all)]
pub mod benchmark_generated;

/// CUDA GPU acceleration via trueno-gpu/realizar
#[cfg(feature = "realizar-gpu")]
pub mod cuda;

/// Re-exports world-class production inference primitives from realizar.
///
/// Provides: Flash Attention, Sliding Window Attention,
/// FusedLayerNormLinear, KVCache, PagedKvCache,
/// Q4_K, Q5_K, Q6_K quantization with fused ops.
#[cfg(feature = "realizar-inference")]
pub mod realizar_inference {
    pub use realizar::layers::{
        Attention, FeedForward, FusedLayerNormLinear, KVCache, LayerNorm, Linear,
        MultiHeadAttention, SlidingWindowAttention,
    };

    pub use realizar::paged_kv::{PagedCacheError, PagedKvCache, SeqId};

    pub use realizar::quantize::{
        dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0, fused_q4k_dot_simd,
        fused_q4k_parallel_matvec, fused_q5k_dot_simd, fused_q5k_parallel_matvec,
        fused_q6k_dot_simd, fused_q6k_parallel_matvec, Q4_KBlock, Q5_KBlock, Q6_KBlock, Q8_0Block,
    };

    pub use realizar::tensor::Tensor;

    // Speculative decoding (Points 66-80)
    pub use realizar::speculative::{
        SpeculativeConfig, SpeculativeError, SpeculativeModel, SpeculativeResult, SpeculativeStats,
        TokenProb,
    };

    // GPU backend detection (Points 26-50)
    #[cfg(feature = "realizar-gpu")]
    pub use realizar::cuda::CudaExecutor;

    /// Check if CUDA GPU is available at runtime
    #[cfg(feature = "realizar-gpu")]
    pub fn gpu_available() -> bool {
        CudaExecutor::is_available()
    }
}

#[cfg(feature = "wasm")]
pub mod wasm;

/// CLI module for native command-line interface
#[cfg(feature = "cli")]
pub mod cli;

/// TUI module for pipeline visualization dashboard
#[cfg(feature = "tui")]
pub mod tui;

pub use error::{WhisperError, WhisperResult};

/// Core types and implementations
#[allow(clippy::all)]
mod core_generated;
pub use core_generated::*;
