//! LFM2 Model Implementation (WAPR-LFM2-002)
//!
//! This module implements the LiquidAI LFM2-2.6B-Transcript architecture
//! for post-transcription summarization.
//!
//! # Architecture Overview
//!
//! LFM2 is a hybrid conv/attention architecture with:
//! - Grouped Query Attention (GQA) with 32 Q heads and 8 KV heads
//! - SwiGLU FFN activation
//! - 1D Convolution layers interleaved with attention
//! - RoPE positional encoding (θ = 1,000,000)
//!
//! # Module Structure
//!
//! ```text
//! lfm2/
//! ├── mod.rs       # This file - module exports
//! ├── gqa.rs       # Grouped Query Attention
//! ├── swiglu.rs    # SwiGLU FFN activation
//! ├── rope.rs      # RoPE positional encoding
//! ├── conv.rs      # 1D Convolution layer
//! ├── model.rs     # LFM2 model struct
//! └── tokenizer.rs # BPE tokenizer (WAPR-LFM2-007)
//! ```
//!
//! # Spec Reference
//!
//! See `docs/specifications/1.0-whisper-apr.md` Section 18 for full specification.

pub mod conv;
pub mod gqa;
pub mod model;
pub mod rope;
pub mod swiglu;
pub mod tokenizer;

pub use conv::Conv1d;
pub use gqa::GroupedQueryAttention;
pub use model::{
    GenerationStats, Lfm2, Lfm2Layer, Lfm2WasmConfig, LoadStats, WasmMemoryEstimate,
    WasmQuantization,
};
pub use rope::RotaryEmbedding;
pub use swiglu::SwiGluFfn;
pub use tokenizer::{ByteLevelTokenizer, Lfm2Tokenizer, SpecialTokens};
