//! Moonshine model blocks (WAPR-MOONSHINE-002)
//!
//! Moonshine encoder and decoder blocks composed from existing LFM2 components:
//! - `RmsNorm` for pre-normalization
//! - `GroupedQueryAttention` for GQA self-attention and cross-attention
//! - `SwiGluFfn` for gated feed-forward
//! - `RotaryEmbedding` for position encoding (applied within attention)

pub mod decoder_block;
pub mod encoder_block;

pub use decoder_block::MoonshineDecoderBlock;
pub use encoder_block::MoonshineEncoderBlock;
