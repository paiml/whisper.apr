//! Moonshine model blocks (WAPR-MOONSHINE-002)
//!
//! Moonshine encoder and decoder blocks composed from existing LFM2 components:
//! - `RmsNorm` for pre-normalization
//! - `GroupedQueryAttention` for MHA self-attention and cross-attention
//! - `MlpFfn` for feed-forward (GELU in encoder, SiLU in decoder)
//! - `RotaryEmbedding` for position encoding (applied within attention)

pub mod decoder_block;
pub mod encoder_block;

pub use decoder_block::MoonshineDecoderBlock;
pub use encoder_block::MoonshineEncoderBlock;
