//! Moonshine model blocks (WAPR-MOONSHINE-002)
//!
//! Moonshine encoder and decoder blocks composed from existing LFM2 components:
//! - `LayerNormNoBias` for pre-normalization (weight-only, no bias)
//! - `GroupedQueryAttention` for MHA self-attention and cross-attention
//! - `MlpFfn` for encoder feed-forward (GELU), `GatedMlpFfn` for decoder (SiLU gate)
//! - `RotaryEmbedding` for position encoding (applied within attention)

pub mod decoder_block;
pub mod encoder_block;

pub use decoder_block::MoonshineDecoderBlock;
pub use encoder_block::MoonshineEncoderBlock;
