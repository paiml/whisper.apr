//! Streaming audio processor - re-exports from implementation module

#[path = "streaming_generated.rs"]
#[allow(clippy::all)]
mod impl_;

pub use impl_::*;
