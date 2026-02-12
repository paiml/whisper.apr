//! Quantized inference - re-exports from implementation module

#[path = "quantized_generated.rs"]
#[allow(clippy::all)]
mod impl_;

pub use impl_::*;
