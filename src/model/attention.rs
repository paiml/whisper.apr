//! Multi-head attention - re-exports from implementation module

#[path = "attention_generated.rs"]
#[allow(clippy::all)]
mod impl_;

pub use impl_::*;
