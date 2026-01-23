//! Multi-head attention - re-exports from implementation module

#[path = "attention_generated.rs"]
mod impl_;

pub use impl_::*;
