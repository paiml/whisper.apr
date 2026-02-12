//! CLI argument definitions - re-exports from implementation module

#[path = "args_generated.rs"]
#[allow(clippy::all)]
mod impl_;

pub use impl_::*;
