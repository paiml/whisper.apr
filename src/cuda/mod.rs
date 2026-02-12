//! CUDA GPU acceleration - re-exports from implementation module

#[path = "impl_generated.rs"]
#[allow(clippy::all)]
mod impl_;

pub use impl_::*;

#[cfg(test)]
mod tests_generated;
