//! Backend abstraction and selection (WAPR-140 to WAPR-141)
//!
//! Provides a unified interface for compute operations that can run on
//! different backends (SIMD, GPU) with automatic selection based on
//! workload characteristics.
//!
//! # Usage
//!
//! ```rust,ignore
//! use whisper_apr::backend::{BackendSelector, SelectorConfig, MatMulOp};
//!
//! let selector = BackendSelector::new(SelectorConfig::for_inference());
//! let op = MatMulOp::new(1024, 768, 1024);
//! let selection = selector.select(&op);
//!
//! println!("Selected backend: {}", selection);
//! ```

mod selector;
mod traits;

pub use selector::{BackendSelection, BackendSelector, SelectorConfig, SelectionStrategy};
pub use traits::{
    BackendCapabilities, BackendType, ComputeOp, GeluOp, LayerNormOp, MatMulOp, SoftmaxOp,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all public types are accessible
        let _ = std::any::type_name::<BackendType>();
        let _ = std::any::type_name::<BackendCapabilities>();
        let _ = std::any::type_name::<BackendSelector>();
        let _ = std::any::type_name::<BackendSelection>();
        let _ = std::any::type_name::<SelectorConfig>();
        let _ = std::any::type_name::<SelectionStrategy>();
        let _ = std::any::type_name::<MatMulOp>();
        let _ = std::any::type_name::<SoftmaxOp>();
        let _ = std::any::type_name::<LayerNormOp>();
        let _ = std::any::type_name::<GeluOp>();
    }
}
