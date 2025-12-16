//! Web Worker threading support for parallel WASM execution
//!
//! Provides thread pool initialization and management for wasm-bindgen-rayon.
//!
//! # Architecture (Spec 10.2)
//!
//! The twin-binary strategy ensures the application always runs:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Capability Detection                     │
//! └──────────────────────────┬──────────────────────────────────┘
//!                            │
//!                   ┌────────▼────────┐
//!                   │ crossOriginIsolated │
//!                   │       true?         │
//!                   └────────┬────────────┘
//!                      yes/  \no
//!                         /    \
//!        ┌───────────────▼─┐  ┌─▼───────────────────┐
//!        │  SIMD+Threaded  │  │   SIMD Sequential   │
//!        │  (SharedArrayBuffer) │  │   (Fallback)   │
//!        └─────────────────┘  └─────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```javascript
//! import init, { initThreadPool, isThreadedAvailable } from 'whisper-apr';
//!
//! await init();
//!
//! if (isThreadedAvailable()) {
//!     // Initialize thread pool with optimal worker count
//!     await initThreadPool(navigator.hardwareConcurrency - 1);
//! }
//! ```
//!
//! # Building
//!
//! ```bash
//! # Threaded build (requires wasm-pack with --target web)
//! RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
//!   wasm-pack build --target web --features parallel -- -Z build-std=std,panic_abort
//!
//! # Sequential fallback
//! wasm-pack build --target web --features simd
//! ```

use wasm_bindgen::prelude::*;

#[cfg(feature = "parallel")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Check if threading is available in current environment
///
/// Returns true if SharedArrayBuffer is available (requires COOP/COEP headers)
#[wasm_bindgen(js_name = isThreadedAvailable)]
pub fn is_threaded_available() -> bool {
    #[wasm_bindgen(
        inline_js = "export function crossOriginIsolated() { return globalThis.crossOriginIsolated === true; }"
    )]
    extern "C" {
        #[wasm_bindgen(js_name = crossOriginIsolated)]
        fn cross_origin_isolated() -> bool;
    }

    cross_origin_isolated()
}

/// Get optimal thread count for this environment
///
/// Per spec 10.3: N_threads = max(1, min(hardwareConcurrency - 1, N_limit))
#[wasm_bindgen(js_name = optimalThreadCount)]
pub fn optimal_thread_count() -> usize {
    #[wasm_bindgen(
        inline_js = "export function hardwareConcurrency() { return navigator.hardwareConcurrency || 1; }"
    )]
    extern "C" {
        #[wasm_bindgen(js_name = hardwareConcurrency)]
        fn hardware_concurrency() -> usize;
    }

    let hw_threads = hardware_concurrency();

    if hw_threads <= 1 {
        return 1;
    }

    // Reserve 1 thread for UI/audio, cap at 8 for diminishing returns
    let available = hw_threads.saturating_sub(1);
    available.clamp(1, 8)
}

/// Execution mode based on available capabilities
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadingMode {
    /// Multi-threaded with SharedArrayBuffer
    Parallel,
    /// Single-threaded sequential execution
    Sequential,
}

impl ThreadingMode {
    /// Detect and return the current threading mode
    #[must_use]
    pub fn detect() -> Self {
        if is_threaded_available() {
            Self::Parallel
        } else {
            Self::Sequential
        }
    }

    /// Get human-readable name
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Parallel => "Parallel (Web Workers)",
            Self::Sequential => "Sequential (Single-threaded)",
        }
    }

    /// Get expected performance multiplier vs parallel
    #[must_use]
    pub const fn performance_ratio(self) -> f32 {
        match self {
            Self::Parallel => 1.0,
            Self::Sequential => 4.0, // ~4x slower without threading
        }
    }
}

/// Get the current threading mode
#[wasm_bindgen(js_name = getThreadingMode)]
pub fn get_threading_mode() -> ThreadingMode {
    ThreadingMode::detect()
}

/// Get threading mode name
#[wasm_bindgen(js_name = getThreadingModeName)]
pub fn get_threading_mode_name() -> String {
    ThreadingMode::detect().name().to_string()
}

// ============================================================================
// Parallel execution helpers (only available with parallel feature)
// ============================================================================

#[cfg(feature = "parallel")]
mod parallel {
    use rayon::prelude::*;

    /// Execute a parallel map operation
    pub fn parallel_map<T, U, F>(items: &[T], f: F) -> Vec<U>
    where
        T: Sync,
        U: Send,
        F: Fn(&T) -> U + Sync + Send,
    {
        items.par_iter().map(f).collect()
    }

    /// Execute a parallel reduce operation
    pub fn parallel_reduce<T, F, R>(items: &[T], identity: T, f: F, reduce: R) -> T
    where
        T: Send + Sync + Clone,
        F: Fn(&T) -> T + Sync + Send,
        R: Fn(T, T) -> T + Sync + Send,
    {
        items.par_iter().map(f).reduce(|| identity.clone(), reduce)
    }

    /// Parallel matrix multiplication helper
    pub fn parallel_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];

        c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                row[j] = sum;
            }
        });

        c
    }
}

#[cfg(feature = "parallel")]
pub use parallel::*;

// ============================================================================
// Sequential fallback (when parallel feature is disabled)
// ============================================================================

#[cfg(not(feature = "parallel"))]
mod sequential {
    /// Sequential map operation (fallback)
    pub fn parallel_map<T, U, F>(items: &[T], f: F) -> Vec<U>
    where
        F: Fn(&T) -> U,
    {
        items.iter().map(f).collect()
    }

    /// Sequential reduce operation (fallback)
    pub fn parallel_reduce<T, F, R>(items: &[T], identity: T, f: F, reduce: R) -> T
    where
        T: Clone,
        F: Fn(&T) -> T,
        R: Fn(T, T) -> T,
    {
        items.iter().map(f).fold(identity, reduce)
    }

    /// Sequential matrix multiplication (fallback)
    pub fn parallel_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }

        c
    }
}

#[cfg(not(feature = "parallel"))]
pub use sequential::*;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threading_mode_name() {
        assert!(ThreadingMode::Parallel.name().contains("Parallel"));
        assert!(ThreadingMode::Sequential.name().contains("Sequential"));
    }

    #[test]
    fn test_threading_mode_performance_ratio() {
        assert!((ThreadingMode::Parallel.performance_ratio() - 1.0).abs() < f32::EPSILON);
        assert!(ThreadingMode::Sequential.performance_ratio() > 1.0);
    }

    #[test]
    fn test_parallel_map() {
        let items = vec![1, 2, 3, 4, 5];
        let result = parallel_map(&items, |x| x * 2);
        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_parallel_reduce() {
        let items = vec![1, 2, 3, 4, 5];
        let result = parallel_reduce(&items, 0, |x| *x, |a, b| a + b);
        assert_eq!(result, 15);
    }

    #[test]
    fn test_parallel_matmul_2x2() {
        // 2x2 identity * 2x2 values
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = parallel_matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_parallel_matmul_3x3() {
        // Simple 3x3 multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let c = parallel_matmul(&a, &b, 3, 3, 3);
        assert_eq!(c, a); // Identity multiplication
    }

    #[test]
    fn test_parallel_matmul_nonsquare() {
        // 2x3 * 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let c = parallel_matmul(&a, &b, 2, 2, 3);

        // Expected: [[22, 28], [49, 64]]
        let expected = vec![22.0, 28.0, 49.0, 64.0];
        assert_eq!(c, expected);
    }
}
