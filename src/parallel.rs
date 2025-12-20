//! Unified parallelism abstraction for CLI and WASM
//!
//! Provides a consistent API for parallel computation that works on both
//! native (via rayon) and WASM (via wasm-bindgen-rayon) targets.
//!
//! # Design (per §11.3.2)
//!
//! - `parallel` feature: Enables multi-threaded execution
//! - Sequential fallback: Works without `parallel` feature
//! - Same API: `parallel_map` works identically on all targets
//!
//! # Citations
//!
//! - [29] Amdahl's Law for speedup analysis
//! - [31] "Attention Is All You Need" - independent head computation
//! - [32] Work-stealing scheduler design (rayon)

use crate::error::WhisperResult;

/// Configure the global thread pool with the specified number of threads.
///
/// This must be called before any parallel operations. If called multiple times,
/// only the first call takes effect (rayon limitation).
///
/// # Arguments
///
/// * `num_threads` - Number of threads to use. If None, uses rayon's default
///   (typically the number of logical CPUs).
///
/// # Returns
///
/// Ok(actual_threads) on success, Err if thread pool already initialized differently.
#[cfg(feature = "parallel")]
pub fn configure_thread_pool(num_threads: Option<u32>) -> WhisperResult<usize> {
    use rayon::ThreadPoolBuilder;

    let builder = ThreadPoolBuilder::new();
    let builder = if let Some(n) = num_threads {
        builder.num_threads(n as usize)
    } else {
        builder // Use rayon's default (logical CPU count)
    };

    match builder.build_global() {
        Ok(()) => Ok(rayon::current_num_threads()),
        Err(_) => {
            // Thread pool already initialized - return current thread count
            Ok(rayon::current_num_threads())
        }
    }
}

/// Sequential fallback - no thread pool to configure.
#[cfg(not(feature = "parallel"))]
pub fn configure_thread_pool(num_threads: Option<u32>) -> WhisperResult<usize> {
    let _ = num_threads;
    Ok(1)
}

/// Parallel map over a range of indices.
///
/// When the `parallel` feature is enabled, this uses rayon's parallel iterator.
/// Otherwise, it falls back to sequential iteration.
///
/// # Arguments
///
/// * `range` - The range of indices to iterate over
/// * `f` - A function that maps each index to a result
///
/// # Returns
///
/// A vector of results in index order (deterministic).
///
/// # Example
///
/// ```ignore
/// let results = parallel_map(0..6, |head| {
///     compute_attention_head(head)
/// });
/// ```
#[cfg(feature = "parallel")]
pub fn parallel_map<T, F>(range: std::ops::Range<usize>, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    use rayon::prelude::*;
    range.into_par_iter().map(f).collect()
}

/// Sequential fallback when `parallel` feature is disabled.
#[cfg(not(feature = "parallel"))]
pub fn parallel_map<T, F>(range: std::ops::Range<usize>, f: F) -> Vec<T>
where
    F: Fn(usize) -> T,
{
    range.map(f).collect()
}

/// Parallel map that collects Results, short-circuiting on first error.
///
/// # Arguments
///
/// * `range` - The range of indices to iterate over
/// * `f` - A function that maps each index to a WhisperResult
///
/// # Returns
///
/// Ok(Vec) if all succeed, Err on first failure.
#[cfg(feature = "parallel")]
pub fn parallel_try_map<T, F>(range: std::ops::Range<usize>, f: F) -> WhisperResult<Vec<T>>
where
    T: Send,
    F: Fn(usize) -> WhisperResult<T> + Send + Sync,
{
    use rayon::prelude::*;
    range.into_par_iter().map(f).collect()
}

/// Sequential fallback for try_map.
#[cfg(not(feature = "parallel"))]
pub fn parallel_try_map<T, F>(range: std::ops::Range<usize>, f: F) -> WhisperResult<Vec<T>>
where
    F: Fn(usize) -> WhisperResult<T>,
{
    range.map(f).collect()
}

/// Check if parallel execution is available.
///
/// Returns true if:
/// - `parallel` feature is enabled, AND
/// - On WASM: SharedArrayBuffer is available
/// - On native: Always true (rayon works)
#[cfg(feature = "parallel")]
pub fn is_parallel_available() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        crate::wasm::threading::is_threaded_available()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        true
    }
}

/// Sequential fallback - parallel not available.
#[cfg(not(feature = "parallel"))]
pub fn is_parallel_available() -> bool {
    false
}

/// Get the number of threads available for parallel execution.
#[cfg(feature = "parallel")]
pub fn thread_count() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        crate::wasm::threading::optimal_thread_count()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        rayon::current_num_threads()
    }
}

/// Sequential fallback - always 1 thread.
#[cfg(not(feature = "parallel"))]
pub fn thread_count() -> usize {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_map_basic() {
        let results = parallel_map(0..4, |i| i * 2);
        assert_eq!(results, vec![0, 2, 4, 6]);
    }

    #[test]
    fn test_parallel_map_order_preserved() {
        // Verify deterministic ordering (important for attention heads)
        let results = parallel_map(0..100, |i| i);
        let expected: Vec<usize> = (0..100).collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_parallel_try_map_success() {
        let results = parallel_try_map(0..4, |i| Ok(i * 2));
        assert!(results.is_ok());
        assert_eq!(results.unwrap(), vec![0, 2, 4, 6]);
    }

    #[test]
    fn test_parallel_try_map_error() {
        let results: WhisperResult<Vec<i32>> = parallel_try_map(0..4, |i| {
            if i == 2 {
                Err(crate::error::WhisperError::Model("test error".into()))
            } else {
                Ok(i as i32)
            }
        });
        assert!(results.is_err());
    }

    #[test]
    fn test_thread_count() {
        let count = thread_count();
        assert!(count >= 1);
        #[cfg(feature = "parallel")]
        {
            // With parallel feature, should have multiple threads on multi-core
            // (unless running on single-core, which is rare)
            println!("Thread count: {}", count);
        }
    }

    #[test]
    fn test_is_parallel_available() {
        let available = is_parallel_available();
        #[cfg(feature = "parallel")]
        assert!(available, "parallel feature enabled but not available");
        #[cfg(not(feature = "parallel"))]
        assert!(!available, "parallel should not be available without feature");
    }
}
