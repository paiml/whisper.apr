//! Tracing utilities for pipeline instrumentation
//!
//! This module provides conditional tracing support for the whisper.apr pipeline.
//! When the `tracing` feature is enabled, spans are emitted for renacer integration.
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::trace::span;
//!
//! fn my_function() {
//!     let _span = span!("step_f_mel");
//!     // ... function body
//! }
//! ```
//!
//! # Feature Flag
//!
//! Enable with `--features tracing` to emit spans.
//! Without the feature, the macros compile to no-ops.

/// Create a tracing span (no-op when tracing feature is disabled)
///
/// This macro creates a span that is recorded by renacer when the `tracing`
/// feature is enabled. When disabled, it compiles to nothing.
#[macro_export]
#[cfg(feature = "tracing")]
macro_rules! trace_span {
    ($name:expr) => {
        tracing::span!(tracing::Level::DEBUG, $name)
    };
    ($name:expr, $($field:tt)*) => {
        tracing::span!(tracing::Level::DEBUG, $name, $($field)*)
    };
}

/// Create a tracing span (no-op when tracing feature is disabled)
#[macro_export]
#[cfg(not(feature = "tracing"))]
macro_rules! trace_span {
    ($name:expr) => {
        ()
    };
    ($name:expr, $($field:tt)*) => {
        ()
    };
}

/// Placeholder for span guard when tracing is disabled
#[cfg(not(feature = "tracing"))]
pub struct NoopSpanGuard;

/// Enter a tracing span (no-op when tracing feature is disabled)
#[macro_export]
#[cfg(feature = "tracing")]
macro_rules! trace_enter {
    ($name:expr) => {
        tracing::span!(tracing::Level::DEBUG, $name).entered()
    };
}

/// Enter a tracing span (no-op when tracing feature is disabled)
#[macro_export]
#[cfg(not(feature = "tracing"))]
macro_rules! trace_enter {
    ($name:expr) => {
        $crate::trace::NoopSpanGuard
    };
}

/// Log a tracing event (no-op when tracing feature is disabled)
#[macro_export]
#[cfg(feature = "tracing")]
macro_rules! trace_event {
    ($($arg:tt)*) => {
        tracing::debug!($($arg)*)
    };
}

/// Log a tracing event (no-op when tracing feature is disabled)
#[macro_export]
#[cfg(not(feature = "tracing"))]
macro_rules! trace_event {
    ($($arg:tt)*) => {};
}

// Re-export macros at module level
pub use trace_enter;
pub use trace_event;
pub use trace_span;

#[cfg(test)]
mod tests {

    #[test]
    fn test_trace_macros_compile() {
        // These should compile regardless of feature flag
        let _span = trace_span!("test_span");
        let _guard = trace_enter!("test_enter");
        trace_event!("test event");
    }
}
