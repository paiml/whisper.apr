//! GPU error types

use std::fmt;

/// Result type for GPU operations
pub type GpuResult<T> = Result<T, GpuError>;

/// GPU operation errors
#[derive(Debug, Clone)]
pub enum GpuError {
    /// WebGPU is not available on this platform
    NotAvailable,
    /// Failed to request GPU adapter
    AdapterNotFound,
    /// Failed to create GPU device
    DeviceCreationFailed(String),
    /// Buffer operation failed
    BufferError(String),
    /// Shader compilation failed
    ShaderError(String),
    /// Pipeline creation failed
    PipelineError(String),
    /// Compute operation failed
    ComputeError(String),
    /// Invalid buffer size
    InvalidBufferSize {
        /// Requested buffer size in bytes
        requested: u64,
        /// Maximum allowed buffer size in bytes
        max: u64,
    },
    /// Buffer alignment error
    AlignmentError {
        /// Actual offset that was provided
        offset: u64,
        /// Required alignment in bytes
        required: u64,
    },
    /// Device lost during operation
    DeviceLost,
    /// Out of GPU memory
    OutOfMemory,
    /// Unsupported operation for this backend
    UnsupportedOperation(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAvailable => write!(f, "WebGPU is not available on this platform"),
            Self::AdapterNotFound => write!(f, "Failed to find a suitable GPU adapter"),
            Self::DeviceCreationFailed(msg) => write!(f, "GPU device creation failed: {msg}"),
            Self::BufferError(msg) => write!(f, "GPU buffer error: {msg}"),
            Self::ShaderError(msg) => write!(f, "Shader compilation error: {msg}"),
            Self::PipelineError(msg) => write!(f, "Pipeline creation error: {msg}"),
            Self::ComputeError(msg) => write!(f, "Compute operation error: {msg}"),
            Self::InvalidBufferSize { requested, max } => {
                write!(
                    f,
                    "Invalid buffer size: requested {requested} bytes, max {max}"
                )
            }
            Self::AlignmentError { offset, required } => {
                write!(
                    f,
                    "Buffer alignment error: offset {offset} not aligned to {required}"
                )
            }
            Self::DeviceLost => write!(f, "GPU device was lost"),
            Self::OutOfMemory => write!(f, "GPU out of memory"),
            Self::UnsupportedOperation(op) => write!(f, "Unsupported GPU operation: {op}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl GpuError {
    /// Create a buffer error
    #[must_use]
    pub fn buffer(msg: impl Into<String>) -> Self {
        Self::BufferError(msg.into())
    }

    /// Create a shader error
    #[must_use]
    pub fn shader(msg: impl Into<String>) -> Self {
        Self::ShaderError(msg.into())
    }

    /// Create a pipeline error
    #[must_use]
    pub fn pipeline(msg: impl Into<String>) -> Self {
        Self::PipelineError(msg.into())
    }

    /// Create a compute error
    #[must_use]
    pub fn compute(msg: impl Into<String>) -> Self {
        Self::ComputeError(msg.into())
    }

    /// Create a device creation error
    #[must_use]
    pub fn device(msg: impl Into<String>) -> Self {
        Self::DeviceCreationFailed(msg.into())
    }

    /// Check if this is a recoverable error
    #[must_use]
    pub fn is_recoverable(&self) -> bool {
        !matches!(
            self,
            Self::NotAvailable | Self::DeviceLost | Self::OutOfMemory
        )
    }

    /// Check if this indicates GPU is unavailable
    #[must_use]
    pub fn is_unavailable(&self) -> bool {
        matches!(self, Self::NotAvailable | Self::AdapterNotFound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        assert!(GpuError::NotAvailable.to_string().contains("not available"));
        assert!(GpuError::AdapterNotFound.to_string().contains("adapter"));
        assert!(GpuError::DeviceLost.to_string().contains("lost"));
        assert!(GpuError::OutOfMemory.to_string().contains("memory"));
    }

    #[test]
    fn test_error_constructors() {
        let err = GpuError::buffer("test");
        assert!(matches!(err, GpuError::BufferError(_)));

        let err = GpuError::shader("test");
        assert!(matches!(err, GpuError::ShaderError(_)));

        let err = GpuError::pipeline("test");
        assert!(matches!(err, GpuError::PipelineError(_)));

        let err = GpuError::compute("test");
        assert!(matches!(err, GpuError::ComputeError(_)));

        let err = GpuError::device("test");
        assert!(matches!(err, GpuError::DeviceCreationFailed(_)));
    }

    #[test]
    fn test_error_is_recoverable() {
        assert!(GpuError::BufferError("test".into()).is_recoverable());
        assert!(GpuError::ShaderError("test".into()).is_recoverable());
        assert!(!GpuError::NotAvailable.is_recoverable());
        assert!(!GpuError::DeviceLost.is_recoverable());
        assert!(!GpuError::OutOfMemory.is_recoverable());
    }

    #[test]
    fn test_error_is_unavailable() {
        assert!(GpuError::NotAvailable.is_unavailable());
        assert!(GpuError::AdapterNotFound.is_unavailable());
        assert!(!GpuError::DeviceLost.is_unavailable());
        assert!(!GpuError::BufferError("test".into()).is_unavailable());
    }

    #[test]
    fn test_invalid_buffer_size_display() {
        let err = GpuError::InvalidBufferSize {
            requested: 1000,
            max: 500,
        };
        let msg = err.to_string();
        assert!(msg.contains("1000"));
        assert!(msg.contains("500"));
    }

    #[test]
    fn test_alignment_error_display() {
        let err = GpuError::AlignmentError {
            offset: 100,
            required: 256,
        };
        let msg = err.to_string();
        assert!(msg.contains("100"));
        assert!(msg.contains("256"));
    }

    #[test]
    fn test_unsupported_operation() {
        let err = GpuError::UnsupportedOperation("sparse matmul".into());
        assert!(err.to_string().contains("sparse matmul"));
    }
}
