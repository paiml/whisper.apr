//! SafeTensors Export Module (WAPR-PUB-001)
//!
//! Exports whisper.apr models to HuggingFace SafeTensors format.
//! Pure Rust implementation - no external safetensors crate dependency.
//!
//! # Format Specification
//!
//! SafeTensors binary format:
//! ```text
//! [8 bytes: u64 LE header length]
//! [N bytes: JSON header (tensor metadata)]
//! [remaining: raw tensor data]
//! ```
//!
//! # Stack Integration
//!
//! This module follows the same patterns as:
//! - `aprender::serialization::safetensors::save_safetensors`
//! - Compatible with `realizar::safetensors::SafetensorsModel::load`
//!
//! # Feature Requirements
//!
//! Requires `cli` feature for full functionality (serde, serde_json).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::{WhisperError, WhisperResult};

/// Tensor data for SafeTensors export.
#[derive(Debug, Clone)]
pub struct TensorData {
    /// F32 tensor values
    pub data: Vec<f32>,
    /// Tensor shape dimensions
    pub shape: Vec<usize>,
}

impl TensorData {
    /// Create new tensor data.
    #[must_use]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    /// Calculate expected element count from shape.
    #[must_use]
    pub fn expected_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Validate tensor data matches shape.
    pub fn validate(&self) -> WhisperResult<()> {
        let expected = self.expected_elements();
        if self.data.len() != expected {
            return Err(WhisperError::Format(format!(
                "Tensor shape {:?} expects {} elements, got {}",
                self.shape,
                expected,
                self.data.len()
            )));
        }
        Ok(())
    }

    /// Byte size of tensor data (F32 = 4 bytes per element).
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.data.len() * 4
    }
}

/// SafeTensors exporter for whisper.apr models.
///
/// Writes tensors in SafeTensors format without external crate dependencies.
pub struct SafeTensorsExporter;

impl SafeTensorsExporter {
    /// Save tensors to SafeTensors format.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `tensors` - Map of tensor names to data (BTreeMap for deterministic ordering)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tensor validation fails
    /// - File I/O fails
    pub fn save<P: AsRef<Path>>(
        path: P,
        tensors: &BTreeMap<String, TensorData>,
    ) -> WhisperResult<()> {
        Self::save_with_metadata(path, tensors, None)
    }

    /// Save tensors with optional metadata.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `tensors` - Map of tensor names to data
    /// * `metadata` - Optional key-value metadata pairs
    pub fn save_with_metadata<P: AsRef<Path>>(
        path: P,
        tensors: &BTreeMap<String, TensorData>,
        metadata: Option<BTreeMap<String, String>>,
    ) -> WhisperResult<()> {
        // Validate all tensors first
        for (name, tensor) in tensors {
            tensor.validate().map_err(|e| {
                WhisperError::Format(format!("Tensor '{}' validation failed: {}", name, e))
            })?;
        }

        // Build JSON header manually (no serde dependency)
        let mut header_parts: Vec<String> = Vec::new();

        // Add __metadata__ if provided
        if let Some(meta) = metadata {
            let meta_entries: Vec<String> = meta
                .iter()
                .map(|(k, v)| format!("\"{}\":\"{}\"", escape_json(k), escape_json(v)))
                .collect();
            if !meta_entries.is_empty() {
                header_parts.push(format!("\"__metadata__\":{{{}}}", meta_entries.join(",")));
            }
        }

        // Calculate offsets and add tensor metadata
        let mut current_offset = 0usize;
        for (name, tensor) in tensors {
            let byte_size = tensor.byte_size();
            let shape_str: Vec<String> = tensor.shape.iter().map(|s| s.to_string()).collect();

            let tensor_meta = format!(
                "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                escape_json(name),
                shape_str.join(","),
                current_offset,
                current_offset + byte_size
            );
            header_parts.push(tensor_meta);
            current_offset += byte_size;
        }

        let header_json = format!("{{{}}}", header_parts.join(","));

        // Pad header to 8-byte alignment
        let header_bytes = header_json.as_bytes();
        let aligned_len = (header_bytes.len() + 7) & !7;
        let padding = aligned_len - header_bytes.len();

        // Write file
        let file = File::create(path.as_ref()).map_err(|e| {
            WhisperError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to create file: {}", e),
            ))
        })?;
        let mut writer = BufWriter::new(file);

        // Write header length (u64 LE)
        writer
            .write_all(&(aligned_len as u64).to_le_bytes())
            .map_err(WhisperError::Io)?;

        // Write header JSON
        writer.write_all(header_bytes).map_err(WhisperError::Io)?;

        // Write padding (spaces per SafeTensors spec)
        writer
            .write_all(&vec![b' '; padding])
            .map_err(WhisperError::Io)?;

        // Write tensor data (F32 LE)
        for (_name, tensor) in tensors {
            for &value in &tensor.data {
                writer
                    .write_all(&value.to_le_bytes())
                    .map_err(WhisperError::Io)?;
            }
        }

        writer.flush().map_err(WhisperError::Io)?;

        Ok(())
    }
}

/// Escape special characters for JSON string values.
fn escape_json(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result
}

/// Statistics from export operation.
#[derive(Debug, Clone)]
pub struct ExportStats {
    /// Number of tensors exported
    pub tensor_count: usize,
    /// Total bytes written (excluding header)
    pub total_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_tensor_data_validation() {
        let valid = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(valid.validate().is_ok());

        let invalid = TensorData::new(vec![1.0, 2.0], vec![2, 2]);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_safetensors_export_basic() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_export_basic.safetensors");

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "weight1".to_string(),
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        );
        tensors.insert(
            "weight2".to_string(),
            TensorData::new(vec![5.0, 6.0, 7.0], vec![3]),
        );

        // Export
        SafeTensorsExporter::save(&path, &tensors).expect("Export should succeed");

        // Verify file exists and has content
        let metadata = fs::metadata(&path).expect("File should exist");
        assert!(metadata.len() > 0);

        // Read back and verify header structure
        let data = fs::read(&path).expect("Should read file");

        // First 8 bytes are header length
        let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        assert!(header_len > 0);

        // Header should contain tensor names
        let header_str = std::str::from_utf8(&data[8..8 + header_len])
            .expect("Header should be UTF-8")
            .trim();
        assert!(header_str.contains("weight1"));
        assert!(header_str.contains("weight2"));
        assert!(header_str.contains("\"dtype\":\"F32\""));

        // Cleanup
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_header_alignment() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_alignment.safetensors");

        let mut tensors = BTreeMap::new();
        tensors.insert("a".to_string(), TensorData::new(vec![1.0], vec![1]));

        SafeTensorsExporter::save(&path, &tensors).expect("Export should succeed");

        let data = fs::read(&path).expect("Should read file");
        let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;

        // Header length should be 8-byte aligned
        assert_eq!(header_len % 8, 0, "Header length should be 8-byte aligned");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_deterministic_output() {
        let temp_dir = std::env::temp_dir();
        let path1 = temp_dir.join("test_det1.safetensors");
        let path2 = temp_dir.join("test_det2.safetensors");

        let mut tensors = BTreeMap::new();
        tensors.insert("z_last".to_string(), TensorData::new(vec![3.0], vec![1]));
        tensors.insert("a_first".to_string(), TensorData::new(vec![1.0], vec![1]));
        tensors.insert("m_middle".to_string(), TensorData::new(vec![2.0], vec![1]));

        SafeTensorsExporter::save(&path1, &tensors).expect("Export 1 should succeed");
        SafeTensorsExporter::save(&path2, &tensors).expect("Export 2 should succeed");

        let data1 = fs::read(&path1).expect("Should read file 1");
        let data2 = fs::read(&path2).expect("Should read file 2");

        assert_eq!(data1, data2, "Exports should be deterministic");

        let _ = fs::remove_file(&path1);
        let _ = fs::remove_file(&path2);
    }

    #[test]
    fn test_with_metadata() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_with_meta.safetensors");

        let mut tensors = BTreeMap::new();
        tensors.insert("w".to_string(), TensorData::new(vec![1.0], vec![1]));

        let mut meta = BTreeMap::new();
        meta.insert("format".to_string(), "whisper.apr".to_string());
        meta.insert("version".to_string(), "0.2.0".to_string());

        SafeTensorsExporter::save_with_metadata(&path, &tensors, Some(meta))
            .expect("Export should succeed");

        let data = fs::read(&path).expect("Should read file");
        let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        let header_str = std::str::from_utf8(&data[8..8 + header_len])
            .expect("Header should be UTF-8")
            .trim();

        assert!(header_str.contains("__metadata__"));
        assert!(header_str.contains("whisper.apr"));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_escape_json() {
        assert_eq!(escape_json("hello"), "hello");
        assert_eq!(escape_json("he\"llo"), "he\\\"llo");
        assert_eq!(escape_json("he\\llo"), "he\\\\llo");
        assert_eq!(escape_json("he\nllo"), "he\\nllo");
    }
}
