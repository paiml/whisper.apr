//! `SharedArrayBuffer` Ring Buffer for Audio Data
//!
//! Lock-free ring buffer using `SharedArrayBuffer` and Atomics.
//! Enables zero-copy audio transfer between `AudioWorklet` and Worker.

use wasm_bindgen::prelude::*;

// Header layout constants
const HEADER_SIZE: usize = 16; // 4 i32s: write_pos, read_pos, capacity, flags
const WRITE_POS_OFFSET: usize = 0;
const READ_POS_OFFSET: usize = 1;
const CAPACITY_OFFSET: usize = 2;
const FLAGS_OFFSET: usize = 3;

// Flag bits
const FLAG_DONE: i32 = 1; // Producer is done writing

/// Shared ring buffer backed by `SharedArrayBuffer`
#[wasm_bindgen]
#[derive(Clone)]
pub struct SharedRingBuffer {
    buffer: js_sys::SharedArrayBuffer,
    header: js_sys::Int32Array,
    data: js_sys::Float32Array,
    capacity: usize,
}

#[wasm_bindgen]
impl SharedRingBuffer {
    /// Create a new ring buffer with given capacity (in samples)
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize) -> Result<SharedRingBuffer, JsValue> {
        // Allocate SharedArrayBuffer for header + data
        let header_bytes = HEADER_SIZE * 4; // 4 bytes per i32
        let data_bytes = capacity * 4; // 4 bytes per f32
        let total_bytes = header_bytes + data_bytes;

        let buffer = js_sys::SharedArrayBuffer::new(total_bytes as u32);
        let header = js_sys::Int32Array::new_with_byte_offset_and_length(
            &buffer,
            0,
            HEADER_SIZE as u32,
        );
        let data = js_sys::Float32Array::new_with_byte_offset_and_length(
            &buffer,
            header_bytes as u32,
            capacity as u32,
        );

        // Initialize header
        js_sys::Atomics::store(&header, WRITE_POS_OFFSET as u32, 0)?;
        js_sys::Atomics::store(&header, READ_POS_OFFSET as u32, 0)?;
        js_sys::Atomics::store(&header, CAPACITY_OFFSET as u32, capacity as i32)?;
        js_sys::Atomics::store(&header, FLAGS_OFFSET as u32, 0)?;

        Ok(SharedRingBuffer {
            buffer,
            header,
            data,
            capacity,
        })
    }

    /// Create from existing `SharedArrayBuffer`
    #[wasm_bindgen(js_name = fromBuffer)]
    pub fn from_buffer(buffer: js_sys::SharedArrayBuffer) -> Result<SharedRingBuffer, JsValue> {
        let header_bytes = HEADER_SIZE * 4;
        let header = js_sys::Int32Array::new_with_byte_offset_and_length(&buffer, 0, HEADER_SIZE as u32);

        let capacity = js_sys::Atomics::load(&header, CAPACITY_OFFSET as u32)
            .map_err(|e| JsValue::from_str(&format!("Failed to read capacity: {e:?}")))?
            as usize;

        let data = js_sys::Float32Array::new_with_byte_offset_and_length(
            &buffer,
            header_bytes as u32,
            capacity as u32,
        );

        Ok(SharedRingBuffer {
            buffer,
            header,
            data,
            capacity,
        })
    }

    /// Get the underlying `SharedArrayBuffer` for transfer
    #[wasm_bindgen(getter)]
    #[must_use] 
    pub fn buffer(&self) -> js_sys::SharedArrayBuffer {
        self.buffer.clone()
    }

    /// Write samples to buffer (producer side)
    pub fn write(&self, samples: &[f32]) -> Result<usize, JsValue> {
        let write_pos = js_sys::Atomics::load(&self.header, WRITE_POS_OFFSET as u32)? as usize;
        let read_pos = js_sys::Atomics::load(&self.header, READ_POS_OFFSET as u32)? as usize;

        // Calculate available space
        let available = if write_pos >= read_pos {
            self.capacity - (write_pos - read_pos) - 1
        } else {
            read_pos - write_pos - 1
        };

        let to_write = samples.len().min(available);
        if to_write == 0 {
            return Ok(0);
        }

        // Write samples with wraparound
        for (i, &sample) in samples.iter().take(to_write).enumerate() {
            let pos = (write_pos + i) % self.capacity;
            self.data.set_index(pos as u32, sample);
        }

        // Update write position atomically
        let new_write_pos = ((write_pos + to_write) % self.capacity) as i32;
        js_sys::Atomics::store(&self.header, WRITE_POS_OFFSET as u32, new_write_pos)?;

        // Notify waiting consumers
        let _ = js_sys::Atomics::notify(&self.header, WRITE_POS_OFFSET as u32)?;

        Ok(to_write)
    }

    /// Read samples from buffer (consumer side)
    pub fn read(&self, max_samples: usize) -> Result<Vec<f32>, JsValue> {
        let write_pos = js_sys::Atomics::load(&self.header, WRITE_POS_OFFSET as u32)? as usize;
        let read_pos = js_sys::Atomics::load(&self.header, READ_POS_OFFSET as u32)? as usize;

        // Calculate available samples
        let available = if write_pos >= read_pos {
            write_pos - read_pos
        } else {
            self.capacity - read_pos + write_pos
        };

        let to_read = max_samples.min(available);
        if to_read == 0 {
            return Ok(Vec::new());
        }

        // Read samples with wraparound
        let mut samples = Vec::with_capacity(to_read);
        for i in 0..to_read {
            let pos = (read_pos + i) % self.capacity;
            samples.push(self.data.get_index(pos as u32));
        }

        // Update read position atomically
        let new_read_pos = ((read_pos + to_read) % self.capacity) as i32;
        js_sys::Atomics::store(&self.header, READ_POS_OFFSET as u32, new_read_pos)?;

        Ok(samples)
    }

    /// Get number of available samples to read
    pub fn available(&self) -> Result<usize, JsValue> {
        let write_pos = js_sys::Atomics::load(&self.header, WRITE_POS_OFFSET as u32)? as usize;
        let read_pos = js_sys::Atomics::load(&self.header, READ_POS_OFFSET as u32)? as usize;

        Ok(if write_pos >= read_pos {
            write_pos - read_pos
        } else {
            self.capacity - read_pos + write_pos
        })
    }

    /// Mark buffer as done (producer finished)
    #[wasm_bindgen(js_name = markDone)]
    pub fn mark_done(&self) -> Result<(), JsValue> {
        let current = js_sys::Atomics::load(&self.header, FLAGS_OFFSET as u32)?;
        js_sys::Atomics::store(&self.header, FLAGS_OFFSET as u32, current | FLAG_DONE)?;
        // Wake up any waiting consumers
        let _ = js_sys::Atomics::notify(&self.header, FLAGS_OFFSET as u32)?;
        Ok(())
    }

    /// Check if producer is done
    #[wasm_bindgen(js_name = isDone)]
    pub fn is_done(&self) -> Result<bool, JsValue> {
        let flags = js_sys::Atomics::load(&self.header, FLAGS_OFFSET as u32)?;
        Ok((flags & FLAG_DONE) != 0)
    }

    /// Reset the buffer
    pub fn reset(&self) -> Result<(), JsValue> {
        js_sys::Atomics::store(&self.header, WRITE_POS_OFFSET as u32, 0)?;
        js_sys::Atomics::store(&self.header, READ_POS_OFFSET as u32, 0)?;
        js_sys::Atomics::store(&self.header, FLAGS_OFFSET as u32, 0)?;
        Ok(())
    }
}
