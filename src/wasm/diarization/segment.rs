//! WASM speaker segment

use wasm_bindgen::prelude::*;

use crate::diarization::segmentation::SpeakerSegment;

/// WASM-friendly speaker segment
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct SpeakerSegmentWasm {
    pub(crate) speaker_id: usize,
    pub(crate) start: f32,
    pub(crate) end: f32,
    pub(crate) confidence: f32,
}

#[wasm_bindgen]
impl SpeakerSegmentWasm {
    /// Get speaker ID (0-indexed)
    #[wasm_bindgen(getter, js_name = speakerId)]
    pub fn speaker_id(&self) -> usize {
        self.speaker_id
    }

    /// Get start time in seconds
    #[wasm_bindgen(getter)]
    pub fn start(&self) -> f32 {
        self.start
    }

    /// Get end time in seconds
    #[wasm_bindgen(getter)]
    pub fn end(&self) -> f32 {
        self.end
    }

    /// Get duration in seconds
    #[wasm_bindgen(getter)]
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }

    /// Get confidence score (0.0 - 1.0)
    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Get speaker label as string (e.g., "SPEAKER_0")
    #[wasm_bindgen(getter, js_name = speakerLabel)]
    pub fn speaker_label(&self) -> String {
        format!("SPEAKER_{}", self.speaker_id)
    }
}

impl From<SpeakerSegment> for SpeakerSegmentWasm {
    fn from(seg: SpeakerSegment) -> Self {
        Self {
            speaker_id: seg.speaker_id(),
            start: seg.start(),
            end: seg.end(),
            confidence: seg.confidence(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speaker_segment_wasm() {
        let segment = SpeakerSegmentWasm {
            speaker_id: 1,
            start: 0.5,
            end: 2.0,
            confidence: 0.85,
        };

        assert_eq!(segment.speaker_id(), 1);
        assert!((segment.start() - 0.5).abs() < f32::EPSILON);
        assert!((segment.end() - 2.0).abs() < f32::EPSILON);
        assert!((segment.duration() - 1.5).abs() < f32::EPSILON);
        assert!((segment.confidence() - 0.85).abs() < f32::EPSILON);
        assert_eq!(segment.speaker_label(), "SPEAKER_1");
    }
}
