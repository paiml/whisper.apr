//! WASM diarization result

use wasm_bindgen::prelude::*;

use super::embedding::SpeakerEmbeddingWasm;
use super::segment::SpeakerSegmentWasm;
use crate::diarization::DiarizationResult;

/// WASM-friendly diarization result
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DiarizationResultWasm {
    pub(crate) segments: Vec<SpeakerSegmentWasm>,
    pub(crate) speaker_embeddings: Vec<SpeakerEmbeddingWasm>,
    pub(crate) speaker_count: usize,
    pub(crate) total_duration: f32,
}

#[wasm_bindgen]
impl DiarizationResultWasm {
    /// Get number of speakers detected
    #[wasm_bindgen(getter, js_name = speakerCount)]
    pub fn speaker_count(&self) -> usize {
        self.speaker_count
    }

    /// Get number of segments
    #[wasm_bindgen(getter, js_name = segmentCount)]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Get total audio duration in seconds
    #[wasm_bindgen(getter, js_name = totalDuration)]
    pub fn total_duration(&self) -> f32 {
        self.total_duration
    }

    /// Get a segment by index
    #[wasm_bindgen(js_name = getSegment)]
    pub fn get_segment(&self, index: usize) -> Option<SpeakerSegmentWasm> {
        self.segments.get(index).cloned()
    }

    /// Get all segments for a specific speaker
    #[wasm_bindgen(js_name = getSegmentsForSpeaker)]
    pub fn get_segments_for_speaker(&self, speaker_id: usize) -> Vec<SpeakerSegmentWasm> {
        self.segments
            .iter()
            .filter(|s| s.speaker_id == speaker_id)
            .cloned()
            .collect()
    }

    /// Get total speaking time for a speaker in seconds
    #[wasm_bindgen(js_name = getSpeakingTime)]
    pub fn get_speaking_time(&self, speaker_id: usize) -> f32 {
        self.segments
            .iter()
            .filter(|s| s.speaker_id == speaker_id)
            .map(|s| s.duration())
            .sum()
    }

    /// Get speaking percentage for a speaker (0.0 - 100.0)
    #[wasm_bindgen(js_name = getSpeakingPercentage)]
    pub fn get_speaking_percentage(&self, speaker_id: usize) -> f32 {
        if self.total_duration <= 0.0 {
            return 0.0;
        }
        (self.get_speaking_time(speaker_id) / self.total_duration) * 100.0
    }

    /// Get speaker embedding by speaker ID
    #[wasm_bindgen(js_name = getSpeakerEmbedding)]
    pub fn get_speaker_embedding(&self, speaker_id: usize) -> Option<SpeakerEmbeddingWasm> {
        self.speaker_embeddings
            .iter()
            .find(|e| e.speaker_id == speaker_id)
            .cloned()
    }

    /// Get all segment start times
    #[wasm_bindgen(js_name = segmentStarts)]
    pub fn segment_starts(&self) -> Vec<f32> {
        self.segments.iter().map(|s| s.start).collect()
    }

    /// Get all segment end times
    #[wasm_bindgen(js_name = segmentEnds)]
    pub fn segment_ends(&self) -> Vec<f32> {
        self.segments.iter().map(|s| s.end).collect()
    }

    /// Get all segment speaker IDs
    #[wasm_bindgen(js_name = segmentSpeakerIds)]
    pub fn segment_speaker_ids(&self) -> Vec<usize> {
        self.segments.iter().map(|s| s.speaker_id).collect()
    }

    /// Get number of speaker turns (segment transitions)
    #[wasm_bindgen(js_name = turnCount)]
    pub fn turn_count(&self) -> usize {
        if self.segments.is_empty() {
            return 0;
        }

        let mut turns = 0;
        let mut prev_speaker = self.segments[0].speaker_id;

        for seg in &self.segments[1..] {
            if seg.speaker_id != prev_speaker {
                turns += 1;
                prev_speaker = seg.speaker_id;
            }
        }

        turns
    }

    /// Export to JSON string
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> String {
        let segments: Vec<_> = self
            .segments
            .iter()
            .map(|s| {
                format!(
                    r#"{{"speaker_id":{},"start":{},"end":{},"confidence":{}}}"#,
                    s.speaker_id, s.start, s.end, s.confidence
                )
            })
            .collect();

        format!(
            r#"{{"speaker_count":{},"total_duration":{},"segments":[{}]}}"#,
            self.speaker_count,
            self.total_duration,
            segments.join(",")
        )
    }
}

impl From<DiarizationResult> for DiarizationResultWasm {
    fn from(result: DiarizationResult) -> Self {
        let segments: Vec<SpeakerSegmentWasm> =
            result.segments().iter().map(|s| s.clone().into()).collect();

        let speaker_embeddings: Vec<SpeakerEmbeddingWasm> = result
            .speaker_embeddings()
            .iter()
            .map(|e| e.clone().into())
            .collect();

        let total_duration = segments.iter().map(|s| s.end).fold(0.0f32, f32::max);

        Self {
            segments,
            speaker_embeddings,
            speaker_count: result.num_speakers(),
            total_duration,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_segment(speaker_id: usize, start: f32, end: f32) -> SpeakerSegmentWasm {
        SpeakerSegmentWasm {
            speaker_id,
            start,
            end,
            confidence: 1.0,
        }
    }

    #[test]
    fn test_diarization_result_wasm_empty() {
        let result = DiarizationResultWasm {
            segments: vec![],
            speaker_embeddings: vec![],
            speaker_count: 0,
            total_duration: 0.0,
        };
        assert_eq!(result.speaker_count(), 0);
        assert_eq!(result.segment_count(), 0);
        assert_eq!(result.turn_count(), 0);
    }

    #[test]
    fn test_diarization_result_wasm_speaking_time() {
        let result = DiarizationResultWasm {
            segments: vec![
                make_segment(0, 0.0, 1.0),
                make_segment(1, 1.0, 2.0),
                make_segment(0, 2.0, 4.0),
            ],
            speaker_embeddings: vec![],
            speaker_count: 2,
            total_duration: 4.0,
        };

        assert!((result.get_speaking_time(0) - 3.0).abs() < 0.01);
        assert!((result.get_speaking_time(1) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_diarization_result_wasm_speaking_percentage() {
        let result = DiarizationResultWasm {
            segments: vec![make_segment(0, 0.0, 1.0), make_segment(1, 1.0, 2.0)],
            speaker_embeddings: vec![],
            speaker_count: 2,
            total_duration: 2.0,
        };

        assert!((result.get_speaking_percentage(0) - 50.0).abs() < 0.01);
        assert!((result.get_speaking_percentage(1) - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_diarization_result_wasm_turn_count() {
        let result = DiarizationResultWasm {
            segments: vec![
                make_segment(0, 0.0, 1.0),
                make_segment(1, 1.0, 2.0),
                make_segment(0, 2.0, 3.0),
            ],
            speaker_embeddings: vec![],
            speaker_count: 2,
            total_duration: 3.0,
        };
        // Two speaker changes: 0->1, 1->0
        assert_eq!(result.turn_count(), 2);
    }

    #[test]
    fn test_diarization_result_wasm_to_json() {
        let result = DiarizationResultWasm {
            segments: vec![make_segment(0, 0.0, 1.0)],
            speaker_embeddings: vec![],
            speaker_count: 1,
            total_duration: 1.0,
        };
        let json = result.to_json();
        assert!(json.contains("speaker_count"));
        assert!(json.contains("total_duration"));
        assert!(json.contains("segments"));
    }
}
