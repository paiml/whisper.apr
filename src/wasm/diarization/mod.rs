//! WASM bindings for speaker diarization (WAPR-153)
//!
//! Provides JavaScript-friendly API for speaker diarization via wasm-bindgen.
//!
//! # Usage
//!
//! ```javascript
//! import { DiarizerWasm, DiarizationConfigWasm } from 'whisper-apr';
//!
//! // Create diarizer with default config
//! const diarizer = new DiarizerWasm();
//!
//! // Or with custom config
//! const config = new DiarizationConfigWasm();
//! config.setMaxSpeakers(4);
//! config.setMinSegmentDuration(0.5);
//! const diarizer = DiarizerWasm.withConfig(config);
//!
//! // Process audio and get speaker-labeled segments
//! const result = diarizer.process(audioFloat32Array, 16000);
//! console.log(`Found ${result.speakerCount} speakers`);
//!
//! for (let i = 0; i < result.segmentCount; i++) {
//!     const segment = result.getSegment(i);
//!     console.log(`Speaker ${segment.speakerId}: ${segment.start}s - ${segment.end}s`);
//! }
//! ```

mod config;
mod detector;
mod diarizer;
mod embedding;
mod extractor;
mod result;
mod segment;

pub use config::DiarizationConfigWasm;
pub use detector::TurnDetectorWasm;
pub use diarizer::DiarizerWasm;
pub use embedding::SpeakerEmbeddingWasm;
pub use extractor::EmbeddingExtractorWasm;
pub use result::DiarizationResultWasm;
pub use segment::SpeakerSegmentWasm;

use wasm_bindgen::prelude::*;

/// Get recommended diarization config for use case
#[wasm_bindgen(js_name = getDiarizationRecommendation)]
pub fn get_diarization_recommendation(use_case: &str) -> String {
    match use_case.to_lowercase().as_str() {
        "meeting" | "conference" | "interview" => {
            "Use forAccuracy() with max 8 speakers for meetings and interviews.".to_string()
        }
        "podcast" | "dialogue" | "conversation" => {
            "Use default config with max 4 speakers for podcasts and dialogues.".to_string()
        }
        "call" | "phone" | "telephony" => {
            "Use default config with max 2 speakers for phone calls.".to_string()
        }
        "realtime" | "live" | "streaming" => {
            "Use forRealtime() for live streaming with reduced latency.".to_string()
        }
        _ => "Unknown use case. Available: meeting, podcast, call, realtime.".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_diarization_recommendation_meeting() {
        let rec = get_diarization_recommendation("meeting");
        assert!(rec.contains("forAccuracy"));
    }

    #[test]
    fn test_get_diarization_recommendation_podcast() {
        let rec = get_diarization_recommendation("podcast");
        assert!(rec.contains("default"));
    }

    #[test]
    fn test_get_diarization_recommendation_call() {
        let rec = get_diarization_recommendation("call");
        assert!(rec.contains("2 speakers"));
    }

    #[test]
    fn test_get_diarization_recommendation_realtime() {
        let rec = get_diarization_recommendation("realtime");
        assert!(rec.contains("forRealtime"));
    }

    #[test]
    fn test_get_diarization_recommendation_unknown() {
        let rec = get_diarization_recommendation("unknown");
        assert!(rec.contains("Unknown"));
    }

    #[test]
    fn test_get_diarization_recommendation_case_insensitive() {
        let rec = get_diarization_recommendation("MEETING");
        assert!(rec.contains("forAccuracy"));
    }
}
