//! VAD data models and types
//!
//! Core data structures for voice activity detection.

/// Voice activity detection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadState {
    /// No speech detected
    Silence,
    /// Speech is being detected
    Speech,
    /// Transitioning from silence to speech
    SpeechStart,
    /// Transitioning from speech to silence
    SpeechEnd,
}

/// Voice activity event for streaming
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    /// Continue in current state (no change)
    Continue,
    /// Speech started
    SpeechStart,
    /// Speech ended
    SpeechEnd,
}

/// Detected speech segment
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Average energy during segment
    pub energy: f32,
}

impl SpeechSegment {
    /// Duration of the segment in seconds
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_state_enum() {
        let state = VadState::Silence;
        assert_eq!(state, VadState::Silence);
        assert_ne!(state, VadState::Speech);
    }

    #[test]
    fn test_vad_state_clone() {
        let state = VadState::Speech;
        let cloned = state;
        assert_eq!(state, cloned);
    }

    #[test]
    fn test_vad_event_enum() {
        let event = VadEvent::SpeechStart;
        assert_eq!(event, VadEvent::SpeechStart);
        assert_ne!(event, VadEvent::SpeechEnd);
    }

    #[test]
    fn test_speech_segment_duration() {
        let segment = SpeechSegment {
            start: 1.0,
            end: 3.5,
            energy: 0.5,
        };
        assert!((segment.duration() - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_speech_segment_clone() {
        let segment = SpeechSegment {
            start: 0.0,
            end: 1.0,
            energy: 0.1,
        };
        let cloned = segment.clone();
        assert!((segment.start - cloned.start).abs() < f32::EPSILON);
        assert!((segment.end - cloned.end).abs() < f32::EPSILON);
    }

    #[test]
    fn test_speech_segment_debug() {
        let segment = SpeechSegment {
            start: 0.0,
            end: 1.0,
            energy: 0.5,
        };
        let debug = format!("{:?}", segment);
        assert!(debug.contains("SpeechSegment"));
    }
}
