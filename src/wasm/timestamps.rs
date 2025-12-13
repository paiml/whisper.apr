//! WASM bindings for word-level timestamps (WAPR-163)
//!
//! Provides JavaScript-friendly API for word-level timestamp extraction.
//!
//! # Usage
//!
//! ```javascript
//! import { WordTimestampExtractorWasm, AlignmentConfigWasm } from 'whisper-apr';
//!
//! // Create extractor with default config
//! const extractor = new WordTimestampExtractorWasm();
//!
//! // Or with custom config
//! const config = new AlignmentConfigWasm();
//! config.setMinAttention(0.05);
//! const extractor = WordTimestampExtractorWasm.withConfig(config);
//!
//! // Extract word timestamps from cross-attention
//! const result = extractor.extractWords(attentionWeights, tokenIds, tokenTexts, numFrames);
//!
//! for (let i = 0; i < result.wordCount; i++) {
//!     const word = result.getWord(i);
//!     console.log(`${word.text}: ${word.start}s - ${word.end}s (conf: ${word.confidence})`);
//! }
//! ```

use wasm_bindgen::prelude::*;

use crate::timestamps::{
    alignment::{AlignmentConfig, CrossAttentionAlignment, TokenAlignment, WordAlignment},
    boundaries::{BoundaryConfig, BoundaryDetector, WordBoundary},
    interpolation::{InterpolationConfig, InterpolationMethod, TimestampInterpolator, TokenTimestamp},
    WordTimestampResult, WordWithTimestamp,
};

/// WASM-friendly alignment configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct AlignmentConfigWasm {
    layers: Vec<usize>,
    min_attention: f32,
    temperature: f32,
    use_median: bool,
}

#[wasm_bindgen]
impl AlignmentConfigWasm {
    /// Create default alignment config
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            layers: vec![0, 1, 2, 3, 4, 5],
            min_attention: 0.1,
            temperature: 1.0,
            use_median: false,
        }
    }

    /// Create config optimized for accuracy
    #[wasm_bindgen(js_name = forAccuracy)]
    pub fn for_accuracy() -> Self {
        Self {
            layers: vec![2, 3, 4, 5],
            min_attention: 0.05,
            temperature: 0.5,
            use_median: true,
        }
    }

    /// Create config optimized for speed
    #[wasm_bindgen(js_name = forSpeed)]
    pub fn for_speed() -> Self {
        Self {
            layers: vec![3, 4],
            min_attention: 0.15,
            temperature: 1.0,
            use_median: false,
        }
    }

    /// Set layers to use for alignment
    #[wasm_bindgen(js_name = setLayers)]
    pub fn set_layers(&mut self, layers: Vec<usize>) {
        self.layers = layers;
    }

    /// Set minimum attention threshold
    #[wasm_bindgen(js_name = setMinAttention)]
    pub fn set_min_attention(&mut self, threshold: f32) {
        self.min_attention = threshold;
    }

    /// Set temperature
    #[wasm_bindgen(js_name = setTemperature)]
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }

    /// Set whether to use median averaging
    #[wasm_bindgen(js_name = setUseMedian)]
    pub fn set_use_median(&mut self, use_median: bool) {
        self.use_median = use_median;
    }

    /// Get min attention
    #[wasm_bindgen(getter, js_name = minAttention)]
    pub fn min_attention(&self) -> f32 {
        self.min_attention
    }
}

impl Default for AlignmentConfigWasm {
    fn default() -> Self {
        Self::new()
    }
}

impl From<AlignmentConfigWasm> for AlignmentConfig {
    fn from(wasm: AlignmentConfigWasm) -> Self {
        AlignmentConfig {
            layers: wasm.layers,
            heads: None,
            min_attention: wasm.min_attention,
            temperature: wasm.temperature,
            use_median: wasm.use_median,
        }
    }
}

/// WASM-friendly word with timestamp
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WordWithTimestampWasm {
    word: String,
    start: f32,
    end: f32,
    confidence: f32,
}

#[wasm_bindgen]
impl WordWithTimestampWasm {
    /// Get word text
    #[wasm_bindgen(getter)]
    pub fn word(&self) -> String {
        self.word.clone()
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

    /// Check if high confidence
    #[wasm_bindgen(getter, js_name = isHighConfidence)]
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.8
    }
}

impl From<WordWithTimestamp> for WordWithTimestampWasm {
    fn from(word: WordWithTimestamp) -> Self {
        Self {
            word: word.word,
            start: word.start,
            end: word.end,
            confidence: word.confidence,
        }
    }
}

impl From<WordAlignment> for WordWithTimestampWasm {
    fn from(alignment: WordAlignment) -> Self {
        Self {
            word: alignment.word,
            start: alignment.start_time,
            end: alignment.end_time,
            confidence: alignment.confidence,
        }
    }
}

/// WASM-friendly word timestamp result
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WordTimestampResultWasm {
    words: Vec<WordWithTimestampWasm>,
    segment_start: f32,
    segment_end: f32,
    alignment_confidence: f32,
}

#[wasm_bindgen]
impl WordTimestampResultWasm {
    /// Get number of words
    #[wasm_bindgen(getter, js_name = wordCount)]
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Get segment start time
    #[wasm_bindgen(getter, js_name = segmentStart)]
    pub fn segment_start(&self) -> f32 {
        self.segment_start
    }

    /// Get segment end time
    #[wasm_bindgen(getter, js_name = segmentEnd)]
    pub fn segment_end(&self) -> f32 {
        self.segment_end
    }

    /// Get overall alignment confidence
    #[wasm_bindgen(getter, js_name = alignmentConfidence)]
    pub fn alignment_confidence(&self) -> f32 {
        self.alignment_confidence
    }

    /// Get word by index
    #[wasm_bindgen(js_name = getWord)]
    pub fn get_word(&self, index: usize) -> Option<WordWithTimestampWasm> {
        self.words.get(index).cloned()
    }

    /// Get all word texts
    #[wasm_bindgen(js_name = wordTexts)]
    pub fn word_texts(&self) -> Vec<String> {
        self.words.iter().map(|w| w.word.clone()).collect()
    }

    /// Get all word start times
    #[wasm_bindgen(js_name = wordStarts)]
    pub fn word_starts(&self) -> Vec<f32> {
        self.words.iter().map(|w| w.start).collect()
    }

    /// Get all word end times
    #[wasm_bindgen(js_name = wordEnds)]
    pub fn word_ends(&self) -> Vec<f32> {
        self.words.iter().map(|w| w.end).collect()
    }

    /// Get all confidence scores
    #[wasm_bindgen(js_name = wordConfidences)]
    pub fn word_confidences(&self) -> Vec<f32> {
        self.words.iter().map(|w| w.confidence).collect()
    }

    /// Check if result is high quality
    #[wasm_bindgen(getter, js_name = isHighQuality)]
    pub fn is_high_quality(&self) -> bool {
        self.alignment_confidence >= 0.7
    }

    /// Export to JSON string
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> String {
        let words_json: Vec<String> = self
            .words
            .iter()
            .map(|w| {
                format!(
                    r#"{{"word":"{}","start":{},"end":{},"confidence":{}}}"#,
                    w.word.replace('"', "\\\""),
                    w.start,
                    w.end,
                    w.confidence
                )
            })
            .collect();

        format!(
            r#"{{"segment_start":{},"segment_end":{},"alignment_confidence":{},"words":[{}]}}"#,
            self.segment_start,
            self.segment_end,
            self.alignment_confidence,
            words_json.join(",")
        )
    }
}

impl From<WordTimestampResult> for WordTimestampResultWasm {
    fn from(result: WordTimestampResult) -> Self {
        Self {
            words: result.words.into_iter().map(|w| w.into()).collect(),
            segment_start: result.segment_start,
            segment_end: result.segment_end,
            alignment_confidence: result.alignment_confidence,
        }
    }
}

/// WASM-friendly token timestamp
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct TokenTimestampWasm {
    index: usize,
    text: String,
    start: f32,
    end: f32,
    interpolated: bool,
    confidence: f32,
}

#[wasm_bindgen]
impl TokenTimestampWasm {
    /// Get token index
    #[wasm_bindgen(getter)]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get token text
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }

    /// Get start time
    #[wasm_bindgen(getter)]
    pub fn start(&self) -> f32 {
        self.start
    }

    /// Get end time
    #[wasm_bindgen(getter)]
    pub fn end(&self) -> f32 {
        self.end
    }

    /// Get duration
    #[wasm_bindgen(getter)]
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }

    /// Check if interpolated
    #[wasm_bindgen(getter)]
    pub fn interpolated(&self) -> bool {
        self.interpolated
    }

    /// Get confidence
    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        self.confidence
    }
}

impl From<TokenTimestamp> for TokenTimestampWasm {
    fn from(ts: TokenTimestamp) -> Self {
        Self {
            index: ts.index,
            text: ts.text,
            start: ts.start,
            end: ts.end,
            interpolated: ts.interpolated,
            confidence: ts.confidence,
        }
    }
}

/// WASM-friendly word boundary
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WordBoundaryWasm {
    start: f32,
    end: f32,
    start_confidence: f32,
    end_confidence: f32,
    audio_refined: bool,
}

#[wasm_bindgen]
impl WordBoundaryWasm {
    /// Get start time
    #[wasm_bindgen(getter)]
    pub fn start(&self) -> f32 {
        self.start
    }

    /// Get end time
    #[wasm_bindgen(getter)]
    pub fn end(&self) -> f32 {
        self.end
    }

    /// Get duration
    #[wasm_bindgen(getter)]
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }

    /// Get overall confidence
    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        (self.start_confidence + self.end_confidence) / 2.0
    }

    /// Check if audio-refined
    #[wasm_bindgen(getter, js_name = audioRefined)]
    pub fn audio_refined(&self) -> bool {
        self.audio_refined
    }
}

impl From<WordBoundary> for WordBoundaryWasm {
    fn from(boundary: WordBoundary) -> Self {
        Self {
            start: boundary.start,
            end: boundary.end,
            start_confidence: boundary.start_confidence,
            end_confidence: boundary.end_confidence,
            audio_refined: boundary.audio_refined,
        }
    }
}

/// WASM bindings for word timestamp extraction
#[wasm_bindgen]
pub struct WordTimestampExtractorWasm {
    alignment: CrossAttentionAlignment,
    interpolator: TimestampInterpolator,
    boundary_detector: BoundaryDetector,
}

#[wasm_bindgen]
impl WordTimestampExtractorWasm {
    /// Create new extractor with default config
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            alignment: CrossAttentionAlignment::default(),
            interpolator: TimestampInterpolator::default(),
            boundary_detector: BoundaryDetector::default(),
        }
    }

    /// Create extractor with custom config
    #[wasm_bindgen(js_name = withConfig)]
    pub fn with_config(config: AlignmentConfigWasm) -> Self {
        Self {
            alignment: CrossAttentionAlignment::new(config.into()),
            interpolator: TimestampInterpolator::default(),
            boundary_detector: BoundaryDetector::default(),
        }
    }

    /// Create extractor optimized for accuracy
    #[wasm_bindgen(js_name = forAccuracy)]
    pub fn for_accuracy() -> Self {
        Self {
            alignment: CrossAttentionAlignment::new(AlignmentConfig::for_accuracy()),
            interpolator: TimestampInterpolator::default(),
            boundary_detector: BoundaryDetector::new(BoundaryConfig::precise()),
        }
    }

    /// Create extractor optimized for speed
    #[wasm_bindgen(js_name = forSpeed)]
    pub fn for_speed() -> Self {
        Self {
            alignment: CrossAttentionAlignment::new(AlignmentConfig::for_speed()),
            interpolator: TimestampInterpolator::new(InterpolationConfig::linear()),
            boundary_detector: BoundaryDetector::new(BoundaryConfig::fast()),
        }
    }

    /// Interpolate word token timestamps
    ///
    /// # Arguments
    /// * `word_start` - Word start time in seconds
    /// * `word_end` - Word end time in seconds
    /// * `tokens` - Token texts within the word
    /// * `start_index` - Starting token index
    #[wasm_bindgen(js_name = interpolateWordTokens)]
    pub fn interpolate_word_tokens(
        &self,
        word_start: f32,
        word_end: f32,
        tokens: Vec<String>,
        start_index: usize,
    ) -> Result<Vec<TokenTimestampWasm>, JsValue> {
        self.interpolator
            .interpolate_word_tokens(word_start, word_end, &tokens, start_index)
            .map(|ts| ts.into_iter().map(|t| t.into()).collect())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for WordTimestampExtractorWasm {
    fn default() -> Self {
        Self::new()
    }
}

/// WASM-friendly timestamp interpolator
#[wasm_bindgen]
pub struct TimestampInterpolatorWasm {
    inner: TimestampInterpolator,
}

#[wasm_bindgen]
impl TimestampInterpolatorWasm {
    /// Create new interpolator
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: TimestampInterpolator::default(),
        }
    }

    /// Create linear interpolator
    #[wasm_bindgen(js_name = linear)]
    pub fn linear() -> Self {
        Self {
            inner: TimestampInterpolator::new(InterpolationConfig::linear()),
        }
    }

    /// Create character-proportional interpolator
    #[wasm_bindgen(js_name = characterProportional)]
    pub fn character_proportional() -> Self {
        Self {
            inner: TimestampInterpolator::new(InterpolationConfig::character_proportional()),
        }
    }

    /// Interpolate timestamps for tokens within a word
    #[wasm_bindgen]
    pub fn interpolate(
        &self,
        word_start: f32,
        word_end: f32,
        tokens: Vec<String>,
        start_index: usize,
    ) -> Result<Vec<TokenTimestampWasm>, JsValue> {
        self.inner
            .interpolate_word_tokens(word_start, word_end, &tokens, start_index)
            .map(|ts| ts.into_iter().map(|t| t.into()).collect())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for TimestampInterpolatorWasm {
    fn default() -> Self {
        Self::new()
    }
}

/// Get recommended word timestamp config for use case
#[wasm_bindgen(js_name = getWordTimestampRecommendation)]
pub fn get_word_timestamp_recommendation(use_case: &str) -> String {
    match use_case.to_lowercase().as_str() {
        "karaoke" | "lyrics" | "subtitles" => {
            "Use forAccuracy() for precise word-level sync in karaoke/subtitles.".to_string()
        }
        "search" | "index" | "indexing" => {
            "Use default config for searchable timestamps in audio indexing.".to_string()
        }
        "realtime" | "live" | "streaming" => {
            "Use forSpeed() for real-time applications with lower latency.".to_string()
        }
        "transcription" | "batch" => {
            "Use default config with smoothing for batch transcription.".to_string()
        }
        _ => "Unknown use case. Available: karaoke, search, realtime, transcription.".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_config_wasm_default() {
        let config = AlignmentConfigWasm::new();
        assert_eq!(config.layers.len(), 6);
        assert!((config.min_attention - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_alignment_config_wasm_for_accuracy() {
        let config = AlignmentConfigWasm::for_accuracy();
        assert!(config.use_median);
        assert!((config.min_attention - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn test_alignment_config_wasm_for_speed() {
        let config = AlignmentConfigWasm::for_speed();
        assert_eq!(config.layers.len(), 2);
    }

    #[test]
    fn test_alignment_config_wasm_setters() {
        let mut config = AlignmentConfigWasm::new();
        config.set_min_attention(0.2);
        assert!((config.min_attention() - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_word_with_timestamp_wasm() {
        let word = WordWithTimestamp::new("hello".to_string(), 0.0, 1.0, 0.9);
        let wasm: WordWithTimestampWasm = word.into();

        assert_eq!(wasm.word(), "hello");
        assert!((wasm.start() - 0.0).abs() < f32::EPSILON);
        assert!((wasm.end() - 1.0).abs() < f32::EPSILON);
        assert!((wasm.duration() - 1.0).abs() < f32::EPSILON);
        assert!((wasm.confidence() - 0.9).abs() < f32::EPSILON);
        assert!(wasm.is_high_confidence());
    }

    #[test]
    fn test_word_timestamp_result_wasm() {
        let words = vec![
            WordWithTimestamp::new("hello".to_string(), 0.0, 0.5, 0.9),
            WordWithTimestamp::new("world".to_string(), 0.5, 1.0, 0.8),
        ];
        let result = WordTimestampResult::new(words, 0.0, 1.0);
        let wasm: WordTimestampResultWasm = result.into();

        assert_eq!(wasm.word_count(), 2);
        assert!((wasm.segment_start() - 0.0).abs() < f32::EPSILON);
        assert!((wasm.segment_end() - 1.0).abs() < f32::EPSILON);
        assert!(wasm.is_high_quality());
    }

    #[test]
    fn test_word_timestamp_result_wasm_get_word() {
        let words = vec![WordWithTimestamp::new("test".to_string(), 0.0, 1.0, 0.85)];
        let result = WordTimestampResult::new(words, 0.0, 1.0);
        let wasm: WordTimestampResultWasm = result.into();

        let word = wasm.get_word(0).expect("should have word");
        assert_eq!(word.word(), "test");
        assert!(wasm.get_word(1).is_none());
    }

    #[test]
    fn test_word_timestamp_result_wasm_arrays() {
        let words = vec![
            WordWithTimestamp::new("a".to_string(), 0.0, 0.5, 0.9),
            WordWithTimestamp::new("b".to_string(), 0.5, 1.0, 0.8),
        ];
        let result = WordTimestampResult::new(words, 0.0, 1.0);
        let wasm: WordTimestampResultWasm = result.into();

        assert_eq!(wasm.word_texts(), vec!["a", "b"]);
        assert_eq!(wasm.word_starts().len(), 2);
        assert_eq!(wasm.word_ends().len(), 2);
        assert_eq!(wasm.word_confidences().len(), 2);
    }

    #[test]
    fn test_word_timestamp_result_wasm_to_json() {
        let words = vec![WordWithTimestamp::new("hello".to_string(), 0.0, 0.5, 0.9)];
        let result = WordTimestampResult::new(words, 0.0, 1.0);
        let wasm: WordTimestampResultWasm = result.into();

        let json = wasm.to_json();
        assert!(json.contains("\"word\":\"hello\""));
        assert!(json.contains("\"start\":0"));
    }

    #[test]
    fn test_token_timestamp_wasm() {
        let ts = TokenTimestamp::interpolated(0, "hel".to_string(), 0.0, 0.3, 0.6);
        let wasm: TokenTimestampWasm = ts.into();

        assert_eq!(wasm.index(), 0);
        assert_eq!(wasm.text(), "hel");
        assert!(wasm.interpolated());
        assert!((wasm.confidence() - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_word_boundary_wasm() {
        let boundary = WordBoundary::new(0.0, 1.0)
            .with_confidence(0.8, 0.9)
            .with_audio_refined(true);
        let wasm: WordBoundaryWasm = boundary.into();

        assert!((wasm.start() - 0.0).abs() < f32::EPSILON);
        assert!((wasm.end() - 1.0).abs() < f32::EPSILON);
        assert!((wasm.confidence() - 0.85).abs() < f32::EPSILON);
        assert!(wasm.audio_refined());
    }

    #[test]
    fn test_word_timestamp_extractor_wasm_new() {
        let _extractor = WordTimestampExtractorWasm::new();
    }

    #[test]
    fn test_word_timestamp_extractor_wasm_with_config() {
        let config = AlignmentConfigWasm::for_accuracy();
        let _extractor = WordTimestampExtractorWasm::with_config(config);
    }

    #[test]
    fn test_word_timestamp_extractor_wasm_for_accuracy() {
        let _extractor = WordTimestampExtractorWasm::for_accuracy();
    }

    #[test]
    fn test_word_timestamp_extractor_wasm_for_speed() {
        let _extractor = WordTimestampExtractorWasm::for_speed();
    }

    #[test]
    fn test_timestamp_interpolator_wasm_new() {
        let _interpolator = TimestampInterpolatorWasm::new();
    }

    #[test]
    fn test_timestamp_interpolator_wasm_linear() {
        let _interpolator = TimestampInterpolatorWasm::linear();
    }

    #[test]
    fn test_timestamp_interpolator_wasm_character_proportional() {
        let _interpolator = TimestampInterpolatorWasm::character_proportional();
    }

    #[test]
    fn test_get_word_timestamp_recommendation() {
        let karaoke = get_word_timestamp_recommendation("karaoke");
        assert!(karaoke.contains("forAccuracy"));

        let realtime = get_word_timestamp_recommendation("realtime");
        assert!(realtime.contains("forSpeed"));
    }
}
