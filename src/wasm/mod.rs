//! WASM bindings for Whisper.apr
//!
//! Provides JavaScript-friendly API via wasm-bindgen for browser deployment.
//!
//! # Usage
//!
//! ```javascript
//! import init, { WhisperAprWasm, TranscribeOptionsWasm } from 'whisper-apr';
//!
//! await init();
//! const whisper = new WhisperAprWasm('tiny');
//! const result = await whisper.transcribe(audioFloat32Array, {});
//! console.log(result.text);
//! ```
//!
//! # Web Worker Usage
//!
//! For non-blocking transcription, use a Web Worker:
//!
//! ```javascript
//! // worker.js
//! import init, { WhisperAprWasm, WorkerProgress } from 'whisper-apr';
//!
//! let whisper = null;
//!
//! self.onmessage = async (e) => {
//!   const { type, ...data } = e.data;
//!
//!   if (type === 'init') {
//!     await init();
//!     whisper = new WhisperAprWasm(data.modelType);
//!     self.postMessage({ type: 'ready' });
//!   } else if (type === 'transcribe') {
//!     const result = whisper.transcribe(data.audio, data.options);
//!     self.postMessage({ type: 'result', ...result });
//!   }
//! };
//! ```

mod capabilities;
mod worker;

pub use capabilities::{Capabilities, ExecutionMode};
pub use worker::{ProgressPhase, WorkerConfig, WorkerMessageType, WorkerProgress, WorkerState};

use wasm_bindgen::prelude::*;

use crate::detection::LanguageProbs;
use crate::model::ModelConfig;
use crate::{DecodingStrategy, Task, TranscribeOptions, TranscriptionResult, WhisperApr};

/// WASM-friendly transcription options
#[wasm_bindgen]
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptionsWasm {
    language: Option<String>,
    task: String,
    beam_size: usize,
    temperature: f32,
}

#[wasm_bindgen]
impl TranscribeOptionsWasm {
    /// Create default transcription options
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            language: None,
            task: "transcribe".to_string(),
            beam_size: 1,
            temperature: 0.0,
        }
    }

    /// Set the language code (e.g., "en", "es", "auto")
    #[wasm_bindgen(js_name = setLanguage)]
    pub fn set_language(&mut self, language: &str) {
        self.language = if language == "auto" || language.is_empty() {
            None
        } else {
            Some(language.to_string())
        };
    }

    /// Set the task ("transcribe" or "translate")
    #[wasm_bindgen(js_name = setTask)]
    pub fn set_task(&mut self, task: &str) {
        self.task = task.to_string();
    }

    /// Set beam size for beam search (1 = greedy)
    #[wasm_bindgen(js_name = setBeamSize)]
    pub fn set_beam_size(&mut self, beam_size: usize) {
        self.beam_size = beam_size;
    }

    /// Set temperature for sampling
    #[wasm_bindgen(js_name = setTemperature)]
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }
}

impl From<TranscribeOptionsWasm> for TranscribeOptions {
    fn from(opts: TranscribeOptionsWasm) -> Self {
        let task = match opts.task.as_str() {
            "translate" => Task::Translate,
            _ => Task::Transcribe,
        };

        let strategy = if opts.beam_size > 1 {
            DecodingStrategy::BeamSearch {
                beam_size: opts.beam_size,
                temperature: opts.temperature,
                patience: 1.0,
            }
        } else if opts.temperature > 0.0 {
            DecodingStrategy::Sampling {
                temperature: opts.temperature,
                top_k: None,
                top_p: None,
            }
        } else {
            DecodingStrategy::Greedy
        };

        Self {
            language: opts.language,
            task,
            strategy,
            word_timestamps: false,
        }
    }
}

/// WASM-friendly transcription segment with timestamps
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct SegmentWasm {
    text: String,
    start: f32,
    end: f32,
}

#[wasm_bindgen]
impl SegmentWasm {
    /// Get the segment text
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
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
}

/// WASM-friendly transcription result
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct TranscriptionResultWasm {
    text: String,
    language: String,
    segments: Vec<SegmentWasm>,
}

#[wasm_bindgen]
impl TranscriptionResultWasm {
    /// Get the transcribed text
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }

    /// Get the detected/specified language
    #[wasm_bindgen(getter)]
    pub fn language(&self) -> String {
        self.language.clone()
    }

    /// Get the number of segments
    #[wasm_bindgen(js_name = segmentCount)]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Get a segment by index
    #[wasm_bindgen(js_name = getSegment)]
    pub fn get_segment(&self, index: usize) -> Option<SegmentWasm> {
        self.segments.get(index).cloned()
    }

    /// Get all segment texts as a single array
    #[wasm_bindgen(js_name = segmentTexts)]
    pub fn segment_texts(&self) -> Vec<String> {
        self.segments.iter().map(|s| s.text.clone()).collect()
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
}

impl From<TranscriptionResult> for TranscriptionResultWasm {
    fn from(result: TranscriptionResult) -> Self {
        let segments = result
            .segments
            .into_iter()
            .map(|s| SegmentWasm {
                text: s.text,
                start: s.start,
                end: s.end,
            })
            .collect();

        Self {
            text: result.text,
            language: result.language,
            segments,
        }
    }
}

/// WASM-friendly language detection result
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct LanguageDetectionResult {
    languages: Vec<String>,
    probabilities: Vec<f32>,
}

#[wasm_bindgen]
impl LanguageDetectionResult {
    /// Get detected languages in order of probability
    #[wasm_bindgen(getter)]
    pub fn languages(&self) -> Vec<String> {
        self.languages.clone()
    }

    /// Get probabilities for each language
    #[wasm_bindgen(getter)]
    pub fn probabilities(&self) -> Vec<f32> {
        self.probabilities.clone()
    }

    /// Get the top detected language
    #[wasm_bindgen(js_name = topLanguage)]
    pub fn top_language(&self) -> Option<String> {
        self.languages.first().cloned()
    }

    /// Get the confidence score (top probability)
    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        self.probabilities.first().copied().unwrap_or(0.0)
    }
}

impl From<LanguageProbs> for LanguageDetectionResult {
    fn from(probs: LanguageProbs) -> Self {
        Self {
            languages: probs.languages,
            probabilities: probs.probabilities,
        }
    }
}

/// WASM bindings for WhisperApr
///
/// This provides a JavaScript-friendly API for the WhisperApr engine.
#[wasm_bindgen]
pub struct WhisperAprWasm {
    inner: WhisperApr,
}

#[wasm_bindgen]
impl WhisperAprWasm {
    /// Create a new Whisper instance with the specified model type
    ///
    /// # Arguments
    /// * `model_type` - Model type: "tiny", "base", or "small"
    #[wasm_bindgen(constructor)]
    pub fn new(model_type: &str) -> Result<Self, JsValue> {
        // Set up panic hook for better error messages
        #[cfg(feature = "wasm")]
        console_error_panic_hook::set_once();

        let config = match model_type {
            "tiny" => ModelConfig::tiny(),
            "base" => ModelConfig::base(),
            "small" => ModelConfig::small(),
            "medium" => ModelConfig::medium(),
            "large" => ModelConfig::large(),
            _ => {
                return Err(JsValue::from_str(&format!(
                    "Unknown model type: {model_type}. Use 'tiny', 'base', 'small', 'medium', or 'large'"
                )))
            }
        };

        Ok(Self {
            inner: WhisperApr::from_config(config),
        })
    }

    /// Create a tiny model
    #[wasm_bindgen(js_name = tiny)]
    pub fn tiny() -> Self {
        Self {
            inner: WhisperApr::tiny(),
        }
    }

    /// Create a base model
    #[wasm_bindgen(js_name = base)]
    pub fn base() -> Self {
        Self {
            inner: WhisperApr::base(),
        }
    }

    /// Get the model type as string
    #[wasm_bindgen(js_name = modelType)]
    pub fn model_type(&self) -> String {
        format!("{:?}", self.inner.model_type())
    }

    /// Get estimated memory usage in bytes
    #[wasm_bindgen(js_name = memorySize)]
    pub fn memory_size(&self) -> usize {
        self.inner.memory_size()
    }

    /// Transcribe audio samples
    ///
    /// # Arguments
    /// * `audio` - Audio samples as Float32Array (mono, 16kHz, normalized to [-1, 1])
    /// * `options` - Transcription options
    ///
    /// # Returns
    /// Transcription result with text and language
    #[wasm_bindgen]
    pub fn transcribe(
        &self,
        audio: &[f32],
        options: TranscribeOptionsWasm,
    ) -> Result<TranscriptionResultWasm, JsValue> {
        let result = self
            .inner
            .transcribe(audio, options.into())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(result.into())
    }

    /// Detect language from audio samples
    ///
    /// # Arguments
    /// * `audio` - Audio samples as Float32Array (mono, 16kHz)
    ///
    /// # Returns
    /// Language detection result with probabilities
    #[wasm_bindgen(js_name = detectLanguage)]
    pub fn detect_language(&self, audio: &[f32]) -> Result<LanguageDetectionResult, JsValue> {
        let probs = self
            .inner
            .detect_language(audio)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(probs.into())
    }

    /// Resample audio from source sample rate to 16kHz
    ///
    /// # Arguments
    /// * `audio` - Audio samples at source sample rate
    /// * `source_rate` - Source sample rate (e.g., 44100, 48000)
    ///
    /// # Returns
    /// Resampled audio at 16kHz
    #[wasm_bindgen]
    pub fn resample(&self, audio: &[f32], source_rate: u32) -> Result<Vec<f32>, JsValue> {
        use crate::audio::SincResampler;

        if source_rate == 16000 {
            return Ok(audio.to_vec());
        }

        let resampler = SincResampler::new(source_rate, 16000)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        resampler
            .resample(audio)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // =========================================================================
    // Memory Estimation API
    // =========================================================================

    /// Get estimated weights memory in MB
    #[wasm_bindgen(js_name = weightsMemoryMb)]
    pub fn weights_memory_mb(&self) -> f32 {
        self.inner.config().weights_memory_mb()
    }

    /// Get estimated peak memory usage in MB
    #[wasm_bindgen(js_name = peakMemoryMb)]
    pub fn peak_memory_mb(&self) -> f32 {
        self.inner.config().peak_memory_mb()
    }

    /// Get recommended WASM memory pages (64KB each)
    #[wasm_bindgen(js_name = recommendedWasmPages)]
    pub fn recommended_wasm_pages(&self) -> u32 {
        self.inner.config().recommended_wasm_pages()
    }

    /// Check if the model can run with available memory
    #[wasm_bindgen(js_name = canRunWithMemory)]
    pub fn can_run_with_memory(&self, available_mb: u32) -> bool {
        self.inner.config().can_run_with_memory(available_mb)
    }

    /// Get human-readable memory summary
    #[wasm_bindgen(js_name = memorySummary)]
    pub fn memory_summary(&self) -> String {
        self.inner.config().memory_summary()
    }

    /// Get parameter count
    #[wasm_bindgen(js_name = parameterCount)]
    pub fn parameter_count(&self) -> usize {
        self.inner.config().parameter_count()
    }

    /// Get vocabulary size
    #[wasm_bindgen(js_name = vocabSize)]
    pub fn vocab_size(&self) -> u32 {
        self.inner.config().n_vocab
    }

    /// Get audio context length
    #[wasm_bindgen(js_name = audioContextLength)]
    pub fn audio_context_length(&self) -> u32 {
        self.inner.config().n_audio_ctx
    }

    /// Get text context length (max tokens)
    #[wasm_bindgen(js_name = textContextLength)]
    pub fn text_context_length(&self) -> u32 {
        self.inner.config().n_text_ctx
    }

    /// Get number of encoder layers
    #[wasm_bindgen(js_name = encoderLayerCount)]
    pub fn encoder_layer_count(&self) -> u32 {
        self.inner.config().n_audio_layer
    }

    /// Get number of decoder layers
    #[wasm_bindgen(js_name = decoderLayerCount)]
    pub fn decoder_layer_count(&self) -> u32 {
        self.inner.config().n_text_layer
    }
}

/// Memory requirements for a model type
#[wasm_bindgen]
pub struct ModelMemoryInfo {
    model_type: String,
    weights_mb: f32,
    peak_mb: f32,
    wasm_pages: u32,
    parameters: usize,
}

impl ModelMemoryInfo {
    /// Create memory info for a known model type (for testing)
    #[must_use]
    pub fn for_model(model_type: &str) -> Option<Self> {
        let config = match model_type {
            "tiny" => ModelConfig::tiny(),
            "base" => ModelConfig::base(),
            "small" => ModelConfig::small(),
            "medium" => ModelConfig::medium(),
            "large" => ModelConfig::large(),
            _ => return None,
        };

        Some(Self {
            model_type: model_type.to_string(),
            weights_mb: config.weights_memory_mb(),
            peak_mb: config.peak_memory_mb(),
            wasm_pages: config.recommended_wasm_pages(),
            parameters: config.parameter_count(),
        })
    }
}

#[wasm_bindgen]
impl ModelMemoryInfo {
    /// Get memory info for a model type
    #[wasm_bindgen(constructor)]
    pub fn new(model_type: &str) -> Result<Self, JsValue> {
        Self::for_model(model_type).ok_or_else(|| {
            JsValue::from_str(&format!(
                "Unknown model type: {model_type}. Use 'tiny', 'base', 'small', 'medium', or 'large'"
            ))
        })
    }

    /// Get model type
    #[wasm_bindgen(getter, js_name = modelType)]
    pub fn model_type(&self) -> String {
        self.model_type.clone()
    }

    /// Get weights memory in MB
    #[wasm_bindgen(getter, js_name = weightsMb)]
    pub fn weights_mb(&self) -> f32 {
        self.weights_mb
    }

    /// Get peak memory in MB
    #[wasm_bindgen(getter, js_name = peakMb)]
    pub fn peak_mb(&self) -> f32 {
        self.peak_mb
    }

    /// Get recommended WASM pages
    #[wasm_bindgen(getter, js_name = wasmPages)]
    pub fn wasm_pages(&self) -> u32 {
        self.wasm_pages
    }

    /// Get parameter count
    #[wasm_bindgen(getter)]
    pub fn parameters(&self) -> usize {
        self.parameters
    }

    /// Get human-readable parameter count (e.g., "39M")
    #[wasm_bindgen(js_name = parametersHuman)]
    pub fn parameters_human(&self) -> String {
        if self.parameters >= 1_000_000_000 {
            format!("{:.1}B", self.parameters as f32 / 1_000_000_000.0)
        } else if self.parameters >= 1_000_000 {
            format!("{:.0}M", self.parameters as f32 / 1_000_000.0)
        } else {
            format!("{:.0}K", self.parameters as f32 / 1_000.0)
        }
    }

    /// Check if model fits in available memory
    #[wasm_bindgen(js_name = fitsInMemory)]
    pub fn fits_in_memory(&self, available_mb: f32) -> bool {
        self.peak_mb <= available_mb
    }
}

/// Get recommended model for available memory
#[wasm_bindgen(js_name = recommendedModelForMemory)]
pub fn recommended_model_for_memory(available_mb: u32) -> Option<String> {
    // Try models from largest to smallest that fit
    let models = [("base", ModelConfig::base()), ("tiny", ModelConfig::tiny())];

    for (name, config) in models {
        if config.can_run_with_memory(available_mb) {
            return Some(name.to_string());
        }
    }

    None
}

/// Log a message to the browser console
#[wasm_bindgen(js_name = logToConsole)]
pub fn log_to_console(message: &str) {
    web_sys::console::log_1(&JsValue::from_str(message));
}

/// Get supported languages
#[wasm_bindgen(js_name = supportedLanguages)]
pub fn supported_languages() -> Vec<String> {
    crate::detection::SUPPORTED_LANGUAGES
        .iter()
        .map(|&s| s.to_string())
        .collect()
}

/// Check if a language is supported
#[wasm_bindgen(js_name = isLanguageSupported)]
pub fn is_language_supported(language: &str) -> bool {
    crate::detection::is_supported(language)
}

/// Get the language name for a code
#[wasm_bindgen(js_name = languageName)]
pub fn language_name(code: &str) -> Option<String> {
    crate::detection::language_name(code).map(|s| s.to_string())
}

/// WASM module version
#[wasm_bindgen(js_name = version)]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TranscribeOptionsWasm Tests
    // =========================================================================

    #[test]
    fn test_transcribe_options_new() {
        let opts = TranscribeOptionsWasm::new();
        assert!(opts.language.is_none());
        assert_eq!(opts.task, "transcribe");
        assert_eq!(opts.beam_size, 1);
        assert!((opts.temperature - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transcribe_options_set_language() {
        let mut opts = TranscribeOptionsWasm::new();

        opts.set_language("es");
        assert_eq!(opts.language, Some("es".to_string()));

        opts.set_language("auto");
        assert!(opts.language.is_none());

        opts.set_language("");
        assert!(opts.language.is_none());
    }

    #[test]
    fn test_transcribe_options_set_task() {
        let mut opts = TranscribeOptionsWasm::new();

        opts.set_task("translate");
        assert_eq!(opts.task, "translate");

        opts.set_task("transcribe");
        assert_eq!(opts.task, "transcribe");
    }

    #[test]
    fn test_transcribe_options_set_beam_size() {
        let mut opts = TranscribeOptionsWasm::new();

        opts.set_beam_size(5);
        assert_eq!(opts.beam_size, 5);
    }

    #[test]
    fn test_transcribe_options_set_temperature() {
        let mut opts = TranscribeOptionsWasm::new();

        opts.set_temperature(0.7);
        assert!((opts.temperature - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transcribe_options_into_native_greedy() {
        let opts = TranscribeOptionsWasm::new();
        let native: TranscribeOptions = opts.into();

        assert!(native.language.is_none());
        assert_eq!(native.task, Task::Transcribe);
        assert!(matches!(native.strategy, DecodingStrategy::Greedy));
    }

    #[test]
    fn test_transcribe_options_into_native_beam_search() {
        let mut opts = TranscribeOptionsWasm::new();
        opts.set_beam_size(5);
        opts.set_temperature(0.5);

        let native: TranscribeOptions = opts.into();

        assert!(matches!(
            native.strategy,
            DecodingStrategy::BeamSearch { beam_size: 5, .. }
        ));
    }

    #[test]
    fn test_transcribe_options_into_native_translate() {
        let mut opts = TranscribeOptionsWasm::new();
        opts.set_task("translate");

        let native: TranscribeOptions = opts.into();
        assert_eq!(native.task, Task::Translate);
    }

    #[test]
    fn test_transcribe_options_into_native_sampling() {
        let mut opts = TranscribeOptionsWasm::new();
        opts.set_temperature(0.8);

        let native: TranscribeOptions = opts.into();

        assert!(matches!(
            native.strategy,
            DecodingStrategy::Sampling { temperature, .. } if (temperature - 0.8).abs() < f32::EPSILON
        ));
    }

    // =========================================================================
    // TranscriptionResultWasm Tests
    // =========================================================================

    #[test]
    fn test_transcription_result_from() {
        let native = TranscriptionResult {
            text: "Hello world".to_string(),
            language: "en".to_string(),
            segments: vec![],
        };

        let wasm: TranscriptionResultWasm = native.into();
        assert_eq!(wasm.text(), "Hello world");
        assert_eq!(wasm.language(), "en");
    }

    // =========================================================================
    // LanguageDetectionResult Tests
    // =========================================================================

    #[test]
    fn test_language_detection_result_from() {
        let probs = LanguageProbs {
            languages: vec!["en".to_string(), "es".to_string()],
            probabilities: vec![0.8, 0.2],
        };

        let result: LanguageDetectionResult = probs.into();

        assert_eq!(result.languages(), vec!["en", "es"]);
        assert_eq!(result.probabilities(), vec![0.8, 0.2]);
        assert_eq!(result.top_language(), Some("en".to_string()));
        assert!((result.confidence() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_language_detection_result_empty() {
        let probs = LanguageProbs {
            languages: vec![],
            probabilities: vec![],
        };

        let result: LanguageDetectionResult = probs.into();

        assert!(result.languages().is_empty());
        assert!(result.top_language().is_none());
        assert!((result.confidence() - 0.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Utility Function Tests
    // =========================================================================

    #[test]
    fn test_supported_languages() {
        let langs = supported_languages();
        assert_eq!(langs.len(), 99);
        assert!(langs.contains(&"en".to_string()));
        assert!(langs.contains(&"es".to_string()));
        assert!(langs.contains(&"ja".to_string()));
    }

    #[test]
    fn test_is_language_supported() {
        assert!(is_language_supported("en"));
        assert!(is_language_supported("es"));
        assert!(!is_language_supported("invalid"));
    }

    #[test]
    fn test_language_name() {
        assert_eq!(language_name("en"), Some("English".to_string()));
        assert_eq!(language_name("es"), Some("Spanish".to_string()));
        assert_eq!(language_name("invalid"), None);
    }

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }

    // =========================================================================
    // SegmentWasm Tests
    // =========================================================================

    #[test]
    fn test_segment_wasm_accessors() {
        let segment = SegmentWasm {
            text: "Hello world".to_string(),
            start: 1.5,
            end: 3.0,
        };

        assert_eq!(segment.text(), "Hello world");
        assert!((segment.start() - 1.5).abs() < f32::EPSILON);
        assert!((segment.end() - 3.0).abs() < f32::EPSILON);
        assert!((segment.duration() - 1.5).abs() < f32::EPSILON);
    }

    // =========================================================================
    // TranscriptionResultWasm with Segments Tests
    // =========================================================================

    #[test]
    fn test_transcription_result_with_segments() {
        let native = TranscriptionResult {
            text: "Hello world. How are you?".to_string(),
            language: "en".to_string(),
            segments: vec![
                crate::Segment {
                    text: "Hello world.".to_string(),
                    start: 0.0,
                    end: 1.5,
                    tokens: vec![1, 2, 3],
                },
                crate::Segment {
                    text: " How are you?".to_string(),
                    start: 1.5,
                    end: 3.0,
                    tokens: vec![4, 5, 6],
                },
            ],
        };

        let wasm: TranscriptionResultWasm = native.into();

        assert_eq!(wasm.text(), "Hello world. How are you?");
        assert_eq!(wasm.language(), "en");
        assert_eq!(wasm.segment_count(), 2);

        let seg0 = wasm.get_segment(0).expect("segment 0");
        assert_eq!(seg0.text(), "Hello world.");
        assert!((seg0.start() - 0.0).abs() < f32::EPSILON);
        assert!((seg0.end() - 1.5).abs() < f32::EPSILON);

        let seg1 = wasm.get_segment(1).expect("segment 1");
        assert_eq!(seg1.text(), " How are you?");
        assert!((seg1.start() - 1.5).abs() < f32::EPSILON);
        assert!((seg1.end() - 3.0).abs() < f32::EPSILON);

        // Out of bounds returns None
        assert!(wasm.get_segment(2).is_none());
    }

    #[test]
    fn test_transcription_result_segment_arrays() {
        let native = TranscriptionResult {
            text: "Test".to_string(),
            language: "en".to_string(),
            segments: vec![
                crate::Segment {
                    text: "A".to_string(),
                    start: 0.0,
                    end: 1.0,
                    tokens: vec![1],
                },
                crate::Segment {
                    text: "B".to_string(),
                    start: 1.0,
                    end: 2.0,
                    tokens: vec![2],
                },
            ],
        };

        let wasm: TranscriptionResultWasm = native.into();

        let texts = wasm.segment_texts();
        assert_eq!(texts, vec!["A", "B"]);

        let starts = wasm.segment_starts();
        assert_eq!(starts.len(), 2);
        assert!((starts[0] - 0.0).abs() < f32::EPSILON);
        assert!((starts[1] - 1.0).abs() < f32::EPSILON);

        let ends = wasm.segment_ends();
        assert_eq!(ends.len(), 2);
        assert!((ends[0] - 1.0).abs() < f32::EPSILON);
        assert!((ends[1] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transcription_result_empty_segments() {
        let native = TranscriptionResult {
            text: "".to_string(),
            language: "en".to_string(),
            segments: vec![],
        };

        let wasm: TranscriptionResultWasm = native.into();

        assert_eq!(wasm.segment_count(), 0);
        assert!(wasm.get_segment(0).is_none());
        assert!(wasm.segment_texts().is_empty());
        assert!(wasm.segment_starts().is_empty());
        assert!(wasm.segment_ends().is_empty());
    }

    // =========================================================================
    // ModelMemoryInfo Tests
    // =========================================================================

    #[test]
    fn test_model_memory_info_tiny() {
        let info = ModelMemoryInfo::for_model("tiny").expect("tiny model info");

        assert_eq!(info.model_type(), "tiny");
        assert!(info.weights_mb() > 0.0);
        assert!(info.peak_mb() > info.weights_mb());
        assert!(info.wasm_pages() > 0);
        assert!(info.parameters() > 0);
    }

    #[test]
    fn test_model_memory_info_base() {
        let info = ModelMemoryInfo::for_model("base").expect("base model info");

        assert_eq!(info.model_type(), "base");
        assert!(info.weights_mb() > 0.0);
        assert!(info.peak_mb() > info.weights_mb());
        assert!(info.wasm_pages() > 0);
        assert!(info.parameters() > 0);
    }

    #[test]
    fn test_model_memory_info_invalid() {
        assert!(ModelMemoryInfo::for_model("invalid").is_none());
    }

    #[test]
    fn test_model_memory_info_parameters_human() {
        let tiny_info = ModelMemoryInfo::for_model("tiny").expect("tiny");
        let human = tiny_info.parameters_human();
        // Should be in "XM" format for millions
        assert!(human.contains('M') || human.contains('K'));
    }

    #[test]
    fn test_model_memory_info_fits_in_memory() {
        let info = ModelMemoryInfo::for_model("tiny").expect("tiny");

        // 10GB should be enough for tiny
        assert!(info.fits_in_memory(10000.0));

        // 1MB should not be enough for tiny
        assert!(!info.fits_in_memory(1.0));
    }

    #[test]
    fn test_model_memory_info_base_larger_than_tiny() {
        let tiny_info = ModelMemoryInfo::for_model("tiny").expect("tiny");
        let base_info = ModelMemoryInfo::for_model("base").expect("base");

        assert!(base_info.weights_mb() > tiny_info.weights_mb());
        assert!(base_info.peak_mb() > tiny_info.peak_mb());
        assert!(base_info.parameters() > tiny_info.parameters());
    }

    // =========================================================================
    // recommended_model_for_memory Tests
    // =========================================================================

    #[test]
    fn test_recommended_model_for_memory_ample() {
        // 2GB should recommend base (largest that fits)
        let model = recommended_model_for_memory(2048);
        assert_eq!(model, Some("base".to_string()));
    }

    #[test]
    fn test_recommended_model_for_memory_limited() {
        // Very limited memory should still recommend tiny if it fits
        let tiny_info = ModelMemoryInfo::for_model("tiny").expect("tiny");
        let needed = tiny_info.peak_mb() as u32 + 50; // Just enough for tiny

        let model = recommended_model_for_memory(needed);
        // Should get at least tiny
        assert!(model.is_some());
    }

    #[test]
    fn test_recommended_model_for_memory_insufficient() {
        // 1MB should not be enough for any model
        let model = recommended_model_for_memory(1);
        assert!(model.is_none());
    }
}
