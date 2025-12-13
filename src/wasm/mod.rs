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
mod diarization;
mod gpu;
mod timestamps;
mod vocabulary;
mod worker;

pub use capabilities::{Capabilities, ExecutionMode};
pub use diarization::{
    get_diarization_recommendation, DiarizationConfigWasm, DiarizationResultWasm, DiarizerWasm,
    EmbeddingExtractorWasm, SpeakerEmbeddingWasm, SpeakerSegmentWasm, TurnDetectorWasm,
};
pub use timestamps::{
    get_word_timestamp_recommendation, AlignmentConfigWasm, TimestampInterpolatorWasm,
    TokenTimestampWasm, WordBoundaryWasm, WordTimestampExtractorWasm, WordTimestampResultWasm,
    WordWithTimestampWasm,
};
pub use vocabulary::{
    DomainAdapterWasm, DomainConfigWasm, DomainTermWasm, DomainTypeWasm, HotwordBoosterWasm,
    HotwordConfigWasm, HotwordWasm, TrieSearchResultWasm, VocabularyCustomizerWasm,
    VocabularyTrieWasm,
};
pub use gpu::{
    estimate_mat_mul_flops, estimate_mat_mul_memory, is_gpu_worthwhile,
    recommended_backend_for_model, BackendSelectionWasm, BackendSelectorWasm, BackendTypeWasm,
    DetectionOptionsWasm, GpuBackendWasm, GpuCapabilitiesWasm, GpuDetectionWasm, GpuLimitsWasm,
    SelectionStrategyWasm, SelectorConfigWasm,
};
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

// =============================================================================
// WAPR-103: Streaming WASM API
// =============================================================================

use crate::audio::{ProcessorState, StreamingConfig, StreamingEvent, StreamingProcessor};
use crate::PartialTranscriptionResult;

/// WASM-friendly streaming configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct StreamingConfigWasm {
    input_sample_rate: u32,
    chunk_duration: f32,
    chunk_overlap: f32,
    partial_threshold: f32,
    enable_vad: bool,
    vad_threshold: f32,
}

#[wasm_bindgen]
impl StreamingConfigWasm {
    /// Create default streaming config
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            input_sample_rate: 44100, // Common browser default
            chunk_duration: 30.0,
            chunk_overlap: 1.0,
            partial_threshold: 3.0,
            enable_vad: true,
            vad_threshold: 0.3,
        }
    }

    /// Set input sample rate
    #[wasm_bindgen(js_name = setInputSampleRate)]
    pub fn set_input_sample_rate(&mut self, rate: u32) {
        self.input_sample_rate = rate;
    }

    /// Set chunk duration in seconds
    #[wasm_bindgen(js_name = setChunkDuration)]
    pub fn set_chunk_duration(&mut self, duration: f32) {
        self.chunk_duration = duration;
    }

    /// Set chunk overlap in seconds
    #[wasm_bindgen(js_name = setChunkOverlap)]
    pub fn set_chunk_overlap(&mut self, overlap: f32) {
        self.chunk_overlap = overlap;
    }

    /// Set partial result threshold in seconds
    #[wasm_bindgen(js_name = setPartialThreshold)]
    pub fn set_partial_threshold(&mut self, threshold: f32) {
        self.partial_threshold = threshold;
    }

    /// Enable or disable VAD
    #[wasm_bindgen(js_name = setEnableVad)]
    pub fn set_enable_vad(&mut self, enable: bool) {
        self.enable_vad = enable;
    }

    /// Set VAD threshold
    #[wasm_bindgen(js_name = setVadThreshold)]
    pub fn set_vad_threshold(&mut self, threshold: f32) {
        self.vad_threshold = threshold;
    }

    /// Get input sample rate
    #[wasm_bindgen(getter, js_name = inputSampleRate)]
    pub fn input_sample_rate(&self) -> u32 {
        self.input_sample_rate
    }

    /// Get chunk duration
    #[wasm_bindgen(getter, js_name = chunkDuration)]
    pub fn chunk_duration(&self) -> f32 {
        self.chunk_duration
    }

    /// Get partial threshold
    #[wasm_bindgen(getter, js_name = partialThreshold)]
    pub fn partial_threshold(&self) -> f32 {
        self.partial_threshold
    }
}

impl Default for StreamingConfigWasm {
    fn default() -> Self {
        Self::new()
    }
}

impl From<StreamingConfigWasm> for StreamingConfig {
    fn from(wasm: StreamingConfigWasm) -> Self {
        StreamingConfig::default()
            .chunk_duration(wasm.chunk_duration)
            .chunk_overlap(wasm.chunk_overlap)
            .vad_threshold(wasm.vad_threshold)
    }
}

// =============================================================================
// Low-Latency WASM Bindings (WAPR-113)
// =============================================================================

use crate::audio::LatencyMode;

/// WASM latency mode enum for JavaScript interop
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyModeWasm {
    /// Standard mode (30s chunks, higher accuracy)
    Standard = 0,
    /// Low-latency mode (500ms chunks, faster response)
    LowLatency = 1,
    /// Ultra-low latency mode (250ms chunks, fastest response)
    UltraLow = 2,
    /// Custom configuration
    Custom = 3,
}

impl From<LatencyMode> for LatencyModeWasm {
    fn from(mode: LatencyMode) -> Self {
        match mode {
            LatencyMode::Standard => Self::Standard,
            LatencyMode::LowLatency => Self::LowLatency,
            LatencyMode::UltraLow => Self::UltraLow,
            LatencyMode::Custom => Self::Custom,
        }
    }
}

impl From<LatencyModeWasm> for LatencyMode {
    fn from(mode: LatencyModeWasm) -> Self {
        match mode {
            LatencyModeWasm::Standard => Self::Standard,
            LatencyModeWasm::LowLatency => Self::LowLatency,
            LatencyModeWasm::UltraLow => Self::UltraLow,
            LatencyModeWasm::Custom => Self::Custom,
        }
    }
}

/// WASM-friendly low-latency streaming configuration
///
/// Provides pre-configured settings optimized for different latency requirements.
///
/// # Example
///
/// ```javascript
/// // Create low-latency config for real-time applications
/// const config = LowLatencyConfigWasm.lowLatency();
/// console.log(`Chunk duration: ${config.chunkDurationMs}ms`);
/// console.log(`Expected latency: ${config.expectedLatencyMs}ms`);
///
/// // Create ultra-low latency for voice assistants
/// const ultraConfig = LowLatencyConfigWasm.ultraLow();
/// ```
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct LowLatencyConfigWasm {
    inner: StreamingConfig,
}

#[wasm_bindgen]
impl LowLatencyConfigWasm {
    /// Create a standard latency configuration (30s chunks)
    ///
    /// Best for batch-style transcription where accuracy is prioritized.
    #[wasm_bindgen]
    pub fn standard() -> Self {
        Self {
            inner: StreamingConfig::default(),
        }
    }

    /// Create a low-latency configuration (500ms chunks)
    ///
    /// Optimized for real-time applications requiring fast response:
    /// - 500ms chunk duration
    /// - 50ms overlap
    /// - 100ms minimum speech duration
    /// - ~500ms expected latency
    #[wasm_bindgen(js_name = lowLatency)]
    pub fn low_latency() -> Self {
        Self {
            inner: StreamingConfig::low_latency(),
        }
    }

    /// Create an ultra-low latency configuration (250ms chunks)
    ///
    /// Optimized for the fastest possible response:
    /// - 250ms chunk duration
    /// - 25ms overlap
    /// - 50ms minimum speech duration
    /// - ~250ms expected latency
    ///
    /// Note: May reduce transcription accuracy due to less context.
    #[wasm_bindgen(js_name = ultraLow)]
    pub fn ultra_low() -> Self {
        Self {
            inner: StreamingConfig::ultra_low_latency(),
        }
    }

    /// Create a custom latency configuration
    ///
    /// # Arguments
    /// * `chunk_duration_ms` - Chunk duration in milliseconds
    /// * `overlap_ms` - Overlap between chunks in milliseconds
    /// * `min_speech_ms` - Minimum speech duration to trigger processing
    #[wasm_bindgen]
    pub fn custom(chunk_duration_ms: u32, overlap_ms: u32, min_speech_ms: u32) -> Self {
        let chunk_duration = chunk_duration_ms as f32 / 1000.0;
        let chunk_overlap = overlap_ms as f32 / 1000.0;
        let buffer_duration = chunk_duration * 4.0; // 4x chunk duration

        Self {
            inner: StreamingConfig::custom_latency(
                chunk_duration,
                chunk_overlap,
                min_speech_ms,
                buffer_duration,
            ),
        }
    }

    /// Get the latency mode
    #[wasm_bindgen(getter, js_name = latencyMode)]
    pub fn latency_mode(&self) -> LatencyModeWasm {
        self.inner.latency_mode().into()
    }

    /// Get chunk duration in milliseconds
    #[wasm_bindgen(getter, js_name = chunkDurationMs)]
    pub fn chunk_duration_ms(&self) -> f32 {
        self.inner.chunk_duration * 1000.0
    }

    /// Get chunk overlap in milliseconds
    #[wasm_bindgen(getter, js_name = chunkOverlapMs)]
    pub fn chunk_overlap_ms(&self) -> f32 {
        self.inner.chunk_overlap * 1000.0
    }

    /// Get expected latency in milliseconds
    ///
    /// This is approximately the chunk duration plus processing overhead.
    #[wasm_bindgen(getter, js_name = expectedLatencyMs)]
    pub fn expected_latency_ms(&self) -> f32 {
        self.inner.expected_latency_ms()
    }

    /// Get minimum speech duration in milliseconds
    #[wasm_bindgen(getter, js_name = minSpeechDurationMs)]
    pub fn min_speech_duration_ms(&self) -> u32 {
        self.inner.min_speech_duration_ms
    }

    /// Check if this is a low-latency configuration
    #[wasm_bindgen(getter, js_name = isLowLatency)]
    pub fn is_low_latency(&self) -> bool {
        self.inner.is_low_latency()
    }

    /// Get chunk size in samples (at 16kHz)
    #[wasm_bindgen(getter, js_name = chunkSamples)]
    pub fn chunk_samples(&self) -> usize {
        self.inner.chunk_samples()
    }

    /// Get overlap size in samples (at 16kHz)
    #[wasm_bindgen(getter, js_name = overlapSamples)]
    pub fn overlap_samples(&self) -> usize {
        self.inner.overlap_samples()
    }

    /// Enable VAD (Voice Activity Detection)
    #[wasm_bindgen(js_name = withVad)]
    pub fn with_vad(mut self, enable: bool) -> Self {
        self.inner = if enable {
            self.inner.with_vad()
        } else {
            self.inner.without_vad()
        };
        self
    }

    /// Set VAD threshold (0.0 - 1.0)
    #[wasm_bindgen(js_name = withVadThreshold)]
    pub fn with_vad_threshold(mut self, threshold: f32) -> Self {
        self.inner = self.inner.vad_threshold(threshold);
        self
    }

    /// Set input sample rate
    #[wasm_bindgen(js_name = withInputSampleRate)]
    pub fn with_input_sample_rate(mut self, rate: u32) -> Self {
        self.inner.input_sample_rate = rate;
        self
    }
}

impl LowLatencyConfigWasm {
    /// Convert to StreamingConfig for internal use
    #[must_use]
    pub fn into_streaming_config(self) -> StreamingConfig {
        self.inner
    }
}

/// WASM-friendly streaming KV cache statistics
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct StreamingCacheStatsWasm {
    seq_len: usize,
    total_tokens: usize,
    slide_count: usize,
    window_size: usize,
    utilization: f32,
    memory_bytes: usize,
}

#[wasm_bindgen]
impl StreamingCacheStatsWasm {
    /// Get current sequence length in cache
    #[wasm_bindgen(getter, js_name = seqLen)]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get total tokens processed
    #[wasm_bindgen(getter, js_name = totalTokens)]
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Get number of window slides
    #[wasm_bindgen(getter, js_name = slideCount)]
    pub fn slide_count(&self) -> usize {
        self.slide_count
    }

    /// Get window size
    #[wasm_bindgen(getter, js_name = windowSize)]
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get cache utilization (0.0 - 1.0)
    #[wasm_bindgen(getter)]
    pub fn utilization(&self) -> f32 {
        self.utilization
    }

    /// Get memory usage in bytes
    #[wasm_bindgen(getter, js_name = memoryBytes)]
    pub fn memory_bytes(&self) -> usize {
        self.memory_bytes
    }

    /// Get memory usage in KB
    #[wasm_bindgen(getter, js_name = memoryKb)]
    pub fn memory_kb(&self) -> f32 {
        self.memory_bytes as f32 / 1024.0
    }
}

impl From<crate::model::StreamingCacheStats> for StreamingCacheStatsWasm {
    fn from(stats: crate::model::StreamingCacheStats) -> Self {
        Self {
            seq_len: stats.seq_len,
            total_tokens: stats.total_tokens,
            slide_count: stats.slide_count,
            window_size: stats.window_size,
            utilization: stats.utilization(),
            memory_bytes: stats.memory_bytes,
        }
    }
}

/// Get latency recommendations for different use cases
#[wasm_bindgen(js_name = getLatencyRecommendation)]
pub fn get_latency_recommendation(use_case: &str) -> String {
    match use_case.to_lowercase().as_str() {
        "batch" | "transcription" | "offline" => {
            "Use standard mode (30s chunks) for best accuracy with batch transcription.".to_string()
        }
        "realtime" | "real-time" | "live" | "streaming" => {
            "Use low-latency mode (500ms chunks) for real-time applications.".to_string()
        }
        "voice-assistant" | "assistant" | "dictation" | "voice" => {
            "Use ultra-low latency mode (250ms chunks) for voice assistants and dictation.".to_string()
        }
        "subtitles" | "captions" | "live-captions" => {
            "Use low-latency mode (500ms chunks) for live captioning.".to_string()
        }
        _ => {
            "Unknown use case. Available options: batch, realtime, voice-assistant, subtitles.".to_string()
        }
    }
}

/// Get expected latency for a given chunk duration
#[wasm_bindgen(js_name = expectedLatencyForChunkMs)]
pub fn expected_latency_for_chunk_ms(chunk_duration_ms: u32) -> f32 {
    // Latency is approximately chunk duration + ~50ms processing overhead
    chunk_duration_ms as f32 + 50.0
}

/// WASM-friendly partial transcription result
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct PartialTranscriptionResultWasm {
    text: String,
    language: String,
    is_final: bool,
    confidence: f32,
    duration_secs: f32,
    processing_time_secs: f32,
}

#[wasm_bindgen]
impl PartialTranscriptionResultWasm {
    /// Get the transcribed text
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }

    /// Get the detected language
    #[wasm_bindgen(getter)]
    pub fn language(&self) -> String {
        self.language.clone()
    }

    /// Check if this is the final result
    #[wasm_bindgen(getter, js_name = isFinal)]
    pub fn is_final(&self) -> bool {
        self.is_final
    }

    /// Get confidence score (0.0 - 1.0)
    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Get audio duration in seconds
    #[wasm_bindgen(getter, js_name = durationSecs)]
    pub fn duration_secs(&self) -> f32 {
        self.duration_secs
    }

    /// Get processing time in seconds
    #[wasm_bindgen(getter, js_name = processingTimeSecs)]
    pub fn processing_time_secs(&self) -> f32 {
        self.processing_time_secs
    }

    /// Get real-time factor
    #[wasm_bindgen(js_name = realTimeFactor)]
    pub fn real_time_factor(&self) -> f32 {
        if self.duration_secs <= 0.0 {
            0.0
        } else {
            self.processing_time_secs / self.duration_secs
        }
    }

    /// Check if result has text
    #[wasm_bindgen(js_name = hasText)]
    pub fn has_text(&self) -> bool {
        !self.text.is_empty()
    }
}

impl From<PartialTranscriptionResult> for PartialTranscriptionResultWasm {
    fn from(result: PartialTranscriptionResult) -> Self {
        Self {
            text: result.text,
            language: result.language,
            is_final: result.is_final,
            confidence: result.confidence,
            duration_secs: result.duration_secs,
            processing_time_secs: result.processing_time_secs,
        }
    }
}

/// WASM-friendly streaming event types
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingEventTypeWasm {
    SpeechStart,
    SpeechEnd,
    PartialReady,
    ChunkReady,
    ProcessingStarted,
    ProcessingCompleted,
    Error,
    Reset,
}

/// WASM-friendly streaming event
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct StreamingEventWasm {
    event_type: StreamingEventTypeWasm,
    accumulated_samples: Option<usize>,
    duration_secs: Option<f32>,
    error_message: Option<String>,
}

#[wasm_bindgen]
impl StreamingEventWasm {
    /// Get event type
    #[wasm_bindgen(getter, js_name = eventType)]
    pub fn event_type(&self) -> StreamingEventTypeWasm {
        self.event_type
    }

    /// Get accumulated samples (for PartialReady events)
    #[wasm_bindgen(getter, js_name = accumulatedSamples)]
    pub fn accumulated_samples(&self) -> Option<usize> {
        self.accumulated_samples
    }

    /// Get duration in seconds (for PartialReady/ChunkReady events)
    #[wasm_bindgen(getter, js_name = durationSecs)]
    pub fn duration_secs(&self) -> Option<f32> {
        self.duration_secs
    }

    /// Get error message (for Error events)
    #[wasm_bindgen(getter, js_name = errorMessage)]
    pub fn error_message(&self) -> Option<String> {
        self.error_message.clone()
    }

    /// Check if this is a speech-related event
    #[wasm_bindgen(js_name = isSpeechEvent)]
    pub fn is_speech_event(&self) -> bool {
        matches!(
            self.event_type,
            StreamingEventTypeWasm::SpeechStart | StreamingEventTypeWasm::SpeechEnd
        )
    }

    /// Check if this is a result-related event
    #[wasm_bindgen(js_name = isResultEvent)]
    pub fn is_result_event(&self) -> bool {
        matches!(
            self.event_type,
            StreamingEventTypeWasm::PartialReady | StreamingEventTypeWasm::ChunkReady
        )
    }
}

impl From<StreamingEvent> for StreamingEventWasm {
    fn from(event: StreamingEvent) -> Self {
        match event {
            StreamingEvent::SpeechStart => Self {
                event_type: StreamingEventTypeWasm::SpeechStart,
                accumulated_samples: None,
                duration_secs: None,
                error_message: None,
            },
            StreamingEvent::SpeechEnd => Self {
                event_type: StreamingEventTypeWasm::SpeechEnd,
                accumulated_samples: None,
                duration_secs: None,
                error_message: None,
            },
            StreamingEvent::PartialReady {
                accumulated_samples,
                duration_secs,
            } => Self {
                event_type: StreamingEventTypeWasm::PartialReady,
                accumulated_samples: Some(accumulated_samples),
                duration_secs: Some(duration_secs),
                error_message: None,
            },
            StreamingEvent::ChunkReady { duration_secs } => Self {
                event_type: StreamingEventTypeWasm::ChunkReady,
                accumulated_samples: None,
                duration_secs: Some(duration_secs),
                error_message: None,
            },
            StreamingEvent::ProcessingStarted => Self {
                event_type: StreamingEventTypeWasm::ProcessingStarted,
                accumulated_samples: None,
                duration_secs: None,
                error_message: None,
            },
            StreamingEvent::ProcessingCompleted => Self {
                event_type: StreamingEventTypeWasm::ProcessingCompleted,
                accumulated_samples: None,
                duration_secs: None,
                error_message: None,
            },
            StreamingEvent::Error(msg) => Self {
                event_type: StreamingEventTypeWasm::Error,
                accumulated_samples: None,
                duration_secs: None,
                error_message: Some(msg),
            },
            StreamingEvent::Reset => Self {
                event_type: StreamingEventTypeWasm::Reset,
                accumulated_samples: None,
                duration_secs: None,
                error_message: None,
            },
        }
    }
}

/// WASM-friendly processor state
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessorStateWasm {
    WaitingForSpeech,
    AccumulatingSpeech,
    PartialResultReady,
    ChunkReady,
    Processing,
    Error,
}

impl From<ProcessorState> for ProcessorStateWasm {
    fn from(state: ProcessorState) -> Self {
        match state {
            ProcessorState::WaitingForSpeech => ProcessorStateWasm::WaitingForSpeech,
            ProcessorState::AccumulatingSpeech => ProcessorStateWasm::AccumulatingSpeech,
            ProcessorState::PartialResultReady => ProcessorStateWasm::PartialResultReady,
            ProcessorState::ChunkReady => ProcessorStateWasm::ChunkReady,
            ProcessorState::Processing => ProcessorStateWasm::Processing,
            ProcessorState::Error => ProcessorStateWasm::Error,
        }
    }
}

/// WASM streaming session for real-time transcription
///
/// This provides a JavaScript-friendly streaming API for real-time
/// speech-to-text transcription in the browser.
///
/// # Usage
///
/// ```javascript
/// const config = new StreamingConfigWasm();
/// config.setInputSampleRate(44100);
///
/// const session = new StreamingSessionWasm(whisper, config);
///
/// // Push audio chunks from MediaRecorder or Web Audio API
/// const partial = session.pushAudio(audioFloat32Array);
/// if (partial) {
///     console.log("Partial:", partial.text);
/// }
///
/// // Check for events
/// while (session.hasEvents()) {
///     const event = session.popEvent();
///     console.log("Event:", event.eventType);
/// }
///
/// // Get final result when chunk is ready
/// if (session.hasChunk()) {
///     const final = session.finalize();
///     console.log("Final:", final.text);
/// }
/// ```
#[wasm_bindgen]
pub struct StreamingSessionWasm {
    whisper: WhisperApr,
    processor: StreamingProcessor,
    options: TranscribeOptions,
    last_partial_text: String,
}

#[wasm_bindgen]
impl StreamingSessionWasm {
    /// Create a new streaming session
    #[wasm_bindgen(constructor)]
    pub fn new(
        whisper: &WhisperAprWasm,
        config: StreamingConfigWasm,
    ) -> Self {
        let streaming_config = StreamingConfig {
            input_sample_rate: config.input_sample_rate,
            output_sample_rate: 16000,
            chunk_duration: config.chunk_duration,
            chunk_overlap: config.chunk_overlap,
            buffer_duration: 5.0,
            enable_vad: config.enable_vad,
            vad_threshold: config.vad_threshold,
            min_speech_duration_ms: 300,
            latency_mode: crate::audio::LatencyMode::Standard,
        };
        let mut processor = StreamingProcessor::new(streaming_config);
        processor.set_partial_threshold(config.partial_threshold);

        Self {
            whisper: whisper.inner.clone(),
            processor,
            options: TranscribeOptions::default(),
            last_partial_text: String::new(),
        }
    }

    /// Set transcription options
    #[wasm_bindgen(js_name = setOptions)]
    pub fn set_options(&mut self, options: TranscribeOptionsWasm) {
        self.options = options.into();
    }

    /// Push audio samples and get partial result if available
    #[wasm_bindgen(js_name = pushAudio)]
    pub fn push_audio(&mut self, audio: &[f32]) -> Option<PartialTranscriptionResultWasm> {
        self.processor.push_audio(audio);
        self.processor.process();

        // Check for partial result
        if self.processor.has_partial() {
            if let Some(partial_audio) = self.processor.get_partial() {
                if let Ok(result) = self.whisper.transcribe_partial(
                    &partial_audio,
                    self.options.clone(),
                    false,
                ) {
                    // Deduplicate
                    if result.text != self.last_partial_text {
                        self.last_partial_text = result.text.clone();
                        return Some(result.into());
                    }
                }
            }
        }

        None
    }

    /// Check if a complete chunk is ready
    #[wasm_bindgen(js_name = hasChunk)]
    pub fn has_chunk(&self) -> bool {
        self.processor.has_chunk()
    }

    /// Check if there are pending events
    #[wasm_bindgen(js_name = hasEvents)]
    pub fn has_events(&self) -> bool {
        self.processor.has_events()
    }

    /// Get the number of pending events
    #[wasm_bindgen(js_name = eventCount)]
    pub fn event_count(&self) -> usize {
        self.processor.event_count()
    }

    /// Pop the next event
    #[wasm_bindgen(js_name = popEvent)]
    pub fn pop_event(&mut self) -> Option<StreamingEventWasm> {
        self.processor.pop_event().map(|e| e.into())
    }

    /// Get the current processor state
    #[wasm_bindgen(getter)]
    pub fn state(&self) -> ProcessorStateWasm {
        self.processor.state().into()
    }

    /// Get chunk progress (0.0 - 1.0)
    #[wasm_bindgen(js_name = chunkProgress)]
    pub fn chunk_progress(&self) -> f32 {
        self.processor.chunk_progress()
    }

    /// Get partial audio duration in seconds
    #[wasm_bindgen(js_name = partialDuration)]
    pub fn partial_duration(&self) -> f32 {
        self.processor.partial_duration()
    }

    /// Finalize and get the transcription result for the current chunk
    #[wasm_bindgen]
    pub fn finalize(&mut self) -> Result<PartialTranscriptionResultWasm, JsValue> {
        let chunk = self.processor.get_chunk().ok_or_else(|| {
            JsValue::from_str("no chunk ready for finalization")
        })?;

        let result = self.whisper.transcribe_partial(&chunk, self.options.clone(), true)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        self.last_partial_text.clear();
        Ok(result.into())
    }

    /// Flush any remaining audio and get final result
    #[wasm_bindgen]
    pub fn flush(&mut self) -> Option<PartialTranscriptionResultWasm> {
        if let Some(chunk) = self.processor.flush() {
            if let Ok(result) = self.whisper.transcribe_partial(&chunk, self.options.clone(), true) {
                self.last_partial_text.clear();
                return Some(result.into());
            }
        }
        None
    }

    /// Reset the session
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.processor.reset();
        self.last_partial_text.clear();
    }

    /// Set partial result threshold in seconds
    #[wasm_bindgen(js_name = setPartialThreshold)]
    pub fn set_partial_threshold(&mut self, seconds: f32) {
        self.processor.set_partial_threshold(seconds);
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;

    // =========================================================================
    // StreamingConfigWasm Tests
    // =========================================================================

    #[test]
    fn test_streaming_config_new() {
        let config = StreamingConfigWasm::new();
        assert_eq!(config.input_sample_rate, 44100);
        assert!((config.chunk_duration - 30.0).abs() < 0.01);
        assert!(config.enable_vad);
    }

    #[test]
    fn test_streaming_config_setters() {
        let mut config = StreamingConfigWasm::new();

        config.set_input_sample_rate(16000);
        assert_eq!(config.input_sample_rate(), 16000);

        config.set_chunk_duration(15.0);
        assert!((config.chunk_duration() - 15.0).abs() < 0.01);

        config.set_partial_threshold(5.0);
        assert!((config.partial_threshold() - 5.0).abs() < 0.01);

        config.set_enable_vad(false);
        assert!(!config.enable_vad);
    }

    // =========================================================================
    // PartialTranscriptionResultWasm Tests
    // =========================================================================

    #[test]
    fn test_partial_result_from() {
        let native = PartialTranscriptionResult {
            text: "hello".to_string(),
            language: "en".to_string(),
            is_final: false,
            confidence: 0.95,
            duration_secs: 2.0,
            processing_time_secs: 0.4,
        };

        let wasm: PartialTranscriptionResultWasm = native.into();

        assert_eq!(wasm.text(), "hello");
        assert_eq!(wasm.language(), "en");
        assert!(!wasm.is_final());
        assert!((wasm.confidence() - 0.95).abs() < 0.01);
        assert!((wasm.duration_secs() - 2.0).abs() < 0.01);
        assert!((wasm.processing_time_secs() - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_partial_result_real_time_factor() {
        let result = PartialTranscriptionResultWasm {
            text: "test".to_string(),
            language: "en".to_string(),
            is_final: true,
            confidence: 1.0,
            duration_secs: 4.0,
            processing_time_secs: 1.0,
        };

        assert!((result.real_time_factor() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_partial_result_has_text() {
        let with_text = PartialTranscriptionResultWasm {
            text: "hello".to_string(),
            language: "en".to_string(),
            is_final: false,
            confidence: 1.0,
            duration_secs: 1.0,
            processing_time_secs: 0.1,
        };
        let empty = PartialTranscriptionResultWasm {
            text: String::new(),
            language: "en".to_string(),
            is_final: false,
            confidence: 0.0,
            duration_secs: 0.5,
            processing_time_secs: 0.05,
        };

        assert!(with_text.has_text());
        assert!(!empty.has_text());
    }

    // =========================================================================
    // StreamingEventWasm Tests
    // =========================================================================

    #[test]
    fn test_streaming_event_speech_start() {
        let native = StreamingEvent::SpeechStart;
        let wasm: StreamingEventWasm = native.into();

        assert_eq!(wasm.event_type(), StreamingEventTypeWasm::SpeechStart);
        assert!(wasm.is_speech_event());
        assert!(!wasm.is_result_event());
    }

    #[test]
    fn test_streaming_event_partial_ready() {
        let native = StreamingEvent::PartialReady {
            accumulated_samples: 48000,
            duration_secs: 3.0,
        };
        let wasm: StreamingEventWasm = native.into();

        assert_eq!(wasm.event_type(), StreamingEventTypeWasm::PartialReady);
        assert_eq!(wasm.accumulated_samples(), Some(48000));
        assert!((wasm.duration_secs().unwrap_or(0.0) - 3.0).abs() < 0.01);
        assert!(wasm.is_result_event());
    }

    #[test]
    fn test_streaming_event_error() {
        let native = StreamingEvent::Error("test error".to_string());
        let wasm: StreamingEventWasm = native.into();

        assert_eq!(wasm.event_type(), StreamingEventTypeWasm::Error);
        assert_eq!(wasm.error_message(), Some("test error".to_string()));
    }

    // =========================================================================
    // ProcessorStateWasm Tests
    // =========================================================================

    #[test]
    fn test_processor_state_conversion() {
        assert_eq!(
            ProcessorStateWasm::from(ProcessorState::WaitingForSpeech),
            ProcessorStateWasm::WaitingForSpeech
        );
        assert_eq!(
            ProcessorStateWasm::from(ProcessorState::AccumulatingSpeech),
            ProcessorStateWasm::AccumulatingSpeech
        );
        assert_eq!(
            ProcessorStateWasm::from(ProcessorState::ChunkReady),
            ProcessorStateWasm::ChunkReady
        );
    }

    // =========================================================================
    // WAPR-113: Low-Latency WASM Bindings Tests
    // =========================================================================

    #[test]
    fn test_latency_mode_wasm_enum() {
        assert_eq!(LatencyModeWasm::Standard as u32, 0);
        assert_eq!(LatencyModeWasm::LowLatency as u32, 1);
        assert_eq!(LatencyModeWasm::UltraLow as u32, 2);
        assert_eq!(LatencyModeWasm::Custom as u32, 3);
    }

    #[test]
    fn test_latency_mode_wasm_from_native() {
        assert_eq!(
            LatencyModeWasm::from(LatencyMode::Standard),
            LatencyModeWasm::Standard
        );
        assert_eq!(
            LatencyModeWasm::from(LatencyMode::LowLatency),
            LatencyModeWasm::LowLatency
        );
        assert_eq!(
            LatencyModeWasm::from(LatencyMode::UltraLow),
            LatencyModeWasm::UltraLow
        );
        assert_eq!(
            LatencyModeWasm::from(LatencyMode::Custom),
            LatencyModeWasm::Custom
        );
    }

    #[test]
    fn test_latency_mode_wasm_to_native() {
        assert_eq!(
            LatencyMode::from(LatencyModeWasm::Standard),
            LatencyMode::Standard
        );
        assert_eq!(
            LatencyMode::from(LatencyModeWasm::LowLatency),
            LatencyMode::LowLatency
        );
        assert_eq!(
            LatencyMode::from(LatencyModeWasm::UltraLow),
            LatencyMode::UltraLow
        );
        assert_eq!(
            LatencyMode::from(LatencyModeWasm::Custom),
            LatencyMode::Custom
        );
    }

    #[test]
    fn test_low_latency_config_standard() {
        let config = LowLatencyConfigWasm::standard();
        assert_eq!(config.latency_mode(), LatencyModeWasm::Standard);
        assert!(!config.is_low_latency());
        assert!((config.chunk_duration_ms() - 30000.0).abs() < 1.0);
    }

    #[test]
    fn test_low_latency_config_low_latency() {
        let config = LowLatencyConfigWasm::low_latency();
        assert_eq!(config.latency_mode(), LatencyModeWasm::LowLatency);
        assert!(config.is_low_latency());
        assert!((config.chunk_duration_ms() - 500.0).abs() < 1.0);
        assert!((config.chunk_overlap_ms() - 50.0).abs() < 1.0);
        assert!((config.expected_latency_ms() - 500.0).abs() < 1.0);
    }

    #[test]
    fn test_low_latency_config_ultra_low() {
        let config = LowLatencyConfigWasm::ultra_low();
        assert_eq!(config.latency_mode(), LatencyModeWasm::UltraLow);
        assert!(config.is_low_latency());
        assert!((config.chunk_duration_ms() - 250.0).abs() < 1.0);
        assert!((config.chunk_overlap_ms() - 25.0).abs() < 1.0);
        assert!((config.expected_latency_ms() - 250.0).abs() < 1.0);
    }

    #[test]
    fn test_low_latency_config_custom() {
        let config = LowLatencyConfigWasm::custom(300, 30, 50);
        assert_eq!(config.latency_mode(), LatencyModeWasm::Custom);
        assert!((config.chunk_duration_ms() - 300.0).abs() < 1.0);
        assert!((config.chunk_overlap_ms() - 30.0).abs() < 1.0);
        assert_eq!(config.min_speech_duration_ms(), 50);
    }

    #[test]
    fn test_low_latency_config_chunk_samples() {
        let config = LowLatencyConfigWasm::low_latency();
        // 500ms at 16kHz = 8000 samples
        assert_eq!(config.chunk_samples(), 8000);
    }

    #[test]
    fn test_low_latency_config_overlap_samples() {
        let config = LowLatencyConfigWasm::low_latency();
        // 50ms at 16kHz = 800 samples
        assert_eq!(config.overlap_samples(), 800);
    }

    #[test]
    fn test_low_latency_config_with_vad() {
        let config = LowLatencyConfigWasm::low_latency().with_vad(true);
        // VAD should be enabled by default anyway
        assert!(config.is_low_latency());
    }

    #[test]
    fn test_low_latency_config_with_vad_threshold() {
        let config = LowLatencyConfigWasm::low_latency().with_vad_threshold(0.5);
        assert!(config.is_low_latency());
    }

    #[test]
    fn test_low_latency_config_with_input_sample_rate() {
        let config = LowLatencyConfigWasm::low_latency().with_input_sample_rate(48000);
        assert_eq!(config.into_streaming_config().input_sample_rate, 48000);
    }

    #[test]
    fn test_streaming_cache_stats_wasm() {
        let native = crate::model::StreamingCacheStats {
            seq_len: 32,
            total_tokens: 100,
            slide_count: 2,
            window_size: 64,
            context_overlap: 16,
            memory_bytes: 8192,
        };
        let wasm: StreamingCacheStatsWasm = native.into();

        assert_eq!(wasm.seq_len(), 32);
        assert_eq!(wasm.total_tokens(), 100);
        assert_eq!(wasm.slide_count(), 2);
        assert_eq!(wasm.window_size(), 64);
        assert!((wasm.utilization() - 0.5).abs() < 0.01);
        assert_eq!(wasm.memory_bytes(), 8192);
        assert!((wasm.memory_kb() - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_get_latency_recommendation_batch() {
        let rec = get_latency_recommendation("batch");
        assert!(rec.contains("standard mode"));
    }

    #[test]
    fn test_get_latency_recommendation_realtime() {
        let rec = get_latency_recommendation("realtime");
        assert!(rec.contains("low-latency mode"));
    }

    #[test]
    fn test_get_latency_recommendation_voice_assistant() {
        let rec = get_latency_recommendation("voice-assistant");
        assert!(rec.contains("ultra-low latency"));
    }

    #[test]
    fn test_get_latency_recommendation_subtitles() {
        let rec = get_latency_recommendation("subtitles");
        assert!(rec.contains("low-latency mode"));
    }

    #[test]
    fn test_get_latency_recommendation_unknown() {
        let rec = get_latency_recommendation("unknown");
        assert!(rec.contains("Unknown use case"));
    }

    #[test]
    fn test_expected_latency_for_chunk_ms() {
        let latency = expected_latency_for_chunk_ms(500);
        assert!((latency - 550.0).abs() < 1.0); // 500 + 50ms overhead
    }

    // =========================================================================
    // WhisperAprWasm Tests
    // =========================================================================

    #[test]
    fn test_whisper_wasm_tiny() {
        let whisper = WhisperAprWasm::tiny();
        assert_eq!(whisper.model_type(), "Tiny");
        assert!(whisper.memory_size() > 0);
    }

    #[test]
    fn test_whisper_wasm_base() {
        let whisper = WhisperAprWasm::base();
        assert_eq!(whisper.model_type(), "Base");
        assert!(whisper.memory_size() > 0);
    }

    #[test]
    fn test_whisper_wasm_new_tiny() {
        let whisper = WhisperAprWasm::new("tiny").expect("should create tiny model");
        assert_eq!(whisper.model_type(), "Tiny");
    }

    #[test]
    fn test_whisper_wasm_new_base() {
        let whisper = WhisperAprWasm::new("base").expect("should create base model");
        assert_eq!(whisper.model_type(), "Base");
    }

    #[test]
    fn test_whisper_wasm_new_small() {
        let whisper = WhisperAprWasm::new("small").expect("should create small model");
        assert_eq!(whisper.model_type(), "Small");
    }

    #[test]
    fn test_whisper_wasm_new_medium() {
        let whisper = WhisperAprWasm::new("medium").expect("should create medium model");
        assert_eq!(whisper.model_type(), "Medium");
    }

    #[test]
    fn test_whisper_wasm_new_large() {
        let whisper = WhisperAprWasm::new("large").expect("should create large model");
        assert_eq!(whisper.model_type(), "Large");
    }

    // Note: test_whisper_wasm_new_invalid removed because JsValue error handling
    // doesn't work in non-WASM test context

    #[test]
    fn test_whisper_wasm_memory_methods() {
        let whisper = WhisperAprWasm::tiny();

        let weights_mb = whisper.weights_memory_mb();
        assert!(weights_mb > 0.0);

        let peak_mb = whisper.peak_memory_mb();
        assert!(peak_mb >= weights_mb);

        let wasm_pages = whisper.recommended_wasm_pages();
        assert!(wasm_pages > 0);

        let param_count = whisper.parameter_count();
        assert!(param_count > 0);

        let vocab_size = whisper.vocab_size();
        assert!(vocab_size > 0);
    }

    #[test]
    fn test_whisper_wasm_resample_passthrough() {
        let whisper = WhisperAprWasm::tiny();
        let audio: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];

        // 16kHz to 16kHz should passthrough
        let result = whisper.resample(&audio, 16000).expect("should passthrough");
        assert_eq!(result.len(), audio.len());
    }

    #[test]
    fn test_whisper_wasm_resample_downsample() {
        let whisper = WhisperAprWasm::tiny();
        // Generate some audio at 48kHz
        let audio: Vec<f32> = (0..4800).map(|i| (i as f32 / 4800.0 * 2.0 * std::f32::consts::PI).sin() * 0.5).collect();

        let result = whisper.resample(&audio, 48000).expect("should resample");
        // Downsampling 48kHz to 16kHz should reduce samples by factor of 3
        assert!(result.len() < audio.len());
    }

    // =========================================================================
    // StreamingConfigWasm Additional Tests
    // =========================================================================

    #[test]
    fn test_streaming_config_into_native() {
        let mut config = StreamingConfigWasm::new();
        config.set_chunk_duration(25.0);
        config.set_chunk_overlap(2.0);
        config.set_partial_threshold(4.0);
        config.set_enable_vad(false);
        config.set_vad_threshold(0.5);

        let native: StreamingConfig = config.into();
        // Verify the config was converted (we can't easily inspect all fields but the conversion should work)
        assert!(native.chunk_duration > 0.0);
    }

    #[test]
    fn test_streaming_config_getters() {
        let mut config = StreamingConfigWasm::new();
        config.set_input_sample_rate(48000);
        config.set_chunk_duration(25.0);
        config.set_partial_threshold(5.0);

        assert_eq!(config.input_sample_rate(), 48000);
        assert!((config.chunk_duration() - 25.0).abs() < f32::EPSILON);
        assert!((config.partial_threshold() - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfigWasm::default();
        assert_eq!(config.input_sample_rate(), 44100);
        assert!((config.chunk_duration() - 30.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // PartialTranscriptionResultWasm Additional Tests
    // =========================================================================

    #[test]
    fn test_partial_result_empty() {
        let native = PartialTranscriptionResult {
            text: String::new(),
            language: "en".to_string(),
            is_final: false,
            confidence: 0.0,
            duration_secs: 0.0,
            processing_time_secs: 0.0,
        };

        let wasm: PartialTranscriptionResultWasm = native.into();
        assert!(wasm.text().is_empty());
        assert_eq!(wasm.language(), "en");
        assert!(!wasm.is_final());
        assert!(!wasm.has_text());
    }

    #[test]
    fn test_partial_result_with_text() {
        let native = PartialTranscriptionResult {
            text: "Final text".to_string(),
            language: "es".to_string(),
            is_final: true,
            confidence: 0.95,
            duration_secs: 5.0,
            processing_time_secs: 1.0,
        };

        let wasm: PartialTranscriptionResultWasm = native.into();
        assert_eq!(wasm.text(), "Final text");
        assert_eq!(wasm.language(), "es");
        assert!(wasm.is_final());
        assert!(wasm.has_text());
        assert!((wasm.confidence() - 0.95).abs() < f32::EPSILON);
        assert!((wasm.duration_secs() - 5.0).abs() < f32::EPSILON);
        assert!((wasm.processing_time_secs() - 1.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Model Memory Info Extended Tests
    // =========================================================================

    #[test]
    fn test_model_memory_info_small() {
        let info = ModelMemoryInfo::for_model("small");
        assert!(info.is_some());
        let info = info.expect("small model info");
        assert_eq!(info.model_type(), "small");
        assert!(info.weights_mb() > 0.0);
    }

    #[test]
    fn test_model_memory_info_medium() {
        let info = ModelMemoryInfo::for_model("medium");
        assert!(info.is_some());
        let info = info.expect("medium model info");
        assert_eq!(info.model_type(), "medium");
        assert!(info.weights_mb() > 0.0);
    }

    #[test]
    fn test_model_memory_info_large() {
        let info = ModelMemoryInfo::for_model("large");
        assert!(info.is_some());
        let info = info.expect("large model info");
        assert_eq!(info.model_type(), "large");
        assert!(info.weights_mb() > 0.0);
    }

    #[test]
    fn test_model_memory_info_hierarchy() {
        let tiny = ModelMemoryInfo::for_model("tiny").expect("tiny");
        let base = ModelMemoryInfo::for_model("base").expect("base");
        let small = ModelMemoryInfo::for_model("small").expect("small");

        // Each larger model should have more parameters
        assert!(base.parameters() > tiny.parameters());
        assert!(small.parameters() > base.parameters());
    }

    // =========================================================================
    // StreamingEventWasm From Implementation Tests
    // =========================================================================

    #[test]
    fn test_streaming_event_wasm_speech_end() {
        use crate::audio::StreamingEvent;
        let event: StreamingEventWasm = StreamingEvent::SpeechEnd.into();
        assert_eq!(event.event_type(), StreamingEventTypeWasm::SpeechEnd);
    }

    #[test]
    fn test_streaming_event_wasm_chunk_ready() {
        use crate::audio::StreamingEvent;
        let event: StreamingEventWasm = StreamingEvent::ChunkReady { duration_secs: 10.0 }.into();
        assert_eq!(event.event_type(), StreamingEventTypeWasm::ChunkReady);
        assert!((event.duration_secs().unwrap_or(0.0) - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_streaming_event_wasm_processing_started() {
        use crate::audio::StreamingEvent;
        let event: StreamingEventWasm = StreamingEvent::ProcessingStarted.into();
        assert_eq!(event.event_type(), StreamingEventTypeWasm::ProcessingStarted);
    }

    #[test]
    fn test_streaming_event_wasm_processing_completed() {
        use crate::audio::StreamingEvent;
        let event: StreamingEventWasm = StreamingEvent::ProcessingCompleted.into();
        assert_eq!(event.event_type(), StreamingEventTypeWasm::ProcessingCompleted);
    }

    #[test]
    fn test_streaming_event_wasm_reset() {
        use crate::audio::StreamingEvent;
        let event: StreamingEventWasm = StreamingEvent::Reset.into();
        assert_eq!(event.event_type(), StreamingEventTypeWasm::Reset);
    }

    // =========================================================================
    // ProcessorStateWasm From Implementation Tests
    // =========================================================================

    #[test]
    fn test_processor_state_wasm_partial_result_ready() {
        use crate::audio::ProcessorState;
        let state: ProcessorStateWasm = ProcessorState::PartialResultReady.into();
        assert_eq!(state, ProcessorStateWasm::PartialResultReady);
    }

    #[test]
    fn test_processor_state_wasm_processing() {
        use crate::audio::ProcessorState;
        let state: ProcessorStateWasm = ProcessorState::Processing.into();
        assert_eq!(state, ProcessorStateWasm::Processing);
    }

    #[test]
    fn test_processor_state_wasm_error() {
        use crate::audio::ProcessorState;
        let state: ProcessorStateWasm = ProcessorState::Error.into();
        assert_eq!(state, ProcessorStateWasm::Error);
    }

    // =========================================================================
    // WhisperAprWasm Config Getter Tests
    // =========================================================================

    #[test]
    fn test_whisper_apr_wasm_can_run_with_memory() {
        let whisper = WhisperAprWasm::tiny();
        // Tiny model should be able to run with 512MB
        assert!(whisper.can_run_with_memory(512));
        // Should not be able to run with 10MB
        assert!(!whisper.can_run_with_memory(10));
    }

    #[test]
    fn test_whisper_apr_wasm_memory_summary() {
        let whisper = WhisperAprWasm::tiny();
        let summary = whisper.memory_summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("MB"));
    }

    #[test]
    fn test_whisper_apr_wasm_parameter_count() {
        let whisper = WhisperAprWasm::tiny();
        let params = whisper.parameter_count();
        // Tiny model has ~39M params
        assert!(params > 30_000_000);
    }

    #[test]
    fn test_whisper_apr_wasm_vocab_size() {
        let whisper = WhisperAprWasm::tiny();
        let vocab = whisper.vocab_size();
        // Whisper has 51865 tokens
        assert_eq!(vocab, 51865);
    }

    #[test]
    fn test_whisper_apr_wasm_context_lengths() {
        let whisper = WhisperAprWasm::tiny();
        assert!(whisper.audio_context_length() > 0);
        assert!(whisper.text_context_length() > 0);
    }

    #[test]
    fn test_whisper_apr_wasm_layer_counts() {
        let whisper = WhisperAprWasm::tiny();
        assert!(whisper.encoder_layer_count() > 0);
        assert!(whisper.decoder_layer_count() > 0);
    }

    #[test]
    fn test_model_memory_info_peak_mb() {
        let info = ModelMemoryInfo::for_model("tiny").expect("tiny model");
        assert!(info.peak_mb() > info.weights_mb());
    }

    #[test]
    fn test_model_memory_info_wasm_pages() {
        let info = ModelMemoryInfo::for_model("tiny").expect("tiny model");
        assert!(info.wasm_pages() > 0);
    }
}
