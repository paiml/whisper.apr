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
mod lfm2;
mod threading;
mod timestamps;
mod vocabulary;
mod worker;

pub use capabilities::{Capabilities, ExecutionMode};
pub use diarization::{
    get_diarization_recommendation, DiarizationConfigWasm, DiarizationResultWasm, DiarizerWasm,
    EmbeddingExtractorWasm, SpeakerEmbeddingWasm, SpeakerSegmentWasm, TurnDetectorWasm,
};
pub use gpu::{
    estimate_mat_mul_flops, estimate_mat_mul_memory, is_gpu_worthwhile,
    recommended_backend_for_model, BackendSelectionWasm, BackendSelectorWasm, BackendTypeWasm,
    DetectionOptionsWasm, GpuBackendWasm, GpuCapabilitiesWasm, GpuDetectionWasm, GpuLimitsWasm,
    SelectionStrategyWasm, SelectorConfigWasm,
};
pub use lfm2::{GenerationResultWasm, Lfm2Wasm, MemoryEstimateWasm, ViabilityCheckWasm};
#[cfg(feature = "parallel")]
pub use threading::init_thread_pool;
pub use threading::{
    get_threading_mode, get_threading_mode_name, is_threaded_available, optimal_thread_count,
    parallel_map, parallel_matmul, parallel_reduce, ThreadingMode,
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
pub use worker::{ProgressPhase, WorkerConfig, WorkerMessageType, WorkerProgress, WorkerState};

// Implementation and types
#[path = "impl_generated.rs"]
#[allow(clippy::all)]
mod impl_;

pub use impl_::*;
