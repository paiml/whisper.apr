//! Whisper TUI Application State
//!
//! Manages the application state for the pipeline visualization dashboard.
//! Follows the state machine defined in WAPR-TUI-001.

// MelFilterbank is available via crate::audio but not needed for TUI state

/// Active panel in the whisper dashboard
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WhisperPanel {
    /// Raw audio waveform visualization
    #[default]
    Waveform,
    /// Mel spectrogram heatmap
    Mel,
    /// Encoder layer activations
    Encoder,
    /// Decoder token generation
    Decoder,
    /// Cross-attention weights
    Attention,
    /// Final transcription output
    Transcription,
    /// Performance metrics
    Metrics,
    /// Help and keyboard bindings
    Help,
}

impl WhisperPanel {
    /// Get panel titles for tab bar
    pub fn titles() -> Vec<&'static str> {
        vec![
            "Waveform [1]",
            "Mel [2]",
            "Encoder [3]",
            "Decoder [4]",
            "Attention [5]",
            "Transcription [6]",
            "Metrics [7]",
            "Help [?]",
        ]
    }

    /// Get panel index
    pub fn index(self) -> usize {
        match self {
            Self::Waveform => 0,
            Self::Mel => 1,
            Self::Encoder => 2,
            Self::Decoder => 3,
            Self::Attention => 4,
            Self::Transcription => 5,
            Self::Metrics => 6,
            Self::Help => 7,
        }
    }

    /// Create from index
    pub fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Waveform,
            1 => Self::Mel,
            2 => Self::Encoder,
            3 => Self::Decoder,
            4 => Self::Attention,
            5 => Self::Transcription,
            6 => Self::Metrics,
            _ => Self::Help,
        }
    }
}

/// Pipeline state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WhisperState {
    /// No audio loaded
    #[default]
    Idle,
    /// Audio loaded, waveform ready
    WaveformReady,
    /// Mel spectrogram computed
    MelReady,
    /// Encoder processing
    Encoding,
    /// Decoder generating tokens
    Decoding,
    /// Transcription complete
    Complete,
    /// Error state
    Error,
}

/// Encoder layer metrics for visualization
#[derive(Debug, Clone, Default)]
pub struct EncoderLayerMetrics {
    /// Layer index
    pub layer: usize,
    /// Mean activation magnitude
    pub mean_activation: f32,
    /// Max activation magnitude
    pub max_activation: f32,
    /// Self-attention entropy
    pub attention_entropy: f32,
}

/// Decoder token for visualization
#[derive(Debug, Clone)]
pub struct DecoderToken {
    /// Token ID
    pub id: u32,
    /// Token text
    pub text: String,
    /// Log probability
    pub log_prob: f32,
    /// Cross-attention weights (to audio frames)
    pub attention_weights: Vec<f32>,
}

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct PipelineMetrics {
    /// Audio duration in seconds
    pub audio_duration_secs: f32,
    /// Mel computation time in ms
    pub mel_time_ms: f32,
    /// Encoder time in ms
    pub encoder_time_ms: f32,
    /// Decoder time in ms
    pub decoder_time_ms: f32,
    /// Total processing time in ms
    pub total_time_ms: f32,
    /// Real-time factor
    pub rtf: f32,
    /// Tokens generated
    pub tokens_generated: usize,
    /// Memory used in bytes
    pub memory_bytes: usize,
}

impl PipelineMetrics {
    /// Compute RTF from audio duration and processing time
    pub fn compute_rtf(&mut self) {
        if self.audio_duration_secs > 0.0 {
            self.rtf = (self.total_time_ms / 1000.0) / self.audio_duration_secs;
        }
    }
}

/// Whisper dashboard application state
#[derive(Debug, Clone)]
pub struct WhisperApp {
    /// Current active panel
    pub current_panel: WhisperPanel,
    /// Pipeline state
    pub state: WhisperState,
    /// Should quit flag
    pub should_quit: bool,
    /// Paused flag
    pub paused: bool,
    /// Raw audio data
    pub audio_data: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Mel spectrogram data (80 bins x N frames)
    pub mel_data: Vec<f32>,
    /// Number of mel frames
    pub mel_frames: usize,
    /// Encoder layer metrics
    pub encoder_metrics: Vec<EncoderLayerMetrics>,
    /// Decoder tokens
    pub decoder_tokens: Vec<DecoderToken>,
    /// Cross-attention weights (tokens x frames)
    pub attention_weights: Vec<Vec<f32>>,
    /// Final transcription
    pub transcription: String,
    /// Performance metrics
    pub metrics: PipelineMetrics,
    /// Error message (if in error state)
    pub error_message: Option<String>,
    /// Status message
    pub status_message: Option<String>,
    /// Horizontal scroll position for panels
    pub scroll_x: usize,
    /// Vertical scroll position for panels
    pub scroll_y: usize,
    /// Selected layer for encoder view
    pub selected_layer: usize,
}

impl Default for WhisperApp {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperApp {
    /// Create new application state
    pub fn new() -> Self {
        Self {
            current_panel: WhisperPanel::Waveform,
            state: WhisperState::Idle,
            should_quit: false,
            paused: false,
            audio_data: Vec::new(),
            sample_rate: 16000,
            mel_data: Vec::new(),
            mel_frames: 0,
            encoder_metrics: Vec::new(),
            decoder_tokens: Vec::new(),
            attention_weights: Vec::new(),
            transcription: String::new(),
            metrics: PipelineMetrics::default(),
            error_message: None,
            status_message: None,
            scroll_x: 0,
            scroll_y: 0,
            selected_layer: 0,
        }
    }

    /// Set current panel
    pub fn set_panel(&mut self, panel: WhisperPanel) {
        self.current_panel = panel;
    }

    /// Handle keyboard input
    pub fn handle_key(&mut self, key: char) {
        match key {
            '1' => self.current_panel = WhisperPanel::Waveform,
            '2' => self.current_panel = WhisperPanel::Mel,
            '3' => self.current_panel = WhisperPanel::Encoder,
            '4' => self.current_panel = WhisperPanel::Decoder,
            '5' => self.current_panel = WhisperPanel::Attention,
            '6' => self.current_panel = WhisperPanel::Transcription,
            '7' => self.current_panel = WhisperPanel::Metrics,
            '?' => self.current_panel = WhisperPanel::Help,
            ' ' => self.paused = !self.paused,
            'r' => self.reset(),
            'q' => self.should_quit = true,
            _ => {}
        }
    }

    /// Reset to initial state
    #[allow(clippy::no_effect_underscore_binding)]
    pub fn reset(&mut self) {
        let _span = crate::trace_enter!("tui.reset");
        self.state = WhisperState::Idle;
        self.audio_data.clear();
        self.mel_data.clear();
        self.mel_frames = 0;
        self.encoder_metrics.clear();
        self.decoder_tokens.clear();
        self.attention_weights.clear();
        self.transcription.clear();
        self.metrics = PipelineMetrics::default();
        self.error_message = None;
        self.status_message = Some("Reset to idle".to_string());
        self.scroll_x = 0;
        self.scroll_y = 0;
        self.selected_layer = 0;
    }

    /// Load audio data
    #[allow(clippy::no_effect_underscore_binding)]
    pub fn load_audio(&mut self, audio: &[f32]) {
        let _span = crate::trace_enter!("tui.load_audio");
        self.audio_data = audio.to_vec();
        self.metrics.audio_duration_secs = audio.len() as f32 / self.sample_rate as f32;
        self.state = WhisperState::WaveformReady;
        self.status_message = Some(format!(
            "Loaded {} samples ({:.2}s)",
            audio.len(),
            self.metrics.audio_duration_secs
        ));
    }

    /// Compute mel spectrogram (mock for TUI testing)
    #[allow(clippy::no_effect_underscore_binding)]
    pub fn compute_mel(&mut self) {
        const N_MELS: usize = 80;
        const HOP_LENGTH: usize = 160; // 10ms at 16kHz

        let _span = crate::trace_enter!("tui.compute_mel");
        if self.audio_data.is_empty() {
            self.error_message = Some("No audio loaded".to_string());
            self.state = WhisperState::Error;
            return;
        }

        let start = std::time::Instant::now();

        // Mock mel computation - in real impl would use MelFilterBank
        let n_frames = (self.audio_data.len() / HOP_LENGTH).max(1);

        // Generate mock mel data
        self.mel_data = vec![0.0; N_MELS * n_frames];
        for frame in 0..n_frames {
            for mel_bin in 0..N_MELS {
                // Create a pattern that looks like real mel spectrogram
                let frame_energy = self.audio_data
                    .get(frame * HOP_LENGTH..(frame + 1) * HOP_LENGTH)
                    .map_or(0.0, |s| s.iter().map(|x| x.powi(2)).sum::<f32>());
                let log_energy = (frame_energy + 1e-10).ln();
                // Lower frequencies have more energy
                let freq_weight = 1.0 - (mel_bin as f32 / N_MELS as f32);
                self.mel_data[frame * N_MELS + mel_bin] = log_energy * freq_weight;
            }
        }
        self.mel_frames = n_frames;

        self.metrics.mel_time_ms = start.elapsed().as_secs_f32() * 1000.0;
        self.state = WhisperState::MelReady;
        self.status_message = Some(format!(
            "Computed {} mel frames in {:.2}ms",
            n_frames, self.metrics.mel_time_ms
        ));
    }

    /// Start encoding (mock for TUI)
    #[allow(clippy::no_effect_underscore_binding)]
    pub fn start_encoding(&mut self) {
        let _span = crate::trace_enter!("tui.start_encoding");
        if self.state != WhisperState::MelReady {
            return;
        }

        let start = std::time::Instant::now();

        // Mock encoder layers (tiny model has 4 layers)
        self.encoder_metrics = (0..4)
            .map(|layer| EncoderLayerMetrics {
                layer,
                mean_activation: 0.5 + (layer as f32 * 0.1),
                max_activation: 2.0 + (layer as f32 * 0.2),
                attention_entropy: 3.5 - (layer as f32 * 0.3),
            })
            .collect();

        self.metrics.encoder_time_ms = start.elapsed().as_secs_f32() * 1000.0;
        self.state = WhisperState::Encoding;
        self.status_message = Some(format!(
            "Encoded through {} layers in {:.2}ms",
            self.encoder_metrics.len(),
            self.metrics.encoder_time_ms
        ));
    }

    /// Start decoding (mock for TUI)
    #[allow(clippy::no_effect_underscore_binding)]
    pub fn start_decoding(&mut self) {
        let _span = crate::trace_enter!("tui.start_decoding");
        if self.state != WhisperState::Encoding {
            return;
        }

        let start = std::time::Instant::now();

        // Mock decoder tokens
        let sample_tokens = ["<|startoftranscript|>", "<|en|>", "Hello", ",", " world", ".", "<|endoftext|>"];

        self.decoder_tokens = sample_tokens
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let attention = (0..self.mel_frames.max(10))
                    .map(|f| {
                        // Peak attention around expected position
                        let peak = (i as f32 / sample_tokens.len() as f32) * self.mel_frames as f32;
                        let dist = (f as f32 - peak).abs();
                        (-dist / 10.0).exp()
                    })
                    .collect();

                DecoderToken {
                    id: i as u32 + 50000,
                    text: text.to_string(),
                    log_prob: -0.1 - (i as f32 * 0.05),
                    attention_weights: attention,
                }
            })
            .collect();

        // Build attention weights matrix
        self.attention_weights = self.decoder_tokens
            .iter()
            .map(|t| t.attention_weights.clone())
            .collect();

        self.metrics.decoder_time_ms = start.elapsed().as_secs_f32() * 1000.0;
        self.metrics.tokens_generated = self.decoder_tokens.len();
        self.state = WhisperState::Decoding;
        self.status_message = Some(format!(
            "Generated {} tokens in {:.2}ms",
            self.decoder_tokens.len(),
            self.metrics.decoder_time_ms
        ));
    }

    /// Complete transcription
    #[allow(clippy::no_effect_underscore_binding)]
    pub fn complete(&mut self) {
        let _span = crate::trace_enter!("tui.complete");
        if self.state != WhisperState::Decoding {
            return;
        }

        // Build transcription from tokens
        self.transcription = self.decoder_tokens
            .iter()
            .filter(|t| !t.text.starts_with("<|"))
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join("");

        // Compute total time and RTF
        self.metrics.total_time_ms = self.metrics.mel_time_ms
            + self.metrics.encoder_time_ms
            + self.metrics.decoder_time_ms;
        self.metrics.compute_rtf();

        self.state = WhisperState::Complete;
        self.status_message = Some(format!(
            "Complete: '{}' (RTF: {:.2}x)",
            self.transcription.trim(),
            self.metrics.rtf
        ));
    }

    /// Get state description
    pub fn state_description(&self) -> &'static str {
        match self.state {
            WhisperState::Idle => "Idle - Load audio to begin",
            WhisperState::WaveformReady => "Waveform ready - Compute mel spectrogram",
            WhisperState::MelReady => "Mel ready - Start encoding",
            WhisperState::Encoding => "Encoding audio features",
            WhisperState::Decoding => "Decoding to text",
            WhisperState::Complete => "Transcription complete",
            WhisperState::Error => "Error occurred",
        }
    }
}
