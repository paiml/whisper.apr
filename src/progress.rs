//! Progress tracking and callbacks
//!
//! Provides progress reporting for long-running operations like model loading.
//!
//! # Usage
//!
//! ```rust,ignore
//! use whisper_apr::progress::{Progress, ProgressCallback};
//!
//! let callback = |progress: &Progress| {
//!     println!("{}% - {}", progress.percent(), progress.message());
//! };
//!
//! loader.load_with_progress(&data, callback)?;
//! ```

/// Progress information for a long-running operation
#[derive(Debug, Clone)]
pub struct Progress {
    /// Current step (0-indexed)
    pub current: usize,
    /// Total number of steps
    pub total: usize,
    /// Bytes processed (if applicable)
    pub bytes_loaded: usize,
    /// Total bytes (if known)
    pub bytes_total: Option<usize>,
    /// Current phase/stage name
    pub phase: String,
    /// Human-readable message
    pub message: String,
}

impl Progress {
    /// Create a new progress instance
    #[must_use]
    pub fn new(current: usize, total: usize) -> Self {
        Self {
            current,
            total,
            bytes_loaded: 0,
            bytes_total: None,
            phase: String::new(),
            message: String::new(),
        }
    }

    /// Create progress with phase and message
    #[must_use]
    pub fn with_phase(current: usize, total: usize, phase: impl Into<String>) -> Self {
        let phase = phase.into();
        Self {
            current,
            total,
            bytes_loaded: 0,
            bytes_total: None,
            message: phase.clone(),
            phase,
        }
    }

    /// Create progress for byte-based operations
    #[must_use]
    pub fn bytes(loaded: usize, total: usize) -> Self {
        Self {
            current: loaded,
            total,
            bytes_loaded: loaded,
            bytes_total: Some(total),
            phase: "Loading".to_string(),
            message: format!("{loaded}/{total} bytes"),
        }
    }

    /// Set the phase
    #[must_use]
    pub fn phase(mut self, phase: impl Into<String>) -> Self {
        self.phase = phase.into();
        self
    }

    /// Set the message
    #[must_use]
    pub fn message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    /// Set bytes info
    #[must_use]
    pub fn with_bytes(mut self, loaded: usize, total: usize) -> Self {
        self.bytes_loaded = loaded;
        self.bytes_total = Some(total);
        self
    }

    /// Get progress percentage (0.0 to 100.0)
    #[must_use]
    pub fn percent(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.current as f32 / self.total as f32) * 100.0
        }
    }

    /// Get normalized progress (0.0 to 1.0)
    #[must_use]
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.current as f32 / self.total as f32
        }
    }

    /// Check if operation is complete
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.current >= self.total
    }

    /// Get human-readable message
    #[must_use]
    pub fn display_message(&self) -> &str {
        &self.message
    }

    /// Get bytes progress as string (e.g., "1.5 MB / 10 MB")
    #[must_use]
    pub fn bytes_display(&self) -> String {
        self.bytes_total.map_or_else(
            || format_bytes(self.bytes_loaded),
            |total| {
                format!(
                    "{} / {}",
                    format_bytes(self.bytes_loaded),
                    format_bytes(total)
                )
            },
        )
    }
}

impl Default for Progress {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

/// Callback type for progress updates
pub type ProgressCallback<'a> = &'a mut dyn FnMut(&Progress);

/// Boxed progress callback for owned callbacks
pub type BoxedProgressCallback = Box<dyn FnMut(&Progress) + Send>;

/// Progress tracker for multi-phase operations
#[derive(Debug, Clone)]
pub struct ProgressTracker {
    phases: Vec<Phase>,
    current_phase: usize,
    overall_progress: f32,
}

/// A phase in a multi-phase operation
#[derive(Debug, Clone)]
struct Phase {
    name: String,
    weight: f32,
    completed: bool,
}

impl ProgressTracker {
    /// Create a new progress tracker with phases and their weights
    ///
    /// Weights should sum to 1.0 for accurate overall progress.
    #[must_use]
    pub fn new(phases: Vec<(&str, f32)>) -> Self {
        let phases = phases
            .into_iter()
            .map(|(name, weight)| Phase {
                name: name.to_string(),
                weight,
                completed: false,
            })
            .collect();

        Self {
            phases,
            current_phase: 0,
            overall_progress: 0.0,
        }
    }

    /// Create tracker for model loading phases
    #[must_use]
    pub fn model_loading() -> Self {
        Self::new(vec![
            ("Parsing header", 0.05),
            ("Loading encoder", 0.35),
            ("Loading decoder", 0.35),
            ("Loading vocabulary", 0.15),
            ("Initializing", 0.10),
        ])
    }

    /// Create tracker for transcription phases
    #[must_use]
    pub fn transcription() -> Self {
        Self::new(vec![
            ("Preprocessing audio", 0.10),
            ("Computing mel spectrogram", 0.15),
            ("Encoding audio", 0.25),
            ("Decoding tokens", 0.40),
            ("Post-processing", 0.10),
        ])
    }

    /// Update progress within current phase
    pub fn update_phase_progress(&mut self, fraction: f32) {
        if self.current_phase < self.phases.len() {
            let completed_weight: f32 = self.phases[..self.current_phase]
                .iter()
                .map(|p| p.weight)
                .sum();

            let current_weight = self.phases[self.current_phase].weight;
            self.overall_progress =
                current_weight.mul_add(fraction.clamp(0.0, 1.0), completed_weight);
        }
    }

    /// Move to next phase
    pub fn next_phase(&mut self) {
        if self.current_phase < self.phases.len() {
            self.phases[self.current_phase].completed = true;
            self.current_phase += 1;

            let completed_weight: f32 = self.phases[..self.current_phase]
                .iter()
                .map(|p| p.weight)
                .sum();
            self.overall_progress = completed_weight;
        }
    }

    /// Complete all phases
    pub fn complete(&mut self) {
        for phase in &mut self.phases {
            phase.completed = true;
        }
        self.current_phase = self.phases.len();
        self.overall_progress = 1.0;
    }

    /// Get current phase name
    #[must_use]
    pub fn current_phase_name(&self) -> Option<&str> {
        self.phases.get(self.current_phase).map(|p| p.name.as_str())
    }

    /// Get overall progress (0.0 to 1.0)
    #[must_use]
    pub fn overall_progress(&self) -> f32 {
        self.overall_progress
    }

    /// Get overall progress as percentage
    #[must_use]
    pub fn overall_percent(&self) -> f32 {
        self.overall_progress * 100.0
    }

    /// Check if all phases are complete
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.current_phase >= self.phases.len()
    }

    /// Build a Progress struct for current state
    #[must_use]
    pub fn to_progress(&self) -> Progress {
        let phase_name = self.current_phase_name().unwrap_or("Complete").to_string();
        Progress {
            current: self.current_phase,
            total: self.phases.len(),
            bytes_loaded: 0,
            bytes_total: None,
            phase: phase_name.clone(),
            message: format!("{} ({:.0}%)", phase_name, self.overall_progress * 100.0),
        }
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::model_loading()
    }
}

/// Format bytes as human-readable string
#[must_use]
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// No-op progress callback (for when progress isn't needed)
pub fn null_callback(_progress: &Progress) {}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Progress Tests
    // =========================================================================

    #[test]
    fn test_progress_new() {
        let progress = Progress::new(5, 10);
        assert_eq!(progress.current, 5);
        assert_eq!(progress.total, 10);
        assert!((progress.percent() - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_progress_with_phase() {
        let progress = Progress::with_phase(3, 10, "Loading");
        assert_eq!(progress.phase, "Loading");
        assert_eq!(progress.message, "Loading");
    }

    #[test]
    fn test_progress_bytes() {
        let progress = Progress::bytes(1024, 2048);
        assert_eq!(progress.bytes_loaded, 1024);
        assert_eq!(progress.bytes_total, Some(2048));
    }

    #[test]
    fn test_progress_percent() {
        assert!((Progress::new(0, 10).percent() - 0.0).abs() < f32::EPSILON);
        assert!((Progress::new(5, 10).percent() - 50.0).abs() < f32::EPSILON);
        assert!((Progress::new(10, 10).percent() - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_progress_percent_zero_total() {
        let progress = Progress::new(5, 0);
        assert!((progress.percent() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_progress_fraction() {
        assert!((Progress::new(5, 10).fraction() - 0.5).abs() < f32::EPSILON);
        assert!((Progress::new(0, 10).fraction() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_progress_is_complete() {
        assert!(!Progress::new(5, 10).is_complete());
        assert!(Progress::new(10, 10).is_complete());
        assert!(Progress::new(15, 10).is_complete());
    }

    #[test]
    fn test_progress_with_bytes() {
        let progress = Progress::new(5, 10).with_bytes(1000, 5000);
        assert_eq!(progress.bytes_loaded, 1000);
        assert_eq!(progress.bytes_total, Some(5000));
    }

    #[test]
    fn test_progress_bytes_display() {
        let progress = Progress::bytes(1024 * 1024, 10 * 1024 * 1024);
        let display = progress.bytes_display();
        assert!(display.contains("MB"));
    }

    #[test]
    fn test_progress_default() {
        let progress = Progress::default();
        assert_eq!(progress.current, 0);
        assert_eq!(progress.total, 0);
    }

    // =========================================================================
    // ProgressTracker Tests
    // =========================================================================

    #[test]
    fn test_progress_tracker_new() {
        let tracker = ProgressTracker::new(vec![("Phase1", 0.5), ("Phase2", 0.5)]);
        assert_eq!(tracker.current_phase_name(), Some("Phase1"));
        assert!((tracker.overall_progress() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_progress_tracker_model_loading() {
        let tracker = ProgressTracker::model_loading();
        assert_eq!(tracker.current_phase_name(), Some("Parsing header"));
    }

    #[test]
    fn test_progress_tracker_transcription() {
        let tracker = ProgressTracker::transcription();
        assert_eq!(tracker.current_phase_name(), Some("Preprocessing audio"));
    }

    #[test]
    fn test_progress_tracker_update_phase_progress() {
        let mut tracker = ProgressTracker::new(vec![("Phase1", 0.5), ("Phase2", 0.5)]);

        tracker.update_phase_progress(0.5);
        assert!((tracker.overall_progress() - 0.25).abs() < 0.01);

        tracker.update_phase_progress(1.0);
        assert!((tracker.overall_progress() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_progress_tracker_next_phase() {
        let mut tracker = ProgressTracker::new(vec![("Phase1", 0.3), ("Phase2", 0.7)]);

        tracker.next_phase();
        assert_eq!(tracker.current_phase_name(), Some("Phase2"));
        assert!((tracker.overall_progress() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_progress_tracker_complete() {
        let mut tracker = ProgressTracker::new(vec![("Phase1", 0.5), ("Phase2", 0.5)]);

        tracker.complete();
        assert!(tracker.is_complete());
        assert!((tracker.overall_progress() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_progress_tracker_to_progress() {
        let tracker = ProgressTracker::new(vec![("Loading", 1.0)]);
        let progress = tracker.to_progress();

        assert_eq!(progress.phase, "Loading");
        assert_eq!(progress.total, 1);
    }

    #[test]
    fn test_progress_tracker_default() {
        let tracker = ProgressTracker::default();
        assert_eq!(tracker.current_phase_name(), Some("Parsing header"));
    }

    // =========================================================================
    // Format Bytes Tests
    // =========================================================================

    #[test]
    fn test_format_bytes_bytes() {
        assert_eq!(format_bytes(500), "500 B");
    }

    #[test]
    fn test_format_bytes_kb() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(2048), "2.00 KB");
    }

    #[test]
    fn test_format_bytes_mb() {
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(10 * 1024 * 1024), "10.00 MB");
    }

    #[test]
    fn test_format_bytes_gb() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    // =========================================================================
    // Callback Tests
    // =========================================================================

    #[test]
    fn test_null_callback() {
        let progress = Progress::new(5, 10);
        null_callback(&progress); // Should not panic
    }

    #[test]
    fn test_callback_invocation() {
        let mut called = false;
        let progress = Progress::new(5, 10);

        let mut callback = |_p: &Progress| {
            called = true;
        };

        callback(&progress);
        assert!(called);
    }

    #[test]
    fn test_progress_message_builder() {
        let progress = Progress::new(5, 10)
            .phase("Loading")
            .message("Loading model weights...");

        assert_eq!(progress.phase, "Loading");
        assert_eq!(progress.message, "Loading model weights...");
    }
}
