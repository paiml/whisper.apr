//! THIS IS SPARTA! - Strict Streaming Validation Mode (WAPR-SPARTA)
//!
//! No mercy. No fallbacks. No excuses.
//! Tests either PASS or go INTO THE PIT.
#![allow(clippy::neg_cmp_op_on_partial_ord)] // Macros use negated comparisons for assertion logic
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    THIS IS SPARTA!                          │
//! │                         ⚔️                                   │
//! │    "Spartans! Ready your breakfast and eat hearty...       │
//! │     For tonight, we dine in hell!"                         │
//! │                                                             │
//! │    - King Leonidas, on running integration tests           │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use std::time::Duration;

/// Dramatic assertion macro - fails LOUD when condition is false
///
/// # Example
/// ```ignore
/// sparta!(caps.shared_array_buffer, "SharedArrayBuffer not available");
/// // On failure:
/// // 🔥 THIS IS SPARTA! 🔥
/// // SharedArrayBuffer not available
/// // ⚔️  INTO THE PIT! ⚔️
/// ```
#[macro_export]
macro_rules! sparta {
    ($cond:expr, $msg:expr) => {
        if !$cond {
            panic!(
                "\n\n\
                ╔═══════════════════════════════════════════════════════════╗\n\
                ║           🔥 THIS IS SPARTA! 🔥                           ║\n\
                ╠═══════════════════════════════════════════════════════════╣\n\
                ║                                                           ║\n\
                ║  {}  \n\
                ║                                                           ║\n\
                ╠═══════════════════════════════════════════════════════════╣\n\
                ║           ⚔️  INTO THE PIT! ⚔️                            ║\n\
                ╚═══════════════════════════════════════════════════════════╝\n",
                $msg
            );
        }
    };
    ($cond:expr) => {
        sparta!($cond, "Assertion failed - no mercy for bugs!");
    };
}

/// Dramatic assertion with formatted message
#[macro_export]
macro_rules! sparta_fmt {
    ($cond:expr, $($arg:tt)*) => {
        sparta!($cond, format!($($arg)*));
    };
}

/// SPARTA validator - strict thresholds, no exceptions
#[derive(Debug, Clone)]
pub struct SpartaValidator {
    /// Maximum allowed latency in milliseconds (default: 100ms)
    pub max_latency_ms: u64,
    /// Minimum required FPS (default: 30)
    pub min_fps: f64,
    /// Maximum allowed dropped frames (default: 0 - ZERO TOLERANCE)
    pub max_dropped_frames: usize,
    /// Required capabilities - ALL must be present
    pub require_shared_array_buffer: bool,
    pub require_cross_origin_isolated: bool,
    pub require_atomics: bool,
    /// Minimum CPU cores required
    pub min_cores: u32,
}

impl Default for SpartaValidator {
    fn default() -> Self {
        Self {
            max_latency_ms: 100,   // 100ms - THIS IS SPARTA, not a slideshow
            min_fps: 30.0,         // 30 FPS minimum - Spartans don't stutter
            max_dropped_frames: 0, // ZERO tolerance for dropped frames
            require_shared_array_buffer: true,
            require_cross_origin_isolated: true,
            require_atomics: true,
            min_cores: 2, // Need warriors, not a single soldier
        }
    }
}

impl SpartaValidator {
    /// Create a new SPARTA validator with maximum strictness
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Even stricter mode - for the bravest Spartans
    #[must_use]
    pub fn leonidas_mode() -> Self {
        Self {
            max_latency_ms: 50, // 50ms - Leonidas doesn't wait
            min_fps: 60.0,      // 60 FPS - smooth as a Spartan blade
            max_dropped_frames: 0,
            require_shared_array_buffer: true,
            require_cross_origin_isolated: true,
            require_atomics: true,
            min_cores: 4, // The 300 had many warriors
        }
    }

    /// Validate WASM threading capabilities
    pub fn validate_capabilities(&self, caps: &probar::capabilities::WasmThreadCapabilities) {
        if self.require_cross_origin_isolated {
            sparta!(
                caps.cross_origin_isolated,
                "crossOriginIsolated is FALSE! COOP/COEP headers missing!"
            );
        }

        if self.require_shared_array_buffer {
            sparta!(
                caps.shared_array_buffer,
                "SharedArrayBuffer not available! Cannot share memory between threads!"
            );
        }

        if self.require_atomics {
            sparta!(
                caps.atomics,
                "Atomics not available! Cannot synchronize threads!"
            );
        }

        sparta_fmt!(
            caps.hardware_concurrency >= self.min_cores,
            "Only {} cores available, need at least {}! Spartans fight in formation!",
            caps.hardware_concurrency,
            self.min_cores
        );
    }

    /// Validate streaming metrics
    pub fn validate_latency(&self, latency: Duration) {
        let latency_ms = latency.as_millis() as u64;
        sparta_fmt!(
            latency_ms <= self.max_latency_ms,
            "Latency {}ms exceeds maximum {}ms! Spartans are FAST!",
            latency_ms,
            self.max_latency_ms
        );
    }

    /// Validate FPS
    pub fn validate_fps(&self, fps: f64) {
        sparta_fmt!(
            fps >= self.min_fps,
            "FPS {:.1} below minimum {:.1}! Spartans don't stutter!",
            fps,
            self.min_fps
        );
    }

    /// Validate dropped frames (ZERO TOLERANCE)
    pub fn validate_dropped_frames(&self, dropped: usize) {
        sparta_fmt!(
            dropped <= self.max_dropped_frames,
            "Dropped {} frames! Maximum allowed: {}! Every frame matters in battle!",
            dropped,
            self.max_dropped_frames
        );
    }

    /// Validate state transition occurred
    pub fn validate_state_reached(&self, current: &str, expected: &str) {
        sparta_fmt!(
            current.contains(expected),
            "State '{}' not reached! Expected '{}' - NO RETREAT!",
            current,
            expected
        );
    }

    /// Full SPARTA validation - all or nothing
    pub fn validate_all(
        &self,
        caps: &probar::capabilities::WasmThreadCapabilities,
        latency: Option<Duration>,
        fps: Option<f64>,
        dropped_frames: Option<usize>,
    ) {
        // Capabilities - non-negotiable
        self.validate_capabilities(caps);

        // Latency - if provided
        if let Some(lat) = latency {
            self.validate_latency(lat);
        }

        // FPS - if provided
        if let Some(f) = fps {
            self.validate_fps(f);
        }

        // Dropped frames - ZERO TOLERANCE
        if let Some(dropped) = dropped_frames {
            self.validate_dropped_frames(dropped);
        }
    }
}

/// Battle cry for test output
pub fn battle_cry() {
    eprintln!(
        r#"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     ⚔️  THIS IS SPARTA! ⚔️                                    ║
║                                                               ║
║     "Spartans! What is your profession?"                      ║
║     "HA-OOH! HA-OOH! HA-OOH!"                                 ║
║                                                               ║
║     Running streaming UX tests with ZERO TOLERANCE...         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"#
    );
}

/// Victory message
pub fn victory() {
    eprintln!(
        r#"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     🏆 VICTORY! 🏆                                            ║
║                                                               ║
║     "Tonight, we dine in... the cafeteria!"                   ║
║     (All tests passed with SPARTA validation)                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"#
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparta_macro_passes() {
        sparta!(true, "This should pass");
        sparta!(1 + 1 == 2, "Math works");
        sparta!(true); // No message variant
    }

    #[test]
    #[should_panic(expected = "THIS IS SPARTA")]
    fn test_sparta_macro_fails_dramatically() {
        sparta!(false, "This fails INTO THE PIT");
    }

    #[test]
    #[should_panic(expected = "INTO THE PIT")]
    fn test_sparta_fmt_macro() {
        sparta_fmt!(false, "Value {} is wrong!", 42);
    }

    #[test]
    fn test_sparta_validator_defaults() {
        let v = SpartaValidator::new();
        assert_eq!(v.max_latency_ms, 100);
        assert!((v.min_fps - 30.0).abs() < f64::EPSILON);
        assert_eq!(v.max_dropped_frames, 0);
        assert!(v.require_shared_array_buffer);
        assert!(v.require_cross_origin_isolated);
        assert!(v.require_atomics);
        assert_eq!(v.min_cores, 2);
    }

    #[test]
    fn test_leonidas_mode_stricter() {
        let v = SpartaValidator::leonidas_mode();
        assert_eq!(v.max_latency_ms, 50); // Stricter than default
        assert!((v.min_fps - 60.0).abs() < f64::EPSILON);
        assert_eq!(v.min_cores, 4);
    }

    #[test]
    fn test_validate_latency_passes() {
        let v = SpartaValidator::new();
        v.validate_latency(Duration::from_millis(50)); // Under 100ms
        v.validate_latency(Duration::from_millis(100)); // Exactly 100ms
    }

    #[test]
    #[should_panic(expected = "Latency")]
    fn test_validate_latency_fails() {
        let v = SpartaValidator::new();
        v.validate_latency(Duration::from_millis(150)); // Over 100ms - INTO THE PIT
    }

    #[test]
    fn test_validate_fps_passes() {
        let v = SpartaValidator::new();
        v.validate_fps(30.0);
        v.validate_fps(60.0);
    }

    #[test]
    #[should_panic(expected = "FPS")]
    fn test_validate_fps_fails() {
        let v = SpartaValidator::new();
        v.validate_fps(20.0); // Below 30 - INTO THE PIT
    }

    #[test]
    fn test_validate_dropped_frames_zero_tolerance() {
        let v = SpartaValidator::new();
        v.validate_dropped_frames(0); // ZERO is acceptable
    }

    #[test]
    #[should_panic(expected = "Dropped")]
    fn test_validate_dropped_frames_fails() {
        let v = SpartaValidator::new();
        v.validate_dropped_frames(1); // Even ONE frame - INTO THE PIT
    }

    #[test]
    fn test_battle_cry_and_victory() {
        // These just print, verify they don't panic
        battle_cry();
        victory();
    }
}
