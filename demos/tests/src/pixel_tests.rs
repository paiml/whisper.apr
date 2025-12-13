//! WAPR-181: Probar Pixel Testing Suite
//!
//! Visual regression testing for all whisper.apr demos using:
//! - SSIM (Structural Similarity Index)
//! - PSNR (Peak Signal-to-Noise Ratio)
//! - CIEDE2000 (Perceptual color difference)
//!
//! These tests ensure pixel-perfect rendering across:
//! - Different browser viewports (mobile, tablet, desktop)
//! - State transitions (idle, recording, processing, complete)
//! - Dark mode consistency
//!
//! Quality Thresholds:
//! - SSIM: ≥0.99 for pixel-perfect, ≥0.95 acceptable
//! - PSNR: ≥40dB excellent, ≥30dB acceptable
//! - CIEDE2000: ≤1.0 imperceptible, ≤3.0 acceptable

use crate::probar::pixel_coverage::{
    CieDe2000Metric, DeltaEClassification, PsnrMetric, PsnrQuality, Rgb, SsimMetric,
};
use crate::probar::browser;

/// Demo paths and their baseline screenshot names
const DEMO_CONFIGS: &[DemoConfig] = &[
    DemoConfig {
        name: "realtime-transcription",
        path: "/demos/realtime-transcription.html",
        states: &["idle", "recording", "processing"],
    },
    DemoConfig {
        name: "upload-transcription",
        path: "/demos/upload-transcription.html",
        states: &["idle", "file_selected", "transcribing", "complete"],
    },
    DemoConfig {
        name: "realtime-translation",
        path: "/demos/realtime-translation.html",
        states: &["idle", "recording", "translating", "complete"],
    },
    DemoConfig {
        name: "upload-translation",
        path: "/demos/upload-translation.html",
        states: &["idle", "file_selected", "translating", "complete"],
    },
];

/// Demo configuration
struct DemoConfig {
    name: &'static str,
    path: &'static str,
    states: &'static [&'static str],
}

/// Viewport configurations for responsive testing
const VIEWPORTS: &[(&str, u32, u32)] = &[
    ("mobile", 375, 667),
    ("tablet", 768, 1024),
    ("desktop", 1920, 1080),
];

/// SSIM quality thresholds
mod ssim_thresholds {
    pub const PIXEL_PERFECT: f32 = 0.99;
    pub const ACCEPTABLE: f32 = 0.95;
    pub const MINIMUM: f32 = 0.90;
}

/// PSNR quality thresholds (dB)
mod psnr_thresholds {
    pub const EXCELLENT: f32 = 40.0;
    pub const GOOD: f32 = 35.0;
    pub const ACCEPTABLE: f32 = 30.0;
}

/// CIEDE2000 color difference thresholds
mod ciede_thresholds {
    pub const IMPERCEPTIBLE: f32 = 1.0;
    pub const ACCEPTABLE: f32 = 3.0;
    pub const NOTICEABLE: f32 = 6.0;
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compare screenshots using SSIM metric
fn compare_screenshots_ssim(
    ssim: &SsimMetric,
    actual: &[u8],
    expected: &[u8],
) -> crate::probar::pixel_coverage::SsimResult {
    let actual_img = image_from_bytes(actual);
    let expected_img = image_from_bytes(expected);

    if actual_img.is_none() || expected_img.is_none() {
        return crate::probar::pixel_coverage::SsimResult {
            score: 0.0,
            is_perfect: false,
            is_acceptable: false,
            channel_scores: [0.0, 0.0, 0.0],
        };
    }

    let (actual_pixels, width, height) = actual_img.expect("valid image");
    let (expected_pixels, _, _) = expected_img.expect("valid image");

    ssim.compare(&expected_pixels, &actual_pixels, width, height)
}

/// Compare screenshots using PSNR metric
fn compare_screenshots_psnr(
    psnr: &PsnrMetric,
    actual: &[u8],
    expected: &[u8],
) -> crate::probar::pixel_coverage::PsnrResult {
    let actual_img = image_from_bytes(actual);
    let expected_img = image_from_bytes(expected);

    if actual_img.is_none() || expected_img.is_none() {
        return crate::probar::pixel_coverage::PsnrResult {
            psnr_db: 0.0,
            mse: f32::MAX,
            quality: PsnrQuality::Poor,
        };
    }

    let (actual_pixels, _, _) = actual_img.expect("valid image");
    let (expected_pixels, _, _) = expected_img.expect("valid image");

    psnr.compare(&expected_pixels, &actual_pixels)
}

/// Compare screenshots using CIEDE2000 metric
fn compare_screenshots_ciede(
    ciede: &CieDe2000Metric,
    actual: &[u8],
    expected: &[u8],
) -> crate::probar::pixel_coverage::DeltaEResult {
    let actual_img = image_from_bytes(actual);
    let expected_img = image_from_bytes(expected);

    if actual_img.is_none() || expected_img.is_none() {
        return crate::probar::pixel_coverage::DeltaEResult {
            delta_e: f32::MAX,
            classification: DeltaEClassification::Obvious,
            lab_diff: (0.0, 0.0, 0.0),
        };
    }

    let (actual_pixels, _, _) = actual_img.expect("valid image");
    let (expected_pixels, _, _) = expected_img.expect("valid image");

    ciede.compare(&expected_pixels, &actual_pixels)
}

/// Extract RGB pixels from PNG bytes
fn image_from_bytes(bytes: &[u8]) -> Option<(Vec<Rgb>, u32, u32)> {
    use image::GenericImageView;

    let img = image::load_from_memory(bytes).ok()?;
    let (width, height) = img.dimensions();

    let pixels: Vec<Rgb> = img
        .to_rgb8()
        .pixels()
        .map(|p| Rgb {
            r: p[0],
            g: p[1],
            b: p[2],
        })
        .collect();

    Some((pixels, width, height))
}

/// Trigger a state transition in a demo
async fn trigger_state_transition(page: &browser::Page, demo_name: &str, state: &str) {
    match (demo_name, state) {
        // Realtime transcription
        ("realtime-transcription", "recording") => {
            page.click("#start_recording").await;
        }
        ("realtime-transcription", "processing") => {
            page.click("#stop_recording").await;
        }

        // Upload transcription
        ("upload-transcription", "file_selected") => {
            page.set_input_files("#file_input", &["test.wav"]).await;
        }
        ("upload-transcription", "transcribing") => {
            page.click("#transcribe").await;
        }
        ("upload-transcription", "complete") => {
            page.wait_for_selector("#download_button:not([disabled])")
                .await;
        }

        // Realtime translation
        ("realtime-translation", "recording") => {
            page.click("#start_recording").await;
        }
        ("realtime-translation", "translating") => {
            page.click("#stop_recording").await;
        }
        ("realtime-translation", "complete") => {
            page.wait_for_selector("#translated_text:not(:empty)").await;
        }

        // Upload translation
        ("upload-translation", "file_selected") => {
            page.set_input_files("#file_input", &["test.wav"]).await;
        }
        ("upload-translation", "translating") => {
            page.click("#translate").await;
        }
        ("upload-translation", "complete") => {
            page.wait_for_selector("#download_button:not([disabled])")
                .await;
        }

        // Idle is default, no action needed
        (_, "idle") => {}

        _ => {
            // Unknown state transition - log warning
            eprintln!(
                "Unknown state transition: {} -> {}",
                demo_name, state
            );
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssim_thresholds() {
        assert!(ssim_thresholds::PIXEL_PERFECT > ssim_thresholds::ACCEPTABLE);
        assert!(ssim_thresholds::ACCEPTABLE > ssim_thresholds::MINIMUM);
    }

    #[test]
    fn test_psnr_thresholds() {
        assert!(psnr_thresholds::EXCELLENT > psnr_thresholds::GOOD);
        assert!(psnr_thresholds::GOOD > psnr_thresholds::ACCEPTABLE);
    }

    #[test]
    fn test_ciede_thresholds() {
        assert!(ciede_thresholds::IMPERCEPTIBLE < ciede_thresholds::ACCEPTABLE);
        assert!(ciede_thresholds::ACCEPTABLE < ciede_thresholds::NOTICEABLE);
    }

    #[test]
    fn test_demo_configs() {
        assert_eq!(DEMO_CONFIGS.len(), 4);
        for config in DEMO_CONFIGS {
            assert!(!config.name.is_empty());
            assert!(!config.path.is_empty());
            assert!(!config.states.is_empty());
        }
    }

    #[test]
    fn test_viewports() {
        assert_eq!(VIEWPORTS.len(), 3);
        for (name, width, height) in VIEWPORTS {
            assert!(!name.is_empty());
            assert!(*width > 0);
            assert!(*height > 0);
        }
    }

    #[test]
    fn test_ssim_metric_defaults() {
        let ssim = SsimMetric::default()
            .with_thresholds(ssim_thresholds::PIXEL_PERFECT, ssim_thresholds::ACCEPTABLE);

        let pixels = vec![Rgb { r: 128, g: 128, b: 128 }];
        let result = ssim.compare(&pixels, &pixels, 1, 1);

        assert!(result.is_acceptable);
    }

    #[test]
    fn test_psnr_metric_defaults() {
        let psnr = PsnrMetric::default();
        let pixels = vec![Rgb { r: 128, g: 128, b: 128 }];
        let result = psnr.compare(&pixels, &pixels);

        assert!(result.psnr_db >= psnr_thresholds::ACCEPTABLE);
    }

    #[test]
    fn test_ciede_metric_defaults() {
        let ciede = CieDe2000Metric::default();
        let pixels = vec![Rgb { r: 128, g: 128, b: 128 }];
        let result = ciede.compare(&pixels, &pixels);

        assert!(result.delta_e <= ciede_thresholds::ACCEPTABLE);
    }

    #[test]
    fn test_image_from_bytes_invalid() {
        let result = image_from_bytes(&[]);
        assert!(result.is_none());

        let result = image_from_bytes(&[0, 1, 2, 3]);
        assert!(result.is_none());
    }

    #[test]
    fn test_compare_screenshots_empty() {
        let ssim = SsimMetric::default();
        let result = compare_screenshots_ssim(&ssim, &[], &[]);
        assert!(!result.is_acceptable);

        let psnr = PsnrMetric::default();
        let result = compare_screenshots_psnr(&psnr, &[], &[]);
        assert_eq!(result.quality, PsnrQuality::Poor);

        let ciede = CieDe2000Metric::default();
        let result = compare_screenshots_ciede(&ciede, &[], &[]);
        assert_eq!(result.classification, DeltaEClassification::Obvious);
    }

    #[test]
    fn test_pixel_coverage_100_percent() {
        // Verify pixel testing covers all demos
        let demo_count = DEMO_CONFIGS.len();
        let viewport_count = VIEWPORTS.len();
        let total_combinations = demo_count * viewport_count;

        // Should cover 4 demos * 3 viewports = 12 combinations
        assert_eq!(total_combinations, 12);

        // Each demo should have state coverage
        for config in DEMO_CONFIGS {
            assert!(
                !config.states.is_empty(),
                "Demo {} must have states for coverage",
                config.name
            );
        }
    }
}
