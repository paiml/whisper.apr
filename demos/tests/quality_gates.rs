//! WAPR-DEMO-005: Quality Gates - Probar Meta Tests
//!
//! These tests verify that ALL demos meet the required quality thresholds:
//! - Button coverage: 100%
//! - State coverage: 100%
//! - Error path coverage: ≥95%
//! - Overall GUI coverage: ≥95%
//! - Accessibility: WCAG AA compliance
//!
//! Run with: `cargo test --package whisper-apr-demos quality_gates`

use probar::{browser, expect, CoverageReport};

/// Demo paths for testing
const DEMO_PATHS: &[(&str, &str)] = &[
    ("realtime-transcription", "/demos/realtime-transcription"),
    ("upload-transcription", "/demos/upload-transcription"),
    ("realtime-translation", "/demos/realtime-translation"),
    ("upload-translation", "/demos/upload-translation"),
];

/// Verify all demos meet GUI coverage requirements
#[probar::test]
async fn verify_all_demos_gui_coverage() {
    for (demo_name, demo_path) in DEMO_PATHS {
        let report = CoverageReport::for_demo(demo_name).await;

        // Button coverage must be 100%
        assert!(
            report.button_coverage() >= 100.0,
            "{}: All buttons must be tested (got {:.1}%)",
            demo_name,
            report.button_coverage()
        );

        // State coverage must be 100%
        assert!(
            report.state_coverage() >= 100.0,
            "{}: All states must be visited (got {:.1}%)",
            demo_name,
            report.state_coverage()
        );

        // Error path coverage must be ≥95%
        assert!(
            report.error_path_coverage() >= 95.0,
            "{}: Error paths must be tested (got {:.1}%)",
            demo_name,
            report.error_path_coverage()
        );

        // Overall GUI coverage must be ≥95%
        assert!(
            report.overall_coverage() >= 95.0,
            "{}: Overall GUI coverage below 95% (got {:.1}%)",
            demo_name,
            report.overall_coverage()
        );

        println!(
            "✓ {} - Coverage: {:.1}% (buttons: {:.1}%, states: {:.1}%, errors: {:.1}%)",
            demo_name,
            report.overall_coverage(),
            report.button_coverage(),
            report.state_coverage(),
            report.error_path_coverage()
        );
    }
}

/// Verify accessibility compliance for all demos
#[probar::test]
async fn verify_accessibility_compliance() {
    for (demo_name, demo_path) in DEMO_PATHS {
        let page = browser::launch().await;
        page.goto(demo_path).await;

        // All interactive elements must have ARIA labels
        let buttons = page.locator("button").all().await;
        for button in &buttons {
            let has_aria = button.get_attribute("aria-label").await.is_some()
                || button.get_attribute("aria-labelledby").await.is_some()
                || !button.text_content().await.is_empty();

            assert!(
                has_aria,
                "{}: Button without accessible label found",
                demo_name
            );
        }

        // Inputs must have labels
        let inputs = page.locator("input").all().await;
        for input in &inputs {
            let input_id = input.get_attribute("id").await;
            if let Some(id) = input_id {
                let has_label = page
                    .locator(&format!("label[for='{}']", id))
                    .count()
                    .await
                    > 0
                    || input.get_attribute("aria-label").await.is_some();

                assert!(has_label, "{}: Input '{}' without label", demo_name, id);
            }
        }

        // Status updates must be announced to screen readers
        let status_element = page.locator("[role='status'], [aria-live]").first().await;
        assert!(
            status_element.is_visible().await,
            "{}: No live region for status announcements",
            demo_name
        );

        // Run automated accessibility audit
        let violations = page.accessibility_audit().await;
        assert!(
            violations.is_empty(),
            "{}: Accessibility violations found: {:?}",
            demo_name,
            violations
        );

        println!("✓ {} - Accessibility compliant", demo_name);
    }
}

/// Verify keyboard navigation for all demos
#[probar::test]
async fn verify_keyboard_navigation() {
    for (demo_name, demo_path) in DEMO_PATHS {
        let page = browser::launch().await;
        page.goto(demo_path).await;

        // Tab through all focusable elements
        let focusable = page
            .locator("button, input, select, textarea, [tabindex]")
            .all()
            .await;

        for _ in 0..focusable.len() {
            page.keyboard().press("Tab").await;

            // Focused element should have visible focus indicator
            let focused = page.locator(":focus").first().await;
            assert!(
                focused.is_visible().await,
                "{}: Focus not visible during keyboard navigation",
                demo_name
            );
        }

        // Escape should close any modals/overlays
        page.keyboard().press("Escape").await;
        let modals = page.locator("[role='dialog']:visible").count().await;
        assert_eq!(
            modals, 0,
            "{}: Escape should close modals",
            demo_name
        );

        println!("✓ {} - Keyboard navigation works", demo_name);
    }
}

/// Verify responsive design breakpoints
#[probar::test]
async fn verify_responsive_design() {
    let viewports = [
        ("mobile", 375, 667),
        ("tablet", 768, 1024),
        ("desktop", 1920, 1080),
    ];

    for (demo_name, demo_path) in DEMO_PATHS {
        for (viewport_name, width, height) in viewports {
            let page = browser::launch_with_viewport(width, height).await;
            page.goto(demo_path).await;

            // Main content should be visible
            let main_content = page.locator("main, #app, .app-container").first().await;
            assert!(
                main_content.is_visible().await,
                "{} @ {}: Main content not visible",
                demo_name,
                viewport_name
            );

            // Primary action button should be visible
            let primary_button = page
                .locator("#start_recording, #transcribe, #translate, .primary-action")
                .first()
                .await;
            assert!(
                primary_button.is_visible().await,
                "{} @ {}: Primary action not visible",
                demo_name,
                viewport_name
            );

            // No horizontal overflow
            let body_width = page.evaluate("document.body.scrollWidth").await;
            let viewport_width = page.evaluate("window.innerWidth").await;
            assert!(
                body_width <= viewport_width,
                "{} @ {}: Horizontal overflow detected",
                demo_name,
                viewport_name
            );
        }

        println!("✓ {} - Responsive design verified", demo_name);
    }
}

/// Verify error handling in all demos
#[probar::test]
async fn verify_error_handling() {
    for (demo_name, demo_path) in DEMO_PATHS {
        let page = browser::launch().await;
        page.goto(demo_path).await;

        // Simulate network error
        page.route("**/*.apr", |route| route.abort()).await;

        // Try to trigger an action that requires network
        if demo_path.contains("upload") {
            page.set_input_files("#file_input", &["test.wav"]).await;
            if page.locator("#transcribe").is_visible().await {
                page.click("#transcribe").await;
            } else if page.locator("#translate").is_visible().await {
                page.click("#translate").await;
            }
        }

        // Error should be displayed gracefully
        page.wait_for_timeout(1000).await;

        // Should not show raw error/stack trace
        let page_content = page.content().await;
        assert!(
            !page_content.contains("Error:") || page.locator("#error_message").is_visible().await,
            "{}: Errors should be shown in UI, not raw",
            demo_name
        );

        // App should still be usable (not crashed)
        let is_interactive = page
            .locator("button:not([disabled])")
            .first()
            .await
            .is_visible()
            .await;
        assert!(
            is_interactive,
            "{}: App should remain interactive after error",
            demo_name
        );

        println!("✓ {} - Error handling verified", demo_name);
    }
}

/// Verify performance budgets
#[probar::test]
async fn verify_performance_budgets() {
    for (demo_name, demo_path) in DEMO_PATHS {
        let page = browser::launch().await;

        // Measure page load time
        let start = std::time::Instant::now();
        page.goto(demo_path).await;
        page.wait_for_load_state("networkidle").await;
        let load_time = start.elapsed();

        // Page should load in under 3 seconds
        assert!(
            load_time.as_secs() < 3,
            "{}: Page load too slow ({:.2}s)",
            demo_name,
            load_time.as_secs_f64()
        );

        // WASM should be cached
        let metrics = page.metrics().await;
        println!(
            "✓ {} - Load time: {:.2}s, JS Heap: {:.1}MB",
            demo_name,
            load_time.as_secs_f64(),
            metrics.js_heap_size_mb()
        );
    }
}

// ============================================================================
// Coverage Report Tests (run after all other tests)
// ============================================================================

/// Generate final coverage report
#[probar::test]
async fn generate_coverage_report() {
    let mut total_coverage = 0.0;
    let mut demo_results = Vec::new();

    for (demo_name, _) in DEMO_PATHS {
        let report = CoverageReport::for_demo(demo_name).await;
        total_coverage += report.overall_coverage();
        demo_results.push((demo_name, report));
    }

    let avg_coverage = total_coverage / DEMO_PATHS.len() as f64;

    println!("\n══════════════════════════════════════════════════");
    println!("           WHISPER.APR DEMO COVERAGE REPORT        ");
    println!("══════════════════════════════════════════════════\n");

    for (name, report) in &demo_results {
        let status = if report.overall_coverage() >= 95.0 {
            "✓"
        } else {
            "✗"
        };
        println!(
            "{} {:30} {:>6.1}%",
            status,
            name,
            report.overall_coverage()
        );
        println!(
            "    Buttons: {:>6.1}%  States: {:>6.1}%  Errors: {:>6.1}%",
            report.button_coverage(),
            report.state_coverage(),
            report.error_path_coverage()
        );
    }

    println!("\n──────────────────────────────────────────────────");
    println!(
        "AVERAGE COVERAGE: {:>6.1}% (target: 95.0%)",
        avg_coverage
    );
    println!("──────────────────────────────────────────────────\n");

    assert!(
        avg_coverage >= 95.0,
        "Average coverage {:.1}% is below 95% threshold",
        avg_coverage
    );
}
