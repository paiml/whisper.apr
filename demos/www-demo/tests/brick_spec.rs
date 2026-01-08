//! Prototype Brick Specification for Whisper Demo
//!
//! This file mimics the proposed #[brick] syntax to visualize the architecture.
//! It is not yet compilable with actual macros.

#![allow(unused)]

// Mock types for visualization
struct ElementSpec;
impl ElementSpec {
    fn button(id: &str) -> Self { Self }
    fn div(id: &str) -> Self { Self }
    fn text(self, t: &str) -> Self { self }
    fn aria_label(self, l: &str) -> Self { self }
    fn aria_live(self, l: &str) -> Self { self }
    fn role(self, r: &str) -> Self { self }
    fn initial_disabled(self, d: bool) -> Self { self }
}

/// Mock element for assertion callbacks
struct MockElement;
impl MockElement {
    fn text(&self) -> &str { "" }
    fn is_disabled(&self) -> bool { false }
    fn has_class(&self, _: &str) -> bool { false }
}

enum AppState { Initial, Loading, Ready, Recording, Processing }
enum Event { Init, ModelLoaded, Click, Stop }
struct TransitionSpec;
impl TransitionSpec {
    fn new() -> Self { Self }
    fn given(self, _s: AppState) -> Self { self }
    fn when(self, _e: Event) -> Self { self }
    fn then(self, _s: AppState) -> Self { self }
    fn assert_element<F: Fn(&MockElement) -> bool>(self, _sel: &str, _f: F) -> Self { self }
}

// ============================================================================
// BRICK DEFINITIONS
// ============================================================================

/// The Status Bar Brick
/// Responsible for showing application state to the user.
mod status_bar {
    use super::*;

    // #[brick(generates = "span#status")]
    fn element() -> ElementSpec {
        ElementSpec::div("status")
            .text("Loading...")
            .aria_live("polite")
            .role("status")
    }

    // #[brick(state_transition)]
    fn on_model_loaded() -> TransitionSpec {
        TransitionSpec::new()
            .given(AppState::Loading)
            .when(Event::ModelLoaded)
            .then(AppState::Ready)
            .assert_element("#status", |el| el.text().contains("Ready"))
    }
}

/// The Record Button Brick
/// Primary interaction point.
mod record_button {
    use super::*;

    // #[brick(generates = "button#record")]
    fn element() -> ElementSpec {
        ElementSpec::button("record")
            .text("Record")
            .aria_label("Start/Stop Recording")
            .initial_disabled(true) // Crucial: Disabled until Ready
    }

    // #[brick(state_transition)]
    fn enable_when_ready() -> TransitionSpec {
        TransitionSpec::new()
            .given(AppState::Loading)
            .when(Event::ModelLoaded)
            .then(AppState::Ready)
            .assert_element("#record", |el| !el.is_disabled())
    }

    // #[brick(state_transition)]
    fn click_to_record() -> TransitionSpec {
        TransitionSpec::new()
            .given(AppState::Ready)
            .when(Event::Click)
            .then(AppState::Recording)
            .assert_element("#record", |el| el.text() == "Stop")
            .assert_element("#record", |el| el.has_class("recording"))
    }
}

/// The Transcript Area Brick
/// Displays output.
mod transcript_area {
    use super::*;

    // #[brick(generates = "div#transcript")]
    fn element() -> ElementSpec {
        ElementSpec::div("transcript")
            .aria_live("polite")
            .role("log")
            .aria_label("Transcription output")
    }
}
