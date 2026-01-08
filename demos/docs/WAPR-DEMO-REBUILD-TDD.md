# WAPR-DEMO-REBUILD-TDD: Test-First Demo Rebuild

**Status:** IN PROGRESS
**Approach:** Extreme TDD - 100 steps, all tests written before implementation
**Reference:** whisper.cpp stream.wasm (simple polling architecture)

## Architecture (Simplified like whisper.cpp)

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Main Thread   │───▶│  SharedMemory    │◀───│     Worker      │
│   (UI + Audio)  │    │  (Ring Buffer)   │    │  (Transcribe)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                                              │
        │  set_audio() ──────────────────────────────▶ │
        │                                              │
        │  ◀─────────────────────────── get_result()   │
        │                                              │
        │  mark_done() ──────────────────────────────▶ │
        │                       (Worker polls isDone)  │
```

**Key Simplifications:**
1. No complex message passing for stop - use `isDone()` flag polling
2. Main thread polls `get_result()` for transcription (like whisper.cpp)
3. Simple state machine: Uninitialized → Loading → Ready → Recording → Ready

## Test Plan (100 Steps)

### Phase 1: UX Flow Tests (Steps 1-30)

| Step | Test Name | Description | Priority |
|------|-----------|-------------|----------|
| 1 | `test_page_loads_without_js_errors` | Page loads, no console errors | P0 |
| 2 | `test_status_shows_loading_initially` | Status indicator shows "Loading..." | P0 |
| 3 | `test_record_button_disabled_while_loading` | Record button disabled until ready | P0 |
| 4 | `test_status_shows_ready_after_model_load` | Status shows "Ready" after model loads | P0 |
| 5 | `test_record_button_enabled_when_ready` | Record button enabled when ready | P0 |
| 6 | `test_click_record_starts_recording` | Clicking record starts recording | P0 |
| 7 | `test_status_shows_recording` | Status shows "Recording..." | P0 |
| 8 | `test_vu_meter_animates_during_recording` | VU meter shows audio level | P0 |
| 9 | `test_click_stop_stops_recording` | Clicking stop ends recording | P0 |
| 10 | `test_final_transcription_appears` | Final text appears after stop | P0 |
| 11 | `test_partial_text_during_recording` | Partial results shown while recording | P1 |
| 12 | `test_clear_button_clears_transcript` | Clear button empties transcript | P1 |
| 13 | `test_error_state_on_mic_denied` | Error shown if mic permission denied | P1 |
| 14 | `test_keyboard_space_toggles_record` | Spacebar toggles recording | P2 |
| 15 | `test_keyboard_escape_stops_record` | Escape stops recording | P2 |
| 16 | `test_double_click_record_no_crash` | Rapid clicks don't crash | P1 |
| 17 | `test_record_stop_record_cycle` | Can record multiple times | P1 |
| 18 | `test_transcript_scrolls_on_overflow` | Long transcript scrolls | P2 |
| 19 | `test_status_transitions_correct_order` | States transition correctly | P0 |
| 20 | `test_no_memory_leak_after_recording` | Memory stable after stop | P1 |
| 21 | `test_audio_context_created_on_record` | AudioContext created when needed | P1 |
| 22 | `test_audio_context_suspended_on_stop` | AudioContext suspended on stop | P1 |
| 23 | `test_worker_spawned_on_record` | Worker created when recording starts | P1 |
| 24 | `test_worker_terminated_on_stop` | Worker cleaned up on stop | P1 |
| 25 | `test_ring_buffer_created` | SharedArrayBuffer allocated | P1 |
| 26 | `test_ring_buffer_cleaned_up` | Buffer freed after stop | P1 |
| 27 | `test_model_loads_once` | Model not reloaded on second record | P1 |
| 28 | `test_responsive_mobile_layout` | UI works on mobile viewport | P2 |
| 29 | `test_responsive_tablet_layout` | UI works on tablet viewport | P2 |
| 30 | `test_aria_labels_present` | Accessibility labels present | P2 |

### Phase 2: Pixel/Design Tests (Steps 31-50)

| Step | Test Name | Description | Priority |
|------|-----------|-------------|----------|
| 31 | `test_dark_theme_background` | Background is dark (#0d1117) | P1 |
| 32 | `test_accent_color_blue` | Accent color is blue (#58a6ff) | P1 |
| 33 | `test_record_button_red_when_recording` | Record button turns red | P0 |
| 34 | `test_vu_meter_green_gradient` | VU meter has green gradient | P1 |
| 35 | `test_status_indicator_dot_pulses` | Status dot has pulse animation | P1 |
| 36 | `test_font_family_system` | Uses system font stack | P2 |
| 37 | `test_button_hover_state` | Buttons have hover effect | P2 |
| 38 | `test_button_focus_ring` | Focus ring visible for a11y | P2 |
| 39 | `test_transcript_monospace_font` | Transcript uses monospace | P2 |
| 40 | `test_loading_spinner_visible` | Spinner shown while loading | P1 |
| 41 | `test_error_state_red_border` | Error state has red indicator | P1 |
| 42 | `test_partial_text_italic` | Partial text is italicized | P2 |
| 43 | `test_final_text_normal` | Final text is normal weight | P2 |
| 44 | `test_timestamp_format` | Timestamps formatted correctly | P2 |
| 45 | `test_button_disabled_opacity` | Disabled buttons have 0.5 opacity | P2 |
| 46 | `test_layout_centered` | Main content is centered | P1 |
| 47 | `test_max_width_constraint` | Content has max-width | P2 |
| 48 | `test_padding_consistent` | Consistent spacing | P2 |
| 49 | `test_pixel_snapshot_initial` | Initial state pixel snapshot | P0 |
| 50 | `test_pixel_snapshot_recording` | Recording state pixel snapshot | P0 |

### Phase 3: Worker/Transcription Tests (Steps 51-80)

| Step | Test Name | Description | Priority |
|------|-----------|-------------|----------|
| 51 | `test_worker_js_generates` | Worker JS code generates | P0 |
| 52 | `test_worker_js_no_syntax_errors` | Generated JS parses | P0 |
| 53 | `test_worker_js_has_onmessage` | Has onmessage handler | P0 |
| 54 | `test_worker_js_has_process_tick` | Has processAudioTick | P0 |
| 55 | `test_worker_js_checks_is_done` | Checks isDone() flag | P0 |
| 56 | `test_worker_js_calls_stop_processing` | Calls stopProcessing on done | P0 |
| 57 | `test_worker_receives_init` | Worker receives init message | P0 |
| 58 | `test_worker_receives_start` | Worker receives start message | P0 |
| 59 | `test_worker_detects_done_flag` | Worker sees isDone flag | P0 |
| 60 | `test_worker_sends_ready` | Worker sends ready message | P0 |
| 61 | `test_worker_sends_model_loaded` | Worker sends model loaded | P0 |
| 62 | `test_worker_sends_partial` | Worker sends partial results | P0 |
| 63 | `test_worker_sends_final` | Worker sends final result | P0 |
| 64 | `test_ring_buffer_write_read` | Buffer write/read works | P0 |
| 65 | `test_ring_buffer_wraparound` | Buffer wraps correctly | P1 |
| 66 | `test_ring_buffer_atomics` | Atomics work cross-thread | P0 |
| 67 | `test_ring_buffer_done_flag` | Done flag propagates | P0 |
| 68 | `test_transcription_worker_init` | TranscriptionWorker creates | P0 |
| 69 | `test_transcription_worker_load_model` | Model loads in worker | P0 |
| 70 | `test_transcription_worker_process` | Audio processing works | P0 |
| 71 | `test_transcription_worker_stop` | Stop processing works | P0 |
| 72 | `test_audio_resampling` | 44.1kHz to 16kHz works | P0 |
| 73 | `test_chunk_accumulation` | Samples accumulate correctly | P0 |
| 74 | `test_chunk_threshold` | Transcription at threshold | P0 |
| 75 | `test_silence_detection` | Silence doesn't crash | P1 |
| 76 | `test_loud_audio_clipping` | Loud audio handled | P1 |
| 77 | `test_worker_error_handling` | Errors sent to main thread | P1 |
| 78 | `test_worker_no_window_access` | Worker doesn't use window | P0 |
| 79 | `test_worker_no_document_access` | Worker doesn't use document | P0 |
| 80 | `test_audioworklet_process` | AudioWorklet processes audio | P0 |

### Phase 4: Integration Tests (Steps 81-100)

| Step | Test Name | Description | Priority |
|------|-----------|-------------|----------|
| 81 | `test_e2e_load_to_ready` | Full flow: load → ready | P0 |
| 82 | `test_e2e_record_short_audio` | Record 1s, get transcript | P0 |
| 83 | `test_e2e_record_long_audio` | Record 10s, get transcript | P1 |
| 84 | `test_e2e_multiple_recordings` | Multiple record cycles | P1 |
| 85 | `test_e2e_with_test_audio_file` | Transcribe test-speech.wav | P0 |
| 86 | `test_e2e_accuracy_test_speech` | Accuracy on test audio | P0 |
| 87 | `test_e2e_rtf_under_2x` | RTF < 2x for tiny model | P1 |
| 88 | `test_e2e_memory_under_150mb` | Memory < 150MB peak | P1 |
| 89 | `test_e2e_no_console_errors` | No JS errors in console | P0 |
| 90 | `test_e2e_no_wasm_traps` | No WASM traps | P0 |
| 91 | `test_stress_rapid_start_stop` | Rapid start/stop cycles | P1 |
| 92 | `test_stress_long_recording` | 60s continuous recording | P1 |
| 93 | `test_stress_memory_stability` | Memory stable over time | P1 |
| 94 | `test_cross_browser_chrome` | Works in Chrome | P0 |
| 95 | `test_cross_browser_firefox` | Works in Firefox | P1 |
| 96 | `test_cross_browser_safari` | Works in Safari | P2 |
| 97 | `test_coop_coep_headers` | Required headers present | P0 |
| 98 | `test_shared_array_buffer_available` | SAB works | P0 |
| 99 | `test_wasm_simd_available` | WASM SIMD works | P0 |
| 100 | `test_full_demo_golden_path` | Complete happy path | P0 |

## Implementation Order

1. Write ALL tests first (they will fail)
2. Implement minimal lib.rs to make step 1-10 pass
3. Implement worker to make step 51-70 pass
4. Implement ring buffer to make step 64-67 pass
5. Wire up integration to make step 81-100 pass
6. Add styling to make step 31-50 pass
7. Verify 100% coverage
8. Run full suite, fix any failures
9. Only THEN ask user to demo

## Files to Create

```
www-demo/src/
├── lib.rs              # Main entry, UI, state machine
├── worker.rs           # TranscriptionWorker
├── worker_js.rs        # Worker JS generator
├── ring_buffer.rs      # SharedArrayBuffer ring buffer
├── audio_worklet.rs    # AudioWorklet integration
└── audioworklet_js.rs  # AudioWorklet JS generator

tests/src/
├── ux_flow_tests.rs    # Steps 1-30
├── pixel_tests.rs      # Steps 31-50 (exists, extend)
├── worker_tests.rs     # Steps 51-80
└── integration_tests.rs # Steps 81-100
```
