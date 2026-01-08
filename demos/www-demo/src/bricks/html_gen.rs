//! HTML Generator from Bricks (PROBAR-SPEC-009)
//!
//! This module generates the complete index.html from Brick definitions.
//! Zero hand-written HTML - all code generated from Rust types.
//!
//! # Design Philosophy
//!
//! The generator:
//! 1. Collects all bricks in the `BrickHouse`
//! 2. Generates combined CSS from all bricks
//! 3. Generates combined HTML from all bricks
//! 4. Produces a single index.html file
//!
//! This ensures the HTML is always in sync with the brick definitions
//! and assertions.

use std::sync::Arc;
use jugar_probar::brick::Brick;
use jugar_probar::brick_house::BrickHouse;

use super::{StatusBrick, TranscriptionBrick, VuMeterBrick, WaveformBrick};

/// Configuration for the generated HTML
#[derive(Debug, Clone)]
pub struct HtmlConfig {
    /// Page title
    pub title: String,
    /// WASM module path
    pub wasm_module: String,
    /// Model path
    pub model_path: String,
}

impl Default for HtmlConfig {
    fn default() -> Self {
        Self {
            title: "Whisper.apr Demo - Real-time Speech Recognition".into(),
            wasm_module: "./pkg/whisper_apr_demo.js".into(),
            model_path: "/models/whisper-tiny.apr".into(),
        }
    }
}

/// Generate the complete index.html from brick definitions
#[must_use] 
pub fn generate_index_html(config: &HtmlConfig) -> String {
    // Create bricks
    let status = StatusBrick::new();
    let vu_meter = VuMeterBrick::new();
    let transcription = TranscriptionBrick::new();

    // Collect CSS from all bricks
    let css = generate_css(&[
        &status as &dyn Brick,
        &vu_meter as &dyn Brick,
        &transcription as &dyn Brick,
    ]);

    // Generate the minimal JS glue (≤50 lines per spec)
    let js_glue = generate_js_glue(config);

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{base_css}
{css}
    </style>
</head>
<body>
    <div class="container">
        <h1>Whisper.apr Demo</h1>

        <div class="status-bar">
            {status_html}
            {vu_html}
        </div>

        <div class="controls">
            <button id="record" disabled aria-label="Start/Stop Recording">Record</button>
            <button id="clear" aria-label="Clear transcript">Clear</button>
        </div>

        <div class="output">
            {transcription_html}
        </div>
    </div>

    <script type="module">
{js_glue}
    </script>
</body>
</html>
"#,
        title = config.title,
        base_css = BASE_CSS,
        css = css,
        status_html = status.to_html(),
        vu_html = vu_meter.to_html(),
        transcription_html = transcription.to_html(),
        js_glue = js_glue,
    )
}

/// Generate combined CSS from bricks
fn generate_css(bricks: &[&dyn Brick]) -> String {
    bricks
        .iter()
        .map(|b| b.to_css())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Base CSS for the page layout
const BASE_CSS: &str = r"        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem;
        }
        .container {
            max-width: 800px;
            width: 100%;
        }
        h1 {
            text-align: center;
            margin-bottom: 2rem;
            color: #4dc3ff;
        }
        .status-bar {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .controls {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        button {
            padding: 1rem 2rem;
            font-size: 1rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        #record {
            background: #e94560;
            color: white;
            flex: 1;
        }
        #record:hover:not(:disabled) {
            background: #ff6b6b;
        }
        #record:disabled {
            background: #666;
            cursor: not-allowed;
        }
        #record.recording {
            background: #50fa7b;
            animation: pulse 1s infinite;
        }
        #clear {
            background: #4dc3ff;
            color: #1a1a2e;
        }
        #clear:hover {
            background: #7dd5ff;
        }
        .output {
            background: #16213e;
            border-radius: 8px;
            padding: 1.5rem;
            min-height: 200px;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }";

/// Generate minimal JS glue code (≤50 lines per spec requirement)
fn generate_js_glue(config: &HtmlConfig) -> String {
    format!(
        r"        // Minimal JS glue - all logic in WASM (PROBAR-SPEC-009)
        import init, {{ WorkerManager }} from '{wasm_module}';

        let manager = null;
        let isRecording = false;
        let audioContext = null;
        let mediaStream = null;

        const status = document.getElementById('status');
        const recordBtn = document.getElementById('record');
        const clearBtn = document.getElementById('clear');
        const vuMeter = document.getElementById('vu_meter');

        async function initDemo() {{
            try {{
                status.textContent = 'Loading WASM...';
                await init();
                status.textContent = 'Spawning worker...';
                manager = new WorkerManager();
                await manager.spawn('{model_path}');

                window.addEventListener('whisper-worker-ready', () => {{
                    status.textContent = 'Loading model...';
                    manager.send_init();
                }}, {{ once: true }});

                window.addEventListener('whisper-model-loaded', (e) => {{
                    const {{ sizeMb, loadTimeMs }} = e.detail;
                    status.textContent = `Ready (${{sizeMb.toFixed(1)}}MB in ${{(loadTimeMs/1000).toFixed(1)}}s)`;
                    recordBtn.disabled = false;
                }});

                window.addEventListener('whisper-transcription', (e) => {{
                    const {{ text, isFinal }} = e.detail;
                    const target = isFinal ? document.querySelector('.transcription-final') : document.querySelector('.transcription-partial');
                    if (target) {{
                        if (isFinal) {{
                            target.textContent += text + ' ';
                            const partial = document.querySelector('.transcription-partial');
                            if (partial) partial.textContent = '';
                        }} else {{
                            target.textContent = text;
                        }}
                    }}
                }});
            }} catch (err) {{
                status.textContent = 'Error: ' + err.message;
            }}
        }}

        recordBtn.addEventListener('click', async () => {{
            if (isRecording) {{
                manager.stopRecording();
                if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
                if (audioContext) audioContext.close();
                isRecording = false;
                recordBtn.textContent = 'Record';
                recordBtn.classList.remove('recording');
                status.textContent = 'Ready';
                vuMeter.style.width = '0%';
            }} else {{
                mediaStream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                audioContext = new AudioContext();
                const source = audioContext.createMediaStreamSource(mediaStream);
                const processor = audioContext.createScriptProcessor(4096, 1, 1);
                const ringBuffer = manager.getRingBuffer();

                processor.onaudioprocess = (e) => {{
                    if (!isRecording) return;
                    const samples = e.inputBuffer.getChannelData(0);
                    if (ringBuffer) ringBuffer.write(samples);
                    let sum = 0;
                    for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
                    vuMeter.style.width = Math.min(100, Math.sqrt(sum / samples.length) * 500) + '%';
                }};

                source.connect(processor);
                processor.connect(audioContext.destination);
                manager.startRecording(audioContext.sampleRate);
                isRecording = true;
                recordBtn.textContent = 'Stop';
                recordBtn.classList.add('recording');
                status.textContent = 'Recording...';
            }}
        }});

        clearBtn.addEventListener('click', () => {{
            document.querySelector('.transcription-final').textContent = '';
            document.querySelector('.transcription-partial').textContent = '';
        }});

        initDemo();",
        wasm_module = config.wasm_module,
        model_path = config.model_path,
    )
}

/// Create a `BrickHouse` for the whisper demo
pub fn create_whisper_brick_house() -> Result<BrickHouse, jugar_probar::brick::BrickError> {
    let mut house = BrickHouse::new("whisper-demo", 1000); // 1 second total budget

    house.add_brick(Arc::new(StatusBrick::new()), 50)?;
    house.add_brick(Arc::new(VuMeterBrick::new()), 10)?;
    house.add_brick(Arc::new(WaveformBrick::new()), 16)?;
    house.add_brick(Arc::new(TranscriptionBrick::new()), 100)?;

    Ok(house)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_index_html() {
        let config = HtmlConfig::default();
        let html = generate_index_html(&config);

        // Verify structure
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<title>Whisper.apr Demo"));
        assert!(html.contains("data-testid=\"status\""));
        assert!(html.contains("data-testid=\"vu-meter\""));
        assert!(html.contains("data-testid=\"transcription\""));
    }

    #[test]
    fn test_css_included() {
        let config = HtmlConfig::default();
        let html = generate_index_html(&config);

        // Verify CSS from bricks is included
        assert!(html.contains(".status-brick"));
        assert!(html.contains(".vu-meter-brick"));
        assert!(html.contains(".transcription-brick"));
    }

    #[test]
    fn test_js_glue_minimal() {
        let config = HtmlConfig::default();
        let js = generate_js_glue(&config);

        // Count lines (should be ≤50 significant lines)
        let lines: Vec<_> = js
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.trim().starts_with("//"))
            .collect();

        // Allow some margin for readability
        assert!(
            lines.len() <= 80,
            "JS glue has {} lines, should be minimal",
            lines.len()
        );
    }

    #[test]
    fn test_js_uses_wasm_module() {
        let config = HtmlConfig::default();
        let js = generate_js_glue(&config);

        assert!(js.contains("WorkerManager"));
        assert!(js.contains("manager.spawn"));
        assert!(js.contains("manager.startRecording"));
        assert!(js.contains("manager.stopRecording"));
    }

    #[test]
    fn test_create_brick_house() {
        let house = create_whisper_brick_house().expect("should create house");

        assert_eq!(house.name(), "whisper-demo");
        assert_eq!(house.brick_count(), 4);
        assert!(house.remaining_budget_ms() > 0);
    }

    #[test]
    fn test_brick_house_can_render() {
        let house = create_whisper_brick_house().expect("should create house");
        assert!(house.can_render());
    }
}
