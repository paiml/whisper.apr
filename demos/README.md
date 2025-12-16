# Whisper.apr Demos

Real-time speech-to-text demos powered by Whisper.apr WASM engine.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
probar serve -d www --cross-origin-isolated --port 8080

# Open browser
open http://localhost:8080
```

## Running Tests

```bash
# All tests
cargo test

# Browser tests (requires Chrome)
probar test --playbook playbooks/realtime-transcription.yaml

# Visual regression
probar test --pixel --update-snapshots

# Full validation suite
probar test --all
```

## Demos

| Demo | Description | Path |
|------|-------------|------|
| Real-time Transcription | Live microphone transcription | `/realtime-transcription.html` |
| Upload Transcription | File upload transcription | `/upload-transcription.html` |
| Real-time Translation | Live translation | `/realtime-translation.html` |
| Upload Translation | File translation | `/upload-translation.html` |

## Test Coverage

See `tests/README.md` for detailed test documentation.

- **Playbooks**: State machine validation
- **Snapshots**: Visual regression testing
- **Recordings**: Deterministic replay testing
- **Performance**: RTF and latency validation
