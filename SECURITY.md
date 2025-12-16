# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it by emailing security@paiml.com.

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days.

## Security Considerations

### WASM Sandboxing
whisper.apr runs in the browser's WASM sandbox, providing memory isolation from the host system.

### Audio Data
Audio data is processed locally in the browser. No audio is transmitted to external servers unless explicitly configured.

### Model Loading
Models are loaded from configured URLs. Ensure models are served over HTTPS in production.
