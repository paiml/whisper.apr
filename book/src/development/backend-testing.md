# Backend End-to-End Testing

The `backend_test` example provides comprehensive validation of all compute backends:
SIMD (CPU), WASM (Browser), and CUDA (GPU).

## Quick Start

```bash
# Run all backends
cargo run --release --example backend_test

# Run specific backend
cargo run --release --example backend_test -- --backend simd
cargo run --release --example backend_test -- --backend wasm
cargo run --release --example backend_test -- --backend cuda
```

## Backends Overview

| Backend | Target | Detection | RTF Goal |
|---------|--------|-----------|----------|
| SIMD | Native CPU | AVX2/SSE2/NEON | < 2.0x |
| WASM | Browser | SIMD128 | < 3.0x |
| CUDA | NVIDIA GPU | wgpu/Vulkan | < 0.5x |

## SIMD Backend

Tests native CPU transcription with SIMD acceleration via trueno:

- **AVX2**: Intel/AMD 64-bit (256-bit vectors)
- **SSE2**: Fallback for older x86_64
- **NEON**: ARM64 (Apple Silicon, mobile)

```bash
cargo run --release --example backend_test -- --backend simd
```

Output:
```
Testing SIMD backend...
  SIMD Backend: AVX2
│ Status:     PASS
│ RTF:        0.47x (target: 2.0x)
│ Timings:    mel=8ms, encode=89ms, decode=1415ms
```

## WASM Backend

Tests browser transcription via headless Chrome:

- Requires local server: `python3 -m http.server 8080 --directory demos/www`
- Uses probar browser test infrastructure
- Validates SIMD128 intrinsics in WASM

```bash
# Start server first
cd demos && python3 -m http.server 8080 --directory www &

# Run WASM tests
cargo run --release --example backend_test -- --backend wasm
```

## CUDA/GPU Backend

Tests GPU-accelerated inference via trueno's wgpu backend:

- Detects NVIDIA GPUs via `nvidia-smi`
- Uses Vulkan for compute shaders
- Falls back gracefully if no GPU available

```bash
cargo run --release --example backend_test -- --backend cuda
```

Output on RTX 4090:
```
Testing CUDA backend...
  GPU: NVIDIA GeForce RTX 4090 (24564MB VRAM)
│ Status:     PASS
│ RTF:        0.35x (target: 0.5x)
```

## CLI Options

```
Backend End-to-End Test CLI

Usage: backend_test [OPTIONS]

Options:
  -b, --backend <BACKEND>  Run specific backend (simd, wasm, cuda)
  -a, --audio <PATH>       Path to test audio file
  -m, --model <PATH>       Path to model file
  -v, --verbose            Verbose output
  -h, --help               Show this help
```

## Result Structure

Each backend test returns:

```rust
pub struct BackendTestResult {
    pub backend: BackendType,
    pub status: TestStatus,      // Pass, Fail, Skipped
    pub rtf: Option<f64>,        // Real-Time Factor
    pub transcript: Option<String>,
    pub timings: Option<Timings>, // mel_ms, encode_ms, decode_ms
    pub memory_mb: Option<f64>,
    pub error: Option<String>,
}
```

## RTF (Real-Time Factor)

RTF measures transcription speed relative to audio duration:

- **RTF 0.5x** = 2x faster than real-time (1s audio in 0.5s)
- **RTF 1.0x** = Real-time (1s audio in 1s)
- **RTF 2.0x** = Half real-time (1s audio in 2s)

Lower RTF is better. Targets by backend:
- SIMD: < 2.0x (acceptable for batch processing)
- WASM: < 3.0x (acceptable for browser, network overhead)
- CUDA: < 0.5x (real-time streaming capable)

## Integration with CI

The backend test can be integrated into CI pipelines:

```yaml
# GitHub Actions example
- name: Run backend tests
  run: |
    cargo run --release --example backend_test -- --backend simd
    # WASM requires browser, skip in CI or use headless
    # CUDA requires GPU runner
```

Exit codes:
- `0`: All required backends pass
- `1`: Any required backend fails

SIMD and WASM are required backends. CUDA is optional (skipped if no GPU).

## Troubleshooting

### WASM: "Demo server not running"
```bash
# Start the demo server
cd demos && python3 -m http.server 8080 --directory www
```

### CUDA: "nvidia-smi not available"
Install NVIDIA drivers and CUDA toolkit.

### CUDA: "wgpu adapter not available"
Ensure Vulkan is installed:
```bash
# Ubuntu/Debian
sudo apt install vulkan-tools mesa-vulkan-drivers

# Verify
vulkaninfo | head -20
```
