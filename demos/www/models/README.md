# Whisper.apr Model Files

Pre-converted OpenAI Whisper tiny models in APR format for WASM deployment.

## Models

| File | Quantization | Size | Description |
|------|-------------|------|-------------|
| `whisper-tiny.apr` | FP32 | 144 MB | Full precision baseline |
| `whisper-tiny-fb.apr` | FP32 + filterbank | 145 MB | Includes embedded mel filterbank |
| `whisper-tiny-int8.apr` | INT8 | 36 MB | 8-bit quantized |
| `whisper-tiny-int8-fb.apr` | INT8 + filterbank | 37 MB | 8-bit with embedded filterbank |
| `whisper-tiny-int4.apr` | INT4 | 23 MB | 4-bit quantized |
| `whisper-tiny-int4-sparse.apr` | INT4 + sparsity | 9 MB | 4-bit with structured sparsity |

## Architecture

- **Base model**: OpenAI Whisper tiny (39M parameters)
- **Encoder**: 4 layers, 384-dim, 6 heads
- **Decoder**: 4 layers, 384-dim, 6 heads
- **Vocabulary**: 51,865 BPE tokens (multilingual)
- **Audio input**: 80-mel filterbank, 16kHz sample rate

## Format

APR (Aprender) binary format with LZ4 block compression for streaming WASM delivery. See `docs/specifications/` for format details.

## License

Model weights derived from OpenAI Whisper, released under MIT License.
