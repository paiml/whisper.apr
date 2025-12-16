#!/usr/bin/env python3
"""
Extract ground truth values from whisper reference implementation.

This script generates golden values at each pipeline step for falsification testing.

Usage:
    python3 tools/extract_ground_truth.py

Outputs:
    golden_traces/step_*.bin - Binary dumps of intermediate values
    golden_traces/step_*.json - Metadata for each step
"""

import numpy as np
import wave
import struct
import json
import subprocess
import os
from pathlib import Path

# Configuration
AUDIO_PATH = "demos/test-audio/test-speech-1.5s.wav"
WHISPER_CPP_PATH = "../whisper.cpp"
WHISPER_CPP_MODEL = "../whisper.cpp/models/ggml-tiny.bin"
OUTPUT_DIR = Path("golden_traces")
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_binary(name: str, data: np.ndarray, metadata: dict = None):
    """Save binary data with metadata."""
    bin_path = OUTPUT_DIR / f"{name}.bin"
    json_path = OUTPUT_DIR / f"{name}.json"

    data.astype(np.float32).tofile(bin_path)

    meta = {
        "name": name,
        "shape": list(data.shape),
        "dtype": "float32",
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "nonzero": int(np.count_nonzero(data)),
        "file": str(bin_path),
    }
    if metadata:
        meta.update(metadata)

    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved {name}: shape={data.shape}, range=[{data.min():.4f}, {data.max():.4f}]")
    return bin_path


def step_a_audio():
    """Step A: Extract raw audio samples."""
    print("\n=== Step A: Audio Samples ===")

    with wave.open(AUDIO_PATH, 'rb') as f:
        params = f.getparams()
        pcm = np.frombuffer(f.readframes(-1), dtype=np.int16)

    # Normalize to [-1, 1]
    samples = pcm.astype(np.float32) / 32768.0

    save_binary("step_a_audio", samples, {
        "source": AUDIO_PATH,
        "sample_rate": params.framerate,
        "channels": params.nchannels,
        "duration_sec": len(samples) / params.framerate,
    })

    return samples


def step_b_filterbank():
    """Step B: Load filterbank from whisper.cpp."""
    print("\n=== Step B: Mel Filterbank ===")

    # Already extracted to /tmp/whisper_cpp_filterbank.bin
    fb_path = "/tmp/whisper_cpp_filterbank.bin"
    if not os.path.exists(fb_path):
        print(f"  ERROR: Run 'python3 tools/extract_filterbank.py {WHISPER_CPP_MODEL}' first")
        return None

    filterbank = np.fromfile(fb_path, dtype=np.float32).reshape(N_MELS, 201)
    save_binary("step_b_filterbank", filterbank, {
        "source": "whisper.cpp ggml-tiny.bin",
        "n_mels": N_MELS,
        "n_freqs": 201,
    })

    return filterbank


def step_c_mel_spectrogram(samples: np.ndarray, filterbank: np.ndarray):
    """Step C: Compute mel spectrogram using whisper's exact method."""
    print("\n=== Step C: Mel Spectrogram ===")

    try:
        import librosa
    except ImportError:
        print("  WARNING: librosa not installed, using numpy FFT")
        return step_c_mel_numpy(samples, filterbank)

    # Use librosa's mel spectrogram with whisper parameters
    # Whisper uses: n_fft=400, hop_length=160, n_mels=80, fmax=8000
    mel = librosa.feature.melspectrogram(
        y=samples,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=8000,
    )

    # Apply log compression (same as whisper)
    log_mel = np.log10(np.maximum(mel, 1e-10))

    # Normalize (whisper specific)
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0

    save_binary("step_c_mel_librosa", log_mel.T, {  # Transpose to (T, n_mels)
        "method": "librosa",
        "n_frames": log_mel.shape[1],
        "n_mels": N_MELS,
    })

    return log_mel.T


def step_c_mel_numpy(samples: np.ndarray, filterbank: np.ndarray):
    """Step C: Compute mel spectrogram using numpy (matches whisper.cpp more closely)."""
    print("  Using numpy FFT implementation")

    # Pad to 30 seconds (whisper requirement)
    n_samples = len(samples)

    # Compute STFT
    window = np.hanning(N_FFT)
    n_frames = (n_samples - N_FFT) // HOP_LENGTH + 1

    # Pre-allocate
    stft_magnitude = np.zeros((n_frames, N_FFT // 2 + 1), dtype=np.float32)

    for i in range(n_frames):
        start = i * HOP_LENGTH
        frame = samples[start:start + N_FFT] * window
        fft = np.fft.rfft(frame)
        stft_magnitude[i] = np.abs(fft) ** 2

    # Apply mel filterbank
    mel = stft_magnitude @ filterbank.T

    # Log compression
    log_mel = np.log10(np.maximum(mel, 1e-10))

    # Whisper normalization
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0

    save_binary("step_c_mel_numpy", log_mel, {
        "method": "numpy",
        "n_frames": n_frames,
        "n_mels": N_MELS,
    })

    return log_mel


def step_n_tokens():
    """Step N: Get token sequence from whisper.cpp."""
    print("\n=== Step N: Token Sequence ===")

    # Run whisper.cpp to get tokens
    cmd = [
        f"{WHISPER_CPP_PATH}/main",
        "-m", WHISPER_CPP_MODEL,
        "-f", AUDIO_PATH,
        "--output-txt",
        "-of", "/tmp/whisper_out",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Parse output for tokens
        output_lines = result.stdout.split('\n')
        transcript = ""
        for line in output_lines:
            if line.startswith('[') and '-->' in line:
                # Extract text after timestamp
                parts = line.split(']')
                if len(parts) > 1:
                    transcript += parts[1].strip()

        # Save transcript
        with open(OUTPUT_DIR / "step_o_text.txt", 'w') as f:
            f.write(transcript)

        print(f"  Transcript: '{transcript}'")

        # Also save raw output
        with open(OUTPUT_DIR / "whisper_cpp_output.txt", 'w') as f:
            f.write(result.stdout)

        return transcript

    except FileNotFoundError:
        print(f"  ERROR: whisper.cpp not found at {WHISPER_CPP_PATH}")
        return None
    except subprocess.TimeoutExpired:
        print("  ERROR: whisper.cpp timed out")
        return None


def run_whisper_apr():
    """Run our whisper.apr and get output for comparison."""
    print("\n=== Running whisper.apr ===")

    cmd = [
        "cargo", "run", "--release", "--example", "debug_transcribe"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # Save output
        with open(OUTPUT_DIR / "whisper_apr_output.txt", 'w') as f:
            f.write(result.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

        print(f"  Output saved to {OUTPUT_DIR}/whisper_apr_output.txt")

        # Extract tokens from output
        for line in result.stdout.split('\n'):
            if 'Generated tokens' in line or 'token' in line.lower():
                print(f"  {line}")

        return result.stdout

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def create_comparison_summary():
    """Create a summary comparing all steps."""
    print("\n=== Creating Comparison Summary ===")

    summary = {
        "audio_path": AUDIO_PATH,
        "whisper_cpp_model": WHISPER_CPP_MODEL,
        "steps": {}
    }

    # Load all step metadata
    for json_path in OUTPUT_DIR.glob("step_*.json"):
        with open(json_path) as f:
            data = json.load(f)
            summary["steps"][data["name"]] = data

    # Save summary
    with open(OUTPUT_DIR / "SUMMARY.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Create markdown summary
    md_lines = [
        "# Ground Truth Extraction Summary",
        "",
        f"**Audio**: `{AUDIO_PATH}`",
        f"**Model**: `{WHISPER_CPP_MODEL}`",
        "",
        "## Steps Extracted",
        "",
        "| Step | Shape | Range | Mean |",
        "|------|-------|-------|------|",
    ]

    for name, data in sorted(summary["steps"].items()):
        shape = "x".join(map(str, data["shape"]))
        range_str = f"[{data['min']:.4f}, {data['max']:.4f}]"
        md_lines.append(f"| {name} | {shape} | {range_str} | {data['mean']:.4f} |")

    md_lines.extend([
        "",
        "## Usage",
        "",
        "```rust",
        "// Load ground truth",
        "let gt_mel = load_f32_binary(\"golden_traces/step_c_mel_numpy.bin\")?;",
        "",
        "// Compare with our mel",
        "let our_mel = model.compute_mel(&audio)?;",
        "let cosine = cosine_similarity(&gt_mel, &our_mel);",
        "assert!(cosine > 0.99, \"Mel divergence: {}\", 1.0 - cosine);",
        "```",
    ])

    with open(OUTPUT_DIR / "SUMMARY.md", 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"  Summary saved to {OUTPUT_DIR}/SUMMARY.md")


def main():
    print("=" * 60)
    print("GROUND TRUTH EXTRACTION FOR PIPELINE FALSIFICATION")
    print("=" * 60)

    ensure_output_dir()

    # Step A: Audio
    samples = step_a_audio()

    # Step B: Filterbank
    filterbank = step_b_filterbank()

    # Step C: Mel Spectrogram
    if filterbank is not None:
        mel = step_c_mel_spectrogram(samples, filterbank)

    # Step N: Tokens from whisper.cpp
    transcript = step_n_tokens()

    # Run our implementation for comparison
    run_whisper_apr()

    # Create summary
    create_comparison_summary()

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nGround truth files saved to: {OUTPUT_DIR}/")
    print("\nNext steps:")
    print("  1. Run: cargo run --example pipeline_falsification")
    print("  2. Compare each step against ground truth")
    print("  3. Identify first divergence point")


if __name__ == "__main__":
    main()
