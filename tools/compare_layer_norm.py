#!/usr/bin/env python3
"""
Compare layer norm weights between whisper.cpp and our model.

This helps identify if the conversion is wrong.

Usage:
    python3 tools/compare_layer_norm.py
"""

import struct
import numpy as np
import os

WHISPER_CPP_MODEL = "../whisper.cpp/models/ggml-tiny.bin"
OUR_MODEL = "models/whisper-tiny.apr"

def read_ggml_tensor_info(f):
    """Read tensor info from GGML format."""
    n_dims = struct.unpack('<I', f.read(4))[0]
    name_len = struct.unpack('<I', f.read(4))[0]
    dtype = struct.unpack('<I', f.read(4))[0]

    dims = []
    for _ in range(n_dims):
        dims.append(struct.unpack('<I', f.read(4))[0])
    # Pad to 4 dims
    while len(dims) < 4:
        dims.append(1)

    name = f.read(name_len).decode('utf-8')

    # GGML data types
    # 0 = f32, 1 = f16, 2 = q4_0, etc.

    return {
        'name': name,
        'dims': dims,
        'dtype': dtype,
        'n_dims': n_dims
    }


def extract_ggml_layer_norm(model_path, tensor_name):
    """Extract a specific tensor from GGML model."""
    with open(model_path, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != b'ggjt':
            print(f"Warning: unexpected magic {magic}")

        version = struct.unpack('<I', f.read(4))[0]

        # Read hyperparameters (tiny model: 384 dim)
        n_vocab = struct.unpack('<I', f.read(4))[0]
        n_audio_ctx = struct.unpack('<I', f.read(4))[0]
        n_audio_state = struct.unpack('<I', f.read(4))[0]
        n_audio_head = struct.unpack('<I', f.read(4))[0]
        n_audio_layer = struct.unpack('<I', f.read(4))[0]
        n_text_ctx = struct.unpack('<I', f.read(4))[0]
        n_text_state = struct.unpack('<I', f.read(4))[0]
        n_text_head = struct.unpack('<I', f.read(4))[0]
        n_text_layer = struct.unpack('<I', f.read(4))[0]
        n_mels = struct.unpack('<I', f.read(4))[0]
        ftype = struct.unpack('<I', f.read(4))[0]

        print(f"Model params: vocab={n_vocab}, state={n_audio_state}, layers={n_audio_layer}")

        # Skip tokens
        for _ in range(n_vocab):
            tok_len = struct.unpack('<I', f.read(4))[0]
            f.read(tok_len)

        # Read tensors
        while True:
            pos = f.tell()
            try:
                info = read_ggml_tensor_info(f)
            except:
                break

            n_elements = 1
            for d in info['dims']:
                n_elements *= d

            # Data size based on dtype
            if info['dtype'] == 0:  # f32
                data_size = n_elements * 4
            elif info['dtype'] == 1:  # f16
                data_size = n_elements * 2
            else:
                print(f"Unknown dtype {info['dtype']} for {info['name']}")
                break

            # Align to 32 bytes
            aligned_pos = (f.tell() + 31) & ~31
            f.seek(aligned_pos)

            if info['name'] == tensor_name:
                # Found it! Read the data
                if info['dtype'] == 0:
                    data = np.frombuffer(f.read(data_size), dtype=np.float32)
                else:
                    data = np.frombuffer(f.read(data_size), dtype=np.float16).astype(np.float32)
                return data

            # Skip data
            f.seek(aligned_pos + data_size)

    return None


def main():
    print("=" * 60)
    print("LAYER NORM WEIGHT COMPARISON")
    print("=" * 60)
    print()

    if not os.path.exists(WHISPER_CPP_MODEL):
        print(f"ERROR: {WHISPER_CPP_MODEL} not found")
        return

    # The tensor names in GGML format
    # decoder.ln.weight corresponds to decoder.layer_norm.weight in HuggingFace
    ggml_tensors_to_check = [
        "decoder.ln.weight",
        "decoder.ln.bias",
        "encoder.ln.weight",
        "encoder.ln.bias",
        "decoder.blocks.0.ln1.weight",
        "decoder.blocks.3.ln3.weight",
    ]

    print("=== Whisper.cpp (GGML) Layer Norm Weights ===\n")

    for name in ggml_tensors_to_check:
        data = extract_ggml_layer_norm(WHISPER_CPP_MODEL, name)
        if data is not None:
            print(f"{name}:")
            print(f"  shape: {data.shape}")
            print(f"  mean:  {data.mean():.6f}")
            print(f"  range: [{data.min():.4f}, {data.max():.4f}]")
            print(f"  first 5: {data[:5]}")
            print()
        else:
            print(f"{name}: NOT FOUND")
            print()


if __name__ == "__main__":
    main()
