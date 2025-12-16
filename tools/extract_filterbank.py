#!/usr/bin/env python3
"""Extract mel filterbank from ggml model file for comparison.

The ggml format stores the filterbank after the header section.
This script extracts it and saves as raw floats for comparison.
"""

import struct
import sys
import numpy as np

def read_ggml_filterbank(model_path):
    """Extract filterbank from ggml model file."""
    with open(model_path, 'rb') as f:
        # Read magic
        magic = f.read(4)
        print(f"Magic: {magic}")

        # Read header based on ggml format
        # We need to find where the filters section is
        # Looking at whisper.cpp:1580-1584:
        # read_safe(loader, filters.n_mel);
        # read_safe(loader, filters.n_fft);
        # filters.data.resize(filters.n_mel * filters.n_fft);
        # loader->read(loader->context, filters.data.data(), filters.data.size() * sizeof(float));

        # Skip to find filters section
        # ggml format has variable header, need to parse properly

        # For now, let's use a simpler approach - search for expected patterns
        f.seek(0)
        data = f.read()

        # Look for the filters section
        # In whisper models, n_mel=80, n_fft=201
        # So filterbank is 80 * 201 = 16080 floats = 64320 bytes

        # Try to find a sequence of 80 followed by 201 (as i32)
        for offset in range(0, min(len(data) - 64320, 10000), 4):
            n_mel = struct.unpack_from('<i', data, offset)[0]
            n_fft = struct.unpack_from('<i', data, offset + 4)[0]

            if n_mel == 80 and n_fft == 201:
                print(f"Found filterbank header at offset {offset}")
                print(f"  n_mel: {n_mel}")
                print(f"  n_fft: {n_fft}")

                # Extract filterbank
                filter_start = offset + 8
                filter_size = n_mel * n_fft

                if filter_start + filter_size * 4 <= len(data):
                    filterbank = struct.unpack_from(f'<{filter_size}f', data, filter_start)
                    filterbank = np.array(filterbank, dtype=np.float32)

                    # Reshape to (n_mel, n_fft)
                    filterbank = filterbank.reshape(n_mel, n_fft)

                    return filterbank

        print("Could not find filterbank in model file")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: extract_filterbank.py <model.bin> [output.npy]")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/whisper_cpp_filterbank.npy"

    filterbank = read_ggml_filterbank(model_path)

    if filterbank is not None:
        print(f"\nFilterbank shape: {filterbank.shape}")
        print(f"Min: {filterbank.min():.6f}")
        print(f"Max: {filterbank.max():.6f}")
        print(f"Mean: {filterbank.mean():.6f}")
        print(f"Std: {filterbank.std():.6f}")

        # Check for rows/cols that are all zeros
        zero_rows = np.sum(np.all(filterbank == 0, axis=1))
        zero_cols = np.sum(np.all(filterbank == 0, axis=0))
        print(f"Zero rows: {zero_rows}")
        print(f"Zero cols: {zero_cols}")

        # Show first few values of first row
        print(f"\nFirst row (first 10 values): {filterbank[0, :10]}")
        print(f"Row 1 (first 10 values): {filterbank[1, :10]}")
        print(f"Row 2 (first 10 values): {filterbank[2, :10]}")

        # Save
        np.save(output_path, filterbank)
        print(f"\nSaved to {output_path}")

        # Also save as raw floats for Rust
        raw_path = output_path.replace('.npy', '.bin')
        filterbank.tofile(raw_path)
        print(f"Saved raw to {raw_path}")

if __name__ == "__main__":
    main()
