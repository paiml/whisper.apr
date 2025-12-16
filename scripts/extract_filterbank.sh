#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Extract mel filterbank from OpenAI Whisper for embedding in .apr models
#
# NPZ format: ZIP archive containing numpy arrays as .npy files
# NPY format: magic + version + header + raw data
#
# Usage:
#   ./scripts/extract_filterbank.sh [output_dir]
#
# Output:
#   mel_80.bin   - 80x201 f32 filterbank (64,320 bytes)
#   mel_128.bin  - 128x201 f32 filterbank (102,912 bytes)

set -euo pipefail

# Get script directory safely
get_script_dir() {
    local source="${BASH_SOURCE[0]}"
    cd "$(dirname "$source")" && pwd
}

readonly SCRIPT_DIR="$(get_script_dir)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly OUTPUT_DIR="${1:-$PROJECT_ROOT/data}"
readonly CACHE_DIR="${XDG_CACHE_HOME:-"$HOME/.cache"}/whisper-apr"
readonly MEL_FILTERS_URL="https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"
readonly MEL_FILTERS_NPZ="$CACHE_DIR/mel_filters.npz"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Validate path doesn't contain traversal or absolute root
validate_path() {
    local path="$1"
    if [[ -z "$path" ]]; then
        log_error "Empty path"
        return 1
    fi
    if [[ "$path" == *".."* ]]; then
        log_error "Path traversal detected: $path"
        return 1
    fi
    if [[ "$path" == "/" ]]; then
        log_error "Root path not allowed"
        return 1
    fi
}

# Safe mkdir with validation
safe_mkdir() {
    local dir="$1"
    validate_path "$dir" || return 1
    # bashrs:ignore SEC010 - path validated above
    # shellcheck disable=SC2174
    mkdir -p -m 0755 "$dir"
}

# Safe rm -rf with validation (only /tmp/* allowed)
safe_rmrf() {
    local dir="$1"
    # Only allow removal of temp directories
    if [[ -z "$dir" ]]; then
        return 0
    fi
    if [[ ! -d "$dir" ]]; then
        return 0
    fi
    # bashrs:ignore SEC010 - explicitly checking /tmp/* prefix
    if [[ "$dir" != /tmp/* ]]; then
        log_error "Refusing to rm -rf non-temp directory: $dir"
        return 1
    fi
    validate_path "$dir" || return 1
    # bashrs:ignore SEC010,SEC011 - path validated above, /tmp/* enforced
    rm -rf "$dir"
}

# Download mel_filters.npz if not cached
download_filterbank() {
    safe_mkdir "$CACHE_DIR"

    if [[ -f "$MEL_FILTERS_NPZ" ]]; then
        log_info "Using cached: $MEL_FILTERS_NPZ"
        return 0
    fi

    log_info "Downloading mel_filters.npz from OpenAI..."
    curl -fsSL "$MEL_FILTERS_URL" -o "$MEL_FILTERS_NPZ"
    log_info "Downloaded to: $MEL_FILTERS_NPZ"
}

# Extract .npy file from .npz (which is just a ZIP)
extract_npy() {
    local npz_file="$1"
    local npy_name="$2"
    local output_file="$3"

    validate_path "$output_file" || return 1
    # bashrs:ignore SEC010 - output_file validated above, unzip -p outputs to stdout
    # NPZ is a ZIP - extract the .npy file (unzip -p outputs to stdout, safe)
    unzip -p "$npz_file" "${npy_name}.npy" > "$output_file"
}

# Parse NPY header and extract raw float data
# NPY v1.0 format:
#   - Magic: \x93NUMPY (6 bytes)
#   - Version: 1.0 (2 bytes)
#   - Header len: u16 LE (2 bytes)
#   - Header: Python dict as ASCII (variable)
#   - Data: raw bytes (rest of file)
parse_npy_to_bin() {
    local npy_file="$1"
    local bin_file="$2"

    validate_path "$bin_file"

    # Read magic and version
    local magic
    magic="$(head -c 6 "$npy_file" | xxd -p)"
    if [[ "$magic" != "934e554d5059" ]]; then
        log_error "Invalid NPY magic: $magic"
        return 1
    fi

    # Read header length (bytes 8-9, little-endian u16)
    local header_len
    header_len="$(dd if="$npy_file" bs=1 skip=8 count=2 2>/dev/null | od -An -tu2 -v | tr -d ' ')"

    # Data starts after: magic(6) + version(2) + header_len(2) + header
    local data_offset
    data_offset=$((10 + header_len))

    # Extract raw data (skip header)
    tail -c +"$((data_offset + 1))" "$npy_file" > "$bin_file"

    local size
    size="$(stat -c%s "$bin_file" 2>/dev/null || stat -f%z "$bin_file")"
    log_info "Extracted $((size / 4)) floats to $bin_file"
}

# Generate Rust constant from binary file
generate_rust_const() {
    local bin_file="$1"
    local const_name="$2"
    local n_mels="$3"
    local n_freqs="$4"

    local total
    total=$((n_mels * n_freqs))

    echo "/// $const_name filterbank from OpenAI Whisper"
    echo "/// Shape: [$n_mels, $n_freqs]"
    echo "/// Source: $MEL_FILTERS_URL"
    echo "#[allow(clippy::excessive_precision)]"
    echo "pub static ${const_name}_FILTERBANK: [f32; $total] = ["

    # Convert binary f32 LE to text, 8 values per line
    od -An -tf4 -v -w32 "$bin_file" | while IFS= read -r line; do
        # Format each float with scientific notation using printf
        local formatted=""
        for val in $line; do
            formatted="${formatted}${val}, "
        done
        echo "    $formatted"
    done

    echo "];"
    echo ""
}

main() {
    log_info "Extracting OpenAI Whisper mel filterbank"

    safe_mkdir "$OUTPUT_DIR"

    local tmp_dir=""
    tmp_dir="$(mktemp -d)"

    # Safe cleanup trap using safe_rmrf
    cleanup() {
        safe_rmrf "${tmp_dir:-}"
    }
    trap cleanup EXIT

    # Download
    download_filterbank

    # Extract mel_80
    log_info "Extracting mel_80 (80x201)..."
    extract_npy "$MEL_FILTERS_NPZ" "mel_80" "$tmp_dir/mel_80.npy"
    parse_npy_to_bin "$tmp_dir/mel_80.npy" "$OUTPUT_DIR/mel_80.bin"

    # Extract mel_128
    log_info "Extracting mel_128 (128x201)..."
    extract_npy "$MEL_FILTERS_NPZ" "mel_128" "$tmp_dir/mel_128.npy"
    parse_npy_to_bin "$tmp_dir/mel_128.npy" "$OUTPUT_DIR/mel_128.bin"

    # Verify sizes
    local mel_80_size
    local mel_128_size
    mel_80_size="$(stat -c%s "$OUTPUT_DIR/mel_80.bin" 2>/dev/null || stat -f%z "$OUTPUT_DIR/mel_80.bin")"
    mel_128_size="$(stat -c%s "$OUTPUT_DIR/mel_128.bin" 2>/dev/null || stat -f%z "$OUTPUT_DIR/mel_128.bin")"

    if [[ "$mel_80_size" -ne 64320 ]]; then
        log_error "mel_80.bin size mismatch: expected 64320, got $mel_80_size"
        return 1
    fi

    if [[ "$mel_128_size" -ne 102912 ]]; then
        log_error "mel_128.bin size mismatch: expected 102912, got $mel_128_size"
        return 1
    fi

    log_info "Successfully extracted filterbanks to $OUTPUT_DIR"
    log_info "  mel_80.bin:  $mel_80_size bytes (80 x 201 x 4)"
    log_info "  mel_128.bin: $mel_128_size bytes (128 x 201 x 4)"

    # Generate Rust module if requested
    if [[ "${GENERATE_RUST:-0}" == "1" ]]; then
        local rust_file="$PROJECT_ROOT/src/audio/mel_filterbank_data.rs"
        log_info "Generating Rust constants: $rust_file"
        {
            echo "//! Pre-computed mel filterbank from OpenAI Whisper"
            echo "//!"
            echo "//! Generated by: scripts/extract_filterbank.sh"
            echo "//! Source: $MEL_FILTERS_URL"
            echo ""
            echo "pub const N_MELS_80: usize = 80;"
            echo "pub const N_MELS_128: usize = 128;"
            echo "pub const N_FREQS: usize = 201;"
            echo ""
            generate_rust_const "$OUTPUT_DIR/mel_80.bin" "MEL_80" 80 201
            generate_rust_const "$OUTPUT_DIR/mel_128.bin" "MEL_128" 128 201
        } > "$rust_file"
        log_info "Generated: $rust_file"
    fi
}

main "$@"
