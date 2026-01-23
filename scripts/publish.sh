#!/usr/bin/env bash
# HuggingFace Publish Script (WAPR-PUB-001)
#
# Transpiled from: scripts/publish.rs
# Verify with: bashrs verify scripts/publish.rs scripts/publish.sh
#
# Usage:
#   ./scripts/publish.sh model.apr paiml/whisper-apr-tiny
#   ./scripts/publish.sh model.apr paiml/whisper-apr-tiny --format both
#   ./scripts/publish.sh model.apr paiml/whisper-apr-tiny --dry-run

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    cat << 'EOF'
Usage: publish.sh <model.apr> <repo-id> [OPTIONS]

Arguments:
    <model.apr>     Path to APR model file
    <repo-id>       HuggingFace repository ID (e.g., paiml/whisper-apr-tiny)

Options:
    --format <fmt>  Output format: apr, safetensors, both (default: both)
    --dry-run       Verify without uploading
    --skip-verify   Skip pre-publish verification
    --message <msg> Custom commit message
    --help          Show this help

Environment:
    HF_TOKEN        HuggingFace API token (required for upload)

Examples:
    # Publish to HuggingFace with both formats
    ./scripts/publish.sh whisper-tiny.apr paiml/whisper-apr-tiny

    # Dry run (verify only)
    ./scripts/publish.sh whisper-tiny.apr paiml/whisper-apr-tiny --dry-run

    # SafeTensors only
    ./scripts/publish.sh whisper-tiny.apr paiml/whisper-apr-tiny --format safetensors
EOF
}

log_step() {
    echo -e "${GREEN}$1${NC}"
}

log_warn() {
    echo -e "${YELLOW}      ⚠ $1${NC}"
}

log_error() {
    echo -e "${RED}Error: $1${NC}" >&2
}

log_success() {
    echo -e "      ${GREEN}✓${NC} $1"
}

command_exists() {
    command -v "$1" &> /dev/null
}

# Parse arguments
if [[ $# -lt 2 ]] || [[ "$1" == "--help" ]]; then
    print_usage
    [[ "$1" == "--help" ]] && exit 0 || exit 1
fi

MODEL_PATH="$1"
REPO_ID="$2"
shift 2

# Default options
FORMAT="both"
DRY_RUN=false
SKIP_VERIFY=false
MESSAGE="Upload ${MODEL_PATH} via whisper.apr publish"

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --format)
            FORMAT="$2"
            if [[ ! "$FORMAT" =~ ^(apr|safetensors|both)$ ]]; then
                log_error "Invalid format '$FORMAT'. Use: apr, safetensors, both"
                exit 1
            fi
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-verify)
            SKIP_VERIFY=true
            shift
            ;;
        --message)
            MESSAGE="$2"
            shift 2
            ;;
        *)
            log_warn "Unknown option '$1'"
            shift
            ;;
    esac
done

echo "=== whisper.apr Publish Workflow ==="
echo ""
echo "Model:    $MODEL_PATH"
echo "Repo:     $REPO_ID"
echo "Format:   $FORMAT"
echo "Dry-run:  $DRY_RUN"
echo ""

# Step 1: Verify model file exists
log_step "[1/6] Checking model file..."
if [[ ! -f "$MODEL_PATH" ]]; then
    log_error "Model file not found: $MODEL_PATH"
    exit 1
fi
log_success "Model file exists"

# Step 2: Check HF_TOKEN
log_step "[2/6] Checking authentication..."
if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ "$DRY_RUN" == "false" ]]; then
        log_error "HF_TOKEN environment variable not set"
        echo "       Set it with: export HF_TOKEN=hf_..." >&2
        exit 1
    else
        log_warn "HF_TOKEN not set (dry-run mode)"
    fi
else
    log_success "HF_TOKEN is set"
fi

# Step 3: Verify APR format
if [[ "$SKIP_VERIFY" == "false" ]]; then
    log_step "[3/6] Verifying APR format..."
    if command_exists whisper-apr-cli; then
        if ! whisper-apr-cli validate "$MODEL_PATH"; then
            log_error "APR verification failed"
            exit 1
        fi
        log_success "APR format valid"
    else
        # Fallback: check magic bytes manually
        MAGIC=$(head -c 4 "$MODEL_PATH" | od -A n -t x1 | tr -d ' \n')
        if [[ "$MAGIC" == "41505200" ]]; then  # APR\0 in hex
            log_success "APR magic bytes valid"
        else
            log_error "Invalid APR magic bytes (expected APR\\0)"
            exit 1
        fi
    fi
else
    log_step "[3/6] Skipping verification (--skip-verify)"
fi

# Step 4: Export to SafeTensors if needed
SAFETENSORS_PATH="${MODEL_PATH%.apr}.safetensors"
if [[ "$FORMAT" == "safetensors" ]] || [[ "$FORMAT" == "both" ]]; then
    log_step "[4/6] Exporting to SafeTensors..."
    if command_exists whisper-apr-cli; then
        if whisper-apr-cli export "$MODEL_PATH" -o "$SAFETENSORS_PATH"; then
            log_success "Exported to $SAFETENSORS_PATH"
        else
            log_error "SafeTensors export failed"
            exit 1
        fi
    else
        log_warn "whisper-apr-cli not found, skipping export"
        echo "        Install with: cargo install --path ."
    fi
else
    log_step "[4/6] Skipping SafeTensors export (format=apr)"
fi

# Step 5: Sign models (if pacha available)
log_step "[5/6] Signing models..."
if command_exists batuta; then
    if batuta pacha sign "$MODEL_PATH" 2>/dev/null; then
        log_success "Model signed"
    else
        log_warn "Signing skipped (pacha not configured)"
    fi
else
    log_warn "batuta not found, skipping signing"
fi

# Step 6: Upload to HuggingFace
log_step "[6/6] Uploading to HuggingFace..."
if [[ "$DRY_RUN" == "true" ]]; then
    log_warn "Dry-run mode, skipping upload"
    echo ""
    echo "      Would upload:"
    if [[ "$FORMAT" == "apr" ]] || [[ "$FORMAT" == "both" ]]; then
        echo "        - $MODEL_PATH"
    fi
    if [[ "$FORMAT" == "safetensors" ]] || [[ "$FORMAT" == "both" ]]; then
        echo "        - $SAFETENSORS_PATH"
    fi
    echo "      To: https://huggingface.co/$REPO_ID"
else
    # Determine files to upload
    FILES_TO_UPLOAD=()
    case "$FORMAT" in
        apr)
            FILES_TO_UPLOAD=("$MODEL_PATH")
            ;;
        safetensors)
            FILES_TO_UPLOAD=("$SAFETENSORS_PATH")
            ;;
        both)
            FILES_TO_UPLOAD=("$MODEL_PATH" "$SAFETENSORS_PATH")
            ;;
    esac

    if command_exists huggingface-cli; then
        # Use huggingface-cli for multi-file upload
        for FILE in "${FILES_TO_UPLOAD[@]}"; do
            echo "      Uploading $(basename "$FILE")..."
            if ! huggingface-cli upload "$REPO_ID" "$FILE" --commit-message "$MESSAGE"; then
                log_error "Upload failed for $FILE"
                exit 1
            fi
        done
    elif command_exists batuta; then
        # Fallback to batuta (one file at a time)
        for FILE in "${FILES_TO_UPLOAD[@]}"; do
            echo "      Uploading $(basename "$FILE")..."
            if ! batuta hf push model "$FILE" --repo "$REPO_ID" --message "$MESSAGE"; then
                log_error "Upload failed for $FILE"
                exit 1
            fi
        done
    else
        log_error "No upload tool found"
        echo "       Install huggingface-cli or batuta" >&2
        exit 1
    fi
    log_success "Uploaded to https://huggingface.co/$REPO_ID"
fi

echo ""
echo "=== Publish Complete ==="
if [[ "$DRY_RUN" == "false" ]]; then
    echo "View at: https://huggingface.co/$REPO_ID"
fi
