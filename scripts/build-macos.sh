#!/bin/bash
# macOS Build Script for Whisper.apr
# WAPR-PERF-002: Click-to-Run WASM Package
#
# This script creates a macOS .app bundle with embedded WASM and model.
# Supports universal binary (x86_64 + arm64) for all Macs.
#
# Usage:
#   ./scripts/build-macos.sh [--skip-wasm] [--skip-sign] [--skip-dmg]
#
# Requirements:
#   - Rust toolchain with x86_64-apple-darwin and aarch64-apple-darwin targets
#   - wasm-pack (cargo install wasm-pack)
#   - create-dmg (brew install create-dmg)

set -euo pipefail

# Configuration
APP_NAME="Whisper.apr"
BUNDLE_ID="com.paiml.whisper-apr"
VERSION="${VERSION:-1.0.0}"
MIN_MACOS="10.15"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$PROJECT_DIR/target"
APP_DIR="$TARGET_DIR/${APP_NAME}.app"
CONTENTS_DIR="$APP_DIR/Contents"

# Parse arguments
SKIP_WASM=false
SKIP_SIGN=false
SKIP_DMG=false

for arg in "$@"; do
    case $arg in
        --skip-wasm) SKIP_WASM=true ;;
        --skip-sign) SKIP_SIGN=true ;;
        --skip-dmg) SKIP_DMG=true ;;
        --help)
            echo "Usage: $0 [--skip-wasm] [--skip-sign] [--skip-dmg]"
            exit 0
            ;;
    esac
done

echo "========================================"
echo "  Building ${APP_NAME} v${VERSION}"
echo "========================================"
echo ""

# Step 1: Build WASM
if [ "$SKIP_WASM" = false ]; then
    echo "[1/6] Building WASM..."
    cd "$PROJECT_DIR"
    wasm-pack build --target web --release --features wasm

    if [ ! -f "pkg/whisper_apr_bg.wasm" ]; then
        echo "ERROR: WASM build failed - pkg/whisper_apr_bg.wasm not found"
        exit 1
    fi

    WASM_SIZE=$(du -h "pkg/whisper_apr_bg.wasm" | cut -f1)
    echo "    WASM size: $WASM_SIZE"
else
    echo "[1/6] Skipping WASM build..."
fi

# Step 2: Build native launcher
echo "[2/6] Building native launcher..."
cd "$PROJECT_DIR"

# Check if we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "    WARNING: Not on macOS, skipping native build"
    echo "    Run this script on macOS or via 'ssh mac'"
else
    # Build for x86_64
    if rustup target list --installed | grep -q "x86_64-apple-darwin"; then
        echo "    Building for x86_64..."
        cargo build --release --bin whisper-apr-launcher --target x86_64-apple-darwin 2>/dev/null || \
            echo "    Note: whisper-apr-launcher binary not yet implemented"
    fi

    # Build for aarch64 (Apple Silicon)
    if rustup target list --installed | grep -q "aarch64-apple-darwin"; then
        echo "    Building for aarch64..."
        cargo build --release --bin whisper-apr-launcher --target aarch64-apple-darwin 2>/dev/null || \
            echo "    Note: whisper-apr-launcher binary not yet implemented"
    fi
fi

# Step 3: Create universal binary
echo "[3/6] Creating universal binary..."
mkdir -p "$TARGET_DIR/release"

if [[ "$(uname)" == "Darwin" ]]; then
    X86_BIN="$TARGET_DIR/x86_64-apple-darwin/release/whisper-apr-launcher"
    ARM_BIN="$TARGET_DIR/aarch64-apple-darwin/release/whisper-apr-launcher"
    UNIVERSAL_BIN="$TARGET_DIR/release/whisper-apr-launcher-universal"

    if [ -f "$X86_BIN" ] && [ -f "$ARM_BIN" ]; then
        lipo -create "$X86_BIN" "$ARM_BIN" -output "$UNIVERSAL_BIN"
        echo "    Created universal binary"
    elif [ -f "$X86_BIN" ]; then
        cp "$X86_BIN" "$UNIVERSAL_BIN"
        echo "    Using x86_64 binary only"
    elif [ -f "$ARM_BIN" ]; then
        cp "$ARM_BIN" "$UNIVERSAL_BIN"
        echo "    Using aarch64 binary only"
    else
        echo "    WARNING: No launcher binary found, creating placeholder"
        echo '#!/bin/bash' > "$UNIVERSAL_BIN"
        echo 'echo "Launcher not yet implemented"' >> "$UNIVERSAL_BIN"
        chmod +x "$UNIVERSAL_BIN"
    fi
else
    echo "    Skipping (not on macOS)"
fi

# Step 4: Create app bundle structure
echo "[4/6] Creating app bundle..."
rm -rf "$APP_DIR"
mkdir -p "$CONTENTS_DIR/MacOS"
mkdir -p "$CONTENTS_DIR/Resources"
mkdir -p "$CONTENTS_DIR/Frameworks"

# Copy launcher
if [ -f "$TARGET_DIR/release/whisper-apr-launcher-universal" ]; then
    cp "$TARGET_DIR/release/whisper-apr-launcher-universal" "$CONTENTS_DIR/MacOS/whisper-apr-launcher"
    chmod +x "$CONTENTS_DIR/MacOS/whisper-apr-launcher"
fi

# Copy WASM and resources
if [ -f "$PROJECT_DIR/pkg/whisper_apr_bg.wasm" ]; then
    cp "$PROJECT_DIR/pkg/whisper_apr_bg.wasm" "$CONTENTS_DIR/Resources/whisper-apr.wasm"
fi

if [ -f "$PROJECT_DIR/pkg/whisper_apr.js" ]; then
    cp "$PROJECT_DIR/pkg/whisper_apr.js" "$CONTENTS_DIR/Resources/"
fi

# Copy web UI if it exists
if [ -d "$PROJECT_DIR/demos/www-demo" ]; then
    cp -r "$PROJECT_DIR/demos/www-demo/"* "$CONTENTS_DIR/Resources/" 2>/dev/null || true
fi

# Copy model if it exists
if [ -f "$PROJECT_DIR/models/whisper-tiny.apr" ]; then
    cp "$PROJECT_DIR/models/whisper-tiny.apr" "$CONTENTS_DIR/Resources/"
fi

# Create Info.plist
cat > "$CONTENTS_DIR/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>whisper-apr-launcher</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>Whisper.apr</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>WAPR</string>
    <key>LSMinimumSystemVersion</key>
    <string>${MIN_MACOS}</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Whisper.apr needs microphone access for real-time speech transcription.</string>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright © 2024-2026 PAIML. MIT License.</string>
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeName</key>
            <string>Audio File</string>
            <key>CFBundleTypeRole</key>
            <string>Viewer</string>
            <key>LSItemContentTypes</key>
            <array>
                <string>public.audio</string>
                <string>public.mp3</string>
                <string>public.mpeg-4-audio</string>
                <string>com.microsoft.waveform-audio</string>
            </array>
        </dict>
    </array>
</dict>
</plist>
EOF

echo "    Created app bundle at: $APP_DIR"

# Step 5: Code sign
if [ "$SKIP_SIGN" = false ] && [[ "$(uname)" == "Darwin" ]]; then
    echo "[5/6] Code signing..."
    codesign --force --deep --sign - "$APP_DIR" 2>/dev/null || \
        echo "    WARNING: Code signing failed (ad-hoc signing)"
else
    echo "[5/6] Skipping code signing..."
fi

# Step 6: Create DMG
if [ "$SKIP_DMG" = false ] && [[ "$(uname)" == "Darwin" ]]; then
    echo "[6/6] Creating DMG..."
    DMG_PATH="$TARGET_DIR/${APP_NAME}-${VERSION}.dmg"
    rm -f "$DMG_PATH"

    if command -v create-dmg &> /dev/null; then
        # Use create-dmg for fancy DMG with app-drop link
        create-dmg \
            --volname "${APP_NAME}" \
            --window-pos 200 120 \
            --window-size 600 400 \
            --icon-size 100 \
            --icon "${APP_NAME}.app" 150 185 \
            --app-drop-link 450 185 \
            "$DMG_PATH" \
            "$APP_DIR" 2>/dev/null || \
            echo "    WARNING: create-dmg failed, trying hdiutil..."
    fi

    # Fallback to hdiutil if create-dmg failed or not available
    if [ ! -f "$DMG_PATH" ]; then
        hdiutil create -volname "${APP_NAME}" \
            -srcfolder "$APP_DIR" \
            -ov -format UDZO \
            "$DMG_PATH" 2>/dev/null || \
            echo "    WARNING: DMG creation failed"
    fi

    if [ -f "$DMG_PATH" ]; then
        DMG_SIZE=$(du -h "$DMG_PATH" | cut -f1)
        echo "    Created DMG: $DMG_PATH ($DMG_SIZE)"
    fi
else
    echo "[6/6] Skipping DMG creation..."
fi

echo ""
echo "========================================"
echo "  Build Complete!"
echo "========================================"
echo ""
echo "App bundle: $APP_DIR"
if [ -f "$TARGET_DIR/${APP_NAME}-${VERSION}.dmg" ]; then
    echo "DMG file:   $TARGET_DIR/${APP_NAME}-${VERSION}.dmg"
fi
echo ""
echo "To test locally:"
echo "  open '$APP_DIR'"
echo ""
echo "To download from remote:"
echo "  scp mac:$TARGET_DIR/${APP_NAME}-${VERSION}.dmg ./"
