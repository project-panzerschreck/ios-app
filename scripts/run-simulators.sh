#!/usr/bin/env bash
# run-simulators.sh
# Builds the RPC worker app and launches it on three iOS simulators,
# each on a different port so they can all serve as RPC workers simultaneously.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT="$PROJECT_DIR/distributed-ml-ggml-client-ios.xcodeproj"
SCHEME="distributed-ml-ggml-client-ios"
BUNDLE_ID="sreehal.distributed-ml-ggml-client-ios"
BUILD_DIR="$PROJECT_DIR/build-sim"

# Simulators and their assigned RPC ports (parallel arrays)
SIM_NAMES=("ip17p"          "iPhone 17 Pro"                        "iPhone 17 Pro Max")
SIM_UDIDS=("4F439C99-BF27-4562-9E90-E0FB4915A1AB" \
           "E4FAA6CD-203B-4657-A21B-B0B0FE527BC8" \
           "FFFF63AC-6842-4EB5-A677-26081BCEFE2A")
SIM_PORTS=(50052 50053 50054)

# ── 1. Build ──────────────────────────────────────────────────────────────────
echo "==> Building $SCHEME for simulator..."
xcodebuild \
    -project "$PROJECT" \
    -scheme "$SCHEME" \
    -configuration Debug \
    -destination "generic/platform=iOS Simulator" \
    -derivedDataPath "$BUILD_DIR" \
    build \
    | xcpretty 2>/dev/null || xcodebuild \
        -project "$PROJECT" \
        -scheme "$SCHEME" \
        -configuration Debug \
        -destination "generic/platform=iOS Simulator" \
        -derivedDataPath "$BUILD_DIR" \
        build

# Locate the .app bundle
APP_PATH=$(find "$BUILD_DIR" -name "${SCHEME}.app" -path "*/Debug-iphonesimulator/*" | head -1)
if [[ -z "$APP_PATH" ]]; then
    echo "ERROR: Could not find built .app in $BUILD_DIR"
    exit 1
fi
echo "==> App bundle: $APP_PATH"

# ── 2. Boot, install, configure, launch ──────────────────────────────────────
for i in 0 1 2; do
    NAME="${SIM_NAMES[$i]}"
    UDID="${SIM_UDIDS[$i]}"
    PORT="${SIM_PORTS[$i]}"

    echo ""
    echo "==> [$NAME] Booting $UDID..."
    xcrun simctl boot "$UDID" 2>/dev/null || true   # ignore "already booted"

    echo "==> [$NAME] Installing app..."
    xcrun simctl install "$UDID" "$APP_PATH"

    echo "==> [$NAME] Setting RPC port to $PORT..."
    xcrun simctl spawn "$UDID" defaults write "$BUNDLE_ID" rpcPort -int "$PORT"

    echo "==> [$NAME] Launching app..."
    xcrun simctl launch "$UDID" "$BUNDLE_ID"
done

# Open Simulator.app so all windows are visible
open -a Simulator

# ── 3. Print llama-cli command ────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "All simulators launched. RPC ports:"
for i in 0 1 2; do
    echo "  ${SIM_NAMES[$i]}  ->  127.0.0.1:${SIM_PORTS[$i]}"
done
echo ""
echo "Run inference with:"
echo ""
echo "  ./build-mac-rpc-only/bin/llama-cli \\"
echo "    --rpc 127.0.0.1:50052,127.0.0.1:50053,127.0.0.1:50054 \\"
echo "    -m ./Models/SmolLM2-135M-Instruct-Q4_K_M.gguf \\"
echo "    -ngl 99 -p \"Your prompt here\" -no-cnv"
echo "========================================================"
