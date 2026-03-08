#!/usr/bin/env bash
#
# build-ggml-ios.sh
#
# Clones llama.cpp (the primary GGML-based LLM runtime) and builds it as
# XCFrameworks for iOS device (arm64 + Metal) and iOS Simulator (arm64 + x86_64).
#
# Usage:
#   cd 2026_ver/distributed-ml-ggml-client-ios
#   bash scripts/build-ggml-ios.sh
#
# Output (after running):
#   Frameworks/
#     llama.xcframework      ← main llama API  (link this first)
#     ggml.xcframework       ← GGML core
#     ggml-base.xcframework  ← GGML utilities
#     ggml-cpu.xcframework   ← CPU backend
#
# After running, in Xcode:
#   Target → General → Frameworks, Libraries, and Embedded Content
#   → (+) → Add Other → Add Files → pick each .xcframework
#   → Set each to "Do Not Embed"
#
# The Header Search Paths in the project already point to
#   vendor/llama.cpp/include   (populated by this script)
#
# Requirements:
#   brew install cmake          # cmake >= 3.24
#   xcode-select --install      # Xcode command-line tools

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="$PROJECT_DIR/vendor/llama.cpp"
BUILD_BASE="$PROJECT_DIR/build-llama"
OUTPUT_DIR="$PROJECT_DIR/Frameworks"
IOS_MIN="17.0"

# ── Pin a specific llama.cpp release ─────────────────────────────────────────
# Update this tag to upgrade.  Find tags at https://github.com/ggml-org/llama.cpp/tags
LLAMA_TAG="b5076"

log()  { echo "[build-ggml-ios] $*"; }
die()  { echo "[build-ggml-ios] ERROR: $*" >&2; exit 1; }

# ── Dependency checks ─────────────────────────────────────────────────────────
command -v cmake       >/dev/null 2>&1 || die "cmake not found – run: brew install cmake"
command -v git         >/dev/null 2>&1 || die "git not found"
command -v xcodebuild  >/dev/null 2>&1 || die "xcodebuild not found – install Xcode"
XCODE_PATH=$(xcode-select -p 2>/dev/null) || die "Xcode CLT not installed (xcode-select --install)"
log "Xcode at: $XCODE_PATH"

CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
log "cmake version: $CMAKE_VER"

# ── Clone / update llama.cpp ──────────────────────────────────────────────────
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    log "Cloning llama.cpp @ $LLAMA_TAG …"
    mkdir -p "$(dirname "$LLAMA_DIR")"
    # Try tagged clone first; fall back to HEAD if tag doesn't exist yet
    git clone --depth 1 --branch "$LLAMA_TAG" \
        https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR" \
    || git clone --depth 1 \
        https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
else
    log "llama.cpp already at $LLAMA_DIR  (to update: cd $LLAMA_DIR && git pull)"
fi

# ── cmake configure + build for one platform slice ───────────────────────────
build_slice() {
    local name="$1"     # "iphoneos" | "iphonesimulator"
    local archs="$2"    # "arm64"    | "arm64 x86_64"
    local metal="$3"    # "ON"       | "OFF"
    local sdk="$4"      # "iphoneos" | "iphonesimulator"
    local build_dir="$BUILD_BASE/$name"

    log "── Configuring $name (archs: $archs, Metal: $metal) …"
    mkdir -p "$build_dir"

    cmake -S "$LLAMA_DIR" -B "$build_dir" \
        -G Xcode \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_ARCHITECTURES="$archs" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN" \
        -DGGML_METAL="$metal" \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DGGML_RPC=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_CURL=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
        -DCMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE=NO \
        -DCMAKE_XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET="$IOS_MIN" \
        -Wno-dev \
        2>&1 | tail -5

    log "── Building $name …"
    # Pass SDK and arch overrides to xcodebuild via cmake
    cmake --build "$build_dir" \
        --config Release \
        --parallel "$(sysctl -n hw.logicalcpu)" \
        -- \
        -sdk "$sdk" \
        ARCHS="$archs" \
        ONLY_ACTIVE_ARCH=NO \
        2>&1 | grep -E "(error:|Build succeeded|FAILED|warning: (deprecated|unused))" || true
}

build_slice "iphoneos"      "arm64"          "ON"  "iphoneos"
build_slice "iphonesimulator" "arm64 x86_64" "OFF" "iphonesimulator"

# ── Locate a static library in a build tree ───────────────────────────────────
find_lib() {
    local build_dir="$1"
    local libname="$2"
    find "$build_dir" \
        \( -path "*/Release-iphoneos/lib${libname}.a"      \
        -o -path "*/Release-iphonesimulator/lib${libname}.a" \
        -o -path "*/Release/lib${libname}.a" \) \
        2>/dev/null | head -1
}

# ── Build one XCFramework (header-free static library) ───────────────────────
# Headers are NOT embedded in the xcframework.  Xcode finds them via the
# HEADER_SEARCH_PATHS build setting:
#   $(PROJECT_DIR)/vendor/llama.cpp/include
#   $(PROJECT_DIR)/vendor/llama.cpp/ggml/include
#
# Embedding the same ggml/include headers in every xcframework causes
# "Multiple commands produce …/include/ggml.h" conflicts at build time.
make_xcframework() {
    local libname="$1"
    local output="$OUTPUT_DIR/${libname}.xcframework"

    local dev_lib sim_lib
    dev_lib=$(find_lib "$BUILD_BASE/iphoneos"        "$libname")
    sim_lib=$(find_lib "$BUILD_BASE/iphonesimulator"  "$libname")

    if [[ -z "$dev_lib" ]]; then
        log "  ⚠ Skipping $libname.xcframework — device lib not found."
        return 0
    fi
    if [[ -z "$sim_lib" ]]; then
        log "  ⚠ Skipping $libname.xcframework — simulator lib not found."
        return 0
    fi

    log "Creating $libname.xcframework …"
    rm -rf "$output"
    xcodebuild -create-xcframework \
        -library "$dev_lib" \
        -library "$sim_lib" \
        -output  "$output"
}

mkdir -p "$OUTPUT_DIR"

make_xcframework "llama"
make_xcframework "ggml"
make_xcframework "ggml-base"
make_xcframework "ggml-cpu"
make_xcframework "ggml-blas"
make_xcframework "ggml-rpc"   # RPC backend (enabled by GGML_RPC=ON above)

# ggml-metal: real device lib + simulator stub.
# Metal was compiled out for the simulator (-DGGML_METAL=OFF), so the simulator
# ggml.a never references ggml_backend_metal_reg.  We add a stub .a so that
# xcodebuild doesn't error with "no library for this platform".
dev_metal="$BUILD_BASE/iphoneos/ggml/src/ggml-metal/Release-iphoneos/libggml-metal.a"
if [[ -f "$dev_metal" ]]; then
    log "Creating ggml-metal simulator stub …"
    SIM_STUB_C="$(mktemp /tmp/ggml_metal_stub_XXXXXX.c)"
    echo 'void ggml_metal_sim_placeholder(void){}' > "$SIM_STUB_C"

    xcrun --sdk iphonesimulator clang -arch arm64 \
        -target arm64-apple-ios${IOS_MIN}-simulator \
        -c "$SIM_STUB_C" -o "${SIM_STUB_C%.c}_arm64.o"
    xcrun --sdk iphonesimulator clang -arch x86_64 \
        -target x86_64-apple-ios${IOS_MIN}-simulator \
        -c "$SIM_STUB_C" -o "${SIM_STUB_C%.c}_x86_64.o"

    xcrun ar rcs "${SIM_STUB_C%.c}_arm64.a"  "${SIM_STUB_C%.c}_arm64.o"
    xcrun ar rcs "${SIM_STUB_C%.c}_x86_64.a" "${SIM_STUB_C%.c}_x86_64.o"
    lipo -create "${SIM_STUB_C%.c}_arm64.a" "${SIM_STUB_C%.c}_x86_64.a" \
         -output "${SIM_STUB_C%.c}_sim.a"

    log "Creating ggml-metal.xcframework (device + simulator stub) …"
    rm -rf "$OUTPUT_DIR/ggml-metal.xcframework"
    xcodebuild -create-xcframework \
        -library "$dev_metal" \
        -library "${SIM_STUB_C%.c}_sim.a" \
        -output  "$OUTPUT_DIR/ggml-metal.xcframework"

    rm -f "$SIM_STUB_C" "${SIM_STUB_C%.c}"_*.{c,o,a}
else
    log "  ⚠ Skipping ggml-metal.xcframework — device lib not found."
fi

# ── Verify output ─────────────────────────────────────────────────────────────
log ""
log "✓ XCFrameworks written to: $OUTPUT_DIR"
ls -1 "$OUTPUT_DIR" | sed 's/^/    /'
log ""
log "══════════════════════════════════════════════════════════════"
log "Next steps in Xcode:"
log "  1. Open the .xcodeproj."
log "  2. Select the 'distributed-ml-ggml-client-ios' target."
log "  3. General → Frameworks, Libraries, and Embedded Content → (+)"
log "  4. Add Other → Add Files → select each .xcframework"
log "     from: $OUTPUT_DIR"
log "  5. Set each framework to 'Do Not Embed' (static libs)."
log "  6. Make sure ggml-rpc.xcframework is added (enables RPC worker mode)."
log "  7. Build (⌘B)."
log "══════════════════════════════════════════════════════════════"
log ""
log "To download a GPT-2 GGUF model for quick testing:"
log "  bash scripts/download-gpt2.sh"
