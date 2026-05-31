#!/usr/bin/env bash
#
# build-ggml-ios.sh
#
# Uses llama.cpp-rpc (our GGML RPC runtime) and builds device-only XCFrameworks
# for iPhone/iPad hardware (arm64 + Metal).
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
# The Header Search Paths in the project point to
#   ../llama.cpp-rpc/include
#   ../llama.cpp-rpc/ggml/include
#
# Requirements:
#   brew install cmake          # cmake >= 3.24
#   xcode-select --install      # Xcode command-line tools

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEFAULT_LLAMA_RPC_DIR="$PROJECT_DIR/../llama.cpp-rpc"
LLAMA_RPC_GIT_SSH="git@github.com:rmcluster/llama.cpp-rpc.git"
LLAMA_RPC_GIT_HTTPS="https://github.com/rmcluster/llama.cpp-rpc.git"
if [[ -n "${LLAMA_DIR:-}" ]]; then
    LLAMA_DIR="$LLAMA_DIR"
elif [[ -d "$DEFAULT_LLAMA_RPC_DIR" ]]; then
    LLAMA_DIR="$DEFAULT_LLAMA_RPC_DIR"
else
    LLAMA_DIR="$DEFAULT_LLAMA_RPC_DIR"
fi
BUILD_BASE="$PROJECT_DIR/build-llama"
OUTPUT_DIR="$PROJECT_DIR/Frameworks"
IOS_MIN="15.6"

log()  { echo "[build-ggml-ios] $*"; }
die()  { echo "[build-ggml-ios] ERROR: $*" >&2; exit 1; }

reset_build_dir_if_needed() {
    local build_dir="$1"
    local cache_file="$build_dir/CMakeCache.txt"

    if [[ ! -f "$cache_file" ]]; then
        return 0
    fi

    local cached_source
    cached_source="$(awk -F= '/^CMAKE_HOME_DIRECTORY:INTERNAL=/{print $2}' "$cache_file" | tail -1)"
    if [[ -n "$cached_source" && "$cached_source" != "$LLAMA_DIR" ]]; then
        log "Removing stale CMake cache in $build_dir"
        log "  cached source: $cached_source"
        log "  requested source: $LLAMA_DIR"
        rm -rf "$build_dir"
    fi
}

ensure_llama_rpc_checkout() {
    if [[ -d "$LLAMA_DIR/.git" ]]; then
        log "Using existing llama.cpp-rpc checkout at $LLAMA_DIR"
        return 0
    fi

    if [[ -f "$LLAMA_DIR/CMakeLists.txt" && -d "$LLAMA_DIR/include" && -d "$LLAMA_DIR/ggml/include" ]]; then
        log "Using existing llama.cpp-rpc source tree at $LLAMA_DIR"
        return 0
    fi

    if [[ -e "$LLAMA_DIR" ]]; then
        die "$LLAMA_DIR exists but does not look like a llama.cpp-rpc checkout. Remove it or set LLAMA_DIR explicitly."
    fi

    mkdir -p "$(dirname "$LLAMA_DIR")"
    log "Cloning llama.cpp-rpc into $LLAMA_DIR …"
    git clone "$LLAMA_RPC_GIT_SSH" "$LLAMA_DIR" \
        || git clone "$LLAMA_RPC_GIT_HTTPS" "$LLAMA_DIR" \
        || die "Failed to clone llama.cpp-rpc from both SSH and HTTPS remotes."
}

verify_llama_rpc_layout() {
    if [[ ! -d "$LLAMA_DIR/ggml/include" ]]; then
        die "ggml headers not found under $LLAMA_DIR"
    fi

    if [[ ! -d "$LLAMA_DIR/include" ]]; then
        die "llama public headers not found under $LLAMA_DIR/include. Set LLAMA_DIR to a checkout that exposes the app-facing headers."
    fi
}

# ── Dependency checks ─────────────────────────────────────────────────────────
command -v cmake       >/dev/null 2>&1 || die "cmake not found – run: brew install cmake"
command -v git         >/dev/null 2>&1 || die "git not found"
command -v xcodebuild  >/dev/null 2>&1 || die "xcodebuild not found – install Xcode"
XCODE_PATH=$(xcode-select -p 2>/dev/null) || die "Xcode CLT not installed (xcode-select --install)"
log "Xcode at: $XCODE_PATH"

CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
log "cmake version: $CMAKE_VER"

# ── Ensure llama.cpp-rpc source tree exists ───────────────────────────────────
ensure_llama_rpc_checkout
verify_llama_rpc_layout

# ── cmake configure + build for the device slice ─────────────────────────────
build_slice() {
    local name="$1"     # "iphoneos"
    local archs="$2"    # "arm64"
    local metal="$3"    # "ON"
    local sdk="$4"      # "iphoneos"
    local build_dir="$BUILD_BASE/$name"

    log "── Configuring $name (archs: $archs, Metal: $metal) …"
    reset_build_dir_if_needed "$build_dir"
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
    # Build only library targets — executables have no bundle ID on iOS
    for target in ggml ggml-base ggml-cpu ggml-blas ggml-rpc ggml-metal llama; do
        cmake --build "$build_dir" \
            --config Release \
            --target "$target" \
            --parallel "$(sysctl -n hw.logicalcpu)" \
            -- \
            -sdk "$sdk" \
            ARCHS="$archs" \
            ONLY_ACTIVE_ARCH=NO \
            2>&1 | grep -E "(error:|Build succeeded|FAILED)" || true
    done
}

build_slice "iphoneos"      "arm64"          "ON"  "iphoneos"

# ── Locate a static library in a build tree ───────────────────────────────────
find_lib() {
    local build_dir="$1"
    local libname="$2"
    find "$build_dir" \
        \( -path "*/Release-iphoneos/lib${libname}.a" \
        -o -path "*/Release/lib${libname}.a" \) \
        2>/dev/null | head -1
}

# ── Build one device-only XCFramework (header-free static library) ───────────
# Headers are NOT embedded in the xcframework.  Xcode finds them via the
# HEADER_SEARCH_PATHS build setting:
#   $(PROJECT_DIR)/../llama.cpp-rpc/include
#   $(PROJECT_DIR)/../llama.cpp-rpc/ggml/include
#
# Embedding the same ggml/include headers in every xcframework causes
# "Multiple commands produce …/include/ggml.h" conflicts at build time.
make_xcframework() {
    local libname="$1"
    local output="$OUTPUT_DIR/${libname}.xcframework"

    local dev_lib
    dev_lib=$(find_lib "$BUILD_BASE/iphoneos" "$libname")

    if [[ -z "$dev_lib" ]]; then
        log "  ⚠ Skipping $libname.xcframework — device lib not found."
        return 0
    fi

    log "Creating device-only $libname.xcframework …"
    rm -rf "$output"
    xcodebuild -create-xcframework \
        -library "$dev_lib" \
        -output  "$output"
}

mkdir -p "$OUTPUT_DIR"

make_xcframework "llama"
make_xcframework "ggml"
make_xcframework "ggml-base"
make_xcframework "ggml-cpu"
make_xcframework "ggml-blas"
make_xcframework "ggml-rpc"   # RPC backend (enabled by GGML_RPC=ON above)

# ggml-metal: package the real device lib only.
dev_metal="$BUILD_BASE/iphoneos/ggml/src/ggml-metal/Release-iphoneos/libggml-metal.a"
if [[ -f "$dev_metal" ]]; then
    log "Creating device-only ggml-metal.xcframework …"
    rm -rf "$OUTPUT_DIR/ggml-metal.xcframework"
    xcodebuild -create-xcframework \
        -library "$dev_metal" \
        -output  "$OUTPUT_DIR/ggml-metal.xcframework"
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
