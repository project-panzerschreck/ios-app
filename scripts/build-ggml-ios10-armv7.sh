#!/usr/bin/env bash
#
# build-ggml-ios10-armv7.sh
#
# Builds separate static libraries for iOS 10.3 armv7/armv7s without touching
# the existing modern arm64 XCFramework output in Frameworks/.
#
# Output:
#   armv7-rebuild-ios10/
#     lib/
#     vendor/llama.cpp/include/
#     vendor/llama.cpp/ggml/include/
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_ROOT="$PROJECT_DIR/armv7-rebuild-ios10"
BUILD_ROOT="$OUTPUT_ROOT/build"
LIB_OUTPUT_DIR="$OUTPUT_ROOT/lib"
VENDOR_OUTPUT_DIR="$OUTPUT_ROOT/vendor/llama.cpp"
SOURCE_ROOT="$OUTPUT_ROOT/src/llama.cpp-rpc-ios10"
IOS_MIN="10.3"
ARCHS=("armv7" "armv7s")
LLAMA_TAG="b5076"
LLAMA_RPC_GIT="${LLAMA_RPC_GIT:-git@github.com:rmcluster/llama.cpp-rpc.git}"
LLAMA_RPC_GIT_HTTPS="${LLAMA_RPC_GIT_HTTPS:-https://github.com/rmcluster/llama.cpp-rpc.git}"
LLAMA_BRANCH="${LLAMA_BRANCH:-iphone-5-build}"

if [[ -d "$PROJECT_DIR/../llama.cpp-rpc/.git" ]]; then
    DEFAULT_LLAMA_DIR="$PROJECT_DIR/../llama.cpp-rpc"
else
    DEFAULT_LLAMA_DIR="$PROJECT_DIR/../llama.cpp"
fi
LLAMA_DIR="${LLAMA_DIR:-$DEFAULT_LLAMA_DIR}"

log() { echo "[build-ggml-ios10-armv7] $*"; }
die() { echo "[build-ggml-ios10-armv7] ERROR: $*" >&2; exit 1; }

command -v cmake >/dev/null 2>&1 || die "cmake not found"
command -v git >/dev/null 2>&1 || die "git not found"
command -v xcrun >/dev/null 2>&1 || die "xcrun not found"
command -v lipo >/dev/null 2>&1 || die "lipo not found"
command -v rsync >/dev/null 2>&1 || die "rsync not found"

LEGACY_TOOLCHAIN_BIN="${LEGACY_TOOLCHAIN_BIN:-${PREFIX:-}}"
LEGACY_TOOLCHAIN_LIBCXX="${LEGACY_TOOLCHAIN_LIBCXX:-}"
IOS10_SDK_ROOT="${IOS10_SDK_ROOT:-}"
if [[ -z "$IOS10_SDK_ROOT" && -n "${THEOS:-}" && -d "$THEOS/sdks/iPhoneOS10.3.sdk" ]]; then
    IOS10_SDK_ROOT="$THEOS/sdks/iPhoneOS10.3.sdk"
fi
if [[ -z "$IOS10_SDK_ROOT" && -d "$HOME/theos/sdks/iPhoneOS10.3.sdk" ]]; then
    IOS10_SDK_ROOT="$HOME/theos/sdks/iPhoneOS10.3.sdk"
fi

[[ -n "$IOS10_SDK_ROOT" ]] || die "Set IOS10_SDK_ROOT to an iPhoneOS10.3.sdk path (or install one under \$THEOS/sdks)."
[[ -d "$IOS10_SDK_ROOT" ]] || die "iOS 10 SDK not found at: $IOS10_SDK_ROOT"
[[ -f "$IOS10_SDK_ROOT/usr/lib/libSystem.tbd" || -f "$IOS10_SDK_ROOT/usr/lib/libSystem.dylib" ]] || die "SDK at $IOS10_SDK_ROOT does not look like a usable iPhoneOS SDK"

if [[ -n "$LEGACY_TOOLCHAIN_BIN" && -x "$LEGACY_TOOLCHAIN_BIN/clang" ]]; then
    log "Using legacy toolchain at $LEGACY_TOOLCHAIN_BIN"
    C_COMPILER="$LEGACY_TOOLCHAIN_BIN/clang"
    CXX_COMPILER="$LEGACY_TOOLCHAIN_BIN/clang++"
    AR_TOOL="$LEGACY_TOOLCHAIN_BIN/ar"
    RANLIB_TOOL="$LEGACY_TOOLCHAIN_BIN/ranlib"
else
    C_COMPILER="$(xcrun --find clang)"
    CXX_COMPILER="$(xcrun --find clang++)"
    AR_TOOL="$(xcrun --find ar)"
    RANLIB_TOOL="$(xcrun --find ranlib)"
fi

if [[ -z "$LEGACY_TOOLCHAIN_LIBCXX" && -n "$LEGACY_TOOLCHAIN_BIN" ]]; then
    LEGACY_TOOLCHAIN_ROOT="$(cd "$LEGACY_TOOLCHAIN_BIN/.." && pwd)"
    if [[ -d "$LEGACY_TOOLCHAIN_ROOT/include/c++/v1" ]]; then
        LEGACY_TOOLCHAIN_LIBCXX="$LEGACY_TOOLCHAIN_ROOT/include/c++/v1"
    fi
fi

if [[ -z "$LEGACY_TOOLCHAIN_LIBCXX" ]]; then
    MODERN_IOS_SDK_ROOT="$(xcrun --sdk iphoneos --show-sdk-path)"
    LIBCXX_INCLUDE_DIR="$MODERN_IOS_SDK_ROOT/usr/include/c++/v1"
else
    LIBCXX_INCLUDE_DIR="$LEGACY_TOOLCHAIN_LIBCXX"
fi

[[ -d "$LIBCXX_INCLUDE_DIR" ]] || die "libc++ headers not found at: $LIBCXX_INCLUDE_DIR"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    log "Cloning llama.cpp-rpc @ $LLAMA_BRANCH ..."
    mkdir -p "$(dirname "$LLAMA_DIR")"
    if ! git clone --depth 1 --branch "$LLAMA_BRANCH" "$LLAMA_RPC_GIT" "$LLAMA_DIR"; then
        log "SSH clone failed; retrying via HTTPS"
        git clone --depth 1 --branch "$LLAMA_BRANCH" "$LLAMA_RPC_GIT_HTTPS" "$LLAMA_DIR"
    fi
else
    log "Using existing llama.cpp-rpc checkout at $LLAMA_DIR"
    git -C "$LLAMA_DIR" fetch origin --prune
    git -C "$LLAMA_DIR" checkout "$LLAMA_BRANCH"
    git -C "$LLAMA_DIR" pull --ff-only origin "$LLAMA_BRANCH" || true
fi

git -C "$LLAMA_DIR" rev-parse --short HEAD >/dev/null 2>&1 \
    || die "llama.cpp-rpc checkout at $LLAMA_DIR is not usable"
log "llama.cpp-rpc source: $(git -C "$LLAMA_DIR" describe --tags --always --dirty 2>/dev/null || git -C "$LLAMA_DIR" rev-parse --short HEAD) @ $(git -C "$LLAMA_DIR" branch --show-current)"

mkdir -p "$LIB_OUTPUT_DIR" "$BUILD_ROOT" "$VENDOR_OUTPUT_DIR"
rm -rf "$VENDOR_OUTPUT_DIR/include" "$VENDOR_OUTPUT_DIR/ggml"
mkdir -p "$VENDOR_OUTPUT_DIR/ggml"

log "Preparing separate legacy source tree at $SOURCE_ROOT"
rm -rf "$SOURCE_ROOT"
mkdir -p "$(dirname "$SOURCE_ROOT")"
rsync -a --delete --exclude '.git' "$LLAMA_DIR/" "$SOURCE_ROOT/"

RPC_SOURCE="$SOURCE_ROOT/ggml/src/ggml-rpc/ggml-rpc.cpp"
if grep -q '#include <filesystem>' "$RPC_SOURCE" 2>/dev/null; then
log "Applying legacy iOS 10 filesystem patches to ggml sources"
python3 - <<'PY' "$SOURCE_ROOT/ggml/src/ggml-rpc/ggml-rpc.cpp"
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()

old_include = """#include <cstring>\n#include <fstream>\n#include <filesystem>\n#include <algorithm>\n"""
new_include = """#include <cstring>\n#include <fstream>\n#include <algorithm>\n"""
if old_include not in source:
    raise SystemExit("expected filesystem include block not found")
source = source.replace(old_include, new_include, 1)

old_namespace = """namespace fs = std::filesystem;\n\nstatic constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB\n"""
new_namespace = """static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB\n"""
if old_namespace not in source:
    raise SystemExit("expected filesystem namespace block not found")
source = source.replace(old_namespace, new_namespace, 1)

old_socket_block = """#ifdef _WIN32\ntypedef SOCKET sockfd_t;\nusing ssize_t = __int64;\n#else\ntypedef int sockfd_t;\n#endif\n\n// cross-platform socket\n"""
new_socket_block = """#ifdef _WIN32\ntypedef SOCKET sockfd_t;\nusing ssize_t = __int64;\n#else\ntypedef int sockfd_t;\n#endif\n\nstatic std::string rpc_join_path(const char * base, const char * leaf) {\n    std::string result = base ? base : \"\";\n    if (!result.empty() && result.back() != '/') {\n        result.push_back('/');\n    }\n    result += leaf;\n    return result;\n}\n\nstatic bool rpc_file_exists(const char * path) {\n#ifdef _WIN32\n    DWORD attrs = GetFileAttributesA(path);\n    return attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY);\n#else\n    return access(path, F_OK) == 0;\n#endif\n}\n\n// cross-platform socket\n"""
if old_socket_block not in source:
    raise SystemExit("expected socket typedef block not found")
source = source.replace(old_socket_block, new_socket_block, 1)

replacements = {
    "        fs::path cache_file = fs::path(cache_dir) / hash_str;\n        std::ofstream ofs(cache_file, std::ios::binary);\n        ofs.write((const char *)data, size);\n        GGML_LOG_INFO(\"[%s] saved to '%s'\\n\", __func__, cache_file.c_str());\n":
    "        std::string cache_file = rpc_join_path(cache_dir, hash_str);\n        std::ofstream ofs(cache_file.c_str(), std::ios::binary);\n        ofs.write((const char *)data, size);\n        GGML_LOG_INFO(\"[%s] saved to '%s'\\n\", __func__, cache_file.c_str());\n",
    "    fs::path cache_file = fs::path(cache_dir) / hash_str;\n    std::error_code ec;\n    if (!fs::exists(cache_file, ec)) {\n        return false;\n    }\n    std::ifstream ifs(cache_file, std::ios::binary);\n":
    "    std::string cache_file = rpc_join_path(cache_dir, hash_str);\n    if (!rpc_file_exists(cache_file.c_str())) {\n        return false;\n    }\n    std::ifstream ifs(cache_file.c_str(), std::ios::binary);\n",
}
for old, new in replacements.items():
    if old not in source:
        raise SystemExit("expected cache path block not found")
    source = source.replace(old, new, 1)

path.write_text(source)
PY
else
    log "Skipping legacy filesystem patches (fork branch already iOS 10 compatible)"
    if ! grep -q 'ggml_backend_rpc_stop_server' "$SOURCE_ROOT/ggml/include/ggml-rpc.h" 2>/dev/null; then
        die "Expected ggml_backend_rpc_stop_server in ggml-rpc.h; use rmcluster/llama.cpp-rpc @ iphone-5-build"
    fi
    python3 - <<'PY' "$RPC_SOURCE"
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
old = "    std::atomic<bool> stop_requested = false;\n"
new = "    std::atomic<bool> stop_requested;\n\n    rpc_server_runtime() : stop_requested(false) {}\n"
if old in source:
    source = source.replace(old, new, 1)
    path.write_text(source)
PY
    if [[ "${GGML_RPC_VERBOSE:-0}" == "1" ]]; then
        log "Applying GGML_RPC_VERBOSE patches (route printf/fprintf to GGML_LOG_*)"
        python3 - <<'PY' "$RPC_SOURCE"
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
source = source.replace(
    "static const char * RPC_DEBUG = std::getenv(\"GGML_RPC_DEBUG\");",
    "static const char * RPC_DEBUG = \"1\";",
    1,
)
source = source.replace('printf("', 'GGML_LOG_INFO("')
source = source.replace('fprintf(stderr, "', 'GGML_LOG_ERROR("')
path.write_text(source)
PY
    fi
fi

if grep -q '#include <filesystem>' "$SOURCE_ROOT/ggml/src/ggml-backend-reg.cpp" 2>/dev/null \
    || grep -q 'namespace fs = std::filesystem' "$SOURCE_ROOT/ggml/src/ggml-backend-dl.h" 2>/dev/null; then
log "Applying legacy iOS 10 backend-reg patches"
python3 - <<'PY' \
    "$SOURCE_ROOT/ggml/src/ggml-backend-dl.h" \
    "$SOURCE_ROOT/ggml/src/ggml-backend-dl.cpp" \
    "$SOURCE_ROOT/ggml/src/ggml-backend-reg.cpp"
from pathlib import Path
import sys

dl_h = Path(sys.argv[1])
dl_cpp = Path(sys.argv[2])
reg_cpp = Path(sys.argv[3])

source = dl_h.read_text()
source = source.replace("#include <filesystem>\n", "#include <string>\n", 1)
source = source.replace("namespace fs = std::filesystem;\n\n", "", 1)
source = source.replace("dl_handle * dl_load_library(const fs::path & path);\n", "dl_handle * dl_load_library(const char * path);\n", 1)
dl_h.write_text(source)

source = dl_cpp.read_text()
source = source.replace("dl_handle * dl_load_library(const fs::path & path) {\n", "dl_handle * dl_load_library(const char * path) {\n", 1)
source = source.replace("    HMODULE handle = LoadLibraryW(path.wstring().c_str());\n", "    HMODULE handle = LoadLibraryA(path);\n", 1)
source = source.replace("dl_handle * dl_load_library(const fs::path & path) {\n", "dl_handle * dl_load_library(const char * path) {\n", 1)
source = source.replace("    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);\n", "    dl_handle * handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);\n", 1)
dl_cpp.write_text(source)

source = reg_cpp.read_text()
source = source.replace("#include <filesystem>\n", "", 1)
source = source.replace("namespace fs = std::filesystem;\n\n", "", 1)

old_path_str = """static std::string path_str(const fs::path & path) {\n    try {\n#if defined(__cpp_lib_char8_t)\n        // C++20 and later: u8string() returns std::u8string\n        const std::u8string u8str = path.u8string();\n        return std::string(reinterpret_cast<const char *>(u8str.data()), u8str.size());\n#else\n        // C++17: u8string() returns std::string\n        return path.u8string();\n#endif\n    } catch (...) {\n        return std::string();\n    }\n}\n\n"""
if old_path_str not in source:
    raise SystemExit("expected ggml-backend-reg path_str block not found")
source = source.replace(old_path_str, "", 1)
source = source.replace("    ggml_backend_reg_t load_backend(const fs::path & path, bool silent) {\n", "    ggml_backend_reg_t load_backend(const char * path, bool silent) {\n", 1)

replacements = {
    "                GGML_LOG_ERROR(\"%s: failed to load %s: %s\\n\", __func__, path_str(path).c_str(), dl_error());\n":
    "                GGML_LOG_ERROR(\"%s: failed to load %s: %s\\n\", __func__, path ? path : \"\", dl_error());\n",
    "                GGML_LOG_INFO(\"%s: backend %s is not supported on this system\\n\", __func__, path_str(path).c_str());\n":
    "                GGML_LOG_INFO(\"%s: backend %s is not supported on this system\\n\", __func__, path ? path : \"\");\n",
    "                GGML_LOG_ERROR(\"%s: failed to find ggml_backend_init in %s\\n\", __func__, path_str(path).c_str());\n":
    "                GGML_LOG_ERROR(\"%s: failed to find ggml_backend_init in %s\\n\", __func__, path ? path : \"\");\n",
    "                        __func__, path_str(path).c_str());\n":
    "                        __func__, path ? path : \"\");\n",
    "                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);\n":
    "                        __func__, path ? path : \"\", reg->api_version, GGML_BACKEND_API_VERSION);\n",
    "        GGML_LOG_INFO(\"%s: loaded %s backend from %s\\n\", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());\n":
    "        GGML_LOG_INFO(\"%s: loaded %s backend from %s\\n\", __func__, ggml_backend_reg_name(reg), path ? path : \"\");\n",
}
for old, new in replacements.items():
    if old not in source:
        raise SystemExit(f"expected ggml-backend-reg replacement not found: {old!r}")
    source = source.replace(old, new, 1)

dynamic_start = source.find("// Dynamic loading\n")
if dynamic_start == -1:
    raise SystemExit("expected ggml-backend-reg dynamic loading section not found")
source = source[:dynamic_start] + """// Dynamic loading
ggml_backend_reg_t ggml_backend_load(const char * path) {
    if (path == nullptr || path[0] == '\\0') {
        return nullptr;
    }
    return get_reg().load_backend(path, false);
}

void ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, true);
}

void ggml_backend_load_all() {
}

void ggml_backend_load_all_from_path(const char * dir_path) {
    (void) dir_path;
}
"""
reg_cpp.write_text(source)
PY
else
    log "Skipping legacy backend-reg patches"
fi

cp -R "$SOURCE_ROOT/include" "$VENDOR_OUTPUT_DIR/"
cp -R "$SOURCE_ROOT/ggml/include" "$VENDOR_OUTPUT_DIR/ggml/"

rebuild_ggml_rpc_lib_only() {
    log "Rebuilding libggml-rpc.a only (legacy clang direct compile)"
    mkdir -p "$BUILD_ROOT"
    local objects=()
    for arch in "${ARCHS[@]}"; do
        local obj="$BUILD_ROOT/ggml-rpc-$arch.o"
        local target_triple="$arch-apple-ios$IOS_MIN"
        local cxx_flags="-arch $arch -target $target_triple -miphoneos-version-min=$IOS_MIN -isysroot $IOS10_SDK_ROOT -isystem $LIBCXX_INCLUDE_DIR -std=gnu++14 -fno-modules"
        log "Compiling ggml-rpc for $arch"
        "$CXX_COMPILER" -c $cxx_flags \
            -I"$SOURCE_ROOT/include" \
            -I"$SOURCE_ROOT/ggml/include" \
            -I"$SOURCE_ROOT/ggml/src" \
            -DGGML_USE_CPU \
            -O2 \
            -o "$obj" \
            "$SOURCE_ROOT/ggml/src/ggml-rpc/ggml-rpc.cpp"
        objects+=("$obj")
    done

    local fat_obj="$BUILD_ROOT/ggml-rpc.o"
    lipo -create "${objects[@]}" -output "$fat_obj"
    rm -f "$LIB_OUTPUT_DIR/libggml-rpc.a"
    "$AR_TOOL" -qc "$LIB_OUTPUT_DIR/libggml-rpc.a" "$fat_obj"
    # Legacy ranlib can hang on fat archives; system ranlib is fine for index tables.
    if [[ "${SKIP_RANLIB:-0}" != "1" ]]; then
        if command -v /usr/bin/ranlib >/dev/null 2>&1; then
            /usr/bin/ranlib "$LIB_OUTPUT_DIR/libggml-rpc.a" || true
        else
            "$RANLIB_TOOL" "$LIB_OUTPUT_DIR/libggml-rpc.a" || true
        fi
    fi
    log "Updated $LIB_OUTPUT_DIR/libggml-rpc.a with ggml_backend_rpc_stop_server"
}

build_arch() {
    local arch="$1"
    local build_dir="$BUILD_ROOT/$arch"
    local target_triple="$arch-apple-ios$IOS_MIN"
    local common_flags="-arch $arch -target $target_triple -miphoneos-version-min=$IOS_MIN -isysroot $IOS10_SDK_ROOT"
    local cxx_common_flags="$common_flags -isystem $LIBCXX_INCLUDE_DIR"

    log "Configuring $arch against SDK: $IOS10_SDK_ROOT"
    rm -rf "$build_dir"
    mkdir -p "$build_dir"
    local cmake_cache="$build_dir/initial-cache.cmake"
    cat > "$cmake_cache" <<EOF
set(CMAKE_C_COMPILER_WORKS TRUE CACHE INTERNAL "")
set(CMAKE_CXX_COMPILER_WORKS TRUE CACHE INTERNAL "")
set(CMAKE_C_ABI_COMPILED TRUE CACHE INTERNAL "")
set(CMAKE_CXX_ABI_COMPILED TRUE CACHE INTERNAL "")
EOF
    cmake -S "$SOURCE_ROOT" -B "$build_dir" -C "$cmake_cache" \
        -G "Unix Makefiles" \
        -DCMAKE_SYSTEM_NAME=Darwin \
        -DCMAKE_CROSSCOMPILING=TRUE \
        -DCMAKE_SYSTEM_PROCESSOR="$arch" \
        -DCMAKE_OSX_SYSROOT="$IOS10_SDK_ROOT" \
        -DCMAKE_OSX_ARCHITECTURES="$arch" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN" \
        -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
        -DCMAKE_C_COMPILER="$C_COMPILER" \
        -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
        -DCMAKE_AR="$AR_TOOL" \
        -DCMAKE_RANLIB="$RANLIB_TOOL" \
        -DCMAKE_C_FLAGS="$common_flags" \
        -DCMAKE_CXX_FLAGS="$cxx_common_flags -std=gnu++17" \
        -DCMAKE_EXE_LINKER_FLAGS="$common_flags" \
        -DGGML_METAL=OFF \
        -DGGML_RPC=ON \
        -DGGML_LLAMAFILE=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_CURL=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -Wno-dev

    log "Building $arch static libraries"
    local jobs
    jobs="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
    for target in ggml ggml-base ggml-cpu ggml-blas ggml-rpc llama; do
        cmake --build "$build_dir" --target "$target" --parallel "$jobs"
    done
}

find_static_lib() {
    local build_dir="$1"
    local libname="$2"
    find "$build_dir" -name "lib${libname}.a" | head -1
}

if [[ "${RPC_ONLY_REBUILD:-0}" == "1" ]]; then
    rebuild_ggml_rpc_lib_only
else
    for arch in "${ARCHS[@]}"; do
        build_arch "$arch"
    done
fi

merge_lib() {
    local libname="$1"
    local input_libs=()
    for arch in "${ARCHS[@]}"; do
        local candidate
        candidate="$(find_static_lib "$BUILD_ROOT/$arch" "$libname")"
        [[ -n "$candidate" ]] || die "Missing lib${libname}.a for $arch"
        input_libs+=("$candidate")
    done

    log "Creating fat lib${libname}.a"
    lipo -create "${input_libs[@]}" -output "$LIB_OUTPUT_DIR/lib${libname}.a"
}

if [[ "${RPC_ONLY_REBUILD:-0}" != "1" ]]; then
    merge_lib "llama"
    merge_lib "ggml"
    merge_lib "ggml-base"
    merge_lib "ggml-cpu"
    merge_lib "ggml-blas"
    merge_lib "ggml-rpc"
fi

cat > "$OUTPUT_ROOT/manifest.txt" <<EOF
llama_dir=$LLAMA_DIR
llama_branch=$LLAMA_BRANCH
llama_commit=$(git -C "$LLAMA_DIR" rev-parse HEAD)
patched_source_root=$SOURCE_ROOT
ios_sdk_root=$IOS10_SDK_ROOT
ios_min=$IOS_MIN
archs=${ARCHS[*]}
built_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

log "Done. Legacy static libs are in: $LIB_OUTPUT_DIR"
