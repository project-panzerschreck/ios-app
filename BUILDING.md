# Building `iphone-5`

This branch is the legacy UIKit / Theos build for:

- `iPhone 5`
- `armv7` / `armv7s`
- iOS `10.3`

It produces an unsigned IPA suitable for sideloading.

## What you need

- macOS
- Xcode command line tools
- `cmake`
- `git`
- Theos installed under `~/theos` or exposed through `THEOS`
- a real device `iPhoneOS10.3.sdk`
- the legacy Apple command-line toolchain used by the branch

Expected default layout:

```text
~/theos/
  makefiles/
  sdks/
    iPhoneOS10.3.sdk/

~/ios-legacy-toolchain/
  bin/
    clang
    clang++
    ld
    ar
    ranlib
    strip
    libtool
    lipo
    dsymutil
    codesign_allocate
  include/
    c++/
      v1/
```

Set the environment explicitly:

```sh
export THEOS="$HOME/theos"
export IOS10_SDK_ROOT="$HOME/theos/sdks/iPhoneOS10.3.sdk"
export LEGACY_TOOLCHAIN_BIN="$HOME/ios-legacy-toolchain/bin"
export LEGACY_TOOLCHAIN_LIBCXX="$HOME/ios-legacy-toolchain/include/c++/v1"
```

Sanity checks:

```sh
test -f "$THEOS/makefiles/common.mk" && echo "THEOS OK"
test -d "$IOS10_SDK_ROOT" && echo "SDK OK"
test -x "$LEGACY_TOOLCHAIN_BIN/clang" && echo "LEGACY CLANG OK"
test -d "$LEGACY_TOOLCHAIN_LIBCXX" && echo "LIBCXX OK"
```

## 1. Build the iOS 10 armv7 native libraries

This branch does not use the modern arm64 XCFramework set at build time.
Instead it relies on the legacy rebuild output in:

```text
armv7-rebuild-ios10/
```

Libraries are built from `rmcluster/llama.cpp-rpc` on branch **`iphone-5-build`**
(sibling checkout recommended). That branch provides `ggml_backend_rpc_stop_server`
and iOS 10-safe RPC sources. Older `armv7-rebuild-ios10/` trees built from upstream
`b5076` do not include RPC stop.

```text
ios-app/
llama.cpp-rpc/    # git@github.com:rmcluster/llama.cpp-rpc.git @ iphone-5-build
```

From the repo root:

```sh
bash scripts/build-ggml-ios10-armv7.sh
```

### Verbose RPC logging (rebuild required)

RPC trace lines appear in the app **Logs** tab only when `libggml-rpc.a` is built with
`GGML_RPC_VERBOSE=1`. Setting `GGML_RPC_DEBUG=1` at runtime does **not** work on iOS
(the library reads that env var at load time).

```sh
export GGML_RPC_VERBOSE=1
# Optional: rebuild only libggml-rpc.a (faster when iterating on RPC logging):
export RPC_ONLY_REBUILD=1
bash scripts/build-ggml-ios10-armv7.sh
```

GitHub Actions (`.github/workflows/iphone-5.yml`) sets `GGML_RPC_VERBOSE=1` and
`RPC_ONLY_REBUILD=1` on the native rebuild step. That workflow expects the same legacy
toolchain and iOS 10.3 SDK on the runner (stock hosted runners do not include them).

Expected output:

```text
armv7-rebuild-ios10/lib/
armv7-rebuild-ios10/vendor/
armv7-rebuild-ios10/manifest.txt
```

Sanity check:

```sh
ls -1 armv7-rebuild-ios10/lib
test -f armv7-rebuild-ios10/manifest.txt && echo "MANIFEST OK"
```

If you already have a good `armv7-rebuild-ios10/manifest.txt`, `make ipa` will reuse that output and skip the rebuild step.

## 2. Build the unsigned IPA with Theos

Build:

```sh
make clean ipa
```

Or use the release-style name:

```sh
bash scripts/build-iphone5-ipa.sh packages/rmclusternode-5-unsigned.ipa
```

Output:

```text
packages/rmclusternode_1.0.0_unsigned.ipa
packages/rmclusternode-5-unsigned.ipa   # copy from build-iphone5-ipa.sh
```

## 3. Verify the artifact

Check the staged binary architecture:

```sh
lipo -info .theos/obj/rmclusternode.app/rmclusternode
```

Expected result:

- `armv7`
- `armv7s`

Check that the IPA does not contain signing payloads:

```sh
unzip -l packages/rmclusternode-5-unsigned.ipa | rg "embedded.mobileprovision|_CodeSignature" || true
```

## Notes

- Modern Xcode alone is not enough for this branch. The legacy Apple command-line toolchain is required for reliable 32-bit device linking.
- The branch-local `scripts/build-ggml-ios10-armv7.sh` contains a small compatibility fix so `LEGACY_TOOLCHAIN_BIN` can be provided explicitly without tripping over an unset `PREFIX`.
- For a deeper historical explanation of the 32-bit porting setup, also read [README.md](README.md).
