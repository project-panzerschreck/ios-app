# Building `iphone6`

This branch is the legacy UIKit / Theos build for:

- `iPhone 6`
- `arm64`
- iOS `12.x`

It produces an unsigned IPA suitable for sideloading.

## What you need

- macOS
- Xcode and command line tools
- `cmake`
- `git`
- Theos installed under `~/theos` or exposed through `THEOS`
- a sibling checkout of `llama.cpp-rpc` at `../llama.cpp-rpc`

Install `cmake` if needed:

```sh
brew install cmake
```

Sanity checks:

```sh
test -f "$HOME/theos/makefiles/common.mk" && echo "THEOS OK"
xcode-select -p
cmake --version
test -d ../llama.cpp-rpc && echo "llama.cpp-rpc OK"
```

## Repository layout expected by this branch

The arm64 native framework build expects:

```text
ios-app/
llama.cpp-rpc/
```

The app build consumes static XCFrameworks from `Frameworks/`.

The shared arm64 framework build assumes your local `../llama.cpp-rpc` checkout is on the branch that carries the iOS 12 compatibility work used for the arm64 builds in this repo:

- `iphone6-build`

Without that branch, the arm64 library build can fail on `std::filesystem` availability checks when targeting iOS 12.

## 1. Build the shared arm64 XCFrameworks

This branch shares the same arm64 framework set as `ios-14.7-6s`.

Build them locally with:

```sh
IOS_MIN=12.0 bash scripts/build-ggml-ios.sh
```

Expected outputs:

```text
Frameworks/
  llama.xcframework
  ggml.xcframework
  ggml-base.xcframework
  ggml-cpu.xcframework
  ggml-blas.xcframework
  ggml-rpc.xcframework
  ggml-metal.xcframework
```

Sanity check:

```sh
ls -1 Frameworks
```

## 2. Build the unsigned IPA with Theos

The branch Makefile is already configured for:

- `TARGET := iphone:clang:12.0:12.0`
- `ARCHS := arm64`
- packaging an unsigned IPA

Build:

```sh
export IOS16_SDK_ROOT="$(xcrun --sdk iphoneos --show-sdk-path)"
make clean ipa
```

Or use the helper script:

```sh
bash scripts/build-iphone6-ipa.sh packages/rmclusternode-6-unsigned.ipa
```

Output:

```text
packages/rmclusternode_1.0.0_unsigned.ipa
```

For convenience, you can copy it to the release-style name:

```sh
cp packages/rmclusternode_1.0.0_unsigned.ipa packages/rmclusternode-6-unsigned.ipa
```

## 3. Verify the artifact

Check the binary architecture:

```sh
lipo -info .theos/obj/arm64/rmclusternode.app/rmclusternode
```

Check that the IPA does not contain signing payloads:

```sh
unzip -l packages/rmclusternode-6-unsigned.ipa | rg "embedded.mobileprovision|_CodeSignature" || true
```

## Notes

- This branch does not use SwiftUI. The UI is the UIKit / Objective-C port.
- The app intentionally uses `GGMLMetalStub.mm` instead of linking the real `ggml-metal` static library into the Theos target. That avoids iPhone 6 runtime issues while still satisfying GGML’s backend registry symbol requirements.
- If you rebuilt the XCFrameworks but Theos does not relink, run `make clean ipa` instead of `make ipa`.
