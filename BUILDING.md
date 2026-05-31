# Building `ios-14.7-6s`

This branch is the SwiftUI / Xcode build for:

- `iPhone 6s`
- `arm64`
- iOS `14.7.x`

It produces an unsigned IPA suitable for sideloading.

## What you need

- macOS with Xcode installed
- Xcode command line tools
- `cmake`
- `git`
- a sibling checkout of `llama.cpp-rpc` at `../llama.cpp-rpc`

Install `cmake` if needed:

```sh
brew install cmake
```

Sanity checks:

```sh
xcode-select -p
xcrun --find clang
xcrun --find clang++
cmake --version
test -d ../llama.cpp-rpc && echo "llama.cpp-rpc OK"
```

## Repository layout expected by this branch

The native framework build script expects this layout:

```text
ios-app/
llama.cpp-rpc/
```

So from this branch root:

```sh
test -d ../llama.cpp-rpc/include
test -d ../llama.cpp-rpc/ggml/include
```

The shared arm64 build also assumes your local `../llama.cpp-rpc` checkout includes the iOS 12 compatibility fixes used by this workspace, specifically in:

- `ggml/src/ggml-backend-dl.cpp`
- `ggml/src/ggml-backend-dl.h`
- `ggml/src/ggml-backend-reg.cpp`
- `ggml/src/ggml-rpc/ggml-rpc.cpp`

Without those patches, the arm64 library build can fail on `std::filesystem` availability checks when targeting iOS 12.

## 1. Build the arm64 XCFrameworks

This branch depends on static XCFramework outputs under `Frameworks/`.

Build them with the branch-local script:

```sh
IOS_MIN=12.0 bash scripts/build-ggml-ios.sh
```

Why `12.0`:

- the shared arm64 libraries are intentionally built with an iOS 12 floor
- that lets the same arm64 framework set be reused by both the `iphone6` and `ios-14.7-6s` branches

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

## 2. Build the app

You can build from Xcode, or from the command line.

Command-line build:

```sh
xcodebuild \
  -project distributed-ml-ggml-client-ios.xcodeproj \
  -scheme distributed-ml-ggml-client-ios \
  -configuration Release \
  -sdk iphoneos \
  -derivedDataPath DerivedData \
  CODE_SIGNING_ALLOWED=NO \
  CODE_SIGNING_REQUIRED=NO \
  CODE_SIGN_IDENTITY= \
  build
```

The built app lands at:

```text
DerivedData/Build/Products/Release-iphoneos/distributed-ml-ggml-client-ios.app
```

## 3. Package an unsigned IPA

Create a plain unsigned IPA from the `.app` bundle:

```sh
APP_DIR="DerivedData/Build/Products/Release-iphoneos/distributed-ml-ggml-client-ios.app"
WORK_DIR=".ipa-payload"
OUT_DIR="packages"
IPA_NAME="rmclusternode-6s-unsigned.ipa"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/Payload" "$OUT_DIR"
cp -R "$APP_DIR" "$WORK_DIR/Payload/"
rm -rf "$WORK_DIR/Payload/distributed-ml-ggml-client-ios.app/_CodeSignature"
rm -f "$WORK_DIR/Payload/distributed-ml-ggml-client-ios.app/embedded.mobileprovision"

(
  cd "$WORK_DIR"
  zip -qry "../$OUT_DIR/$IPA_NAME" Payload
)
```

Output:

```text
packages/rmclusternode-6s-unsigned.ipa
```

## 4. Verify the artifact

Check that the app binary is arm64:

```sh
lipo -info DerivedData/Build/Products/Release-iphoneos/distributed-ml-ggml-client-ios.app/distributed-ml-ggml-client-ios
```

Check that the IPA is unsigned:

```sh
unzip -l packages/rmclusternode-6s-unsigned.ipa | rg "embedded.mobileprovision|_CodeSignature" || true
```

## Notes

- This branch is SwiftUI-based and targets iOS `14.7`.
- The shared arm64 `llama/ggml` framework set is built with `IOS_MIN=12.0` on purpose.
- If the native library build fails after switching SDK floors or source trees, remove `build-llama/` and rerun the script.
