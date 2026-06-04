#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DERIVED_DATA="${DERIVED_DATA:-DerivedData}"
APP_DIR="$DERIVED_DATA/Build/Products/Release-iphoneos/distributed-ml-ggml-client-ios.app"
WORK_DIR="${WORK_DIR:-.ipa-payload}"
OUT_DIR="${OUT_DIR:-packages}"
IPA_BASENAME="$(basename "${1:-rmclusternode-6s-unsigned.ipa}")"

if [[ ! -d Frameworks/llama.xcframework ]]; then
  echo "Frameworks/ missing. Build with: IOS_MIN=12.0 bash scripts/build-ggml-ios.sh" >&2
  exit 1
fi

VERBOSE_RPC="${VERBOSE_RPC:-0}"
XCODE_EXTRA_ARGS=()
if [[ "$VERBOSE_RPC" == "1" ]]; then
  echo "[build-ios-14-ipa] Building with VERBOSE_RPC_DEFAULT (GGML/RPC logs enabled by default in app)"
  XCODE_EXTRA_ARGS+=(
    'GCC_PREPROCESSOR_DEFINITIONS=VERBOSE_RPC_DEFAULT=1 $(inherited)'
    'SWIFT_ACTIVE_COMPILATION_CONDITIONS=VERBOSE_RPC_DEFAULT $(inherited)'
  )
fi

xcodebuild \
  -project distributed-ml-ggml-client-ios.xcodeproj \
  -scheme distributed-ml-ggml-client-ios \
  -configuration Release \
  -sdk iphoneos \
  -derivedDataPath "$DERIVED_DATA" \
  CODE_SIGNING_ALLOWED=NO \
  CODE_SIGNING_REQUIRED=NO \
  CODE_SIGN_IDENTITY= \
  "${XCODE_EXTRA_ARGS[@]}" \
  build

if [[ ! -d "$APP_DIR" ]]; then
  echo "App bundle not found at $APP_DIR" >&2
  exit 1
fi

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/Payload" "$OUT_DIR"
cp -R "$APP_DIR" "$WORK_DIR/Payload/"
rm -rf "$WORK_DIR/Payload/distributed-ml-ggml-client-ios.app/_CodeSignature"
rm -f "$WORK_DIR/Payload/distributed-ml-ggml-client-ios.app/embedded.mobileprovision"

(
  cd "$WORK_DIR"
  zip -qry "../$OUT_DIR/$IPA_BASENAME" Payload
)

lipo -info "$APP_DIR/distributed-ml-ggml-client-ios"
echo "Wrote $OUT_DIR/$IPA_BASENAME"
