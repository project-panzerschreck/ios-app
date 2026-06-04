#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export THEOS="${THEOS:-$HOME/theos}"
export IOS16_SDK_ROOT="${IOS16_SDK_ROOT:-$(xcrun --sdk iphoneos --show-sdk-path)}"

if [[ ! -d Frameworks ]]; then
  echo "Frameworks/ missing. Build with: IOS_MIN=12.0 bash scripts/build-ggml-ios.sh" >&2
  exit 1
fi

VERBOSE_RPC="${VERBOSE_RPC:-0}"
if [[ "$VERBOSE_RPC" == "1" ]]; then
  echo "[build-iphone6-ipa] Building with VERBOSE_RPC_DEFAULT (GGML/RPC logs enabled by default in app)"
fi

VERBOSE_RPC="$VERBOSE_RPC" make clean ipa
cp -f packages/rmclusternode_1.0.0_unsigned.ipa "${1:-packages/rmclusternode-6-unsigned.ipa}"
echo "Wrote ${1:-packages/rmclusternode-6-unsigned.ipa}"
