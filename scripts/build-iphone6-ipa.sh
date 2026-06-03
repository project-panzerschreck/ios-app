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

make clean ipa
cp -f packages/rmclusternode_1.0.0_unsigned.ipa "${1:-packages/rmclusternode-6-unsigned.ipa}"
echo "Wrote ${1:-packages/rmclusternode-6-unsigned.ipa}"
