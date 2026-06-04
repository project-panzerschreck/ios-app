#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export THEOS="${THEOS:-$HOME/theos}"
export IOS10_SDK_ROOT="${IOS10_SDK_ROOT:-$HOME/theos/sdks/iPhoneOS10.3.sdk}"
export LEGACY_TOOLCHAIN_BIN="${LEGACY_TOOLCHAIN_BIN:-$HOME/ios-legacy-toolchain/bin}"
export LEGACY_TOOLCHAIN_LIBCXX="${LEGACY_TOOLCHAIN_LIBCXX:-$HOME/ios-legacy-toolchain/include/c++/v1}"

if [[ ! -f armv7-rebuild-ios10/manifest.txt ]]; then
  echo "armv7-rebuild-ios10/manifest.txt missing. Build native libs with:" >&2
  echo "  GGML_RPC_VERBOSE=1 bash scripts/build-ggml-ios10-armv7.sh" >&2
  exit 1
fi

make clean ipa
cp -f packages/rmclusternode_1.0.0_unsigned.ipa "${1:-packages/rmclusternode-5-unsigned.ipa}"
echo "Wrote ${1:-packages/rmclusternode-5-unsigned.ipa}"
