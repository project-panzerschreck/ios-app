#!/bin/sh
set -eu

LEGACY_TOOLCHAIN_BIN="${LEGACY_TOOLCHAIN_BIN:-$HOME/ios-legacy-toolchain/bin}"
IOS10_SDK_ROOT="${IOS10_SDK_ROOT:-$HOME/theos/sdks/iPhoneOS10.3.sdk}"
CLANGXX="${LEGACY_TOOLCHAIN_BIN}/clang++"

if [ ! -x "$CLANGXX" ]; then
  echo "legacy-link.sh: missing clang++ at $CLANGXX" >&2
  exit 1
fi

exec "$CLANGXX" "$@" \
  "$IOS10_SDK_ROOT/usr/lib/libSystem.tbd"
