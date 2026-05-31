# rmcluster node on iPhone 5 (iOS 10.3)

For current branch-specific build steps, read [BUILDING.md](BUILDING.md).

This repository contains a legacy UIKit / Objective-C port of the original one-screen SwiftUI app, rebuilt so it can run on an iPhone 5-class device:

- iOS `10.3`
- `armv7` / `armv7s`
- Theos application project
- unsigned IPA output for Sideloadly

This README is the full "how we got this working" guide, including the awkward parts: Theos, the legacy iOS SDK, the old Apple command-line toolchain, the separate ggml rebuild, IPA packaging, and sideloading to an iPhone 5.

## What this project is now

The app is no longer SwiftUI-based. It uses:

- `main.m`
- `AppDelegate`
- one UIKit `UIViewController`
- Objective-C UI
- Objective-C++ bridge into the ggml / llama worker code

Relevant files:

- [distributed-ml-ggml-client-ios/AppDelegate.m](distributed-ml-ggml-client-ios/AppDelegate.m)
- [distributed-ml-ggml-client-ios/RMRootViewController.m](distributed-ml-ggml-client-ios/RMRootViewController.m)
- [distributed-ml-ggml-client-ios/RMInferenceService.m](distributed-ml-ggml-client-ios/RMInferenceService.m)
- [distributed-ml-ggml-client-ios/Bridge/LlamaBridge.mm](distributed-ml-ggml-client-ios/Bridge/LlamaBridge.mm)
- [distributed-ml-ggml-client-ios/RMStorageServer.m](distributed-ml-ggml-client-ios/RMStorageServer.m)

The Theos build entrypoint is:

- [Makefile](Makefile)

The armv7-native rebuild output is intentionally separate from the modern framework build:

- [armv7-rebuild-ios10/README.md](armv7-rebuild-ios10/README.md)

## What you need

You need four things:

1. A modern macOS machine that can run Theos.
2. Theos itself.
3. A real **device** `iPhoneOS10.3.sdk`.
4. A **legacy Apple command-line toolchain** that can still link `armv7` / `armv7s`.

The important lesson from this port:

- modern Xcode alone is **not enough**
- an old Xcode GUI app is **not required**
- the winning setup is:
  - modern macOS
  - Theos
  - old iPhoneOS 10.3 device SDK
  - old Apple clang/ld toolchain used from the shell

## Why modern Xcode alone is not enough

If you try to build this with only modern Xcode and a dropped-in iOS 10 SDK, you will usually hit some combination of:

- simulator/device stub mismatches
- missing old Objective-C runtime symbols
- missing SJLJ unwind symbols
- bad `libSystem.tbd` behavior
- linker/runtime incompatibilities for 32-bit iOS

The fix is to use a legacy Apple command-line toolchain directly instead of relying on the active Xcode linker.

## Tested working shape

This project was made to build successfully with:

- Theos installed under `~/theos`
- a device SDK at `~/theos/sdks/iPhoneOS10.3.sdk`
- a legacy toolchain under `~/ios-legacy-toolchain`

Expected layout:

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

The current repo defaults match that layout. You can override them with environment variables.

## 1. Install Theos

Use the official Theos installation docs:

- Theos installation: https://theos.dev/docs/installation

Theos says it supports macOS and should be installed as a normal user, not root. It also uses a rolling release model and can be updated with `update-theos` or `make update-theos`. Source: [Theos installation docs](https://theos.dev/docs/installation).

At minimum, you want `THEOS` to point at your Theos root:

```sh
export THEOS="$HOME/theos"
```

Sanity check:

```sh
test -f "$THEOS/makefiles/common.mk" && echo "THEOS OK"
```

## 2. Install a real device iPhoneOS 10.3 SDK

You need:

- `iPhoneOS10.3.sdk`
- **device** SDK, not simulator SDK

Put it here:

```text
~/theos/sdks/iPhoneOS10.3.sdk
```

Sanity checks:

```sh
plutil -p ~/theos/sdks/iPhoneOS10.3.sdk/SDKSettings.plist | head -40
head -20 ~/theos/sdks/iPhoneOS10.3.sdk/usr/lib/libobjc.tbd
head -20 ~/theos/sdks/iPhoneOS10.3.sdk/usr/lib/libc++.tbd
head -20 ~/theos/sdks/iPhoneOS10.3.sdk/usr/lib/libgcc_s.1.tbd
```

What you want to see:

- platform is `ios`
- `armv7` and `armv7s` are present
- runtime stubs exist under `usr/lib`

## 3. Install a legacy Apple toolchain

This is the piece that makes `armv7` linking work on a modern Mac without needing to open an old Xcode.app.

You need a toolchain with binaries like:

- `clang`
- `clang++`
- `ld`
- `ar`
- `ranlib`
- `strip`
- `libtool`
- `lipo`
- `dsymutil`
- `codesign_allocate`

Put it somewhere like:

```text
~/ios-legacy-toolchain
```

Sanity checks:

```sh
~/ios-legacy-toolchain/bin/clang --version
~/ios-legacy-toolchain/bin/ld -v
~/ios-legacy-toolchain/bin/lipo -h | head
```

You do **not** need to launch an old Xcode GUI app for this project.

## 4. Verify repo-local expectations

This repo’s build expects these defaults:

```sh
export THEOS="$HOME/theos"
export IOS10_SDK_ROOT="$HOME/theos/sdks/iPhoneOS10.3.sdk"
export LEGACY_TOOLCHAIN_BIN="$HOME/ios-legacy-toolchain/bin"
export LEGACY_TOOLCHAIN_LIBCXX="$HOME/ios-legacy-toolchain/include/c++/v1"
```

Sanity check:

```sh
echo "$THEOS"
echo "$IOS10_SDK_ROOT"
echo "$LEGACY_TOOLCHAIN_BIN"
echo "$LEGACY_TOOLCHAIN_LIBCXX"
```

## 5. Understand the two native build paths

This repo has two different worlds:

1. Modern framework/Xcode build outputs under `Frameworks/`
2. Separate legacy iOS 10 armv7 rebuild outputs under `armv7-rebuild-ios10/`

The legacy build is deliberately isolated so it does not overwrite the modern output.

Read:

- [armv7-rebuild-ios10/README.md](armv7-rebuild-ios10/README.md)

## 6. Rebuild ggml / llama for iOS 10 armv7

The legacy native rebuild script is:

```sh
bash scripts/build-ggml-ios10-armv7.sh
```

What it does:

- builds into `armv7-rebuild-ios10/`
- copies the upstream `llama.cpp` source into `armv7-rebuild-ios10/src/llama.cpp-rpc-ios10/`
- applies legacy-only patches there
- keeps the main checkout untouched
- builds static libraries for:
  - `armv7`
  - `armv7s`

Expected output:

```text
armv7-rebuild-ios10/
  lib/
    libllama.a
    libggml.a
    libggml-base.a
    libggml-cpu.a
    libggml-blas.a
    libggml-rpc.a
  vendor/llama.cpp/include/
  vendor/llama.cpp/ggml/include/
  manifest.txt
```

Sanity check:

```sh
ls armv7-rebuild-ios10/lib
cat armv7-rebuild-ios10/manifest.txt
```

You can also verify architectures with the legacy `lipo`:

```sh
~/ios-legacy-toolchain/bin/lipo -info armv7-rebuild-ios10/lib/libllama.a
```

## 7. Build the app

To compile the app:

```sh
export THEOS="$HOME/theos"
export IOS10_SDK_ROOT="$HOME/theos/sdks/iPhoneOS10.3.sdk"
export LEGACY_TOOLCHAIN_BIN="$HOME/ios-legacy-toolchain/bin"
export LEGACY_TOOLCHAIN_LIBCXX="$HOME/ios-legacy-toolchain/include/c++/v1"

make messages=yes
```

Theos command docs:

- `make` compiles
- `make clean` forces a rebuild
- `make stage` stages files into `.theos/_/`
- `make package` builds a package

Source: [Theos commands](https://theos.dev/docs/commands)

Useful commands:

```sh
make clean
make messages=yes
make clean messages=yes
```

## 8. Build the IPA

This repo has a custom IPA target:

```sh
make ipa
```

That creates:

```text
packages/rmclusternode_1.0.0_unsigned.ipa
```

Why unsigned?

- Theos builds the app bundle and signs it locally with `ldid` for staging
- the final IPA is meant to be signed and installed by Sideloadly

## 9. Verify the IPA before installing

Check the archive contents:

```sh
unzip -l packages/rmclusternode_1.0.0_unsigned.ipa | sed -n '1,80p'
```

You should see at least:

```text
Payload/rmclusternode.app/rmclusternode
Payload/rmclusternode.app/Info.plist
Payload/rmclusternode.app/AppIcon...
Payload/rmclusternode.app/Default-568h@2x.png
```

Check the plist:

```sh
unzip -p packages/rmclusternode_1.0.0_unsigned.ipa Payload/rmclusternode.app/Info.plist > /tmp/rmcluster-info.plist
plutil -p /tmp/rmcluster-info.plist
```

Important values:

- bundle identifier: `com.rmcluster.rmcluster-node`
- minimum OS: `10.3`
- supported platform: `iPhoneOS`
- launch image present for iPhone 5 4-inch mode

Check architectures:

```sh
rm -rf /tmp/rmcluster-ipa-check
mkdir -p /tmp/rmcluster-ipa-check
unzip -q packages/rmclusternode_1.0.0_unsigned.ipa -d /tmp/rmcluster-ipa-check
~/ios-legacy-toolchain/bin/lipo -info /tmp/rmcluster-ipa-check/Payload/rmclusternode.app/rmclusternode
file /tmp/rmcluster-ipa-check/Payload/rmclusternode.app/rmclusternode
```

You want both:

- `armv7`
- `armv7s`

## 10. Install with Sideloadly

Official site:

- https://sideloadly.io/?lang=en

Sideloadly says:

- it installs any IPA on iPhone/iPad
- it works with free or paid Apple IDs
- free Apple IDs are valid for 7 days
- macOS download is available directly

Source: [Sideloadly homepage](https://sideloadly.io/?lang=en)

### Basic install steps

1. Connect the iPhone 5 by USB.
2. Open Sideloadly.
3. Drag `packages/rmclusternode_1.0.0_unsigned.ipa` into the window.
4. Select the iPhone 5 as the target.
5. Enter your Apple ID.
6. Leave the default install mode alone.
7. Click `Start`.

### Recommended Sideloadly settings

Use:

- `Use automatic bundle ID`: on
- no dylib/framework injection
- no custom entitlements
- no tweak injection

Leave off unless needed:

- `Try to support older iOS versions (7+)`
- `Remove limitation on supported devices`

### After install

On the phone:

1. Open `Settings`
2. Go to `General`
3. Go to `VPN & Device Management`
4. Trust the profile for your Apple ID
5. Launch the app

### Free Apple ID notes

Sideloadly says a free account works, but apps are valid for 7 days. A paid Apple Developer account extends validity to 1 year. Source: [Sideloadly FAQ on the homepage](https://sideloadly.io/?lang=en).

## 11. iPhone 5-specific notes

### Black bars top and bottom

This app includes a `Default-568h@2x.png` launch image specifically so iPhone 5-class devices do **not** run the app in old 3.5-inch compatibility mode.

If you see black bars after reinstalling, it usually means one of:

- you are still running an older IPA build
- the installed app was not replaced cleanly

Delete the app from the phone, then sideload the newest IPA again.

### Memory reporting

The worker no longer advertises raw installed RAM.

It now advertises a conservative worker budget and available headroom based on:

- `os_proc_available_memory()` when available
- otherwise a Mach-based iOS 10 fallback:
  - current footprint
  - reclaimable system pages
  - an OS safety reserve
  - a conservative per-process cap

So the iPhone 5 should no longer claim the whole 1 GB as worker-usable memory.

## 12. Common failure modes

### `Your chosen SDK ... does not appear to exist`

The path is wrong or the SDK is missing.

Check:

```sh
ls ~/theos/sdks
```

You want:

```text
iPhoneOS10.3.sdk
```

### Modern clang complains about `module.map` deprecations

This repo already disables/suppresses the problematic path enough for this build. If you hit it again, make sure you are building from the current `Makefile`.

### Linker errors with old runtime symbols

If you see failures around:

- `_objc_msgSend_stret`
- `___gxx_personality_sj0`
- simulator/device `tbd` mismatches

you are probably not using the legacy toolchain, or your SDK dump is bad.

Re-check:

```sh
echo "$LEGACY_TOOLCHAIN_BIN"
echo "$IOS10_SDK_ROOT"
~/ios-legacy-toolchain/bin/clang --version
~/ios-legacy-toolchain/bin/ld -v
```

### Sideloadly says `This does not look like valid iOS app`

This usually means the IPA bundle structure is malformed or missing expected bundle resources.

This repo’s current IPA target already includes:

- executable
- `Info.plist`
- app icons
- iPhone 5 launch image

So if you see that now, verify you are using the latest IPA from `packages/`.

### Sideloadly says the bundle ID already exists

Uncheck `Use automatic bundle ID` and give it a slightly different one, for example:

```text
com.rmcluster.rmcluster-node2
```

### App installs but will not open

Check:

1. you trusted the developer profile on-device
2. the IPA is the latest build
3. the app was signed with the Apple ID currently trusted on the phone

## 13. Daily rebuild workflow

Once everything is installed, the normal loop is:

```sh
export THEOS="$HOME/theos"
export IOS10_SDK_ROOT="$HOME/theos/sdks/iPhoneOS10.3.sdk"
export LEGACY_TOOLCHAIN_BIN="$HOME/ios-legacy-toolchain/bin"
export LEGACY_TOOLCHAIN_LIBCXX="$HOME/ios-legacy-toolchain/include/c++/v1"

make ipa
```

Then:

1. open Sideloadly
2. drag in the new IPA
3. install to the iPhone 5

If you change only Objective-C / UIKit app code, `make ipa` is usually enough.

If you change ggml / llama native code or anything in the legacy native rebuild path, do:

```sh
make clean
bash scripts/build-ggml-ios10-armv7.sh
make ipa
```

## 14. Files worth knowing

- [Makefile](Makefile)  
  Theos target, legacy toolchain wiring, IPA packaging.

- [control](control)  
  Debian/Theos package metadata.

- [scripts/build-ggml-ios10-armv7.sh](scripts/build-ggml-ios10-armv7.sh)  
  Separate iOS 10 armv7 native rebuild.

- [scripts/legacy-link.sh](scripts/legacy-link.sh)  
  Forces the legacy linker path for final app linking.

- [distributed-ml-ggml-client-ios/Info.plist](distributed-ml-ggml-client-ios/Info.plist)  
  Old-iOS app bundle metadata, icons, launch image config.

- [distributed-ml-ggml-client-ios/Default-568h@2x.png](distributed-ml-ggml-client-ios/Default-568h@2x.png)  
  Prevents iPhone 5 letterboxing.

## 15. Current output path

Successful IPA output:

```text
packages/rmclusternode_1.0.0_unsigned.ipa
```

That is the file you should drag into Sideloadly.
