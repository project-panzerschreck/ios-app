# Distributed Inference – Build & Run Guide

Phones are **leaf/worker nodes only**. The Mac coordinator handles tokenization and sampling (CPU); all tensor compute runs on the phone(s) via GGML RPC.

---

## 1. iOS app (RPC server on phone)

Build and run the Xcode project normally. The app starts a GGML RPC server on port **50052** (configurable in the UI). The phone's Wi-Fi IP and the llama-cli command are displayed in the RPC Worker panel.

Prerequisites: run the xcframework build script first so `Frameworks/` is populated.

```bash
cd scripts
./build-ggml-ios.sh
```

Then build & run the app from Xcode onto a physical device.

---

## 2. Mac coordinator binary (no Metal, RPC only)

Build a dedicated `llama-cli` that has Metal **disabled**. Without Metal, the only GPU backend is RPC, so all layers go to the phone(s).

### Configure (one-time)

> **Apple Silicon note:** CMake's SVE probe hangs indefinitely on M-series chips. Use the provided initial-cache file (`rpc-only-init.cmake`) which pre-seeds the result and skips the hang.

```bash
cmake -C rpc-only-init.cmake \
  -B build-mac-rpc-only \
  -DCMAKE_BUILD_TYPE=Release \
  vendor/llama.cpp
```

`rpc-only-init.cmake` sets `GGML_METAL=OFF`, `GGML_RPC=ON`, and pre-caches the ARM SVE feature-detection result.

### Build

```bash
cmake --build build-mac-rpc-only --target llama-cli -j$(sysctl -n hw.logicalcpu)
```

Binary lands at `build-mac-rpc-only/bin/llama-cli`.

> **Note:** `build-mac/` is the original Metal-enabled build (kept for single-device testing).
> `build-mac-rpc-only/` is for distributed runs where phones are the only workers.

---

## 3. Run distributed inference

Start the iOS app on the phone first, then on the Mac:

```bash
./build-mac-rpc-only/bin/llama-cli \
  --rpc <phone-ip>:50052 \
  -m ./Models/<model>.gguf \
  -ngl 99 \
  -p "Your prompt here" \
  -no-cnv
```

**Multiple phones:**

```bash
./build-mac-rpc-only/bin/llama-cli \
  --rpc <phone1-ip>:50052,<phone2-ip>:50052 \
  -m ./Models/<model>.gguf \
  -ngl 99 \
  -p "Your prompt here" \
  -no-cnv
```

### Key flags

| Flag              | Purpose                                               |
| ----------------- | ----------------------------------------------------- |
| `--rpc <ip:port>` | RPC worker address(es), comma-separated               |
| `-ngl 99`         | Offload all layers to GPU backends (phones via RPC)   |
| `-no-cnv`         | Disable chat-template wrapping (use for bare prompts) |
| `-sys "..."`      | Set system prompt when using chat mode                |

### What to expect in output

```
load_tensors: RPC[<ip>:50052] model buffer size = ~98 MiB
load_tensors:        CPU model buffer size = ~28 MiB   ← coordinator CPU only
```

Metal lines should be absent. `graph splits` count equals number of active backends.

---

## 4. Device registry server (optional)

Tracks which phones are online and generates the llama-cli command automatically.

```bash
pip install -r server/requirements.txt
python server/inference_server.py \
  --model /path/to/model.gguf \
  --llama-cli ./build-mac-rpc-only/bin/llama-cli
```

Endpoints: `POST /register`, `GET /devices`, `POST /keepalive/{id}`, `DELETE /deregister/{id}`, `POST /run-inference`.

---

## 5. llama.cpp version

Pinned to tag **b5076** in `scripts/build-ggml-ios.sh`. `llama_batch_add` was removed at this tag; an inline helper `llama_batch_add_token` is defined in `Bridge/LlamaBridge.mm`.
