#!/usr/bin/env bash
#
# download-smollm2.sh
#
# Downloads SmolLM2-135M-Instruct in GGUF format (Q4_K_M, ~85 MB).
# SmolLM2 is an instruction-tuned model from HuggingFace that fits
# comfortably on retired smartphones and produces much more coherent
# output than GPT-2.
#
# Prompt format (chat template):
#   <|im_start|>system\nYou are a helpful AI assistant.<|im_end|>
#   <|im_start|>user\n{prompt}<|im_end|>
#   <|im_start|>assistant\n
#
# Usage:
#   bash scripts/download-smollm2.sh
#   HF_TOKEN=hf_xxx bash scripts/download-smollm2.sh   # if rate-limited

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/Models"

log() { echo "[download-smollm2] $*"; }

mkdir -p "$MODELS_DIR"

MODEL_URL="https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"
MODEL_FILE="$MODELS_DIR/SmolLM2-135M-Instruct-Q4_K_M.gguf"
MIN_SIZE=$((60 * 1024 * 1024))   # 60 MB lower bound (real file ~85 MB)

if [[ -f "$MODEL_FILE" ]]; then
    actual=$(wc -c < "$MODEL_FILE")
    if [[ "$actual" -ge "$MIN_SIZE" ]]; then
        log "Model already exists at $MODEL_FILE  ($(( actual / 1024 / 1024 )) MB)"
        exit 0
    fi
    log "Found incomplete file ($(( actual )) bytes) – re-downloading…"
    rm -f "$MODEL_FILE"
fi

# Load .env from project root if present (never committed — see .gitignore)
ENV_FILE="$PROJECT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    set -a; source "$ENV_FILE"; set +a
fi

AUTH_HEADER=""
if [[ -n "${HF_TOKEN:-}" ]]; then
    AUTH_HEADER="Authorization: Bearer $HF_TOKEN"
    log "Using HF_TOKEN for authentication."
fi

log "Downloading SmolLM2-135M-Instruct Q4_K_M (~85 MB) …"

if command -v curl >/dev/null 2>&1; then
    curl -L --progress-bar \
        ${AUTH_HEADER:+-H "$AUTH_HEADER"} \
        -o "$MODEL_FILE" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
    wget -q --show-progress \
        ${AUTH_HEADER:+--header "$AUTH_HEADER"} \
        -O "$MODEL_FILE" "$MODEL_URL"
else
    echo "ERROR: curl or wget required" >&2
    exit 1
fi

actual=$(wc -c < "$MODEL_FILE" 2>/dev/null || echo 0)
if [[ "$actual" -lt "$MIN_SIZE" ]]; then
    content=$(head -c 200 "$MODEL_FILE" 2>/dev/null)
    rm -f "$MODEL_FILE"
    echo "ERROR: Download failed (got $actual bytes). Server said:" >&2
    echo "  $content" >&2
    echo "" >&2
    echo "Try with your HuggingFace token:" >&2
    echo "  HF_TOKEN=<your_token> bash scripts/download-smollm2.sh" >&2
    exit 1
fi

log "✓ Model saved to: $MODEL_FILE  ($(( actual / 1024 / 1024 )) MB)"
log ""
log "Simulator: run  bash scripts/push-model-to-sim.sh  after building the app."
log "Device:    AirDrop or Finder → Devices → Files to copy into the app sandbox."
log ""
log "Prompt format for best results:"
log "  <|im_start|>system"
log "  You are a helpful AI assistant.<|im_end|>"
log "  <|im_start|>user"
log "  {your question}<|im_end|>"
log "  <|im_start|>assistant"
