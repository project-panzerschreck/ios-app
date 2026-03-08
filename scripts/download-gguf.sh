#!/usr/bin/env bash
#
# download-gguf.sh
#
# Downloads any GGUF model from a URL into the project's Models/ directory.
# The filename is taken from the last path segment of the URL.
#
# Usage:
#   bash scripts/download-gguf.sh <URL> [MIN_SIZE_MB]
#
# Examples:
#   bash scripts/download-gguf.sh \
#     https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
#
#   HF_TOKEN=hf_xxx bash scripts/download-gguf.sh \
#     https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
#     600

set -euo pipefail
# shellcheck source=lib/gguf-download.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/gguf-download.sh"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 <URL> [MIN_SIZE_MB]" >&2
    echo "Example:" >&2
    echo "  bash $0 https://huggingface.co/.../model-Q4_K_M.gguf 600" >&2
    exit 1
fi

URL="$1"
MIN_MB="${2:-50}"
FILENAME="${URL##*/}"   # last path segment of the URL

gguf_setup_dirs
gguf_load_token "download-gguf"
download_gguf "$URL" "$MODELS_DIR/$FILENAME" "$MIN_MB" "download-gguf"
