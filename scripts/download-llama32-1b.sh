#!/usr/bin/env bash
# Shortcut: downloads Llama 3.2 1B Instruct Q4_K_M (~700 MB, gated — needs HF_TOKEN)
# Accept license first: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
bash "$(dirname "${BASH_SOURCE[0]}")/download-gguf.sh" \
    "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
    600
