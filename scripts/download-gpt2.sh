#!/usr/bin/env bash
# Shortcut: downloads GPT-2 117M Q4_K_M (~70 MB)
bash "$(dirname "${BASH_SOURCE[0]}")/download-gguf.sh" \
    "https://huggingface.co/RichardErkhov/openai-community_-_gpt2-gguf/resolve/main/gpt2.Q4_K_M.gguf" \
    50
