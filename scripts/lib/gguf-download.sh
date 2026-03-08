#!/usr/bin/env bash
#
# lib/gguf-download.sh
#
# Shared helpers for GGUF model download scripts.
# Source this file; do not execute it directly.
#
# Provides:
#   gguf_setup_dirs   – sets SCRIPT_DIR / PROJECT_DIR / MODELS_DIR
#   gguf_load_token   – loads .env and sets AUTH_HEADER from HF_TOKEN
#   download_gguf URL DEST MIN_SIZE_MB [LOG_PREFIX]
#     Downloads URL → DEST, verifying the file is at least MIN_SIZE_MB.
#     Exits non-zero on failure with a clear error message.

# Guard against double-sourcing
[[ -n "${_GGUF_DOWNLOAD_LIB:-}" ]] && return 0
_GGUF_DOWNLOAD_LIB=1

# ── Directory helpers ─────────────────────────────────────────────────────────

gguf_setup_dirs() {
    # Call from the top of each model script.
    # Sets SCRIPT_DIR, PROJECT_DIR, MODELS_DIR relative to the caller.
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
    MODELS_DIR="$PROJECT_DIR/Models"
    mkdir -p "$MODELS_DIR"
}

# ── Logging ───────────────────────────────────────────────────────────────────

# Call  _gguf_log PREFIX message …
_gguf_log() {
    local prefix="$1"; shift
    echo "[$prefix] $*"
}

# ── HuggingFace auth ──────────────────────────────────────────────────────────

# Sets AUTH_HEADER (may be empty if no token found anywhere).
# Call after gguf_setup_dirs so PROJECT_DIR is available.
#
# Token lookup order:
#   1. HF_TOKEN env var
#   2. Project .env file  ($PROJECT_DIR/.env)
#   3. ~/.cache/huggingface/token  (written by huggingface-cli login)
#   4. ~/.huggingface/token        (legacy location)
gguf_load_token() {
    local prefix="${1:-gguf-download}"
    AUTH_HEADER=""

    # 1. Load project .env so it can set HF_TOKEN
    local env_file="$PROJECT_DIR/.env"
    if [[ -f "$env_file" ]]; then
        # shellcheck disable=SC1090
        set -a; source "$env_file"; set +a
    fi

    # 2. Fall back to cached token files if HF_TOKEN still unset
    if [[ -z "${HF_TOKEN:-}" ]]; then
        local cache_token="$HOME/.cache/huggingface/token"
        local legacy_token="$HOME/.huggingface/token"
        if [[ -f "$cache_token" ]]; then
            HF_TOKEN="$(< "$cache_token")"
            _gguf_log "$prefix" "Using token from ~/.cache/huggingface/token"
        elif [[ -f "$legacy_token" ]]; then
            HF_TOKEN="$(< "$legacy_token")"
            _gguf_log "$prefix" "Using token from ~/.huggingface/token"
        fi
    fi

    if [[ -n "${HF_TOKEN:-}" ]]; then
        AUTH_HEADER="Authorization: Bearer $HF_TOKEN"
    else
        _gguf_log "$prefix" "No HuggingFace token found — attempting unauthenticated download."
        _gguf_log "$prefix" "If this fails with 401, either:"
        _gguf_log "$prefix" "  pip install huggingface_hub && huggingface-cli login"
        _gguf_log "$prefix" "  HF_TOKEN=hf_xxx bash $0"
    fi
}

# ── Core download ─────────────────────────────────────────────────────────────

# download_gguf URL DEST MIN_SIZE_MB [LOG_PREFIX]
download_gguf() {
    local url="$1"
    local dest="$2"
    local min_mb="$3"
    local prefix="${4:-gguf-download}"
    local min_bytes=$(( min_mb * 1024 * 1024 ))

    # Skip if already complete
    if [[ -f "$dest" ]]; then
        local actual
        actual=$(wc -c < "$dest")
        if [[ "$actual" -ge "$min_bytes" ]]; then
            _gguf_log "$prefix" "Already downloaded: $dest  ($(( actual / 1024 / 1024 )) MB)"
            return 0
        fi
        _gguf_log "$prefix" "Incomplete file ($(( actual )) bytes) — re-downloading…"
        rm -f "$dest"
    fi

    _gguf_log "$prefix" "Downloading → $dest"

    if command -v curl >/dev/null 2>&1; then
        curl -L --progress-bar \
            ${AUTH_HEADER:+-H "$AUTH_HEADER"} \
            -o "$dest" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress \
            ${AUTH_HEADER:+--header "$AUTH_HEADER"} \
            -O "$dest" "$url"
    else
        echo "ERROR: curl or wget is required" >&2
        return 1
    fi

    # Verify size — catches HTML auth-error pages saved as the output file
    local actual
    actual=$(wc -c < "$dest" 2>/dev/null || echo 0)
    if [[ "$actual" -lt "$min_bytes" ]]; then
        local snippet
        snippet=$(head -c 200 "$dest" 2>/dev/null || true)
        rm -f "$dest"
        echo "ERROR: Download failed (got $actual bytes). Server response:" >&2
        echo "  $snippet" >&2
        echo "" >&2
        echo "If this is a gated model, re-run with your HuggingFace token:" >&2
        echo "  HF_TOKEN=hf_xxx bash $0" >&2
        return 1
    fi

    _gguf_log "$prefix" "✓ Saved to: $dest  ($(( actual / 1024 / 1024 )) MB)"
}
