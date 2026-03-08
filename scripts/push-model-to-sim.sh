#!/usr/bin/env bash
#
# push-model-to-sim.sh
#
# Copies all .gguf files from Models/ into the booted simulator's
# Documents directory for the app.  Run this after building and
# launching the app at least once so the container exists.
#
# Usage:
#   bash scripts/push-model-to-sim.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUNDLE_ID="sreehal.distributed-ml-ggml-client-ios"

log() { echo "[push-model-to-sim] $*"; }

CONTAINER=$(xcrun simctl get_app_container booted "$BUNDLE_ID" data 2>/dev/null) \
    || { echo "ERROR: App '$BUNDLE_ID' not found on booted simulator."; \
         echo "       Build and run the app in Xcode first, then re-run this script."; exit 1; }

DEST="$CONTAINER/Documents"
mkdir -p "$DEST"

count=0
for f in "$PROJECT_DIR"/Models/*.gguf; do
    [[ -f "$f" ]] || continue
    log "Copying $(basename "$f") …"
    cp "$f" "$DEST/"
    (( count++ )) || true
done

if [[ "$count" -eq 0 ]]; then
    echo "No .gguf files found in Models/ — run scripts/download-gpt2.sh first."
    exit 1
fi

log "✓ $count model(s) pushed to simulator Documents."
log "  Restart or re-open the app to see them in the model list."
