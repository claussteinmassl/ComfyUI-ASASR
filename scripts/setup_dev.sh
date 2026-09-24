#!/usr/bin/env bash
# Sets up the development environment: venv + pinned ComfyUI checkout.
set -euo pipefail
cd "$(dirname "$0")/.."

COMFY_COMMIT="6f7cd7fceaaf60d2669b554936394a7412c6fde5"

if [ ! -d .dev/ComfyUI ]; then
    mkdir -p .dev
    git clone https://github.com/comfyanonymous/ComfyUI.git .dev/ComfyUI
fi
git -C .dev/ComfyUI fetch --depth 1 origin "$COMFY_COMMIT" || true
git -C .dev/ComfyUI checkout "$COMFY_COMMIT" 2>/dev/null || {
    git -C .dev/ComfyUI fetch origin "$COMFY_COMMIT"
    git -C .dev/ComfyUI checkout "$COMFY_COMMIT"
}

python3.11 -m venv .venv --clear
.venv/bin/pip install -q -r requirements-dev.txt
echo "dev environment ready"
