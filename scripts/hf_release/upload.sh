#!/bin/bash
# Upload the staged release (scripts/hf_release/stage.py) to the Hugging Face Hub through the local
# throttle, so the Mac's uplink keeps room for other sessions' API calls. Re-running resumes.
#   bash scripts/hf_release/upload.sh [model|dataset|all]   (KBPS=3000 by default)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="/opt/homebrew/bin:$PATH"
what=${1:-all}
KBPS=${KBPS:-3000}
PORT=${PORT:-8899}
python3 scripts/hf_release/throttle_proxy.py --port "$PORT" --kbps "$KBPS" &
proxy=$!
trap 'kill $proxy 2>/dev/null' EXIT
sleep 1
export HTTPS_PROXY=http://127.0.0.1:$PORT https_proxy=http://127.0.0.1:$PORT
if [[ $what == model || $what == all ]]; then
    hf upload molinzhao/vlm-place-cells-lora data/hf_release/model . --repo-type model \
        --exclude ".cache/*" --commit-message "LoRA adapters" --format human
fi
if [[ $what == dataset || $what == all ]]; then
    hf upload molinzhao/vlm-place-cells data/hf_release/dataset . --repo-type dataset \
        --exclude ".cache/*" --commit-message "Data and per-trial results" --format human
fi
echo "upload done ($what)"
