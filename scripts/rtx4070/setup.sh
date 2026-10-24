#!/usr/bin/env bash
# One-time setup of the 4070 host: the project folder and its .venv (torch 2.14.0 from
# R4_TORCH_INDEX, as on the GPU host), VGGT cloned at the GPU host's commit, and VGGT-1B's
# weights (~5 GB) cached. Uses uv if the host has it, else R4_PYTHON's venv module.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/rtx4070/common.sh"
"${R4_SSH[@]}" bash -s -- "$R4_ROOT" "$R4_PYTHON" "$R4_TORCH_INDEX" "$R4_VGGT" "$R4_VGGT_COMMIT" <<'REMOTE'
set -euo pipefail
root="$1" py="$2" index="$3" vggt="$4" commit="$5"
renice -n 10 -p $$ >/dev/null      # the host is shared: installs and downloads at low priority
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
mkdir -p "$root"
cd "$root"
pkgs=(numpy scipy scikit-image scikit-learn matplotlib pillow huggingface_hub einops safetensors)
if command -v uv >/dev/null; then
  [[ -x .venv/bin/python ]] || uv venv --quiet --python 3.12 .venv
  uv pip install --quiet --python .venv/bin/python torch==2.14.0 torchvision==0.29.1 --index-url "$index"
  uv pip install --quiet --python .venv/bin/python "${pkgs[@]}"
else
  [[ -x .venv/bin/python ]] || "$py" -m venv .venv
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet torch==2.14.0 torchvision==0.29.1 --index-url "$index"
  .venv/bin/python -m pip install --quiet "${pkgs[@]}"
fi
[[ -d "$vggt/.git" ]] || git clone --quiet https://github.com/facebookresearch/vggt "$vggt"
git -C "$vggt" checkout --quiet "$commit"
.venv/bin/python - "$vggt" <<'PY'
import sys
import torch
print("torch", torch.__version__, "| cuda build works with this driver:", torch.cuda.is_available())  # no context made
sys.path.insert(0, sys.argv[1])
from vggt.models.vggt import VGGT
VGGT.from_pretrained("facebook/VGGT-1B")
print("VGGT-1B weights cached")
PY
REMOTE
