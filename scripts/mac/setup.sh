#!/usr/bin/env bash
# Create .venv-mac for running the Qwen behaviour tests (two_choice.py --device mps) on Apple silicon,
# pinned to the GPU host's versions so the only difference from a CUDA run is the device.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[[ -x .venv-mac/bin/python ]] || uv venv .venv-mac --python 3.12
uv pip install --python .venv-mac/bin/python torch==2.14.0 torchvision==0.29.1 transformers==5.20.0 peft==0.21.2 \
  accelerate==1.15.0 qwen-vl-utils numpy scipy scikit-image scikit-learn matplotlib pillow huggingface_hub
.venv-mac/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available(), '| transformers', transformers.__version__)"
