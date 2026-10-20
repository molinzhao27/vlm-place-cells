#!/usr/bin/env bash
# Two-choice test on Hy-Embodied-VLM-1.0 (Tencent, Hunyuan MoE 30B-A3B; at most 128 images, so the
# walk is an evenly spaced 125 of its 256 frames plus both options). Needs transformers 4.57 from
# data/tf457 (pip install --target) and the model's remote code. Qwen3.6 gets the same 128-image
# walk for comparison. Through the GPU lock:
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-two-choice-hy.log bash scripts/run_two_choice_hy.sh
set -euo pipefail
P=.venv/bin/python
echo "RUN_START=$(date -u +%FT%TZ)"
HY="env PYTHONPATH=data/tf457 $P -m spatialcells.two_choice --model tencent/Hy-Embodied-VLM-1.0 --trust-remote-code"
$HY --trials dissimilar --no-walk
$HY --trials dissimilar --max-images 128
$HY --trials original --max-images 128
$P -m spatialcells.two_choice --model Qwen/Qwen3.6-35B-A3B --trials dissimilar --max-images 128
$P -m spatialcells.two_choice --score
echo "RUN_END=$(date -u +%FT%TZ)"
