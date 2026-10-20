#!/usr/bin/env bash
# Two-choice test on the Qwen models with the generic runner: does the 64-token frame limit
# explain their failure, and does the walk help or hurt? Through the GPU lock:
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-two-choice-qwen.log bash scripts/run_two_choice_qwen.sh
set -euo pipefail
P=.venv/bin/python
echo "RUN_START=$(date -u +%FT%TZ)"
TC="$P -m spatialcells.two_choice"
# sanity check: should reproduce behave_reason's 35% on the original novel trials
$TC --model Qwen/Qwen3-VL-8B-Instruct --trials original
for m in Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3.8-27B; do
  $TC --model $m --trials dissimilar --size 256
  for s in 256 512 1024; do $TC --model $m --trials dissimilar --size $s --no-walk; done
done
$TC --model Qwen/Qwen3-VL-8B-Instruct --trials dissimilar --size 512    # 256 tokens per frame, ~70k-token prompt
$TC --score
echo "RUN_END=$(date -u +%FT%TZ)"
