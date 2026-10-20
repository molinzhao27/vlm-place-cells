#!/usr/bin/env bash
# Two-choice test on newer open-weight models (generic runner). Through the GPU lock:
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-two-choice-new.log bash scripts/run_two_choice_new.sh
set -euo pipefail
P=.venv/bin/python
echo "RUN_START=$(date -u +%FT%TZ)"
TC="$P -m spatialcells.two_choice"
for m in Qwen/Qwen3.6-35B-A3B google/gemma-4-31B-it; do
  $TC --model $m --trials dissimilar
  $TC --model $m --trials dissimilar --no-walk
  $TC --model $m --trials original
done
$TC --score
echo "RUN_END=$(date -u +%FT%TZ)"
