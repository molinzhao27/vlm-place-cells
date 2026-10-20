#!/usr/bin/env bash
# Behavioural nearest-frame test across model sizes, inside the container (through the GPU lock):
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-behave.log bash scripts/run_behave.sh [models...]
set -euo pipefail
P=.venv/bin/python
ALL="qwen qwen8b qwen38 qwen32b"
MODELS="${*:-$ALL}"
echo "RUN_START=$(date -u +%FT%TZ)"
for m in $MODELS; do
  $P -m spatialcells.behave --model "$m"
  $P -m spatialcells.analyze_behave --models $ALL   # table over whatever has finished
done
echo "RUN_END=$(date -u +%FT%TZ)"
