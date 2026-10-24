#!/usr/bin/env bash
# Experiment 13 on Apple silicon (pre-registered in docs/experiments.md). 13a: the LoRA model (seed 0)
# in each held-out room with the 64-frame walk reversed, window-shuffled in blocks of 8 and
# renumbered. 13b: base Qwen3-VL-8B on room A's lure-free trials with the 64-frame walk's frames,
# all but the two options, taken from room B's walk with the same index. --device mps; rooms outer.
# Each run skips trials already answered, so it can be stopped and restarted. Check a presentation
# first with --dry-run N (no model). From the project root, kept awake:
#   nohup caffeinate -i bash scripts/mac/run_exp13.sh > data/logs/mac_exp13.log 2>&1 &
set -euo pipefail
P=.venv-mac/bin/python
B=Qwen/Qwen3-VL-8B-Instruct
ROOMS=${ROOMS:-"P5000 P5001 P5002 P5003 P5004 P5005 P5006 P5007 P5008 P5009"}
ADAPTER=${ADAPTER:-"data/ft/qwen8b-proc"}
export PYTORCH_ENABLE_MPS_FALLBACK=0     # fail loudly rather than run an op on the CPU
mkdir -p data/logs
echo "RUN_START=$(date -u +%FT%TZ)"
# 13b first (short): the irrelevant walk, plus its no-walk and 64-frame references rerun on MPS so all
# three are on one device (the references were run on CUDA).
for C in "--irrelevant-walk roomB" "--no-walk" ""; do
  echo "$(date -u +%T) roomA base ${C:-walk}"
  # shellcheck disable=SC2086
  $P -m spatialcells.two_choice --model $B --trials dissimilar --size 256 --max-images 64 $C --device mps 2>&1 \
    | grep -E "trials to run|: done|Error|Traceback" || true
done
for R in $ROOMS; do
  for C in "--reverse-walk" "--window-shuffle 8" "--renumber"; do
    echo "$(date -u +%T) room$R $ADAPTER $C"
    # shellcheck disable=SC2086
    $P -m spatialcells.two_choice --model $B --adapter "$ADAPTER" --trials "room$R" --max-images 64 $C --device mps 2>&1 \
      | grep -E "trials to run|: done|Error|Traceback" || true
  done
done
$P -m spatialcells.two_choice --score > data/logs/tc_mps_score.log 2>&1
echo "RUN_END=$(date -u +%FT%TZ)"
