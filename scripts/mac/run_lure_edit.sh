#!/usr/bin/env bash
# Experiment 14c on Apple silicon (pre-registered in docs/experiments.md): base Qwen3-VL-8B, 256 px, with
# the full 256-frame walk and with no walk, on room A's 38 qualifying lure trials, unedited (roomAlure,
# from data/walks), edited and control (the roomAno<posters> rooms of spatialcells/lure_edit.py, which
# must have been run first). --device mps; no-walk runs first (short), then the walk. Each run skips
# trials already answered, so it can be stopped and restarted. Check a presentation first with
# two_choice.py --dry-run N (no model). From the project root, kept awake:
#   nohup caffeinate -i bash scripts/mac/run_lure_edit.sh > data/logs/mac_lure_edit.log 2>&1 &
# then .venv-mac/bin/python scripts/lure_edit_analysis.py (run at the end too).
set -euo pipefail
P=.venv-mac/bin/python
B=Qwen/Qwen3-VL-8B-Instruct
SETS=${SETS:-"roomAlure $(ls -d results/export/dissimilar-roomAno* | sed 's|.*/dissimilar-||' | tr '\n' ' ')"}
export PYTORCH_ENABLE_MPS_FALLBACK=0     # fail loudly rather than run an op on the CPU
mkdir -p data/logs
echo "RUN_START=$(date -u +%FT%TZ)"
for C in "--no-walk" ""; do
  for S in $SETS; do
    echo "$(date -u +%T) $S base ${C:-walk}"
    # shellcheck disable=SC2086
    $P -m spatialcells.two_choice --model $B --trials "$S" --size 256 $C --device mps 2>&1 \
      | grep -E "trials to run|^1/|: done|Error|Traceback" || true
  done
done
$P -m spatialcells.two_choice --score > data/logs/tc_mps_score.log 2>&1
$P scripts/lure_edit_analysis.py || true
echo "RUN_END=$(date -u +%FT%TZ)"
