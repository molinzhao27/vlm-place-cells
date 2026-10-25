#!/usr/bin/env bash
# Experiment 15 on Apple silicon (pre-registered in docs/experiments.md): which walk frames the LoRA model
# (seed 0) reads. Experiment 10's exact 64-frame prompts, all 1,299 novel trials of P5000-P5009, with 8 shown
# walk frames cut from attention in every decoder layer (spatialcells/frame_cut.py, --device mps). Order: the
# DINOv2 look-alike cosines on the CPU if missing (never alongside a Qwen run), clean, the registered check
# (clean reproduces experiment 10's answers on >= 99% of trials, else the script stops before any cut), then
# near neighbours, look-alikes, random, far neighbours, then near neighbours and look-alikes cut in layers
# 1-18 or 19-36 only on P5000-P5002, then the analysis (results/frame_cut_summary.json). One job at a time;
# each run skips trials already done, so the script can be stopped and restarted; any failure stops it.
# About 6.5 h + 1.5 h. From the project root, kept awake:
#   nohup caffeinate -i bash scripts/mac/run_frame_cut.sh > data/logs/mac_frame_cut.log 2>&1 &
set -euo pipefail
P=.venv-mac/bin/python
ADAPTER=${ADAPTER:-"data/ft/qwen8b-proc"}
export PYTORCH_ENABLE_MPS_FALLBACK=0     # fail loudly rather than run an op on the CPU
mkdir -p data/logs
quiet() { grep -vE "processor_kwargs|Redirects|Loading weights" || true; }
run_cut() {
  echo "$(date -u +%T) frame_cut $*"
  $P -m spatialcells.frame_cut --adapter "$ADAPTER" --device mps "$@" 2>&1 | quiet
}
echo "RUN_START=$(date -u +%FT%TZ)"
if [[ ! -f data/frame_cut/dino_lookalike.json ]]; then
  echo "$(date -u +%T) DINOv2 look-alike cosines (CPU)"
  $P -m spatialcells.frame_cut --dino 2>&1 | quiet
fi
run_cut --condition clean
echo "$(date -u +%T) check: clean vs experiment 10"
$P -m spatialcells.frame_cut --adapter "$ADAPTER" --device mps --check 2>&1 | quiet
for C in near lookalike random far; do
  run_cut --condition $C
done
for C in near lookalike; do
  for L in 1-18 19-36; do
    run_cut --condition $C --layers $L --rooms P5000 P5001 P5002
  done
done
echo "$(date -u +%T) analysis"
$P scripts/frame_cut_analysis.py
echo "RUN_END=$(date -u +%FT%TZ)"
