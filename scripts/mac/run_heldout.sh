#!/usr/bin/env bash
# Experiment 10 on Apple silicon (pre-registered in docs/experiments.md): base Qwen3-VL-8B and the
# LoRA model in each held-out room, with the 64-frame walk, no walk and the shuffled 64-frame walk,
# --device mps. Rooms outer, so partial results cover rooms evenly. Each run skips trials already
# answered, so it can be stopped and restarted. Seeds 1-2 later, e.g.
#   ADAPTERS="data/ft/qwen8b-proc-s1 data/ft/qwen8b-proc-s2" SKIP_BASE=1 bash scripts/mac/run_heldout.sh
# MODEL (default Qwen/Qwen3-VL-8B-Instruct) runs another model, e.g. experiment 18's Gemma 4:
#   MODEL=google/gemma-4-12B-it ADAPTERS=data/ft/gemma12b-proc-e4-s3 SKIP_BASE=1 ROOMS="..." bash scripts/mac/run_heldout.sh
# and ADAPTERS=none runs the base model only.
# From the project root, kept awake:
#   nohup caffeinate -i bash scripts/mac/run_heldout.sh > data/logs/mac_heldout.log 2>&1 &
set -euo pipefail
P=.venv-mac/bin/python
B=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
ROOMS=${ROOMS:-"P5000 P5001 P5002 P5003 P5004 P5005 P5006 P5007 P5008 P5009"}
ADAPTERS=${ADAPTERS:-"data/ft/qwen8b-proc"}
[[ $ADAPTERS == none ]] && ADAPTERS=""            # base model only
MODELS=$([[ -n "${SKIP_BASE:-}" ]] && echo "$ADAPTERS" || echo "base $ADAPTERS")
export PYTORCH_ENABLE_MPS_FALLBACK=0     # fail loudly rather than run an op on the CPU
mkdir -p data/logs
echo "RUN_START=$(date -u +%FT%TZ)"
for R in $ROOMS; do
  for M in $MODELS; do
    ad=(); [[ $M == base ]] || ad=(--adapter "$M")
    for C in "--max-images 64" "--no-walk" "--max-images 64 --shuffle-walk"; do
      echo "$(date -u +%T) room$R $M $C"
      # shellcheck disable=SC2086
      $P -m spatialcells.two_choice --model $B ${ad[@]+"${ad[@]}"} --trials "room$R" $C --device mps 2>&1 \
        | grep -E "trials to run|: done|Error|Traceback" || true
    done
  done
done
$P -m spatialcells.two_choice --score > data/logs/tc_mps_score.log 2>&1
echo "RUN_END=$(date -u +%FT%TZ)"
