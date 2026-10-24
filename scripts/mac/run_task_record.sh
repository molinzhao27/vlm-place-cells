#!/usr/bin/env bash
# Experiment 14a on Apple silicon (pre-registered in docs/experiments.md): the residual stream of base
# Qwen3-VL-8B and of the LoRA model (seed 0) on experiment 10's exact 64-frame-walk prompts, all 1,920
# trials of P5000-P5009, one forward pass each (task_record.py, --device mps), then the readouts on the
# CPU (task_probe.py -> results/task_probe.json). Recording skips trials already done, so the script can
# be stopped and restarted. About 2.5 h per model; data/acts_task/ takes ~4.5 GB. From the project root,
# kept awake:
#   nohup caffeinate -i bash scripts/mac/run_task_record.sh > data/logs/mac_task_record.log 2>&1 &
set -euo pipefail
P=.venv-mac/bin/python
ADAPTER=${ADAPTER:-"data/ft/qwen8b-proc"}
export PYTORCH_ENABLE_MPS_FALLBACK=0     # fail loudly rather than run an op on the CPU
mkdir -p data/logs
echo "RUN_START=$(date -u +%FT%TZ)"
for M in base "$ADAPTER"; do
  ad=(); [[ $M == base ]] || ad=(--adapter "$M")
  echo "$(date -u +%T) task_record $M"
  $P -m spatialcells.task_record ${ad[@]+"${ad[@]}"} --device mps 2>&1 \
    | grep -vE "processor_kwargs|Redirects" || true
done
echo "$(date -u +%T) task_probe"
$P -m spatialcells.task_probe --models qwen8b "qwen8b+$(basename "$ADAPTER")" 2>&1 | grep -v Redirects || true
echo "RUN_END=$(date -u +%FT%TZ)"
