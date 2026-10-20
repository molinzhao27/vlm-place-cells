#!/usr/bin/env bash
# Teleport-probe test across model sizes, inside the container (run through
# scripts/vast1/gpu-run.sh so the shared GPU lock is held throughout):
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-scaling.log bash scripts/run_scaling.sh
set -euo pipefail
P=.venv/bin/python
MODELS="qwen qwen8b qwen38 qwen32b"
echo "RUN_START=$(date -u +%FT%TZ)"
for m in $MODELS; do
  for mode in walk shuffled probes-only; do
    [[ $mode == probes-only && -d data/acts/$m-probes-only ]] && continue  # already done by the smoke test
    $P -m spatialcells.extract --model "$m" --mode "$mode"
  done
done
$P -m spatialcells.analyze_probes --models $MODELS
echo "PROBES_DONE=$(date -u +%FT%TZ)"
# single frames for the larger models (memory-smoothing control on single units)
for m in qwen8b qwen38 qwen32b; do
  $P -m spatialcells.extract --model "$m" --mode single
done
echo "RUN_END=$(date -u +%FT%TZ)"
