#!/usr/bin/env bash
# Single-unit place/grid/HD scoring for the larger models: walk in context, one frame,
# and one frame + 8-step moving average (the memory-smoothing control). Needs the
# single-frame activations from scripts/run_scaling.sh. Run through the GPU lock:
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-units.log bash scripts/run_scaling_units.sh
set -euo pipefail
P=.venv/bin/python
echo "RUN_START=$(date -u +%FT%TZ)"
for m in qwen8b qwen38 qwen32b; do
  $P -m spatialcells.analyze --tags "$m-walk" "$m-single" "$m-single+ema8"
  echo "UNITS_DONE $m $(date -u +%FT%TZ)"
done
echo "RUN_END=$(date -u +%FT%TZ)"
