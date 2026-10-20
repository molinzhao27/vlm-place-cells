#!/usr/bin/env bash
# Place/grid/head-direction scoring of VGGT's camera-token neurons in three conditions.
# Through the GPU lock:
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-vggt-cells.log bash scripts/run_vggt_cells.sh
set -euo pipefail
P=.venv/bin/python
echo "RUN_START=$(date -u +%FT%TZ)"
for mode in alone single walk; do
  $P -m spatialcells.vggt_cells --mode $mode
done
$P -m spatialcells.analyze --tags vggt-alone vggt-single vggt-walk
echo "RUN_END=$(date -u +%FT%TZ)"
