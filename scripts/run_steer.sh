#!/usr/bin/env bash
# Experiment 8: steer VGGT's place neurons. A small smoke run on block 19, then block 21's
# activations (vggt_cells --blocks 21, ~1 h), then the full steering run (~1 h).
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-steer.log bash scripts/run_steer.sh
PY=.venv/bin/python
mkdir -p data/logs
echo "RUN_START=$(date -u +%FT%TZ)"
$PY -m spatialcells.vggt_steer --blocks 19 --n-fit 96 --n-test 8 --n-units 3 --n-null 2 --out data/logs/vggt_steer_smoke \
  > data/logs/vggt_steer_smoke.log 2>&1 || echo "SMOKE FAILED"
[ -f data/acts/vggt-single/L21.npy ] || $PY -m spatialcells.vggt_cells --mode single --blocks 21 > data/logs/vggt_cells_l21.log 2>&1
$PY -m spatialcells.vggt_steer > data/logs/vggt_steer.log 2>&1 || echo "STEER FAILED"
echo "RUN_END=$(date -u +%FT%TZ)"
