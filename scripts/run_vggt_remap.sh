#!/usr/bin/env bash
# VGGT remapping (second room) and position-patching tests. Run inside the container under the GPU lock:
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-vggt-remap.log bash scripts/run_vggt_remap.sh
# Needs data/walks-roomB (python -m spatialcells.render_data --room B --out data/walks-roomB, CPU).
set -uo pipefail
P=/workspace/MechInterp/.venv/bin/python
echo "RUN_START=$(date -u +%FT%TZ)"
$P -m spatialcells.vggt_patch --n-fit 64 --n-test 8 --blocks 19 --out results/vggt_patch_smoke.json || echo "FAILED patch smoke"
$P -m spatialcells.vggt_cells --mode single --room B --data data/walks-roomB || echo "FAILED vggt_cells room B"
$P -m spatialcells.vggt_remap || echo "FAILED remap"
$P -m spatialcells.vggt_patch || echo "FAILED patch"
echo "RUN_END=$(date -u +%FT%TZ)"
