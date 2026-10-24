#!/usr/bin/env bash
# Experiment 14d (VGGT cross-room patching) on the 12 GB 4070, with the probe sweep's batch size (BATCH,
# default 2; batching does not change the numbers): a smoke test, then blocks 19 and 21 in room B ->
# results/vggt_patch_crossroom.json (~2 h). Needs data/walks and data/walks-roomB (frames.npy, meta.npz,
# and anchors.npz rendered in room B: vggt_cells.anchors("data/walks-roomB", "B")). The clean target error
# in room B should be near the remap run's 0.85 m; the room-A fit frames' near 0.43 m. (First run on
# 2026-10-23 as part of a combined 14d/14e job, logs/vggt_14de.log.)
# Through the lock, from here:
#   scripts/rtx4070/run.sh --detach logs/vggt_14d.log bash scripts/rtx4070/run_vggt_14d.sh
set -euo pipefail
P=.venv/bin/python
V="--vggt ${VGGT_DIR:?run through scripts/rtx4070/run.sh} --batch ${BATCH:-2}"
X="--crossroom A --data data/walks-roomB --room B"
mkdir -p results
echo "RUN_START=$(date -u +%FT%TZ)"
$P -m spatialcells.vggt_patch $X --blocks 21 --n-fit 256 --n-test 8 $V --out /tmp/vggt_patch_crossroom_smoke.json
$P -m spatialcells.vggt_patch $X --blocks 19 21 $V --out results/vggt_patch_crossroom.json
echo "RUN_END=$(date -u +%FT%TZ)"
