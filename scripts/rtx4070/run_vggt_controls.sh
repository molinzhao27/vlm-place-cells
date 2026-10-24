#!/usr/bin/env bash
# Experiment 13c's VGGT controls (vggt_patch.py --controls) on the 12 GB 4070: a smoke test, then blocks 19
# and 21, with the probe sweep's batch size (BATCH, default 2; batching does not change the numbers). Writes
# results/vggt_patch_controls.json. Check against the published runs before using it: the clean target error
# should be ~0.43 m and the clean heading error under 2 deg; at block 21, "full" transfer ~0.87,
# "position PCA 10" ~0.85 and "erase position PCA 10" ~3.38 m (block 19: 0.41, 0.40, 1.65 m).
# Through the lock, from here:
#   scripts/rtx4070/run.sh --detach logs/vggt_controls.log bash scripts/rtx4070/run_vggt_controls.sh
set -euo pipefail
P=.venv/bin/python
V="--vggt ${VGGT_DIR:?run through scripts/rtx4070/run.sh} --batch ${BATCH:-2}"
mkdir -p results
echo "RUN_START=$(date -u +%FT%TZ)"
$P -m spatialcells.vggt_patch --controls --blocks 21 --n-fit 512 --n-test 16 $V --out /tmp/vggt_patch_controls_smoke.json
$P -m spatialcells.vggt_patch --controls --blocks 19 21 $V --out results/vggt_patch_controls.json
echo "RUN_END=$(date -u +%FT%TZ)"
