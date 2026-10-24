#!/usr/bin/env bash
# The probe-choice sweep (vggt_patch.py --probes) on the 12 GB 4070: the GPU host's smoke test, then
# blocks 15, 19 and 21, with smaller batches (BATCH, default 2: batch 8 ran out of the 4070's 11.6 GB;
# the GPU host used 48; batching does not change the numbers). Writes
# results/vggt_patch_probes_4070.json and .png. Check against the published runs before using it: the
# clean target error should be ~0.43 m and block 21's "position PCA 10" transfer ~0.855.
# Through the lock, from here:
#   scripts/rtx4070/run.sh --detach logs/vggt_probes.log bash scripts/rtx4070/run_vggt_probes.sh
set -euo pipefail
P=.venv/bin/python
V="--vggt ${VGGT_DIR:?run through scripts/rtx4070/run.sh} --batch ${BATCH:-2}"
mkdir -p results
echo "RUN_START=$(date -u +%FT%TZ)"
$P -m spatialcells.vggt_patch --probes --blocks 21 --n-fit 512 --n-test 16 $V --out /tmp/vggt_patch_probes_smoke.json
$P -m spatialcells.vggt_patch --probes --blocks 15 19 21 $V --out results/vggt_patch_probes_4070.json
echo "RUN_END=$(date -u +%FT%TZ)"
