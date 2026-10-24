#!/usr/bin/env bash
# Experiment 12's VGGT positive control on the 4070 (vggt_control.py --trial-set), one trial at a
# time (every 8th walk frame: 35 images each; 67 do not fit 12 GB). Needs results/export/dissimilar-room<R>/trials.json copied over and the
# walks rendered there. Through the lock, from here:
#   scripts/rtx4070/run.sh --detach logs/vggt_replica.log bash scripts/rtx4070/run_vggt_replica.sh
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "RUN_START=$(date -u +%FT%TZ)"
for R in ${ROOMS:-ReplicaApt0 ReplicaApt1 ReplicaApt2}; do
  .venv/bin/python -m spatialcells.vggt_control --trial-set "$R" --stride "${STRIDE:-8}" --batch "${BATCH:-1}" --vggt "${VGGT_DIR:?run through run.sh}"
done
echo "RUN_END=$(date -u +%FT%TZ)"
