#!/usr/bin/env bash
# Experiment 14e (VGGT remapping with a changed reference, amended 2026-10-23) on the 12 GB 4070: block 19,
# single mode, room A, with one anchor set -> data/acts/vggt-single-anchors-<set> (~3.7 h at BATCH 2).
# One job per set (standard, reordered, rot45), so other projects' jobs can run in between. The job that
# finds all three sets recorded then runs vggt_reference -> results/vggt_remap_reference.json (~10 min).
# Needs data/walks with anchors.npz and anchors-rot45.npz. The standard set's camera error should be ~0.42 m.
# Through the lock, from here, one per set:
#   scripts/rtx4070/run.sh --detach logs/vggt_14e_standard.log bash scripts/rtx4070/run_vggt_14e.sh standard
set -euo pipefail
P=.venv/bin/python
SET=${1:?anchor set: standard, reordered or rot45}
echo "RUN_START=$(date -u +%FT%TZ)"
$P -m spatialcells.vggt_cells --mode single --blocks 19 --anchor-set "$SET" \
  --vggt "${VGGT_DIR:?run through scripts/rtx4070/run.sh}" --batch "${BATCH:-2}"
if [[ -f data/acts/vggt-single-anchors-standard/pose.npz && -f data/acts/vggt-single-anchors-reordered/pose.npz &&
      -f data/acts/vggt-single-anchors-rot45/pose.npz ]]; then
  mkdir -p results
  $P -m spatialcells.vggt_reference
fi
echo "RUN_END=$(date -u +%FT%TZ)"
