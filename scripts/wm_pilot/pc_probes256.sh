#!/bin/bash
# Follow-up pilot WP1 reference on the RTX 4070 PC: room A's 256-per-walk probes rendered here (seed 17, as on the
# Mac; checksums printed for comparison), the probe arena, then VGGT single (8 anchors + the probe) at blocks 19, 21.
# usage: scripts/wm_pilot/pc_probes256.sh render|vggt
set -e
P=.venv/bin/python
case "$1" in
  render)
    $P -m spatialcells.render_probes --data data/walks-p256 --per-walk 256 --seed 17 --workers 12
    $P -m spatialcells.wm_pilot arena
    cp data/walks/anchors.npz data/walks-p256-arena/
    md5sum data/walks-p256/probes.npy data/walks-p256/probes_meta.npz ;;
  vggt)
    mkdir -p /mnt/bigd/mechinterp/acts/vggt-single-probes256
    ln -sfn /mnt/bigd/mechinterp/acts/vggt-single-probes256 data/acts/vggt-single-probes256
    $P -m spatialcells.vggt_cells --mode single --data data/walks-p256-arena --vggt "$VGGT_DIR" --blocks 19 21 \
      --batch "${BATCH:-4}" --out-tag vggt-single-probes256 ;;
esac
