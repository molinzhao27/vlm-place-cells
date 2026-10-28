#!/bin/bash
# Follow-up pilot WP, amendment 1 (docs/experiments.md), on the RTX 4070 PC: the first 16 walks of room A and of
# Replica apartment 0 re-rendered with every turn split into sub-steps of at most 15 degrees (spatialcells/substep.py;
# the re-rendered originals are checked byte for byte), then CUT3R's and TTT3R's gates on them, scoring the original
# steps only (results/wm_pilot/gate_<rule>_<scene>-sub15.json).
# usage: scripts/wm_pilot/pc_sub15.sh render-A|render-apt|gate cut3r|ttt3r
set -e
B=/mnt/bigd/wm_models
case "$1" in
  render-A) .venv/bin/python -m spatialcells.substep --data data/walks --out data/walks-sub15 --room A --workers ${WORKERS:-12} ;;
  render-apt)
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
    ${HABITAT_PYTHON:-$HOME/miniforge3/envs/hab033/bin/python} -m spatialcells.substep --data data/walks-roomReplicaApt0 \
      --out data/walks-roomReplicaApt0-sub15 --habitat ${HABITAT_DATA:-$HOME/habitat-data}/replica/replica.scene_dataset_config.json \
      --scene apartment_0 --zup --seed 200 ;;
  gate)
    REPO=$([ "$2" = ttt3r ] && echo TTT3R || echo CUT3R)
    A="--rule $2 --repo $B/$REPO --ckpt $B/weights/cut3r/cut3r_512_dpt_4_64.pth --device cuda --batch ${BATCH:-2}"
    $B/envs/cut3r/bin/python -m spatialcells.wm_cut3r gate $A --data data/walks-sub15 --scene A-sub15
    $B/envs/cut3r/bin/python -m spatialcells.wm_cut3r gate $A --data data/walks-roomReplicaApt0-sub15 --scene ReplicaApt0-sub15 ;;
  *) echo "unknown step $1" >&2; exit 2 ;;
esac
