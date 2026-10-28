#!/bin/bash
# Follow-up pilot WP, amendment 2 (docs/experiments.md), on the RTX 4070 PC: TTT3R's WP1 and WP2 in the Replica
# apartments. Large outputs live on /mnt/bigd/mechinterp and are linked into data/.
# usage: scripts/wm_pilot/pc_apartments.sh render K      256 probes per walk + the 8 anchors (Habitat), probe arena
#        scripts/wm_pilot/pc_apartments.sh gate K        TTT3R's WP0 gate (apartments 1 and 2)
#        scripts/wm_pilot/pc_apartments.sh acts K walk|single|probes-single
#        scripts/wm_pilot/pc_apartments.sh vggt K        VGGT single, blocks 19 and 21, walk frames and probes
# Scoring: scripts/wm_pilot/apt_score.sh.
set -e
K=$2
S=ReplicaApt$K
W=data/walks-room$S
P=$W-p256
B=/mnt/bigd/wm_models
PY=.venv/bin/python
TTT="$B/envs/cut3r/bin/python -m spatialcells.wm_cut3r"
TA="--rule ttt3r --repo $B/TTT3R --ckpt $B/weights/cut3r/cut3r_512_dpt_4_64.pth --device cuda --batch ${BATCH:-2}"
big() {   # data/<name> -> /mnt/bigd/mechinterp/<name>
  mkdir -p /mnt/bigd/mechinterp/$1
  [ -e data/$1 ] || ln -s /mnt/bigd/mechinterp/$1 data/$1
}
case "$1" in
  render)
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
    big walks-room$S-p256
    ${HABITAT_PYTHON:-$HOME/miniforge3/envs/hab033/bin/python} -m spatialcells.habitat_scene \
      --dataset ${HABITAT_DATA:-$HOME/habitat-data}/replica/replica.scene_dataset_config.json --scene apartment_$K --zup \
      --probes-for $W --probes-out $P --probes-per-walk 256 --seed $((200 + K)) --probe-seed 17 --margin 0.15
    $PY -m spatialcells.wm_pilot arena --data $P --arena-dir $P-arena
    md5sum $W/anchors.npz $P/probes.npy $P/probes_meta.npz ;;
  gate) $TTT gate $TA --data $W --scene $S ;;
  acts)
    case "$3" in
      walk) big acts/ttt3r-walk-$S; big acts/ttt3r-probes-walk-$S
            $TTT acts $TA --cond walk --data $W --scene $S --probes $P ;;
      single) big acts/ttt3r-single-$S
              $TTT acts $TA --cond single --data $W --scene $S ;;
      probes-single) big acts/ttt3r-probes-single-$S
                     $TTT acts $TA --cond single --frames probes --data $P --scene $S ;;
      *) echo "unknown condition $3" >&2; exit 2 ;;
    esac ;;
  vggt)
    big acts/vggt-single-$S; big acts/vggt-single-probes-$S
    $PY -m spatialcells.vggt_cells --mode single --data $W --vggt "$VGGT_DIR" --blocks 19 21 --batch ${BATCH:-4} --out-tag vggt-single-$S
    $PY -m spatialcells.vggt_cells --mode single --data $P-arena --vggt "$VGGT_DIR" --blocks 19 21 --batch ${BATCH:-4} \
      --out-tag vggt-single-probes-$S ;;
  *) echo "unknown step $1" >&2; exit 2 ;;
esac
