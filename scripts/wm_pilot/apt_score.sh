#!/bin/bash
# Follow-up pilot WP, amendment 2: score the apartments (CPU; the Mac or the PC). analyze.py --quadrant --arena into
# results/wm_pilot/apartments/ReplicaApt<K>/{units,probes}/<model>/ (one folder per model, so jobs never share a file),
# and WP2 with the own-walk decoder into .../novelty_ownwalk_<model>.json.
# usage: scripts/wm_pilot/apt_score.sh K pixels|pixels-probes|ttt3r|vggt
set -e
K=$1
S=ReplicaApt$K
W=data/walks-room$S
P=$W-p256
O=results/wm_pilot/apartments/$S
PY=${PY:-$([ -x .venv-mac/bin/python ] && echo .venv-mac/bin/python || echo .venv/bin/python)}
A="$PY -m spatialcells.analyze --quadrant --arena"
case "$2" in
  pixels) $A --data $W --out $O/units/pixels --tags pixels pixels+ema8 ;;
  pixels-probes) $A --data $P-arena --out $O/probes/pixels --tags pixels ;;
  ttt3r)
    $A --data $W --out $O/units/ttt3r --tags ttt3r-walk-$S ttt3r-single-$S ttt3r-single-$S+ema8
    $A --data $P-arena --out $O/probes/ttt3r --tags ttt3r-probes-walk-$S ttt3r-probes-single-$S
    $PY -m spatialcells.wm_pilot novelty --own-walk --data $P --arena-dir $P-arena --out $O/novelty_ownwalk_ttt3r.json \
      --pairs ttt3r-walk-$S:ttt3r-probes-walk-$S ttt3r-single-$S:ttt3r-probes-single-$S ;;
  vggt)
    $A --data $W --out $O/units/vggt --tags vggt-single-$S
    $A --data $P-arena --out $O/probes/vggt --tags vggt-single-probes-$S
    $PY -m spatialcells.wm_pilot novelty --own-walk --data $P --arena-dir $P-arena --out $O/novelty_ownwalk_vggt.json \
      --pairs vggt-single-$S:vggt-single-probes-$S ;;
  *) echo "unknown step $2" >&2; exit 2 ;;
esac
