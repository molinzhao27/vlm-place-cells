#!/bin/bash
# Follow-up pilot WP0, WP1, WP2 and WP4 for CUT3R or TTT3R (same checkpoint; Mac, MPS), one step per job.
# usage: scripts/wm_pilot/cut3r_runs.sh cut3r|ttt3r gate|walk|single|alone|shuffled|probes-single [I/N]
#        scripts/wm_pilot/cut3r_runs.sh cut3r patch "1 2 3 4 5 6"|"7 8 9 10 11 12"|walk
set -e
PY=${PY:-$HOME/wm_models/envs/cut3r/bin/python}
R=$1; DEV=${DEV:-mps}; PART=${3:-1/1}
B=$([ "$R" = ttt3r ] && echo 4 || echo 16)       # TTT3R's rule needs the full attention maps
case "$2" in
  gate) $PY -m spatialcells.wm_cut3r gate --rule $R --device $DEV --batch $B
        $PY -m spatialcells.wm_cut3r gate --rule $R --device $DEV --batch $B --data data/walks-roomReplicaApt0 --scene ReplicaApt0 ;;
  walk) $PY -m spatialcells.wm_cut3r acts --rule $R --cond walk --probes data/walks-p256 --device $DEV --batch $B --part $PART ;;
  single|alone|shuffled) $PY -m spatialcells.wm_cut3r acts --rule $R --cond $2 --device $DEV --batch $B --part $PART ;;
  probes-single) $PY -m spatialcells.wm_cut3r acts --rule $R --cond single --frames probes --data data/walks-p256 --device $DEV --batch $B --part $PART ;;
  patch) if [ "$3" = walk ]; then $PY -m spatialcells.wm_cut3r patch --rule $R --device $DEV --batch $B --walk-mode
         else $PY -m spatialcells.wm_cut3r patch --rule $R --device $DEV --batch $B --blocks $3; fi ;;
  *) echo "unknown step $2" >&2; exit 2 ;;
esac
