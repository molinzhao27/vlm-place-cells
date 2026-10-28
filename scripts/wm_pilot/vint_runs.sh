#!/bin/bash
# Follow-up pilot WP1 and WP5 for one navigation policy (Mac, MPS): units in room A (walk, single, shuffled,
# and on the 256-per-walk probes when they exist) and the distance-head two-choice test on every registered trial.
# usage: scripts/wm_pilot/vint_runs.sh vint|nomad|gnm [acts|twochoice|all]
set -e
PY=${PY:-$HOME/wm_models/envs/vint/bin/python}
M=$1; WHAT=${2:-all}
if [ "$WHAT" != twochoice ]; then
  for c in walk single shuffled; do $PY -m spatialcells.wm_vint acts --model $M --cond $c --device mps; done
  if [ -f data/walks-p256/probes_meta.npz ]; then
    $PY -m spatialcells.wm_vint acts --model $M --cond probes-single --data data/walks-p256 --device mps
  fi
fi
if [ "$WHAT" != acts ]; then $PY -m spatialcells.wm_vint twochoice --model $M --device mps; fi
