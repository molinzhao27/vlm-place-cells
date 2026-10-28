#!/bin/bash
# Follow-up pilot WP0: ViNT, NoMaD and GNM competence gates in room A and Replica apartment 0 (Mac, MPS).
set -e
PY=${PY:-$HOME/wm_models/envs/vint/bin/python}
for m in vint nomad gnm; do
  $PY -m spatialcells.wm_vint gate --model $m --device mps
  $PY -m spatialcells.wm_vint gate --model $m --device mps --data data/walks-roomReplicaApt0 --scene ReplicaApt0
done
