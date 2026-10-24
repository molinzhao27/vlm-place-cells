#!/usr/bin/env bash
# Photoreal rooms, step 5 (docs/photoreal_plan.md): walks and teleport probes in Replica scenes (default: the
# three apartments; the single rooms gave almost no novel trials), rendered with habitat-sim 0.3.3 (vertex colours, --zup frame fix), 32 walks of 256
# steps and 64 probes per walk each, into data/walks-room<Name>/ (the files render_data.py and
# render_probes.py write, plus landmarks.json). Through the lock, from here:
#   scripts/rtx4070/run.sh --detach logs/render_replica.log bash scripts/rtx4070/render_replica.sh
set -euo pipefail
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
R=${HABITAT_DATA:-$HOME/habitat-data}/replica                       # setup_habitat.sh / get_replica.sh put it here
PY=${HABITAT_PYTHON:-$HOME/miniforge3/envs/hab033/bin/python}     # the habitat-sim 0.3.3 env (setup_habitat.sh)
echo "RUN_START=$(date -u +%FT%TZ)"
# SCENES: "<replica scene> <our name> <seed>" triples, separated by ";"
SCENES=${SCENES:-"apartment_0 ReplicaApt0 200;apartment_1 ReplicaApt1 201;apartment_2 ReplicaApt2 202"}
IFS=";" read -ra SPECS <<< "$SCENES"
for spec in "${SPECS[@]}"; do
  set -- $spec
  $PY -m spatialcells.habitat_scene --dataset "$R/replica.scene_dataset_config.json" --scene "$1" --zup \
    --walks-out "data/walks-room$2" --walks 32 --steps 256 --probes-per-walk "${PROBES:-128}" --margin "${MARGIN:-0.15}" --seed "$3" 2>&1 | grep '^{'
done
echo "RUN_END=$(date -u +%FT%TZ)"
