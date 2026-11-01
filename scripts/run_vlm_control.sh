#!/usr/bin/env bash
# Experiment 29 amendment 1 (docs/experiments.md) on vast-3. MODE=build (CPU): the control frames and the stimulus check,
# which stops the chain if an agent gives the wrong verdict. MODE=run (GPU): one fine-tuned run in the three new conditions.
#   gq submit --on vast3 --project mechinterp --name e29c_build --cpu --est 40m --env MODE=build -- /bin/bash scripts/run_vlm_control.sh
#   gq submit --on vast3 --project mechinterp --name e29c_s5 --gpu --est 45m --after <build> --env MODE=run \
#     --env A=data/ft/qwen8b-proc-e4-s5 -- /bin/bash scripts/run_vlm_control.sh
set -euo pipefail
P=.venv/bin/python
echo "RUN_START=$(date -u +%FT%TZ) ${MODE:?} ${A:-}"
md5sum spatialcells/vlm_intervention.py scripts/vlm_control_analysis.py results/vlm_intervention/specs.json
if [[ $MODE == build ]]; then
  $P -m spatialcells.vlm_intervention --build-control --workers 4
  $P -m spatialcells.vlm_intervention --control-stimulus --workers 4 --threads 4
  $P scripts/vlm_control_analysis.py --check-stimulus
else
  $P scripts/vlm_control_analysis.py --check-stimulus | tail -1
  $P -m spatialcells.vlm_intervention --run block-there far-far far-close --adapter "${A:?}" --device cuda --workers 4
fi
echo "RUN_END=$(date -u +%FT%TZ)"
