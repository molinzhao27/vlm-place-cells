#!/usr/bin/env bash
# Experiment 29 (pre-registered in docs/experiments.md) on CUDA (vast-3): one fine-tuned run on the novel trials of
# P5000-P5009 with the walk frames around the near option swapped for frames at the same places facing toward or away
# from what the new view shows, or at places 3 m or more away (spatialcells/vlm_intervention.py). First a quick check
# that this machine renders the swap frames exactly as specs.json records them, then one process for every condition:
# clean first, then its check against experiment 10 (the job stops below 99% identical answers), then the swaps, the
# random-frames control and the near-set attention cut. Each condition skips trials already answered, so the job can be
# restarted. Needs data/frame_cut/dino_lookalike.json, results/vlm_intervention/specs.json and stimulus_check.json, which
# the Mac's prep job (scripts/mac/vlm_intervention_prep.sh) copies here only if the stimulus check passed.
#   gq submit --on vast3 --project mechinterp --name e29_s5 --gpu --est 2h --priority 150 \
#     --env A=data/ft/qwen8b-proc-e4-s5 -- /bin/bash scripts/run_vlm_intervention.sh
set -euo pipefail
P=.venv/bin/python
A=${A:?set A to the adapter folder, e.g. data/ft/qwen8b-proc-e4-s5}
CONDS=${CONDS:-"clean here-toward here-away there-toward there-away random-away near-cut"}
echo "RUN_START=$(date -u +%FT%TZ) $A"
md5sum spatialcells/vlm_intervention.py spatialcells/frame_cut.py spatialcells/task_record.py spatialcells/two_choice.py \
  spatialcells/room.py results/vlm_intervention/specs.json data/frame_cut/dino_lookalike.json "$A"/adapter_model.safetensors
echo "$(date -u +%T) stimulus gate (the Mac's stimulus check must have passed and its files arrived)"
$P scripts/vlm_intervention_analysis.py --check-stimulus | tail -1
echo "$(date -u +%T) render check"
$P -m spatialcells.vlm_intervention --render-check --per-room 1 --adapter "$A" --device cuda
echo "$(date -u +%T) conditions: $CONDS"
# shellcheck disable=SC2086
$P -m spatialcells.vlm_intervention --run $CONDS --adapter "$A" --device cuda --workers 4
echo "RUN_END=$(date -u +%FT%TZ)"
