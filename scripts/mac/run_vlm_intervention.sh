#!/usr/bin/env bash
# Experiment 29 (pre-registered in docs/experiments.md) on Apple silicon: run 0 (the original two-epoch LoRA, whose
# experiment-10 and experiment-15 answers were made on this Mac) on the novel trials of P5000-P5009, with the walk
# frames around the near option swapped (spatialcells/vlm_intervention.py, --device mps). One process: clean first,
# then its check against experiment 10 (the job stops below 99% identical answers), then the four swaps and the
# random-frames control. The near-set attention cut for run 0 is experiment 15's (same trials, same device). Each
# condition skips trials already answered, so the job can be restarted. Through gq, behind experiment 24:
#   gq submit --on mac --project mechinterp --name e29_s0 --gpu --est 11h30m --priority 150 \
#     -- /bin/bash scripts/mac/run_vlm_intervention.sh
set -euo pipefail
P=.venv-mac/bin/python
A=${A:-data/ft/qwen8b-proc}
CONDS=${CONDS:-"clean here-toward here-away there-toward there-away random-away"}
export PYTORCH_ENABLE_MPS_FALLBACK=0     # fail loudly rather than run an op on the CPU
echo "RUN_START=$(date -u +%FT%TZ) $A"
md5 -r spatialcells/vlm_intervention.py spatialcells/frame_cut.py spatialcells/task_record.py spatialcells/two_choice.py \
  spatialcells/room.py results/vlm_intervention/specs.json data/frame_cut/dino_lookalike.json "$A"/adapter_model.safetensors
set +e
# shellcheck disable=SC2086
$P -m spatialcells.vlm_intervention --run $CONDS --adapter "$A" --device mps --workers 3 2>&1 \
  | grep --line-buffered -vE "processor_kwargs|Redirects|Loading weights"
rc=${PIPESTATUS[0]}
set -e
[[ $rc -eq 0 ]] || { echo "FAILED rc=$rc"; exit "$rc"; }
echo "RUN_END=$(date -u +%FT%TZ)"
