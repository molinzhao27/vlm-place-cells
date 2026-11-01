#!/usr/bin/env bash
# Four-epoch internals experiment, part 3 (docs/experiments.md; vast-3 CPU, through gq, after part 2).
# Experiment 14a's readouts (task_probe.py, unchanged except for its output file) and the two readouts it ran
# by hand (e4_internals task-extras), on the recordings made by scripts/e4_internals_task.sh.
#   gq submit --on vast3 --project mechinterp --name e4int_readout --cpu --priority 150 --after <part 2> \
#     --est 1h15m -- /bin/bash scripts/e4_internals_readout.sh
set -uo pipefail
P=.venv/bin/python
MODELS=${MODELS:-"qwen8b+qwen8b-proc-cuda qwen8b+qwen8b-proc-e4-s5 qwen8b+qwen8b-proc-e4-s7 qwen8b+qwen8b-proc-e4-s3 qwen8b+qwen8b-proc-e4-s4 qwen8b+qwen8b-proc-e4-s6"}
OUT=results/e4_internals
mkdir -p $OUT
echo "RUN_START=$(date -u +%FT%TZ)"
md5sum spatialcells/task_probe.py spatialcells/e4_internals.py | tee $OUT/readout_inputs.md5
have=""; for m in $MODELS; do [[ -f data/acts_task/$m/done.jsonl ]] && have="$have $m"; done
echo "recordings:$have"
fail=0
$P -m spatialcells.task_probe --models $have --out $OUT/task_probe.json || fail=1
$P -m spatialcells.e4_internals task-extras --models $have --out $OUT/task_extras.json || fail=1
echo "RUN_END=$(date -u +%FT%TZ) failures=$fail"
exit $fail
