#!/usr/bin/env bash
# Four-epoch internals experiment, part 2 (docs/experiments.md; vast-3's RTX 5090, through gq).
# Experiment 14a's recording under the real two-choice prompt (task_record.py: all 1,920 lure-free trials of
# P5000-P5009, the residual stream after every layer at four sites) for run 0 again on CUDA (the parity check
# against its MPS recording on the Mac) and for the four-epoch runs. task_record resumes from done.jsonl if
# the job is stopped, and stops if a prompt's token count differs from experiment 10's run of it.
#   gq submit --on vast3 --project mechinterp --name e4int_task --gpu --priority 150 --est 2h30m \
#     -- /bin/bash scripts/e4_internals_task.sh
set -uo pipefail
P=.venv/bin/python
# adapter:name pairs, in order (run 0 first, then the most and least shuffle-sensitive runs, then the rest)
RUNS=${RUNS:-"data/ft/qwen8b-proc:qwen8b+qwen8b-proc-cuda data/ft/qwen8b-proc-e4-s5:qwen8b+qwen8b-proc-e4-s5 data/ft/qwen8b-proc-e4-s7:qwen8b+qwen8b-proc-e4-s7 data/ft/qwen8b-proc-e4-s3:qwen8b+qwen8b-proc-e4-s3 data/ft/qwen8b-proc-e4-s4:qwen8b+qwen8b-proc-e4-s4 data/ft/qwen8b-proc-e4-s6:qwen8b+qwen8b-proc-e4-s6"}
OUT=results/e4_internals
LOG=data/logs/e4_internals
mkdir -p $OUT $LOG
echo "RUN_START=$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
free_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
echo "free disk: ${free_gb} GB (needs about 14)"
[[ $free_gb -ge 20 ]] || { echo "NOT ENOUGH DISK"; exit 1; }
md5sum spatialcells/task_record.py spatialcells/two_choice.py | tee $OUT/task_inputs.md5
fail=0
for pair in $RUNS; do
  adapter=${pair%%:*}; name=${pair#*:}
  md5sum $adapter/adapter_model.safetensors | tee -a $OUT/task_inputs.md5
  if [[ -f data/acts_task/$name/done.jsonl ]] && [[ $(wc -l < data/acts_task/$name/done.jsonl) -eq 1920 ]]; then
    echo "$name: recorded already"; continue
  fi
  echo "$name: record $(date -u +%T)"
  $P -m spatialcells.task_record --device cuda --adapter $adapter --name $name > $LOG/task_record_$name.log 2>&1 \
    || { echo "TASK_RECORD $name FAILED"; fail=1; }
  tail -2 $LOG/task_record_$name.log
done
echo "RUN_END=$(date -u +%FT%TZ) failures=$fail"
exit $fail
