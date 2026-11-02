#!/usr/bin/env bash
# Experiment 33 (docs/experiments.md) on vast-3's RTX 5090, through gq: every room-A teleport view on its own as the new
# view of the two-choice prompt (spatialcells/single_query.py), for the five four-epoch runs and the base model: parity
# trials, then the ordered walk, the shuffled walk and no walk; MLP neurons and the residual stream at the view's image-end
# token, the last prompt token and the walk frames. single_query.py resumes from its done files and stops on a failed check.
# SMOKE=1: single_query.py --smoke for the first model only (into data/acts_sq_smoke/; the checks, a few views each).
#   gq submit --on vast3 --project mechinterp --name e33_smoke --gpu --est 15m --env SMOKE=1 -- /bin/bash scripts/exp33_record.sh
#   gq submit --on vast3 --project mechinterp --name e33_record --gpu --est 3h30m --after <smoke id> \
#     -- /bin/bash scripts/exp33_record.sh
set -uo pipefail
P=.venv/bin/python
MODELS=${MODELS:-"qwen8b-ft-e4-s5 qwen8b-ft-e4-s7 qwen8b-ft-e4-s3 qwen8b-ft-e4-s4 qwen8b-ft-e4-s6 qwen8b"}
SMOKE=${SMOKE:-0}
OUT=results/exp33
LOG=data/logs/exp33
mkdir -p $OUT $LOG
echo "RUN_START=$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
free_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
echo "free disk: ${free_gb} GB (the full run writes about 20)"
[[ $SMOKE == 1 || $free_gb -ge 30 ]] || { echo "NOT ENOUGH DISK"; exit 1; }
md5sum spatialcells/single_query.py spatialcells/two_choice.py spatialcells/extract.py spatialcells/behave.py \
  spatialcells/analyze_behave.py spatialcells/analyze_probes.py spatialcells/dissimilar_trials.py spatialcells/room.py \
  data/walks/meta.npz data/walks/probes_meta.npz data/walks/frames.npy data/walks/probes.npy \
  results/export/single_query/trials.json results/export/dissimilar/trials.json \
  data/ft/qwen8b-proc-e4-s*/adapter_model.safetensors | tee $OUT/record_inputs$([[ $SMOKE == 1 ]] && echo _smoke).md5
if [[ $SMOKE == 1 ]]; then
  m=${MODELS%% *}
  $P -u -m spatialcells.single_query record --model $m --smoke 2>&1 | tee $LOG/smoke_$m.log
  rc=${PIPESTATUS[0]}
  echo "RUN_END=$(date -u +%FT%TZ) rc=$rc"
  exit $rc
fi
complete() {   # every done file full: 3 x 3,072 views, 3 x 192 parity trials, 2 x 48 walks
  local d=data/acts_sq/$1 n=0
  for c in walk shuffled nowalk; do n=$((n + $(cat $d/done_$c.jsonl 2>/dev/null | wc -l))); done
  for c in walk shuffled nowalk; do n=$((n + $(cat $d/done_parity_$c.jsonl 2>/dev/null | wc -l))); done
  for c in walk shuffled; do n=$((n + $(cat $d/done_walkframes_$c.jsonl 2>/dev/null | wc -l))); done
  [[ $n -eq $((3 * 3072 + 3 * 192 + 2 * 48)) ]]
}
fail=0
for m in $MODELS; do
  if complete $m; then echo "$m: recorded already"; continue; fi
  echo "$m: record $(date -u +%T)"
  $P -u -m spatialcells.single_query record --model $m 2>&1 | tee -a $LOG/record_$m.log
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "RECORD $m FAILED"; fail=1; break; }
done
du -sh data/acts_sq
echo "RUN_END=$(date -u +%FT%TZ) failures=$fail"
exit $fail
