#!/usr/bin/env bash
# Experiment 35 (pre-registered in docs/experiments.md) on vast-3's RTX 5090, through gq: copying activations between
# trials inside the fine-tuned VLM (spatialcells/vlm_patch.py). The five four-epoch runs get every stage (sweep, fit,
# subspace); then the base model gets the view copies of the sweep and the fit at the union of the runs' layers.
# vlm_patch.py resumes from its done files and stops on a failed check. The pairs are rebuilt here from the committed
# inputs and must match the committed file's md5 (PAIRS_MD5). The summary is written at the end
# (results/exp35_summary.json); scripts/exp35_analysis.py --fetch copies the small files to the Mac.
# SMOKE=1: the first model only, vlm_patch.py run --smoke (data/vlm_patch_smoke/: 4 recipients, every check, the fit on
# walks 0-3); it prints the copied rows per second, which sets the main job's --est.
#   gq submit --on vast3 --project mechinterp --name e35_smoke --gpu --est 20m --env SMOKE=1 \
#     --env PAIRS_MD5=<md5 of results/export/vlm_patch/pairs.json> -- /bin/bash scripts/exp35_run.sh
#   gq submit --on vast3 --project mechinterp --name e35_run --gpu --est 5h --after <smoke id> \
#     --env PAIRS_MD5=<same md5> -- /bin/bash scripts/exp35_run.sh
set -uo pipefail
P=.venv/bin/python
MODELS=${MODELS:-"qwen8b-ft-e4-s5 qwen8b-ft-e4-s7 qwen8b-ft-e4-s3 qwen8b-ft-e4-s4 qwen8b-ft-e4-s6"}
BASE=${BASE:-qwen8b}
SMOKE=${SMOKE:-0}
BATCH=${BATCH:-8}
OUT=results/exp35
LOG=data/logs/exp35
mkdir -p $OUT $LOG
quiet() { grep --line-buffered -vE "processor_kwargs|Redirects|Loading weights" || true; }
echo "RUN_START=$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
free_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
echo "free disk: ${free_gb} GB (the full run writes about 1)"
[[ $free_gb -ge 5 ]] || { echo "NOT ENOUGH DISK"; exit 1; }
$P -m spatialcells.vlm_patch pairs 2>&1 | quiet > $LOG/pairs.log
got=$(md5sum results/export/vlm_patch/pairs.json | cut -d' ' -f1)
[[ -n ${PAIRS_MD5:-} && $got == "$PAIRS_MD5" ]] || { echo "PAIRS: md5 $got, expected ${PAIRS_MD5:-unset}"; exit 1; }
md5sum spatialcells/vlm_patch.py spatialcells/single_query.py spatialcells/two_choice.py spatialcells/extract.py \
  spatialcells/behave.py spatialcells/analyze_behave.py spatialcells/analyze_probes.py scripts/exp35_analysis.py \
  data/walks/meta.npz data/walks/probes_meta.npz data/walks/frames.npy data/walks/probes.npy \
  results/export/single_query/trials.json results/export/vlm_patch/pairs.json \
  data/ft/qwen8b-proc-e4-s*/adapter_model.safetensors \
  $(ls results/exp33/*/done_walk.jsonl data/acts_sq/*/done_walk.jsonl 2>/dev/null) \
  | tee $OUT/run_inputs$([[ $SMOKE == 1 ]] && echo _smoke).md5
if [[ $SMOKE == 1 ]]; then
  m=${MODELS%% *}
  $P -u -m spatialcells.vlm_patch run --model $m --smoke --batch $BATCH 2>&1 | quiet | tee $LOG/smoke_$m.log
  rc=${PIPESTATUS[0]}
  echo "RUN_END=$(date -u +%FT%TZ) rc=$rc"
  exit $rc
fi
fail=0
for m in $MODELS; do
  echo "$m: $(date -u +%T)"
  $P -u -m spatialcells.vlm_patch run --model $m --batch $BATCH 2>&1 | quiet | tee -a $LOG/run_$m.log
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "RUN $m FAILED"; fail=1; break; }
done
if [[ $fail == 0 ]]; then
  LAYERS=$($P -c "import json; print(*sorted({L for m in '$MODELS'.split() for L in json.load(open(f'data/vlm_patch/{m}/layers.json'))['layers']}))")
  echo "$BASE: view copies, fit at layers $LAYERS $(date -u +%T)"
  # shellcheck disable=SC2086
  $P -u -m spatialcells.vlm_patch run --model $BASE --stages sweep fit --sweep-sites view --fit-layers $LAYERS \
    --batch $BATCH 2>&1 | quiet | tee -a $LOG/run_$BASE.log
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "RUN $BASE FAILED"; fail=1; }
fi
[[ $fail == 0 ]] && $P scripts/exp35_analysis.py --root data/vlm_patch > $LOG/summary.log 2>&1 \
  && md5sum results/exp35_summary.json
du -sh data/vlm_patch
echo "RUN_END=$(date -u +%FT%TZ) failures=$fail"
exit $fail
