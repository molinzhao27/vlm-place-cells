#!/usr/bin/env bash
# Four-epoch internals experiment, part 1 (docs/experiments.md; vast-3's RTX 5090, through gq).
# Experiment 9's unit recordings (extract.py: walk, shuffled, probes-only, single; room A's 48 walks and
# their teleport probes) for run 0 again under the name qwen8b-ft-rerun (the parity check against the
# original vast-1 recording) and for the four-epoch runs, each followed by the strict scoring:
#   analyze.py --quadrant --out results/e4_internals   (walk, single, single+ema8: experiment 13e's unit test)
#   e4_internals probes                                (teleport probes, opposite 90-degree sectors)
# Runs go in the order given (MODELS); a recording whose 7 layer files exist is not redone.
#   gq submit --on vast3 --project mechinterp --name e4int_units --gpu --priority 150 --est 1h30m \
#     --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -- /bin/bash scripts/e4_internals_units.sh
set -uo pipefail
P=.venv/bin/python
MODELS=${MODELS:-"qwen8b-ft-rerun qwen8b-ft-e4-s5 qwen8b-ft-e4-s7 qwen8b-ft-e4-s3 qwen8b-ft-e4-s4 qwen8b-ft-e4-s6"}
OUT=results/e4_internals
LOG=data/logs/e4_internals
mkdir -p $OUT $LOG
echo "RUN_START=$(date -u +%FT%TZ) models: $MODELS"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
free_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
echo "free disk: ${free_gb} GB (needs about 50)"
[[ $free_gb -ge 60 ]] || { echo "NOT ENOUGH DISK"; exit 1; }
md5sum spatialcells/extract.py spatialcells/analyze.py spatialcells/e4_internals.py data/walks/*.np* | tee $OUT/units_inputs.md5
complete() { [[ -d data/acts/$1 ]] && [[ $(ls data/acts/$1/L*.npy 2>/dev/null | wc -l) -eq 7 ]]; }
fail=0
for m in $MODELS; do
  adapter=$($P -c "from spatialcells.extract import FINETUNED; print(FINETUNED['$m'][1])") || { echo "UNKNOWN MODEL $m"; fail=1; continue; }
  md5sum $adapter/adapter_model.safetensors | tee -a $OUT/units_inputs.md5
  for mode in walk shuffled probes-only single; do
    case $mode in
      walk|shuffled) done_tags="$m-$mode $m-probes-$mode" ;;
      *) done_tags="$m-$mode" ;;
    esac
    ok=1; for t in $done_tags; do complete $t || ok=0; done
    if [[ $ok -eq 1 ]]; then echo "$m $mode: recorded already"; continue; fi
    echo "$m $mode: extract $(date -u +%T)"
    $P -m spatialcells.extract --model $m --mode $mode > $LOG/extract_${m}_${mode}.log 2>&1 \
      || { echo "EXTRACT $m $mode FAILED (see $LOG/extract_${m}_${mode}.log)"; tail -3 $LOG/extract_${m}_${mode}.log; fail=1; }
    grep -E "context|frames, batch" $LOG/extract_${m}_${mode}.log
  done
  echo "$m: score $(date -u +%T)"
  $P -m spatialcells.analyze --quadrant --out $OUT --tags $m-walk $m-single $m-single+ema8 > $LOG/analyze_$m.log 2>&1 \
    || { echo "ANALYZE $m FAILED"; tail -3 $LOG/analyze_$m.log; fail=1; }
  $P -m spatialcells.e4_internals probes --models $m > $LOG/probes_$m.log 2>&1 \
    || { echo "PROBES $m FAILED"; tail -3 $LOG/probes_$m.log; fail=1; }
  $P - "$OUT/summary_quadrant.json" "$m" <<'EOF'
import json, sys
S, m = json.load(open(sys.argv[1])), sys.argv[2]
for c in ("walk", "single", "single+ema8"):
    v = S.get(f"{m}-{c}", {})
    print(f"  {m}-{c}: strictly invariant share of place units by layer: "
          + " ".join(f"{L}:{x['frac_place_heading_invariant_quadrant'] / x['frac_place']:.3f}" for L, x in v.items() if x.get("frac_place")))
EOF
done
echo "RUN_END=$(date -u +%FT%TZ) failures=$fail"
exit $fail
