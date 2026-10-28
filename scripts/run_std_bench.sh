#!/usr/bin/env bash
# Experiment 17: one benchmark for a list of models, stopping after about MAX_MIN minutes (on the Mac, so
# that experiment 9c's Mac tests never wait long). Each run skips answered questions, so the same job
# submitted again carries on; once everything is answered a job exits at once. The Mac uses .venv-mac
# and MPS, the container hosts .venv and CUDA.
#   gq submit --on mac --project mechinterp --name e17_mc_1 --gpu --est 25m --priority 150 \
#     --env MODELS="base run0" -- /bin/bash scripts/run_std_bench.sh
#   gq submit --on vast3 --project mechinterp --name e17_vsi_base --gpu --est 1h30m --priority 150 \
#     --env BENCH=vsibench --env MODELS=base --env MAX_MIN=600 -- /bin/bash scripts/run_std_bench.sh
# MODELS: base, run0, run1, run2, e4s3..e4s7, each optionally with ":blind". BENCH: mindcube (default) or
# vsibench. OUT: output folder (default results/std_bench).
set -u
if [[ $(uname) == Darwin ]]; then P=.venv-mac/bin/python; DEV=mps; else P=.venv/bin/python; DEV=cuda; fi
BENCH=${BENCH:-mindcube}
MAX_MIN=${MAX_MIN:-20}
OUT=${OUT:-results/std_bench}
LIMIT=${LIMIT:-}
adapter() {    # macOS bash 3.2: no associative arrays
  case $1 in
    base) echo "" ;; run0) echo data/ft/qwen8b-proc ;; run1) echo data/ft/qwen8b-proc-s1 ;;
    run2) echo data/ft/qwen8b-proc-s2 ;; e4s[3-7]) echo "data/ft/qwen8b-proc-e4-s${1#e4s}" ;;
    *) echo "unknown model $1" >&2; exit 1 ;;
  esac
}
start=$(date +%s)
echo "RUN_START=$(date -u +%FT%TZ) $BENCH $MODELS"
for M in $MODELS; do
  name=${M%%:*}; extra=()
  [[ $M == *:blind ]] && extra+=(--blind)
  [[ -n $LIMIT ]] && extra+=(--limit "$LIMIT")
  A=$(adapter "$name") || exit 1
  [[ -n $A ]] && extra+=(--adapter "$A")
  left=$(( MAX_MIN - ($(date +%s) - start) / 60 ))
  (( left < 3 )) && { echo "time is up before $M"; break; }
  file=$OUT/$BENCH/$name$([[ $M == *:blind ]] && echo -blind).jsonl
  want=$([[ $BENCH == vsibench ]] && echo 5130 || echo 1050)
  [[ -n $LIMIT ]] && want=$LIMIT
  if [[ -f $file ]] && (( $(wc -l < "$file") >= want )); then echo "$M complete"; continue; fi
  echo "$(date -u +%T) $BENCH $M (${left} min left)"
  $P -m spatialcells.std_bench --bench "$BENCH" --name "$name" --device "$DEV" --out "$OUT" \
     --max-minutes "$left" ${extra[@]+"${extra[@]}"} || exit 1
done
echo "RUN_END=$(date -u +%FT%TZ)"
