#!/usr/bin/env bash
# Experiment 9b: retrain run_finetune.sh's LoRA with another seed (the same data and settings) and test it
# on room A's and room B's lure-free trials: 64-frame walk, no walk and shuffled walk.
#   gq submit --on vast3 --project mechinterp --name lora_seed1 --gpu --est 2h -- /bin/bash scripts/run_ft_seed.sh 1
# An optional second argument names the adapter folder, e.g. the GPU control, a second seed-0 run on
# another GPU without touching experiment 9's adapter: run_ft_seed.sh 0 data/ft/qwen8b-proc-s0-5090
# EPOCHS (default 2, experiment 9's schedule) sets the training length, e.g. experiment 9c:
#   gq submit ... --env EPOCHS=4 -- /bin/bash scripts/run_ft_seed.sh 3 data/ft/qwen8b-proc-e4-s3
set -uo pipefail
S=${1:?usage: run_ft_seed.sh SEED [ADAPTER_DIR]}
P=.venv/bin/python
B=Qwen/Qwen3-VL-8B-Instruct
A=${2:-data/ft/qwen8b-proc-s$S}
E=${EPOCHS:-2}
N=$(basename $A)
echo "RUN_START=$(date -u +%FT%TZ) seed $S, $E epochs -> $A"
$P -m spatialcells.finetune --seed $S --epochs $E --out $A > data/logs/finetune_$N.log 2>&1 || { echo "FINETUNE FAILED"; exit 1; }
echo "FINETUNE_DONE=$(date -u +%FT%TZ)"
for T in dissimilar roomB; do
  $P -m spatialcells.two_choice --model $B --adapter $A --trials $T --max-images 64 > data/logs/tc_${N}_${T}_max64.log 2>&1
  $P -m spatialcells.two_choice --model $B --adapter $A --trials $T --no-walk > data/logs/tc_${N}_${T}_nowalk.log 2>&1
  $P -m spatialcells.two_choice --model $B --adapter $A --trials $T --max-images 64 --shuffle-walk > data/logs/tc_${N}_${T}_shuffled.log 2>&1
done
$P -m spatialcells.two_choice --score > data/logs/tc_score.log 2>&1
echo "RUN_END=$(date -u +%FT%TZ)"
