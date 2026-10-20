#!/usr/bin/env bash
# Experiment 9b: retrain run_finetune.sh's LoRA with another seed (the same data and settings) and test it
# on room A's and room B's lure-free trials: 64-frame walk, no walk and shuffled walk.
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-ft-s1.log bash scripts/run_ft_seed.sh 1
set -uo pipefail
S=${1:?usage: run_ft_seed.sh SEED}
P=.venv/bin/python
B=Qwen/Qwen3-VL-8B-Instruct
A=data/ft/qwen8b-proc-s$S
echo "RUN_START=$(date -u +%FT%TZ)"
$P -m spatialcells.finetune --seed $S --out $A > data/logs/finetune_s$S.log 2>&1 || { echo "FINETUNE FAILED"; exit 1; }
echo "FINETUNE_DONE=$(date -u +%FT%TZ)"
for T in dissimilar roomB; do
  $P -m spatialcells.two_choice --model $B --adapter $A --trials $T --max-images 64 > data/logs/tc_s${S}_${T}_max64.log 2>&1
  $P -m spatialcells.two_choice --model $B --adapter $A --trials $T --no-walk > data/logs/tc_s${S}_${T}_nowalk.log 2>&1
  $P -m spatialcells.two_choice --model $B --adapter $A --trials $T --max-images 64 --shuffle-walk > data/logs/tc_s${S}_${T}_shuffled.log 2>&1
done
$P -m spatialcells.two_choice --score > data/logs/tc_score.log 2>&1
echo "RUN_END=$(date -u +%FT%TZ)"
