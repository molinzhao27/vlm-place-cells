#!/usr/bin/env bash
# Experiment 9: teach Qwen3-VL-8B the two-choice task on procedural rooms (LoRA), test it on room A's
# lure-free trials, then rerun the unit and teleport-probe analyses on the fine-tuned model.
# Needs data/proc (python -m spatialcells.procedural_data, CPU). Through the GPU lock:
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-finetune.log bash scripts/run_finetune.sh
set -uo pipefail
P=.venv/bin/python
B=Qwen/Qwen3-VL-8B-Instruct
A=data/ft/qwen8b-proc
mkdir -p data/logs
echo "RUN_START=$(date -u +%FT%TZ)"
$P -m spatialcells.finetune --max-rooms 16 --epochs 1 --accum 4 --val 4 --eval-every 2 --out data/ft/smoke \
  > data/logs/finetune_smoke.log 2>&1 || { echo "SMOKE FAILED"; exit 1; }
$P -m spatialcells.finetune --out $A > data/logs/finetune.log 2>&1 || { echo "FINETUNE FAILED"; exit 1; }
echo "FINETUNE_DONE=$(date -u +%FT%TZ)"
# room A, lure-free trials: 64-frame walk (the training format), no walk, and the full 256-frame walk
$P -m spatialcells.two_choice --model $B --trials dissimilar --max-images 64 > data/logs/tc_base_max64.log 2>&1
$P -m spatialcells.two_choice --model $B --adapter $A --trials dissimilar --max-images 64 > data/logs/tc_ft_max64.log 2>&1
$P -m spatialcells.two_choice --model $B --adapter $A --trials dissimilar --no-walk > data/logs/tc_ft_nowalk.log 2>&1
$P -m spatialcells.two_choice --model $B --adapter $A --trials dissimilar > data/logs/tc_ft_256.log 2>&1
$P -m spatialcells.two_choice --model $B --adapter $A --trials original --max-images 64 > data/logs/tc_ft_orig.log 2>&1
$P -m spatialcells.two_choice --model $B --trials original --max-images 64 > data/logs/tc_base_orig.log 2>&1
$P -m spatialcells.two_choice --score > data/logs/tc_score.log 2>&1
echo "EVAL_DONE=$(date -u +%FT%TZ)"
for mode in walk shuffled probes-only single; do
  $P -m spatialcells.extract --model qwen8b-ft --mode $mode > data/logs/extract_ft_$mode.log 2>&1 || echo "EXTRACT $mode FAILED"
done
$P -m spatialcells.analyze_probes --models qwen qwen8b qwen32b qwen38 sn8b qwen8b-ft > data/logs/probes_ft.log 2>&1
$P -m spatialcells.analyze --tags qwen8b-ft-walk qwen8b-ft-single qwen8b-ft-single+ema8 > data/logs/analyze_ft.log 2>&1
echo "RUN_END=$(date -u +%FT%TZ)"
