#!/usr/bin/env bash
# Experiment 9 follow-up: the fine-tuned Qwen3-VL-8B on room A's lure-free trials with its 64-frame
# walk shuffled (real frame numbers) and shuffled + relabelled, as Opus got in experiment 6.
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-ft-shuffle.log bash scripts/run_ft_shuffle.sh
P=.venv/bin/python
B=Qwen/Qwen3-VL-8B-Instruct
A=data/ft/qwen8b-proc
echo "RUN_START=$(date -u +%FT%TZ)"
$P -m spatialcells.two_choice --model $B --adapter $A --trials dissimilar --max-images 64 --shuffle-walk > data/logs/tc_ft_shuffled.log 2>&1
$P -m spatialcells.two_choice --model $B --adapter $A --trials dissimilar --max-images 64 --shuffle-walk --relabel > data/logs/tc_ft_relabel.log 2>&1
$P -m spatialcells.two_choice --score > data/logs/tc_score.log 2>&1
echo "RUN_END=$(date -u +%FT%TZ)"
