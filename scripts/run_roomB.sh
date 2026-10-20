#!/usr/bin/env bash
# Experiment 9b: base and fine-tuned Qwen3-VL-8B on room B's lure-free trials. Needs room B's walks,
# probes and trials (CPU):
#   python -m spatialcells.render_data --room B --out data/walks-roomB
#   python -m spatialcells.render_probes --room B --data data/walks-roomB
#   python -m spatialcells.dissimilar_trials --room B
# then through the GPU lock:
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-roomB.log bash scripts/run_roomB.sh
P=.venv/bin/python
B=Qwen/Qwen3-VL-8B-Instruct
A=data/ft/qwen8b-proc
echo "RUN_START=$(date -u +%FT%TZ)"
$P -m spatialcells.two_choice --model $B --trials roomB --max-images 64 > data/logs/tc_roomB_base_max64.log 2>&1
$P -m spatialcells.two_choice --model $B --trials roomB --no-walk > data/logs/tc_roomB_base_nowalk.log 2>&1
$P -m spatialcells.two_choice --model $B --adapter $A --trials roomB --max-images 64 > data/logs/tc_roomB_ft_max64.log 2>&1
$P -m spatialcells.two_choice --model $B --adapter $A --trials roomB --no-walk > data/logs/tc_roomB_ft_nowalk.log 2>&1
$P -m spatialcells.two_choice --model $B --adapter $A --trials roomB --max-images 64 --shuffle-walk > data/logs/tc_roomB_ft_shuffled.log 2>&1
$P -m spatialcells.two_choice --score > data/logs/tc_score.log 2>&1
echo "RUN_END=$(date -u +%FT%TZ)"
