#!/usr/bin/env bash
# Experiment 9b on Apple silicon: scripts/run_roomB.sh's five runs with --device mps, after a check
# against the published room-A result (fine-tuned, 64-frame walk: 94/102 novel on CUDA). Answers go to
# results/two_choice/*-mps.jsonl, never mixed with a CUDA run's. Needs .venv-mac (scripts/mac/setup.sh),
# the adapter (scripts/vast1/fetch.sh adapter), room B's trials.json from the GPU host, and the walks.
# Rendering the walks here takes minutes and gives the GPU host's arrays byte for byte (seeded):
#   python -m spatialcells.render_data --room B --out data/walks-roomB    (then render_probes --room B
#   --data data/walks-roomB, and the same without --room for room A in data/walks)
# The first run downloads Qwen3-VL-8B (~17.5 GB) into the Hugging Face cache. From the project root,
# kept awake:
#   nohup caffeinate -i bash scripts/mac/run_roomB.sh > data/logs/mac_roomB.log 2>&1 &
# Each run skips trials already answered, so it can be stopped and restarted.
set -euo pipefail
P=.venv-mac/bin/python
B=Qwen/Qwen3-VL-8B-Instruct
A=data/ft/qwen8b-proc
D="--device mps"
export PYTORCH_ENABLE_MPS_FALLBACK=0     # fail loudly rather than run an op on the CPU
mkdir -p data/logs
echo "RUN_START=$(date -u +%FT%TZ)"
$P -m spatialcells.two_choice --model $B --adapter $A --trials dissimilar --max-images 64 $D > data/logs/tc_mps_roomA_ft_max64.log 2>&1
$P -m spatialcells.two_choice --model $B --trials roomB --max-images 64 $D > data/logs/tc_mps_roomB_base_max64.log 2>&1
$P -m spatialcells.two_choice --model $B --trials roomB --no-walk $D > data/logs/tc_mps_roomB_base_nowalk.log 2>&1
$P -m spatialcells.two_choice --model $B --adapter $A --trials roomB --max-images 64 $D > data/logs/tc_mps_roomB_ft_max64.log 2>&1
$P -m spatialcells.two_choice --model $B --adapter $A --trials roomB --no-walk $D > data/logs/tc_mps_roomB_ft_nowalk.log 2>&1
$P -m spatialcells.two_choice --model $B --adapter $A --trials roomB --max-images 64 --shuffle-walk $D > data/logs/tc_mps_roomB_ft_shuffled.log 2>&1
$P -m spatialcells.two_choice --score > data/logs/tc_mps_score.log 2>&1
echo "RUN_END=$(date -u +%FT%TZ)"
