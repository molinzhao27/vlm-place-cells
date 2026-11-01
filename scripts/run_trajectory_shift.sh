#!/usr/bin/env bash
# Experiment 30 (pre-registered in docs/experiments.md) on CUDA (vast-3): one four-epoch LoRA run on every novel trial of
# the shifted-walk experiment (standard and neighborhood trials of all four walk conditions, nine rooms), with
# experiment 10's 64-frame prompt, in one two_choice.py call (--trials roomE30, so the model loads once).
# First: data/walks-roomE30 must be unpacked from the Mac's payload (spatialcells/trajectory_shift.py --unpack; done
# here if the e30_unpack CPU job has not), then the parity check: experiment 10's 192 P5000 trials answered again and
# compared with this run's stored vast-3 answers; below 98% identical the job stops before any experiment-30 trial.
# The parity answers go to results/two_choice/e30/, out of two_choice --score. Restartable: answered trials are skipped.
#   gq submit --on vast3 --project mechinterp --name e30_s3 --gpu --est 1h20m --priority 150 --after <id> \
#     --env A=data/ft/qwen8b-proc-e4-s3 -- /bin/bash scripts/run_trajectory_shift.sh
set -euo pipefail
P=.venv/bin/python
B=Qwen/Qwen3-VL-8B-Instruct
A=${A:?set A to the adapter folder, e.g. data/ft/qwen8b-proc-e4-s3}
NAME=Qwen3-VL-8B-Instruct+$(basename "${A%/}")
echo "RUN_START=$(date -u +%FT%TZ) $A"
md5sum spatialcells/trajectory_shift.py spatialcells/two_choice.py spatialcells/room.py scripts/run_trajectory_shift.sh \
  results/export/dissimilar-roomE30/trials.json data/e30/payload/index.json "$A"/adapter_model.safetensors
$P -m spatialcells.trajectory_shift --check-unpacked || $P -m spatialcells.trajectory_shift --unpack
echo "$(date -u +%T) parity: experiment 10's P5000 trials"
mkdir -p results/two_choice/e30
$P -m spatialcells.two_choice --model $B --adapter "$A" --trials roomP5000 --max-images 64 \
  --tag "e30/$NAME-roomP5000-256px-max64-e30parity"
$P -m spatialcells.trajectory_shift --parity "$A"
echo "$(date -u +%T) experiment 30 trials"
$P -m spatialcells.two_choice --model $B --adapter "$A" --trials roomE30 --max-images 64
echo "RUN_END=$(date -u +%FT%TZ)"
