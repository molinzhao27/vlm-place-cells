#!/usr/bin/env bash
# SenseNova-SI-1.1-Qwen3-VL-8B (Qwen3-VL-8B fine-tuned for spatial intelligence) on every test
# the base Qwen3-VL-8B had, plus the two-choice question for both. Through the GPU lock:
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-sensenova.log bash scripts/run_sensenova.sh
set -euo pipefail
P=.venv/bin/python
M=sn8b
echo "RUN_START=$(date -u +%FT%TZ)"
for mode in walk shuffled probes-only single; do
  $P -m spatialcells.extract --model $M --mode $mode
done
$P -m spatialcells.analyze_probes --models qwen qwen8b qwen32b qwen38 $M
$P -m spatialcells.analyze --tags $M-walk $M-single $M-single+ema8
echo "UNITS_DONE=$(date -u +%FT%TZ)"
$P -m spatialcells.behave --model $M
$P -m spatialcells.analyze_behave --models qwen qwen8b qwen32b qwen38 $M
for m in qwen8b $M; do
  $P -m spatialcells.behave_reason --model $m --walks 16 --direct-only
  $P -m spatialcells.behave_reason --model $m --score
done
echo "RUN_END=$(date -u +%FT%TZ)"
