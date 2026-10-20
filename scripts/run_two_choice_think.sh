#!/usr/bin/env bash
# Lure-free two-choice trials with thinking on (Opus 5.5 always thinks, so this is the fair
# comparison). No walk first (fast), then with the walk (cached walk, batched lockstep decoding).
# Qwen3.6 has no effort setting and thinks long, so it gets 8192 thinking tokens; Qwen3.8 runs
# at its medium effort with 4096. Through the GPU lock:
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-two-choice-think.log bash scripts/run_two_choice_think.sh
set -euo pipefail
P=.venv/bin/python
TC="$P -m spatialcells.two_choice"
echo "RUN_START=$(date -u +%FT%TZ)"
for s in 512 256; do
  $TC --model Qwen/Qwen3.6-35B-A3B --trials dissimilar --size $s --no-walk --think --max-new 8192 --batch 16
  $TC --model Qwen/Qwen3.8-27B --trials dissimilar --size $s --no-walk --think --effort medium --max-new 4096 --batch 16
done
$TC --score
$P -m spatialcells.behave_reason --model qwen38 --trials dissimilar --effort medium --budget 4096 --batch 12
$TC --score
$P -m spatialcells.behave_reason --model qwen36 --trials dissimilar --budget 8192 --batch 12
$TC --score
echo "RUN_END=$(date -u +%FT%TZ)"
