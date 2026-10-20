#!/usr/bin/env bash
# Follow-ups: Qwen3.8 no-walk with room to answer, no-walk at higher
# resolution for Qwen3.6 and Gemma, and no-walk on the original trials (Claude's no-walk control).
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-two-choice-more.log bash scripts/run_two_choice_more.sh
set -euo pipefail
P=.venv/bin/python
TC="$P -m spatialcells.two_choice"
echo "RUN_START=$(date -u +%FT%TZ)"
# Hy-Embodied-VLM-1.0 was dropped: with transformers 4.57.6 its remote modeling code leaves layer 0's
# dense MLP and the q/k norms unused and initialises experts at random, so its answers are noise.
for s in 512 1024; do
  $TC --model Qwen/Qwen3.6-35B-A3B --trials dissimilar --size $s --no-walk
  $TC --model google/gemma-4-31B-it --trials dissimilar --size $s --no-walk --image-tokens 280
done
for s in 256 512; do $TC --model Qwen/Qwen3-VL-8B-Instruct --trials original --size $s --no-walk; done
$TC --model Qwen/Qwen3.6-35B-A3B --trials dissimilar --max-images 128
# Qwen3.8 answers in prose without thinking: let it finish (the earlier 16-token runs are kept as -cut)
for s in 256 512 1024; do
  f=results/two_choice/Qwen3.8-27B-dissimilar-${s}px-nowalk.jsonl
  [[ -f ${f%.jsonl}-cut16.jsonl.bak ]] || { [[ -f $f ]] && mv $f ${f%.jsonl}-cut16.jsonl.bak; } || true
  $TC --model Qwen/Qwen3.8-27B --trials dissimilar --size $s --no-walk --max-new 768
done
$TC --score
echo "RUN_END=$(date -u +%FT%TZ)"
