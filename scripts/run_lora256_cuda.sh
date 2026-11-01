#!/usr/bin/env bash
# Experiment 24 on vast-3 (CUDA), amendment 1 in docs/experiments.md: one four-epoch LoRA run (experiment 9c)
# with the full 256-frame walk Opus saw, on the novel trials of the ten held-out procedural rooms. First the
# 64-frame walk on CUDA for P5005's novel trials (the one room whose 64-frame answers came from the Mac; tag
# ...-max64-parity-cuda so it never mixes with experiment 9c's files), then the 256-frame walk in every room
# (two_choice's default tag, ...-room<R>-256px.jsonl). Each run skips trials already answered, so the job can
# be restarted. One job per run, through gq:
#   gq submit --on vast3 --project mechinterp --name e24_256f_s5 --gpu --est 2h --priority 150 \
#     --env A=data/ft/qwen8b-proc-e4-s5 -- /bin/bash scripts/run_lora256_cuda.sh
# Analysis (Mac, after fetching the files): .venv-claude/bin/python scripts/lora256_analysis.py
set -euo pipefail
P=.venv/bin/python
B=Qwen/Qwen3-VL-8B-Instruct
A=${A:?set A to the adapter folder, e.g. data/ft/qwen8b-proc-e4-s5}
ROOMS=${ROOMS:-"P5000 P5001 P5002 P5003 P5004 P5005 P5006 P5007 P5008 P5009"}
PARITY=${PARITY:-"P5005"}
NAME=Qwen3-VL-8B-Instruct+$(basename "${A%/}")
echo "RUN_START=$(date -u +%FT%TZ) $A"
for R in $PARITY; do
  echo "$(date -u +%T) room$R $A --max-images 64 (CUDA parity)"
  $P scripts/lora256_run.py --model $B --adapter "$A" --trials "room$R" --max-images 64 \
    --tag "$NAME-room$R-256px-max64-parity-cuda"
done
for R in $ROOMS; do
  echo "$(date -u +%T) room$R $A 256-frame walk"
  $P scripts/lora256_run.py --model $B --adapter "$A" --trials "room$R"
done
echo "RUN_END=$(date -u +%FT%TZ)"
