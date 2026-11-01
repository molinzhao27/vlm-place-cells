#!/usr/bin/env bash
# Experiment 24 (pre-registered in docs/experiments.md): one four-epoch LoRA run (experiment 9c) with the full
# 256-frame walk Opus saw, on the novel trials of the ten held-out procedural rooms, on the Mac (MPS). First a
# 64-frame parity subset on MPS (P5000 and P5001 novel trials, compared with the CUDA answers from vast-3,
# tag ...-max64-parity-mps so it never mixes with the experiment-9c files). Each run skips trials already
# answered, so the job can be restarted. One job per run, through gq, behind experiment 19's Mac jobs:
#   gq submit --on mac --project mechinterp --name e24_256f_s5 --gpu --est 9h30m --priority 150 --pin \
#     --env A=data/ft/qwen8b-proc-e4-s5 -- /bin/bash scripts/mac/run_lora256.sh
# Analysis: .venv-claude/bin/python scripts/lora256_analysis.py
set -euo pipefail
P=.venv-mac/bin/python
B=Qwen/Qwen3-VL-8B-Instruct
A=${A:?set A to the adapter folder, e.g. data/ft/qwen8b-proc-e4-s5}
ROOMS=${ROOMS:-"P5000 P5001 P5002 P5003 P5004 P5005 P5006 P5007 P5008 P5009"}
PARITY=${PARITY:-"P5000 P5001"}
NAME=Qwen3-VL-8B-Instruct+$(basename "${A%/}")
export PYTORCH_ENABLE_MPS_FALLBACK=0     # fail loudly rather than run an op on the CPU
echo "RUN_START=$(date -u +%FT%TZ) $A"
for R in $PARITY; do
  echo "$(date -u +%T) room$R $A --max-images 64 (MPS parity)"
  $P scripts/lora256_run.py --model $B --adapter "$A" --trials "room$R" --max-images 64 --device mps \
    --tag "$NAME-room$R-256px-max64-parity-mps"
done
for R in $ROOMS; do
  echo "$(date -u +%T) room$R $A 256-frame walk"
  $P scripts/lora256_run.py --model $B --adapter "$A" --trials "room$R" --device mps
done
echo "RUN_END=$(date -u +%FT%TZ)"
