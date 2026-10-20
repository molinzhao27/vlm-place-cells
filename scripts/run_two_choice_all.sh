#!/usr/bin/env bash
# New open-weight models first, then the Qwen token-count / no-walk sweep. Through the GPU lock:
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-two-choice.log bash scripts/run_two_choice_all.sh
set -euo pipefail
bash scripts/run_two_choice_new.sh
bash scripts/run_two_choice_qwen.sh
