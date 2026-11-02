#!/usr/bin/env bash
# Experiment 34 (pre-registered in docs/experiments.md) on CUDA (vast-3): the 64-frame walk cut into experiment 31's
# blocks of 8 shown frames and reversed at one level only, for one four-epoch adapter, on all 192 trials of each
# held-out room. Order:
#  1. parity: P5000's window-shuffled and reversed walks again (2 x 192 trials, into results/two_choice/exp34/); the job
#     stops unless each gives at least 98% identical answers to experiment 31's stored ones
#     (scripts/exp34_analysis.py --parity);
#  2. in each held-out room, the blocks last to first, each in time order (--reverse-blocks 8, tag -revblocks8), and
#     the blocks first to last, each in reverse time order (--reverse-within-blocks 8, tag -revwithin8), labelled
#     1, 2, ... as shown, as experiment 31's reversed and window-shuffled walks were;
#  3. every reply file of this job checked with scripts/check_replies.py (the job fails if any is damaged).
# Each step skips trials already answered, so the job can be restarted.
#   gq submit --on vast3 --project mechinterp --name e34_s5 --gpu --est 55m \
#     --env A=data/ft/qwen8b-proc-e4-s5 -- /bin/bash scripts/run_exp34.sh
set -euo pipefail
P=.venv/bin/python
B=Qwen/Qwen3-VL-8B-Instruct
A=${A:?set A to the adapter folder, e.g. data/ft/qwen8b-proc-e4-s5}
ROOMS=${ROOMS:-"P5000 P5001 P5002 P5003 P5004 P5005 P5006 P5007 P5008 P5009"}
NAME=Qwen3-VL-8B-Instruct+$(basename "$A")
quiet() { grep --line-buffered -vE "processor_kwargs|Redirects|Loading weights" || true; }
tc() { echo "$(date -u +%T) two_choice $*"; $P -m spatialcells.two_choice --model $B --adapter "$A" --max-images 64 "$@" 2>&1 | quiet; }
echo "RUN_START=$(date -u +%FT%TZ) $A"
md5sum spatialcells/two_choice.py scripts/exp34_analysis.py scripts/exp31_analysis.py scripts/check_replies.py \
  "$A"/adapter_model.safetensors
mkdir -p results/two_choice/exp34
tc --trials roomP5000 --window-shuffle 8 --tag "exp34/$NAME-roomP5000-256px-max64-window8-parity"
tc --trials roomP5000 --reverse-walk --tag "exp34/$NAME-roomP5000-256px-max64-reversed-parity"
$P scripts/exp34_analysis.py --parity "$A"
for R in $ROOMS; do
  for C in "--reverse-blocks 8" "--reverse-within-blocks 8"; do
    # shellcheck disable=SC2086
    tc --trials "room$R" $C
  done
done
$P scripts/check_replies.py "results/two_choice/exp34/$NAME-*.jsonl" \
  "results/two_choice/$NAME-roomP500[0-9]-256px-max64-revblocks8.jsonl" \
  "results/two_choice/$NAME-roomP500[0-9]-256px-max64-revwithin8.jsonl"
echo "RUN_END=$(date -u +%FT%TZ)"
