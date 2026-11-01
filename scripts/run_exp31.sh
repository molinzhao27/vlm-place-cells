#!/usr/bin/env bash
# Experiment 31 (pre-registered in docs/experiments.md) on CUDA (vast-3): experiment 13a's walk-order controls, a
# shuffled walk relabeled 1, 2, ... as shown, and experiment 15's attention cuts, for one four-epoch adapter, with
# two_choice.py and frame_cut.py unchanged. Order:
#  1. parity: P5000's ordered walk again (192 trials, results/two_choice/exp31/); the job stops below 98% identical
#     answers to experiment 9c's stored ones (scripts/exp31_analysis.py --parity);
#  2. P5005's ordered and shuffled walks on CUDA (experiment 9c answered P5005 on the Mac), into results/two_choice/exp31/;
#  3. 31a/31c: in each held-out room, all 192 trials with the 64-frame walk reversed, window-shuffled (blocks of 8),
#     renumbered, and shuffled with --relabel (same shuffle order as experiment 9c's --shuffle-walk runs);
#  4. 31b: frame_cut clean on the 1,299 novel trials, experiment 15's check (stops below 99%), the near, look-alike,
#     random and far cuts in all layers, then near and look-alike cut in layers 1-18 or 19-36 on P5000-P5002.
# Each step skips trials already answered, so the job can be restarted. Needs data/frame_cut/dino_lookalike.json.
#   gq submit --on vast3 --project mechinterp --name e31_s5 --gpu --est 3h30m --priority 150 --after 335 \
#     --env A=data/ft/qwen8b-proc-e4-s5 -- /bin/bash scripts/run_exp31.sh
set -euo pipefail
P=.venv/bin/python
B=Qwen/Qwen3-VL-8B-Instruct
A=${A:?set A to the adapter folder, e.g. data/ft/qwen8b-proc-e4-s5}
ROOMS=${ROOMS:-"P5000 P5001 P5002 P5003 P5004 P5005 P5006 P5007 P5008 P5009"}
NAME=Qwen3-VL-8B-Instruct+$(basename "$A")
quiet() { grep --line-buffered -vE "processor_kwargs|Redirects|Loading weights" || true; }
tc() { echo "$(date -u +%T) two_choice $*"; $P -m spatialcells.two_choice --model $B --adapter "$A" --max-images 64 "$@" 2>&1 | quiet; }
fc() { echo "$(date -u +%T) frame_cut $*"; $P -m spatialcells.frame_cut --adapter "$A" --device cuda "$@" 2>&1 | quiet; }
echo "RUN_START=$(date -u +%FT%TZ) $A"
md5sum spatialcells/two_choice.py spatialcells/frame_cut.py spatialcells/task_record.py scripts/exp31_analysis.py \
  data/frame_cut/dino_lookalike.json "$A"/adapter_model.safetensors
mkdir -p results/two_choice/exp31
tc --trials roomP5000 --tag "exp31/$NAME-roomP5000-256px-max64-parity"
$P scripts/exp31_analysis.py --parity "$A"
tc --trials roomP5005 --tag "exp31/$NAME-roomP5005-256px-max64"
tc --trials roomP5005 --shuffle-walk --tag "exp31/$NAME-roomP5005-256px-max64-shuffled"
for R in $ROOMS; do
  for C in "--reverse-walk" "--window-shuffle 8" "--renumber" "--shuffle-walk --relabel"; do
    # shellcheck disable=SC2086
    tc --trials "room$R" $C
  done
done
fc --condition clean
fc --check
for C in near lookalike random far; do
  fc --condition $C
done
for C in near lookalike; do
  for L in 1-18 19-36; do
    fc --condition $C --layers $L --rooms P5000 P5001 P5002
  done
done
echo "RUN_END=$(date -u +%FT%TZ)"
