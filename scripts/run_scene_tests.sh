#!/usr/bin/env bash
# Seeds 1-2 and experiment 9c on CUDA (vast-3): one LoRA adapter in the scenes rendered there, each with
# the 64-frame walk, no walk and the shuffled 64-frame walk. The Mac runs the other scenes
# (scripts/mac/run_heldout.sh). Each run skips trials already answered, so it can be restarted.
#   gq submit --on vast3 --project mechinterp --name e4_s3_scenes --gpu --est 1h45m \
#     --env A=data/ft/qwen8b-proc-e4-s3 -- /bin/bash scripts/run_scene_tests.sh
set -u
P=.venv/bin/python
B=Qwen/Qwen3-VL-8B-Instruct
A=${A:?set A to the adapter folder}
ROOMS=${ROOMS:-"HSSD1 HSSD4 HSSD5 HSSD6 HSSD7 HSSD8 P5000 P5001 P5002 P5003 P5004 P5006 P5007 P5008 P5009"}
echo "RUN_START=$(date -u +%FT%TZ) $A"
for R in $ROOMS; do
  for C in "--max-images 64" "--no-walk" "--max-images 64 --shuffle-walk"; do
    echo "$(date -u +%T) room$R $A $C"
    # shellcheck disable=SC2086
    $P -m spatialcells.two_choice --model $B --adapter "$A" --trials "room$R" $C || exit 1
  done
done
echo "RUN_END=$(date -u +%FT%TZ)"
