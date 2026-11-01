#!/usr/bin/env bash
# Experiment 19 on CUDA (vast-3; docs/experiments.md): every registered Qwen model in the new photoreal scenes
# ROOMS, each with the 64-frame walk, no walk and the shuffled 64-frame walk (scripts/run_scene_tests.sh per
# model): the base model, run 0, the two-epoch runs 1-2 and the four-epoch runs 3-7. The Mac runs its scenes with
# scripts/mac/run_heldout.sh (ADAPTERS set to the same list). Each run skips trials already answered.
#   gq submit --on vast3 --project mechinterp --name e19_HSSD11 --gpu --est 45m \
#     --env ROOMS=HSSD11 -- /bin/bash scripts/run_exp19_scene.sh
set -u
ROOMS=${ROOMS:?set ROOMS, e.g. HSSD11}
MODELS=${MODELS:-"base data/ft/qwen8b-proc data/ft/qwen8b-proc-s1 data/ft/qwen8b-proc-s2 data/ft/qwen8b-proc-e4-s3 data/ft/qwen8b-proc-e4-s4 data/ft/qwen8b-proc-e4-s5 data/ft/qwen8b-proc-e4-s6 data/ft/qwen8b-proc-e4-s7"}
for M in $MODELS; do
  A=$M ROOMS=$ROOMS bash scripts/run_scene_tests.sh || exit 1
done
