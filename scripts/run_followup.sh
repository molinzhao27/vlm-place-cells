#!/usr/bin/env bash
# 1) VGGT patching with pose-PCA subspaces at blocks 15, 19, 21 (~35 min).
# 2) Qwen3.8 and Qwen3.6 thinking on the lure-free trials with the walk (cached walk). The first
#    attempt (run_two_choice_think.sh, batch 12) ran out of memory in Qwen3.8's second batch, so
#    smaller batches here; behave_reason resumes from the 12 trials already written.
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-followup.log bash scripts/run_followup.sh
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
echo "RUN_START=$(date -u +%FT%TZ)"
$P -m spatialcells.vggt_patch --blocks 15 19 21 --out results/vggt_patch_subspaces.json || echo "FAILED patch subspaces"
$P -m spatialcells.behave_reason --model qwen38 --trials dissimilar --effort medium --budget 4096 --batch 6 || echo "FAILED qwen38 walk think"
$P -m spatialcells.two_choice --score > /dev/null || true
$P -m spatialcells.behave_reason --model qwen36 --trials dissimilar --budget 8192 --batch 4 || echo "FAILED qwen36 walk think"
$P -m spatialcells.two_choice --score > /dev/null || true
echo "RUN_END=$(date -u +%FT%TZ)"
