#!/usr/bin/env bash
# Experiment 25 (docs/experiments.md) on vast-1's CPU, from /workspace/MechInterp/exp25 (own copies of the code,
# data/ linked to the project's). Submitted only after the controls job had passed the registered validation rule.
#   PART=assay: the population assay on the model conditions (layers already in the results file are skipped)
#   PART=gain:  experiment 21's gain index (P4 add-on) on the summary_quadrant.json conditions that experiment 21's
#               amendment 1 does not cover, one tag per call so each tag is saved as it finishes, most relevant first
#   gq submit --on vast1 --project mechinterp --name e25_<part> --cpu --est <h> --cwd /workspace/MechInterp/exp25 \
#     --env PART=<part> --env POP_THREADS=8 --env OMP_NUM_THREADS=8 --env MKL_NUM_THREADS=8 \
#     -- taskset -c 8-15,24-31 /bin/bash scripts/exp25_vast1_conditions.sh
# (The first run, vast1:60, ran both parts in one job; it was stopped after the probes so that the gain index, which
# saves only at the end of a call, could run per tag beside the rest of the assay.)
set -uo pipefail
P=/workspace/MechInterp/.venv/bin/python
PART=${PART:-assay}
echo "RUN_START=$(date -u +%FT%TZ) part=$PART"
md5sum scripts/*.py spatialcells/*.py data/walks/meta.npz data/walks/probes_meta.npz
fail=0
run() { $P -u scripts/population_assay.py real --group conditions "$@" || { echo "FAILED: $*"; fail=1; }; }
if [[ $PART == assay ]]; then
  # 1. the paper's main conditions
  run --tags qwen8b-ft-walk qwen8b-ft-single qwen8b-ft-single+ema8 qwen8b-walk qwen8b-single+ema8 vggt-single vggt-walk \
    qwen-walk qwen-shuffled qwen-single+ema8 qwen-single-random pixels+ema8
  # 2. teleport probes (secondary)
  run --probe-tags qwen8b-ft-probes-walk qwen8b-ft-probes-shuffled qwen8b-ft-probes-only qwen8b-probes-walk qwen8b-probes-only
  # 3. the rest of summary_quadrant.json
  run --tags dinov2 dinov2-random sn8b-walk sn8b-single sn8b-single+ema8 qwen38-walk qwen38-single qwen38-single+ema8 \
    qwen32b-walk qwen32b-single qwen32b-single+ema8 qwen-single+ema2 qwen-single+ema4 qwen-single+ema16 qwen-single+ema32 \
    pixels+ema2 pixels+ema4 pixels+ema16 pixels+ema32
fi
if [[ $PART == gain ]]; then
  # into its own file, which starts as a copy of the Mac's results/heading_stat_validation.json (the synthetic
  # reference that --skip-synthetic reads)
  OUT=results/heading_stat_validation_addon.json
  [[ -f $OUT ]] || cp results/heading_stat_validation.json $OUT
  for spec in vggt-alone vggt-single:L2,L5,L9,L12,L15,L23 vggt-walk:L2,L5,L9,L12,L15,L23 qwen-single qwen-single+ema8 \
      qwen8b-single+ema8 qwen-shuffled qwen-single-random dinov2 dinov2-random sn8b-walk sn8b-single sn8b-single+ema8 \
      pixels+ema2 pixels+ema4 pixels+ema16 pixels+ema32 qwen-single+ema2 qwen-single+ema4 qwen-single+ema16 \
      qwen-single+ema32 qwen38-walk qwen38-single qwen38-single+ema8 qwen32b-walk qwen32b-single qwen32b-single+ema8; do
    echo "gain $spec $(date -u +%T)"
    $P -u scripts/heading_stat_validation.py --skip-synthetic --out $OUT --real $spec || { echo "GAIN FAILED $spec"; fail=1; }
  done
fi
echo "RUN_END=$(date -u +%FT%TZ) failures=$fail"
exit $fail
