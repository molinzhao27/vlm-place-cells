#!/usr/bin/env bash
# Experiment 25 (docs/experiments.md), part 3 on vast-3's CPU: the population assay on experiment 22's four-epoch
# recordings (job vast3:305 makes them: data/acts/qwen8b-ft-rerun-* and qwen8b-ft-e4-s<k>-*, room A's walks).
# Runs from /workspace/MechInterp/exp25 on vast-3 (own copies of the code, data/ linked to the project's):
#   gq submit --on vast3 --project mechinterp --name e25_e4 --cpu --after 305 --est 7h --cwd /workspace/MechInterp/exp25 \
#     --env POP_THREADS=6 --env OMP_NUM_THREADS=6 --env MKL_NUM_THREADS=6 -- /bin/bash scripts/exp25_vast3_e4.sh
# Then on the Mac: fetch exp25/results/population_assay_e4.json and run
#   .venv-mac/bin/python scripts/population_assay.py merge --from <that file>
set -uo pipefail
P=/workspace/MechInterp/.venv/bin/python
OUT=results/population_assay_e4.json
echo "RUN_START=$(date -u +%FT%TZ)"
md5sum scripts/*.py spatialcells/*.py data/walks/meta.npz data/walks/probes_meta.npz
fail=0
for m in qwen8b-ft-rerun qwen8b-ft-e4-s5 qwen8b-ft-e4-s7 qwen8b-ft-e4-s3 qwen8b-ft-e4-s4 qwen8b-ft-e4-s6; do
  tags=""; ptags=""
  for c in walk single; do
    if [[ $(ls data/acts/$m-$c/L*.npy 2>/dev/null | wc -l) -eq 7 ]]; then tags="$tags $m-$c"; else echo "MISSING data/acts/$m-$c"; fail=1; fi
  done
  [[ " $tags " == *" $m-single "* ]] && tags="$tags $m-single+ema8"
  for c in probes-walk probes-shuffled probes-only; do
    if [[ $(ls data/acts/$m-$c/L*.npy 2>/dev/null | wc -l) -eq 7 ]]; then ptags="$ptags $m-$c"; else echo "MISSING data/acts/$m-$c"; fi
  done
  [[ -n "$tags$ptags" ]] || continue
  echo "$m: $(date -u +%T)"
  $P -u scripts/population_assay.py real --group conditions_e4 --out $OUT --tags $tags --probe-tags $ptags \
    || { echo "FAILED $m"; fail=1; }
done
echo "RUN_END=$(date -u +%FT%TZ) failures=$fail"
exit $fail
