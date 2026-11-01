#!/usr/bin/env bash
# Experiment 25 amendment 1 (docs/experiments.md) on vast-3's CPU: the population assay on the four-epoch runs'
# shuffled-walk frames, recorded by job vast3:305 but left out of exp25_vast3_e4.sh.
# Runs from /workspace/MechInterp/exp25 on vast-3 (own copies of the code, data/ linked to the project's):
#   gq submit --on vast3 --project mechinterp --name e25_shuf --cpu --est 45m --cwd /workspace/MechInterp/exp25 \
#     --env POP_THREADS=6 --env OMP_NUM_THREADS=6 --env MKL_NUM_THREADS=6 -- /bin/bash scripts/exp25_shuffled_e4.sh
# Then on the Mac: fetch exp25/results/population_assay_e4_shuffled.json into results/.
set -uo pipefail
P=/workspace/MechInterp/.venv/bin/python
OUT=results/population_assay_e4_shuffled.json
echo "RUN_START=$(date -u +%FT%TZ)"
md5sum scripts/population_assay.py spatialcells/analyze.py data/walks/meta.npz data/walks/probes_meta.npz
fail=0
for m in qwen8b-ft-e4-s5 qwen8b-ft-e4-s7 qwen8b-ft-e4-s3 qwen8b-ft-e4-s4 qwen8b-ft-e4-s6; do
  if [[ $(ls data/acts/$m-shuffled/L*.npy 2>/dev/null | wc -l) -ne 7 ]]; then echo "MISSING data/acts/$m-shuffled"; fail=1; continue; fi
  echo "$m: $(date -u +%T)"
  $P -u scripts/population_assay.py real --group conditions_e4 --out $OUT --tags $m-shuffled || { echo "FAILED $m"; fail=1; }
done
echo "RUN_END=$(date -u +%FT%TZ) failures=$fail"
exit $fail
