#!/usr/bin/env bash
# Experiment 33 (docs/experiments.md) on vast-3's CPU, after scripts/exp33_record.sh: experiment 25's population assay on
# the single-query recordings (data/acts_sq/), on experiment 25's own teleport-view and walk-frame recordings of the same
# models (data/acts/), and the summary with the registered rules.
#   gq submit --on vast3 --project mechinterp --name e33_assay --cpu --after <record id> --est 2h30m \
#     --env POP_THREADS=6 --env OMP_NUM_THREADS=6 --env MKL_NUM_THREADS=6 -- /bin/bash scripts/exp33_analysis.sh
# Then on the Mac: fetch results/exp33_single_query.json (check its md5); optionally data/exp33/errors/ and
# data/acts_sq/*/{done_*.jsonl,checks.jsonl,meta.json} (small), to rerun `summarize` there.
set -uo pipefail
P=.venv/bin/python
echo "RUN_START=$(date -u +%FT%TZ)"
mkdir -p results/exp33
md5sum scripts/exp33_analysis.py scripts/population_assay.py scripts/heading_stat_validation.py \
  spatialcells/single_query.py spatialcells/analyze.py results/exp33_calibration.json results/population_assay.json \
  results/export/single_query/trials.json data/walks/meta.npz data/walks/probes_meta.npz | tee results/exp33/analysis_inputs.md5
$P -u scripts/exp33_analysis.py assay && $P -u scripts/exp33_analysis.py summarize
rc=$?
md5sum results/exp33_single_query.json
echo "RUN_END=$(date -u +%FT%TZ) rc=$rc"
exit $rc
