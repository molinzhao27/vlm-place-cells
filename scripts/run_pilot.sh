#!/usr/bin/env bash
# Full pilot inside the container (run through scripts/vast1/gpu-run.sh so the
# shared GPU lock is held throughout):
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-pilot.log bash scripts/run_pilot.sh
set -euo pipefail
P=.venv/bin/python
echo "RUN_START=$(date -u +%FT%TZ)"
rm -rf data/acts/*-test results-test
$P -m spatialcells.extract --model dinov2
$P -m spatialcells.extract --model dinov2 --random
for mode in single walk shuffled; do
  $P -m spatialcells.extract --model qwen --mode "$mode"
done
$P -m spatialcells.extract --model qwen --mode single --random
$P -m spatialcells.analyze --tags pixels dinov2 dinov2-random qwen-single qwen-walk qwen-shuffled qwen-single-random
echo "RUN_END=$(date -u +%FT%TZ)"
