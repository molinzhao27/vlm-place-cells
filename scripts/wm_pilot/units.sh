#!/bin/bash
# Follow-up pilot WP1: score units with analyze.py --quadrant (unchanged), on room A's walk frames and on the
# 256-per-walk probe arena, into results/wm_pilot/{units,probes}/<model>/summary_quadrant.json (one folder per
# model so that concurrent jobs never write the same file).
# usage: scripts/wm_pilot/units.sh <model> "<walk tags>" "<probe tags>"
set -e
PY=${PY:-.venv-mac/bin/python}
M=$1
if [ -n "$2" ]; then $PY -m spatialcells.analyze --quadrant --out results/wm_pilot/units/$M --tags $2; fi
if [ -n "$3" ]; then $PY -m spatialcells.analyze --quadrant --data data/walks-p256-arena --out results/wm_pilot/probes/$M --tags $3; fi
