#!/usr/bin/env bash
# Experiment 16 (docs/experiments.md): the fixed draw of 8 HSSD houses, each checked, rendered as in
# experiment 12 (habitat-sim 0.3.3, 32 walks of 256 steps, 128 probes per walk, 0.15 m clearance) into
# /mnt/bigd/mechinterp/walks-roomHSSD<k> (linked from data/), with its lure-free trials built; see
# spatialcells/hssd_select.py. Needs HSSD (get_hssd.sh). Through the lock, from here:
#   scripts/rtx4070/run.sh --detach logs/render_hssd.log bash scripts/rtx4070/render_hssd.sh
set -euo pipefail
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
echo "RUN_START=$(date -u +%FT%TZ)"
.venv/bin/python -m spatialcells.hssd_select --houses "${HOUSES:-8}"
echo "RUN_END=$(date -u +%FT%TZ)"
