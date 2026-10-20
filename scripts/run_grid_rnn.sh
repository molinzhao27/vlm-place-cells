#!/usr/bin/env bash
# Experiment 7: grid-cell positive control. Sorscher-style path-integration RNN trained on foraging
# walks with Sorscher-sized steps (0.093 m = their 2 cm in a 2.2 m box), fp32, weight decay after a
# 5k-step warm-up ("s093") or from the start ("s093wd0"); each scored along data/walks (3 sub-steps
# per walk step) and on 1,000 dense walks; then Sorscher's own trajectory generator scaled to the
# room, weight decay from the start ("sorscher"). The scorer calibration (--synthetic) runs on CPU.
#   scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-grid-rnn2.log bash scripts/run_grid_rnn.sh
PY=.venv/bin/python
echo "RUN_START=$(date -u +%FT%TZ)"
$PY -m spatialcells.grid_rnn --name s093 --step 0.093 --sub 3 --train --score --dense 1000 > data/logs/grid_rnn_s093.log 2>&1 || echo "s093 FAILED"
$PY -m spatialcells.grid_rnn --name s093wd0 --step 0.093 --sub 3 --wd-start 0 --train --score --dense 1000 > data/logs/grid_rnn_s093wd0.log 2>&1 || echo "s093wd0 FAILED"
$PY -m spatialcells.grid_rnn --name sorscher --traj sorscher --sub 3 --wd-start 0 --train --score > data/logs/grid_rnn_sorscher.log 2>&1 || echo "sorscher FAILED"
echo "RUN_END=$(date -u +%FT%TZ)"
