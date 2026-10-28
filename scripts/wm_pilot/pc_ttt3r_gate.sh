#!/bin/bash
# Follow-up pilot WP0 for TTT3R on the RTX 4070 PC (CUDA; the CUT3R environment and checkpoint on /mnt/bigd).
set -e
B=/mnt/bigd/wm_models
A="--rule ${RULE:-ttt3r} --repo $B/${REPO:-TTT3R} --ckpt $B/weights/cut3r/cut3r_512_dpt_4_64.pth --device cuda --batch ${BATCH:-2}"
$B/envs/cut3r/bin/python -m spatialcells.wm_cut3r gate $A
$B/envs/cut3r/bin/python -m spatialcells.wm_cut3r gate $A --data data/walks-roomReplicaApt0 --scene ReplicaApt0
