#!/usr/bin/env bash
# HSSD for experiment 16 (docs/experiments.md) on the 4070 host: hssd/hssd-hab from Hugging Face
# (CC BY-NC 4.0, not gated; download approved by the user 2026-10-24) at a pinned revision, without
# the parts the default scenes never use: ~11 GB in ~57k files, into /mnt/bigd/mechinterp/hssd-hab
# (the system disk is small; HF_HOME goes there too), linked as ~/habitat-data/hssd-hab. The houses
# the draw needs first come first (remote/get_hssd_job.py). Detached, low priority, no GPU; log in
# ~/MechInterp/logs/get_hssd.log. A rerun resumes.
#   scripts/rtx4070/get_hssd.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/rtx4070/common.sh"
REV=${HSSD_REVISION:-4369cb9876214c7fbebcf552eb532380e4d287e4}     # main on 2026-10-24
DEST=${HSSD_DEST:-/mnt/bigd/mechinterp/hssd-hab}
"${R4_SSH[@]}" mkdir -p "$R4_ROOT/scripts/rtx4070/remote"
rsync -a -e "$R4_RSYNC_SHELL" "$ROOT/scripts/rtx4070/remote/get_hssd_job.py" "$R4_HOST:$R4_ROOT/scripts/rtx4070/remote/"
"${R4_SSH[@]}" "mkdir -p ~/MechInterp/logs '$DEST' /mnt/bigd/mechinterp/hf_home; ln -sfn '$DEST' ~/habitat-data/hssd-hab; \
  HF_HOME=/mnt/bigd/mechinterp/hf_home nohup nice -n 10 ~/miniforge3/envs/hab033/bin/python -I \
  '$R4_ROOT/scripts/rtx4070/remote/get_hssd_job.py' '$REV' '$DEST' ${HSSD_PRIORITY:-24} \
  > ~/MechInterp/logs/get_hssd.log 2>&1 < /dev/null &"
echo "started on $R4_HOST; log: ~/MechInterp/logs/get_hssd.log"
