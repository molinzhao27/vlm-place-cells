#!/usr/bin/env bash
# Experiment 30 (pre-registered in docs/experiments.md): the Mac's part before any model run, on its CPU through gq.
# For each condition and room: render the walks and new views and build the trial lists
# (spatialcells/trajectory_shift.py --prepare), copy the payload (the images the novel trials' 64-frame prompts show),
# the frame MD5s and the poses into the vast-3 container, check them there by MD5, and delete the local payload. Then
# the combined novel-trial list (--combine), its copy, and the unpack on vast-3 (a gq CPU job there). Finished rooms
# are skipped, so the job can be restarted.
#   gq submit --on mac --project mechinterp --name e30_prep --cpu --est 2h30m -- /bin/bash scripts/mac/trajectory_shift_prep.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
P=.venv-mac/bin/python
ROOMS=${ROOMS:-"P5000 P5001 P5002 P5003 P5004 P5006 P5007 P5008 P5009"}
CONDS=${CONDS:-"control long-step low-turn patrol"}
WORKERS=${WORKERS:-6}
export MI_HOST=vast3
source scripts/vast1/common.sh
retry() {    # vast-3's sshd sometimes resets new connections for a few minutes
  local n
  for n in 1 2 3 4 5; do "$@" && return 0; echo "attempt $n failed; retrying in 60 s" >&2; sleep 60; done
  return 1
}
push() {     # repo-relative files -> the same paths in the container, then MD5 there against here
  retry rsync -a -R --bwlimit=3000 -e "$MI_RSYNC_SHELL" "$@" "$MI_CONTAINER:$MI_REMOTE_ROOT/"
  local there here
  there=$(retry "${MI_SSH[@]}" "docker exec $MI_CONTAINER sh -c 'cd $MI_REMOTE_ROOT && md5sum $*'")
  here=$(md5 -r "$@" | awk '{print $1"  "$2}')
  if [[ "$there" != "$here" ]]; then
    echo "MD5 MISMATCH on vast-3"; diff <(echo "$there") <(echo "$here"); return 1
  fi
}
echo "RUN_START=$(date -u +%FT%TZ)"
for C in $CONDS; do
  for R in $ROOMS; do
    S=$R-$C
    if [[ -f data/e30/$S/pushed ]]; then echo "$S: already copied"; continue; fi
    nice -n 10 $P -m spatialcells.trajectory_shift --prepare --rooms "$R" --conditions "$C" --workers "$WORKERS"
    push "data/e30/payload/$S.npz" "data/e30/$S/md5.json" "data/e30/$S/poses.npz"
    rm "data/e30/payload/$S.npz"
    touch "data/e30/$S/pushed"
    echo "$(date -u +%T) $S: copied to vast-3, MD5 match"
  done
done
$P -m spatialcells.trajectory_shift --combine
push data/e30/payload/index.json results/export/dissimilar-roomE30/trials.json spatialcells/trajectory_shift.py \
  scripts/run_trajectory_shift.sh
echo "$(date -u +%T) combined trial list and code copied to vast-3, MD5 match"
"$(command -v gq || echo "$HOME/.local/bin/gq")" submit --on vast3 --project mechinterp --name e30_unpack --cpu --est 30m --priority 150 \
  -- .venv/bin/python -m spatialcells.trajectory_shift --unpack
echo "RUN_END=$(date -u +%FT%TZ)"
