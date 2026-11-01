#!/usr/bin/env bash
# Copy rendered photoreal scenes from here to the mechinterp container (experiment 19): each scene's walks
# (data/walks-room<S>) and trial export (results/export/dissimilar-room<S>), compressed on the wire and capped at
# 3 MB/s (the Mac's uplink is shared), then the MD5 of every file compared between the two ends. One line per
# scene; exits 1 on any mismatch. Host settings as for sync.sh (MI_HOST=vast3).
#   MI_HOST=vast3 scripts/vast1/push_scenes.sh HSSD11 HSSD12
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/vast1/common.sh"
cd "$ROOT"
retry() {    # vast-3's sshd sometimes resets new connections for a few minutes
  local n
  for n in 1 2 3 4 5; do "$@" && return 0; echo "attempt $n failed; retrying in 60 s" >&2; sleep 60; done
  return 1
}
for S in "$@"; do
  retry rsync -a -R -z --bwlimit="${BWLIMIT:-3000}" -e "$MI_RSYNC_SHELL" "data/walks-room$S" "results/export/dissimilar-room$S" \
    "$MI_CONTAINER:$MI_REMOTE_ROOT/"
  there=$(retry "${MI_SSH[@]}" "docker exec $MI_CONTAINER sh -c 'cd $MI_REMOTE_ROOT && find data/walks-room$S \
    results/export/dissimilar-room$S -type f | LC_ALL=C sort | xargs md5sum'")    # one string: ssh re-splits its arguments
  here=$(find data/walks-room$S results/export/dissimilar-room$S -type f | LC_ALL=C sort | xargs md5 -r | awk '{print $1"  "$2}')
  if [[ "$there" == "$here" ]]; then
    echo "room$S: $(wc -l <<< "$here" | tr -d ' ') files, MD5 match on $MI_VAST_HOST"
  else
    echo "room$S: MD5 MISMATCH"; diff <(echo "$there") <(echo "$here") | head; exit 1
  fi
done
