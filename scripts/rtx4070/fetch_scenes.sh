#!/usr/bin/env bash
# Copy rendered photoreal scenes from the 4070 host to here and check them (experiment 19): each scene's walks
# (data/walks-room<S>, a link into /mnt/bigd there) and trial export (results/export/dissimilar-room<S>), with the
# SHA-256 of every file compared between the two machines. Prints one line per scene; exits 1 on any mismatch.
#   scripts/rtx4070/fetch_scenes.sh HSSD9 HSSD10
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/rtx4070/common.sh"
cd "$ROOT"
for S in "$@"; do
  for d in "data/walks-room$S" "results/export/dissimilar-room$S"; do
    mkdir -p "$d"
    rsync -a -e "$R4_RSYNC_SHELL" "$R4_HOST:$R4_ROOT/$d/" "$d/"
  done
  there=$("${R4_SSH[@]}" "cd $R4_ROOT && find -L data/walks-room$S results/export/dissimilar-room$S -type f | LC_ALL=C sort | xargs sha256sum")
  here=$(find data/walks-room$S results/export/dissimilar-room$S -type f | LC_ALL=C sort | xargs shasum -a 256)
  if [[ "$there" == "$here" ]]; then
    echo "room$S: $(wc -l <<< "$here" | tr -d ' ') files, SHA-256 match"
  else
    echo "room$S: SHA-256 MISMATCH"; diff <(echo "$there") <(echo "$here") | head; exit 1
  fi
done
