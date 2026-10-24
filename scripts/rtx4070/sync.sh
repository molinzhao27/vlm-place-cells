#!/usr/bin/env bash
# Move code, data and results between here and the 4070 host.
#   sync.sh push    code -> host (data/, results/ and venvs are not touched)
#   sync.sh data    data/walks (frames, meta, anchors; 2.4 GB) -> host, for the vggt_* modules
#                   (fetch it here first: scripts/vast1/fetch.sh roomA)
#   sync.sh pull    the host's results/ -> here
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/rtx4070/common.sh"
case "${1:-push}" in
  push) rsync -a --delete -e "$R4_RSYNC_SHELL" \
          --exclude '.venv*' --exclude data --exclude results --exclude logs --exclude __pycache__ --exclude .DS_Store \
          "$ROOT/" "$R4_HOST:$R4_ROOT/" ;;
  data) "${R4_SSH[@]}" mkdir -p "$R4_ROOT/data/walks"
        for f in frames.npy meta.npz anchors.npz; do
          rsync -a --partial --progress -e "$R4_RSYNC_SHELL" "$ROOT/data/walks/$f" "$R4_HOST:$R4_ROOT/data/walks/"
        done ;;
  pull) rsync -a -e "$R4_RSYNC_SHELL" "$R4_HOST:$R4_ROOT/results/" "$ROOT/results/" ;;
  *) echo "usage: $0 push|data|pull" >&2; exit 2 ;;
esac
