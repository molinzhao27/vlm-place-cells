#!/usr/bin/env bash
# Push code to the container (sync.sh push) or pull results back (sync.sh pull).
# Large arrays stay remote; pull brings back only results/ (figures, tables).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/vast1/common.sh"
case "${1:-push}" in
  push) rsync -a --no-owner --no-group --delete -e "$MI_RSYNC_SHELL" \
          --exclude '.venv*' --exclude data --exclude results --exclude __pycache__ --exclude .DS_Store \
          "$ROOT/" "$MI_CONTAINER:$MI_REMOTE_ROOT/" ;;
  pull) rsync -a -e "$MI_RSYNC_SHELL" "$MI_CONTAINER:$MI_REMOTE_ROOT/results/" "$ROOT/results/" ;;
  *) echo "usage: $0 push|pull" >&2; exit 2 ;;
esac
