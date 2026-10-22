#!/usr/bin/env bash
# Copy data from the container to this machine, for jobs run on other machines (docs/gpu_host.md).
# Only the files those jobs read are copied; existing copies are kept unless the container's differ.
#   fetch.sh roomA     data/walks: frames, probes, meta, anchors (2.9 GB)
#   fetch.sh roomB     data/walks-roomB (2.9 GB) and results/export/dissimilar-roomB/trials.json
#   fetch.sh adapter   data/ft/qwen8b-proc, the experiment-9 LoRA adapter (167 MB)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/vast1/common.sh"
(($#)) || { echo "usage: $0 roomA|roomB|adapter ..." >&2; exit 2; }
get() {     # get SRC_DIR DEST_DIR [FILE ...]: whole folder, or only the named files in it
  local src="$1" dest="$2"; shift 2
  mkdir -p "$ROOT/$dest"
  if (($#)); then
    for f in "$@"; do rsync -a --partial --progress -e "$MI_RSYNC_SHELL" "$MI_CONTAINER:$MI_REMOTE_ROOT/$src/$f" "$ROOT/$dest/"; done
  else
    rsync -a --partial --progress -e "$MI_RSYNC_SHELL" "$MI_CONTAINER:$MI_REMOTE_ROOT/$src/" "$ROOT/$dest/"
  fi
}
for what in "$@"; do
  case "$what" in
    roomA) get data/walks data/walks frames.npy probes.npy meta.npz anchors.npz ;;
    roomB) get data/walks-roomB data/walks-roomB frames.npy probes.npy meta.npz anchors.npz
           get results/export/dissimilar-roomB results/export/dissimilar-roomB trials.json ;;
    adapter) get data/ft/qwen8b-proc data/ft/qwen8b-proc ;;
    *) echo "unknown: $what (roomA|roomB|adapter)" >&2; exit 2 ;;
  esac
done
