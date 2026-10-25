#!/usr/bin/env bash
# LEGACY (2026-10-25): GPU jobs on the 4070 now go through gq (docs/gpu_host.md), e.g.
#   gq submit --on pc --project mechinterp --name NAME --gpu --est 2h --env VGGT_DIR=<R4_VGGT> -- .venv/bin/python -m ...
# Kept so older run scripts' usage lines still make sense; don't start new jobs with it.
#
# Run a command on the 4070 host from the project root there, holding R4_GPU_LOCK for the whole
# command; it refuses to start if anything else is computing on the GPU.
#   scripts/rtx4070/run.sh py -m spatialcells.vggt_patch --plot --out results/vggt_patch_probes_4070.json
#   scripts/rtx4070/run.sh --detach logs/vggt_probes.log bash scripts/rtx4070/run_vggt_probes.sh
# A leading "py" is replaced by the venv's python. The detached log path is relative to R4_ROOT
# (or absolute). Jobs see VGGT_DIR=R4_VGGT.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/rtx4070/common.sh"
log=""
if [[ "${1:-}" == --detach ]]; then
  log="${2:?--detach needs a log path}"; shift 2
fi
(($#)) || { echo "usage: $0 [--detach LOG] COMMAND [ARGS...]" >&2; exit 2; }
[[ "$1" == py ]] && set -- .venv/bin/python "${@:2}"
guard='active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit
if [[ -n "$active" ]]; then
  echo "GPU busy with another process; refusing to overlap: $active" >&2
  exit 75
fi
exec "$@"'
printf -v command '%q ' env VGGT_DIR="$R4_VGGT" flock --exclusive "$R4_GPU_LOCK" bash -c "$guard" gpu-slot "$@"
printf -v root '%q' "$R4_ROOT"
if [[ -n "$log" ]]; then
  printf -v log_q '%q' "$log"
  command="cd $root && mkdir -p \"\$(dirname -- $log_q)\" && { nohup $command >$log_q 2>&1 </dev/null & printf 'PID %s\\n' \"\$!\"; }"
else
  command="cd $root && $command"
fi
exec "${R4_SSH[@]}" "$command"
