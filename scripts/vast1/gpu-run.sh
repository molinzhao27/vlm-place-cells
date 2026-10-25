#!/usr/bin/env bash
# LEGACY (2026-10-25): GPU jobs now go through gq (docs/gpu_host.md), e.g.
#   gq submit --on vast1 --project mechinterp --name NAME --gpu --est 2h -- .venv/bin/python -m ...
# Kept so older run scripts' usage lines still make sense; don't start new jobs with it.
#
# Hold the shared host GPU lock for an entire foreground container command.
#   gpu-run.sh py -m spatialcells.extract --model dinov2
#   gpu-run.sh --detach $HOST_LOGS/mechinterp-x.log py -m ...   (HOST_LOGS: a log dir on the host)
#   gpu-run.sh --after-pattern 'other-queu[e].sh' --detach ...   (wait, before taking the lock, until
#                                                     no host process matches pgrep -f PAT)
#   gpu-run.sh --turn mechinterp --detach ...       (wait for the project's turn in the host's shared
#                                                     GPU order: MI_GPU_TURN PROJECT, then the lock)
# The optional detached log is a HOST path. Never use docker exec -d here:
# that would release the lock while the process was still running.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/vast1/common.sh"
host_log="" after="" turn=""
while [[ "${1:-}" == --detach || "${1:-}" == --after-pattern || "${1:-}" == --turn ]]; do
  case "$1" in
    --detach) host_log="${2:?--detach needs an absolute host log path}"
              [[ "$host_log" == /* ]] || { echo "Use an absolute HOST log path" >&2; exit 2; } ;;
    --after-pattern) after="${2:?--after-pattern needs a pgrep -f pattern}" ;;
    --turn) turn="${2:?--turn needs a project name}"
            [[ -n "$MI_GPU_TURN" ]] || { echo "--turn needs MI_GPU_TURN in host.env" >&2; exit 2; } ;;
  esac
  shift 2
done
(($#)) || { echo "usage: $0 [--turn PROJECT] [--after-pattern PAT] [--detach HOST_LOG] COMMAND [ARGS...]" >&2; exit 2; }
[[ "$1" == py ]] && set -- "$MI_PYTHON" "${@:2}"
guard='active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit
if [[ -n "$active" ]]; then
  echo "GPU occupied by a process outside the shared lock; refusing overlap: $active" >&2
  exit 75
fi
exec "$@"'
printf -v command '%q ' flock --exclusive "$MI_GPU_LOCK" \
  bash -c "$guard" gpu-slot docker exec -i -u "$MI_USER" "${MI_EXEC_ENV[@]}" -w "$MI_REMOTE_ROOT" "$MI_CONTAINER" "$@"
if [[ -n "$after" ]]; then       # a bracket in PAT (queu[e]) keeps pgrep from matching this loop itself
  printf -v command '%q ' bash -c "while pgrep -f $(printf '%q' "$after") >/dev/null; do sleep 300; done; exec $command"
fi
if [[ -n "$turn" ]]; then        # the turn script waits, then execs the rest of its arguments
  printf -v command '%q %q %s' "$MI_GPU_TURN" "$turn" "$command"
fi
if [[ -n "$host_log" ]]; then
  printf -v log_parent '%q' "$(dirname "$host_log")"
  printf -v log_path '%q' "$host_log"
  command="mkdir -p $log_parent && { nohup $command >$log_path 2>&1 </dev/null & printf 'Host lock-holder PID: %s\\n' \"\$!\"; }"
fi
exec "${MI_SSH[@]}" "$command"
