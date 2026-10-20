#!/usr/bin/env bash
# Hold the shared host GPU lock for an entire foreground container command.
#   gpu-run.sh py -m spatialcells.extract --model dinov2
#   gpu-run.sh --detach $HOST_LOGS/mechinterp-x.log py -m ...   (HOST_LOGS: a log dir on the host)
#   gpu-run.sh --after-pattern 'other-queu[e].sh' --detach ...   (wait, before taking the lock, until
#                                                     no host process matches pgrep -f PAT)
# The optional detached log is a HOST path. Never use docker exec -d here:
# that would release the lock while the process was still running.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/vast1/common.sh"
host_log="" after=""
while [[ "${1:-}" == --detach || "${1:-}" == --after-pattern ]]; do
  if [[ "$1" == --detach ]]; then
    host_log="${2:?--detach needs an absolute host log path}"
    [[ "$host_log" == /* ]] || { echo "Use an absolute HOST log path" >&2; exit 2; }
  else
    after="${2:?--after-pattern needs a pgrep -f pattern}"
  fi
  shift 2
done
(($#)) || { echo "usage: $0 [--after-pattern PAT] [--detach HOST_LOG] COMMAND [ARGS...]" >&2; exit 2; }
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
if [[ -n "$host_log" ]]; then
  printf -v log_parent '%q' "$(dirname "$host_log")"
  printf -v log_path '%q' "$host_log"
  command="mkdir -p $log_parent && { nohup $command >$log_path 2>&1 </dev/null & printf 'Host lock-holder PID: %s\\n' \"\$!\"; }"
fi
exec "${MI_SSH[@]}" "$command"
