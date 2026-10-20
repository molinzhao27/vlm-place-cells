#!/usr/bin/env bash
# Run a CPU command inside the MechInterp container from the project root, e.g.
#   scripts/vast1/run.sh py -m spatialcells.render --out data/room
# A leading "py" is replaced by the venv's python.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/vast1/common.sh"
(($#)) || { echo "usage: $0 COMMAND [ARGS...]" >&2; exit 2; }
[[ "$1" == py ]] && set -- "$MI_PYTHON" "${@:2}"
# ssh joins its arguments into one remote shell line, so quote each one.
printf -v command '%q ' docker exec -i -u "$MI_USER" "${MI_EXEC_ENV[@]}" -w "$MI_REMOTE_ROOT" "$MI_CONTAINER" "$@"
exec "${MI_SSH[@]}" "$command"
