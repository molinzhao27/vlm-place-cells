#!/usr/bin/env bash
# Manage the MechInterp container on vast-1 from this machine.
#   container.sh start    create the volume and start the container
#   container.sh env      create/update the venv on the volume from requirements.txt
#   container.sh stop     stop the container (the volume is kept)
#   container.sh status   show container, volume and GPU state
#   container.sh shell    interactive shell in the container
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/vast1/common.sh"

start() {
  : "${MI_IMAGE:?set MI_IMAGE in scripts/vast1/host.env}"
  "${MI_SSH[@]}" bash -s -- "$MI_CONTAINER" "$MI_IMAGE" "$MI_VOLUME" "$MI_REMOTE_ROOT" "$MI_USER" <<'REMOTE'
set -euo pipefail
container="$1" image="$2" volume="$3" project="$4" user="$5"
docker volume inspect "$volume" >/dev/null 2>&1 || docker volume create "$volume" >/dev/null
if docker container inspect "$container" >/dev/null 2>&1; then
  docker start "$container" >/dev/null
  echo "started existing $container"
else
  # GPU via CDI (the host's "nvidia" runtime is the Vast.ai shim; do not use it).
  # The host's memory is shared with other projects' containers, so this one stays small.
  docker run -d --name "$container" --hostname vast1-mechinterp --init --restart unless-stopped \
    --device nvidia.com/gpu=all \
    --memory 48g --memory-swap 48g \
    --shm-size 16g --ulimit memlock=-1 \
    -e HF_HOME=/workspace/.cache/huggingface \
    -v "$volume":/workspace \
    "$image" sleep infinity >/dev/null
  echo "created $container"
fi
docker exec -u 0 "$container" bash -c "mkdir -p '$project' /workspace/.cache && chown -R $user:$user /workspace"
REMOTE
}

env() {
  rsync -a -e "$MI_RSYNC_SHELL" "$ROOT/requirements.txt" "$MI_CONTAINER:$MI_REMOTE_ROOT/requirements.txt"
  "${MI_SSH[@]}" docker exec -i -u "$MI_USER" -w "$MI_REMOTE_ROOT" "$MI_CONTAINER" bash -s -- "$MI_BASE_PYTHON" <<'REMOTE'
set -euo pipefail
# --system-site-packages keeps the image's CUDA build of torch (and numpy,
# scipy, scikit-image, scikit-learn); everything else lands in .venv.
[[ -x .venv/bin/python ]] || "$1" -m venv --system-site-packages .venv
.venv/bin/python -m pip install --quiet --upgrade -r requirements.txt
.venv/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
REMOTE
}

case "${1:-status}" in
  start) start ;;
  env) env ;;
  stop) "${MI_SSH[@]}" docker stop "$MI_CONTAINER" ;;
  status)
    "${MI_SSH[@]}" "docker ps -a --filter name=^/$MI_CONTAINER\$ --format '{{.Names}}\t{{.Image}}\t{{.Status}}';
      docker volume inspect -f '{{.Name}} {{.Mountpoint}}' '$MI_VOLUME' 2>/dev/null; df -h /var/lib/docker | tail -1;
      nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader" ;;
  shell) ssh -t "$MI_VAST_HOST" docker exec -it -u "$MI_USER" -w "$MI_REMOTE_ROOT" "$MI_CONTAINER" bash ;;
  *) echo "usage: $0 start|env|stop|status|shell" >&2; exit 2 ;;
esac
