#!/usr/bin/env bash
# Manage the MechInterp container on a GPU host (vast-1; MI_HOST=vast3 for vast-3) from this machine.
#   container.sh image    build MI_IMAGE on the host from Dockerfile (for hosts without a torch image)
#   container.sh start    create the volume and start the container
#   container.sh env      create/update the venv on the volume from MI_REQUIREMENTS
#   container.sh stop     stop the container (the volume is kept)
#   container.sh status   show container, volume and GPU state
#   container.sh shell    interactive shell in the container
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/vast1/common.sh"

start() {
  : "${MI_IMAGE:?set MI_IMAGE in $MI_HOST_ENV}"
  "${MI_SSH[@]}" bash -s -- "$MI_CONTAINER" "$MI_IMAGE" "$MI_VOLUME" "$MI_REMOTE_ROOT" "$MI_USER" \
    "$MI_HOSTNAME" "$MI_MEMORY" "$MI_SHM" "${MI_GPU_ARGS[@]}" <<'REMOTE'
set -euo pipefail
container="$1" image="$2" volume="$3" project="$4" user="$5" hostname="$6" memory="$7" shm="$8"
shift 8     # the rest: the host's GPU flags (MI_GPU_ARGS)
docker volume inspect "$volume" >/dev/null 2>&1 || docker volume create "$volume" >/dev/null
if docker container inspect "$container" >/dev/null 2>&1; then
  docker start "$container" >/dev/null
  echo "started existing $container"
else
  docker run -d --name "$container" --hostname "$hostname" --init --restart unless-stopped \
    "$@" \
    --memory "$memory" --memory-swap "$memory" \
    --shm-size "$shm" --ulimit memlock=-1 \
    -e HF_HOME=/workspace/.cache/huggingface -e TRITON_CACHE_DIR=/workspace/.cache/triton \
    -v "$volume":/workspace \
    "$image" sleep infinity >/dev/null
  echo "created $container"
fi
docker exec -u 0 "$container" bash -c "mkdir -p '$project' /workspace/.cache && chown -R $user:$user /workspace"
REMOTE
}

env() {
  local req; req="$(basename "$MI_REQUIREMENTS")"
  rsync -a -e "$MI_RSYNC_SHELL" "$ROOT/$MI_REQUIREMENTS" "$MI_CONTAINER:$MI_REMOTE_ROOT/$req"
  "${MI_SSH[@]}" docker exec -i -u "$MI_USER" -w "$MI_REMOTE_ROOT" "$MI_CONTAINER" bash -s -- "$MI_BASE_PYTHON" "$req" <<'REMOTE'
set -euo pipefail
# --system-site-packages keeps the image's CUDA build of torch (and numpy,
# scipy, scikit-image, scikit-learn); everything else lands in .venv. On an image
# without torch (Dockerfile), requirements-lock.txt puts torch in .venv too.
[[ -x .venv/bin/python ]] || "$1" -m venv --system-site-packages .venv
.venv/bin/python -m pip install --quiet --upgrade -r "$2"
.venv/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
REMOTE
}

image() {
  : "${MI_IMAGE:?set MI_IMAGE in $MI_HOST_ENV}"
  "${MI_SSH[@]}" docker build -q -t "$MI_IMAGE" - < "$ROOT/scripts/vast1/Dockerfile"
}

case "${1:-status}" in
  image) image ;;
  start) start ;;
  env) env ;;
  stop) "${MI_SSH[@]}" docker stop "$MI_CONTAINER" ;;
  status)
    "${MI_SSH[@]}" "docker ps -a --filter name=^/$MI_CONTAINER\$ --format '{{.Names}}\t{{.Image}}\t{{.Status}}';
      docker volume inspect -f '{{.Name}} {{.Mountpoint}}' '$MI_VOLUME' 2>/dev/null; df -h /var/lib/docker | tail -1;
      nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader" ;;
  shell) ssh -t "$MI_VAST_HOST" docker exec -it -u "$MI_USER" -w "$MI_REMOTE_ROOT" "$MI_CONTAINER" bash ;;
  *) echo "usage: $0 image|start|env|stop|status|shell" >&2; exit 2 ;;
esac
