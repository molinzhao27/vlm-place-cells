# Shared settings for running MechInterp on a remote GPU host: a Docker container
# reached over SSH. Source this file. Host-specific values go in host.env next to
# this file (gitignored; start from host.env.example) or in the environment.
# MI_HOST=NAME reads host-NAME.env instead (another host, e.g. MI_HOST=vast3).
MI_HOST_ENV="$(dirname "${BASH_SOURCE[0]}")/host${MI_HOST:+-$MI_HOST}.env"
if [[ -f "$MI_HOST_ENV" ]]; then
  source "$MI_HOST_ENV"
elif [[ -n "${MI_HOST:-}" ]]; then
  echo "MI_HOST=$MI_HOST: no $MI_HOST_ENV" >&2; exit 2
fi
MI_VAST_HOST="${MI_VAST_HOST:?set MI_VAST_HOST=user@host in $MI_HOST_ENV}"
# MechInterp has its own container and volume, separate from other projects on the host.
MI_CONTAINER="${MI_CONTAINER:-mechinterp}"
# MI_IMAGE is only a read-only base (Python 3.12 and a CUDA build of torch); packages go
# into a venv on the MechInterp volume, created with the image's MI_BASE_PYTHON.
MI_IMAGE="${MI_IMAGE:-}"
MI_BASE_PYTHON="${MI_BASE_PYTHON:-python3}"
# User that runs jobs inside the container.
MI_USER="${MI_USER:-root}"
MI_VOLUME="${MI_VOLUME:-mechinterp-data}"
MI_REMOTE_ROOT="${MI_REMOTE_ROOT:-/workspace/MechInterp}"
MI_PYTHON="${MI_PYTHON:-/workspace/MechInterp/.venv/bin/python}"
# Packages for container.sh env: requirements.txt on top of the image's torch, or
# requirements-lock.txt (vast-1's exact versions, torch included) on an image without torch.
MI_REQUIREMENTS="${MI_REQUIREMENTS:-requirements.txt}"
# docker run settings for container.sh start. The GPU flags depend on the host's runtime:
# vast-1 uses CDI (its "nvidia" runtime is the Vast.ai shim); vast-3 needs the nvidia runtime
# (host.env.example). Memory is shared with other projects' containers, so this one stays small.
if [[ -z "${MI_GPU_ARGS+set}" ]]; then
  MI_GPU_ARGS=(--device nvidia.com/gpu=all)
fi
MI_HOSTNAME="${MI_HOSTNAME:-vast1-mechinterp}"
MI_MEMORY="${MI_MEMORY:-48g}"
MI_SHM="${MI_SHM:-16g}"
# Triton (used by flash-linear-attention) needs a C compiler; set MI_CC if the
# image's is not on PATH. Its kernel cache lives on the MechInterp volume.
MI_EXEC_ENV=()
if [[ -n "${MI_CC:-}" ]]; then
  MI_EXEC_ENV+=(-e "CC=$MI_CC")
fi
MI_EXEC_ENV+=(-e TRITON_CACHE_DIR=/workspace/.cache/triton)
# Host-wide lock shared with the other projects on this GPU.
MI_GPU_LOCK="${MI_GPU_LOCK:-/tmp/gpu-0.lock}"
# Optional host script that waits for a project's turn in a shared GPU order, then execs its
# remaining arguments (gpu-run.sh --turn PROJECT).
MI_GPU_TURN="${MI_GPU_TURN:-}"
MI_SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 "$MI_VAST_HOST")
# rsync remote shell: "<container>:<path>" runs rsync inside the container.
MI_RSYNC_SHELL="ssh -o BatchMode=yes -o ServerAliveInterval=30 $MI_VAST_HOST docker exec -i"
