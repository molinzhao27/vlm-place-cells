# Shared settings for running MechInterp directly (no container) on an Ubuntu machine with a
# 12 GB RTX 4070, reached over SSH. Source this file. Host-specific values go in host.env next to
# this file (gitignored; start from host.env.example) or in the environment.
if [[ -f "$(dirname "${BASH_SOURCE[0]}")/host.env" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/host.env"
fi
R4_HOST="${R4_HOST:?set R4_HOST=user@host in scripts/rtx4070/host.env}"
R4_ROOT="${R4_ROOT:?set R4_ROOT (absolute project folder on the host) in scripts/rtx4070/host.env}"
R4_VGGT="${R4_VGGT:?set R4_VGGT (absolute VGGT clone path on the host) in scripts/rtx4070/host.env}"
R4_PYTHON="${R4_PYTHON:-python3}"
# Same torch as the GPU host (2.14.0); the CUDA build must suit the host's driver.
R4_TORCH_INDEX="${R4_TORCH_INDEX:-https://download.pytorch.org/whl/cu132}"
R4_GPU_LOCK="${R4_GPU_LOCK:-/tmp/mechinterp-gpu.lock}"
# The VGGT commit the GPU host's results were computed with.
R4_VGGT_COMMIT=a288dd0f14786c93483e45524328726ab7b1b4ce
R4_SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 "$R4_HOST")
R4_RSYNC_SHELL="ssh -o BatchMode=yes -o ServerAliveInterval=30"
