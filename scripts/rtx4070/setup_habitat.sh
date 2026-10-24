#!/usr/bin/env bash
# Photoreal rooms (docs/photoreal_plan.md): a per-user Miniforge in ~/miniforge3 on the 4070 host
# with two Habitat-Sim environments. hab024 has habitat-sim 0.2.4, the last release with PTex,
# which Replica's full-quality textures need. hab033 has 0.3.3 (current), for ReplicaCAD. No
# `conda init`, so the host's shell setup is unchanged. CPU and disk only, at low priority; no GPU.
#   scripts/rtx4070/setup_habitat.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/rtx4070/common.sh"
"${R4_SSH[@]}" bash -s <<'REMOTE'
set -euo pipefail
renice -n 10 -p $$ >/dev/null
if [[ ! -x ~/miniforge3/bin/conda ]]; then
  curl -fsSLo /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/Miniforge3.sh -b -p ~/miniforge3
  rm /tmp/Miniforge3.sh
fi
C=~/miniforge3/bin/conda
for spec in "hab024 0.2.4" "hab033 0.3.3"; do
  set -- $spec
  [[ -d ~/miniforge3/envs/$1 ]] || $C create -y -q -n "$1" python=3.9
  # libopengl: the host lacks libOpenGL.so.0 (Ubuntu's libopengl0), which habitat-sim links against
  $C install -y -q -n "$1" "habitat-sim=$2" headless libopengl -c conda-forge -c aihabitat
  $C run -n "$1" python -c "import habitat_sim as h; print('$1: habitat-sim', h.__version__, '| CUDA build:', getattr(h, 'cuda_enabled', '?'))"
done
df -h ~ | tail -1
REMOTE
