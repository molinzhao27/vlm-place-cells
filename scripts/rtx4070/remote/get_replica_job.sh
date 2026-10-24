#!/usr/bin/env bash
# Runs on the 4070 host; started by scripts/rtx4070/get_replica.sh. Argument: the Replica scenes to keep.
set -euo pipefail
D=~/habitat-data
mkdir -p "$D/replica"
echo "start $(date -u +%T)"
# Habitat's datasets_download clones with git and needs git-lfs, which this host lacks: use huggingface_hub
~/miniforge3/envs/hab033/bin/python - "$D" <<'PY'
import sys
from huggingface_hub import snapshot_download
d = sys.argv[1]
for repo, sub in (("ai-habitat/habitat_test_scenes", "scene_datasets/habitat-test-scenes"),
                  ("ai-habitat/ReplicaCAD_baked_lighting", "replica_cad_baked_lighting")):
    print(snapshot_download(repo, repo_type="dataset", local_dir=f"{d}/{sub}"))
PY
echo "pilot data done $(date -u +%T)"
for p in a b c d e f g h i j k l m n o p q; do
  curl -fsSL --retry 5 "https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.parta$p"
done | gzip -dc | tar -x -C "$D/replica" $1      # scenes are top-level folders (apartment_0/, room_0/, ...)
curl -fsSL -o "$D/replica/additional_habitat_configs.zip" http://dl.fbaipublicfiles.com/habitat/Replica/additional_habitat_configs.zip
(cd "$D/replica" && python3 -m zipfile -e additional_habitat_configs.zip . && rm additional_habitat_configs.zip)   # no unzip on this host
du -sh "$D"/* "$D"/replica/*/* 2>/dev/null | head -40
df -h ~ | tail -1
echo "done $(date -u +%T)"
