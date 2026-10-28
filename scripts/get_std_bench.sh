#!/usr/bin/env bash
# Experiment 17 data: VSI-Bench (videos unzipped to videos/<dataset>/) and MindCube, at the registered
# Hugging Face revisions, into data/bench/. Prints the md5 of the files the runs read.
#   gq submit --on vast3 --project mechinterp --name e17_data --cpu --est 15m -- /bin/bash scripts/get_std_bench.sh vsibench
set -eu
if [[ $(uname) == Darwin ]]; then P=.venv-mac/bin/python; else P=.venv/bin/python; fi
for B in "${@:-vsibench mindcube}"; do
  case $B in
    vsibench) R=nyu-visionx/VSI-Bench; V=3a6a1cf9ac54c403509b15260897cc6ddc41b408 ;;
    mindcube) R=MLL-Lab/MindCube; V=9c941b46a6bd65b6914669ef7a579948fc9c8467 ;;
  esac
  mkdir -p data/bench/$B
  $P -I -c "import sys; from huggingface_hub import snapshot_download as s; s(sys.argv[1], repo_type='dataset', revision=sys.argv[2], local_dir=sys.argv[3], max_workers=4)" "$R" "$V" "data/bench/$B"
  if [[ $B == vsibench ]]; then
    $P -I -c "import sys, zipfile; [zipfile.ZipFile(f'{sys.argv[1]}/{z}.zip').extractall(f'{sys.argv[1]}/videos') for z in ('arkitscenes', 'scannet', 'scannetpp')]" data/bench/vsibench
    (cd data/bench/vsibench && md5sum test.jsonl pruned_ids.txt && find videos -name '*.mp4' | sort | xargs md5sum | md5sum)
  else
    $P -I -c "import sys, zipfile; zipfile.ZipFile(f'{sys.argv[1]}/data.zip').extractall(sys.argv[1])" data/bench/mindcube
    (cd data/bench/mindcube/data && md5sum raw/MindCube_tinybench.jsonl && find other_all_image -type f | sort | xargs md5sum | md5sum)
  fi
done
