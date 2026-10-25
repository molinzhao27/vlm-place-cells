"""Runs on the 4070 host with hab033's python; started by scripts/rtx4070/get_hssd.sh.
    python get_hssd_job.py REVISION DEST [PRIORITY]
Downloads hssd/hssd-hab at REVISION into DEST file by file, in an order that lets experiment 16's draw
start before the whole set is in:
1. the top-level configs, scenes/, semantics/, metadata/ and stages/;
2. every file of the objects used by the first PRIORITY houses of the fixed draw (sorted scene ids,
   random.Random(16).shuffle, as spatialcells/hssd_select.py), then DEST/.priority_done lists them;
3. everything else, except what the default (cluttered, non-articulated) scenes never use: urdf/,
   scenes-articulated/, scenes-uncluttered/ and scene_filter_files/; then DEST/.download_done.
Anonymous requests may resolve 3,000 files per 5 minutes; huggingface_hub waits out a 429 itself.
Files already there are skipped, so a rerun resumes."""
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download

REPO = "hssd/hssd-hab"
SKIP = ("urdf/", "scenes-articulated/", "scenes-uncluttered/", "scene_filter_files/")
rev, dest = sys.argv[1], sys.argv[2]
priority = int(sys.argv[3]) if len(sys.argv) > 3 else 24
t0 = time.time()
stamp = lambda: f"{time.strftime('%FT%TZ', time.gmtime())} +{(time.time() - t0) / 60:.1f} min"
print("RUN_START", stamp(), flush=True)
files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=rev) if not f.startswith(SKIP)]


def fetch(batch, label):
    todo = [f for f in batch if not os.path.exists(os.path.join(dest, f))]
    print(f"{label}: {len(batch)} files, {len(todo)} to fetch", stamp(), flush=True)

    def get(f):
        for attempt in range(30):            # beyond huggingface_hub's own retries on a 429
            try:
                return hf_hub_download(REPO, f, repo_type="dataset", revision=rev, local_dir=dest)
            except Exception as e:
                print(f"  retry {f}: {type(e).__name__}", flush=True)
                time.sleep(60)
        raise RuntimeError(f"could not fetch {f}")

    with ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(get, todo), 1):
            if i % 1000 == 0:
                print(f"  {i}/{len(todo)}", stamp(), flush=True)


fetch([f for f in files if "/" not in f or f.startswith(("scenes/", "semantics/", "metadata/", "stages/"))], "configs, scenes, semantics, metadata, stages")
ids = sorted(f[len("scenes/"):-len(".scene_instance.json")] for f in files
             if f.startswith("scenes/") and f.endswith(".scene_instance.json"))
order = list(ids)
random.Random(16).shuffle(order)
by_stem = {}
for f in files:
    if f.startswith("objects/"):
        by_stem.setdefault(os.path.basename(f).split(".")[0], []).append(f)
want = []
for sid in order[:priority]:
    scene = json.load(open(f"{dest}/scenes/{sid}.scene_instance.json"))
    for o in scene.get("object_instances", []):
        want += by_stem.get(o["template_name"].split("/")[-1], [])
want = sorted(set(want))
fetch(want, f"objects of the first {priority} drawn houses")
json.dump(order[:priority], open(f"{dest}/.priority_done", "w"))
fetch(files, "everything else")
open(f"{dest}/.download_done", "w").write(rev + "\n")
print("RUN_END", stamp(), flush=True)
