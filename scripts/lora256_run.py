"""Experiment 24's runner (docs/experiments.md): spatialcells.two_choice on a room's novel trials only.

    .venv-mac/bin/python scripts/lora256_run.py --model Qwen/Qwen3-VL-8B-Instruct \
        --adapter data/ft/qwen8b-proc-e4-s5 --trials roomP5000 --device mps [--max-images 64 --tag ...]

It takes two_choice's arguments and changes one thing: the trial list is the room's manifest
(results/export/dissimilar-room<R>/trials.json) with the familiar trials left out, written to a temporary
file for the run. The prompt, frames, decoding, output file and default tag are two_choice's, so a file
written here can later be completed with the familiar trials by two_choice itself. Not for --score
(it would score against the novel-only list).
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatialcells import two_choice as tc  # noqa: E402

_manifest = tc.trials_file
_tmp = []


def novel_only(name):
    trials = [t for t in json.load(open(_manifest(name))) if not t["familiar"]]
    fd, path = tempfile.mkstemp(prefix=f"novel-{name}-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(trials, f)
    _tmp.append(path)
    print(f"{name}: {len(trials)} novel trials", flush=True)
    return path


if __name__ == "__main__":
    if "--score" in sys.argv:
        raise SystemExit("use python -m spatialcells.two_choice --score")
    tc.trials_file = novel_only
    try:
        tc.main()
    finally:
        for p in _tmp:
            os.remove(p)
