"""Export two-choice trials for tests that run outside the GPU host: the Claude API test and
the human-baseline page. Same probes and option frames as the VLM and VGGT tests
(behave_reason's records); presentation order is re-drawn with a fixed seed.

    python -m spatialcells.export_trials

Writes results/export/:
  claude/walk_WW/frame_KKK.jpg   every frame of the 16 test walks (JPEG q92, 256 px)
  claude/probe_PPPP.jpg          the probes
  claude/trials.json             all 104 novel trials + 3 familiar per walk
  human/walk_WW.jpg              16 x 16 sprite of a walk's 256 frames (4 walks)
  human/probes.jpg               sprite of the human trials' probes
  human/trials.json              the human trials (novel ones of those 4 walks + 3 familiar each)
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from .analyze_behave import angdiff, geometry

OUT = "results/export"
HUMAN_WALKS = 4
FAMILIAR_PER_WALK = 3


def main():
    meta, pmeta, G = geometry()
    frames = np.load("data/walks/frames.npy", mmap_mode="r")
    probes = np.load("data/walks/probes.npy", mmap_mode="r")
    z = np.load("results/behave/qwen38-reason.npz")
    rows = [(int(p), int(nr), int(fr)) for p, q, th, nr, fr in
            zip(z["probe"], z["question"], z["think"], z["near"], z["far"]) if q == "choice" and not th]
    rng = np.random.default_rng(7)
    walks = sorted({int(pmeta["walk"][p]) for p, _, _ in rows})
    trials = []
    for w in walks:
        mine = [r for r in rows if pmeta["walk"][r[0]] == w]
        novel = [r for r in mine if not G[r[0]]["familiar"]]
        fam = [r for r in mine if G[r[0]]["familiar"]]
        fam = [fam[i] for i in sorted(rng.choice(len(fam), FAMILIAR_PER_WALK, replace=False))]
        for p, nr, fr in novel + fam:
            fi, d = G[p]["fi"], G[p]["d"]
            a, b = (nr, fr) if rng.random() < 0.5 else (fr, nr)
            trials.append({
                "id": f"w{w:02d}p{p:04d}", "walk": w, "probe": p, "familiar": G[p]["familiar"],
                "near": nr, "far": fr, "a": a, "b": b,
                "near_dist_m": round(float(d[nr - 1]), 3), "far_dist_m": round(float(d[fr - 1]), 3),
                "pixels_favour_near": bool(G[p]["sim"][nr - 1] > G[p]["sim"][fr - 1]),
                "near_heading_diff_deg": round(float(np.degrees(angdiff(meta["theta"][fi[nr - 1]], pmeta["theta"][p]))), 1),
                "far_heading_diff_deg": round(float(np.degrees(angdiff(meta["theta"][fi[fr - 1]], pmeta["theta"][p]))), 1),
                "dist_to_frames_m": [round(float(x), 3) for x in d],
            })

    os.makedirs(f"{OUT}/claude", exist_ok=True)
    jpg = lambda a, path: Image.fromarray(np.asarray(a)).save(path, quality=92)
    for w in walks:
        os.makedirs(f"{OUT}/claude/walk_{w:02d}", exist_ok=True)
        fi = G[next(t["probe"] for t in trials if t["walk"] == w)]["fi"]
        for k, f in enumerate(fi, 1):
            jpg(frames[f], f"{OUT}/claude/walk_{w:02d}/frame_{k:03d}.jpg")
    for t in trials:
        jpg(probes[t["probe"]], f"{OUT}/claude/probe_{t['probe']:04d}.jpg")
    json.dump(trials, open(f"{OUT}/claude/trials.json", "w"))

    # human page: the walks with the most novel trials, frames as sprites
    by_novel = sorted(walks, key=lambda w: -sum(1 for t in trials if t["walk"] == w and not t["familiar"]))
    hw = sorted(by_novel[:HUMAN_WALKS])
    ht = [{k: v for k, v in t.items() if k != "dist_to_frames_m"} for t in trials if t["walk"] in hw]
    os.makedirs(f"{OUT}/human", exist_ok=True)
    for w in hw:
        fi = G[next(t["probe"] for t in ht if t["walk"] == w)]["fi"]
        sheet = np.zeros((16 * 256, 16 * 256, 3), np.uint8)
        for k, f in enumerate(fi):
            r, c = divmod(k, 16)
            sheet[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] = frames[f]
        Image.fromarray(sheet).save(f"{OUT}/human/walk_{w:02d}.jpg", quality=85)
    n = len(ht)
    cols = int(np.ceil(np.sqrt(n)))
    sheet = np.zeros((int(np.ceil(n / cols)) * 256, cols * 256, 3), np.uint8)
    for i, t in enumerate(ht):
        r, c = divmod(i, cols)
        sheet[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] = probes[t["probe"]]
        t["sprite"] = i
    Image.fromarray(sheet).save(f"{OUT}/human/probes.jpg", quality=90)
    json.dump({"walks": hw, "probe_cols": cols, "trials": ht}, open(f"{OUT}/human/trials.json", "w"))
    nn = sum(not t["familiar"] for t in trials)
    print(f"claude: {len(trials)} trials ({nn} novel) over {len(walks)} walks; "
          f"human: {n} trials ({sum(not t['familiar'] for t in ht)} novel) over walks {hw}")


if __name__ == "__main__":
    main()
