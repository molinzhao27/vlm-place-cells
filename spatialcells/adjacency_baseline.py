"""Can a simple "look-alike, then time" rule do the lure-free two-choice task?

    python -m spatialcells.adjacency_baseline

For each lure-free trial (results/export/dissimilar/trials.json): find the walk frames that look
most like the new view, then answer with the option closer in frame number to them. Looks are
compared with pixels (32 x 32 RGB, as dissimilar_trials.py) or DINOv2-base CLS features. The rule
uses either all 256 frames or the 64 a 64-image prompt shows (two_choice.py --max-images 64), and
either the single best match or the median frame number of the top 3 matches. If it scores like
the fine-tuned Qwen3-VL-8B (experiment 9), that model needs nothing more than view matching plus the
order of the walk. Writes results/adjacency_baseline.json.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from .analyze_probes import pixel_feats


@torch.no_grad()
def dino(images, batch=256):
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(np.array(images[i:i + batch])).cuda().permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=224, mode="bicubic", antialias=True)
        out.append(model(pixel_values=(x - mean) / std).last_hidden_state[:, 0].float().cpu())
    return torch.cat(out).numpy()


def main():
    frames = np.load("data/walks/frames.npy", mmap_mode="r")
    probes = np.load("data/walks/probes.npy", mmap_mode="r")
    meta = dict(np.load("data/walks/meta.npz"))
    trials = json.load(open("results/export/dissimilar/trials.json"))
    pidx = sorted({t["probe"] for t in trials})
    feats = {"pixels": (pixel_feats(frames), pixel_feats(probes[pidx])),
             "dinov2": (dino(frames), dino(probes[pidx]))}
    order = {w: np.flatnonzero(meta["walk"] == w)[np.argsort(meta["step"][meta["walk"] == w])] for w in np.unique(meta["walk"])}
    res = {}
    for name, (W, P) in feats.items():
        mu, sd = W.mean(0), W.std(0) + 1e-6
        unit = lambda X: (X - mu) / sd / np.linalg.norm((X - mu) / sd, axis=1, keepdims=True)
        Wn, Pn = unit(W), unit(P)
        prow = {p: i for i, p in enumerate(pidx)}
        for shown_name in ("all 256", "64 shown"):
            for k in (1, 3):
                right = {"novel": [], "familiar": []}
                for t in trials:
                    shown = np.arange(1, 257)
                    if shown_name == "64 shown":
                        shown = np.array(sorted(set(np.linspace(1, 256, 61).round().astype(int).tolist()) | {t["a"], t["b"]}))
                    sim = Wn[order[t["walk"]][shown - 1]] @ Pn[prow[t["probe"]]]
                    ref = np.median(shown[np.argsort(-sim)[:k]])
                    pick = t["near"] if abs(t["near"] - ref) < abs(t["far"] - ref) else t["far"]
                    right["familiar" if t["familiar"] else "novel"].append(pick == t["near"])
                key = f"{name}, {shown_name}, top {k}"
                res[key] = {c: float(np.mean(v)) for c, v in right.items()}
                print(f"{key:28s} novel {res[key]['novel']:.0%}  familiar {res[key]['familiar']:.0%}", flush=True)
    json.dump(res, open("results/adjacency_baseline.json", "w"), indent=1)


if __name__ == "__main__":
    main()
