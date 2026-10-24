"""Can a simple "look-alike, then time" rule do the lure-free two-choice task?

    python -m spatialcells.adjacency_baseline
    python -m spatialcells.adjacency_baseline --presentations [--trials roomP5000 ...] [--device cpu]

For each lure-free trial (results/export/dissimilar/trials.json): find the walk frames that look
most like the new view, then answer with the option closer in frame number to them. Looks are
compared with pixels (32 x 32 RGB, as dissimilar_trials.py) or DINOv2-base CLS features. The rule
uses either all 256 frames or the 64 a 64-image prompt shows (two_choice.py --max-images 64), and
either the single best match or the median frame number of the top 3 matches. If it scores like
the fine-tuned Qwen3-VL-8B (experiment 9), that model needs nothing more than view matching plus the
order of the walk. Writes results/adjacency_baseline.json.
--presentations (experiment 13a) scores the rule on the 64-frame prompt as two_choice.py presents
it: ordered and shuffled (experiment 10), reversed, window-shuffled in blocks of 8 and renumbered,
by default in the ten held-out rooms (--trials as two_choice.py's). The look-alikes are found among
the frames shown, whatever their order; their time and the options' is read either from the labels
shown or from the positions in the prompt. Writes results/adjacency_presentations.json: novel and
familiar accuracy per room, and the mean and range over rooms.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from .analyze_probes import pixel_feats
from .two_choice import present, trials_file, walk_order

PRESENTATIONS = {"ordered": {}, "shuffled": {"shuffle_walk": True}, "reversed": {"reverse_walk": True},
                 "window-shuffled 8": {"window_shuffle": 8}, "renumbered": {"renumber": True}}


@torch.no_grad()
def dino(images, batch=256, device="cuda"):
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(np.array(images[i:i + batch])).to(device).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=224, mode="bicubic", antialias=True)
        out.append(model(pixel_values=(x - mean) / std).last_hidden_state[:, 0].float().cpu())
    return torch.cat(out).numpy()


def presentations(names, device):
    res = {}                                        # key -> trial set -> {"novel": acc, "familiar": acc}
    for name in names:
        data = f"data/walks-{name}" if name.startswith("room") else "data/walks"
        frames = np.load(f"{data}/frames.npy", mmap_mode="r")
        probes = np.load(f"{data}/probes.npy", mmap_mode="r")
        order = walk_order(dict(np.load(f"{data}/meta.npz")))
        trials = json.load(open(trials_file(name)))
        pidx = sorted({t["probe"] for t in trials})
        prow = {p: i for i, p in enumerate(pidx)}
        batch = 256 if device == "cuda" else 64
        feats = {"pixels": (pixel_feats(frames), pixel_feats(probes[pidx])),
                 "dinov2": (dino(frames, batch, device), dino(probes[pidx], batch, device))}
        for feat, (W, P) in feats.items():
            mu, sd = W.mean(0), W.std(0) + 1e-6
            unit = lambda X: (X - mu) / sd / np.linalg.norm((X - mu) / sd, axis=1, keepdims=True)
            Wn, Pn = unit(W), unit(P)
            for pres, flags in PRESENTATIONS.items():
                args = argparse.Namespace(**{"no_walk": False, "max_images": 64, "shuffle_walk": False, "relabel": False,
                                             "reverse_walk": False, "window_shuffle": None, "renumber": False, **flags})
                for k in (1, 3):
                    right = {(w, c): [] for w in ("labels", "positions") for c in ("novel", "familiar")}
                    for t in trials:
                        shown, lab = present(t, len(order[t["walk"]]), args)
                        srt = np.array(sorted(shown))
                        sim = Wn[order[t["walk"]][srt - 1]] @ Pn[prow[t["probe"]]]
                        best = srt[np.argsort(-sim)[:k]]
                        pos = {f: i + 1 for i, f in enumerate(shown)}
                        for w, when in (("labels", lab), ("positions", pos)):
                            ref = np.median([when[f] for f in best])
                            near = abs(when[t["near"]] - ref) < abs(when[t["far"]] - ref)
                            right[w, "familiar" if t["familiar"] else "novel"].append(near)
                    for w in ("labels", "positions"):
                        res.setdefault(f"{feat}, {pres}, time from {w}, top {k}", {})[name] = {
                            c: float(np.mean(right[w, c])) for c in ("novel", "familiar")}
        print(name + ": " + ", ".join(f"{p} {res[f'pixels, {p}, time from labels, top 1'][name]['novel']:.0%}"
                                      for p in PRESENTATIONS) + " (pixels, labels, top 1, novel)", flush=True)
    out = {}
    for key, by in res.items():
        nov = [v["novel"] for v in by.values()]
        out[key] = {"novel": float(np.mean(nov)), "novel_min": min(nov), "novel_max": max(nov),
                    "familiar": float(np.mean([v["familiar"] for v in by.values()])), "per_room": by}
        print(f"{key:58s} novel {out[key]['novel']:.1%} [{min(nov):.0%}, {max(nov):.0%}]  "
              f"familiar {out[key]['familiar']:.1%}")
    json.dump(out, open("results/adjacency_presentations.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presentations", action="store_true", help="experiment 13a: the 64 frames shown in each "
                                                                 "two_choice.py presentation")
    ap.add_argument("--trials", nargs="+", default=[f"roomP{5000 + i}" for i in range(10)],
                    help="with --presentations: trial sets, as two_choice.py's --trials")
    ap.add_argument("--device", default="cuda", help="for DINOv2: cuda or cpu")
    args = ap.parse_args()
    if args.presentations:
        return presentations(args.trials, args.device)
    frames = np.load("data/walks/frames.npy", mmap_mode="r")
    probes = np.load("data/walks/probes.npy", mmap_mode="r")
    meta = dict(np.load("data/walks/meta.npz"))
    trials = json.load(open("results/export/dissimilar/trials.json"))
    pidx = sorted({t["probe"] for t in trials})
    feats = {"pixels": (pixel_feats(frames), pixel_feats(probes[pidx])),
             "dinov2": (dino(frames, device=args.device), dino(probes[pidx], device=args.device))}
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
