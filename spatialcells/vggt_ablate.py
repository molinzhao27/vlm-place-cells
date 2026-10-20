"""Does VGGT use its heading-invariant place units to localise? Mean-ablation of those units.

    python -m spatialcells.vggt_ablate [--every 16]

Same input as vggt_cells.py --mode single: the 8 anchor views, then one walk frame. The
units are MLP neurons (fc2 input) at the walk frame's camera token. A neuron is
"invariant" if one of its poles is a place unit with heading invariance > 0.5 in
vggt-single (analyze.py), and "heading-specific" if it has a place pole but none of them is
invariant. Each condition replaces K of those neurons by their mean over all walk frames,
in block 19 alone or in blocks 15, 19 and 23 together, and measures the camera head's
position and heading error on every --every'th walk frame. K is the smaller of the two
sets, so invariant and heading-specific neurons are compared at equal count, against
random neurons of the same count. --all-neurons is the positive control: every neuron at that
token in those blocks. Writes results/vggt_ablate.json (or vggt_ablate_all.json).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .analyze import analyze_layer, load_units
from .vggt_cells import anchors
from .vggt_control import RES, umeyama

BLOCK_SETS = {"block 19": [19], "blocks 15+19+23": [15, 19, 23]}


def unit_sets(layer, meta, n_neurons):
    X = load_units("vggt-single", layer, "data/walks", meta)
    _, pu = analyze_layer(X, meta, np.random.default_rng(0))
    pole, place, hi = (pu[k].cpu().numpy() for k in ("pole", "place", "hi"))
    neuron = pole % n_neurons                      # poles are [positive half, negative half]
    inv = np.unique(neuron[place & (hi > 0.5)])
    spec = np.setdiff1d(np.unique(neuron[place & (hi <= 0.5)]), inv)
    return inv, spec


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/walks")
    ap.add_argument("--vggt", default="/workspace/vggt")
    ap.add_argument("--every", type=int, default=16, help="use every Nth walk frame")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--all-neurons", action="store_true", help="positive control: ablate every neuron")
    args = ap.parse_args()
    sys.path.insert(0, args.vggt)
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    frames = np.load(f"{args.data}/frames.npy", mmap_mode="r")
    meta = dict(np.load(f"{args.data}/meta.npz"))
    A = anchors(args.data)
    model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
    agg = model.aggregator
    P = agg.patch_start_idx + (RES // 14) ** 2
    n_anchor = len(A["images"])
    n_neurons = agg.global_blocks[19].mlp.fc2.in_features
    rng = np.random.default_rng(0)

    blocks = sorted({b for bs in BLOCK_SETS.values() for b in bs})
    sets, means = {}, {}
    for b in blocks:
        inv, spec = unit_sets(f"L{b}", meta, n_neurons)
        K = min(len(inv), len(spec))
        sets[b] = {"invariant": rng.choice(inv, K, replace=False), "heading-specific": rng.choice(spec, K, replace=False),
                   "random 1": rng.choice(n_neurons, K, replace=False), "random 2": rng.choice(n_neurons, K, replace=False),
                   "all neurons": np.arange(n_neurons)}
        means[b] = torch.from_numpy(np.load(f"data/acts/vggt-single/L{b}.npy").astype(np.float32).mean(0)).cuda()
        print(f"block {b}: {len(inv)} invariant, {len(spec)} heading-specific place neurons; ablating K={K}", flush=True)

    active = {}                                     # block -> neuron indices to ablate in this condition
    def hook(b):
        def f(_, inp):
            if b not in active:
                return None
            x = inp[0].clone()
            u = active[b]
            x[:, n_anchor * P, u] = means[b][u].to(x.dtype)     # the walk frame's camera token
            return (x,)
        return f
    for b in blocks:
        agg.global_blocks[b].mlp.fc2.register_forward_pre_hook(hook(b))

    def prep(ims):
        x = torch.from_numpy(np.stack([np.asarray(a) for a in ims])).cuda().permute(0, 3, 1, 2).float() / 255
        return F.interpolate(x, size=(RES, RES), mode="bicubic", align_corners=False).clamp(0, 1)

    anchor_px = prep(A["images"])
    gt_anchor = np.c_[A["poses"][:, :2], np.ones(n_anchor)]
    idx = np.arange(0, len(frames), args.every)

    def pose_errors():
        err, herr = [], []
        for b0 in range(0, len(idx), args.batch):
            g = idx[b0:b0 + args.batch]
            seqs = torch.stack([torch.cat([anchor_px, prep(frames[[f]])]) for f in g])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens, _ = agg(seqs)
                pose = model.camera_head(tokens)[-1]
            ext, _ = pose_encoding_to_extri_intri(pose.float(), (RES, RES))
            for E, f in zip(ext.cpu().numpy().astype(np.float64), g):
                R, t = E[:, :, :3], E[:, :, 3]
                C = -np.einsum("sji,sj->si", R, t)
                fwd = np.einsum("sji,j->si", R, np.array([0.0, 0.0, 1.0]))
                s, Ra, ta = umeyama(C[:n_anchor], gt_anchor)
                c, fr = s * (Ra @ C[-1]) + ta, Ra @ fwd[-1]
                err.append(np.hypot(c[0] - meta["x"][f], c[1] - meta["y"][f]))
                d = np.arctan2(fr[1], fr[0]) - meta["theta"][f]
                herr.append(np.degrees(abs((d + np.pi) % (2 * np.pi) - np.pi)))
        return np.array(err), np.array(herr)

    out = {"n_frames": len(idx), "K": {f"L{b}": int(len(sets[b]["invariant"])) for b in blocks}, "conditions": {}}
    kinds = ("all neurons",) if args.all_neurons else ("invariant", "heading-specific", "random 1", "random 2")
    conds = [("none", None, None)] + [(f"{name}: {s}", name, s) for name in BLOCK_SETS for s in kinds]
    t0 = time.time()
    base = None
    for label, name, s in conds:
        active.clear()
        if name:
            active.update({b: torch.from_numpy(sets[b][s]).cuda() for b in BLOCK_SETS[name]})
        err, herr = pose_errors()
        if base is None:
            base = err
        o = {"median_pos_err_m": float(np.median(err)), "mean_pos_err_m": float(err.mean()),
             "frac_within_1m": float((err < 1).mean()), "median_heading_err_deg": float(np.median(herr)),
             "median_increase_m": float(np.median(err - base))}
        out["conditions"][label] = o
        print(f"{label:40s} position {o['median_pos_err_m']:.2f} m (mean {o['mean_pos_err_m']:.2f}, "
              f"within 1 m {o['frac_within_1m']:.0%}), heading {o['median_heading_err_deg']:.1f} deg, "
              f"{time.time() - t0:.0f}s", flush=True)
    path = "results/vggt_ablate_all.json" if args.all_neurons else "results/vggt_ablate.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
