"""Decode teleport-probe positions from each condition's activations.

    python -m spatialcells.analyze_probes --models qwen qwen8b qwen38 qwen32b

For every model, condition (probes after the walk, after the shuffled walk, or
alone) and layer: ridge decoding of the probe's (x, y), trained on even walks and
tested on odd walks, within the same facing half and across to the opposite one.
Baselines:
  pixels              the probe's own pixels (a view, no memory)
  pixels-recall-ema8  the walk frame that looks most like the probe, represented
                      by its 8-step moving average of pixels: what recognising the
                      view and recalling what was seen around then would give
Writes results/probes.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .analyze import causal_ema, decoding_suite, dev, heading_matrix


def pixel_feats(frames):
    return np.stack([np.asarray(f).reshape(32, 8, 32, 8, 3).mean((1, 3)).ravel() for f in frames]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", default="data/walks")
    args = ap.parse_args()
    meta = dict(np.load(f"{args.data}/meta.npz"))
    pmeta = dict(np.load(f"{args.data}/probes_meta.npz"))
    path = "results/probes.json"
    res = json.load(open(path)) if os.path.exists(path) else {}

    W = pixel_feats(np.load(f"{args.data}/frames.npy", mmap_mode="r"))
    P = pixel_feats(np.load(f"{args.data}/probes.npy", mmap_mode="r"))
    mu, sd = W.mean(0), W.std(0) + 1e-6
    Wz = torch.from_numpy((W - mu) / sd).to(dev)
    Pz = torch.from_numpy((P - mu) / sd).to(dev)
    res["pixels"] = {"pix": decoding_suite(Pz, pmeta)}
    Wema = causal_ema(Wz, meta, 8)
    recall = torch.empty_like(Pz)
    for w in np.unique(pmeta["walk"]):
        wi = torch.from_numpy(np.flatnonzero(meta["walk"] == w)).to(dev)
        pi = torch.from_numpy(np.flatnonzero(pmeta["walk"] == w)).to(dev)
        sim = F.normalize(Pz[pi], dim=1) @ F.normalize(Wz[wi], dim=1).T
        recall[pi] = Wema[wi[sim.argmax(1)]]
    res["pixels-recall-ema8"] = {"pix": decoding_suite(recall, pmeta)}
    mats = {"pixels": {"layer": "pix", "sectors": ["E", "N", "W", "S"], "error_m": heading_matrix(Pz, pmeta)},
            "pixels-recall-ema8": {"layer": "pix", "sectors": ["E", "N", "W", "S"], "error_m": heading_matrix(recall, pmeta)}}

    for model in args.models:
        for cond in ("probes-walk", "probes-shuffled", "probes-only"):
            tag = f"{model}-{cond}"
            if not os.path.isdir(f"data/acts/{tag}"):
                continue
            layers = sorted((f[:-4] for f in os.listdir(f"data/acts/{tag}") if f.endswith(".npy")), key=lambda s: int(s[1:]))
            for layer in layers:
                X = torch.from_numpy(np.load(f"data/acts/{tag}/{layer}.npy").astype(np.float32)).to(dev)
                res.setdefault(tag, {})[layer] = decoding_suite(X, pmeta)
                del X
            # train-heading x test-heading matrix at the layer that transfers best across headings
            best = min(res[tag], key=lambda l: res[tag][l]["cross_heading"])
            X = torch.from_numpy(np.load(f"data/acts/{tag}/{best}.npy").astype(np.float32)).to(dev)
            mats[tag] = {"layer": best, "sectors": ["E", "N", "W", "S"], "error_m": heading_matrix(X, pmeta)}
            del X
            json.dump(res, open(path, "w"), indent=1)
            json.dump(mats, open("results/probes_heading_matrix.json", "w"), indent=1)

    print("median error (m) decoding probe position | within facing half / across to opposite half | chance %.2f"
          % res["pixels"]["pix"]["chance"])
    for tag, by_layer in res.items():
        cells = "  ".join(f"{l}:{v['within_heading']:.2f}/{v['cross_heading']:.2f}" for l, v in by_layer.items())
        print(f"{tag:28s} {cells}")
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
