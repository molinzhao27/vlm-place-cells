"""Score the behavioural answers from behave.py.

    python -m spatialcells.analyze_behave --models qwen qwen8b qwen38 qwen32b

For each probe, the error is the distance from where the probe was taken to where
the chosen frame was taken. Probes are split by whether the walk ever saw that view:
  familiar  some walk frame within 1 m faced within 45 deg of the probe's heading
  novel     the place was visited but never seen facing this way
Baselines on the same probes:
  pixel match  the walk frame whose pixels look most like the probe (pure view matching)
  random       a uniformly random walk frame (expected value)
  best         the walk frame actually nearest the probe
Writes results/behave.json and results/behave.png.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .analyze_probes import pixel_feats

NEAR, SAME_HEADING = 1.0, np.pi / 4


def angdiff(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def geometry(data="data/walks"):
    """Per probe: distances to its walk's frames (in walk order), familiar or novel, and the
    walk frame whose pixels look most like it. Returns (meta, pmeta, G)."""
    meta = dict(np.load(f"{data}/meta.npz"))
    pmeta = dict(np.load(f"{data}/probes_meta.npz"))
    W = pixel_feats(np.load(f"{data}/frames.npy", mmap_mode="r"))
    P = pixel_feats(np.load(f"{data}/probes.npy", mmap_mode="r"))
    mu, sd = W.mean(0), W.std(0) + 1e-6
    unit = lambda X: (X - mu) / sd / np.linalg.norm((X - mu) / sd, axis=1, keepdims=True)
    Wn, Pn = unit(W), unit(P)
    walk_frames = {}
    for w in np.unique(meta["walk"]):
        i = np.flatnonzero(meta["walk"] == w)
        walk_frames[w] = i[np.argsort(meta["step"][i])]
    G = {}
    for p in range(len(pmeta["walk"])):
        fi = walk_frames[pmeta["walk"][p]]
        d = np.hypot(meta["x"][fi] - pmeta["x"][p], meta["y"][fi] - pmeta["y"][p])
        dh = angdiff(meta["theta"][fi], pmeta["theta"][p])
        G[p] = {"d": d, "familiar": bool(((d < NEAR) & (dh < SAME_HEADING)).any()),
                "sim": Wn[fi] @ Pn[p], "pix": int(np.argmax(Wn[fi] @ Pn[p])), "fi": fi}
    return meta, pmeta, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", default="data/walks")
    args = ap.parse_args()
    meta, pmeta, G = geometry(args.data)

    def summarise(errs, hits):
        errs, hits = np.asarray(errs, float), np.asarray(hits, float)
        return {"median_error_m": float(np.median(errs)), "hit_rate_1m": float(hits.mean())}

    res = {}
    for model in args.models:
        path = f"results/behave/{model}.npz"
        if not os.path.exists(path):
            continue
        z = np.load(path)
        out = {}
        for cat in ("all", "familiar", "novel"):
            rows = [(p, a) for p, a in zip(z["probe"], z["answer"])
                    if cat == "all" or G[p]["familiar"] == (cat == "familiar")]
            parsed = [(p, a) for p, a in rows if a > 0]
            g = [G[p] for p, _ in parsed]
            dm = [G[p]["d"][a - 1] for p, a in parsed]
            dp = [x["d"][x["pix"]] for x in g]
            out[cat] = {
                "n": len(rows), "parsed": len(parsed) / max(len(rows), 1),
                "model": summarise(dm, np.array(dm) < NEAR),
                "pixel_match": summarise(dp, np.array(dp) < NEAR),
                "random": {"median_error_m": float(np.median([np.median(x["d"]) for x in g])),
                           "hit_rate_1m": float(np.mean([(x["d"] < NEAR).mean() for x in g]))},
                "best": summarise([x["d"].min() for x in g], [x["d"].min() < NEAR for x in g]),
                # did the model pick the frame that looks most like the probe?
                "same_as_pixel_match": float(np.mean([a - 1 == G[p]["pix"] for p, a in parsed])),
                "heading_diff_deg_median": float(np.degrees(np.median(
                    [angdiff(meta["theta"][G[p]["fi"][a - 1]], pmeta["theta"][p]) for p, a in parsed]))),
                # picks facing within 45 deg of the probe: for novel probes these are all >= 1 m away
                "picks_facing_same_way": float(np.mean(
                    [angdiff(meta["theta"][G[p]["fi"][a - 1]], pmeta["theta"][p]) < SAME_HEADING for p, a in parsed])),
            }
        vals, counts = np.unique(z["answer"][z["answer"] > 0], return_counts=True)
        top = np.argsort(-counts)[:5]
        out["top_answers"] = {int(vals[i]): int(counts[i]) for i in top}
        res[model] = out

    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/behave.json", "w"), indent=1)
    print(f"{'':10s} {'cat':9s} {'n':>5s} {'parsed':>6s} | hit rate within 1 m: model / pixel match / random / best"
          f" | median error (m): model / pixel / random | picks pixel match, heading diff, picks facing same way")
    for model, out in res.items():
        for cat in ("all", "familiar", "novel"):
            o = out[cat]
            print(f"{model:10s} {cat:9s} {o['n']:5d} {o['parsed']:6.0%} | "
                  f"{o['model']['hit_rate_1m']:.2f} / {o['pixel_match']['hit_rate_1m']:.2f} / "
                  f"{o['random']['hit_rate_1m']:.2f} / {o['best']['hit_rate_1m']:.2f} | "
                  f"{o['model']['median_error_m']:.2f} / {o['pixel_match']['median_error_m']:.2f} / "
                  f"{o['random']['median_error_m']:.2f} | {o['same_as_pixel_match']:.0%}, "
                  f"{o['heading_diff_deg_median']:.0f} deg, {o['picks_facing_same_way']:.0%}")
        print(f"{model:10s} most common answers: {out['top_answers']}")
    figure(res)


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .summarize import INK2, MODELS, S1, S2, family_divider, model_x, style
    models = [m for m in MODELS if m in res]
    if not models:
        return
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    x = model_x(models)
    for ax, cat, title in [(axs[0], "familiar", "a  View seen before nearby (same facing direction)"),
                           (axs[1], "novel", "b  Place visited, but never seen facing this way")]:
        bars = ax.bar(x, [res[m][cat]["model"]["hit_rate_1m"] for m in models], 0.6, color=S1 if cat == "familiar" else S2)
        ax.bar_label(bars, fmt=lambda v: f"{v:.0%}", fontsize=8, color=INK2, padding=2,
                     bbox=dict(fc="white", ec="none", pad=0.5))
        for key, lab, ls in [("pixel_match", "pick the most similar-looking frame", ":"), ("random", "random frame", "--")]:
            v = np.mean([res[m][cat][key]["hit_rate_1m"] for m in models])
            ax.axhline(v, color=INK2, lw=1, ls=ls, label=f"{lab} ({v:.0%})")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        family_divider(ax, models, x)
        n = res[models[0]][cat]["n"]
        style(ax, f"{title}, n = {n}", "fraction of answers within 1 m of the true place")
        ax.set_xticks(x, [MODELS[m] for m in models], fontsize=8)
    axs[0].set_ylim(0, 1)
    fig.suptitle("Which earlier frame was taken nearest to this new view?", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/behave.png", dpi=110, facecolor="white")
    print("wrote results/behave.png")


if __name__ == "__main__":
    main()
