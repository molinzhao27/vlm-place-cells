"""Summary figure and table from results/summary.json.

    python -m spatialcells.summarize
"""
from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["pixels", "pixels+ema8", "dinov2-random", "dinov2", "qwen-single-random", "qwen-single",
         "qwen-shuffled", "qwen-walk", "qwen-single+ema8"]
LABEL = {"pixels": "raw pixels", "pixels+ema8": "pixels\n+ 8-step avg", "dinov2-random": "DINOv2\nrandom", "dinov2": "DINOv2",
         "qwen-single-random": "Qwen3-VL\nrandom", "qwen-single": "Qwen3-VL\none frame",
         "qwen-shuffled": "Qwen3-VL\nshuffled walk", "qwen-walk": "Qwen3-VL\nwalk in context"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"   # reference categorical slots 1-5
LINES = {"qwen-walk": S1, "qwen-single": S2, "qwen-shuffled": S3, "qwen-single-random": S4,
         "qwen-single+ema8": S5}
LABEL["qwen-single+ema8"] = "Qwen3-VL one\nframe + 8-step avg"
MODELS = {"qwen": "Qwen3-VL\n2B", "qwen8b": "Qwen3-VL\n8B", "qwen32b": "Qwen3-VL\n32B",
          "qwen38": "Qwen3.8 27B\n(newer family)", "sn8b": "SenseNova-SI 8B\n(Qwen3-VL-8B + spatial FT)"}
VL_SERIES = ("qwen", "qwen8b", "qwen32b")   # one family, one recipe: the size series


def model_x(models):
    """x positions: the Qwen3-VL sizes side by side, other families after a gap."""
    xs = []
    for m in models:
        xs.append(0.0 if not xs else xs[-1] + (1 if m in VL_SERIES else 1.5))
    return np.array(xs)


def series_plot(ax, models, xs, ys, color, label=None):
    """Connect the Qwen3-VL sizes; other families are lone diamonds."""
    ys = np.asarray(ys, float)
    vl = [i for i, m in enumerate(models) if m in VL_SERIES]
    other = [i for i, m in enumerate(models) if m not in VL_SERIES]
    ax.plot(xs[vl], ys[vl], color=color, lw=2, marker="o", ms=6, mec="white", mew=0.8, label=label)
    ax.plot(xs[other], ys[other], color=color, lw=0, marker="D", ms=6, mec="white", mew=0.8)


def family_divider(ax, models, xs):
    vl = [x for x, m in zip(xs, models) if m in VL_SERIES]
    other = [x for x, m in zip(xs, models) if m not in VL_SERIES]
    if vl and other:
        ax.axvline((max(vl) + min(other)) / 2, color=GRID, lw=1)
PROBE_CONDS = {"probes-walk": ("after the walk", S1), "probes-shuffled": ("after shuffled walk", S3),
               "probes-only": ("no walk", S2)}


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def layer_num(name):
    return 0 if name == "pix" else int(name[1:])


def main():
    S = json.load(open("results/summary.json"))
    tags = [t for t in ORDER if t in S]
    rows = []
    for t in tags:
        for layer, st in sorted(S[t].items(), key=lambda kv: layer_num(kv[0])):
            rows.append({"condition": t, "layer": layer, **st})
    with open("results/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, axs = plt.subplots(2, 3, figsize=(20, 9.5))
    x = np.arange(len(tags))
    xt = [LABEL[t] for t in tags]

    # (a) place-cell fraction per layer, as dots
    ax = axs[0, 0]
    for i, t in enumerate(tags):
        v = [100 * s["frac_place"] for s in S[t].values()]
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=22, color=S1, edgecolor="white", lw=0.6, zorder=3)
    style(ax, "a  Units passing rodent place-cell criteria", "% of active neuron poles (one dot per layer)")
    ax.set_xticks(x, xt, fontsize=8)

    # (b) heading invariance of those place units
    ax = axs[0, 1]
    for i, t in enumerate(tags):
        v = [s["place_inv_over_stab_median"] for s in S[t].values() if np.isfinite(s["place_inv_over_stab_median"])]
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=22, color=S1, edgecolor="white", lw=0.6, zorder=3)
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.text(len(tags) - 0.5, 1.02, "true place cell: same map facing any direction", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.axhline(0, color=GRID, lw=1)
    style(ax, "b  Are they place cells or view cells?", "heading invariance / stability (median over place units)")
    ax.set_xticks(x, xt, fontsize=8)

    # (c) decoding at each condition's best layer
    ax = axs[0, 2]
    best = {t: min(S[t].values(), key=lambda s: s["decode_cross_heading"]) for t in tags}
    wdt = 0.36
    b1 = ax.bar(x - wdt / 2, [best[t]["decode_within_heading"] for t in tags], wdt - 0.03, color=S1, label="same facing direction")
    b2 = ax.bar(x + wdt / 2, [best[t]["decode_cross_heading"] for t in tags], wdt - 0.03, color=S2, label="opposite facing direction")
    chance = np.mean([best[t]["decode_chance"] for t in tags])
    ax.axhline(chance, color=INK2, lw=1, ls="--")
    ax.text(-0.5, chance + 0.05, "chance (guess the room centre)", fontsize=8, color=INK2, va="bottom")
    ax.bar_label(b2, fmt="%.1f", fontsize=7, color=INK2, padding=2)
    style(ax, "c  Decoding position from a layer (best layer)", "median error on held-out walks (m)")
    ax.set_xticks(x, xt, fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.93))

    # (d-f) Qwen3-VL by layer
    qtags = [t for t in LINES if t in S]
    panels = [("frac_place", 100, "d  Qwen3-VL: place-criteria units by layer", "% of active poles"),
              ("place_inv_over_stab_median", 1, "e  Qwen3-VL: heading invariance by layer", "invariance / stability"),
              ("decode_cross_heading", 1, "f  Qwen3-VL: decoding across facing directions", "median error (m)")]
    for ax, (key, k, title, yl) in zip(axs[1], panels):
        for t in qtags:
            ls = sorted(S[t], key=layer_num)
            xs = [layer_num(l) for l in ls]
            ys = [k * S[t][l][key] for l in ls]
            ax.plot(xs, ys, color=LINES[t], lw=2, marker="o", ms=5, mec="white", mew=0.8)
            ax.annotate(LABEL[t].replace("\n", " "), (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=INK, va="center")
        if key == "decode_cross_heading":
            ax.axhline(chance, color=INK2, lw=1, ls="--")
        style(ax, title, yl)
        ax.set_xlabel("language-model layer", fontsize=9, color=INK2)
        ax.set_xlim(0, 36)
        ax.legend([plt.Line2D([], [], color=LINES[t], lw=2) for t in qtags],
                  [LABEL[t].replace("\n", " ") for t in qtags], frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig("results/summary.png", dpi=110, facecolor="white")
    print("wrote results/summary.png and results/summary.csv")
    probes_figure()
    units_scaling_figure(S, chance)
    vggt_figure(S, chance)


def vggt_figure(S, chance):
    """VGGT's camera-token neurons by layer: alone, with anchors only, with a stretch of the walk."""
    conds = {"vggt-alone": ("each frame alone", S2), "vggt-single": ("frame + 8 anchor views", S1),
             "vggt-walk": ("64 walk frames + anchors", S5)}
    if not all(t in S for t in conds):
        return
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    panels = [("frac_place", 100, "a  VGGT: units passing place-cell criteria", "% of active poles"),
              ("place_inv_over_stab_median", 1, "b  VGGT: heading invariance of those units", "invariance / stability"),
              ("decode_cross_heading", 1, "c  VGGT: decoding position across facing directions", "median error (m)")]
    for ax, (key, k, title, yl) in zip(axs, panels):
        for t, (lab, col) in conds.items():
            ls = sorted(S[t], key=layer_num)
            ax.plot([layer_num(l) for l in ls], [k * S[t][l][key] for l in ls], color=col, lw=2, marker="o", ms=5,
                    mec="white", mew=0.8, label=lab)
        if key == "decode_cross_heading":
            ax.axhline(chance, color=INK2, lw=1, ls="--")
            ax.text(23, chance + 0.05, "chance", ha="right", va="bottom", fontsize=8, color=INK2)
        if key == "place_inv_over_stab_median":
            ax.axhline(1, color=INK2, lw=1, ls="--")
        style(ax, title, yl)
        ax.set_xlabel("global-attention block (of 24)", fontsize=9, color=INK2)
        ax.set_ylim(0, None)
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig("results/vggt_units.png", dpi=110, facecolor="white")
    print("wrote results/vggt_units.png")


def units_scaling_figure(S, chance):
    """Single units across model sizes: one frame, walk in context, one frame + 8-step average."""
    conds = {"single": ("one frame", S2), "walk": ("walk in context", S1), "single+ema8": ("one frame + 8-step avg", S5)}
    models = [m for m in MODELS if f"{m}-walk" in S]
    if len(models) < 2:
        return
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    x = model_x(models)
    for ax, key, agg, title, yl in [
            (axs[0], "decode_cross_heading", min, "a  Decoding position across facing directions (best layer)",
             "median error (m)"),
            (axs[1], "place_inv_over_stab_median", max, "b  Heading invariance of place-criteria units (best layer)",
             "invariance / stability")]:
        for c, (lab, col) in conds.items():
            ys = [agg(v[key] for v in S[f"{m}-{c}"].values() if np.isfinite(v[key])) if f"{m}-{c}" in S else np.nan
                  for m in models]
            series_plot(ax, models, x, ys, col, label=lab)
        ref, reflab = (chance, "chance") if key == "decode_cross_heading" else (1, "true place cell")
        ax.axhline(ref, color=INK2, lw=1, ls="--")
        ax.text(x[-1], ref + 0.03, reflab, ha="right", va="bottom", fontsize=8, color=INK2)
        family_divider(ax, models, x)
        style(ax, title, yl)
        ax.set_xticks(x, [MODELS[m] for m in models], fontsize=8)
        ax.set_ylim(0, None)
        ax.legend(frameon=False, fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig("results/units_scaling.png", dpi=110, facecolor="white")
    print("wrote results/units_scaling.png")


def probes_figure():
    """Teleport-probe decoding across model sizes."""
    import os
    if not os.path.exists("results/probes.json"):
        return
    R = json.load(open("results/probes.json"))
    models = [m for m in MODELS if any(f"{m}-{c}" in R for c in PROBE_CONDS)]
    chance = R["pixels"]["pix"]["chance"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    x = model_x(models)
    refs = [(R["pixels"]["pix"], "probe's own pixels"), (R["pixels-recall-ema8"]["pix"], "recall the matching walk view")]
    for ax, key, title in [(axs[0], "cross_heading", "a  Teleport probes: decoding across facing directions"),
                           (axs[1], "within_heading", "b  Teleport probes: decoding within a facing direction")]:
        for cond, (lab, col) in PROBE_CONDS.items():
            ys = [min(v[key] for v in R[f"{m}-{cond}"].values()) if f"{m}-{cond}" in R else np.nan for m in models]
            series_plot(ax, models, x, ys, col, label=f"probes {lab}")
        lo = min(r[key] for r, _ in refs)
        for (ref, lab), ls in zip(refs, (":", "-.")):
            ax.axhline(ref[key], color=INK2, lw=1, ls=ls)
            below = ref[key] == lo and abs(refs[0][0][key] - refs[1][0][key]) < 0.3   # keep close labels apart
            ax.text(x[-1] + 0.4, ref[key] + (-0.04 if below else 0.04), lab, ha="right",
                    va="top" if below else "bottom", fontsize=8, color=INK2)
        ax.axhline(chance, color=INK2, lw=1, ls="--")
        ax.text(x[-1] + 0.4, chance + 0.04, "chance", ha="right", va="bottom", fontsize=8, color=INK2)
        family_divider(ax, models, x)
        style(ax, title, "median error, best layer (m)")
        ax.set_xticks(x, [MODELS[m] for m in models], fontsize=8)
        ax.set_ylim(0, chance + 0.6)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax = axs[2]
    cols = [S1, S2, S3, S4]
    for m, col in zip(models, cols):
        tag = f"{m}-probes-walk"
        if tag not in R:
            continue
        ls = sorted(R[tag], key=lambda s: int(s[1:]))
        depth = np.array([int(l[1:]) for l in ls], float)
        ys = [R[tag][l]["cross_heading"] for l in ls]
        ax.plot(depth / depth.max(), ys, color=col, lw=2, marker="o" if m in VL_SERIES else "D", ms=5,
                mec="white", mew=0.8, ls="-" if m in VL_SERIES else "--", label=MODELS[m].replace("\n", " "))
    ax.axhline(chance, color=INK2, lw=1, ls="--")
    style(ax, "c  Probes after the walk, by layer", "median error across facing directions (m)")
    ax.set_xlabel("relative depth in the language model", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig("results/probes.png", dpi=110, facecolor="white")
    print("wrote results/probes.png")


if __name__ == "__main__":
    main()
