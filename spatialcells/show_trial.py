"""Show one behavioural trial: the new view (with its 64-token grid), the two frames the
model chose between, and where each was taken.

    python -m spatialcells.show_trial --near 130 --far 149
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analyze_behave import geometry
from .room import L, PILLARS, WALL_TEX
from .summarize import INK, INK2, S1, S2, S3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", type=int, required=True)
    ap.add_argument("--far", type=int, required=True)
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--out", default="results/example_trial.png")
    args = ap.parse_args()
    z = np.load(f"results/behave/{args.model}-reason.npz")
    i = next(i for i in range(len(z["probe"])) if z["question"][i] == "choice" and z["think"][i]
             and z["near"][i] == args.near and z["far"][i] == args.far)
    meta, pmeta, G = geometry()
    frames = np.load("data/walks/frames.npy", mmap_mode="r")
    probes = np.load("data/walks/probes.npy", mmap_mode="r")
    p, ans = int(z["probe"][i]), int(z["answer"][i])
    fi = G[p]["fi"]
    pts = {"new view": (pmeta["x"][p], pmeta["y"][p], pmeta["theta"][p], S1),
           f"Frame {args.near:03d} (nearer)": (*(meta[k][fi[args.near - 1]] for k in ("x", "y", "theta")), S3),
           f"Frame {args.far:03d} (farther)": (*(meta[k][fi[args.far - 1]] for k in ("x", "y", "theta")), S2)}
    imgs = [probes[p], frames[fi[args.near - 1]], frames[fi[args.far - 1]]]

    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
    for ax, (lab, (x, y, th, col)), im in zip(axs, pts.items(), imgs):
        ax.imshow(im)
        d = np.hypot(x - pts["new view"][0], y - pts["new view"][1])
        sub = "256 x 256 px; grid = the 64 tokens" if lab == "new view" else f"{d:.1f} m from the new view"
        if ans == int(lab[6:9] if lab.startswith("Frame") else -1):
            sub += "\nthe model's answer"
        ax.set_title(f"{lab}\n{sub}", fontsize=9, color=col)
        if lab == "new view":
            for k in range(32, 256, 32):
                ax.axhline(k - 0.5, color="white", lw=0.6, alpha=0.7)
                ax.axvline(k - 0.5, color="white", lw=0.6, alpha=0.7)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(col), s.set_linewidth(3)

    ax = axs[3]
    w = meta["walk"][fi[0]]
    ax.plot(meta["x"][fi], meta["y"][fi], color="#c9c8c4", lw=0.8, zorder=1)
    ax.add_patch(plt.Rectangle((0, 0), L, L, fill=False, ec=INK, lw=1.5))
    for side, (tx, ty, rot) in {"N": (L / 2, L + 0.25, 0), "S": (L / 2, -0.45, 0),
                                "E": (L + 0.3, L / 2, -90), "W": (-0.45, L / 2, 90)}.items():
        ax.text(tx, ty, f"{WALL_TEX[side]} wall", ha="center", va="center", rotation=rot, fontsize=8, color=INK2)
    for px, py, pr, _, tex in PILLARS:
        ax.add_patch(plt.Circle((px, py), pr, color="#d94040" if tex == "red" else "#8a6d55", zorder=2))
    for lab, (x, y, th, col) in pts.items():
        r = 1.6   # 90 deg field of view wedge
        wedge = np.array([[x, y], [x + r * np.cos(th + np.pi / 4), y + r * np.sin(th + np.pi / 4)],
                          [x + r * np.cos(th - np.pi / 4), y + r * np.sin(th - np.pi / 4)]])
        ax.add_patch(plt.Polygon(wedge, color=col, alpha=0.25, lw=0, zorder=3))
        ax.annotate("", (x + 0.9 * np.cos(th), y + 0.9 * np.sin(th)), (x, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2), zorder=4)
        ax.plot(x, y, "o", color=col, ms=7, mec="white", zorder=5, label=lab)
    ax.set_xlim(-0.8, L + 0.8), ax.set_ylim(-0.8, L + 0.8)
    ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3)
    ax.set_title(f"Top-down: walk {w} (grey), positions and facing", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, facecolor="white")
    print(f"probe {p}: answer {ans}, wrote {args.out}")


if __name__ == "__main__":
    main()
