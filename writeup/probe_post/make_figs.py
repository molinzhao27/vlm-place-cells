"""Figures for the probe-vs-causal-direction post, from saved VGGT results (CPU only).

    .venv-claude/bin/python writeup/probe_post/make_figs.py

Reads results/vggt_patch.json, results/vggt_patch_subspaces.json and results/vggt_steer.json.
Writes writeup/probe_post/figs/fig1_depth.png, fig2_subspaces.png, fig3_neurons.png and
fig4_joystick.png and fig5_lambda.png (the last from results/vggt_patch_probes.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figs"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"     # categorical slot 1
ORANGE = "#eb6834"   # categorical slot 2
GRAY = "#b9b7ae"     # de-emphasised controls

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.linewidth": 1,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "axes.grid": False,
})
RING = dict(markeredgecolor=SURFACE, markeredgewidth=2, markersize=8)


def tidy(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def hbar(ax, y, value, height, color, r_px=4):
    """Horizontal bar from 0 with a rounded data end and a square baseline end.

    A nonzero value too small to see is drawn as a 2 px stub so it stays visible."""
    ax.figure.canvas.draw()
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    if value != 0 and abs(value) * abs(x1 - x0) < 2:
        value = (1 if value > 0 else -1) * 2 / abs(x1 - x0)
    s = 1 if value >= 0 else -1
    rx = min(r_px / abs(x1 - x0), abs(value))
    ry = min(r_px / abs(y1 - y0), height / 2)
    top, bot, end = y + height / 2, y - height / 2, value
    k = 0.5523  # cubic approximation of a quarter circle
    verts = [(0, bot), (end - s * rx, bot),
             (end - s * rx + s * k * rx, bot), (end, bot + ry - k * ry), (end, bot + ry),
             (end, top - ry),
             (end, top - ry + k * ry), (end - s * rx + s * k * rx, top), (end - s * rx, top),
             (0, top), (0, bot)]
    codes = [MPath.MOVETO, MPath.LINETO, *[MPath.CURVE4] * 3, MPath.LINETO, *[MPath.CURVE4] * 3,
             MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none"))


def bar_panel(ax, rows, xmax, ticks, ticklabels, fmt, title, labels=True):
    """rows: (label, value, colour), drawn top to bottom."""
    tidy(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)  # baseline drawn under the bars so 2 px stubs show
    ax.axvline(0, color=AXIS, lw=1, zorder=0)
    n = len(rows)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
    for i, (label, v, col) in enumerate(rows):
        y = n - 1 - i
        hbar(ax, y, max(v, 0), 0.6, col)
        ax.text(max(v, 0) + xmax * 0.015, y, fmt(v), va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels([r[0] for r in rows] if labels else [], fontsize=9.5, color=INK2)


def fig1_depth():
    d = json.load(open(RES / "vggt_patch.json"))
    blocks = [b for b in sorted(int(b) for b in d["blocks"]) if b <= 21]
    r2 = [d["blocks"][str(b)]["probe_r2"]["position 2"] for b in blocks]
    tr = [d["blocks"][str(b)]["full"]["transfer_mean"] for b in blocks]
    head = d["head input"]["transfer_mean"]

    fig, ax = plt.subplots(figsize=(6.6, 3.7), dpi=200)
    tidy(ax)
    ax.set_xlim(0.5, 24.5)
    ax.set_ylim(-0.03, 1.1)
    ax.set_xticks(blocks + [23.5])
    ax.set_xticklabels([str(b) for b in blocks] + ["camera\nhead"])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("VGGT block")

    ax.axvspan(13.7, 16.3, color="#f0efec", zorder=0, linewidth=0)
    ax.text(15, 0.55, "readable,\nnot used", ha="center", va="center", color=INK2, fontsize=9)

    ax.plot(blocks, r2, color=BLUE, lw=2, marker="o", label="probe R² for camera (x, y)", **RING)
    ax.plot(blocks, tr, color=ORANGE, lw=2, marker="o", label="patch whole token: transfer", **RING)
    ax.plot([23.5], [head], marker="o", color=ORANGE, **RING)

    i15 = blocks.index(15)
    ax.annotate(f"{r2[i15]:.2f}", (15, r2[i15]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.annotate(f"{tr[i15]:.2f}", (15, tr[i15]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.annotate(f"{tr[-1]:.2f}", (21, tr[-1]), xytext=(-8, 2), textcoords="offset points", ha="right", fontsize=9)
    ax.annotate(f"{head:.2f}", (23.5, head), xytext=(0, -17), textcoords="offset points", ha="center", fontsize=9)

    ax.legend(loc="center left", bbox_to_anchor=(0.06, 0.4), frameon=False, fontsize=9, handlelength=1.6,
              labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_depth.png")
    plt.close(fig)


def fig2_subspaces(block="21"):
    d = json.load(open(RES / "vggt_patch_subspaces.json"))["blocks"][block]
    sh = d["share"]
    rows = [  # label, share, transfer, colour
        ("probe directions, 2-d", sh["position 2"], d["position 2"]["transfer_mean"], ORANGE),
        ("probe directions, 10-d", sh["position 10"], d["position 10"]["transfer_mean"], ORANGE),
        ("random, 10-d", sh["random 10"], d["random 10"]["transfer_mean"], GRAY),
        ("position PCA, 10-d", sh["position PCA 10"], d["position PCA 10"]["transfer_mean"], BLUE),
        ("whole token, 1,024-d", 1.0, d["full"]["transfer_mean"], GRAY),
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.9), dpi=200, gridspec_kw={"wspace": 0.12})
    pct = lambda v: f"{100 * v:.2f}%" if v < 0.01 else f"{100 * v:.1f}%" if v < 0.1 else f"{100 * v:.0f}%"
    bar_panel(a1, [(r[0], r[1], r[3]) for r in rows], 1.18, [0, 0.5, 1], ["0", "50%", "100%"], pct,
              "share of the difference between frames")
    bar_panel(a2, [(r[0], r[2], r[3]) for r in rows], 1.18, [0, 0.5, 1], ["0", "0.5", "1"],
              lambda v: f"{max(v, 0):.2f}", "transfer to the donor's position", labels=False)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.86, bottom=0.1)
    fig.savefig(OUT / "fig2_subspaces.png")
    plt.close(fig)


def fig3_neurons():
    c = json.load(open(RES / "vggt_steer.json"))["a"]["conditions"]
    rows = [
        ("whole residual token", c["block 21: residual token -> mean at L"]["transfer_mean"], BLUE),
        ("heading-invariant place neurons", c["blocks 21: invariant, all 1981"]["transfer_mean"], GRAY),
        ("heading-specific place neurons", c["blocks 21: heading-specific, top 1377"]["transfer_mean"], GRAY),
        ("random neurons", c["blocks 21: random 1981"]["transfer_mean"], GRAY),
        ("all 4,096 neurons", c["blocks 21: all 4096"]["transfer_mean"], GRAY),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=200)
    bar_panel(ax, rows, 0.8, [0, 0.25, 0.5, 0.75], ["0", "0.25", "0.5", "0.75"],
              lambda v: f"{v:.2f}" if abs(v) >= 0.05 else f"{v:.3f}".replace("-0.000", "0.000"),
              "set to their average at location L (block 21): transfer toward L")
    fig.subplots_adjust(left=0.36, right=0.97, top=0.86, bottom=0.1)
    fig.savefig(OUT / "fig3_neurons.png")
    plt.close(fig)


def fig4_joystick(block="block 21"):
    c = json.load(open(RES / "vggt_steer.json"))["c"][block]
    east = sorted((r["alpha"], r["dx_mean"]) for r in c["rows"] if r["dir"] == "east")
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=200)
    tidy(ax)
    ax.plot([-3.2, 3.2], [-3.2, 3.2], color=AXIS, lw=1)
    ax.text(1.0, 2.2, "gain 1", color=MUTED, fontsize=9, ha="right", va="center")
    ax.plot(*zip(*east), color=BLUE, lw=2, marker="o", **RING)
    ax.annotate(f"gain {c['east_gain_m_per_m']:.2f}", (3, east[-1][1]), xytext=(0, 10),
                textcoords="offset points", ha="right", fontsize=9)
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.3, 3.3)
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([-3, 0, 3])
    ax.set_xlabel("push east (metres' worth of token change)")
    ax.set_ylabel("predicted camera moves east (m)")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_joystick.png")
    plt.close(fig)


def fig5_lambda(block="21"):
    d = json.load(open(RES / "vggt_patch_probes.json"))["blocks"][block]
    lams = [k for k in d["readout_r2"] if k.startswith("ridge")]
    ridge = [(d["readout_r2"][k], d[k]["transfer_mean"]) for k in lams]
    pat = (d["readout_r2"]["pattern"], d["pattern"]["transfer_mean"])
    pca = (d["readout_r2"]["position PCA 10"], d["position PCA 10"]["transfer_mean"])

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=200)
    tidy(ax)
    ax.set_xlim(0.65, 1.005)
    ax.set_ylim(-0.03, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("R² of camera (x, y) read from the written directions")
    ax.set_ylabel("transfer to the donor's position")

    ax.plot(*zip(*ridge), color=BLUE, lw=2, marker="o", label="ridge probe, λ = 0.1 to 10⁶ (2-d)", **RING)
    ax.plot(*pat, marker="o", color=ORANGE, linestyle="none", label="Haufe pattern (2-d)", **RING)
    ax.plot(*pca, marker="o", color=GRAY, linestyle="none", label="position PCA (10-d)", **RING)

    best = max(ridge, key=lambda p: p[0])
    ax.annotate(f"best-R² probe: {best[1]:.2f}", best, xytext=(-34, 0), textcoords="offset points",
                ha="right", va="center", fontsize=8.5, color=INK2)
    ax.annotate("λ = 10⁶", ridge[-1], xytext=(0, 10), textcoords="offset points", ha="center",
                fontsize=8.5, color=INK2)
    ax.annotate(f"{pca[1]:.2f}", pca, xytext=(-10, 0), textcoords="offset points", ha="right",
                va="center", fontsize=9, color=INK)
    ax.annotate(f"{pat[1]:.2f}", pat, xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, color=INK)

    ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=1.6, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_lambda.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    fig1_depth()
    fig2_subspaces()
    fig3_neurons()
    fig4_joystick()
    fig5_lambda()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
