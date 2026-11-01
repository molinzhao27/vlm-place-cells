"""What are the ten directions of VGGT's pose subspace? Unregistered and descriptive; no model is run.

    PYTHONPATH=. .venv-mac/bin/python scripts/vggt_token_pca.py          # -> results/vggt_token_pca.json
    .venv-mac/bin/python scripts/vggt_token_pca.py --fig                 # -> writeup/paper/figs_r3/fig_tokenpca.pdf, preview png

Uses experiment 27's recordings (data/subspace_controls/{raw,resid}.npz, copied from the PC): the walk frame's camera
token after global blocks 19 and 21 in single mode, room A. The directions are experiment 27's own ordinary PCA (Qord30,
fit with no labels on the 2,048 fit frames, the subspace whose patching gives the transfer in the paper). Everything is
scored on frames the PCA did not see: the 4,096 other even-walk frames (6-fold cross-validation grouped by walk), then
the 256 odd-walk targets and donors. Bootstrap CIs resample the 24 even walks.

The features are functions of the true pose. With (x, y) measured from the room center, f = (cos t, sin t) the facing
direction and r = (sin t, -cos t) the camera's right:
  heading        cos t, sin t
  room frame     x, y
  camera frame   (x, y).f and (x, y).r: position in the camera's axes. This is the form of the translation VGGT's camera
                 head outputs (the extrinsic t = -R C; vggt/utils/pose_enc.py), up to the origin, which the heading terms absorb
  mirror         x cos t - y sin t, x sin t + y cos t: the same four products with the rotation reversed, as a control
  second order   cos 2t, sin 2t, x^2 + y^2
The same features are scored on raw pixels (32 x 32 RGB, analyze_probes.pixel_feats) with their own PCA on the fit frames.
A family's share is the variance it explains in cross-validation, over the total held-out variance of the token.

Neurons (a consistency check, not independent evidence, since a unit that reads room position is heading-invariant by
construction): each fc2-input neuron's share of held-out variance in each pair of directions, by experiment 23's groups
(data/units_subspace/units.npz). Directions 3-6 go to the room or camera pair by which family explains them better; at
block 19 that is not 3-4 and 5-6. CI: bootstrap over neurons.
"""
import json
import sys

import numpy as np
from sklearn.linear_model import Ridge

OUT = "results/vggt_token_pca.json"
N_FOLD, N_BOOT = 6, 2000
BLOCKS = (21, 19)
FAMILIES = ("heading", "room", "camera", "mirror", "second")
SECTORS = (("east", 0.0), ("north", np.pi / 2), ("west", np.pi), ("south", 3 * np.pi / 2))


def features(x, y, t):
    x, y = x - 5.0, y - 5.0
    c, s = np.cos(t), np.sin(t)
    return {"heading": np.c_[c, s], "room": np.c_[x, y], "camera": np.c_[x * c + y * s, x * s - y * c],
            "mirror": np.c_[x * c - y * s, x * s + y * c], "second": np.c_[np.cos(2 * t), np.sin(2 * t), x ** 2 + y ** 2]}


def predict(D, Y, fold):
    P = np.zeros_like(Y)
    for f in range(N_FOLD):
        te = fold == f
        P[te] = Ridge(1e-6).fit(D[~te], Y[~te]).predict(D[te])
    return P


def explained(Y, P, rows=None):
    Y, P = (Y, P) if rows is None else (Y[rows], P[rows])
    m = Y.mean(0)
    return ((Y - m) ** 2).sum(0) - ((Y - P) ** 2).sum(0), ((Y - m) ** 2).sum(0)


def boot(fn, walk, rng):
    w = np.unique(walk)
    rows = {k: np.flatnonzero(walk == k) for k in w}
    return [fn(np.concatenate([rows[k] for k in rng.choice(w, len(w))])) for _ in range(N_BOOT)]


def score(X_fit, X_test, Q, F, walk, seed=0):
    """Family shares and per-PC R^2 for held-out frames. Q: orthonormal directions by variance. Variance outside the
    directions kept here stays in the denominator of the token-wide shares."""
    mu = X_fit.mean(0)
    X = X_test - mu
    Y = X @ Q
    total = ((X - X.mean(0)) ** 2).sum()
    w = np.unique(walk)
    fold = np.searchsorted(w, walk) % N_FOLD
    res = {"pc_var": (((Y - Y.mean(0)) ** 2).sum(0) / total).tolist()}
    designs = {**{k: F[k] for k in FAMILIES}, "heading+room+camera": np.c_[F["heading"], F["room"], F["camera"]],
               "heading+room+camera+second": np.c_[F["heading"], F["room"], F["camera"], F["second"]],
               "heading+room+camera+mirror": np.c_[F["heading"], F["room"], F["camera"], F["mirror"]]}
    preds = {k: predict(D, Y, fold) for k, D in designs.items()}
    res["r2_by_pc"], res["share_token"], res["share_top10"] = {}, {}, {}
    rng = np.random.default_rng(seed)
    for k, P in preds.items():
        e, t = explained(Y, P)
        res["r2_by_pc"][k] = (e / t)[:12].tolist()
        res["share_token"][k] = float(e.sum() / total)
        res["share_top10"][k] = float(e[:10].sum() / t[:10].sum())
    Xc = ((X - X.mean(0)) ** 2).sum(1)

    def one(rows):
        out = {}
        tot = Xc[rows].sum()
        for k, P in preds.items():
            e, t = explained(Y, P, rows)
            out[k] = e.sum() / tot, e[:10].sum() / t[:10].sum()
        return out
    draws = boot(one, walk, rng)
    res["share_token_ci95"] = {k: np.percentile([d[k][0] for d in draws], [2.5, 97.5]).tolist() for k in preds}
    res["share_top10_ci95"] = {k: np.percentile([d[k][1] for d in draws], [2.5, 97.5]).tolist() for k in preds}
    return res


def maps(Y, F, x, y, t):
    """Mean score per 1 m bin and heading sector, along the direction of a PC pair that best follows one feature."""
    out = {"sectors": [s[0] for s in SECTORS]}
    for key, pair, target in (("camera", (2, 3), F["camera"][:, 0]), ("room", (4, 5), F["room"][:, 0])):
        P = Y[:, list(pair)]
        w = np.linalg.lstsq(P, target - target.mean(), rcond=None)[0]
        w /= np.linalg.norm(w)
        s = P @ w
        grid = np.full((len(SECTORS), 10, 10), np.nan)
        for i, (_, c) in enumerate(SECTORS):
            a = np.abs((t - c + np.pi) % (2 * np.pi) - np.pi) <= np.pi / 4
            ix, iy = np.clip(x[a].astype(int), 0, 9), np.clip(y[a].astype(int), 0, 9)
            n = np.bincount(ix * 10 + iy, minlength=100).reshape(10, 10)
            v = np.bincount(ix * 10 + iy, weights=s[a], minlength=100).reshape(10, 10)
            grid[i] = np.where(n >= 3, v / n.clip(1), np.nan)
        out[key] = {"map_x_by_y": np.round(grid, 3).tolist(), "direction": w.tolist(), "corr_with_feature": float(np.corrcoef(s, target)[0, 1])}
    return out


def neurons(b, Y, r, rest):
    """Share of each neuron's held-out variance in each pair of directions, by experiment 23's groups."""
    g = np.load("data/units_subspace/units.npz")[f"vggt-single_L{b}_group"]
    A = np.load(f"data/acts/vggt-single/L{b}.npy", mmap_mode="r")[rest].astype(np.float32)
    A = A - A.mean(0)
    var = (A ** 2).sum(0)
    ok = var > 0
    room_like = [i for i in range(2, 6) if r["r2_by_pc"]["room"][i] > r["r2_by_pc"]["camera"][i]]
    pairs = {"heading": [0, 1], "room": room_like, "camera": [i for i in range(2, 6) if i not in room_like]}
    share = {}
    for k, cols in pairs.items():
        P = Y[:, cols] - Y[:, cols].mean(0)
        share[k] = ((np.linalg.qr(P)[0].T @ A) ** 2).sum(0) / np.where(ok, var, 1)
    out = {"pairs": pairs}
    for name, gv in (("invariant", 2), ("noninvariant_place", 1), ("nonplace", 0)):
        k = (g == gv) & ok
        out[name] = {"n": int(k.sum()), **{f: float(np.median(v[k])) for f, v in share.items()},
                     "room_of_room_plus_camera": float(np.median(share["room"][k] / (share["room"][k] + share["camera"][k] + 1e-9)))}
    a, b_ = share["room"][(g == 2) & ok], share["room"][(g == 1) & ok]
    rng = np.random.default_rng(0)
    d = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b_, len(b_))) for _ in range(N_BOOT)]
    out["room_invariant_minus_noninvariant"] = {"diff": float(np.median(a) - np.median(b_)), "ci95": np.percentile(d, [2.5, 97.5]).tolist()}
    return out


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    INK, INK2, GRID, AXIS, MUTED = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#898781"     # make_figs.py's style
    C_R, C_C, C_H, GRAY, GRAY2 = "#1baf7a", "#2a78d6", "#eb6834", "#9a988f", "#cfcdc4"
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 7.5, "axes.titlesize": 7.5,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "text.color": INK,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False})

    def tidy(ax, grid="y"):
        ax.spines[["top", "right"]].set_visible(False)
        if grid:
            ax.grid(axis=grid, color=GRID, lw=0.6)
            ax.set_axisbelow(True)

    def panel(ax, s, x, y=1.04):
        ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom")

    R = json.load(open(OUT))
    fig = plt.figure(figsize=(6.5, 3.35))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.75, 1], width_ratios=[1.3, 0.8, 1.4], hspace=0.6, wspace=0.42, left=0.07, right=0.995, top=0.95, bottom=0.03)
    a = fig.add_subplot(gs[0, 0])
    a.bar(range(1, 13), np.array(R["blocks"]["21"]["pc_var"][:12]) * 100, color=[C_H] * 2 + [C_C] * 2 + [C_R] * 2 + [GRAY] * 4 + [GRAY2] * 2, width=0.78)
    for x, y, t, c in ((2.75, 25.5, "heading", C_H), (3.5, 12.4, "camera-frame\nposition", C_C), (5.75, 6.8, "room-frame\nposition", C_R), (8.2, 2.6, "second order", INK2)):
        a.text(x, y, t, color=c, ha="left", va="bottom", fontsize=6.8)
    a.set_xticks(range(1, 13))
    a.set_xlim(0.3, 12.7)
    a.set_xlabel("PCA direction (fit with no labels)")
    a.set_ylabel("share of token variance (%)")
    tidy(a)
    panel(a, "a", -0.17)
    b = fig.add_subplot(gs[0, 1])
    th, p1, p2 = (np.array(R["ring"][k]) for k in ("theta", "pc1", "pc2"))
    b.scatter(p1, p2, c=th, cmap="twilight", s=1.6, lw=0, vmin=0, vmax=2 * np.pi, rasterized=True)
    rad = 1.17 * np.hypot(p1, p2).max()
    for lab, c in (("E", 0), ("N", np.pi / 2), ("W", np.pi), ("S", 1.5 * np.pi)):
        s = np.abs((th - c + np.pi) % (2 * np.pi) - np.pi) < 0.25
        ang = np.arctan2(p2[s].mean(), p1[s].mean())
        b.text(rad * np.cos(ang), rad * np.sin(ang), lab, ha="center", va="center", fontsize=7, color=INK2)
    b.set_aspect("equal")
    b.set_xlim(-1.3 * rad / 1.17, 1.3 * rad / 1.17)
    b.set_ylim(-1.3 * rad / 1.17, 1.3 * rad / 1.17)
    b.set_xticks([])
    b.set_yticks([])
    b.set_xlabel("direction 1")
    b.set_ylabel("direction 2")
    tidy(b, None)
    panel(b, "b", -0.14)
    d = fig.add_subplot(gs[0, 2])
    groups = (("block 21", R["blocks"]["21"]), ("block 19", R["blocks"]["19"]), ("raw pixels", R["pixels"]))
    w = 0.2
    for i, (_, r) in enumerate(groups):
        for j, (k, c, lab) in enumerate((("heading", C_H, "heading"), ("room", C_R, "room frame"), ("camera", C_C, "camera frame"), ("mirror", GRAY, "mirror control"))):
            y, (lo, hi) = r["share_token"][k] * 100, np.array(r["share_token_ci95"][k]) * 100
            d.bar(i + (j - 1.5) * w, max(y, 0), w * 0.92, color=c, label=lab if i == 0 else None)
            d.plot([i + (j - 1.5) * w] * 2, [max(lo, 0), max(hi, 0)], color=INK2, lw=0.7)
        y = r["share_token"]["heading+room+camera"] * 100
        d.plot([i - 0.42, i + 0.42], [y, y], color=INK, lw=1.1, label="all three" if i == 0 else None)
    d.set_xticks(range(3))
    d.set_xticklabels([g[0] for g in groups])
    d.set_ylabel("held-out variance explained (%)")
    d.set_ylim(0, 100)
    tidy(d)
    d.legend(loc="upper right", bbox_to_anchor=(1.02, 1.03), fontsize=6.2, handlelength=1.2, labelspacing=0.2)
    panel(d, "c", -0.17)
    sub = GridSpecFromSubplotSpec(1, 9, subplot_spec=gs[1, :], width_ratios=[1, 1, 1, 1, 0.28, 1, 1, 1, 1], wspace=0.08)
    for off, key, name, colr in ((0, "camera", "camera-frame direction", C_C), (5, "room", "room-frame direction", C_R)):
        G = np.array(R["maps"][key]["map_x_by_y"], float)
        lim = np.nanpercentile(np.abs(G), 99)
        for i, lab in enumerate(("E", "N", "W", "S")):
            ax = fig.add_subplot(sub[0, off + i])
            ax.imshow(G[i].T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=(0, 10, 0, 10), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("facing " + lab, fontsize=6.6, pad=1.5, color=INK2)
            for sp in ax.spines.values():
                sp.set_edgecolor(AXIS)
            if i == 0:
                ax.text(0, 1.3, name, transform=ax.transAxes, fontsize=7, color=colr, va="bottom", ha="left")
                if off == 0:
                    panel(ax, "d", -0.22, 1.25)
    fig.savefig("writeup/paper/figs_r3/fig_tokenpca.pdf")
    fig.savefig("writeup/paper/kit/fig_previews/fig_tokenpca.png", dpi=170)


def main():
    if "--fig" in sys.argv:
        return figure()
    from spatialcells.analyze_probes import pixel_feats
    Z, R = np.load("data/subspace_controls/resid.npz"), np.load("data/subspace_controls/raw.npz")
    m = dict(np.load("data/walks/meta.npz"))
    fit, rest = R["fit"], R["rest"]
    odd = np.r_[R["targets"], R["donors"]]
    pose = lambda i: (m["x"][i], m["y"][i], m["theta"][i])
    F, Fo = features(*pose(rest)), features(*pose(odd))
    walk = m["walk"][rest]
    res = {"about": __doc__.split("\n")[0], "n_fit": len(fit), "n_test": len(rest), "n_walks": len(np.unique(walk)), "n_odd": len(odd),
           "families": list(FAMILIES), "blocks": {}}
    for b in BLOCKS:
        Xf, Xr, Xo = (Z[f"{k}_{b}"].astype(np.float64) for k in ("fit", "rest", "t"))
        Xo = np.r_[Xo, Z[f"d_{b}"].astype(np.float64)]
        Q = R[f"b{b}/Qord30"].astype(np.float64)
        r = score(Xf, Xr, Q, F, walk)
        Y = (Xr - Xf.mean(0)) @ Q
        Yo = (Xo - Xf.mean(0)) @ Q
        D, Do = np.c_[F["heading"], F["room"], F["camera"]], np.c_[Fo["heading"], Fo["room"], Fo["camera"]]
        Po = Ridge(1e-6).fit(D, Y[:, :10]).predict(Do)
        e, t = explained(Yo[:, :10], Po)
        r["odd_walks"] = {"r2_by_pc": (e / t).tolist(), "share_top10": float(e.sum() / t.sum())}
        if b == 21:
            res["maps"] = maps(Y, F, *pose(rest))
            ring = np.random.default_rng(0).permutation(len(rest))[:1500]
            res["ring"] = {"pc1": np.round(Y[ring, 0], 2).tolist(), "pc2": np.round(Y[ring, 1], 2).tolist(), "theta": np.round(pose(rest)[2][ring], 3).tolist()}
        r["neurons"] = neurons(b, Y, r, rest)
        res["blocks"][str(b)] = r
    frames = np.load("data/walks/frames.npy", mmap_mode="r")
    px = pixel_feats(frames).astype(np.float64)
    Xf, Xr = px[fit], px[rest]
    Q = np.linalg.svd(Xf - Xf.mean(0), full_matrices=False)[2][:30].T
    res["pixels"] = {"dim": px.shape[1], **score(Xf, Xr, Q, F, walk)}
    json.dump(res, open(OUT, "w"))
    for name, r in [("block 21", res["blocks"]["21"]), ("block 19", res["blocks"]["19"]), ("pixels", res["pixels"])]:
        print(f"\n{name}: top-10 variance {sum(r['pc_var'][:10]):.3f}; per PC " + " ".join(f"{v:.3f}" for v in r["pc_var"][:10]))
        for k in r["share_token"]:
            lo, hi = r["share_token_ci95"][k]
            lo2, hi2 = r["share_top10_ci95"][k]
            pcs = " ".join(f"{v:5.2f}" for v in r["r2_by_pc"][k][:10])
            print(f"  {k:28s} token {r['share_token'][k]:6.3f} [{lo:6.3f},{hi:6.3f}]  top-10 {r['share_top10'][k]:6.3f} [{lo2:6.3f},{hi2:6.3f}]  R2 {pcs}")
        if "odd_walks" in r:
            print("  odd walks, R2 by PC: " + " ".join(f"{v:5.2f}" for v in r["odd_walks"]["r2_by_pc"]) + f"  share {r['odd_walks']['share_top10']:.3f}")
        if "neurons" in r:
            print("  neurons: " + json.dumps(r["neurons"]))


if __name__ == "__main__":
    main()
