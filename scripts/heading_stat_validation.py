"""Experiment 21 (docs/experiments.md): the paper's strict heading-invariance statistic on synthetic
units whose answer is known, and a gain index beside it.

    .venv-mac/bin/python scripts/heading_stat_validation.py        # synthetic sweep + the real units on this machine
    python scripts/heading_stat_validation.py --skip-synthetic --real qwen8b-ft-walk qwen8b-ft-single \
        --rnn base:1 s093:3                                          # on vast-1, where those activations live

With --skip-synthetic, the real units' comparison with synthetic invariant units (their G > 1 rate per
stability bin) is read from the --out file's synthetic part, so copy the Mac's results file there first.

Synthetic units are read out along room A's 48 walks (data/walks/meta.npz), so occupancy and the mix
of headings per bin are the real ones. Four kinds: (a) invariant place units, (b) gain-modulated place
units (rate at the preferred heading R times the rate at the opposite one), (c) conjunctive
place-by-heading units (fire only within a heading window of width w), (d) view-dependent units (fire
when a wall point or a pillar is in the 90-degree field of view; the place they fire at moves with
heading). Noise is AR(1) within each walk, k times the SD of the unit's noise-free signal.

Every population goes through analyze.analyze_layer(..., quadrant=True) unchanged. The per-unit
opposite-sector score is recomputed with analyze.py's own functions exactly as analyze_layer does,
and the script stops unless that reproduces analyze_layer's frac_place_heading_invariant_quadrant.

Gain index G: for each opposite pair (N vs S, E vs W), the field amplitude in each sector map (mean over
the in-field bins of the all-heading map minus mean over the other bins both sectors visit), log2 of
the ratio of the two amplitudes (each floored at 5% of the all-heading amplitude); G is the length of
the vector of the two pair indices. G = 1 is a twofold change between opposite headings.

Writes results/heading_stat_validation.json (merging into an existing file, so a later --real pass
elsewhere adds to it).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from spatialcells import analyze                                  # noqa: E402
from spatialcells.room import HFOV, L as ROOM_L, PILLARS          # noqa: E402

THRESH = 0.5                       # \HeadInvThresh: a place unit counts as invariant above this
OUT = "results/heading_stat_validation.json"
K_GRID = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
RHOS = (0.5, 0.0, 0.9)             # first is primary
GAIN_R = (1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0)
CONJ_W = (60, 90, 120, 180, 270)   # 270 is a boundary case outside the decision rule
RULE_W = (60, 90, 120, 180)
N_A, N_B, N_C, N_D = 300, 100, 100, 200
WINDOW_HALF = 0.05
MIN_N = 30
BINS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001)]
FLOOR = 0.05
dev = analyze.dev


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- synthetic units (noise-free)
def place_fields(pos, rng, n):
    """n fields: one Gaussian bump (70%) or two, SD 0.4-0.9 m, centres drawn from the walk positions."""
    F = np.zeros((len(pos), n))
    for u in range(n):
        for _ in range(1 if rng.random() < 0.7 else 2):
            c, s = pos[rng.integers(len(pos))], rng.uniform(0.4, 0.9)
            F[:, u] += np.exp(-((pos - c) ** 2).sum(1) / (2 * s * s))
    return F


def line_of_sight(pos, q, skip=None):
    """True where the segment from each position to q misses every pillar (except pillar index skip)."""
    ok = np.ones(len(pos), bool)
    d = q[None] - pos
    dd = (d ** 2).sum(1).clip(1e-9)
    for j, (px, py, pr, _, _) in enumerate(PILLARS):
        if j == skip:
            continue
        c = np.array([px, py])
        t = np.clip(((c[None] - pos) * d).sum(1) / dd, 0, 1)
        ok &= np.hypot(*(pos + t[:, None] * d - c).T) >= pr
    return ok


def view_unit(pos, th, q, rng, skip=None):
    """Fires when q is inside the 90-degree field of view and not hidden by a pillar, with a Gaussian
    preference for where in the image q appears and a response that grows as q gets closer."""
    bearing = np.arctan2(q[1] - pos[:, 1], q[0] - pos[:, 0])
    delta = wrap(bearing - th)                      # angle of q from the image centre (positive = left)
    off, width, d0 = np.deg2rad(rng.uniform(-30, 30)), np.deg2rad(rng.uniform(8, 25)), rng.uniform(0.75, 2.5)
    dist = np.hypot(q[0] - pos[:, 0], q[1] - pos[:, 1])
    r = np.exp(-(delta - off) ** 2 / (2 * width ** 2)) * (np.abs(delta) <= HFOV / 2)
    return r * np.minimum(1.0, d0 / dist.clip(1e-3)) * line_of_sight(pos, q, skip)


def wall_point(rng):
    s = rng.uniform(0, 4 * ROOM_L)
    side, t = int(s // ROOM_L), s % ROOM_L
    return np.array([(t, ROOM_L), (ROOM_L, t), (t, 0.0), (0.0, t)][side])   # N, E, S, W walls


def build_units(meta, seed=21):
    """Noise-free signals S (N, U) and a label per unit: (kind, cell name, parameter)."""
    rng = np.random.default_rng(seed)
    pos = np.stack([meta["x"], meta["y"]], 1)
    th = meta["theta"]
    cols, labels = [], []
    F = place_fields(pos, rng, N_A)
    cols.append(F)
    labels += [("a", "a", None)] * N_A
    for R in GAIN_R:
        F = place_fields(pos, rng, N_B)
        phi = rng.uniform(0, 2 * np.pi, N_B)
        cols.append(F * np.exp(0.5 * np.log(R) * np.cos(th[:, None] - phi[None])))
        labels += [("b", f"b:R={R:g}", R)] * N_B
    for w in CONJ_W:
        F = place_fields(pos, rng, N_C)
        phi = rng.uniform(0, 2 * np.pi, N_C)
        cols.append(F * (np.abs(wrap(th[:, None] - phi[None])) <= np.deg2rad(w) / 2))
        labels += [("c", f"c:w={w}", w)] * N_C
    cols.append(np.stack([view_unit(pos, th, wall_point(rng), rng) for _ in range(N_D)], 1))
    labels += [("d", "d:wall", "wall")] * N_D
    pill = []
    for i in range(N_D):
        j = i % len(PILLARS)
        pill.append(view_unit(pos, th, np.array(PILLARS[j][:2]), rng, skip=j))
    cols.append(np.stack(pill, 1))
    labels += [("d", "d:pillar", "pillar")] * N_D
    S = np.concatenate(cols, 1)
    S = S + rng.uniform(0, 0.1, S.shape[1]) * S.max(0)              # constant offset, 0-10% of the peak
    return S, labels


N_E = 100


def axis_units(meta, seed=2101):
    """Amendment 1, kind (e): one field (f1) on the north-south axis and another (f2) on the east-west axis.
    hard: f1 in the N and S sectors (|sin| > |cos|), f2 in E and W; smooth: sin^2 f1 + cos^2 f2."""
    rng = np.random.default_rng(seed)
    pos = np.stack([meta["x"], meta["y"]], 1)
    th = meta["theta"]
    ns = (np.abs(np.sin(th)) > np.abs(np.cos(th)))[:, None]
    F1, F2 = place_fields(pos, rng, N_E), place_fields(pos, rng, N_E)
    hard = np.where(ns, F1, F2)
    F1, F2 = place_fields(pos, rng, N_E), place_fields(pos, rng, N_E)
    smooth = np.sin(th)[:, None] ** 2 * F1 + np.cos(th)[:, None] ** 2 * F2
    S = np.concatenate([hard, smooth], 1)
    S = S + rng.uniform(0, 0.1, S.shape[1]) * S.max(0)
    return S, [("e", "e:hard", "hard")] * N_E + [("e", "e:smooth", "smooth")] * N_E


def ar1_noise(meta, U, rho, seed):
    """Unit-variance AR(1) noise within each walk (frames walk-major, step-minor), shape (N, U)."""
    rng = np.random.default_rng(seed)
    W, S = int(meta["walk"].max()) + 1, int(meta["step"].max()) + 1
    e = np.empty((W, S, U), np.float32)
    e[:, 0] = rng.standard_normal((W, U))
    c = math.sqrt(1 - rho ** 2)
    for s in range(1, S):
        e[:, s] = rho * e[:, s - 1] + c * rng.standard_normal((W, U))
    return e.reshape(W * S, U)


# ---------------------------------------------------------------- scoring with analyze.py
def bins_of(meta):
    """analyze.spatial_bins; vast-1's earlier analyze.py (the one that wrote summary_quadrant.json) bins
    inline in analyze_layer with this same expression (identical in room A, whose grid corner is 0, 0)."""
    if hasattr(analyze, "spatial_bins"):
        return analyze.spatial_bins(meta)
    ix = np.clip((meta["x"] / analyze.L * analyze.NB).astype(int), 0, analyze.NB - 1)
    iy = np.clip((meta["y"] / analyze.L * analyze.NB).astype(int), 0, analyze.NB - 1)
    return torch.from_numpy(iy * analyze.NB + ix).to(dev)


# the six sector pairs of amendment 1 (sectors 0..3 = E, N, W, S as in analyze.sectors)
PAIRS6 = {"N-S": (1, 3), "E-W": (0, 2), "N-E": (1, 0), "N-W": (1, 2), "S-E": (3, 0), "S-W": (3, 2)}


def score(X, meta):
    """analyze_layer unchanged, plus each pole's opposite-sector score (recomputed as analyze_layer's
    quadrant block does), its 180-degree score, its gain index and (amendment 1) the correlation of
    every pair of the four sector maps."""
    stats, pu = analyze.analyze_layer(X, meta, np.random.default_rng(0), quadrant=True)
    R = torch.cat([X.clamp_min(0), (-X).clamp_min(0)], 1)
    active = (R > 0).float().mean(0) >= 0.01
    assert torch.equal(torch.nonzero(active).squeeze(1), pu["pole"])
    R = R[:, active]
    bins = bins_of(meta)
    sec = analyze.sectors(meta["theta"])
    smaps = [analyze.rate_maps(R, bins, torch.from_numpy(sec == i).to(dev)) for i in range(4)]
    del R
    corr, pairs = {}, []
    for name, (i, j) in PAIRS6.items():
        (ma, oa), (mb, ob) = smaps[i], smaps[j]
        both = (oa >= analyze.MIN_OCC) & (ob >= analyze.MIN_OCC)
        corr[name] = analyze.map_corr(ma, mb, both)
        if name in ("N-S", "E-W"):                        # as in analyze_layer's quadrant block
            pairs.append((ma, mb, both))
    inv_q = (corr["N-S"] + corr["E-W"]) / 2
    frac = (pu["place"] & (inv_q > THRESH)).float().mean().item()
    if frac != stats["frac_place_heading_invariant_quadrant"]:
        sys.exit(f"per-unit score does not reproduce analyze_layer: {frac} vs "
                 f"{stats['frac_place_heading_invariant_quadrant']}")
    G, Graw, gpairs = gain_index(pu["maps"], pu["valid"], pairs)
    c6 = torch.stack([corr[k] for k in PAIRS6], 1)
    del pairs, smaps
    per = {"pole": pu["pole"].cpu().numpy(), "place": pu["place"].cpu().numpy(), "stab": pu["stab"].cpu().numpy(),
           "inv_q": inv_q.cpu().numpy(), "hi": pu["hi"].cpu().numpy(), "G": G, "Graw": Graw, "g_pairs": gpairs,
           "ns": corr["N-S"].cpu().numpy(), "ew": corr["E-W"].cpu().numpy(),
           "min6": c6.min(1).values.cpu().numpy(), "c6": c6.cpu().numpy()}
    return stats, per


def gain_index(maps, valid, pairs):
    """G (amplitude, baseline-subtracted) and Graw (raw in-field rate), log2 units, per pole."""
    inf = torch.tensor(float("inf"), device=maps.device)
    lo = torch.where(valid, maps, inf).amin((1, 2))
    hi = torch.where(valid, maps, -inf).amax((1, 2))
    infield = (maps > (lo + 0.5 * (hi - lo))[:, None, None]) & valid
    g, graw = [], []
    for ma, mb, both in pairs:
        Fm, Om = (infield & both).float(), (~infield & both).float()
        nF, nO = Fm.sum((1, 2)), Om.sum((1, 2))
        mean = lambda m, M, n: (m * M).sum((1, 2)) / n.clamp_min(1)
        ra, rb, rall = mean(ma, Fm, nF), mean(mb, Fm, nF), mean(maps, Fm, nF)
        aa, ab, aall = ra - mean(ma, Om, nO), rb - mean(mb, Om, nO), rall - mean(maps, Om, nO)
        eps = FLOOR * aall
        gi = torch.log2(torch.maximum(aa, eps) / torch.maximum(ab, eps))
        gi[(nF == 0) | (nO == 0) | (aall <= 0)] = float("nan")
        epsr = FLOOR * rall
        gr = torch.log2(torch.maximum(ra, epsr) / torch.maximum(rb, epsr))
        gr[(nF == 0) | (rall <= 0)] = float("nan")
        g.append(gi)
        graw.append(gr)
    G = torch.sqrt(g[0] ** 2 + g[1] ** 2)
    Graw = torch.sqrt(graw[0] ** 2 + graw[1] ** 2)
    return G.cpu().numpy(), Graw.cpu().numpy(), torch.stack(g, 1).cpu().numpy()


# ---------------------------------------------------------------- summaries
def med(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def summary(t, m, n_units=None):
    """t: per-unit table (dict of arrays) of selected positive poles; m: mask over it."""
    s = {"n_selected": int(m.sum())}
    if n_units is not None:
        s["n_units"] = n_units
        s["frac_selected"] = s["n_selected"] / n_units
    if m.sum():
        G = t["G"][m]
        s.update(frac_invariant=float((t["inv_q"][m] > THRESH).mean()),
                 frac_invariant_180=float((t["hi"][m] > THRESH).mean()),
                 inv_q_median=med(t["inv_q"][m]), stab_median=med(t["stab"][m]),
                 inv_over_stab_median=med(t["inv_q"][m] / np.maximum(t["stab"][m], 1e-3)),
                 G_median=med(G), frac_G_above_1=float((G[np.isfinite(G)] > 1).mean()) if np.isfinite(G).any() else None,
                 Graw_median=med(t["Graw"][m]))
    return s


def cells_of(labels):
    names = list(dict.fromkeys(c for _, c, _ in labels))
    return names, np.array([c for _, c, _ in labels]), np.array([k for k, _, _ in labels])


def groups(cell, kind):
    """Pooled groups used by the decision rule (and kind (b) pooled for reference)."""
    rule_c = np.isin(cell, [f"c:w={w}" for w in RULE_W])
    return {"a": kind == "a", "b (all R)": kind == "b", "c (w 60-180)": rule_c, "d (wall + pillar)": kind == "d"}


def rule_check(t, lo, hi):
    w = (t["stab"] >= lo) & (t["stab"] <= hi)
    g = groups(t["cell"], t["kind"])
    out = {}
    for name, want_inv in (("a", True), ("c (w 60-180)", False), ("d (wall + pillar)", False)):
        m = w & g[name]
        n = int(m.sum())
        correct = float(((t["inv_q"][m] > THRESH) == want_inv).mean()) if n else None
        out[name] = {"n": n, "frac_correct": correct,
                     "meets": None if n < MIN_N else bool(correct >= 0.9)}
    return out


def r_half(rows):
    """R at which kind (b)'s pass rate falls below one half (linear in log R between tested values)."""
    pts = [(math.log(R), rows[f"b:R={R:g}"].get("frac_invariant")) for R in GAIN_R]
    pts = [(lr, p) for lr, p in pts if p is not None]
    if not pts:
        return None
    if pts[0][1] < 0.5:
        return f"<{GAIN_R[0]:g}"
    for (l0, p0), (l1, p1) in zip(pts, pts[1:]):
        if p1 < 0.5 <= p0:
            return float(math.exp(l0 + (p0 - 0.5) / (p0 - p1) * (l1 - l0)))
    return f">{GAIN_R[-1]:g}"


def real_reference(res=None):
    if not os.path.exists("results/summary_quadrant.json") and res and "real_reference" in res:
        return res["real_reference"]            # e.g. vast-3, whose results/ has no summary_quadrant.json
    s = json.load(open("results/summary_quadrant.json"))
    v = [st["place_stability_median"] for tag, layers in s.items() if not tag.startswith("_")
         for st in layers.values()
         if st.get("frac_place") and st.get("place_stability_median") is not None
         and st["place_stability_median"] == st["place_stability_median"]]
    v = np.array(v)
    m = float(np.median(v))
    return {"n_entries": len(v), "median": m, "min": float(v.min()), "max": float(v.max()),
            "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
            "window": [m - WINDOW_HALF, m + WINDOW_HALF]}


# ---------------------------------------------------------------- the synthetic sweep
def synthetic(meta, ref, res):
    S, labels = build_units(meta)
    U = S.shape[1]
    names, cell, kind = cells_of(labels)
    sd = S.std(0)
    print(f"{U} synthetic units; real place-unit stability median {ref['median']:.3f} "
          f"(window {ref['window'][0]:.3f}-{ref['window'][1]:.3f})", flush=True)
    out = {}
    for ri, rho in enumerate(RHOS):
        eps = ar1_noise(meta, U, rho, seed=100 + ri)
        rows, per_k = [], {}
        for k in K_GRID:
            t0 = time.time()
            X = torch.from_numpy((S + k * sd * eps).astype(np.float32)).to(dev)
            stats, per = score(X, meta)
            del X
            pos_pole = per["pole"] < U
            sel = per["place"] & pos_pole
            u = per["pole"][sel]
            t = {"cell": cell[u], "kind": kind[u], "k": np.full(len(u), k), **{f: per[f][sel] for f in
                 ("stab", "inv_q", "hi", "G", "Graw")}}
            rows.append(t)
            # every unit, selected or not, for the selection rate
            n_per_cell = {c: int((cell == c).sum()) for c in names}
            cells = {c: summary(t, t["cell"] == c, n_per_cell[c]) for c in names}
            # kind (a) G among all positive poles (not only selected), for the false-flag rate
            per_k[f"{k:g}"] = {"n_poles_active": stats["n_poles_active"],
                               "analyze_layer_frac_place_heading_invariant_quadrant":
                                   stats["frac_place_heading_invariant_quadrant"],
                               "cells": cells}
            print(f"rho {rho} k {k:g} ({time.time() - t0:.0f} s): " + "; ".join(
                f"{c} sel {cells[c]['n_selected']}/{cells[c]['n_units']} inv "
                f"{cells[c].get('frac_invariant', float('nan')):.2f} stab {cells[c].get('stab_median') or float('nan'):.2f}"
                for c in names), flush=True)
        T = {f: np.concatenate([r[f] for r in rows]) for f in rows[0]}
        lo, hi = ref["window"]
        w = (T["stab"] >= lo) & (T["stab"] <= hi)
        g = groups(T["cell"], T["kind"])
        window = {"cells": {c: summary(T, w & (T["cell"] == c)) for c in names},
                  "groups": {n: summary(T, w & m) for n, m in g.items()}}
        window["r_half"] = r_half(window["cells"])
        bins = {}
        for b0, b1 in BINS:
            m = (T["stab"] >= b0) & (T["stab"] < b1)
            bins[f"{b0:.1f}-{min(b1, 1):.1f}"] = {
                "cells": {c: summary(T, m & (T["cell"] == c)) for c in names},
                "groups": {n: summary(T, m & gm) for n, gm in g.items()},
                "rule": rule_check(T, b0, b1 - 1e-9),
                "r_half": r_half({c: summary(T, m & (T["cell"] == c)) for c in names})}
        out[f"rho={rho:g}"] = {"per_k": per_k, "window": window, "stability_bins": bins,
                               "rule": rule_check(T, lo, hi)}
        print(f"rho {rho}: rule in window " + json.dumps(out[f"rho={rho:g}"]["rule"]), flush=True)
    res["synthetic"] = out
    res["verdict"] = verdict(out)
    print("verdict:", res["verdict"], flush=True)


def verdict(out):
    def ok(r):
        parts = r.values()
        if any(p["meets"] is None for p in parts):
            return None
        return all(p["meets"] for p in parts)
    p = ok(out["rho=0.5"]["rule"])
    sens = {k: ok(out[k]["rule"]) for k in ("rho=0", "rho=0.9")}
    fails = lambda r: [n for n, x in r.items() if x["meets"] is False]
    if p is None:
        return {"verdict": "not determined", "primary": out["rho=0.5"]["rule"]}
    if not p:
        return {"verdict": "not validated", "fails": fails(out["rho=0.5"]["rule"])}
    undetermined = [k for k, v in sens.items() if v is None]
    if not any(v is False for v in sens.values()):
        return {"verdict": "validated", "sensitivity": sens, "sensitivity_not_determined": undetermined}
    return {"verdict": "validated, with a caveat about the noise's time structure", "sensitivity": sens,
            "fails": {k: fails(out[k]["rule"]) for k, v in sens.items() if v is False},
            "sensitivity_not_determined": undetermined}


# ---------------------------------------------------------------- amendment 1: the axes compared
FIELDS = ("stab", "inv_q", "hi", "G", "Graw", "ns", "ew", "min6")


def axis_summary(t, m, n_units=None):
    s = summary(t, m, n_units)
    if m.sum():
        s.update(frac_mean2=float((t["inv_q"][m] > THRESH).mean()),
                 frac_both=float(((t["ns"][m] > THRESH) & (t["ew"][m] > THRESH)).mean()),
                 frac_min6=float((t["min6"][m] > THRESH).mean()), min6_median=med(t["min6"][m]))
    return s


def axis_groups(cell, kind):
    return {"a": kind == "a", "b (all R)": kind == "b",
            "c (w 60-180)": np.isin(cell, [f"c:w={w}" for w in RULE_W]), "c:w=270": cell == "c:w=270",
            "d (wall + pillar)": kind == "d", "e (axis-split)": kind == "e"}


def axis_synthetic(meta, ref):
    S1, l1 = build_units(meta)                        # the main run's units, same seeds
    S2, l2 = axis_units(meta)
    S, labels = np.concatenate([S1, S2], 1), l1 + l2
    U, U1 = S.shape[1], S1.shape[1]
    names, cell, kind = cells_of(labels)
    sd = S.std(0)
    out = {}
    for ri, rho in enumerate(RHOS):
        # kinds a-d get the main run's noise draw; kind (e) its own
        eps = np.concatenate([ar1_noise(meta, U1, rho, seed=100 + ri), ar1_noise(meta, U - U1, rho, seed=200 + ri)], 1)
        rows, per_k = [], {}
        for k in K_GRID:
            t0 = time.time()
            X = torch.from_numpy((S + k * sd * eps).astype(np.float32)).to(dev)
            stats, per = score(X, meta)
            del X
            sel = per["place"] & (per["pole"] < U)
            u = per["pole"][sel]
            t = {"cell": cell[u], "kind": kind[u], **{f: per[f][sel] for f in FIELDS}}
            rows.append(t)
            per_k[f"{k:g}"] = {c: axis_summary(t, t["cell"] == c, int((cell == c).sum())) for c in names}
            print(f"axis rho {rho} k {k:g} ({time.time() - t0:.0f} s): " + "; ".join(
                f"{c} {v['n_selected']} {v.get('frac_mean2', float('nan')):.2f}/{v.get('frac_both', float('nan')):.2f}/"
                f"{v.get('frac_min6', float('nan')):.2f}" for c, v in per_k[f'{k:g}'].items()), flush=True)
        T = {f: np.concatenate([r[f] for r in rows]) for f in rows[0]}
        g = axis_groups(T["cell"], T["kind"])
        lo, hi = ref["window"]
        w = (T["stab"] >= lo) & (T["stab"] <= hi)
        res_rho = {"per_k": per_k,
                   "window": {"cells": {c: axis_summary(T, w & (T["cell"] == c)) for c in names},
                              "groups": {n: axis_summary(T, w & m) for n, m in g.items()}},
                   "stability_bins": {}}
        for b0, b1 in BINS:
            m = (T["stab"] >= b0) & (T["stab"] < b1)
            res_rho["stability_bins"][f"{b0:.1f}-{min(b1, 1):.1f}"] = {
                "cells": {c: axis_summary(T, m & (T["cell"] == c)) for c in names},
                "groups": {n: axis_summary(T, m & gm) for n, gm in g.items()}}
        out[f"rho={rho:g}"] = res_rho
        print(f"axis rho {rho} window: " + json.dumps({n: [v["n_selected"], v.get("frac_mean2"), v.get("frac_both"),
                                                          v.get("frac_min6")] for n, v in res_rho["window"]["groups"].items()}),
              flush=True)
    return out


def bin_of(stab):
    for b0, b1 in BINS:
        if b0 <= stab < b1:
            return f"{b0:.1f}-{min(b1, 1):.1f}"
    return None


def axis_reference(ac, stab, rho="rho=0.5"):
    """Synthetic invariant units' pass rate under (2) in the stability bin of stab."""
    b = bin_of(stab) if stab is not None and stab == stab else None
    if b is None or "synthetic" not in ac:
        return None
    return ac["synthetic"][rho]["stability_bins"][b]["groups"]["a"].get("frac_min6")


def axis_rule(ac):
    out = {}
    for name in ("rnn-base", "rnn-s093"):
        for mode in ("anchored", "free"):
            r = ac.get("real", {}).get(name, {}).get(mode)
            if not r:
                continue
            s, st = r["axis"]["share_min6"], r["axis"]["place_stability_median"]
            ref = axis_reference(ac, st)
            status = ("meets" if s >= 0.95 else
                      "reliability" if ref is not None and s >= ref - 0.05 else "falls short")
            out[f"{name}/{mode}"] = {"share_min6": s, "place_stability_median": st,
                                     "synthetic_a_min6_in_bin": ref, "status": status}
    if len(out) < 4:
        verdict = f"not determined ({len(out)} of 4 RNN conditions scored)"
    elif all(v["status"] == "meets" for v in out.values()):
        verdict = "closed"
    elif any(v["status"] == "falls short" for v in out.values()):
        verdict = "not closed: an RNN condition falls more than 5 points below invariant units of its reliability under (2)"
    else:
        verdict = "closed, with a reliability caveat"
    return {"rnn": out, "verdict": verdict}


# ---------------------------------------------------------------- real units
def a_false_flag_by_bin(res):
    """Kind (a)'s fraction with G > 1 and fraction counted invariant per stability bin (primary rho)."""
    bins = res.get("synthetic", {}).get("rho=0.5", {}).get("stability_bins", {}) or \
        res.get("axis_check", {}).get("synthetic", {}).get("rho=0.5", {}).get("stability_bins", {})
    return {b: {"frac_G_above_1": v["groups"]["a"].get("frac_G_above_1"),
                "frac_invariant": v["groups"]["a"].get("frac_invariant"),
                "n": v["groups"]["a"]["n_selected"]} for b, v in bins.items()}


def real_condition(X, meta, ref_stats, ff):
    stats, per = score(X, meta)
    pl = per["place"]
    inv = pl & (per["inv_q"] > THRESH)
    non = pl & ~(per["inv_q"] > THRESH)
    G = per["G"]
    def gsum(m):
        g = G[m & np.isfinite(G)]
        return {"n": int(m.sum()), "n_G_finite": len(g), "G_median": med(g),
                "G_q75": float(np.percentile(g, 75)) if len(g) else None,
                "frac_G_above_1": float((g > 1).mean()) if len(g) else None,
                "Graw_median": med(per["Graw"][m])}
    # expected false flags if every invariant place unit were a pure place unit at its own stability
    exp_ff = []
    for s in per["stab"][inv]:
        for b, v in ff.items():
            b0, b1 = (float(x) for x in b.split("-"))
            if b0 <= s < b1 or (b1 == 1.0 and s >= b0):
                if v["frac_G_above_1"] is not None:
                    exp_ff.append(v["frac_G_above_1"])
                break
    out = {"frac_place": stats["frac_place"],
           "frac_place_heading_invariant_quadrant": stats["frac_place_heading_invariant_quadrant"],
           "invariant_share_of_place": float(inv.sum() / max(pl.sum(), 1)),
           "place_stability_median": stats["place_stability_median"],
           "invariant_units": gsum(inv), "not_invariant_units": gsum(non),
           "invariant_stab_median": med(per["stab"][inv]),
           "expected_frac_G_above_1_if_pure_place": float(np.mean(exp_ff)) if exp_ff else None,
           "axis": {"n_place": int(pl.sum()), "place_stability_median": stats["place_stability_median"],
                    "share_mean2": float((per["inv_q"][pl] > THRESH).mean()) if pl.any() else None,
                    "share_both": float(((per["ns"][pl] > THRESH) & (per["ew"][pl] > THRESH)).mean()) if pl.any() else None,
                    "share_min6": float((per["min6"][pl] > THRESH).mean()) if pl.any() else None,
                    "min6_median": med(per["min6"][pl]), "ns_median": med(per["ns"][pl]), "ew_median": med(per["ew"][pl])}}
    if ref_stats:
        out["reproduces_summary_quadrant"] = {
            k: {"here": stats[k], "summary_quadrant": ref_stats.get(k), "abs_diff": abs(stats[k] - ref_stats[k])}
            for k in ("frac_place", "frac_place_heading_invariant_quadrant") if k in ref_stats}
    return out


def real_units(meta, specs, res, ref_summary="results/summary_quadrant.json", dest=None, save=None):
    sq = json.load(open(ref_summary)) if os.path.exists(ref_summary) else {}
    ff = a_false_flag_by_bin(res)
    out = res.setdefault("real_units", {}) if dest is None else dest
    for spec in specs:
        tag, _, lay = spec.partition(":")
        base = tag.partition("+ema")[0]
        if lay:
            layers = lay.split(",")
        elif base == "pixels":
            layers = ["pix"]
        else:
            try:
                layers = sorted((f[:-4] for f in os.listdir(f"data/acts/{base}") if f.endswith(".npy")),
                                key=lambda s: int(s[1:]))
            except OSError as e:
                print(f"{tag}: SKIPPED ({e})", flush=True)
                out.setdefault(tag, {})["error"] = str(e)
                continue
        for layer in layers:
            t0 = time.time()
            try:
                X = analyze.load_units(tag, layer, "data/walks", meta)
                r = real_condition(X, meta, sq.get(tag, {}).get(layer), ff)
            except Exception as e:                      # noqa: BLE001 - record and go on to the next layer
                print(f"{tag} {layer}: FAILED ({type(e).__name__}: {e})", flush=True)
                out.setdefault(tag, {})[layer] = {"error": f"{type(e).__name__}: {e}"}
                continue
            del X
            out.setdefault(tag, {})[layer] = r
            print(f"{tag} {layer} ({time.time() - t0:.0f} s): " + json.dumps(
                {k: r[k] for k in ("invariant_share_of_place", "invariant_units", "expected_frac_G_above_1_if_pure_place")}),
                flush=True)
            if "reproduces_summary_quadrant" in r:
                print("   reproduces:", json.dumps(r["reproduces_summary_quadrant"]), flush=True)
            print("   axis:", json.dumps(r["axis"]), flush=True)
            if save:
                save()


def rnn_units(specs, res, root, dest=None, save=None):
    """The path-integrating RNN (experiment 13f), anchored and free, as grid_rnn.score builds it."""
    from spatialcells import grid_rnn
    meta = dict(np.load("data/walks/meta.npz"))
    W, S = int(meta["walk"].max()) + 1, int(meta["step"].max()) + 1
    xy = torch.tensor(np.stack([meta["x"], meta["y"]], 1).reshape(W, S, 2), dtype=torch.float32,
                      device=dev) / grid_rnn.SCALE
    keep = meta["step"] >= 1
    sub_meta = {k: v[keep] for k, v in meta.items()}
    sub_meta["step"] = sub_meta["step"] - 1
    ref = json.load(open("results/grid_rnn_quadrant.json")) if os.path.exists("results/grid_rnn_quadrant.json") else {}
    ff = a_false_flag_by_bin(res)
    out = res.setdefault("real_units", {}) if dest is None else dest
    for spec in specs:
        name, _, sub = spec.partition(":")
        sub = int(sub or 1)
        d = os.path.join(root, "grid_rnn" + ("" if name == "base" else f"_{name}"))
        pc, model = grid_rnn.PlaceCells(), grid_rnn.PathRNN().to(dev)
        model.load_state_dict(torch.load(f"{d}/model.pt", map_location=dev))
        for mode, chunk in (("anchored", max(1, round(grid_rnn.T / sub))), ("free", S)):
            with torch.no_grad():
                g, _ = grid_rnn.hidden(model, pc, xy, chunk, sub)
            X = g.reshape(-1, grid_rnn.NG)
            r = real_condition(X, sub_meta, ref.get(name, {}).get(mode), ff)
            out.setdefault(f"rnn-{name}", {})[mode] = r
            print(f"rnn {name} {mode}: " + json.dumps(
                {k: r[k] for k in ("invariant_share_of_place", "invariant_units", "expected_frac_G_above_1_if_pure_place")}),
                flush=True)
            if "reproduces_summary_quadrant" in r:
                print("   reproduces:", json.dumps(r["reproduces_summary_quadrant"]), flush=True)
            print("   axis:", json.dumps(r["axis"]), flush=True)
            if save:
                save()


E4_REMOTE = "results/heading_axis_check_e4.json"
E4_MODES = ("walk", "single", "single+ema8")


def fetch_e4(out):
    """Copy only vast-3's axis-check output for experiment 22's recordings back (md5 checked) and merge its
    axis_check.real_e4 into out; nothing else in out changes. Run 0's vast-3 re-record (qwen8b-ft-rerun-*)
    is compared layer by layer with vast-1's run 0 (axis_check.real["qwen8b-ft-*"])."""
    import hashlib
    import subprocess
    env = {}
    for line in open("scripts/vast1/host-vast3.env"):
        k, eq, v = line.strip().partition("=")
        if eq and not k.startswith("#"):
            env[k] = v.strip().strip('"')
    host, box = env["MI_VAST_HOST"], env.get("MI_CONTAINER", "mechinterp")
    root = env.get("MI_REMOTE_ROOT", "/workspace/MechInterp")
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host]
    remote_md5 = subprocess.run(ssh + [f"docker exec -w {root} {box} md5sum {E4_REMOTE}"], capture_output=True,
                                text=True, check=True).stdout.split()[0]
    data = subprocess.run(ssh + [f"docker exec -w {root} {box} cat {E4_REMOTE}"], capture_output=True,
                          check=True).stdout
    local_md5 = hashlib.md5(data).hexdigest()
    if local_md5 != remote_md5:
        sys.exit(f"md5 mismatch: vast-3 {remote_md5}, copy {local_md5}")
    src = json.loads(data)["axis_check"]
    res = json.load(open(out))
    ac = res.setdefault("axis_check", {})
    ac["real_e4"] = src["real_e4"]
    parity = {}
    for mode in E4_MODES:
        new, old = src["real_e4"].get(f"qwen8b-ft-rerun-{mode}", {}), ac.get("real", {}).get(f"qwen8b-ft-{mode}", {})
        for layer, r in new.items():
            if "axis" not in r or "axis" not in old.get(layer, {}):
                continue
            a, b = r["axis"], old[layer]["axis"]
            parity.setdefault(mode, {})[layer] = {
                k: {"vast3_rerun": a[k], "vast1_run0": b[k], "diff": None if a[k] is None or b[k] is None else a[k] - b[k]}
                for k in ("n_place", "place_stability_median", "share_mean2", "share_both", "share_min6")}
    diffs = [abs(v["diff"]) for m in parity.values() for L in m.values() for k, v in L.items()
             if k.startswith("share") and v["diff"] is not None]
    ac["real_e4_about"] = {
        "source": f"vast-3 {root}/{E4_REMOTE}", "md5": remote_md5,
        "fetched": time.strftime("%Y-%m-%d %H:%M %Z"),
        "what": "experiment 22's recordings (four-epoch runs qwen8b-ft-e4-s3..s7 and run 0's vast-3 re-record "
                "qwen8b-ft-rerun), walk / single / single+ema8, scored as amendment 1 (axis_check.real)",
        "parity_run0_max_abs_share_diff": max(diffs) if diffs else None}
    ac["real_e4_parity_run0"] = parity
    json.dump(res, open(out, "w"), indent=1)
    print(f"merged axis_check.real_e4 ({len(src['real_e4'])} conditions) into {out}; md5 {remote_md5}")
    for tag, layers in src["real_e4"].items():
        for layer, r in layers.items():
            if "axis" in r:
                a = r["axis"]
                print(f"  {tag:30s} {layer:4s} place {a['n_place']:5d} stab {a['place_stability_median']:.3f} "
                      f"mean2 {a['share_mean2']:.4f} both {a['share_both']:.4f} min6 {a['share_min6']:.4f} "
                      f"ref {a.get('synthetic_a_min6_in_bin')}")
            else:
                print(f"  {tag:30s} {layer}: {r}")
    print("run-0 parity, largest share difference (vast-3 re-record vs vast-1):", ac["real_e4_about"]["parity_run0_max_abs_share_diff"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-synthetic", action="store_true", help="only the real units (reuses the file's synthetic part)")
    ap.add_argument("--real", nargs="*", default=["vggt-single:L19,L21", "pixels:pix", "pixels+ema8:pix"],
                    help="tag[:layer,layer] on data/walks (all layers in data/acts/<tag> if none given)")
    ap.add_argument("--rnn", nargs="*", default=[], help="name:sub, e.g. base:1 s093:3 (data/grid_rnn[_name]/model.pt)")
    ap.add_argument("--rnn-root", default="data")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--ref-summary", default="results/summary_quadrant.json",
                    help="analyze.py --quadrant summary the real units' fractions are checked against "
                         "(e.g. results/e4_internals/summary_quadrant.json for experiment 22's recordings)")
    ap.add_argument("--axis-check", action="store_true",
                    help="amendment 1: the six sector-pair shares, synthetic (with axis-split units) then --real/--rnn, "
                         "into the --out file's axis_check key")
    ap.add_argument("--real-key", default="real",
                    help="with --axis-check: the axis_check key the real conditions go under (real_e4 for "
                         "experiment 22's four-epoch recordings on vast-3)")
    ap.add_argument("--fetch-e4", action="store_true",
                    help="on the Mac: copy vast-3's results/heading_axis_check_e4.json (md5 checked) and merge its "
                         "axis_check.real_e4 into --out, with run 0's vast-3 re-record checked against vast-1's run 0")
    args = ap.parse_args()
    if args.fetch_e4:
        fetch_e4(args.out)
        return
    t0 = time.time()
    meta = dict(np.load("data/walks/meta.npz"))
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    ref = real_reference(res)
    res["_about"] = {
        "experiment": "21 (docs/experiments.md)",
        "statistic": "analyze.analyze_layer(..., quadrant=True) unchanged; per-unit opposite-sector score (mean of "
                     "N-vs-S and E-vs-W rate-map correlations over bins both 90-degree sectors visit) recomputed "
                     "as analyze_layer does and checked against its frac_place_heading_invariant_quadrant",
        "threshold": THRESH,
        "frac_invariant": "among selected place units (positive poles), fraction with opposite-sector score > 0.5; "
                          "frac_invariant_180 uses the older 180-degree halves",
        "gain_index": "G: per opposite pair, log2 ratio of field amplitude (in-field minus out-of-field mean of the "
                      "sector map; in-field = all-heading map above half its range; bins both sectors visit), "
                      "each floored at 5% of the all-heading amplitude; G = length of the (N/S, E/W) vector. "
                      "Graw: same with raw in-field rates",
        "units": {"a": N_A, "b": {"R": GAIN_R, "per_R": N_B}, "c": {"w_deg": CONJ_W, "per_w": N_C, "rule_w": RULE_W},
                  "d": {"wall": N_D, "pillar": N_D}},
        "noise": {"k": K_GRID, "rho": RHOS, "primary_rho": RHOS[0],
                  "model": "AR(1) within each walk, k x SD of the unit's noise-free signal"},
        "window": "selected units whose own split-half stability is within 0.05 of the real median",
        "rule": "in the window at rho 0.5: >= 90% of (a) above 0.5; >= 90% of (c, w 60-180) and of (d) at or below; "
                ">= 30 units each",
    }
    res["real_reference"] = ref
    if args.axis_check:
        ac = res.setdefault("axis_check", {})
        ac["_about"] = {
            "amendment": "Exp 21 amendment 1 (docs/experiments.md): the two axes compared with each other",
            "pairs": list(PAIRS6), "threshold": THRESH,
            "share_mean2": "place units with (N-S + E-W) / 2 > 0.5 (the paper's measure)",
            "share_both": "(1) N-S > 0.5 and E-W > 0.5",
            "share_min6": "(2) the smallest of the six sector-pair correlations > 0.5",
            "axis_units": {"e:hard": "f1 in the N and S sectors, f2 in E and W", "e:smooth": "sin^2 f1 + cos^2 f2",
                           "per_variant": N_E},
            "rule": "closed if each RNN condition keeps >= 95% under (2); below that, within 5 points of synthetic "
                    "invariant units' (2) rate in its stability bin = reliability caveat; further = not closed",
        }
        def save():
            json.dump(res, open(args.out, "w"), indent=1)
        if not args.skip_synthetic:
            ac["synthetic"] = axis_synthetic(meta, ref)
            save()
        real = ac.setdefault(args.real_key, {})
        if args.real:
            real_units(meta, args.real, res, args.ref_summary, dest=real, save=save)
        if args.rnn:
            rnn_units(args.rnn, res, args.rnn_root, dest=real, save=save)
        for conds in real.values():
            for r in conds.values():
                if isinstance(r, dict) and "axis" in r:
                    r["axis"]["synthetic_a_min6_in_bin"] = axis_reference(ac, r["axis"]["place_stability_median"])
        if args.real_key == "real":
            ac["rule"] = axis_rule(ac)
            print("axis rule:", json.dumps(ac["rule"]), flush=True)
        res.setdefault("runtime_s", {})["axis_check"] = round(time.time() - t0)
        save()
        print(f"wrote {args.out} ({time.time() - t0:.0f} s)", flush=True)
        return
    if not args.skip_synthetic:
        synthetic(meta, ref, res)
        json.dump(res, open(args.out, "w"), indent=1)
    if args.real:
        real_units(meta, args.real, res, args.ref_summary)
        json.dump(res, open(args.out, "w"), indent=1)
    if args.rnn:
        rnn_units(args.rnn, res, args.rnn_root)
    res.setdefault("runtime_s", {})["last_run"] = round(time.time() - t0)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
