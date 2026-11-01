"""Experiment 25 (docs/experiments.md): a population-level test of a heading-invariant place code, validated on
controls whose answer is known before it is applied to any model.

    # synthetic populations (room A's real trajectories, experiment 21's unit generator)
    python scripts/population_assay.py synthetic --seed 2400
    # real conditions on room A's walks (activations in data/acts/<tag>/<layer>.npy), probes, the RNNs
    python scripts/population_assay.py real --group controls --rnn base:1 s093:3 --tags pixels vggt-alone qwen-single qwen8b-single
    python scripts/population_assay.py real --group conditions --tags qwen8b-ft-walk ... --probe-tags qwen8b-ft-probes-walk ...
    # the registered validation rule, the decision rule and the summary the paper reads
    python scripts/population_assay.py summarize
    # merge another machine's file (e.g. vast-3's four-epoch runs) into this one, then summarize
    python scripts/population_assay.py merge --from results/e4_internals/population_assay_e4.json

Primary assay: heading-held-out decoding. For each 90-degree facing sector s (centred on E, N, W, S) a linear
readout of (x, y) is trained on frames whose heading is at least 90 degrees from s's centre (the opposite half
of the compass, so training mixes 180 degrees of heading) and tested on the frames facing s, 45 degrees or more
from any training heading. A matched readout is trained on the 180 degrees centred on s and tested on the same
frames. Every frame of every walk is tested once (training walks even, test walks odd, then the reverse).
    HG = (chance_ho - err_ho) / (chance_seen - err_seen)
is the share of the position information a readout has for headings it was trained on that survives a heading
it never saw (errors are median Euclidean errors in metres over all frames; chance predicts the training
frames' mean position). Units are z-scored over all frames (no positions used), and the ridge penalty is
picked on held-out walks inside the training arc, so the test heading never touches the fit.
Secondary: the same with no guard band (train 270 degrees, test the remaining 90) and with only the opposite
sector (train 90 degrees, experiment 13e's strict split); and a population-vector correlation across
opposite sectors (assay a).
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
from spatialcells import analyze                                  # noqa: E402
import heading_stat_validation as hsv                             # noqa: E402  (experiment 21's generator)

OUT = "results/population_assay.json"
dev = analyze.dev
CENTERS = np.array([0.0, 0.5, 1.0, 1.5]) * np.pi                  # E, N, W, S (analyze.sectors' order)
# variant: (training arc = heading at least `ho` degrees from the test sector's centre,
#           matched seen arc = heading less than `seen` degrees from it); both arcs have the same width
VARIANTS = {"guard45": (90.0, 90.0), "loso": (45.0, 135.0), "strict": (135.0, 45.0)}
PRIMARY = "guard45"
LAMBDAS = 10.0 ** np.arange(0, 8.01, 0.5)                         # 1 ... 1e8 per 1000 training frames, picked on
                                                                  # held-out walks inside the training arc
HG_POS = 0.5                     # call: positive when HG >= this, partial when HG_NEG <= HG < HG_POS, negative below,
HG_NEG = 0.25                    # all only when ...
MIN_CODE_M = 0.5                 # ... the seen-heading readout beats chance by at least this many metres
MONO_TOL = 0.05                  # validation: HG may dip by at most this between successive mixing levels
P_STAR = 0.5                     # validation: synthetic populations with at least this share of invariant
                                 # place units must be called positive
N_BOOT = 1000
N_PERM = 200

# synthetic populations
SYN_U = 2000
SYN_P = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
SYN_K = (1.0, 3.0, 5.0)          # AR(1) noise, k x the unit's SD; k = 3 gives the real units' median stability
SYN_RHO = 0.5
SYN_PRIMARY_K = 3.0
CONJ_W_BG = (60, 90)             # heading-dependent background: conjunctive windows no wider than the field of view


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- the readout
def standardize(X):
    """z-score every unit over all frames (no positions used); drop constant units."""
    X = X.float()
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-6 * sd.max().clamp_min(1e-12)
    return (X[:, keep] - mu[keep]) / sd[keep]


def gram(Z):
    return (Z @ Z.T).double()


def _center(G, tr, te):
    K = G[tr][:, tr]
    a = K.mean(1)
    b = a.mean()
    Kc = K - a[:, None] - a[None, :] + b
    Kst = G[te][:, tr]
    Kstc = Kst - Kst.mean(1, keepdim=True) - a[None, :] + b
    return Kc, Kstc


def _solve(Kc, yc, lam):
    n = len(Kc)
    L = torch.linalg.cholesky(Kc + lam * torch.eye(n, dtype=Kc.dtype))
    return torch.cholesky_solve(yc, L)


def ridge(G, Y, tr, te, fit, val):
    """Kernel ridge (linear kernel, unpenalized intercept) on the Gram G: penalty picked by fitting on `fit`
    and scoring on `val` (both inside the training set), then refit on `tr` and predict `te`."""
    Kc, Kvc = _center(G, fit, val)
    yf = Y[fit]
    ym = yf.mean(0)
    best, best_err = None, math.inf
    for lam in LAMBDAS:
        try:
            alpha = _solve(Kc, yf - ym, lam * len(fit) / 1000.0)
        except RuntimeError:
            continue
        err = (Kvc @ alpha + ym - Y[val]).norm(dim=1).median().item()
        if err < best_err:
            best, best_err = lam, err
    Kc, Ktc = _center(G, tr, te)
    ym = Y[tr].mean(0)
    alpha = _solve(Kc, Y[tr] - ym, best * len(tr) / 1000.0)
    pred = Ktc @ alpha + ym
    return pred, ym, best


def heldout_decoding(Z, meta, variants=VARIANTS):
    """Every frame is tested once: by the readout of its fold (test walks odd or even) and its own sector."""
    G = gram(Z)
    Y = torch.tensor(np.stack([meta["x"], meta["y"]], 1), dtype=torch.float64)
    walk, th = meta["walk"], meta["theta"]
    sec = analyze.sectors(th)
    N = len(walk)
    out = {}
    for name, (ho_deg, seen_deg) in variants.items():
        err = {k: np.full(N, np.nan) for k in ("ho", "seen", "chance_ho", "chance_seen")}
        lams = {"ho": [], "seen": []}
        for par in (0, 1):                                   # training walks even, test odd; then the reverse
            trw, tew = walk % 2 == par, walk % 2 == 1 - par
            a_w, b_w = walk % 4 == par, walk % 4 == par + 2  # inside the training walks: fit / validate the penalty
            for s in range(4):
                d = np.abs(wrap(th - CENTERS[s]))
                te_m = tew & (sec == s)
                te = torch.from_numpy(np.flatnonzero(te_m))
                for kind, arc in (("ho", d >= np.deg2rad(ho_deg)), ("seen", d < np.deg2rad(seen_deg))):
                    t = lambda m: torch.from_numpy(np.flatnonzero(m))
                    pred, ym, lam = ridge(G, Y, t(trw & arc), te, t(a_w & arc), t(b_w & arc))
                    err[kind][te_m] = (pred - Y[te]).norm(dim=1).numpy()
                    err[f"chance_{kind}"][te_m] = (ym[None] - Y[te]).norm(dim=1).numpy()
                    lams[kind].append(float(lam))
        out[name] = summarize_errors(err, walk, sec)
        out[name]["lambda"] = lams
    del G
    return out


def hg_from(e):
    """HG and its parts from per-frame error arrays (pooled medians)."""
    m = {k: float(np.median(v)) for k, v in e.items()}
    gain_seen = m["chance_seen"] - m["seen"]
    gain_ho = m["chance_ho"] - m["ho"]
    return m, gain_ho, gain_seen, (gain_ho / gain_seen if gain_seen > 0 else float("nan"))


def summarize_errors(err, walk, sec):
    m, gain_ho, gain_seen, hg = hg_from(err)
    res = {"err_ho_m": m["ho"], "err_seen_m": m["seen"], "chance_ho_m": m["chance_ho"],
           "chance_seen_m": m["chance_seen"], "gain_ho_m": gain_ho, "gain_seen_m": gain_seen, "HG": hg,
           "per_sector_err_ho_m": [float(np.median(err["ho"][sec == s])) for s in range(4)],
           "per_sector_err_seen_m": [float(np.median(err["seen"][sec == s])) for s in range(4)]}
    # bootstrap over walks
    W = int(walk.max()) + 1
    idx = [np.flatnonzero(walk == w) for w in range(W)]
    rng = np.random.default_rng(0)
    hgs, gho = [], []
    for _ in range(N_BOOT):
        sel = np.concatenate([idx[w] for w in rng.integers(0, W, W)])
        _, gh, gs, h = hg_from({k: v[sel] for k, v in err.items()})
        hgs.append(h)
        gho.append(gh)
    hgs = np.array(hgs)
    res["HG_ci95"] = [float(np.nanpercentile(hgs, 2.5)), float(np.nanpercentile(hgs, 97.5))]
    res["gain_ho_ci95_m"] = [float(np.percentile(gho, 2.5)), float(np.percentile(gho, 97.5))]
    res["call"] = call(res)
    return res


def call(r):
    if not (r["gain_seen_m"] >= MIN_CODE_M):
        return "no position code"
    return "positive" if r["HG"] >= HG_POS else ("partial" if r["HG"] >= HG_NEG else "negative")


# ---------------------------------------------------------------- assay (a): population vectors
def pv_index(Z, meta):
    """Population-vector correlation across opposite sectors (assay a). Maps per sector from even and from odd
    walks; each sector's population vectors are centred across bins (removing any heading offset); r is the
    correlation over units between the two population vectors of the same bin, averaged over bins both visit."""
    bins = analyze.spatial_bins(meta)
    sec = analyze.sectors(meta["theta"])
    even = meta["walk"] % 2 == 0
    maps = {}
    for s in range(4):
        for p, m in (("e", even), ("o", ~even)):
            mm, occ = analyze.rate_maps(Z, bins, torch.from_numpy(m & (sec == s)).to(dev))
            maps[s, p] = (mm.reshape(len(mm), -1), occ.reshape(-1))

    def pair(A, B, perms=0):
        (ma, oa), (mb, ob) = A, B
        v = (oa >= analyze.MIN_OCC) & (ob >= analyze.MIN_OCC)
        a, b = ma[:, v], mb[:, v]
        a = a - a.mean(1, keepdim=True)
        b = b - b.mean(1, keepdim=True)
        a = (a - a.mean(0)) / a.std(0).clamp_min(1e-12)       # per bin over units: correlation = mean product
        b = (b - b.mean(0)) / b.std(0).clamp_min(1e-12)
        r = (a * b).mean(0)
        null = []
        g = torch.Generator().manual_seed(0)
        for _ in range(perms):
            null.append(float((a * b[:, torch.randperm(b.shape[1], generator=g)]).mean(0).mean()))
        return float(r.mean()), int(v.sum()), null

    cross, null = [], []
    for s, s2 in ((1, 3), (3, 1), (0, 2), (2, 0)):
        r, n, nl = pair(maps[s, "e"], maps[s2, "o"], N_PERM // 4)
        cross.append(r)
        null += nl
    within = [pair(maps[s, "e"], maps[s, "o"])[0] for s in range(4)]
    rc, rw = float(np.mean(cross)), float(np.mean(within))
    return {"r_cross": rc, "r_within": rw, "index": rc / rw if rw > 0 else float("nan"),
            "null_r_cross_q99": float(np.percentile(null, 99)), "p_perm": float((1 + sum(x >= rc for x in null)) / (1 + len(null)))}


def assay(X, meta, pv=True):
    t0 = time.time()
    Z = standardize(X.cpu())
    res = {"n_units": int(Z.shape[1]), "n_frames": int(Z.shape[0])}
    res.update(heldout_decoding(Z, meta))
    if pv:
        res["pv"] = pv_index(Z.to(dev), meta)
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ---------------------------------------------------------------- synthetic populations
def hd_units(th, rng, n):
    phi, kappa = rng.uniform(0, 2 * np.pi, n), rng.uniform(1, 6, n)
    return np.exp(kappa[None] * (np.cos(th[:, None] - phi[None]) - 1))


def conj_units(pos, th, rng, n, widths):
    F = hsv.place_fields(pos, rng, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    w = np.deg2rad(np.array([widths[i % len(widths)] for i in range(n)]))
    return F * (np.abs(wrap(th[:, None] - phi[None])) <= w[None] / 2)


def view_units(pos, th, rng, n):
    cols = []
    for i in range(n):
        if i % 2 == 0:
            cols.append(hsv.view_unit(pos, th, hsv.wall_point(rng), rng))
        else:
            j = (i // 2) % len(hsv.PILLARS)
            cols.append(hsv.view_unit(pos, th, np.array(hsv.PILLARS[j][:2]), rng, skip=j))
    return np.stack(cols, 1)


def population(meta, p, rng, kind="mixed"):
    """Noise-free signals (N, SYN_U): a share p of heading-invariant place units (experiment 21's kind a); the
    rest split evenly into conjunctive place-by-heading units (windows 60 and 90 degrees), view-dependent units
    (wall points and pillars, experiment 21's kind d) and head-direction units. kind="views": view units only.
    kind="conj180": invariant share p, the rest conjunctive with 180-degree windows (a boundary case)."""
    pos = np.stack([meta["x"], meta["y"]], 1)
    th = meta["theta"]
    n_inv = int(round(p * SYN_U))
    cols = [hsv.place_fields(pos, rng, n_inv)] if n_inv else []
    rest = SYN_U - n_inv
    if kind == "views":
        cols.append(view_units(pos, th, rng, rest))
    elif kind == "conj180":
        cols.append(conj_units(pos, th, rng, rest, (180,)))
    else:
        n3 = [rest // 3 + (i < rest % 3) for i in range(3)]
        if n3[0]:
            cols.append(conj_units(pos, th, rng, n3[0], CONJ_W_BG))
        if n3[1]:
            cols.append(view_units(pos, th, rng, n3[1]))
        if n3[2]:
            cols.append(hd_units(th, rng, n3[2]))
    S = np.concatenate(cols, 1)
    return S + rng.uniform(0, 0.1, S.shape[1]) * S.max(0)          # constant offset, 0-10% of the peak


def unit_test(X, meta):
    """The paper's strict unit statistic (analyze_layer, quadrant): invariant share of place units."""
    st, _ = analyze.analyze_layer(X.to(dev), meta, np.random.default_rng(0), quadrant=True)
    fp = st["frac_place"]
    return {"frac_place": fp, "invariant_share_of_place": st["frac_place_heading_invariant_quadrant"] / fp if fp else None,
            "place_stability_median": st["place_stability_median"]}


def synthetic(args, res):
    meta = dict(np.load("data/walks/meta.npz"))
    out = res.setdefault("synthetic", {})
    out["_design"] = {"seed": args.seed, "units": SYN_U, "p": SYN_P, "k": args.ks, "rho": SYN_RHO,
                      "background": "1/3 conjunctive (w 60, 90 deg), 1/3 view (wall, pillar), 1/3 head direction",
                      "rotated": "z-scored population times a random orthogonal matrix: every unit a mixture"}
    pops = [("mixed", p) for p in SYN_P] + [("views", 0.0), ("conj180", 0.0), ("conj180", 0.1)]
    for kind, p in pops:
        rng = np.random.default_rng([args.seed, int(p * 1000), ["mixed", "views", "conj180"].index(kind)])
        S = population(meta, p, rng, kind)
        sd = S.std(0)
        for ki, k in enumerate(args.ks):
            eps = hsv.ar1_noise(meta, S.shape[1], SYN_RHO, seed=args.seed + 7 * ki + 1)
            X = torch.from_numpy((S + k * sd * eps).astype(np.float32))
            Xs = standardize(X)
            Q, _ = torch.linalg.qr(torch.randn(Xs.shape[1], Xs.shape[1], generator=torch.Generator().manual_seed(args.seed + ki)))
            for rot in (False, True):
                key = f"{kind}:p={p:g}:k={k:g}:{'rot' if rot else 'units'}"
                if key in out and not args.redo:
                    continue
                Xr = Xs @ Q if rot else Xs
                r = assay(Xr, meta, pv=not args.no_pv)
                if k == SYN_PRIMARY_K and kind == "mixed" and not args.no_unit_test:
                    r["unit_test"] = unit_test(Xr, meta)
                r.update(kind=kind, p=p, k=k, rotated=rot)
                out[key] = r
                v = r[PRIMARY]
                print(f"{key}: HG {v['HG']:.2f} [{v['HG_ci95'][0]:.2f}, {v['HG_ci95'][1]:.2f}] err ho {v['err_ho_m']:.2f} "
                      f"seen {v['err_seen_m']:.2f} chance {v['chance_ho_m']:.2f}; loso {r['loso']['HG']:.2f} strict "
                      f"{r['strict']['HG']:.2f}" + (f"; pv {r['pv']['index']:.2f}" if "pv" in r else "")
                      + (f"; unit {r['unit_test']['invariant_share_of_place']}" if "unit_test" in r else "")
                      + f" ({r['seconds']} s)", flush=True)
                save(res, args.out)


# ---------------------------------------------------------------- real conditions
def layers_of(tag):
    base = tag.partition("+ema")[0]
    if base == "pixels":
        return ["pix"]
    return sorted((f[:-4] for f in os.listdir(f"data/acts/{base}") if f.endswith(".npy")), key=lambda s: int(s[1:]))


def real(args, res):
    meta = dict(np.load("data/walks/meta.npz"))
    pmeta = dict(np.load("data/walks/probes_meta.npz"))
    out = res.setdefault(args.group, {})
    for spec in args.tags:
        tag, _, lay = spec.partition(":")
        for layer in (lay.split(",") if lay else layers_of(tag)):
            if layer in out.get(tag, {}) and not args.redo:
                continue
            X = analyze.load_units(tag, layer, "data/walks", meta).cpu()
            r = assay(X, meta, pv=not args.no_pv)
            del X
            out.setdefault(tag, {})[layer] = r
            report(tag, layer, r)
            save(res, args.out)
    for spec in args.probe_tags:
        tag, _, lay = spec.partition(":")
        for layer in (lay.split(",") if lay else layers_of(tag)):
            if layer in out.get(tag, {}) and not args.redo:
                continue
            X = torch.from_numpy(np.load(f"data/acts/{tag}/{layer}.npy").astype(np.float32))
            r = assay(X, pmeta, pv=False)
            del X
            out.setdefault(tag, {})[layer] = r
            report(tag, layer, r)
            save(res, args.out)
    for spec in args.rnn:
        rnn(spec, out, args)
        save(res, args.out)


def rnn(spec, out, args):
    """The path-integrating RNN (experiment 13f), anchored and free, as grid_rnn.score and experiment 21 build it."""
    from spatialcells import grid_rnn
    meta = dict(np.load("data/walks/meta.npz"))
    W, S = int(meta["walk"].max()) + 1, int(meta["step"].max()) + 1
    xy = torch.tensor(np.stack([meta["x"], meta["y"]], 1).reshape(W, S, 2), dtype=torch.float32, device=dev) / grid_rnn.SCALE
    keep = meta["step"] >= 1
    sub_meta = {k: v[keep] for k, v in meta.items()}
    sub_meta["step"] = sub_meta["step"] - 1
    name, _, sub = spec.partition(":")
    sub = int(sub or 1)
    d = os.path.join(args.rnn_root, "grid_rnn" + ("" if name == "base" else f"_{name}"))
    pc, model = grid_rnn.PlaceCells(), grid_rnn.PathRNN().to(dev)
    model.load_state_dict(torch.load(f"{d}/model.pt", map_location=dev))
    for mode, chunk in (("anchored", max(1, round(grid_rnn.T / sub))), ("free", S)):
        tag = f"rnn-{name}"
        if mode in out.get(tag, {}) and not args.redo:
            continue
        with torch.no_grad():
            g, _ = grid_rnn.hidden(model, pc, xy, chunk, sub)
        r = assay(g.reshape(-1, grid_rnn.NG).cpu(), sub_meta, pv=not args.no_pv)
        out.setdefault(tag, {})[mode] = r
        report(tag, mode, r)


def report(tag, layer, r):
    v = r[PRIMARY]
    print(f"{tag} {layer}: HG {v['HG']:.2f} [{v['HG_ci95'][0]:.2f}, {v['HG_ci95'][1]:.2f}] ({v['call']}); err ho "
          f"{v['err_ho_m']:.2f} m, seen {v['err_seen_m']:.2f} m, chance {v['chance_ho_m']:.2f} m; loso {r['loso']['HG']:.2f}, "
          f"strict {r['strict']['HG']:.2f} ({r['strict']['err_ho_m']:.2f} m)"
          + (f"; pv {r['pv']['index']:.2f} (r {r['pv']['r_cross']:.3f}, p {r['pv']['p_perm']:.3f})" if "pv" in r else "")
          + f" [{r['seconds']} s]", flush=True)


# ---------------------------------------------------------------- rules and summary
def layer_rows(layers, variant=PRIMARY):
    return [(l, r[variant]) for l, r in layers.items() if isinstance(r, dict) and variant in r]


def condition_call(layers, variant=PRIMARY):
    """A condition's call is the call at its best layer: the highest HG among layers whose seen-heading readout
    beats chance by MIN_CODE_M (if no layer does, the condition has no position code to test)."""
    rows = layer_rows(layers, variant)
    if not rows:
        return None
    coded = [(l, v) for l, v in rows if v["gain_seen_m"] >= MIN_CODE_M and np.isfinite(v["HG"])]
    if not coded:
        l, v = min(rows, key=lambda lv: lv[1]["err_seen_m"])
        c = "no position code"
    else:
        l, v = max(coded, key=lambda lv: lv[1]["HG"])
        c = v["call"]
    return {"best_layer": l, "HG": v["HG"], "HG_ci95": v["HG_ci95"], "call": c, "err_ho_m": v["err_ho_m"],
            "err_seen_m": v["err_seen_m"], "chance_ho_m": v["chance_ho_m"], "gain_ho_m": v["gain_ho_m"],
            "gain_ho_ci95_m": v["gain_ho_ci95_m"], "gain_seen_m": v["gain_seen_m"],
            "n_layers": len(rows), "HG_by_layer": {ll: vv["HG"] for ll, vv in rows}}


POS_CONTROLS = ["rnn-base:anchored", "rnn-s093:anchored"]          # in the rule
POS_REPORTED = ["rnn-base:free", "rnn-s093:free"]                  # reported beside it
NEG_CONTROLS = ["pixels", "vggt-alone", "qwen-single", "qwen8b-single"]


def combine(flags):
    """False if any part fails, None (not determined) if any part is missing, else True."""
    if any(f is False for f in flags):
        return False
    if any(f is None for f in flags):
        return None
    return True


def validation(res, variant=PRIMARY):
    ctl, syn = res.get("controls", {}), res.get("synthetic", {})
    v = {"variant": variant, "positive_controls": {}, "negative_controls": {}, "synthetic": {}}
    flags = []
    for spec in POS_CONTROLS + POS_REPORTED:
        tag, mode = spec.split(":")
        r = ctl.get(tag, {}).get(mode)
        x = r[variant] if r else None
        meets = None if x is None else x["call"] == "positive"
        v["positive_controls"][spec] = None if x is None else {
            "HG": x["HG"], "HG_ci95": x["HG_ci95"], "err_ho_m": x["err_ho_m"], "err_seen_m": x["err_seen_m"],
            "chance_ho_m": x["chance_ho_m"], "call": x["call"], "meets": meets, "in_rule": spec in POS_CONTROLS}
        if spec in POS_CONTROLS:
            flags.append(meets)
    for tag in NEG_CONTROLS:
        rows = layer_rows(ctl.get(tag, {}), variant)
        if not rows:
            v["negative_controls"][tag] = None
            flags.append(None)
            continue
        fin = [x["HG"] for _, x in rows if np.isfinite(x["HG"])]
        meets = all(x["call"] in ("negative", "no position code") for _, x in rows)
        v["negative_controls"][tag] = {"max_HG": max(fin) if fin else None, "HG_by_layer": {l: x["HG"] for l, x in rows},
                                       "calls": {l: x["call"] for l, x in rows}, "meets": meets}
        flags.append(meets)
    for k in SYN_K:
        for rot in ("units", "rot"):
            rows = [(p, syn.get(f"mixed:p={p:g}:k={k:g}:{rot}")) for p in SYN_P]
            if any(r is None for _, r in rows):
                v["synthetic"][f"mixed k={k:g} {rot}"] = None
                flags.append(None)
                continue
            hg = [r[variant]["HG"] if np.isfinite(r[variant]["HG"]) else -1.0 for _, r in rows]
            calls = [r[variant]["call"] for _, r in rows]
            mono = all(b >= a - MONO_TOL for a, b in zip(hg, hg[1:]))
            pos_ok = all(c == "positive" for (p, _), c in zip(rows, calls) if p >= P_STAR)
            neg_ok = calls[0] in ("negative", "no position code")
            meets = bool(mono and pos_ok and neg_ok)
            v["synthetic"][f"mixed k={k:g} {rot}"] = {"p": list(SYN_P), "HG": hg, "calls": calls, "monotone": mono,
                                                     "positive_at_p_star_and_above": pos_ok, "negative_at_p0": neg_ok,
                                                     "meets": meets}
            flags.append(meets)
    views = {f"k={k:g} {rot}": syn.get(f"views:p=0:k={k:g}:{rot}") for k in SYN_K for rot in ("units", "rot")}
    if all(r is not None for r in views.values()):
        meets = all(r[variant]["call"] in ("negative", "no position code") for r in views.values())
        v["synthetic"]["views only"] = {"HG": {n: r[variant]["HG"] for n, r in views.items()},
                                        "calls": {n: r[variant]["call"] for n, r in views.items()}, "meets": meets}
        flags.append(meets)
    else:
        v["synthetic"]["views only"] = None
        flags.append(None)
    v["validated"] = combine(flags)
    return v


def calibration(res, k=SYN_PRIMARY_K, rot="units", variant=PRIMARY):
    """(p, HG) of the synthetic mixtures at the primary noise level, and the invariant share p a given HG matches."""
    syn = res.get("synthetic", {})
    pts = [(p, syn[f"mixed:p={p:g}:k={k:g}:{rot}"][variant]["HG"]) for p in SYN_P if f"mixed:p={p:g}:k={k:g}:{rot}" in syn]
    return pts


def equivalent_p(hg, pts):
    if not pts or hg is None or not np.isfinite(hg):
        return None
    if hg <= pts[0][1]:
        return 0.0
    for (p0, h0), (p1, h1) in zip(pts, pts[1:]):
        if h0 <= hg <= h1 and h1 > h0:
            return float(p0 + (hg - h0) / (h1 - h0) * (p1 - p0))
    return 1.0


_UNIT_CACHE = {}


def unit_share(tag, grp):
    """The strict unit statistic (experiment 13e/22) per layer: frac_place_heading_invariant_quadrant / frac_place."""
    path = "results/e4_internals/summary_quadrant.json" if grp == "conditions_e4" else "results/summary_quadrant.json"
    if path not in _UNIT_CACHE:
        _UNIT_CACHE[path] = json.load(open(path)) if os.path.exists(path) else {}
    layers = _UNIT_CACHE[path].get(tag, {})
    return {l: v["frac_place_heading_invariant_quadrant"] / v["frac_place"] for l, v in layers.items()
            if isinstance(v, dict) and v.get("frac_place")}


def verdict(s, val):
    """The registered decision rule: the fine-tuned model with the walk in context (run 0 and the four-epoch runs)."""
    if not val["validated"]:
        return {"verdict": "assay not validated: no model result is read as evidence either way", "validated": val["validated"]}
    runs = {t: s[t]["call"] for t in ["qwen8b-ft-walk"] + [f"qwen8b-ft-e4-s{k}-walk" for k in range(3, 8)] if t in s}
    calls = set(runs.values())
    v = "positive" if "positive" in calls else ("partial" if "partial" in calls else ("negative" if runs else None))
    return {"question": "does the fine-tuned model with the walk in context carry a heading-invariant population code?",
            "calls": runs, "four_epoch_runs_present": sum(t != "qwen8b-ft-walk" for t in runs), "verdict": v,
            "rerun_parity": parity(s)}


def parity(s):
    """vast-3 rerun of run 0 against vast-1's original: HG within 0.05 at every layer of walk and single."""
    out = {}
    for c in ("walk", "single"):
        a, b = s.get(f"qwen8b-ft-{c}"), s.get(f"qwen8b-ft-rerun-{c}")
        if a and b:
            d = {l: abs(a["HG_by_layer"][l] - b["HG_by_layer"][l]) for l in a["HG_by_layer"] if l in b["HG_by_layer"]}
            out[c] = {"max_abs_diff": max(d.values()), "meets": bool(max(d.values()) <= 0.05)}
    return out or None


def summarize(res):
    val = validation(res)
    res["validation"] = val
    res["validation_secondary_variants"] = {var: validation(res, var) for var in VARIANTS if var != PRIMARY}
    pts = calibration(res)
    s = {"_about": "Experiment 25 (docs/experiments.md). HG = (chance - error) on a heading the readout never saw, "
                   "divided by (chance - error) on the same frames for a readout trained on the 180 degrees centred on "
                   "their heading: the share of the seen-heading position information that survives an unseen heading "
                   "(1 = fully heading-general, 0 = chance, below 0 = worse than chance, e.g. mirrored). Errors are "
                   "median Euclidean errors in metres over all frames of room A's 48 walks. Per condition: best layer "
                   "= highest HG among layers whose seen-heading readout beats chance by >= 0.5 m. Calls: positive "
                   "HG >= 0.5, partial 0.25-0.5, negative < 0.25. equivalent_invariant_share: the share of invariant "
                   "place units in a synthetic mixture (2,000 units, noise k = 3) with the same HG. "
                   "unit_invariant_share: the strict unit statistic (experiment 13e) on the same population.",
         "validated": val["validated"],
         "validation_rule": {"positive_controls_HG": {k: (x or {}).get("HG") for k, x in val["positive_controls"].items()},
                             "negative_controls_max_HG": {k: (x or {}).get("max_HG") for k, x in val["negative_controls"].items()}},
         "calibration_k3": [{"p": p, "HG": h} for p, h in pts]}
    rows = []
    for grp in ("controls", "conditions", "conditions_e4"):
        for tag, layers in sorted(res.get(grp, {}).items()):
            if tag.startswith("rnn-"):                     # one row per mode (anchored, free), not a best "layer"
                rows += [(grp, f"{tag}:{mode}", {mode: r}) for mode, r in layers.items()]
            else:
                rows.append((grp, tag, layers))
    for grp, tag, layers in rows:
        c = condition_call(layers)
        if not c:
            continue
        c["group"] = grp
        c["equivalent_invariant_share"] = equivalent_p(c["HG"], pts) if c["call"] != "no position code" else None
        r = layers[c["best_layer"]]
        if "pv" in r:
            c["pv_index"] = r["pv"]["index"]
            c["pv_p_perm"] = r["pv"]["p_perm"]
        for var in VARIANTS:
            if var != PRIMARY:
                c[f"HG_{var}"] = r[var]["HG"]
                c[f"err_ho_{var}_m"] = r[var]["err_ho_m"]
        uq = unit_share(tag, grp)
        if uq:
            c["unit_invariant_share_at_best_layer"] = uq.get(c["best_layer"])
            c["unit_invariant_share_max"] = max(uq.values())
        s[tag] = c
    s["moving_average_comparison"] = {}
    for walk in [t for t in s if isinstance(s[t], dict) and t.endswith("-walk") and not t.endswith("probes-walk")]:
        m = walk[:-len("-walk")]
        if f"{m}-single+ema8" in s:
            w, e = s[walk]["HG"], s[f"{m}-single+ema8"]["HG"]
            s["moving_average_comparison"][m] = {"walk_HG": w, "single+ema8_HG": e, "ema_reaches_walk": bool(e >= w - 0.05)}
    s["verdict"] = verdict(s, val)
    syn = res.get("synthetic", {})
    for p in SYN_P:
        for rot in ("units", "rot"):
            key = f"mixed:p={p:g}:k={SYN_PRIMARY_K:g}:{rot}"
            if key in syn:
                r = syn[key]
                s[f"synthetic {key}"] = {"HG": r[PRIMARY]["HG"], "call": r[PRIMARY]["call"],
                                         "pv_index": r.get("pv", {}).get("index"),
                                         "unit_invariant_share": (r.get("unit_test") or {}).get("invariant_share_of_place")}
    res["summary"] = s
    return res


# ---------------------------------------------------------------- io
def save(res, path):
    tmp = path + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synthetic", "real", "summarize", "merge", "merge-gain"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=2400)
    ap.add_argument("--ks", type=float, nargs="*", default=list(SYN_K))
    ap.add_argument("--group", default="conditions", help="controls | conditions | conditions_e4")
    ap.add_argument("--tags", nargs="*", default=[], help="tag[:layer,layer] on room A's walks")
    ap.add_argument("--probe-tags", nargs="*", default=[], help="teleport-probe activations (probes_meta.npz)")
    ap.add_argument("--rnn", nargs="*", default=[], help="name:sub, e.g. base:1 s093:3")
    ap.add_argument("--rnn-root", default="data")
    ap.add_argument("--from", dest="src", help="merge: another results file")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--no-pv", action="store_true")
    ap.add_argument("--no-unit-test", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("POP_THREADS", torch.get_num_threads())))
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res["_about"] = {"experiment": "25 (docs/experiments.md)", "primary": PRIMARY, "variants": VARIANTS,
                     "lambdas": LAMBDAS.tolist(), "HG_pos": HG_POS, "min_code_m": MIN_CODE_M, "HG_neg": HG_NEG,
                     "mono_tol": MONO_TOL, "p_star": P_STAR}
    if args.mode in ("synthetic", "real"):
        import hashlib
        md5 = lambda f: hashlib.md5(open(f, "rb").read()).hexdigest()
        files = ["scripts/population_assay.py", "scripts/heading_stat_validation.py", "spatialcells/analyze.py",
                 "spatialcells/room.py", "spatialcells/grid_rnn.py", "data/walks/meta.npz", "data/walks/probes_meta.npz"]
        res.setdefault("inputs_md5", {})[f"{args.mode}:{args.group if args.mode == 'real' else args.seed}"] = {
            "host": os.uname().nodename, "time": time.strftime("%Y-%m-%d %H:%M %Z"),
            **{f: md5(f) for f in files if os.path.exists(f)}}
    if args.mode == "merge-gain":
        # the P4 add-on: experiment 21's gain index run elsewhere (heading_stat_validation.py --skip-synthetic --out SRC);
        # copy its real_units entries into results/heading_stat_validation.json, only tags/layers not there yet
        dst_path = "results/heading_stat_validation.json"
        src, dst = json.load(open(args.src)), json.load(open(dst_path))
        added = []
        for tag, layers in src.get("real_units", {}).items():
            for layer, r in layers.items():
                if layer not in dst.setdefault("real_units", {}).get(tag, {}):
                    dst["real_units"].setdefault(tag, {})[layer] = r
                    added.append(f"{tag}:{layer}")
        dst.setdefault("runtime_s", {})["exp25_addon_merge"] = time.strftime("%Y-%m-%d %H:%M %Z")
        save(dst, dst_path)
        print(f"merged {len(added)} layers into {dst_path}: {added}")
        return
    t0 = time.time()
    if args.mode == "synthetic":
        synthetic(args, res)
    elif args.mode == "real":
        real(args, res)
    elif args.mode == "merge":
        src = json.load(open(args.src))
        res.setdefault("inputs_md5", {}).update(src.get("inputs_md5", {}))
        for grp in ("synthetic", "controls", "conditions", "conditions_e4"):
            for tag, v in src.get(grp, {}).items():
                if isinstance(v, dict) and not tag.startswith("_") and grp != "synthetic":
                    res.setdefault(grp, {}).setdefault(tag, {}).update(v)
                elif grp == "synthetic":
                    res.setdefault(grp, {})[tag] = v
    summarize(res)
    save(res, args.out)
    print(f"wrote {args.out} ({time.time() - t0:.0f} s); validated: {res['validation']['validated']}", flush=True)


if __name__ == "__main__":
    main()
