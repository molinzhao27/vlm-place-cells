"""Follow-up pilot WP (docs/experiments.md): the analyses on top of wm_vint.py and wm_cut3r.py.

    python -m spatialcells.wm_pilot arena
    python -m spatialcells.analyze --quadrant --out results/wm_pilot/units/<model> --tags <model>-walk ...
    python -m spatialcells.analyze --quadrant --data data/walks-p256-arena --out results/wm_pilot/probes/<model> --tags ...
    python -m spatialcells.wm_pilot novelty --pairs vint-single:vint-probes-single cut3r-walk:cut3r-probes-walk ...
    python -m spatialcells.wm_pilot novelty --own-walk --pairs ... [--data D --arena-dir A --out F]     (amendment 2)
    python -m spatialcells.wm_pilot arena --data data/walks-roomReplicaApt0-p256 --arena-dir data/walks-roomReplicaApt0-p256-arena
    python -m spatialcells.wm_pilot twochoice --models vint nomad gnm
    python -m spatialcells.wm_pilot summary

  arena      data/walks-p256-arena: room A's 256-per-walk teleport probes as an analyze.py arena (walk = the probe's
             walk, step = its index; frames.npy links the probe images, so the pixels tag works). The probes come
             in random order, so analyze.py's circular shifts act as permutations.
  novelty    WP2: ridge (analyze.decode's lambda grid and walk-level validation split) trained on the walk frames
             of the even walks (TRAIN tag, all headings) and tested on the odd walks' probes (TEST tag), split into
             familiar probes (a walk frame within 1 m faced within 45 deg) and novel ones. Novelty cost = median
             novel error minus median familiar error. Baselines under the same protocol: pixels (walk-frame
             pixels -> probe pixels) and recall (walk frames' 8-step moving average of pixels -> the moving
             average at the walk frame most like the probe). 95% CIs from a paired bootstrap over the odd walks.
             Also the strict cross-sector decoding of the probes (13f's caveat) and, for CUT3R and TTT3R, the
             pose head's own probe positions. -> results/wm_pilot/novelty.json
  novelty --own-walk  amendment 2's WP2: familiarity as above, but every walk's probes are decoded by a ridge trained
             on that walk's own frames (novelty_own_walk), so the decoder has seen exactly the walk familiarity refers to.
             -> results/wm_pilot/novelty_ownwalk.json, or --out (the apartments: results/wm_pilot/apartments/<scene>/)
  twochoice  WP5 statistics from results/wm_pilot/twochoice_<model>.jsonl -> results/wm_pilot/twochoice.json
  summary    WP0-WP5 in one file with the reference numbers (path-integrating RNN, VGGT) ->
             results/wm_pilot/summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from .analyze import _ridge, causal_ema, decoding_suite
from .analyze_behave import NEAR, SAME_HEADING, angdiff, geometry
from .analyze_probes import pixel_feats
from .wm_common import bootstrap_ci

OUT = "results/wm_pilot"
ARENA = "data/walks-p256-arena"
P256 = "data/walks-p256"
LAMS = (1e1, 1e2, 1e3, 1e4, 1e5)


def arena(data=P256, out=ARENA):
    os.makedirs(out, exist_ok=True)
    p = dict(np.load(f"{data}/probes_meta.npz"))
    order = np.lexsort((p["idx"], p["walk"]))
    n = np.bincount(p["walk"])
    assert (order == np.arange(len(order))).all() and (n == n[0]).all()
    np.savez(f"{out}/meta.npz", walk=p["walk"], step=p["idx"], x=p["x"], y=p["y"], theta=p["theta"])
    src = os.path.relpath(data, out)
    for f, target in (("frames.npy", "probes.npy"), ("anchors.npz", "anchors.npz")):
        if not os.path.lexists(f"{out}/{f}") and os.path.exists(f"{data}/{target}"):
            os.symlink(f"{src}/{target}", f"{out}/{f}")
    print(f"{out}: {len(order)} probes, {len(n)} walks")


def fit_predict(Xtr, Ytr, walk_tr, Xte):
    """analyze.decode's ridge (standardized on the training rows, lambda picked on walks 0 mod 4 -> 2 mod 4),
    returning predictions for Xte instead of an error."""
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Ztr, Zte, ym = (Xtr - mu) / sd, (Xte - mu) / sd, Ytr.mean(0)
    a, b = torch.from_numpy(walk_tr % 4 == 0), torch.from_numpy(walk_tr % 4 == 2)
    errs = [((Ztr[b] @ _ridge(Ztr[a], Ytr[a] - ym, lam) + ym - Ytr[b]).norm(dim=1).median().item(), lam) for lam in LAMS]
    lam = min(errs)[1]
    return (Zte @ _ridge(Ztr, Ytr - ym, lam) + ym).numpy(), lam


def medians(err, fam, walks, ws):
    sel = np.isin(walks, ws)
    nov, famv = np.median(err[sel & ~fam]), np.median(err[sel & fam])
    return nov, famv, nov - famv


def novelty(pairs):
    meta = dict(np.load(f"{P256}/meta.npz"))
    _, pmeta, G = geometry(P256)
    fam = np.array([G[p]["familiar"] for p in range(len(pmeta["walk"]))])
    tr = meta["walk"] % 2 == 0
    te = pmeta["walk"] % 2 == 1
    Y = torch.tensor(np.stack([meta["x"], meta["y"]], 1)[tr], dtype=torch.float32)
    P = np.stack([pmeta["x"], pmeta["y"]], 1)[te]
    fam_te, walks_te = fam[te], pmeta["walk"][te]
    odd = np.unique(walks_te)
    res = json.load(open(f"{OUT}/novelty.json")) if os.path.exists(f"{OUT}/novelty.json") else {}
    res["_about"] = {"n_test_probes": int(te.sum()), "n_novel": int((~fam_te).sum()), "n_familiar": int(fam_te.sum()),
                     "unit": "median position error (m); CIs from a bootstrap over the 24 odd walks (10,000, seed 0)"}
    errs = {}

    def score(name, Xw, Xp, layer="pix"):
        pred, lam = fit_predict(torch.from_numpy(Xw[tr]), Y, meta["walk"][tr], torch.from_numpy(Xp[te]))
        e = np.hypot(*(pred - P).T)
        errs[name, layer] = e
        nov, famv, cost = medians(e, fam_te, walks_te, odd)
        res.setdefault(name, {})[layer] = {"novel": float(nov), "familiar": float(famv), "novelty_cost": float(cost),
                                           "lambda": lam,
                                           "novelty_cost_ci95": bootstrap_ci(lambda ws: medians(e, fam_te, walks_te, ws)[2], odd)}
        print(f"{name} {layer}: novel {nov:.2f} m, familiar {famv:.2f} m, cost {cost:+.2f} m", flush=True)

    W = pixel_feats(np.load(f"{P256}/frames.npy", mmap_mode="r"))
    Pp = pixel_feats(np.load(f"{P256}/probes.npy", mmap_mode="r"))
    mu, sd = W.mean(0), W.std(0) + 1e-6
    Wz, Pz = (W - mu) / sd, (Pp - mu) / sd
    score("pixels", Wz, Pz)
    Wema = causal_ema(torch.from_numpy(Wz), meta, 8).numpy()
    recall = np.empty_like(Pz)
    for w in np.unique(pmeta["walk"]):
        wi, pi = np.flatnonzero(meta["walk"] == w), np.flatnonzero(pmeta["walk"] == w)
        unit = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
        recall[pi] = Wema[wi[(unit(Pz[pi]) @ unit(Wz[wi]).T).argmax(1)]]
    score("pixels-recall-ema8", Wema, recall)

    pm_arena = {k: v for k, v in np.load(f"{ARENA}/meta.npz").items()}
    for pair in pairs:
        train, test = pair.split(":")
        layers = sorted({f[:-4] for f in os.listdir(f"data/acts/{train}") if f.startswith("L")}
                        & {f[:-4] for f in os.listdir(f"data/acts/{test}") if f.startswith("L")}, key=lambda s: int(s[1:]))
        for layer in layers:
            Xw = np.load(f"data/acts/{train}/{layer}.npy").astype(np.float32)
            Xp = np.load(f"data/acts/{test}/{layer}.npy").astype(np.float32)
            name = f"{train} -> {test}"
            score(name, Xw, Xp, layer)
            r = res[name][layer]
            for base, key in (("pixels-recall-ema8", "novel"), ("pixels", "novelty_cost")):
                eb = errs[base, "pix"]
                f = (lambda ws, e=errs[name, layer], eb=eb, k=key:
                     medians(e, fam_te, walks_te, ws)[0 if k == "novel" else 2] - medians(eb, fam_te, walks_te, ws)[0 if k == "novel" else 2])
                r[f"{key}_minus_{base}"] = float(f(odd))
                r[f"{key}_minus_{base}_ci95"] = bootstrap_ci(f, odd)
            r["beyond_view_recognition"] = bool(r["novel_minus_pixels-recall-ema8_ci95"][1] < 0
                                                and r["novelty_cost_minus_pixels_ci95"][1] < 0)
            strict = decoding_suite(torch.from_numpy(Xp), pm_arena, quadrant=True)
            r["probe_cross_quadrant"] = strict["cross_quadrant"]
            r["probe_within_quadrant"] = strict["within_quadrant"]
            r["probe_cross_quadrant_chance"] = strict["cross_quadrant_chance"]
            json.dump(res, open(f"{OUT}/novelty.json", "w"), indent=1)
        pose = f"data/acts/{test}/pose.npz"
        if os.path.exists(pose):                    # CUT3R / TTT3R: the pose head's own probe positions
            q = np.load(pose)
            e = np.hypot(q["x"] - pmeta["x"], q["y"] - pmeta["y"])
            nov, famv, cost = medians(e, fam, pmeta["walk"], np.unique(pmeta["walk"]))
            res[f"{test} pose head"] = {"novel": float(nov), "familiar": float(famv), "novelty_cost": float(cost),
                                        "all": float(np.median(e)), "within_1m": float((e < 1).mean())}
            print(f"{test} pose head: novel {nov:.2f} m, familiar {famv:.2f} m, all {np.median(e):.2f} m", flush=True)
    json.dump(res, open(f"{OUT}/novelty.json", "w"), indent=1)


def walk_rows(meta):
    """Walk -> its rows in step (or probe index) order."""
    key = "step" if "step" in meta else "idx"
    return {int(w): (lambda i: i[np.argsort(meta[key][i])])(np.flatnonzero(meta["walk"] == w)) for w in np.unique(meta["walk"])}


def own_walk_familiar(meta, pmeta):
    """analyze_behave.geometry's split without its pixel work: a probe is familiar if a frame of its own walk lies
    within NEAR (1 m) and faced within SAME_HEADING (45 deg) of it."""
    wr = walk_rows(meta)
    fam = np.zeros(len(pmeta["walk"]), bool)
    for w, pi in walk_rows(pmeta).items():
        fi = wr[w]
        d = np.hypot(meta["x"][fi][None] - pmeta["x"][pi][:, None], meta["y"][fi][None] - pmeta["y"][pi][:, None])
        fam[pi] = ((d < NEAR) & (angdiff(meta["theta"][fi][None], pmeta["theta"][pi][:, None]) < SAME_HEADING)).any(1)
    return fam


def ridge_pred(Xtr, Ytr, Xte, lam):
    """analyze.decode's ridge (standardized on the training rows), returning predictions."""
    mu, sd, ym = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6), Ytr.mean(0)
    return ((Xte - mu) / sd) @ _ridge((Xtr - mu) / sd, Ytr - ym, lam) + ym


def own_walk_errors(Xw, Xp, meta, pmeta):
    """Amendment 2's WP2 decoder: for every walk, ridge from its own walk frames to (x, y), tested on its own probes,
    so that the decoder has seen exactly the walk that familiarity is defined against (and that a model's state
    carries in the probes-walk condition). Lambda (LAMS) is picked once per representation: within each walk the first
    half of the steps predicts the second half and vice versa, median error over all walks."""
    wr, pr = walk_rows(meta), walk_rows(pmeta)
    Xw, Xp = torch.as_tensor(Xw, dtype=torch.float32), torch.as_tensor(Xp, dtype=torch.float32)
    Y = torch.tensor(np.stack([meta["x"], meta["y"]], 1), dtype=torch.float32)
    P = np.stack([pmeta["x"], pmeta["y"]], 1)
    val = {}
    for lam in LAMS:
        e = []
        for fi in wr.values():
            a, b = fi[:len(fi) // 2], fi[len(fi) // 2:]
            for tr, va in ((a, b), (b, a)):
                e.append((ridge_pred(Xw[tr], Y[tr], Xw[va], lam) - Y[va]).norm(dim=1).numpy())
        val[lam] = float(np.median(np.concatenate(e)))
    lam = min(val, key=val.get)
    err = np.full(len(P), np.nan)
    for w, pi in pr.items():
        fi = wr[w]
        err[pi] = np.hypot(*(ridge_pred(Xw[fi], Y[fi], Xp[pi], lam).numpy() - P[pi]).T)
    return err, lam


def novelty_own_walk(pairs, data=P256, arena_dir=ARENA, out=f"{OUT}/novelty_ownwalk.json"):
    """WP2 with amendment 2's decoder (own_walk_errors); familiarity as registered (the probe's own walk). Novelty
    cost = median error on novel probes minus that on familiar ones, pooled over every walk's probes; baselines pixels
    and recall (the 8-step moving average of pixels at the own-walk frame that looks most like the probe) under the
    same protocol; 95% CIs from a paired bootstrap over walks (resampled with repetition, 10,000, seed 0). Also, as
    in novelty(): strict cross-sector decoding of the probes and, for CUT3R and TTT3R, the pose head's own probe
    positions."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta, pmeta = dict(np.load(f"{data}/meta.npz")), dict(np.load(f"{data}/probes_meta.npz"))
    fam = own_walk_familiar(meta, pmeta)
    pr = walk_rows(pmeta)
    walks = np.array(sorted(pr))
    res = json.load(open(out)) if os.path.exists(out) else {}
    res["_about"] = {"data": data, "n_probes": len(fam), "n_novel": int((~fam).sum()), "n_familiar": int(fam.sum()),
                     "n_walks": len(walks), "decoder": own_walk_errors.__doc__.split("\n")[0],
                     "unit": "median position error (m); CIs from a bootstrap over walks (10,000, seed 0)"}
    errs = {}

    def stats(e, ws):
        idx = np.concatenate([pr[w] for w in ws])
        nov, famv = np.median(e[idx][~fam[idx]]), np.median(e[idx][fam[idx]])
        return nov, famv, nov - famv

    def score(name, layer, Xw, Xp):
        e, lam = own_walk_errors(Xw, Xp, meta, pmeta)
        errs[name, layer] = e
        nov, famv, cost = stats(e, walks)
        r = res.setdefault(name, {})[layer] = {
            "novel": float(nov), "familiar": float(famv), "all": float(np.median(e)), "novelty_cost": float(cost),
            "lambda": lam, "novelty_cost_ci95": bootstrap_ci(lambda ws: stats(e, ws)[2], walks)}
        print(f"{name} {layer}: novel {nov:.2f} m, familiar {famv:.2f} m, cost {cost:+.2f} m {r['novelty_cost_ci95']}", flush=True)
        return r

    W = pixel_feats(np.load(f"{data}/frames.npy", mmap_mode="r"))
    Pp = pixel_feats(np.load(f"{data}/probes.npy", mmap_mode="r"))
    mu, sd = W.mean(0), W.std(0) + 1e-6
    Wz, Pz = (W - mu) / sd, (Pp - mu) / sd
    score("pixels", "pix", Wz, Pz)
    Wema = causal_ema(torch.from_numpy(Wz), meta, 8).numpy()
    recall = np.empty_like(Pz)
    unit = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
    for w, pi in pr.items():
        wi = np.flatnonzero(meta["walk"] == w)
        recall[pi] = Wema[wi[(unit(Pz[pi]) @ unit(Wz[wi]).T).argmax(1)]]
    score("pixels-recall-ema8", "pix", Wema, recall)

    pm_arena = dict(np.load(f"{arena_dir}/meta.npz"))
    for pair in pairs:
        train, test = pair.split(":")
        layers = sorted({f[:-4] for f in os.listdir(f"data/acts/{train}") if f.startswith("L")}
                        & {f[:-4] for f in os.listdir(f"data/acts/{test}") if f.startswith("L")}, key=lambda s: int(s[1:]))
        name = f"{train} -> {test}"
        for layer in layers:
            Xp = np.load(f"data/acts/{test}/{layer}.npy").astype(np.float32)
            r = score(name, layer, np.load(f"data/acts/{train}/{layer}.npy").astype(np.float32), Xp)
            e = errs[name, layer]
            for base, k in (("pixels-recall-ema8", 0), ("pixels", 2)):
                eb = errs[base, "pix"]
                f = lambda ws, e=e, eb=eb, k=k: stats(e, ws)[k] - stats(eb, ws)[k]
                key = f"{'novel' if k == 0 else 'novelty_cost'}_minus_{base}"
                r[key], r[f"{key}_ci95"] = float(f(walks)), bootstrap_ci(f, walks)
            r["beyond_view_recognition"] = bool(r["novel_minus_pixels-recall-ema8_ci95"][1] < 0
                                                and r["novelty_cost_minus_pixels_ci95"][1] < 0)
            strict = decoding_suite(torch.from_numpy(Xp), pm_arena, quadrant=True)
            r.update(probe_cross_quadrant=strict["cross_quadrant"], probe_within_quadrant=strict["within_quadrant"],
                     probe_cross_quadrant_chance=strict["cross_quadrant_chance"])
            json.dump(res, open(out, "w"), indent=1)
        pose = f"data/acts/{test}/pose.npz"
        if os.path.exists(pose):                    # CUT3R / TTT3R: the pose head's own probe positions
            q = np.load(pose)
            e = np.hypot(q["x"] - pmeta["x"], q["y"] - pmeta["y"])
            nov, famv, cost = stats(e, walks)
            res[f"{test} pose head"] = {"novel": float(nov), "familiar": float(famv), "novelty_cost": float(cost),
                                        "novelty_cost_ci95": bootstrap_ci(lambda ws: stats(e, ws)[2], walks),
                                        "all": float(np.median(e)), "within_1m": float((e < 1).mean())}
            print(f"{test} pose head: novel {nov:.2f} m, familiar {famv:.2f} m, all {np.median(e):.2f} m", flush=True)
    json.dump(res, open(out, "w"), indent=1)


def binom_ci(k, n):
    from scipy.stats import binomtest
    ci = binomtest(int(round(k)), n).proportion_ci(method="exact")
    return [float(ci.low), float(ci.high)]


def twochoice(models):
    from scipy.stats import binomtest
    res = {}
    P_ROOMS = [f"roomP{5000 + i}" for i in range(10)]
    APTS = [f"roomReplicaApt{i}" for i in range(3)]
    HOUSES = [f"roomHSSD{i}" for i in range(1, 9)]
    for m in models:
        path = f"{OUT}/twochoice_{m}.jsonl"
        if not os.path.exists(path):
            continue
        recs = [json.loads(l) for l in open(path)]
        by = {}
        for r in recs:
            by.setdefault(r["set"], []).append(r)
        acc = lambda rs, c: float(np.mean([r[c]["correct"] for r in rs])) if rs else float("nan")
        out = {"n_trials": len(recs)}
        for c in ("walk", "nowalk", "reverse"):
            o = {}
            for s in ("dissimilar", "roomB"):
                rs = by.get(s, [])
                nov = [r for r in rs if not r["familiar"]]
                o[s] = {"all": acc(rs, c), "all_ci95": binom_ci(sum(r[c]["correct"] for r in rs), len(rs)),
                        "novel": acc(nov, c), "novel_ci95": binom_ci(sum(r[c]["correct"] for r in nov), len(nov)),
                        "n": len(rs), "n_novel": len(nov)}
            per = np.array([acc([r for r in by.get(s, []) if not r["familiar"]], c) for s in P_ROOMS])
            if np.isfinite(per).all():
                above, below = int((per > 0.5).sum()), int((per < 0.5).sum())
                o["P rooms novel"] = {"mean": float(per.mean()), "ci95": bootstrap_ci(np.mean, per), "per_room": per.tolist(),
                                      "rooms_above_below": [above, below],
                                      "sign_p": float(binomtest(above, above + below).pvalue) if above + below else 1.0}
            for label, sets in (("apartments", APTS), ("houses", HOUSES)):
                rs = [r for s in sets for r in by.get(s, [])]
                if len(rs):
                    k = sum(r[c]["correct"] for r in rs)
                    per = np.array([acc(by.get(s, []), c) for s in sets])
                    o[f"{label} all"] = {"pooled": k / len(rs), "ci95": binom_ci(k, len(rs)), "n": len(rs),
                                         "per_scene": per.tolist(), "novel_pooled": acc([r for r in rs if not r["familiar"]], c)}
                    if label == "houses":
                        o[f"{label} all"]["house_mean_ci95"] = bootstrap_ci(np.mean, per)
            out[c] = o
        d = np.array([acc([r for r in by.get(s, []) if not r["familiar"]], "walk")
                      - acc([r for r in by.get(s, []) if not r["familiar"]], "nowalk") for s in P_ROOMS])
        if np.isfinite(d).all():
            out["P rooms novel, walk minus no walk"] = {"mean": float(d.mean()), "ci95": bootstrap_ci(np.mean, d),
                                                         "uses_walk": bool(bootstrap_ci(np.mean, d)[0] > 0)}
        res[m] = out
        w = out["walk"]
        print(f"{m}: room A novel {w['dissimilar']['novel']:.1%}, P rooms novel {w.get('P rooms novel', {}).get('mean', np.nan):.1%} "
              f"{w.get('P rooms novel', {}).get('ci95')}, apartments {w.get('apartments all', {}).get('pooled', np.nan):.1%}, "
              f"houses {w.get('houses all', {}).get('pooled', np.nan):.1%}; no walk P novel "
              f"{out['nowalk'].get('P rooms novel', {}).get('mean', np.nan):.1%}", flush=True)
    json.dump(res, open(f"{OUT}/twochoice.json", "w"), indent=1)


def shares(path):
    """Per tag: the share of place units strictly heading-invariant at every layer, and the best layer."""
    if not os.path.exists(path):
        return {}
    S = json.load(open(path))
    out = {}
    for tag, layers in S.items():
        if tag.startswith("_"):
            continue
        sh = {l: (v["frac_place_heading_invariant_quadrant"] / v["frac_place"] if v["frac_place"] > 0 else float("nan"))
              for l, v in layers.items()}
        ok = {l: s for l, s in sh.items() if np.isfinite(s)}
        best = max(ok, key=ok.get) if ok else None
        out[tag] = {"best_layer": best, "best_share": ok.get(best, float("nan")),
                    "frac_place_at_best": layers[best]["frac_place"] if best else float("nan"),
                    "n_place_at_best": round(layers[best]["frac_place"] * layers[best]["n_poles_active"]) if best else 0,
                    "share_by_layer": sh, "frac_place_by_layer": {l: v["frac_place"] for l, v in layers.items()}}
    return out


def label(share):
    return "place-like" if share >= 0.45 else "view-like" if share <= 0.10 else "intermediate"


def summary():
    res = {"gates": {os.path.basename(p)[5:-5]: json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/gate_*.json"))}}
    for tag in ("cut3r-walk", "cut3r-single", "ttt3r-walk", "ttt3r-single"):
        p = f"data/acts/{tag}/pose.npz"
        if os.path.exists(p):
            q, m = np.load(p), np.load("data/walks/meta.npz")
            e = np.hypot(q["x"] - m["x"], q["y"] - m["y"])
            res["gates"][f"{tag} anchored"] = {"median_error_m": float(np.nanmedian(e)),
                                               "anchor_fit_rms_m": float(q["anchor_fit_rms"])}
    walk, probes = {}, {}
    for f in sorted(glob.glob(f"{OUT}/units/*/summary_quadrant.json")):
        walk.update(shares(f))
    for f in sorted(glob.glob(f"{OUT}/probes/*/summary_quadrant.json")):
        probes.update(shares(f))
    for d in (walk, probes):
        for v in d.values():
            v["label"] = label(v["best_share"]) if np.isfinite(v["best_share"]) else None
    res["WP1 along the walk"] = walk
    res["WP1 teleport probes"] = probes
    ref = {}
    rnn = json.load(open("results/grid_rnn_quadrant.json"))
    for run, conds in rnn.items():
        for c, v in conds.items():
            if isinstance(v, dict) and "frac_place" in v:
                ref[f"RNN {run} {c}"] = v["frac_place_heading_invariant_quadrant"] / v["frac_place"]
    ref.update({f"{k} (exp 13e)": v for k, v in
                {t: s["best_share"] for t, s in shares("results/summary_quadrant.json").items()
                 if t in ("vggt-single", "vggt-walk", "vggt-alone", "pixels", "dinov2", "qwen-walk", "qwen-single+ema8")}.items()})
    res["references_along_the_walk"] = ref
    for name in ("novelty", "novelty_ownwalk", "twochoice"):
        if os.path.exists(f"{OUT}/{name}.json"):
            res[{"novelty": "WP2", "novelty_ownwalk": "WP2 own-walk decoder (amendment 2)", "twochoice": "WP5"}[name]] = \
                json.load(open(f"{OUT}/{name}.json"))
    apts = {}
    for d in sorted(glob.glob(f"{OUT}/apartments/*/")):              # amendment 2: TTT3R in the photoreal apartments
        scene = os.path.basename(d.rstrip("/"))
        a = {}
        for k in ("units", "probes"):                                # one folder per model, as units/ and probes/
            a[k] = {}
            for f in sorted(glob.glob(f"{d}{k}/*/summary_quadrant.json")):
                a[k].update(shares(f))
        for v in (*a["units"].values(), *a["probes"].values()):
            v["label"] = label(v["best_share"]) if np.isfinite(v["best_share"]) else None
        a = {"WP1 along the walk": a["units"], "WP1 teleport probes": a["probes"]}
        for f in sorted(glob.glob(f"{d}novelty_ownwalk*.json")):
            a.setdefault("WP2 own-walk decoder", {}).update(json.load(open(f)))
        if os.path.exists(f"{OUT}/gate_ttt3r_{scene}.json"):
            a["WP0 gate"] = json.load(open(f"{OUT}/gate_ttt3r_{scene}.json"))
        pr = a["WP1 teleport probes"]
        pw, ps, px = (pr.get(t) for t in (f"ttt3r-probes-walk-{scene}", f"ttt3r-probes-single-{scene}", "pixels"))
        if pw and ps and px:                     # amendment 2's comparison rule (12:00 PDT): one layer, 10 points
            L = pw["best_layer"]
            sw, ss, sp = pw["share_by_layer"][L], ps["share_by_layer"].get(L, float("nan")), px["best_share"]
            vg = pr.get(f"vggt-single-probes-{scene}")
            a["comparison rule"] = {
                "layer": L, "probes_walk": sw, "probes_single": ss, "pixels": sp,
                "walk_minus_single": sw - ss, "walk_minus_pixels": sw - sp,
                "n_place_probes_walk": pw["n_place_at_best"],
                "state_adds_place_code": bool(sw - ss >= 0.10 and sw - sp >= 0.10),
                "vggt_minus_pixels": (vg["best_share"] - sp) if vg else None,
                "vggt_clears_floor": bool(vg["best_share"] - sp >= 0.10) if vg else None}
        apts[scene] = a
    if apts:
        res["apartments (amendment 2)"] = apts
    for p in sorted(glob.glob(f"{OUT}/patch_*.json")):
        res[f"WP4 {os.path.basename(p)[6:-5]}"] = json.load(open(p))
    json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
    print("WP1, share of place units strictly heading-invariant (best layer):")
    for name, d in (("walk frames", walk), ("probes", probes)):
        for t, v in d.items():
            print(f"  {name:11s} {t:28s} {v['best_layer']:>4s} {v['best_share']:6.1%}  (place {v['frac_place_at_best']:.1%}, "
                  f"{v['n_place_at_best']} units)  {v['label']}")
    print("references: " + ", ".join(f"{k} {v:.0%}" for k, v in ref.items()))
    for scene, a in apts.items():
        for name in ("WP1 along the walk", "WP1 teleport probes"):
            for t, v in a[name].items():
                print(f"  {scene} {name[4:]:15s} {t:36s} {v['best_layer']:>4s} {v['best_share']:6.1%}  "
                      f"(place {v['frac_place_at_best']:.1%}, {v['n_place_at_best']} units)  {v['label']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["arena", "novelty", "twochoice", "summary"])
    ap.add_argument("--pairs", nargs="*", default=[], help="novelty: TRAIN_TAG:TEST_TAG")
    ap.add_argument("--models", nargs="*", default=["vint", "nomad", "gnm"])
    ap.add_argument("--own-walk", action="store_true", help="novelty: amendment 2's own-walk decoder (novelty_own_walk)")
    ap.add_argument("--data", default=P256, help="arena, novelty --own-walk: the walks and probes")
    ap.add_argument("--arena-dir", default=ARENA, help="arena: output; novelty --own-walk: the probe arena")
    ap.add_argument("--out", default=f"{OUT}/novelty_ownwalk.json", help="novelty --own-walk: output")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    nov = (lambda: novelty_own_walk(args.pairs, args.data, args.arena_dir, args.out)) if args.own_walk else lambda: novelty(args.pairs)
    {"arena": lambda: arena(args.data, args.arena_dir), "novelty": nov, "twochoice": lambda: twochoice(args.models),
     "summary": summary}[args.what]()


if __name__ == "__main__":
    main()
