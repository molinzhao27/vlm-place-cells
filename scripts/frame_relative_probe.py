"""Experiment 26 (proposal P12): is the new view's position coded in walk-relative coordinates while the model
does the task? Experiment 14a's readout, with the target moved into the walk's own frame (CPU only).

    .venv-mac/bin/python scripts/frame_relative_probe.py [--models qwen8b qwen8b+qwen8b-proc] [--no-mlp]
        [--sites ...] [--layers 20 ...] [--targets ...] [--out results/frame_relative_probe.json] [--threads 6]
    .venv-mac/bin/python scripts/frame_relative_probe.py --selftest      # synthetic activations, no recording
    .venv-mac/bin/python scripts/frame_relative_probe.py --exploratory   # after the run; not registered (see exploratory())

Reads experiment 14a's recordings (data/acts_task/<model>/, spatialcells/task_record.py: the residual stream
after every decoder layer at the new view's image-end token, the mean of its visual tokens, the question's last
token and the last prompt token, for all 1,920 lure-free trials of P5000-P5009 under the exact experiment-10
prompt). The protocol is task_probe.py's, using its own ridge_fit, mlp_fit and r2: for every model, site and
layer, each readout is trained on five rooms and tested on the other five, both ways (P5000-P5004 ->
P5005-P5009 and back), and the two directions are averaged; inside the five training rooms the last one is the
validation room (the ridge's lambda; the MLP's early stopping). Targets, from the trial's own walk (two_choice's
walk_order; the 64-frame prompt always shows frame 1 and frame 256, the walk's first and last steps):
  world_xy       the new view's (x, y) in room coordinates: experiment 14a's target, recomputed as a check
  first_xy       its position in the walk's first-frame frame: translated to frame 1's position and rotated by
                 frame 1's heading, so frame 1 faces +x (x forward, y to the left); in m
  last_xy        the same in frame 256's frame (the walk's last frame, shown just before "New view:")
  first_heading  its heading relative to frame 1's, as (cos, sin)
Walk-shuffled controls (<target>_shuf<seed>): the same targets computed from another walk of the same room (a
random derangement of the room's walks, seeds 0, 1 and 2). They keep the room's geometry and the targets'
distribution but break the tie to the walk the model saw; reported per seed and as the mean (<target>_shuf_r2)
and max over seeds.
Metrics: R^2 (1 - SS_res / SS_tot about the test rooms' mean, averaged over the two columns) and the median
error: Euclidean in m for positions, the absolute angle between atan2 of the predicted and the true (sin, cos)
in degrees for the heading. Chance: predicting the training rooms' mean (R^2 and median error).
Writes results/frame_relative_probe.json: per model, site, layer and readout; _chance; _exp14a (the stored
world-frame values from results/task_probe.json, and the largest difference from the recomputation here); and a
top-level summary with the registered decision rule's verdicts (docs/experiments.md, experiment 26).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from spatialcells.task_probe import mean_dicts, mlp_fit, r2, ridge_fit  # noqa: E402
from spatialcells.two_choice import walk_order  # noqa: E402

OUT = "results/frame_relative_probe.json"
EXP14A = "results/task_probe.json"
SEEDS = (0, 1, 2)
FRAMES = ("first_xy", "last_xy", "first_heading")         # the registered walk-relative targets
TARGETS = ("world_xy",) + FRAMES
R2_ABSENT, R2_PRESENT, MARGIN = 0.10, 0.30, 0.20        # the registered decision rule
WALK_LEN = 256


def rel(x, y, th, ox, oy, oth):
    """(x, y) in the frame of a camera at (ox, oy) facing oth: x forward, y to its left."""
    dx, dy = x - ox, y - oy
    c, s = np.cos(oth), np.sin(oth)
    return np.stack([c * dx + s * dy, -s * dx + c * dy], 1)


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if (p != np.arange(n)).all():
            return p


def unit(name):
    return "deg" if name.startswith("first_heading") else "m"


def load(model):
    """Recorded rows, room per row, and every target (true and walk-shuffled) as float32 tensors."""
    d = f"data/acts_task/{model}"
    meta = json.load(open(f"{d}/meta.json"))
    rows = np.array(sorted({json.loads(l)["row"] for l in open(f"{d}/done.jsonl")}), int)
    pick = lambda k, dt=None: np.asarray(meta[k], dt)[rows]
    room, walk = pick("room"), pick("walk", int)
    x, y, th = pick("x", float), pick("y", float), pick("theta", float)
    a, b = pick("a", int), pick("b", int)
    poses = {}                                          # room -> walk -> (frame 1's pose, frame 256's pose)
    for R in sorted(set(room)):
        m = dict(np.load(f"data/walks-room{R}/meta.npz"))
        order = walk_order(m)
        assert all(len(fi) == WALK_LEN for fi in order.values()), R
        poses[R] = {w: tuple((float(m["x"][fi[k]]), float(m["y"][fi[k]]), float(m["theta"][fi[k]])) for k in (0, -1))
                    for w, fi in order.items()}
    # Frames 1 and 256 are always in the 64-frame prompt (two_choice.present: linspace(1, 256, 61) plus a and b).
    keep = set(np.linspace(1, WALK_LEN, 61).round().astype(int).tolist())
    assert {1, WALK_LEN} <= keep and ((a >= 1) & (a <= WALK_LEN) & (b >= 1) & (b <= WALK_LEN)).all()

    def targets(walk_of):
        f = np.array([poses[R][w][0] for R, w in zip(room, walk_of)])
        l = np.array([poses[R][w][1] for R, w in zip(room, walk_of)])
        return {"first_xy": rel(x, y, th, *f.T), "last_xy": rel(x, y, th, *l.T),
                "first_heading": np.stack([np.cos(th - f[:, 2]), np.sin(th - f[:, 2])], 1)}

    T = {"world_xy": np.stack([x, y], 1), **targets(walk)}
    for s in SEEDS:
        rng = np.random.default_rng(s)
        w2 = walk.copy()
        for R in sorted(set(room)):
            ws = np.array(sorted(poses[R]))
            perm = dict(zip(ws.tolist(), ws[derangement(len(ws), rng)].tolist()))
            w2[room == R] = [perm[w] for w in walk[room == R]]
        assert (w2 != walk).all()
        T.update({f"{k}_shuf{s}": v for k, v in targets(w2).items()})
    return meta, rows, room, {k: torch.tensor(v, dtype=torch.float32) for k, v in T.items()}


def metrics(name, pred, y):
    if unit(name) == "deg":
        da = torch.atan2(pred[:, 1], pred[:, 0]) - torch.atan2(y[:, 1], y[:, 0])
        err = torch.rad2deg(torch.atan2(torch.sin(da), torch.cos(da)).abs())
    else:
        err = (pred - y).norm(dim=1)
    return {"r2": r2(pred, y), f"median_{unit(name)}": float(err.median())}


def split_rows(room, split):
    tr_rooms, te_rooms = split
    in_tr, in_te = np.isin(room, tr_rooms), np.isin(room, te_rooms)
    is_val = room == tr_rooms[-1]
    idx = lambda m: torch.from_numpy(np.flatnonzero(m))
    return idx(in_tr & ~is_val), idx(is_val), idx(in_tr), idx(in_te)


def evaluate(X, room, T, names, split, readout):
    fit, val, train, test = split_rows(room, split)
    fitter = ridge_fit if readout == "ridge" else mlp_fit
    o = {}
    for k in names:
        p = fitter(X, T[k], fit, val, train)(test)
        o.update({f"{k}_{m}": v for m, v in metrics(k, p, T[k][test]).items()})
    return o


def chance(room, T, names, split):
    _, _, train, test = split_rows(room, split)
    o = {}
    for k in names:
        p = T[k][train].mean(0, keepdim=True).expand(len(test), -1)
        o.update({f"{k}_{m}": v for m, v in metrics(k, p, T[k][test]).items()})
    return o


def add_shuffle_means(d):
    """<frame>_shuf_r2 / _shuf_r2_max / _shuf_median_<unit>: over the shuffle seeds present."""
    for f in FRAMES:
        ks = [f"{f}_shuf{s}_r2" for s in SEEDS if f"{f}_shuf{s}_r2" in d]
        if ks:
            d[f"{f}_shuf_r2"] = float(np.mean([d[k] for k in ks]))
            d[f"{f}_shuf_r2_max"] = float(np.max([d[k] for k in ks]))
            u = unit(f)
            d[f"{f}_shuf_median_{u}"] = float(np.mean([d[f"{f}_shuf{s}_median_{u}"] for s in SEEDS
                                                       if f"{f}_shuf{s}_median_{u}" in d]))
    return d


def run(args):
    out = args.out
    targets = args.targets
    names = [t for t in targets] + [f"{t}_shuf{s}" for t in targets if t in FRAMES for s in SEEDS]
    readouts = ["ridge"] if args.no_mlp else ["ridge", "mlp"]
    models = args.models or sorted(m for m in os.listdir("data/acts_task") if os.path.exists(f"data/acts_task/{m}/meta.json"))
    res = json.load(open(out)) if os.path.exists(out) else {}
    res["_about"] = __doc__.split("\n\n", 2)[2].strip()
    t0 = time.time()
    for model in models:
        meta, rows, room, T = load(model)
        rooms = sorted(set(room))
        if len(rows) < len(meta["id"]):
            print(f"{model}: {len(rows)}/{len(meta['id'])} trials recorded; using those", flush=True)
        h = len(rooms) // 2
        splits = [(rooms[:h], rooms[h:]), (rooms[h:], rooms[:h])]
        res.setdefault("_split", {})[model] = {"rooms": [rooms[:h], rooms[h:]], "validation_room": [s[0][-1] for s in splits],
                                               "n_trials": int(len(rows))}
        res.setdefault("_chance", {})[model] = add_shuffle_means(mean_dicts([chance(room, T, names, s) for s in splits]))
        res.setdefault(model, {})
        layers = [l for l in meta["layers"] if not args.layers or l in args.layers]
        for site in args.sites:
            A = np.load(f"data/acts_task/{model}/{site}.npy", mmap_mode="r")
            S = res[model].setdefault(site, {})
            for j, layer in enumerate(meta["layers"]):
                if layer not in layers:
                    continue
                cell = S.get(f"L{layer}", {})
                if all(ro in cell and all(f"{k}_r2" in cell[ro] for k in names) for ro in readouts):
                    continue                                   # done in an earlier (stopped) run
                X = torch.from_numpy(np.asarray(A[rows, j], np.float32))
                for ro in readouts:
                    cell[ro] = add_shuffle_means({**cell.get(ro, {}),
                                                  **mean_dicts([evaluate(X, room, T, names, s, ro) for s in splits])})
                S[f"L{layer}"] = cell
            json.dump(res, open(out, "w"), indent=1)
            print(f"{model} {site}: {len(layers)} layers, {time.time() - t0:.0f}s", flush=True)
    res["_exp14a"] = exp14a_reference(res, models)
    res["summary"] = summarise(res, models)
    json.dump(res, open(out, "w"), indent=1)
    show(res["summary"])


def cells(res, model, key):
    return [(site, L, ro, v[ro]) for site, by in res[model].items() for L, v in by.items() for ro in v
            if key in v[ro]]


def exp14a_reference(res, models):
    """Experiment 14a's stored world-frame (x, y) readouts, and the largest difference from this recomputation."""
    if not os.path.exists(EXP14A):
        return {}
    q = json.load(open(EXP14A))
    o = {}
    for model in models:
        if model not in q:
            continue
        stored = [(s, L, ro, v[ro]) for s, by in q[model].items() for L, v in by.items() for ro in v]
        best = max(stored, key=lambda c: c[3]["xy_r2"])
        low = min(stored, key=lambda c: c[3]["xy_median_m"])
        diffs = [abs(res[model][s][L][ro]["world_xy_r2"] - v["xy_r2"]) for s, L, ro, v in stored
                 if "world_xy_r2" in res[model].get(s, {}).get(L, {}).get(ro, {})]
        o[model] = {"best_r2": best[3]["xy_r2"], "best_r2_at": " ".join(best[:3]),
                    "best_median_m": low[3]["xy_median_m"], "best_median_at": " ".join(low[:3]),
                    "chance_median_m": q["_chance"][model]["xy_median_m"],
                    "recomputed_cells": len(diffs), "recomputed_max_abs_diff_r2": max(diffs) if diffs else None}
    return o


def verdict(cs, f):
    """The registered rule for one model and target (cs: its cells)."""
    best = max(c[3][f"{f}_r2"] for c in cs)
    hits = [c for c in cs if c[3][f"{f}_r2"] >= R2_PRESENT and c[3][f"{f}_r2"] - c[3][f"{f}_shuf_r2"] >= MARGIN]
    if hits:
        return "present"
    if best <= R2_ABSENT:
        return "absent"
    return "not walk-specific" if best >= R2_PRESENT else "weak"


def summarise(res, models):
    S = {"rule": (f"Per model and target, over every site, layer and readout: 'absent' if the best cross-room R^2 is at "
                  f"most {R2_ABSENT}; 'present' if some cell reaches R^2 >= {R2_PRESENT} and beats the mean of its "
                  f"three walk-shuffled controls (same model, site, layer, readout) by >= {MARGIN}; 'not walk-specific' "
                  f"if R^2 >= {R2_PRESENT} somewhere but no such cell beats its control by {MARGIN}; 'weak' otherwise. "
                  f"first_xy is the primary target; last_xy and first_heading are secondary, under the same rule."),
         "units": "r2 values are fractions (R^2); *_m in meters; *_deg in degrees", "models": {}}
    for model in models:
        if model not in res:
            continue
        ch, M = res["_chance"][model], {}
        for f in ("world_xy",) + FRAMES:
            cs = cells(res, model, f"{f}_r2")
            if not cs:
                continue
            u = unit(f)
            b = max(cs, key=lambda c: c[3][f"{f}_r2"])
            lo = min(cs, key=lambda c: c[3][f"{f}_median_{u}"])
            e = {"best_r2": b[3][f"{f}_r2"], "best_r2_at": " ".join(b[:3]), f"best_r2_median_{u}": b[3][f"{f}_median_{u}"],
                 f"best_median_{u}": lo[3][f"{f}_median_{u}"], "best_median_at": " ".join(lo[:3]),
                 "chance_r2": ch[f"{f}_r2"], f"chance_median_{u}": ch[f"{f}_median_{u}"], "n_cells": len(cs)}
            for ro in ("ridge", "mlp"):
                c = [x for x in cs if x[2] == ro]
                if c:
                    e[f"best_r2_{ro}"] = max(x[3][f"{f}_r2"] for x in c)
            if f in FRAMES:
                e["shuf_r2_at_best"] = b[3][f"{f}_shuf_r2"]
                e["margin_at_best"] = e["best_r2"] - e["shuf_r2_at_best"]
                e["shuf_best_r2"] = max(c[3][f"{f}_shuf_r2"] for c in cs)
                e["shuf_best_r2_any_seed"] = max(c[3][f"{f}_shuf_r2_max"] for c in cs)
                e[f"shuf_best_median_{u}"] = min(c[3][f"{f}_shuf_median_{u}"] for c in cs)
                e["chance_shuf_r2"] = ch[f"{f}_shuf_r2"]
                e["verdict"] = verdict(cs, f)
            M[f] = e
        S["models"][model] = M
    for f in FRAMES:
        vs = {m: S["models"][m][f]["verdict"] for m in S["models"] if f in S["models"][m]}
        if vs:
            S[f"verdict_{f}"] = (next(iter(vs.values())) if len(set(vs.values())) == 1
                                 else "; ".join(f"{m}: {v}" for m, v in vs.items()))
            S[f"best_r2_{f}_all_models"] = max(S["models"][m][f]["best_r2"] for m in vs)
    if "exploratory" in res:
        S["exploratory"] = res["exploratory"]["summary"]
    return S


def r2_cols(pred, y):
    return (1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-12)).tolist()


def exploratory(args):
    """Not registered; run after the registered readouts (--exploratory). (1) At each model's and target's best ridge
    cell, the R^2 of each column separately (forward / left for positions, cos / sin for the heading), for the true
    target and the mean of the three shuffles. (2) An appearance baseline with no model in it: the cosine similarity of
    DINOv2-base CLS features (as experiment 13d feeds DINOv2) between the new view and frame 1, and between the new view
    and frame 256, read out with the same cross-room ridge and MLP. If the anchor frame's similarity alone predicts a
    target as well as the residual stream does, that readout reflects view matching with the anchor frame rather than
    a coordinate code."""
    res = json.load(open(args.out))
    models = [m for m in (args.models or res["summary"]["models"]) if m in res]
    E = {"_about": exploratory.__doc__.strip(), "columns": {}, "appearance": {}}
    for model in models:
        meta, rows, room, T = load(model)
        rooms = sorted(set(room))
        splits = [(rooms[:5], rooms[5:]), (rooms[5:], rooms[:5])]
        E["columns"][model] = {}
        for f in FRAMES:
            b = max([c for c in cells(res, model, f"{f}_r2") if c[2] == "ridge"], key=lambda c: c[3][f"{f}_r2"])
            site, L = b[0], int(b[1][1:])
            A = np.load(f"data/acts_task/{model}/{site}.npy", mmap_mode="r")
            X = torch.from_numpy(np.asarray(A[rows, meta["layers"].index(L)], np.float32))
            col = lambda k: np.mean([r2_cols(ridge_fit(X, T[k], *split_rows(room, s)[:3])(split_rows(room, s)[3]),
                                             T[k][split_rows(room, s)[3]]) for s in splits], 0).tolist()
            E["columns"][model][f] = {"cell": f"{site} L{L} ridge", "columns": ["cos", "sin"] if unit(f) == "deg" else
                                      ["forward", "left"], "r2": col(f),
                                      "shuf_r2": np.mean([col(f"{f}_shuf{s}") for s in SEEDS], 0).tolist()}
    columns_r2 = {m: {f: dict(zip(v["columns"], v["r2"])) for f, v in E["columns"][m].items()} for m in models}
    if args.columns_only:  # the appearance baseline has no model in it, and needs the walks' frames
        E["summary"] = {"note": "exploratory, not registered", "columns_r2": columns_r2}
        res["exploratory"] = E
        res.setdefault("summary", {})["exploratory"] = E["summary"]
        json.dump(res, open(args.out, "w"), indent=1)
        print(json.dumps(E["summary"], indent=1))
        return
    from appearance_cue import dino
    from transformers import AutoModel

    # appearance baseline (the trials and targets are the same in every model's recording)
    meta, rows, room, T = load(models[0])
    pick = lambda k, dt=None: np.asarray(meta[k], dt)[rows]
    walk, probe = pick("walk", int), pick("probe", int)
    net = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    S1, S256 = np.zeros(len(rows)), np.zeros(len(rows))
    for R in sorted(set(room)):
        d = f"data/walks-room{R}"
        frames, probes = np.load(f"{d}/frames.npy", mmap_mode="r"), np.load(f"{d}/probes.npy", mmap_mode="r")
        order = walk_order(dict(np.load(f"{d}/meta.npz")))
        i = np.flatnonzero(room == R)
        new = dino(net, [np.asarray(probes[p]) for p in probe[i]])
        ws = sorted(set(walk[i].tolist()))
        f1 = dict(zip(ws, dino(net, [np.asarray(frames[order[w][0]]) for w in ws])))
        f256 = dict(zip(ws, dino(net, [np.asarray(frames[order[w][-1]]) for w in ws])))
        S1[i] = [float(new[j] @ f1[w]) for j, w in enumerate(walk[i])]
        S256[i] = [float(new[j] @ f256[w]) for j, w in enumerate(walk[i])]
        print(f"appearance {R}: {len(i)} trials", flush=True)
    X = torch.tensor(np.stack([S1, S256], 1), dtype=torch.float32)
    names = list(FRAMES) + [f"{f}_shuf{s}" for f in FRAMES for s in SEEDS]
    for ro in ("ridge", "mlp"):
        E["appearance"][ro] = add_shuffle_means(mean_dicts([evaluate(X, room, T, names, s, ro) for s in splits]))
    E["appearance"]["features"] = "cosine similarity of DINOv2-base CLS: new view vs frame 1, new view vs frame 256"
    E["summary"] = {
        "note": "exploratory, not registered",
        "appearance_r2": {f: max(E["appearance"][ro][f"{f}_r2"] for ro in ("ridge", "mlp")) for f in FRAMES},
        "appearance_shuf_r2": {f: max(E["appearance"][ro][f"{f}_shuf_r2"] for ro in ("ridge", "mlp")) for f in FRAMES},
        "columns_r2": columns_r2}
    res["exploratory"] = E
    res.setdefault("summary", {})["exploratory"] = E["summary"]
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(E["summary"], indent=1))


def show(S):
    print("\n" + S["rule"])
    for model, M in S["models"].items():
        print(f"\n== {model}")
        for f, e in M.items():
            u = "deg" if f == "first_heading" else "m"
            line = (f"  {f:14s} best R2 {e['best_r2']:+.3f} ({e['best_r2_at']}); best median {e[f'best_median_{u}']:.2f} {u} "
                    f"({e['best_median_at']}); chance {e[f'chance_median_{u}']:.2f} {u}, R2 {e['chance_r2']:+.3f}")
            if "verdict" in e:
                line += (f"; shuffled R2 at best {e['shuf_r2_at_best']:+.3f} (best anywhere {e['shuf_best_r2']:+.3f}); "
                         f"verdict: {e['verdict']}")
            print(line)
    for f in FRAMES:
        if f"verdict_{f}" in S:
            print(f"verdict {f}: {S[f'verdict_{f}']} (best R2 over models {S[f'best_r2_{f}_all_models']:+.3f})")


def selftest():
    """Synthetic activations from the real targets (no recording read): a planted first-frame code must read out
    as 'present' and its shuffled control must not; a planted world code must leave first_xy 'absent'."""
    _, rows, room, T = load("qwen8b")
    g = torch.Generator().manual_seed(0)
    n, d = len(rows), 512
    rooms = sorted(set(room))
    splits = [(rooms[:5], rooms[5:]), (rooms[5:], rooms[:5])]
    names = ["world_xy", "first_xy", "first_xy_shuf0", "first_xy_shuf1", "first_xy_shuf2"]
    for plant in ("first_xy", "world_xy"):
        X = torch.randn(n, d, generator=g)
        Y = (T[plant] - T[plant].mean(0)) / T[plant].std(0)
        X[:, :8] += Y @ torch.randn(2, 8, generator=g)
        for ro in ("ridge", "mlp"):
            o = add_shuffle_means(mean_dicts([evaluate(X, room, T, names, s, ro) for s in splits]))
            print(f"planted {plant:8s} {ro:5s}: world R2 {o['world_xy_r2']:+.2f}, first_xy R2 {o['first_xy_r2']:+.2f} "
                  f"({o['first_xy_median_m']:.2f} m), shuffled R2 {o['first_xy_shuf_r2']:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="folders under data/acts_task (default: all)")
    ap.add_argument("--sites", nargs="+", default=["img_end", "img_mean", "question", "last"])
    ap.add_argument("--layers", nargs="+", type=int, help="only these layers (default all)")
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=TARGETS)
    ap.add_argument("--no-mlp", action="store_true", help="ridge only")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--exploratory", action="store_true", help="after the registered run: per-column R^2 at the best "
                    "ridge cells and the DINOv2 anchor-similarity baseline (not registered)")
    ap.add_argument("--columns-only", action="store_true", help="with --exploratory: the per-column R^2 only")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    selftest() if args.selftest else exploratory(args) if args.exploratory else run(args)


if __name__ == "__main__":
    main()
