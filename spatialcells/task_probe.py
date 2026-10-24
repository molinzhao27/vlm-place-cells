"""Experiment 14a readouts: what the residual stream holds while the model does the task (CPU only).

    python -m spatialcells.task_probe [--models qwen8b qwen8b+qwen8b-proc] [--no-mlp]

Reads task_record.py's recordings (data/acts_task/<model>/). For every model, site and layer, each
readout is trained on five held-out rooms and tested on the other five, both ways (P5000-P5004 ->
P5005-P5009 and back), and the two are averaged; no room is in both. Inside the five training rooms
the last one is the validation room (the ridge's lambda; the MLP's early stopping). Targets:
  xy        the new view's position: R^2 (mean of x and y) and median error (m)
  d_diff    its distance to option a minus its distance to option b: R^2
  a_nearer  whether option a is the nearer one: accuracy (all test trials, and novel ones)
  sectors   xy trained on one 90-degree heading sector of the new view (centred on E, N, W, S) in the
            training rooms and tested on the opposite sector of the test rooms (strict, as experiment
            13e; experiment 13f's caveat applies), with the same sector as the within-sector reference;
            median error (m), mean over the 4 sectors
Readouts: ridge (analyze._ridge on inputs z-scored over the training rooms; lambda 1e1-1e5 by validation
MSE, then refit on all five) and a one-hidden-layer MLP (256 ReLU units, Adam, minibatches of 64, early
stopping on the validation room's loss, patience 10 epochs; one per target). Chance: R^2 0 (predicting
the test mean), accuracy of the training rooms' majority answer, median error of predicting the training
set's (sector's) mean position. R^2 is 1 - SS_res / SS_tot about the test rooms' mean.
Writes results/task_probe.json (per model, site, layer and readout) and prints a summary, with the
model's behavioural accuracy on the same trials (experiment 10) and, descriptively for question (3),
the earlier recording of the same models at the image-end token (extract.py walk mode, room A,
MLP neurons at 7 layers; results/probes_quadrant.json).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .analyze import _ridge, sectors

LAMBDAS = (1e1, 1e2, 1e3, 1e4, 1e5)
OUT = "results/task_probe.json"


def r2(pred, y):
    """1 - SS_res / SS_tot about the test mean, averaged over columns."""
    return float((1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-12)).mean())


def ridge_fit(X, Y, fit, val, train):
    """z-score on the training rows, pick lambda by validation MSE (per target column set), refit on all
    training rows; returns a predict function."""
    mu, sd = X[train].mean(0), X[train].std(0).clamp_min(1e-6)
    Z = (X - mu) / sd
    ym = Y[fit].mean(0)
    errs = [((Z[val] @ _ridge(Z[fit], Y[fit] - ym, lam) + ym - Y[val]) ** 2).mean().item() for lam in LAMBDAS]
    lam = LAMBDAS[int(np.argmin(errs))]
    ym = Y[train].mean(0)
    W = _ridge(Z[train], Y[train] - ym, lam)
    return lambda rows: Z[rows] @ W + ym


def mlp_fit(X, Y, fit, val, train, binary=False, epochs=200, patience=10, seed=0):
    """One hidden layer of 256 ReLU units, trained on the fit rows with early stopping on the validation rows."""
    torch.manual_seed(seed)
    mu, sd = X[train].mean(0), X[train].std(0).clamp_min(1e-6)
    Z = (X - mu) / sd
    ym, ys = (torch.zeros(Y.shape[1]), torch.ones(Y.shape[1])) if binary else (Y[fit].mean(0), Y[fit].std(0).clamp_min(1e-6))
    T = (Y - ym) / ys
    net = nn.Sequential(nn.Linear(Z.shape[1], 256), nn.ReLU(), nn.Linear(256, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss() if binary else nn.MSELoss()
    best, best_state, wait = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for b in fit[torch.randperm(len(fit), generator=g)].split(64):
            opt.zero_grad()
            loss_fn(net(Z[b]), T[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(Z[val]), T[val]).item()
        if v < best - 1e-4:
            best, best_state, wait = v, {k: p.clone() for k, p in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(rows):
        with torch.no_grad():
            out = net(Z[rows])
        return out if binary else out * ys + ym
    return predict


def evaluate(X, M, split, readout):
    """All targets for one feature matrix X (trials x hidden), one split (train rooms, test rooms)."""
    tr_rooms, te_rooms = split
    room = M["room"]
    idx = lambda m: torch.from_numpy(np.flatnonzero(m))
    in_tr, in_te = np.isin(room, tr_rooms), np.isin(room, te_rooms)
    is_val = room == tr_rooms[-1]
    fit, val, train, test = idx(in_tr & ~is_val), idx(is_val), idx(in_tr), idx(in_te)
    xy, dd, an = M["xy"], M["dd"], M["an"]
    novel = idx(in_te & ~M["familiar"])
    fitter = ridge_fit if readout == "ridge" else mlp_fit
    o = {}
    p = fitter(X, xy, fit, val, train)(test)
    o["xy_r2"], o["xy_median_m"] = r2(p, xy[test]), (p - xy[test]).norm(dim=1).median().item()
    o["d_diff_r2"] = r2(fitter(X, dd, fit, val, train)(test), dd[test])
    if readout == "ridge":
        f = fitter(X, an * 2 - 1, fit, val, train)
        acc = lambda rows: ((f(rows)[:, 0] > 0).float() == an[rows, 0]).float().mean().item()
    else:
        f = fitter(X, an, fit, val, train, binary=True)
        acc = lambda rows: ((f(rows)[:, 0] > 0).float() == an[rows, 0]).float().mean().item()
    o["a_nearer_acc"], o["a_nearer_acc_novel"] = acc(test), acc(novel)
    sec = M["sector"]
    within, cross = [], []
    for i in range(4):
        s, opp = sec == i, sec == (i + 2) % 4
        f = fitter(X, xy, idx(in_tr & ~is_val & s), idx(is_val & s), idx(in_tr & s))
        within.append((f(idx(in_te & s)) - xy[idx(in_te & s)]).norm(dim=1).median().item())
        cross.append((f(idx(in_te & opp)) - xy[idx(in_te & opp)]).norm(dim=1).median().item())
    o["within_sector_m"], o["cross_sector_m"] = float(np.mean(within)), float(np.mean(cross))
    return o


def chance(M, split):
    tr_rooms, te_rooms = split
    room, xy, an, sec = M["room"], M["xy"], M["an"][:, 0], M["sector"]
    tr, te = np.isin(room, tr_rooms), np.isin(room, te_rooms)
    med = lambda a, b: (xy[torch.from_numpy(b)] - xy[torch.from_numpy(a)].mean(0)).norm(dim=1).median().item()
    maj = float(an[torch.from_numpy(tr)].mean() >= 0.5)
    return {"xy_median_m": med(tr, te),
            "a_nearer_acc": (an[torch.from_numpy(te)] == maj).float().mean().item(),
            "within_sector_m": float(np.mean([med(tr & (sec == i), te & (sec == i)) for i in range(4)])),
            "cross_sector_m": float(np.mean([med(tr & (sec == i), te & (sec == (i + 2) % 4)) for i in range(4)]))}


def mean_dicts(ds):
    return {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}


def load_meta(model):
    d = f"data/acts_task/{model}"
    meta = json.load(open(f"{d}/meta.json"))
    done = sorted({json.loads(l)["row"] for l in open(f"{d}/done.jsonl")}) if os.path.exists(f"{d}/done.jsonl") else []
    rows = np.array(done, int)
    pick = lambda k, dt=None: np.asarray(meta[k], dt)[rows]
    M = {"room": pick("room"), "familiar": pick("familiar", bool), "sector": sectors(pick("theta", float)),
         "xy": torch.tensor(np.stack([pick("x", float), pick("y", float)], 1), dtype=torch.float32),
         "dd": torch.tensor(pick("d_a_minus_d_b", float)[:, None], dtype=torch.float32),
         "an": torch.tensor(pick("a_nearer", float)[:, None], dtype=torch.float32)}
    correct = [c for c in pick("correct", object) if c is not None]
    nov = [c for c, f in zip(pick("correct", object), M["familiar"]) if c is not None and not f]
    beh = {"n": len(correct), "accuracy": float(np.mean(correct)) if correct else None,
           "accuracy_novel": float(np.mean(nov)) if nov else None}
    return meta, rows, M, beh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="folders under data/acts_task (default: all)")
    ap.add_argument("--sites", nargs="+", default=["img_end", "img_mean", "question", "last"])
    ap.add_argument("--no-mlp", action="store_true", help="ridge only")
    args = ap.parse_args()
    models = args.models or sorted(m for m in os.listdir("data/acts_task") if os.path.exists(f"data/acts_task/{m}/meta.json"))
    readouts = ["ridge"] if args.no_mlp else ["ridge", "mlp"]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res["_about"] = __doc__.split("\n\n", 2)[2].strip()
    t0 = time.time()
    for model in models:
        meta, rows, M, beh = load_meta(model)
        rooms = sorted(set(M["room"]))
        if len(rooms) < 2 or len(rows) < len(meta["id"]):
            print(f"{model}: {len(rows)}/{len(meta['id'])} trials recorded in {len(rooms)} rooms"
                  + ("; skipped" if len(rooms) < 2 else "; using those"), flush=True)
            if len(rooms) < 2:
                continue
        h = len(rooms) // 2
        splits = [(rooms[:h], rooms[h:]), (rooms[h:], rooms[:h])]
        res.setdefault("_split", {})[model] = {"rooms": [rooms[:h], rooms[h:]], "validation_room": [s[0][-1] for s in splits],
                                               "n_trials": len(rows)}
        res.setdefault("_behaviour", {})[model] = beh
        res.setdefault("_chance", {})[model] = mean_dicts([chance(M, s) for s in splits])
        res[model] = {}
        for site in args.sites:
            A = np.load(f"data/acts_task/{model}/{site}.npy", mmap_mode="r")
            res[model][site] = {}
            for j, layer in enumerate(meta["layers"]):
                X = torch.from_numpy(np.asarray(A[rows, j], np.float32))
                res[model][site][f"L{layer}"] = {ro: mean_dicts([evaluate(X, M, s, ro) for s in splits]) for ro in readouts}
            json.dump(res, open(OUT, "w"), indent=1)
            print(f"{model} {site}: {len(meta['layers'])} layers, {time.time() - t0:.0f}s", flush=True)
    res["_reference_earlier_recording"] = earlier_recording()
    json.dump(res, open(OUT, "w"), indent=1)
    summary(res, models, args.sites, readouts)


def earlier_recording():
    """The earlier recording of the same models at the image-end token: teleport probes after the walk
    (extract.py, walk mode; room A; MLP neurons at 7 layers; even/odd walks), per layer."""
    if not os.path.exists("results/probes_quadrant.json"):
        return {}
    q = json.load(open("results/probes_quadrant.json"))
    keys = ("all", "within_quadrant", "cross_quadrant", "chance", "cross_quadrant_chance")
    return {tag: {l: {k: v[k] for k in keys} for l, v in q[tag].items()}
            for tag in ("qwen8b-probes-walk", "qwen8b-ft-probes-walk") if tag in q}


def summary(res, models, sites, readouts):
    best = lambda by_layer, ro, k, low=False: (min if low else max)(by_layer, key=lambda l: by_layer[l][ro][k])
    for model in models:
        if model not in res:
            continue
        b, c = res["_behaviour"][model], res["_chance"][model]
        fmt = lambda v: "n/a" if v is None else f"{v:.1%}"
        print(f"\n== {model}: {res['_split'][model]['n_trials']} trials; behaviour {fmt(b['accuracy'])} "
              f"(novel {fmt(b['accuracy_novel'])}); chance: a_nearer {c['a_nearer_acc']:.1%}, xy {c['xy_median_m']:.2f} m, "
              f"sector within / cross {c['within_sector_m']:.2f} / {c['cross_sector_m']:.2f} m")
        print("  best layer per site and readout: a_nearer acc | d_diff R2 | xy R2, median m | sector within / cross m")
        for site in sites:
            for ro in readouts:
                L = res[model][site]
                la, ld, lx, ls = best(L, ro, "a_nearer_acc"), best(L, ro, "d_diff_r2"), best(L, ro, "xy_r2"), \
                    best(L, ro, "cross_sector_m", low=True)
                print(f"  {site:9s} {ro:5s}  {L[la][ro]['a_nearer_acc']:.1%} ({la})  {L[ld][ro]['d_diff_r2']:+.2f} ({ld})  "
                      f"{L[lx][ro]['xy_r2']:+.2f}, {L[lx][ro]['xy_median_m']:.2f} m ({lx})  "
                      f"{L[ls][ro]['within_sector_m']:.2f} / {L[ls][ro]['cross_sector_m']:.2f} m ({ls})")
        for site in ("question", "last"):
            for ro in readouts:
                L = res[model].get(site)
                if not L:
                    continue
                print(f"  {site} {ro} by layer, a_nearer acc: " + " ".join(f"{v[ro]['a_nearer_acc'] * 100:.0f}" for v in L.values()))
                print(f"  {site} {ro} by layer, d_diff R2:    " + " ".join(f"{v[ro]['d_diff_r2']:.2f}" for v in L.values()))
    ref = res.get("_reference_earlier_recording", {})
    for tag, by_layer in ref.items():
        print(f"earlier recording {tag} (room A, MLP neurons; all / within / cross sector m, chance "
              f"{next(iter(by_layer.values()))['chance']:.2f}): "
              + " ".join(f"{l}:{v['all']:.2f}/{v['within_quadrant']:.2f}/{v['cross_quadrant']:.2f}" for l, v in by_layer.items()))


if __name__ == "__main__":
    main()
