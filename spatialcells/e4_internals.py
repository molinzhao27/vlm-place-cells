"""The four-epoch internals experiment: run 0's unit and task-readout measures repeated on the four-epoch runs.

    python -m spatialcells.e4_internals probes --models qwen8b-ft-rerun qwen8b-ft-e4-s5 ...
    python -m spatialcells.e4_internals task-extras --models qwen8b+qwen8b-proc-e4-s5 ... [--out FILE]

probes       Teleport-probe decoding on strict opposite 90-degree sectors, exactly as
             `analyze_probes.py --quadrant` (experiment 13e) scores a model: analyze.decoding_suite on every
             recorded layer of <model>-probes-walk / -probes-shuffled / -probes-only (data/acts/, from
             extract.py), ridge trained on even walks and tested on odd walks. Only the per-model loop is
             repeated here, so that the scores go to results/e4_internals/probes_quadrant.json instead of the
             shared results/probes_quadrant.json; the pixel baselines do not depend on the model and stay there.
             _best_layer is the layer with the lowest cross_quadrant, as in analyze_probes.

task-extras  Two readouts that experiment 14a ran by hand (its "exploratory" notes) and the paper quotes,
             written down here so that every run gets the same computation, run 0 included. From
             task_record.py's recordings (data/acts_task/<model>/), with task_probe's split (five held-out
             rooms -> the other five, both ways, averaged; the training rooms' last room validates lambda):
  answer     Is the distance-difference readout (d_a - d_b) more than the model's answer? R^2 of predicting
             d_a - d_b from the answer alone (the training rooms' mean for each answer), from the true nearer
             option alone, and of a ridge readout (task_probe.ridge_fit) of the residual left after the
             answer, per layer at the last and question tokens. Trials whose recorded answer is neither
             option are left out (counted). The answer is experiment 10's on the same prompt (meta.json).
  within     The new view's (x, y) inside each room: five folds of walks (walk index mod 5), ridge per fold
             (lambda validated on the next fold), median error over the room's trials, mean over rooms;
             chance predicts the training folds' mean. Per layer at the visual-token mean and the last token.
Writes --out (default results/e4_internals/task_extras.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

OUT_DIR = "results/e4_internals"
CODE = ("spatialcells/analyze.py", "spatialcells/extract.py", "spatialcells/task_record.py",
        "spatialcells/task_probe.py", "spatialcells/e4_internals.py")


def code_md5():
    return {f: hashlib.md5(open(f, "rb").read()).hexdigest() for f in CODE if os.path.exists(f)}


# ---------------------------------------------------------------- probes (strict sectors)
def score_probes(models, data, out):
    from .analyze import decoding_suite, dev
    pmeta = dict(np.load(f"{data}/probes_meta.npz"))
    res = json.load(open(out)) if os.path.exists(out) else {}
    res["_about"] = ("analyze_probes --quadrant's per-model scores (analyze.decoding_suite with quadrant=True) for the "
                     "runs recorded by the four-epoch internals experiment; baselines (pixels, recall) are in "
                     "results/probes_quadrant.json. _best_layer: the layer with the lowest cross_quadrant.")
    t0 = time.time()
    for model in models:
        for cond in ("probes-walk", "probes-shuffled", "probes-only"):
            tag = f"{model}-{cond}"
            if not os.path.isdir(f"data/acts/{tag}"):
                print(f"{tag}: no recording, skipped", flush=True)
                continue
            layers = sorted((f[:-4] for f in os.listdir(f"data/acts/{tag}") if f.endswith(".npy")), key=lambda s: int(s[1:]))
            res[tag] = {}
            for layer in layers:
                X = torch.from_numpy(np.load(f"data/acts/{tag}/{layer}.npy").astype(np.float32)).to(dev)
                res[tag][layer] = decoding_suite(X, pmeta, True)
                del X
            print(f"{tag}: " + "  ".join(f"{l}:{v['within_quadrant']:.2f}/{v['cross_quadrant']:.2f}" for l, v in res[tag].items())
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
            json.dump(res, open(out, "w"), indent=1)
    res["_best_layer"] = {t: min(v, key=lambda l: v[l]["cross_quadrant"]) for t, v in res.items() if not t.startswith("_")}
    res["_code_md5"] = code_md5()
    json.dump(res, open(out, "w"), indent=1)


# ---------------------------------------------------------------- task-extras
def answer_readouts(X, M, splits, ridge_fit, r2):
    """Per split: R^2 of d_a - d_b from the answer alone, from the true sign alone, and of a ridge readout of
    the residual after the answer. M has room, dd (n x 1), an (true a-nearer, n x 1), ans_a (n x 1), ok (bool)."""
    out = []
    room = M["room"]
    for tr_rooms, te_rooms in splits:
        idx = lambda m: torch.from_numpy(np.flatnonzero(m & M["ok"]))
        in_tr, in_te = np.isin(room, tr_rooms), np.isin(room, te_rooms)
        is_val = room == tr_rooms[-1]
        fit, val, train, test = idx(in_tr & ~is_val), idx(is_val), idx(in_tr), idx(in_te)
        y = M["dd"]

        def by_label(lab):
            m1 = y[train][lab[train, 0] > 0.5].mean()
            m0 = y[train][lab[train, 0] <= 0.5].mean()
            return torch.where(lab[:, :1] > 0.5, m1, m0)
        pred_ans = by_label(M["ans_a"])
        o = {"answer_only_r2": r2(pred_ans[test], y[test]), "true_sign_r2": r2(by_label(M["an"])[test], y[test])}
        if X is not None:
            resid = y - pred_ans
            o["residual_ridge_r2"] = r2(ridge_fit(X, resid, fit, val, train)(test), resid[test])
        out.append(o)
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def within_room(X, M, ridge_fit, n_folds=5):
    """Median error (m) of the new view's (x, y) read out inside each room, folds by walk; mean over rooms."""
    errs, chance = [], []
    for R in sorted(set(M["room"])):
        rows = np.flatnonzero(M["room"] == R)
        fold = M["walk"][rows] % n_folds
        e, c = [], []
        for k in range(n_folds):
            te, va = rows[fold == k], rows[fold == (k + 1) % n_folds]
            tr = rows[fold != k]
            fi = rows[(fold != k) & (fold != (k + 1) % n_folds)]
            if not len(te) or not len(va) or not len(fi):
                continue
            t = lambda a: torch.from_numpy(a)
            xy = M["xy"]
            if X is not None:
                p = ridge_fit(X, xy, t(fi), t(va), t(tr))(t(te))
                e.append((p - xy[t(te)]).norm(dim=1))
            c.append((xy[t(te)] - xy[t(tr)].mean(0)).norm(dim=1))
        if e:
            errs.append(torch.cat(e).median().item())
        chance.append(torch.cat(c).median().item())
    return (float(np.mean(errs)) if errs else None), float(np.mean(chance))


def task_extras(models, sites_answer, sites_within, out):
    from .task_probe import load_meta, r2, ridge_fit
    res = json.load(open(out)) if os.path.exists(out) else {}
    res["_about"] = __doc__.split("task-extras  ", 1)[1].split("Writes --out")[0].strip()
    t0 = time.time()
    for model in models:
        meta, rows, M, beh = load_meta(model)
        rooms = sorted(set(M["room"]))
        h = len(rooms) // 2
        splits = [(rooms[:h], rooms[h:]), (rooms[h:], rooms[:h])]
        a = np.asarray(meta["a"])[rows]
        b = np.asarray(meta["b"])[rows]
        ans = np.array([-1 if v is None else v for v in np.asarray(meta["answer"], object)[rows]])
        M["ok"] = (ans == a) | (ans == b)
        M["ans_a"] = torch.tensor((ans == a).astype(np.float32)[:, None])
        M["walk"] = np.asarray(meta["walk"])[rows]
        r = {"n_trials": int(len(rows)), "n_answer_ok": int(M["ok"].sum()), "device": meta.get("device", "mps (not recorded)"),
             "exp10_devices": sorted({d for d in np.asarray(meta.get("exp10_device", [None] * len(meta["id"])), object)[rows] if d}),
             "behaviour": beh}
        base = answer_readouts(None, M, splits, ridge_fit, r2)
        r["answer_only_r2"], r["true_sign_r2"] = base["answer_only_r2"], base["true_sign_r2"]
        r["within_room_chance_m"] = within_room(None, M, ridge_fit)[1]
        for site in sorted(set(sites_answer) | set(sites_within)):
            A = np.load(f"data/acts_task/{model}/{site}.npy", mmap_mode="r")
            per = {}
            for j, layer in enumerate(meta["layers"]):
                X = torch.from_numpy(np.asarray(A[rows, j], np.float32))
                d = {}
                if site in sites_answer:
                    d["residual_ridge_r2"] = answer_readouts(X, M, splits, ridge_fit, r2)["residual_ridge_r2"]
                if site in sites_within:
                    d["within_room_m"] = within_room(X, M, ridge_fit)[0]
                per[f"L{layer}"] = d
            r[site] = per
            print(f"{model} {site}: {len(meta['layers'])} layers, {time.time() - t0:.0f}s", flush=True)
        res[model] = r
        json.dump(res, open(out, "w"), indent=1)
        s = r.get("last", {})
        mid = [v["residual_ridge_r2"] for L, v in s.items() if 20 <= int(L[1:]) <= 30 and "residual_ridge_r2" in v]
        print(f"== {model}: answer-only R2 {r['answer_only_r2']:.2f}, true sign {r['true_sign_r2']:.2f}, residual after the "
              f"answer (last token, layers 20-30) {min(mid):.2f}-{max(mid):.2f}" if mid else f"== {model}", flush=True)
    res["_code_md5"] = code_md5()
    json.dump(res, open(out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probes")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--data", default="data/walks")
    p.add_argument("--out", default=f"{OUT_DIR}/probes_quadrant.json")
    t = sub.add_parser("task-extras")
    t.add_argument("--models", nargs="+", required=True, help="folders under data/acts_task")
    t.add_argument("--answer-sites", nargs="+", default=["question", "last"])
    t.add_argument("--within-sites", nargs="+", default=["img_mean", "last"])
    t.add_argument("--out", default=f"{OUT_DIR}/task_extras.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.cmd == "probes":
        score_probes(args.models, args.data, args.out)
    else:
        task_extras(args.models, args.answer_sites, args.within_sites, args.out)


if __name__ == "__main__":
    main()
