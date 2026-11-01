"""Experiment 32 (docs/experiments.md): which parts of VGGT's ten pose directions does the camera depend on? Each pair
of experiment 27's ordinary PCA directions is written alone and left out of the ten, on the same 128 target-donor pairs
and with the same patching path. Regression-derived directions repeat the test without the mixing between PCA directions.

    # the RTX 4070 through gq (GPU): the clean passes, every patch, then the analysis
    .venv/bin/python scripts/vggt_pair_patch.py run --vggt $VGGT_DIR --batch 2 [--smoke]
    # anywhere (CPU): intervals, the registered rule, the summary
    .venv-mac/bin/python scripts/vggt_pair_patch.py analyze [--smoke]

run      VGGT single mode (8 anchors + 1 walk frame, room A). Takes experiment 27's targets, donors, fit frames, Qord30 and
         residuals from data/subspace_controls/. Reruns the clean targets and donors, checks their residuals against the
         stored ones, and writes each subspace's part of the donor's residual into the target's run after global block g.
         Keeps the walk frame's (x, y), heading and camera-from-world translation (meters, in the camera's axes) for
         every pair -> data/pair_patch/raw.{npz,json} (--smoke: data/pair_patch/smoke/, the first 8 pairs)
analyze  experiment 27's per-pair measures, translation transfer, percentile bootstrap intervals over pairs (10,000
         resamples, seed 0, paired across conditions), the registered rule -> results/vggt_pair_patch.json ("summary")
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

from vggt_subspace_controls import STATS, angdiff, ci, pair_measures, stat  # noqa: E402

OUT = "data/pair_patch"
E27 = "data/subspace_controls"
BLOCKS = [21, 19]                  # block 21 is primary
# zero-based directions of Qord30, assigned by scripts/vggt_token_pca.py before any patch (which family explains each better)
PAIRS = {21: {"heading": [0, 1], "camera": [2, 3], "room": [4, 5], "second": [6, 7, 8, 9]},
         19: {"heading": [0, 1], "camera": [2, 5], "room": [3, 4], "second": [6, 7, 8, 9]}}
FEATS = {"heading": [0, 1], "room": [2, 3], "camera": [4, 5]}
FAMILIES = ("heading", "camera", "room", "second")
N_BOOT = 10000
LOSS_NEEDED, LOSS_SPARE, LOSS_SPARE_CI = 0.25, 0.10, 0.25     # share of the ten's transfer lost when a family is left out
ALONE_MOVES, ALONE_STILL, ALONE_STILL_CI = 0.25, 0.12, 0.25   # share of the ten's transfer reached by a family alone


def features(x, y, t):
    """vggt_token_pca.py's six: heading, room frame, camera frame, with (x, y) from the room center."""
    x, y = x - 5.0, y - 5.0
    c, s = np.cos(t), np.sin(t)
    return np.c_[c, s, x, y, x * c + y * s, x * s - y * c]


def label_free(Q, pairs):
    """name -> orthonormal directions: each family alone, each left out of the ten, and the ten."""
    ten = list(range(10))
    conds = {"ten": Q[:, ten]}
    for k in FAMILIES:
        conds[f"alone_{k}"] = Q[:, pairs[k]]
        conds[f"ten_minus_{k}"] = Q[:, [i for i in ten if i not in pairs[k]]]
    return conds


def fit_directions(Hf, F):
    """Regression of the fit frames' residuals on the six features: A (C, 6) and its pseudo-inverse."""
    X, Y = F - F.mean(0), Hf - Hf.mean(0)
    A = np.linalg.lstsq(X, Y, rcond=None)[0].T
    return A, np.linalg.pinv(A)


def oblique(A, Ap, keep, d):
    """The difference d written along the kept features' directions only, leaving the others' codes as they were."""
    return (d @ Ap.T)[:, keep] @ A[:, keep].T


# ------------------------------------------------------------------------------------------------ run
@torch.no_grad()
def run(args):
    import torch.nn.functional as F
    sys.path.insert(0, args.vggt)
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from spatialcells.vggt_cells import anchors
    from spatialcells.vggt_control import RES, umeyama
    from spatialcells.vggt_patch import to_room

    out = f"{OUT}/smoke" if args.smoke else OUT
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)

    R27, Z27 = np.load(f"{E27}/raw.npz"), np.load(f"{E27}/resid.npz")
    n_test = 8 if args.smoke else 128
    targets, donors, fit = R27["targets"][:n_test], R27["donors"][:n_test], R27["fit"]
    frames = np.load("data/walks/frames.npy", mmap_mode="r")
    meta = dict(np.load("data/walks/meta.npz"))

    A = anchors("data/walks", "A")
    model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
    agg = model.aggregator
    P = agg.patch_start_idx + (RES // 14) ** 2
    n_anchor = len(A["images"])
    cam = n_anchor * P
    rec, patch = {}, {}

    def hook(g):
        def f(_, __, o):
            rec[g] = o[:, cam].float().cpu()
            if g in patch:
                o = o.clone()
                o[:, cam] = patch[g].to(o.device, o.dtype)
                return o
        return f

    for g in BLOCKS:
        agg.global_blocks[g].register_forward_hook(hook(g))

    def prep(ims):
        x = torch.from_numpy(np.stack([np.asarray(a) for a in ims])).cuda().permute(0, 3, 1, 2).float() / 255
        return F.interpolate(x, size=(RES, RES), mode="bicubic", align_corners=False).clamp(0, 1)

    anchor_px = prep(A["images"])
    gt_anchor = np.c_[A["poses"][:, :2], np.ones(n_anchor)]

    def predict(idx):
        """The walk frame's (x, y), heading and translation (camera axes, meters), each run aligned by its own anchor fit."""
        seqs = torch.stack([torch.cat([anchor_px, prep(frames[[i]])]) for i in idx])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, _ = agg(seqs)
            pose = model.camera_head(tokens)[-1]
        ext, _ = pose_encoding_to_extri_intri(pose.float(), (RES, RES))
        ext = ext.cpu().numpy().astype(np.float64)
        R, t = ext[..., :3], ext[..., 3]
        cams = np.stack([-np.einsum("bsji,bsj->bsi", R, t), np.einsum("bsji,j->bsi", R, np.array([0.0, 0.0, 1.0]))], 1)
        fits = [umeyama(c[0][:n_anchor], gt_anchor) for c in cams]
        xy, th = zip(*(to_room(c, T) for c, T in zip(cams, fits)))
        tr = np.stack([T[0] * t[b, n_anchor] for b, T in enumerate(fits)])
        return np.stack(xy)[:, n_anchor], np.stack(th)[:, n_anchor], tr

    def sweep(idx, repl=None, g=None):
        out_, got = [], {}
        for b0 in range(0, len(idx), args.batch):
            patch.clear()
            rec.clear()
            if repl is not None:
                patch[g] = repl[b0:b0 + args.batch]
            out_.append(predict(idx[b0:b0 + args.batch]))
            for k, v in rec.items():
                got.setdefault(k, []).append(v)
        patch.clear()
        return (*(np.concatenate(x) for x in zip(*out_)), {k: torch.cat(v) for k, v in got.items()})

    pred_t, th_t, tr_t, H_t = sweep(targets)
    pred_d, th_d, tr_d, H_d = sweep(donors)
    rep = {"experiment": 32, "smoke": bool(args.smoke), "n_test": int(n_test), "batch": args.batch, "blocks": {}, "residual_check": {}}
    for g in BLOCKS:
        same = [float(np.abs(H_t[g].numpy() - Z27[f"t_{g}"][:n_test]).max()), float(np.abs(H_d[g].numpy() - Z27[f"d_{g}"][:n_test]).max())]
        rep["residual_check"][str(g)] = {"max_abs_diff_targets": same[0], "max_abs_diff_donors": same[1]}
    log(f"clean passes done; residuals vs experiment 27 {rep['residual_check']}")
    raw = {"pred_t": pred_t, "th_t": th_t, "tr_t": tr_t, "pred_d": pred_d, "th_d": th_d, "tr_d": tr_d, "targets": targets, "donors": donors,
           "xy_true_t": np.c_[meta["x"][targets], meta["y"][targets]]}

    def save():
        np.savez(f"{out}/raw.npz", **raw)
        json.dump(rep, open(f"{out}/raw.json", "w"), indent=1)

    Ffit = features(meta["x"][fit], meta["y"][fit], meta["theta"][fit])
    for g in BLOCKS:
        Hf = Z27[f"fit_{g}"].astype(np.float64)
        Hc = Hf - Hf.mean(0)
        ht, hd = H_t[g].numpy().astype(np.float64), H_d[g].numpy().astype(np.float64)
        d = hd - ht
        Q = R27[f"b{g}/Qord30"].astype(np.float64)
        Af, Ap = fit_directions(Hf, Ffit)
        conds = {k: (d @ q) @ q.T for k, q in label_free(Q, PAIRS[g]).items()}
        for k, keep in FEATS.items():
            conds[f"fit_{k}"] = oblique(Af, Ap, keep, d)
            conds[f"fit_all_minus_{k}"] = oblique(Af, Ap, [i for f, v in FEATS.items() if f != k for i in v], d)
        conds["fit_all"] = oblique(Af, Ap, list(range(6)), d)
        rep["blocks"][str(g)] = {"pairs": PAIRS[g], "conditions": {}, "feature_r2_fit": float(1 - ((Hc - (Hc @ Ap.T) @ Af.T) ** 2).sum() / (Hc ** 2).sum())}
        for name, w in conds.items():
            xy, th, tr, _ = sweep(targets, torch.from_numpy((ht + w).astype(np.float32)), g)
            raw[f"b{g}/{name}/xy"], raw[f"b{g}/{name}/th"], raw[f"b{g}/{name}/tr"] = xy, th, tr
            raw[f"b{g}/{name}/share"] = (w ** 2).sum(1) / (d ** 2).sum(1)
            gap = pred_d - pred_t
            tf = (((xy - pred_t) * gap).sum(1) / (gap ** 2).sum(1)).mean()
            rep["blocks"][str(g)]["conditions"][name] = {"share_median": float(np.median(raw[f"b{g}/{name}/share"])), "transfer_mean": float(tf)}
            log(f"block {g} {name}: share {np.median(raw[f'b{g}/{name}/share']):.3f}, transfer {tf:.3f}")
            save()
    rep["elapsed_s"] = time.time() - t0
    save()
    log(f"done; wrote {out}/raw.npz, raw.json")
    run_analyze(argparse.Namespace(smoke=args.smoke, out=args.out))


# ------------------------------------------------------------------------------------------------ analyze
def verdict(L, Lci, A, Aci):
    """The registered rule for one family: L, share of the ten's transfer lost without it; A, share reached by it alone."""
    if L >= LOSS_NEEDED and Lci[0] > 0:
        return "needed"
    if L < LOSS_SPARE and Lci[1] < LOSS_SPARE_CI:
        if A >= ALONE_MOVES and Aci[0] > 0:
            return "read but redundant"
        if A < ALONE_STILL and Aci[1] < ALONE_STILL_CI:
            return "not read"
    return "unclear"


def run_analyze(args):
    d = f"{OUT}/smoke" if args.smoke else OUT
    Z, RJ = np.load(f"{d}/raw.npz"), json.load(open(f"{d}/raw.json"))
    R27 = np.load(f"{E27}/raw.npz")
    R = {k: Z[k] for k in ("pred_t", "pred_d", "th_t", "th_d", "xy_true_t")}
    n = len(R["pred_t"])
    idx = np.random.default_rng(0).integers(0, n, (N_BOOT, n))
    res = {"about": "Experiment 32 (docs/experiments.md). VGGT single mode, room A, the walk frame's camera token after global block g; "
                    "experiment 27's 128 target-donor pairs. Transfer is the fraction of the target-donor line the prediction moves "
                    "(mean); translation transfer is the same in the camera-from-world translation. Intervals: 95% percentile "
                    "bootstrap over pairs (10,000 resamples, seed 0), paired across conditions.",
           "residual_check": RJ["residual_check"], "blocks": {}}
    gap_tr = Z["tr_d"] - Z["tr_t"]

    def tr_transfer(x):
        return ((x - Z["tr_t"]) * gap_tr).sum(1) / (gap_tr ** 2).sum(1)

    summary = {}
    for g in BLOCKS:
        B = RJ["blocks"][str(g)]
        names = list(B["conditions"])
        M = {nm: pair_measures(Z[f"b{g}/{nm}/xy"], Z[f"b{g}/{nm}/th"], R) for nm in names}
        for nm in names:
            M[nm]["translation_transfer"] = tr_transfer(Z[f"b{g}/{nm}/tr"])
        how = {**STATS, "translation_transfer": "mean"}
        conds = {}
        for nm in names:
            c = {"share_median": B["conditions"][nm]["share_median"]}
            for meas, h in how.items():
                x = M[nm][meas]
                c[meas], c[f"{meas}_ci95"] = float(stat(x, h)), ci(stat(x[idx], h, axis=1))
            conds[nm] = c
        ten27 = pair_measures(R27[f"b{g}/ordinary_pca10/xy"][:n], R27[f"b{g}/ordinary_pca10/th"][:n], R)["transfer"]
        rep = {"transfer_exp27": float(ten27.mean()), "transfer_here": conds["ten"]["transfer"],
               "max_abs_diff": float(np.abs(M["ten"]["transfer"] - ten27).max())}
        rep["ok"] = bool(abs(rep["transfer_exp27"] - rep["transfer_here"]) <= 0.01)
        if not args.smoke and not rep["ok"]:
            sys.exit(f"block {g}: the ten directions do not reproduce experiment 27: {rep}")

        def rule(ref, alone, left):
            T, Ta, Tl = M[ref]["transfer"], M[alone]["transfer"], M[left]["transfer"]
            r = lambda a, b: a.mean(-1) / b.mean(-1)
            L, A = 1 - r(Tl, T), r(Ta, T)
            Lci, Aci = ci(1 - r(Tl[idx], T[idx])), ci(r(Ta[idx], T[idx]))
            return {"loss": float(L), "loss_ci95": Lci, "alone": float(A), "alone_ci95": Aci, "verdict": verdict(L, Lci, A, Aci)}

        families = {k: rule("ten", f"alone_{k}", f"ten_minus_{k}") for k in FAMILIES}
        fitted = {k: rule("fit_all", f"fit_{k}", f"fit_all_minus_{k}") for k in FEATS}
        res["blocks"][str(g)] = {"pairs": B["pairs"], "feature_r2_fit": B["feature_r2_fit"], "conditions": conds, "reproduces_exp27": rep,
                                 "families": families, "feature_fit": fitted}
        summary[f"block{g}"] = {"reproduces_exp27": rep, "families": families, "feature_fit": fitted,
                                "transfer": {nm: conds[nm]["transfer"] for nm in names},
                                "transfer_ci95": {nm: conds[nm]["transfer_ci95"] for nm in names},
                                "heading_gap_closed": {nm: conds[nm]["heading_gap_closed"] for nm in names},
                                "translation_transfer": {nm: conds[nm]["translation_transfer"] for nm in names}}
    # does the head's translation have the form the camera-frame features assume? (the clean targets and donors)
    th = np.r_[Z["th_t"], Z["th_d"]]
    xy = np.r_[Z["pred_t"], Z["pred_d"]] - 5.0
    tr = np.r_[Z["tr_t"], Z["tr_d"]]
    fwd, right = xy[:, 0] * np.cos(th) + xy[:, 1] * np.sin(th), xy[:, 0] * np.sin(th) - xy[:, 1] * np.cos(th)
    summary["translation_form"] = {"corr_forward_with_tz": float(np.corrcoef(fwd, tr[:, 2])[0, 1]), "corr_right_with_tx": float(np.corrcoef(right, tr[:, 0])[0, 1])}
    res["summary"] = {"experiment": 32, **summary}
    path = args.out if not args.smoke else f"{d}/vggt_pair_patch_smoke.json"
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps({"residual_check": RJ["residual_check"], "translation_form": summary["translation_form"]}), flush=True)
    for g in BLOCKS:
        b = summary[f"block{g}"]
        print(f"block {g}: reproduces {b['reproduces_exp27']}", flush=True)
        for nm in b["transfer"]:
            print(f"  {nm:24s} transfer {b['transfer'][nm]:+.3f} [{b['transfer_ci95'][nm][0]:+.3f}, {b['transfer_ci95'][nm][1]:+.3f}]  "
                  f"heading closed {b['heading_gap_closed'][nm]:+.2f}  translation {b['translation_transfer'][nm]:+.3f}", flush=True)
        for arm in ("families", "feature_fit"):
            for k, v in b[arm].items():
                print(f"  {arm:11s} {k:8s} loss {v['loss']:+.2f} [{v['loss_ci95'][0]:+.2f}, {v['loss_ci95'][1]:+.2f}]  "
                      f"alone {v['alone']:+.2f} [{v['alone_ci95'][0]:+.2f}, {v['alone_ci95'][1]:+.2f}]  -> {v['verdict']}", flush=True)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--vggt", default=os.environ.get("VGGT_DIR", "/workspace/vggt"))
    r.add_argument("--batch", type=int, default=2)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--out", default="results/vggt_pair_patch.json")
    a = sub.add_parser("analyze")
    a.add_argument("--out", default="results/vggt_pair_patch.json")
    a.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    {"run": run, "analyze": run_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
