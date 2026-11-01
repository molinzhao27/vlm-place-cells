"""Experiment 27 (docs/experiments.md): is VGGT's causal position subspace specific to position, or is it where the
camera token varies? Position PCA 10 is compared with ordinary PCA, random and difference-matched subspaces, a short
rank sweep, and position fits on heading-balanced frames, all on vggt_patch.py's targets and donors and with its
patching path (as experiment 23 re-ran it on the RTX 4070).

    # the RTX 4070 through gq (GPU): recording, reproduction check, every patch, then the analysis
    .venv/bin/python scripts/vggt_subspace_controls.py run --vggt $VGGT_DIR --batch 2 [--smoke]
    # anywhere (CPU): bootstrap intervals, the registered rule, the summary
    .venv-mac/bin/python scripts/vggt_subspace_controls.py analyze [--smoke]

run      VGGT single mode (8 anchors + 1 walk frame, room A) on vggt_patch.py's draw (default_rng(0): 2,048 fit frames
         from even walks, 128 targets from odd walks, one donor > 3 m from each). Records the residual at the walk
         frame's camera token after global blocks 21 and 19, builds each subspace Q, writes h_t + Q Q^T (h_d - h_t)
         into the target's run, and keeps the patched walk frame's (x, y) and heading for every pair. Then it records
         the other 4,096 even-walk frames and runs the two 6,144-frame position fits (unbalanced, heading-balanced).
         -> data/subspace_controls/raw.{npz,json}, resid.npz (--smoke: data/subspace_controls/smoke/)
analyze  per-pair measures (transfer, remaining distance to the donor, perpendicular error, heading change, heading gap
         closed), percentile bootstrap intervals over pairs (10,000 resamples, seed 0, paired across conditions), the
         registered decision rule -> results/vggt_subspace_controls.json (top-level "summary")
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

OUT = "data/subspace_controls"
BLOCKS = [21, 19]                  # block 21 (the paper's 0.85) is primary
REF_4070 = {19: {"share": 0.9258939623832703, "transfer": 0.4000054934943613},      # vggt_patch_probes_4070.json
            21: {"share": 0.934897243976593, "transfer": 0.853985131158677}}
SWEEP = (2, 4, 10, 20)
KMAX = 30                          # share matching searches k = 1..30
N_FOLDS = 8                        # cross-fitting of difference PCA
MARGIN = 0.2                       # decision rule
HEAD_GAP_MIN = 30.0                # heading gap closed: pairs whose clean predicted headings differ by >= 30 deg
N_BOOT = 10000
LABELS = {
    "full": "full token",
    "position_pca10": "position PCA 10 (standard, 2,048 fit frames)",
    "position_pca2": "position PCA 2", "position_pca4": "position PCA 4", "position_pca20": "position PCA 20",
    "ordinary_pca2": "ordinary PCA 2", "ordinary_pca4": "ordinary PCA 4", "ordinary_pca10": "ordinary PCA 10",
    "ordinary_pca20": "ordinary PCA 20",
    "random10": "random 10",
    "difference_pca10": "difference PCA 10 (cross-fitted over 8 folds of pairs)",
    "oracle_difference_pca10": "oracle difference PCA 10 (in-sample, descriptive)",
    "position_pca10_all_even": "position PCA 10 fit on all 6,144 even-walk frames",
    "position_pca10_heading_balanced": "position PCA 10 fit on heading-balanced even-walk frames",
}


def pos_bins(xy):
    return np.clip(xy[:, 0].astype(int), 0, 9) * 10 + np.clip(xy[:, 1].astype(int), 0, 9)


def sectors(theta):
    """90-degree sectors of true heading from 0, as experiment 13c's heading-matched donors."""
    return (theta % (2 * np.pi) / (np.pi / 2)).astype(int) % 4


def bin_mean_subspace(H, labels, k, min_count=5):
    """vggt_patch.mean_subspace with the minimum bin count as a parameter (5 there; the smoke run lowers it)."""
    labels = torch.as_tensor(labels, device=H.device)
    means = torch.stack([H[labels == u].mean(0) for u in labels.unique() if (labels == u).sum() >= min_count])
    _, _, Vt = torch.linalg.svd(means - means.mean(0), full_matrices=False)
    return Vt[:k].T


def balanced_set(xy, theta, rng, min_per_sector=2):
    """Indices with, in each 1 m bin, the same number of frames from each 90-degree heading sector (the smallest
    sector's count, drawn at random); bins with fewer than min_per_sector frames in some sector are dropped."""
    pb, sec = pos_bins(xy), sectors(theta)
    keep, n_bins = [], 0
    for b in np.unique(pb):
        cells = [np.flatnonzero((pb == b) & (sec == s)) for s in range(4)]
        m = min(len(c) for c in cells)
        if m < min_per_sector:
            continue
        n_bins += 1
        for c in cells:
            keep.extend(rng.choice(c, m, replace=False).tolist())
    return np.sort(np.array(keep, dtype=int)), n_bins


def resultant_median(xy, theta, min_count=5):
    """Median over 1 m bins (>= min_count frames) of the length of the mean heading vector (cos, sin): 0 when every
    heading is equally represented, 1 when all frames in a bin face the same way."""
    pb = pos_bins(xy)
    r = [np.hypot(np.cos(theta[pb == b]).mean(), np.sin(theta[pb == b]).mean())
         for b in np.unique(pb) if (pb == b).sum() >= min_count]
    return float(np.median(r)), len(r)


# ------------------------------------------------------------------------------------------------ run
@torch.no_grad()
def run(args):
    import torch.nn.functional as F
    sys.path.insert(0, args.vggt)
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from spatialcells.vggt_cells import anchors
    from spatialcells.vggt_control import RES, umeyama
    from spatialcells.vggt_patch import mean_subspace, to_room

    out = f"{OUT}/smoke" if args.smoke else OUT
    os.makedirs(out, exist_ok=True)
    n_fit, n_test = (384, 8) if args.smoke else (args.n_fit, args.n_test)
    min_count, min_per_sector = (4, 1) if args.smoke else (5, 2)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)

    rng = np.random.default_rng(0)                      # the same draw as vggt_patch.py (no --controls)
    frames = np.load("data/walks/frames.npy", mmap_mode="r")
    meta = dict(np.load("data/walks/meta.npz"))
    xy_true = np.stack([meta["x"], meta["y"]], 1)
    theta_true = meta["theta"]
    even = np.flatnonzero(meta["walk"] % 2 == 0)
    odd = np.flatnonzero(meta["walk"] % 2 == 1)
    fit = np.sort(rng.choice(even, n_fit, replace=False))
    targets = rng.choice(odd, n_test, replace=False)
    donors = np.array([rng.choice(odd[np.hypot(*(xy_true[odd] - xy_true[t]).T) > 3]) for t in targets])
    rest = np.setdiff1d(even, fit)                      # the other even-walk frames, sorted
    if args.smoke:
        rest = np.sort(np.random.default_rng(3).choice(rest, 256, replace=False))
    perm = np.random.default_rng(1).permutation(n_test)
    folds = np.array_split(perm, N_FOLDS)
    pos_bin = pos_bins(xy_true[fit])

    A = anchors("data/walks", "A")
    model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
    agg = model.aggregator
    P = agg.patch_start_idx + (RES // 14) ** 2
    n_anchor = len(A["images"])
    cam = n_anchor * P                                  # the walk frame's camera token in the global blocks' layout
    rec, patch, recording = {}, {}, [True]

    def hook(g):
        def f(_, __, o):
            if recording[0]:
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
        """The walk frame's (x, y) (n, 2) and heading (n,) in room coordinates, each run aligned by its own anchor fit
        (as vggt_patch.predict / walk_pose)."""
        seqs = torch.stack([torch.cat([anchor_px, prep(frames[[i]])]) for i in idx])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, _ = agg(seqs)
            pose = model.camera_head(tokens)[-1]
        ext, _ = pose_encoding_to_extri_intri(pose.float(), (RES, RES))
        ext = ext.cpu().numpy().astype(np.float64)
        R, t = ext[..., :3], ext[..., 3]
        cams = np.stack([-np.einsum("bsji,bsj->bsi", R, t), np.einsum("bsji,j->bsi", R, np.array([0.0, 0.0, 1.0]))], 1)
        xy, th = zip(*(to_room(c, umeyama(c[0][:n_anchor], gt_anchor)) for c in cams))
        return np.stack(xy)[:, n_anchor], np.stack(th)[:, n_anchor]

    def run_clean(idx):
        xys, ths, got = [], [], {}
        for b0 in range(0, len(idx), args.batch):
            rec.clear()
            xy, th = predict(idx[b0:b0 + args.batch])
            xys.append(xy)
            ths.append(th)
            for k, v in rec.items():
                got.setdefault(k, []).append(v)
        return np.concatenate(xys), np.concatenate(ths), {k: torch.cat(v) for k, v in got.items()}

    def run_patched(g, repl):
        """repl: (n_test, C) replacement tokens at block g for the targets, in order."""
        recording[0] = False
        xys, ths = [], []
        for b0 in range(0, n_test, args.batch):
            patch.clear()
            patch[g] = repl[b0:b0 + args.batch]
            xy, th = predict(targets[b0:b0 + args.batch])
            xys.append(xy)
            ths.append(th)
        patch.clear()
        recording[0] = True
        return np.concatenate(xys), np.concatenate(ths)

    # ---- clean passes, in experiment 23's order (fit, targets, donors), then the other even frames later
    pred_f, th_f, H_f = run_clean(fit)
    pred_t, th_t, H_t = run_clean(targets)
    pred_d, th_d, H_d = run_clean(donors)
    clean_err = float(np.median(np.hypot(*(pred_t - xy_true[targets]).T)))
    log(f"clean passes done; clean target error {clean_err:.3f} m (4070 run 0.432)")

    raw = {"pred_t": pred_t, "th_t": th_t, "pred_d": pred_d, "th_d": th_d, "fit": fit, "targets": targets,
           "donors": donors, "rest": rest, "xy_true_t": xy_true[targets], "theta_true_t": theta_true[targets],
           "pred_fit": pred_f, "th_fit": th_f, "perm": perm}
    resid = {f"fit_{g}": H_f[g].numpy() for g in BLOCKS}
    resid.update({f"t_{g}": H_t[g].numpy() for g in BLOCKS})
    resid.update({f"d_{g}": H_d[g].numpy() for g in BLOCKS})
    rep = {"experiment": 27, "smoke": bool(args.smoke), "n_fit": int(n_fit), "n_test": int(n_test), "n_rest": int(len(rest)),
           "batch": args.batch, "n_folds": N_FOLDS, "clean_error_m": clean_err,
           "target_donor_distance_m_median": float(np.median(np.hypot(*(pred_d - pred_t).T))),
           "labels": LABELS, "blocks": {}}

    # the 2,048 fit residuals against experiment 23's recording (descriptive)
    e23 = "data/units_subspace/record.npz"
    rep["vs_exp23_recording"] = {}
    if os.path.exists(e23) and not args.smoke:
        R23 = np.load(e23)
        for g in BLOCKS:
            a, b = H_f[g].numpy(), R23[f"post_fit_{g}"]
            if a.shape == b.shape:
                rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
                rep["vs_exp23_recording"][str(g)] = {"median_rel_diff": float(np.median(rel)), "max_rel_diff": float(rel.max()),
                                                     "identical": bool(np.array_equal(a, b))}
        log(f"fit residuals vs experiment 23: {rep['vs_exp23_recording']}")

    def save():
        np.savez(f"{out}/raw.npz", **raw)
        json.dump(rep, open(f"{out}/raw.json", "w"), indent=1)

    proj = lambda Q, v: (v @ Q) @ Q.T
    share_pp = lambda Q, d: ((d @ Q) ** 2).sum(1) / (d ** 2).sum(1)
    overlap = lambda Q1, Q2: float(((Q1.T @ Q2) ** 2).sum() / min(Q1.shape[1], Q2.shape[1]))

    def var_frac(Q, Hc):
        return float(((Hc @ Q) ** 2).sum() / (Hc ** 2).sum())

    def patch_cond(g, name, repl, shares, desc):
        xy, th = run_patched(g, repl.cpu())
        raw[f"b{g}/{name}/xy"], raw[f"b{g}/{name}/th"] = xy, th
        raw[f"b{g}/{name}/share"] = np.asarray(shares, np.float64)
        moved = xy - pred_t
        gap = pred_d - pred_t
        tr = (moved * gap).sum(1) / (gap ** 2).sum(1)
        desc.update(share_median=float(np.median(shares)), transfer_mean=float(tr.mean()),
                    transfer_sem=float(tr.std() / np.sqrt(len(tr))))
        rep["blocks"][str(g)]["subspaces"][name] = desc
        log(f"block {g} {name}: dims {desc.get('dims')}, share {desc['share_median']:.3f}, transfer {tr.mean():.3f}")
        save()
        return tr

    # ---- phase A: subspaces fit on the 2,048 fit frames, and the difference subspaces
    for g in BLOCKS:
        Hf = H_f[g].cuda()
        Hc = Hf - Hf.mean(0)
        ht, hd = H_t[g].cuda(), H_d[g].cuda()
        d = hd - ht
        C = Hf.shape[1]
        rep["blocks"][str(g)] = {"subspaces": {}, "token_norm_median": float(ht.norm(dim=1).median())}
        blk = rep["blocks"][str(g)]

        Qpos = mean_subspace(Hf, pos_bin, 10)           # exactly as vggt_patch.py
        Qpos20 = mean_subspace(Hf, pos_bin, 20)
        Qord = torch.linalg.svd(Hc, full_matrices=False)[2][:KMAX].T
        gen = torch.Generator("cuda").manual_seed(g)      # vggt_patch.py's "random 10" at this block
        Qr, _ = torch.linalg.qr(torch.randn(C, 10, device="cuda", generator=gen))
        Qdiff = []                                        # per fold: fit on the other folds' differences (uncentered)
        for f in folds:
            others = np.setdiff1d(np.arange(n_test), f)
            Qdiff.append(torch.linalg.svd(d[others], full_matrices=False)[2][:KMAX].T)
        Qorc = torch.linalg.svd(d, full_matrices=False)[2][:10].T

        def diff_shares(k):
            s = torch.empty(n_test, device="cuda")
            for f, Qf in zip(folds, Qdiff):
                s[f] = share_pp(Qf[:, :k], d[f])
            return s

        def diff_repl(k):
            r = ht.clone()
            for f, Qf in zip(folds, Qdiff):
                r[f] = ht[f] + proj(Qf[:, :k], d[f])
            return r

        # medians: numpy's (mean of the two middle pairs) everywhere; the reproduction check also uses torch's
        # lower-middle median, which is how vggt_patch.py computed the reference shares
        med = lambda x: float(np.median(x.cpu().numpy()))
        s_pos = med(share_pp(Qpos, d))
        s_pos_torch = float(share_pp(Qpos, d).median())
        kmax_d = min(Qf.shape[1] for Qf in Qdiff)
        med_ord = [med(share_pp(Qord[:, :k], d)) for k in range(1, Qord.shape[1] + 1)]
        med_diff = [med(diff_shares(k)) for k in range(1, kmax_d + 1)]
        first = lambda med: next((k for k, m in enumerate(med, 1) if m >= s_pos), None)
        k_ord, k_diff = first(med_ord), first(med_diff)
        k_d = max(10, k_diff if k_diff is not None else kmax_d)
        k_d = min(k_d, kmax_d)
        blk["share_matching"] = {
            "position_pca10_share_median": s_pos,
            "k_match_ordinary": k_ord, "k_match_ordinary_reached": k_ord is not None,
            "k_match_difference": k_diff, "k_match_difference_reached": k_diff is not None,
            "difference_matched_dims": k_d,
            "ordinary_share_median_by_k": med_ord, "difference_share_median_by_k": med_diff}
        if k_ord is None:
            k_ord = Qord.shape[1]
        log(f"block {g}: position PCA 10 share {s_pos:.3f}; k_match ordinary {blk['share_matching']['k_match_ordinary']}, "
            f"difference {blk['share_matching']['k_match_difference']}; difference-matched control {k_d} dims")

        def gdesc(Q, kind):
            return {"dims": int(Q.shape[1]), "kind": kind, "fit_variance_frac": var_frac(Q, Hc),
                    "overlap_with_position_pca10": overlap(Qpos, Q)}

        conds = [("position_pca10", Qpos, "position"), ("full", None, "full"), ("ordinary_pca10", Qord[:, :10], "ordinary"),
                 ("difference_pca10", "diff10", "difference"), ("random10", Qr, "random"),
                 ("oracle_difference_pca10", Qorc, "oracle")]
        if k_d != 10:
            conds.append((f"difference_pca{k_d}", f"diff{k_d}", "difference"))
        conds += [(f"position_pca{k}", Qpos20[:, :k] if k > 10 else Qpos[:, :k], "position") for k in SWEEP if k != 10]
        conds += [(f"ordinary_pca{k}", Qord[:, :k], "ordinary") for k in SWEEP if k != 10]
        if k_ord not in SWEEP:
            conds.append((f"ordinary_pca{k_ord}", Qord[:, :k_ord], "ordinary"))
        blk["difference_matched"] = f"difference_pca{k_d}"
        blk["ordinary_share_matched"] = f"ordinary_pca{k_ord}"
        for name, Q, kind in conds:
            if kind == "full":
                tr = patch_cond(g, name, hd, np.ones(n_test), {"dims": int(C), "kind": kind, "fit_variance_frac": 1.0})
                continue
            if kind == "difference":
                k = int(Q[4:])
                vf = float(np.mean([var_frac(Qf[:, :k], Hc) for Qf in Qdiff]))
                ov = float(np.mean([overlap(Qpos, Qf[:, :k]) for Qf in Qdiff]))
                patch_cond(g, name, diff_repl(k), diff_shares(k).cpu().numpy(),
                           {"dims": k, "kind": kind, "fit_variance_frac": vf, "overlap_with_position_pca10": ov,
                            "fold_overlap_mean": float(np.mean([overlap(Qdiff[i][:, :k], Qdiff[j][:, :k])
                                                                for i in range(N_FOLDS) for j in range(i + 1, N_FOLDS)]))})
                continue
            tr = patch_cond(g, name, ht + proj(Q, d), share_pp(Q, d).cpu().numpy(), gdesc(Q, kind))
            if name == "position_pca10":
                ref = REF_4070[g]
                blk["reproduces_4070"] = {"share": s_pos_torch, "share_ref": ref["share"], "share_numpy_median": s_pos,
                                          "transfer": float(tr.mean()), "transfer_ref": ref["transfer"],
                                          "ok": bool(abs(s_pos_torch - ref["share"]) <= 0.02
                                                     and abs(tr.mean() - ref["transfer"]) <= 0.05)}
                log(f"block {g} reproduction: {blk['reproduces_4070']}")
        raw[f"b{g}/Qpos10"] = Qpos.cpu().numpy()
        raw[f"b{g}/Qord{KMAX}"] = Qord.cpu().numpy()
        save()

    # ---- phase B: the other even-walk frames, then position PCA 10 on all of them, unbalanced and heading-balanced
    log(f"recording the other {len(rest)} even-walk frames")
    pred_r, th_r, H_r = run_clean(rest)
    raw.update(pred_rest=pred_r, th_rest=th_r)
    resid.update({f"rest_{g}": H_r[g].numpy() for g in BLOCKS})
    np.savez(f"{out}/resid.npz", **resid)
    all_even = np.concatenate([fit, rest])
    xy_e, th_e = xy_true[all_even], theta_true[all_even]
    bal, n_bal_bins = balanced_set(xy_e, th_e, np.random.default_rng(2), min_per_sector)
    raw["balanced_rows"] = bal
    rep["heading_balance"] = {
        "n_all_even": int(len(all_even)), "n_balanced": int(len(bal)), "balanced_bins": int(n_bal_bins),
        "resultant_median_fit2048": resultant_median(xy_true[fit], theta_true[fit], min_count)[0],
        "resultant_median_all_even": resultant_median(xy_e, th_e, min_count)[0],
        "resultant_median_balanced": resultant_median(xy_e[bal], th_e[bal], min_count)[0],
        "sector_counts_fit2048": np.bincount(sectors(theta_true[fit]), minlength=4).tolist(),
        "sector_counts_balanced": np.bincount(sectors(th_e[bal]), minlength=4).tolist()}
    log(f"heading balance: {rep['heading_balance']}")
    for g in BLOCKS:
        He = torch.cat([H_f[g], H_r[g]]).cuda()
        Hc = H_f[g].cuda() - H_f[g].cuda().mean(0)
        ht, hd = H_t[g].cuda(), H_d[g].cuda()
        d = hd - ht
        Qpos = torch.from_numpy(raw[f"b{g}/Qpos10"]).cuda()
        Qe = bin_mean_subspace(He, pos_bins(xy_e), 10, min_count)
        Qb = bin_mean_subspace(He[bal], pos_bins(xy_e[bal]), 10, min_count)
        rep["blocks"][str(g)]["overlap_balanced_vs_all_even"] = float(((Qe.T @ Qb) ** 2).sum() / 10)
        for name, Q in (("position_pca10_all_even", Qe), ("position_pca10_heading_balanced", Qb)):
            patch_cond(g, name, ht + proj(Q, d), share_pp(Q, d).cpu().numpy(),
                       {"dims": int(Q.shape[1]), "kind": "position", "fit_variance_frac": var_frac(Q, Hc),
                        "overlap_with_position_pca10": float(((Qpos.T @ Q) ** 2).sum() / 10)})
    rep["elapsed_s"] = time.time() - t0
    save()
    log(f"done; wrote {out}/raw.npz, raw.json, resid.npz")
    run_analyze(argparse.Namespace(smoke=args.smoke, out=args.out))


# ------------------------------------------------------------------------------------------------ analyze
def angdiff(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def ci(b):
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def pair_measures(xy, th, R):
    """Per pair: transfer, remaining distance to the donor, perpendicular error, moved, error to the true position,
    heading change, heading gap closed (NaN where the clean headings differ by < HEAD_GAP_MIN)."""
    pt, pd = R["pred_t"], R["pred_d"]
    m, gap = xy - pt, pd - pt
    tr = (m * gap).sum(1) / (gap ** 2).sum(1)
    hgap = angdiff(R["th_t"], R["th_d"])
    closed = np.where(np.degrees(hgap) >= HEAD_GAP_MIN, 1 - angdiff(th, R["th_d"]) / np.maximum(hgap, 1e-9), np.nan)
    return {"transfer": tr,
            "distance_to_donor_m": np.linalg.norm(xy - pd, axis=1),
            "perpendicular_error_m": np.linalg.norm(m - tr[:, None] * gap, axis=1),
            "moved_m": np.linalg.norm(m, axis=1),
            "error_m": np.linalg.norm(xy - R["xy_true_t"], axis=1),
            "heading_change_deg": np.degrees(angdiff(th, R["th_t"])),
            "heading_to_donor_deg": np.degrees(angdiff(th, R["th_d"])),
            "heading_gap_closed": closed}


STATS = {"transfer": "mean", "distance_to_donor_m": "median", "perpendicular_error_m": "median", "moved_m": "median",
         "error_m": "median", "heading_change_deg": "median", "heading_to_donor_deg": "median",
         "heading_gap_closed": "mean"}


def stat(x, how, axis=None):
    if how == "mean":
        return np.nanmean(x, axis=axis)
    return np.nanmedian(x, axis=axis)


def run_analyze(args):
    d = f"{OUT}/smoke" if args.smoke else OUT
    Z = np.load(f"{d}/raw.npz")
    RJ = json.load(open(f"{d}/raw.json"))
    R = {k: Z[k] for k in ("pred_t", "pred_d", "th_t", "th_d", "xy_true_t")}
    n = len(R["pred_t"])
    idx = np.random.default_rng(0).integers(0, n, (N_BOOT, n))     # one resample set: paired across conditions
    clean_hgap = np.degrees(angdiff(R["th_t"], R["th_d"]))
    res = {"about": "Experiment 27 (docs/experiments.md). VGGT single mode, room A, the walk frame's camera token after "
                    "global block g; 128 target-donor pairs (vggt_patch.py's draw). Transfer is the fraction of the "
                    "target-donor line the prediction moves (mean); meters and degrees are medians over pairs; heading "
                    "gap closed is a mean over pairs whose clean predicted headings differ by >= 30 deg. Intervals: "
                    "95% percentile bootstrap over pairs (10,000 resamples, seed 0), paired across conditions.",
           "labels": RJ["labels"], "run": {k: v for k, v in RJ.items() if k not in ("blocks", "labels")},
           "clean": {"target_error_m_median": RJ["clean_error_m"],
                     "target_donor_distance_m_median": float(np.median(np.linalg.norm(R["pred_d"] - R["pred_t"], axis=1))),
                     "target_donor_heading_deg_median": float(np.median(clean_hgap)),
                     "pairs_with_heading_gap_ge_30deg": int((clean_hgap >= HEAD_GAP_MIN).sum())},
           "blocks": {}}
    checks = {f"block{g}_reproduces_4070": RJ["blocks"][str(g)]["reproduces_4070"]["ok"]
              for g in BLOCKS if "reproduces_4070" in RJ["blocks"].get(str(g), {})}
    res["checks"] = checks
    print("checks:", json.dumps(checks), flush=True)
    if not args.smoke and (len(checks) < len(BLOCKS) or not all(checks.values())):
        sys.exit("the reproduction check failed: no comparison is reported (registration)")

    def boot_diff(a, b, how="mean"):
        """Paired difference of a statistic (a minus b) with its bootstrap interval."""
        dd = a - b
        return {"diff": float(stat(dd, how)), "ci95": ci(stat(dd[idx], how, axis=1))}

    summary_blocks = {}
    for g in BLOCKS:
        B = RJ["blocks"][str(g)]
        names = list(B["subspaces"])
        M = {nm: pair_measures(Z[f"b{g}/{nm}/xy"], Z[f"b{g}/{nm}/th"], R) for nm in names}
        conds = {}
        for nm in names:
            c = {k: v for k, v in B["subspaces"][nm].items() if k not in ("transfer_mean", "transfer_sem")}
            for meas, how in STATS.items():
                x = M[nm][meas]
                c[meas] = float(stat(x, how))
                c[f"{meas}_ci95"] = ci(stat(x[idx], how, axis=1))
            conds[nm] = c
        dm = B["difference_matched"]
        D_ord = boot_diff(M["position_pca10"]["transfer"], M["ordinary_pca10"]["transfer"])
        D_diff = boot_diff(M["position_pca10"]["transfer"], M[dm]["transfer"])
        ord_ok = D_ord["diff"] >= MARGIN and D_ord["ci95"][0] > 0
        diff_ok = D_diff["diff"] >= MARGIN and D_diff["ci95"][0] > 0
        verdict = ("position-specific" if ord_ok and diff_ok else
                   "not the token's main variance, but not specific to position" if ord_ok else "generic")
        blk = {"conditions": conds, "share_matching": B["share_matching"], "difference_matched": dm,
               "ordinary_share_matched": B["ordinary_share_matched"], "reproduces_4070": B.get("reproduces_4070"),
               "position_minus_ordinary_pca10_transfer": D_ord, "position_minus_difference_matched_transfer": D_diff,
               "rule": {"margin": MARGIN, "ordinary_condition_met": bool(ord_ok), "difference_condition_met": bool(diff_ok)},
               "verdict": verdict, "token_norm_median": B["token_norm_median"]}
        # descriptive paired differences against position PCA 10
        blk["position_pca10_minus"] = {
            nm: {meas: boot_diff(M["position_pca10"][meas], M[nm][meas], STATS[meas])
                 for meas in ("transfer", "distance_to_donor_m", "perpendicular_error_m", "heading_gap_closed")}
            for nm in names if nm != "position_pca10"}
        # heading-balanced fits (secondary)
        if "position_pca10_heading_balanced" in M:
            Mb, Me = M["position_pca10_heading_balanced"], M["position_pca10_all_even"]
            dt = boot_diff(Mb["transfer"], Me["transfer"])
            dc = boot_diff(Me["heading_gap_closed"], Mb["heading_gap_closed"])     # how much less the balanced fit closes
            within = abs(dt["diff"]) <= 0.1
            leak = dc["diff"] >= 0.2 and dc["ci95"][0] > 0
            blk["heading_balanced"] = {
                "balanced_minus_all_even_transfer": dt,
                "all_even_minus_balanced_heading_gap_closed": dc,
                "balanced_minus_all_even_heading_change_deg": boot_diff(Mb["heading_change_deg"], Me["heading_change_deg"], "median"),
                "balanced_minus_standard2048_transfer": boot_diff(Mb["transfer"], M["position_pca10"]["transfer"]),
                "overlap_balanced_vs_all_even": B.get("overlap_balanced_vs_all_even"),
                "transfer_within_0.1": bool(within), "heading_leak": bool(leak),
                "reading": (("position transfer does not depend on heading in the bin means" if within else
                             "balancing changes the position transfer by more than 0.1") + "; " +
                            ("part of the heading content came from heading leaking into the position bins" if leak else
                             "heading moves with position even after balancing (a pose subspace)"))}
        res["blocks"][str(g)] = blk

        def short(nm):
            c = conds[nm]
            return {k: c[k] for k in ("dims", "share_median", "fit_variance_frac", "transfer", "transfer_ci95",
                                      "distance_to_donor_m", "distance_to_donor_m_ci95", "perpendicular_error_m",
                                      "perpendicular_error_m_ci95", "heading_change_deg", "heading_change_deg_ci95",
                                      "heading_gap_closed", "heading_gap_closed_ci95") if k in c} | (
                {"overlap_with_position_pca10": c["overlap_with_position_pca10"]} if "overlap_with_position_pca10" in c else {})

        summary_blocks[f"block{g}"] = {
            "verdict": verdict,
            "position_minus_ordinary_pca10_transfer": D_ord,
            "position_minus_difference_matched_transfer": D_diff,
            "difference_matched_key": dm, "difference_matched_dims": B["share_matching"]["difference_matched_dims"],
            "k_match_ordinary": B["share_matching"]["k_match_ordinary"],
            "k_match_difference": B["share_matching"]["k_match_difference"],
            "conditions": {nm: short(nm) for nm in names},
            **({"heading_balanced": {k: blk["heading_balanced"][k] for k in
                                     ("balanced_minus_all_even_transfer", "all_even_minus_balanced_heading_gap_closed",
                                      "transfer_within_0.1", "heading_leak", "reading")}} if "heading_balanced" in blk else {})}

    res["summary"] = {
        "experiment": 27,
        "units": "transfer and heading_gap_closed are fractions (0 = no effect, 1 = the donor's value; means over pairs); "
                 "*_m are meters and *_deg degrees (medians over pairs); share_median is the median fraction of the "
                 "target-donor residual difference (squared norm) inside the subspace; fit_variance_frac the fraction "
                 "of the 2,048 fit frames' residual variance inside it; ci95 = paired percentile bootstrap over pairs",
        "verdict": summary_blocks.get("block21", {}).get("verdict"),
        "verdict_block19": summary_blocks.get("block19", {}).get("verdict"),
        "reproduction": {f"block{g}": RJ["blocks"][str(g)].get("reproduces_4070") for g in BLOCKS},
        "fit_residuals_vs_exp23": RJ.get("vs_exp23_recording"),
        "clean": res["clean"],
        "heading_balance_check": RJ.get("heading_balance"),
        **summary_blocks}
    path = args.out if not args.smoke else f"{d}/vggt_subspace_controls_smoke.json"
    json.dump(res, open(path, "w"), indent=1)
    S = res["summary"]
    for g in BLOCKS:
        b = S[f"block{g}"]
        print(f"block {g}: verdict {b['verdict']}; position - ordinary PCA 10 {b['position_minus_ordinary_pca10_transfer']}; "
              f"position - {b['difference_matched_key']} {b['position_minus_difference_matched_transfer']}", flush=True)
        for nm, c in b["conditions"].items():
            print(f"  {nm:34s} dims {c['dims']:5d} share {c['share_median']:.3f} var {c.get('fit_variance_frac', float('nan')):.3f} "
                  f"transfer {c['transfer']:+.3f} [{c['transfer_ci95'][0]:+.3f}, {c['transfer_ci95'][1]:+.3f}] "
                  f"to donor {c['distance_to_donor_m']:.2f} m perp {c['perpendicular_error_m']:.2f} m "
                  f"heading {c['heading_change_deg']:.1f} deg closed {c['heading_gap_closed']:+.2f}", flush=True)
        if "heading_balanced" in b:
            print(f"  heading-balanced: {b['heading_balanced']}", flush=True)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--vggt", default=os.environ.get("VGGT_DIR", "/workspace/vggt"))
    r.add_argument("--batch", type=int, default=2)
    r.add_argument("--n-fit", type=int, default=2048)
    r.add_argument("--n-test", type=int, default=128)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--out", default="results/vggt_subspace_controls.json")
    a = sub.add_parser("analyze")
    a.add_argument("--out", default="results/vggt_subspace_controls.json")
    a.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    {"run": run, "analyze": run_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
