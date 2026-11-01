"""Experiment 23 (docs/experiments.md): do VGGT's strictly heading-invariant place units write into, or read
from, the 10-d position subspace that carries position causally?

    # the RTX 4070 through gq (GPU): unit groups, then the recording, checks and causal check
    .venv/bin/python scripts/vggt_units_subspace.py score
    .venv/bin/python scripts/vggt_units_subspace.py record --vggt $VGGT_DIR --batch 2 [--smoke]
    # the Mac (CPU): fractions, bootstrap intervals, decision rule
    .venv-mac/bin/python scripts/vggt_units_subspace.py analyze

score     runs analyze.analyze_layer(..., quadrant=True) unchanged on the stored activations of VGGT-single
          blocks 19 and 21 and VGGT-walk block 19, recomputes each pole's opposite-sector score, checks both
          against results/summary_quadrant.json, and groups the 4,096 neurons of each (strictly invariant
          place / non-invariant place / non-place). -> data/units_subspace/units.{npz,json}
record    runs VGGT single mode (8 anchors + 1 walk frame) on vggt_patch.py's 2,048 fit frames and 128
          target-donor pairs (same random draw), recording at the walk frame's camera token in global blocks
          19 and 21: the block output (the residual the causal test patches), the residual entering the MLP
          (x_mid) and the MLP neurons (fc2 input). It rebuilds position PCA 10 as vggt_patch.py does, checks it
          against the stored 4070 run (share and transfer), checks that block output - x_mid equals
          LayerScale * fc2(a), and runs the causal check (interchange and mean replacement of neuron sets at
          fc2's input). -> data/units_subspace/record.{npz,json} (--smoke: data/units_subspace/smoke/)
analyze   fraction of each neuron's write vector (LayerScale * fc2 column) inside the subspace, per group,
          with bootstrap intervals and the registered rule; secondary measures (a)-(f); the causal summary.
          -> results/vggt_units_subspace.json (top-level "summary")
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
from spatialcells import analyze  # noqa: E402

OUT = "data/units_subspace"
CONDS = [("vggt-single", "L19"), ("vggt-single", "L21"), ("vggt-walk", "L19")]
BLOCKS = [19, 21]
THRESH = 0.5                       # \HeadInvThresh: opposite-sector score for a strictly invariant place unit
K = 10                             # position PCA 10
D = 1024
R_NULL = K / D
SQ_KEYS = ("frac_place", "frac_place_heading_invariant_quadrant")
REF_4070 = {19: {"share": 0.9258939623832703, "transfer": 0.4000054934943613},      # vggt_patch_probes_4070.json
            21: {"share": 0.934897243976593, "transfer": 0.853985131158677}}
GROUP_NAMES = {0: "nonplace", 1: "noninvariant_place", 2: "invariant"}


# ------------------------------------------------------------------------------------------------ score
def set_device(d):
    analyze.dev = d
    analyze.KER = analyze.KER.to(d)


def score_layer(X, meta):
    """analyze_layer unchanged, plus each pole's opposite-sector score, recomputed as its quadrant block does."""
    stats, pu = analyze.analyze_layer(X, meta, np.random.default_rng(0), quadrant=True)
    R = torch.cat([X.clamp_min(0), (-X).clamp_min(0)], 1)
    active = (R > 0).float().mean(0) >= 0.01
    assert torch.equal(torch.nonzero(active).squeeze(1), pu["pole"])
    R = R[:, active]
    bins = analyze.spatial_bins(meta)
    sec = analyze.sectors(meta["theta"])
    hq = []
    for i in (1, 0):                                      # N vs S, then E vs W, as in analyze_layer
        ma, oa = analyze.rate_maps(R, bins, torch.from_numpy(sec == i).to(analyze.dev))
        mb, ob = analyze.rate_maps(R, bins, torch.from_numpy(sec == i + 2).to(analyze.dev))
        hq.append(analyze.map_corr(ma, mb, (oa >= analyze.MIN_OCC) & (ob >= analyze.MIN_OCC)))
    inv_q = (hq[0] + hq[1]) / 2
    frac = (pu["place"] & (inv_q > THRESH)).float().mean().item()
    if frac != stats["frac_place_heading_invariant_quadrant"]:
        sys.exit(f"per-pole score does not reproduce analyze_layer: {frac} vs {stats['frac_place_heading_invariant_quadrant']}")
    per = {"pole": pu["pole"].cpu().numpy(), "place": pu["place"].cpu().numpy(), "inv_q": inv_q.cpu().numpy(),
           "hi": pu["hi"].cpu().numpy(), "stab": pu["stab"].cpu().numpy(), "si": pu["si"].cpu().numpy()}
    return stats, per


def neuron_groups(per, n):
    """Per neuron: 2 strictly invariant (a pole is a place unit with opposite-sector score > THRESH), 1 place but
    none strictly invariant, 0 no place pole (as vggt_ablate.unit_sets, with the strict score)."""
    neuron = per["pole"] % n                              # poles are [positive half, negative half]
    lab = np.zeros(n, np.int8)
    lab[np.unique(neuron[per["place"]])] = 1
    lab[np.unique(neuron[per["place"] & (per["inv_q"] > THRESH)])] = 2
    return lab


def score(args):
    os.makedirs(OUT, exist_ok=True)
    meta = dict(np.load("data/walks/meta.npz"))
    sq = json.load(open(args.summary))
    arrays, report = {}, {"threshold": THRESH, "summary_quadrant": args.summary, "conditions": {}}
    for tag, layer in CONDS:
        t0 = time.time()
        for device in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
            set_device(device)
            X = analyze.load_units(tag, layer, "data/walks", meta)
            stats, per = score_layer(X, meta)
            diff = {k: abs(stats[k] - sq[tag][layer][k]) for k in SQ_KEYS}
            print(f"{tag} {layer} on {device}: " + ", ".join(f"{k} {stats[k]:.4f} (summary_quadrant "
                  f"{sq[tag][layer][k]:.4f})" for k in SQ_KEYS) + f", {time.time() - t0:.0f}s", flush=True)
            if max(diff.values()) <= 0.001:
                break
        n = X.shape[1]
        lab = neuron_groups(per, n)
        key = f"{tag}_{layer}"
        for k, v in per.items():
            arrays[f"{key}_{k}"] = v
        arrays[f"{key}_group"] = lab
        arrays[f"{key}_mean"] = X.mean(0).float().cpu().numpy()      # for mean replacement (as vggt_ablate)
        place_poles = int(per["place"].sum())
        inv_poles = int((per["place"] & (per["inv_q"] > THRESH)).sum())
        report["conditions"][key] = {
            "device": device, "n_poles_active": int(len(per["pole"])), "place_poles": place_poles,
            "strict_invariant_place_poles": inv_poles, "strict_share_of_place_poles": inv_poles / max(place_poles, 1),
            "neurons": {GROUP_NAMES[g]: int((lab == g).sum()) for g in (2, 1, 0)},
            "strict_invariant_negative_pole_share": float(np.mean(per["pole"][per["place"] & (per["inv_q"] > THRESH)] >= n)),
            "here": {k: stats[k] for k in SQ_KEYS}, "summary_quadrant": {k: sq[tag][layer][k] for k in SQ_KEYS},
            "abs_diff": diff, "within_0.001": bool(max(diff.values()) <= 0.001)}
        print(f"  {json.dumps(report['conditions'][key]['neurons'])}", flush=True)
        del X
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    np.savez(f"{OUT}/units.npz", **arrays)
    json.dump(report, open(f"{OUT}/units.json", "w"), indent=1)
    print(f"wrote {OUT}/units.npz, units.json", flush=True)
    if not all(c["within_0.001"] for c in report["conditions"].values()):
        sys.exit("unit scoring does not reproduce summary_quadrant.json within 0.001")


# ------------------------------------------------------------------------------------------------ record
@torch.no_grad()
def record(args):
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
    U = np.load(f"{OUT}/units.npz")
    t0 = time.time()

    rng = np.random.default_rng(0)                      # the same draw as vggt_patch.py (no --controls)
    frames = np.load("data/walks/frames.npy", mmap_mode="r")
    meta = dict(np.load("data/walks/meta.npz"))
    xy_true = np.stack([meta["x"], meta["y"]], 1)
    even = np.flatnonzero(meta["walk"] % 2 == 0)
    odd = np.flatnonzero(meta["walk"] % 2 == 1)
    fit = np.sort(rng.choice(even, n_fit, replace=False))
    targets = rng.choice(odd, n_test, replace=False)
    donors = np.array([rng.choice(odd[np.hypot(*(xy_true[odd] - xy_true[t]).T) > 3]) for t in targets])
    x, y = xy_true[fit, 0], xy_true[fit, 1]
    pos_bin = np.clip(x.astype(int), 0, 9) * 10 + np.clip(y.astype(int), 0, 9)

    A = anchors("data/walks", "A")
    model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
    agg = model.aggregator
    P = agg.patch_start_idx + (RES // 14) ** 2
    n_anchor = len(A["images"])
    cam = n_anchor * P                                  # the walk frame's camera token in the global blocks' layout
    rec, patch_out, patch_fc2, recording = {}, {}, {}, [True]

    def out_hook(g):
        def f(_, __, o):
            if recording[0]:
                rec[f"post{g}"] = o[:, cam].float().cpu()
            if g in patch_out:
                o = o.clone()
                o[:, cam] = patch_out[g].to(o.device, o.dtype)
                return o
        return f

    def mid_hook(g):
        def f(_, inp):
            if recording[0]:
                rec[f"mid{g}"] = inp[0][:, cam].float().cpu()
        return f

    def fc2_hook(g):
        def f(_, inp):
            if recording[0]:
                rec[f"act{g}"] = inp[0][:, cam].float().cpu()
            if g in patch_fc2:
                u, v = patch_fc2[g]
                xx = inp[0].clone()
                xx[:, cam, u] = v.to(xx.device, xx.dtype)
                return (xx,)
        return f

    for g in BLOCKS:
        blk = agg.global_blocks[g]
        blk.register_forward_hook(out_hook(g))
        blk.norm2.register_forward_pre_hook(mid_hook(g))
        blk.mlp.fc2.register_forward_pre_hook(fc2_hook(g))

    def prep(ims):
        xx = torch.from_numpy(np.stack([np.asarray(a) for a in ims])).cuda().permute(0, 3, 1, 2).float() / 255
        return F.interpolate(xx, size=(RES, RES), mode="bicubic", align_corners=False).clamp(0, 1)

    anchor_px = prep(A["images"])
    gt_anchor = np.c_[A["poses"][:, :2], np.ones(n_anchor)]

    def predict(idx):
        """The walk frame's (x, y) in room coordinates, each run aligned by its own anchor fit (as vggt_patch)."""
        seqs = torch.stack([torch.cat([anchor_px, prep(frames[[i]])]) for i in idx])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, _ = agg(seqs)
            pose = model.camera_head(tokens)[-1]
        ext, _ = pose_encoding_to_extri_intri(pose.float(), (RES, RES))
        ext = ext.cpu().numpy().astype(np.float64)
        R, t = ext[..., :3], ext[..., 3]
        cams = np.stack([-np.einsum("bsji,bsj->bsi", R, t), np.einsum("bsji,j->bsi", R, np.array([0.0, 0.0, 1.0]))], 1)
        return np.stack([to_room(c, umeyama(c[0][:n_anchor], gt_anchor))[0][n_anchor] for c in cams])

    def run_clean(idx):
        preds, got = [], {}
        for b0 in range(0, len(idx), args.batch):
            rec.clear()
            preds.append(predict(idx[b0:b0 + args.batch]))
            for k, v in rec.items():
                got.setdefault(k, []).append(v)
        return np.concatenate(preds), {k: torch.cat(v) for k, v in got.items()}

    def run_patched(idx, out_repl=None, fc2_repl=None):
        """out_repl: block -> (n, C) block outputs; fc2_repl: block -> (neuron ids, (n, k) or (k,) values)."""
        recording[0] = False
        preds = []
        for b0 in range(0, len(idx), args.batch):
            sl = slice(b0, b0 + args.batch)
            patch_out.clear()
            patch_fc2.clear()
            patch_out.update({g: v[sl] for g, v in (out_repl or {}).items()})
            patch_fc2.update({g: (u, v[sl] if v.ndim == 2 else v) for g, (u, v) in (fc2_repl or {}).items()})
            preds.append(predict(idx[b0:b0 + args.batch]))
        patch_out.clear()
        patch_fc2.clear()
        recording[0] = True
        return np.concatenate(preds)

    pred_f, H_f = run_clean(fit)
    pred_t, H_t = run_clean(targets)
    pred_d, H_d = run_clean(donors)
    clean_err = float(np.median(np.hypot(*(pred_t - xy_true[targets]).T)))
    print(f"clean passes done, {time.time() - t0:.0f}s; clean target error {clean_err:.3f} m (4070 run 0.432)", flush=True)

    rep = {"n_fit": int(n_fit), "n_test": int(n_test), "batch": args.batch, "clean_error_m": clean_err,
           "fit_clean_error_m": float(np.median(np.hypot(*(pred_f - xy_true[fit]).T))),
           "target_donor_distance_m": float(np.median(np.hypot(*(pred_d - pred_t).T))), "blocks": {}}
    arrays = {"fit": fit, "targets": targets, "donors": donors, "pos_bin": pos_bin,
              "pred_fit": pred_f, "pred_t": pred_t, "pred_d": pred_d}
    gap = pred_d - pred_t
    proj = lambda Q, v: (v @ Q) @ Q.T

    def summarize(pred, erase=False):
        moved = pred - pred_t
        r = {"moved_m": np.linalg.norm(moved, axis=1).tolist(),
             "error_m": np.hypot(*(pred - xy_true[targets]).T).tolist()}
        if not erase:
            r["transfer"] = ((moved * gap).sum(1) / (gap ** 2).sum(1)).tolist()
        r["summary"] = {"moved_m_median": float(np.median(r["moved_m"])), "error_m_median": float(np.median(r["error_m"]))}
        if not erase:
            tr = np.array(r["transfer"])
            r["summary"].update(transfer_mean=float(tr.mean()), transfer_sem=float(tr.std() / np.sqrt(len(tr))))
        return r

    for g in BLOCKS:
        blk = agg.global_blocks[g]
        W2, b2, gam = (p.detach().float() for p in (blk.mlp.fc2.weight, blk.mlp.fc2.bias, blk.ls2.gamma))
        arrays.update({f"W2_{g}": W2.cpu().numpy(), f"b2_{g}": b2.cpu().numpy(), f"gamma2_{g}": gam.cpu().numpy(),
                       f"W1_{g}": blk.mlp.fc1.weight.detach().float().cpu().numpy(),
                       f"b1_{g}": blk.mlp.fc1.bias.detach().float().cpu().numpy(),
                       f"ln2_w_{g}": blk.norm2.weight.detach().float().cpu().numpy(),
                       f"ln2_b_{g}": blk.norm2.bias.detach().float().cpu().numpy()})
        for k in ("post", "mid", "act"):
            arrays[f"{k}_fit_{g}"] = H_f[f"{k}{g}"].numpy()
            arrays[f"{k}_t_{g}"] = H_t[f"{k}{g}"].numpy()
            arrays[f"{k}_d_{g}"] = H_d[f"{k}{g}"].numpy()
        # what block g's MLP adds: block output - x_mid = LayerScale * (fc2 a + b)
        post, mid, act = (H_f[f"{k}{g}"].cuda() for k in ("post", "mid", "act"))
        dlt = post - mid
        recon = (act @ W2.T + b2) * gam
        write_rel = float(((dlt - recon).norm(dim=1) / dlt.norm(dim=1)).median())
        # the recorded neurons against the stored (vast-1, batch 48) activations at the fit frames
        stored = torch.from_numpy(np.load(f"data/acts/vggt-single/L{g}.npy", mmap_mode="r")[fit].astype(np.float32)).cuda()
        a0, s0 = act - act.mean(1, keepdim=True), stored - stored.mean(1, keepdim=True)
        act_r = float(((a0 * s0).sum(1) / (a0.norm(dim=1) * s0.norm(dim=1))).median())
        act_rel = float(((act - stored).norm(dim=1) / stored.norm(dim=1)).median())
        # position PCA 10, as vggt_patch.py
        Hf = post
        mu = Hf.mean(0)
        Q = mean_subspace(Hf, pos_bin, K)
        Qm = mean_subspace(mid, pos_bin, K)
        arrays[f"Q_post_{g}"], arrays[f"Q_mid_{g}"] = Q.cpu().numpy(), Qm.cpu().numpy()
        ht, hd = H_t[f"post{g}"].cuda(), H_d[f"post{g}"].cuda()
        d = hd - ht
        share = float((((proj(Q, d)) ** 2).sum(1) / (d ** 2).sum(1)).median())
        res = {"write_check_median_rel_err": write_rel, "acts_vs_stored_median_r": act_r,
               "acts_vs_stored_median_rel_diff": act_rel, "share_pca10": share,
               "token_norm_median": float(ht.norm(dim=1).median()), "conditions": {}}
        res["conditions"]["position PCA 10, written"] = summarize(run_patched(targets, out_repl={g: (ht + proj(Q, d)).cpu()}))
        res["conditions"]["position PCA 10, erased"] = summarize(
            run_patched(targets, out_repl={g: (ht + proj(Q, mu - ht)).cpu()}), erase=True)
        ref = REF_4070[g]
        tr = res["conditions"]["position PCA 10, written"]["summary"]["transfer_mean"]
        res["reproduces_4070"] = {"share": share, "share_ref": ref["share"], "transfer": tr, "transfer_ref": ref["transfer"],
                                  "ok": bool(abs(share - ref["share"]) <= 0.02 and abs(tr - ref["transfer"]) <= 0.05)}
        res["write_check_ok"] = bool(write_rel < 0.02)
        print(f"block {g} ({time.time() - t0:.0f}s): write check {write_rel:.4f}; neurons vs stored r {act_r:.4f}, "
              f"rel diff {act_rel:.4f}; share {share:.3f} (ref {ref['share']:.3f}); PCA-10 transfer {tr:.3f} "
              f"(ref {ref['transfer']:.3f}); erased error "
              f"{res['conditions']['position PCA 10, erased']['summary']['error_m_median']:.2f} m", flush=True)
        rep["blocks"][g] = res
        np.savez(f"{out}/record.npz", **arrays)
        json.dump(rep, open(f"{out}/record.json", "w"), indent=1)

    # causal check: neuron sets at fc2's input, at the walk frame's camera token
    rng_sets = np.random.default_rng(0)
    for g in BLOCKS:
        lab = U[f"vggt-single_L{g}_group"]
        inv, spec, place = (np.flatnonzero(lab == 2), np.flatnonzero(lab == 1), np.flatnonzero(lab >= 1))
        N = len(inv)
        sets = {"invariant": inv,
                "noninvariant_place": np.sort(rng_sets.choice(spec, min(N, len(spec)), replace=False)),
                "random_place": np.sort(rng_sets.choice(place, N, replace=False)),
                "all_neurons": np.arange(len(lab))}
        means = torch.from_numpy(U[f"vggt-single_L{g}_mean"])
        act_d = H_d[f"act{g}"]
        res = rep["blocks"][g]
        res["sets"] = {k: v.tolist() for k, v in sets.items()}
        for name, u in sets.items():
            ut = torch.from_numpy(u).cuda()
            res["conditions"][f"interchange: {name}"] = summarize(run_patched(targets, fc2_repl={g: (ut, act_d[:, u])}))
            res["conditions"][f"mean replacement: {name}"] = summarize(
                run_patched(targets, fc2_repl={g: (ut, means[u])}), erase=True)
            print(f"block {g} ({time.time() - t0:.0f}s) {name} ({len(u)}): interchange transfer "
                  f"{res['conditions'][f'interchange: {name}']['summary']['transfer_mean']:.4f}; mean replacement moves "
                  f"{res['conditions'][f'mean replacement: {name}']['summary']['moved_m_median']:.3f} m", flush=True)
        json.dump(rep, open(f"{out}/record.json", "w"), indent=1)
    print(f"done, {time.time() - t0:.0f}s; wrote {out}/record.npz, record.json", flush=True)


# ------------------------------------------------------------------------------------------------ analyze
def frac_in(Q, V):
    """Fraction of each column of V (C, n) inside the column space of the orthonormal Q (C, k)."""
    return ((Q.T @ V) ** 2).sum(0) / (V ** 2).sum(0)


def boot_median(x, rng, B=10000):
    x = np.asarray(x, np.float32)
    return np.median(x[rng.integers(0, len(x), (B, len(x)))], axis=1)


def ci(b):
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def compare(f, lab, rng, null):
    """The registered statistics for one measure: group medians with intervals, the two differences, the rule."""
    out = {"groups": {}}
    boots = {}
    for g, name in GROUP_NAMES.items():
        x = f[lab == g]
        boots[name] = boot_median(x, rng)
        out["groups"][name] = {"n": int(len(x)), "median": float(np.median(x)), "median_ci95": ci(boots[name]),
                               "mean": float(x.mean()), "frac_above_null_p95": float((x > null["p95"]).mean())}
    inv = out["groups"]["invariant"]["median"]
    d1, d2 = boots["invariant"] - boots["noninvariant_place"], boots["invariant"] - boots["nonplace"]
    out["invariant_minus_noninvariant_place"] = {
        "diff": inv - out["groups"]["noninvariant_place"]["median"], "ci95": ci(d1)}
    out["invariant_minus_nonplace"] = {"diff": inv - out["groups"]["nonplace"]["median"], "ci95": ci(d2)}
    c = {"i_at_least_2x_random": bool(inv >= 2 * R_NULL), "ii_above_noninvariant_place": bool(ci(d1)[0] > 0),
         "iii_above_nonplace": bool(ci(d2)[0] > 0)}
    if all(c.values()):
        verdict = "connect"
    elif not c["i_at_least_2x_random"] and not c["iii_above_nonplace"]:
        verdict = "don't connect"
    elif c["i_at_least_2x_random"]:
        verdict = "partial, not specific to invariant units"
    else:
        verdict = "partial"
    out.update(conditions=c, verdict=verdict, invariant_over_random=inv / R_NULL)
    return out


def bin_means(Z, labels):
    """Centered mean of Z's rows per label with >= 5 rows (as vggt_patch.mean_subspace)."""
    M = np.stack([Z[labels == u].mean(0) for u in np.unique(labels) if (labels == u).sum() >= 5])
    return M - M.mean(0)


def magnitude(Rc, U, g, key, rng):
    """(f): each part's share of the residual's bin-mean position signal inside Q (additive over parts)."""
    Q = Rc[f"Q_post_{g}"].astype(np.float64)
    V = Rc[f"gamma2_{g}"][:, None].astype(np.float64) * Rc[f"W2_{g}"].astype(np.float64)
    act = Rc[f"act_fit_{g}"].astype(np.float64)
    lab = U[f"{key}_group"]
    pb = Rc["pos_bin"]
    MH = bin_means(Rc[f"post_fit_{g}"].astype(np.float64) @ Q, pb)
    den = (MH ** 2).sum()
    CQ = V.T @ Q                                            # each neuron's write, in subspace coordinates
    share = lambda Z: float((bin_means(Z, pb) * MH).sum() / den)
    out = {"incoming_residual_x_mid": share(Rc[f"mid_fit_{g}"].astype(np.float64) @ Q), "all_neurons": share(act @ CQ)}
    for gi, name in GROUP_NAMES.items():
        m = lab == gi
        out[name] = share(act[:, m] @ CQ[m])
        out[f"{name}_per_neuron"] = out[name] / max(m.sum(), 1)
    out["parts_sum_check"] = out["incoming_residual_x_mid"] + out["all_neurons"]
    # post hoc (added after the smoke run): the size of each part's bin-mean signal in Q relative to the residual's,
    # whatever its sign or alignment
    rel = lambda Z: float(np.sqrt((bin_means(Z, pb) ** 2).sum() / den))
    out["relative_norm_post_hoc"] = {"all_neurons": rel(act @ CQ), **{
        name: rel(act[:, lab == gi] @ CQ[lab == gi]) for gi, name in GROUP_NAMES.items()}}
    inv, place = np.flatnonzero(lab == 2), np.flatnonzero(lab >= 1)
    null = np.array([share(act[:, s] @ CQ[s]) for s in (rng.choice(place, len(inv), replace=False) for _ in range(1000))])
    out["random_place_sets_same_size"] = {"median": float(np.median(null)), "ci95": ci(null),
                                          "invariant_percentile": float((null < out["invariant"]).mean())}
    return out


def causal_summary(blk):
    """Means, SEMs and the paired differences of the registered causal rule."""
    C = blk["conditions"]
    tr = lambda k: np.array(C[f"interchange: {k}"]["transfer"])
    out = {"interchange_transfer": {}, "mean_replacement_moved_m_median": {}, "mean_replacement_error_m_median": {}}
    for k in ("invariant", "noninvariant_place", "random_place", "all_neurons"):
        t = tr(k)
        out["interchange_transfer"][k] = {"mean": float(t.mean()), "sem": float(t.std() / np.sqrt(len(t))),
                                          "n_neurons": len(blk["sets"][k])}
        out["mean_replacement_moved_m_median"][k] = C[f"mean replacement: {k}"]["summary"]["moved_m_median"]
        out["mean_replacement_error_m_median"][k] = C[f"mean replacement: {k}"]["summary"]["error_m_median"]
    p = C["position PCA 10, written"]["summary"]
    out["interchange_transfer"]["position_pca10_reference"] = {"mean": p["transfer_mean"], "sem": p["transfer_sem"]}
    out["position_pca10_erased_error_m_median"] = C["position PCA 10, erased"]["summary"]["error_m_median"]
    out["clean_error_m_median"] = None
    diffs = {}
    for k in ("noninvariant_place", "random_place"):
        dd = tr("invariant") - tr(k)
        m, s = float(dd.mean()), float(dd.std() / np.sqrt(len(dd)))
        diffs[k] = {"mean": m, "ci95": [m - 1.96 * s, m + 1.96 * s]}
    out["invariant_minus"] = diffs
    inv = out["interchange_transfer"]["invariant"]["mean"]
    carries = inv >= 0.1 and all(v["ci95"][0] > 0 for v in diffs.values())
    out["verdict"] = "carries the causal code" if carries else "no measurable causal role at this token"
    out["all_neuron_ceiling_below_0.1"] = bool(out["interchange_transfer"]["all_neurons"]["mean"] < 0.1)
    return out


def run_analyze(args):
    d = f"{OUT}/smoke" if args.smoke else OUT
    U = np.load(f"{OUT}/units.npz")
    UJ = json.load(open(f"{OUT}/units.json"))
    Rc = np.load(f"{d}/record.npz")
    RJ = json.load(open(f"{d}/record.json"))
    checks = {"units_reproduce_summary_quadrant": all(c["within_0.001"] for c in UJ["conditions"].values()),
              **{f"block{g}_subspace_reproduces_4070": RJ["blocks"][str(g)]["reproduces_4070"]["ok"] for g in BLOCKS},
              **{f"block{g}_write_check": RJ["blocks"][str(g)]["write_check_ok"] for g in BLOCKS}}
    print("checks:", json.dumps(checks), flush=True)
    if not all(checks.values()) and not args.smoke:
        sys.exit("a reproduction check failed: no fraction is reported (registration)")
    rng = np.random.default_rng(0)

    # the isotropic null: fraction of a random direction inside each 10-d subspace (expected exactly K / D)
    def null_of(Q):
        f = np.concatenate([frac_in(Q, rng.standard_normal((Q.shape[0], 10000))) for _ in range(10)])
        return {"expected": R_NULL, "median": float(np.median(f)), "mean": float(f.mean()), "p95": float(np.percentile(f, 95))}

    Q = {f"{s}{g}": Rc[f"Q_{s}_{g}"].astype(np.float64) for g in BLOCKS for s in ("post", "mid")}
    V = {g: Rc[f"gamma2_{g}"][:, None].astype(np.float64) * Rc[f"W2_{g}"].astype(np.float64) for g in BLOCKS}
    W2 = {g: Rc[f"W2_{g}"].astype(np.float64) for g in BLOCKS}
    Rin = {}
    for g in BLOCKS:
        r = Rc[f"ln2_w_{g}"][None, :].astype(np.float64) * Rc[f"W1_{g}"].astype(np.float64)   # (4096, C)
        Rin[g] = (r - r.mean(1, keepdims=True)).T                                              # centred, (C, 4096)
    nulls = {k: null_of(q) for k, q in Q.items()}
    lab = {k: U[f"{k}_group"] for k in ("vggt-single_L19", "vggt-single_L21", "vggt-walk_L19")}

    res = {"about": "Experiment 23 (docs/experiments.md). Fractions are of squared norm inside position PCA 10; "
                    "groups are per neuron (strictly invariant / non-invariant place / non-place); intervals are "
                    "95% bootstrap (10,000 resamples of neurons within each group, seed 0).",
           "checks": checks, "units": UJ["conditions"],
           "record": {k: v for k, v in RJ.items() if k != "blocks"},
           "record_blocks": {g: {k: v for k, v in b.items() if k not in ("conditions", "sets")} for g, b in RJ["blocks"].items()},
           "nulls": nulls}
    m = {}
    m["primary_single_block19_write"] = compare(frac_in(Q["post19"], V[19]), lab["vggt-single_L19"], rng, nulls["post19"])
    m["a_single_block21_write"] = compare(frac_in(Q["post21"], V[21]), lab["vggt-single_L21"], rng, nulls["post21"])
    m["b_walk_block19_write"] = compare(frac_in(Q["post19"], V[19]), lab["vggt-walk_L19"], rng, nulls["post19"])
    m["c_single_block19_read"] = compare(frac_in(Q["mid19"], Rin[19]), lab["vggt-single_L19"], rng, nulls["mid19"])
    m["c_single_block21_read"] = compare(frac_in(Q["mid21"], Rin[21]), lab["vggt-single_L21"], rng, nulls["mid21"])
    m["d_single_block19_write_raw_fc2"] = compare(frac_in(Q["post19"], W2[19]), lab["vggt-single_L19"], rng, nulls["post19"])
    m["e_single_block19_write_vs_block21_subspace"] = compare(frac_in(Q["post21"], V[19]), lab["vggt-single_L19"], rng,
                                                              nulls["post21"])
    res["measures"] = m
    res["subspace_overlap"] = {
        "post19_post21": float(((Q["post19"].T @ Q["post21"]) ** 2).sum() / K),
        "mid19_post19": float(((Q["mid19"].T @ Q["post19"]) ** 2).sum() / K),
        "mid21_post21": float(((Q["mid21"].T @ Q["post21"]) ** 2).sum() / K)}
    res["f_magnitude"] = {"block19": magnitude(Rc, U, 19, "vggt-single_L19", rng),
                          "block21": magnitude(Rc, U, 21, "vggt-single_L21", rng)}

    # post hoc diagnostics (added after the smoke run showed that the raw fc2 columns load at chance while the
    # LayerScale-weighted write vectors do not): how unevenly LayerScale weights the residual's coordinates, how
    # much of each subspace sits on its largest coordinates, and a LayerScale-shaped null (gamma * a random
    # direction: what a neuron with a random fc2 column would write). Descriptive; not part of the rule.
    def pr(e):
        e = np.asarray(e, np.float64)
        return float(e.sum() ** 2 / (e ** 2).sum())
    diag = {}
    for g in BLOCKS:
        gam = Rc[f"gamma2_{g}"].astype(np.float64)
        e = (Q[f"post{g}"] ** 2).sum(1)                      # each coordinate's share of the subspace (sums to K)
        order = np.argsort(-np.abs(gam))
        fg = np.concatenate([frac_in(Q[f"post{g}"], gam[:, None] * rng.standard_normal((D, 10000))) for _ in range(10)])
        diag[f"block{g}"] = {
            "gamma2_abs_median": float(np.median(np.abs(gam))), "gamma2_abs_max": float(np.abs(gam).max()),
            "gamma2_sq_participation_ratio": pr(gam ** 2),
            "subspace_coordinate_participation_ratio": pr(e),
            "subspace_share_on_top10_gamma_coords": float(e[order[:10]].sum() / K),
            "subspace_share_on_top51_gamma_coords": float(e[order[:51]].sum() / K),
            "corr_gamma2_sq_vs_subspace_coord_share": float(np.corrcoef(gam ** 2, e)[0, 1]),
            "gamma2_negative_share": float(np.mean(gam < 0)),
            "layerscale_null": {"median": float(np.median(fg)), "mean": float(fg.mean()), "p95": float(np.percentile(fg, 95))}}
        # the block's shared write direction (mean write vector) and its main output directions (left singular
        # vectors of the write matrix), and the group medians once the shared direction is removed
        Vg, lab_g = V[g], lab[f"vggt-single_L{g}"]
        mdir = Vg.mean(1) / np.linalg.norm(Vg.mean(1))
        Vp = Vg - np.outer(mdir, mdir @ Vg)
        fp = frac_in(Q[f"post{g}"], Vp)
        Uv = np.linalg.svd(Vg, full_matrices=False)[0]
        diag[f"block{g}"].update({
            "mean_write_direction_frac_in_subspace": float(frac_in(Q[f"post{g}"], mdir[:, None])[0]),
            "medians_without_mean_write_direction": {name: float(np.median(fp[lab_g == gi])) for gi, name in GROUP_NAMES.items()},
            "top10_output_directions_frac_in_subspace": frac_in(Q[f"post{g}"], Uv[:, :10]).tolist()})
    res["diagnostics_post_hoc"] = diag
    res["causal"] = {f"block{g}": causal_summary(RJ["blocks"][str(g)]) for g in BLOCKS}
    for g in BLOCKS:
        res["causal"][f"block{g}"]["clean_error_m_median"] = RJ["clean_error_m"]

    def short(c):
        G = c["groups"]
        return {"invariant_median": G["invariant"]["median"], "invariant_median_ci95": G["invariant"]["median_ci95"],
                "noninvariant_place_median": G["noninvariant_place"]["median"],
                "noninvariant_place_median_ci95": G["noninvariant_place"]["median_ci95"],
                "nonplace_median": G["nonplace"]["median"], "nonplace_median_ci95": G["nonplace"]["median_ci95"],
                "invariant_minus_noninvariant_place_ci95": c["invariant_minus_noninvariant_place"]["ci95"],
                "invariant_minus_nonplace_ci95": c["invariant_minus_nonplace"]["ci95"],
                "invariant_over_random": c["invariant_over_random"], "verdict": c["verdict"]}

    p = m["primary_single_block19_write"]
    cz = res["causal"]
    res["summary"] = {
        "experiment": 23,
        "verdict": p["verdict"], "verdict_conditions": p["conditions"],
        "random_expected_frac": R_NULL, "random_empirical_median_frac": nulls["post19"]["median"],
        **short(p),
        "n_neurons": {k: v["n"] for k, v in p["groups"].items()},
        "block21": short(m["a_single_block21_write"]),
        "walk_block19": short(m["b_walk_block19_write"]),
        "input_side_block19": short(m["c_single_block19_read"]),
        "input_side_block21": short(m["c_single_block21_read"]),
        "raw_fc2_block19": short(m["d_single_block19_write_raw_fc2"]),
        "block19_writes_in_block21_subspace": short(m["e_single_block19_write_vs_block21_subspace"]),
        "subspace_overlap_block19_block21": res["subspace_overlap"]["post19_post21"],
        "magnitude_share_block19": {k: res["f_magnitude"]["block19"][k] for k in
                                    ("invariant", "noninvariant_place", "nonplace", "all_neurons", "incoming_residual_x_mid")},
        "magnitude_share_block21": {k: res["f_magnitude"]["block21"][k] for k in
                                    ("invariant", "noninvariant_place", "nonplace", "all_neurons", "incoming_residual_x_mid")},
        "magnitude_relative_size_block19_post_hoc": res["f_magnitude"]["block19"]["relative_norm_post_hoc"],
        "magnitude_relative_size_block21_post_hoc": res["f_magnitude"]["block21"]["relative_norm_post_hoc"],
        "medians_without_mean_write_direction_block19_post_hoc": diag["block19"]["medians_without_mean_write_direction"],
        "layerscale_null_median_block19_post_hoc": diag["block19"]["layerscale_null"]["median"],
        "causal_block19": {"interchange_transfer_invariant": cz["block19"]["interchange_transfer"]["invariant"]["mean"],
                           "interchange_transfer_noninvariant_place": cz["block19"]["interchange_transfer"]["noninvariant_place"]["mean"],
                           "interchange_transfer_random_place": cz["block19"]["interchange_transfer"]["random_place"]["mean"],
                           "interchange_transfer_all_neurons": cz["block19"]["interchange_transfer"]["all_neurons"]["mean"],
                           "interchange_transfer_position_pca10": cz["block19"]["interchange_transfer"]["position_pca10_reference"]["mean"],
                           "mean_replacement_moved_m_invariant": cz["block19"]["mean_replacement_moved_m_median"]["invariant"],
                           "mean_replacement_moved_m_all_neurons": cz["block19"]["mean_replacement_moved_m_median"]["all_neurons"],
                           "verdict": cz["block19"]["verdict"]},
        "causal_block21": {"interchange_transfer_invariant": cz["block21"]["interchange_transfer"]["invariant"]["mean"],
                           "interchange_transfer_noninvariant_place": cz["block21"]["interchange_transfer"]["noninvariant_place"]["mean"],
                           "interchange_transfer_random_place": cz["block21"]["interchange_transfer"]["random_place"]["mean"],
                           "interchange_transfer_all_neurons": cz["block21"]["interchange_transfer"]["all_neurons"]["mean"],
                           "interchange_transfer_position_pca10": cz["block21"]["interchange_transfer"]["position_pca10_reference"]["mean"],
                           "mean_replacement_moved_m_invariant": cz["block21"]["mean_replacement_moved_m_median"]["invariant"],
                           "mean_replacement_moved_m_all_neurons": cz["block21"]["mean_replacement_moved_m_median"]["all_neurons"],
                           "verdict": cz["block21"]["verdict"]},
    }
    path = args.out if not args.smoke else f"{d}/vggt_units_subspace_smoke.json"
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1), flush=True)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--summary", default="results/summary_quadrant.json")
    r = sub.add_parser("record")
    r.add_argument("--vggt", default=os.environ.get("VGGT_DIR", "/workspace/vggt"))
    r.add_argument("--batch", type=int, default=2)
    r.add_argument("--n-fit", type=int, default=2048)
    r.add_argument("--n-test", type=int, default=128)
    r.add_argument("--smoke", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--out", default="results/vggt_units_subspace.json")
    a.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    {"score": score, "record": record, "analyze": run_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
