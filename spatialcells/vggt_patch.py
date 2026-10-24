"""Is VGGT's position code at the walk frame's camera token used? Interchange interventions.

    python -m spatialcells.vggt_patch [--n-fit 2048] [--n-test 128]

Single mode (8 anchors + 1 walk frame, room A). For a target frame and a donor frame
more than 3 m away, part of the donor's residual stream at the walk frame's camera token,
right after global block g, is written into the target's run. The camera head's position
is read in room coordinates (similarity fit on the anchors, as in vggt_cells). Transfer is
how far the prediction moves toward the donor's predicted position, as a fraction of the
distance between the two (0 = no effect, 1 = the target is placed where the donor was).
What is written:
  full             the whole 1024-d token
  position 2       a 2-d subspace read by a ridge probe to (x, y), fitted on other frames
  position 10      10-d: probes to x, y and Fourier features of position (10 and 5 m)
  heading 2        2-d: a probe to (cos, sin) of heading
  random 10        a random 10-d subspace (control)
  all but position everything except the position-10 subspace
Probe directions can sit in low-variance corners of the token, so the subspaces are also
taken from where the token actually varies with pose: PCA of the mean token per 1 m
position bin ("position PCA k"), per 30 deg heading bin ("heading PCA 6"), and per pose bin
(2 m x 90 deg, "pose PCA 30"). "share" is the median fraction of the target-donor
difference (squared norm) that lies in each subspace.
Erasure conditions set the target's position-10 (or random-10) component to its mean over
the fit frames and report the position error instead of transfer. "head input" patches
the full token after both block 23's frame and global attention, which is what the camera
head reads (the endpoint). Writes results/vggt_patch.json.

--probes asks whether the probe's 2-d subspace failed because of how the probe was chosen.
At each block it writes (and erases) 2-d subspaces for (x, y): the ridge probe at each
lambda in LAMS, the activation pattern Cov(H) W of the best-R^2 probe (Haufe et al. 2014),
the cross-covariance direction H^T Y (mass-mean-like), and position PCA 2 / 10 for
reference. "readout_r2" is the held-out R^2 of a least-squares readout of (x, y) from each
subspace. Writes the JSON and a figure (same name, .png); --plot redraws the figure only.
    python -m spatialcells.vggt_patch --probes --blocks 15 19 21 --out results/vggt_patch_probes.json

--controls runs experiment 13c's controls (same targets, donors and fit frames). At each block:
  all but position PCA 10      the exact orthogonal complement of position PCA 10, written
  erase position PCA 10 / random 10   the subspace set to its fit-frame mean (random 10 is the
                               standard run's "random 10" at that block); position error, no transfer
  full / position PCA 10       written and scored as usual; ", fixed alignment" scores the same runs
                               with each clean target run's similarity transform instead of one
                               refitted to the patched anchors, with the median movement of the
                               anchor cameras (both mapped by the clean transform)
  ..., heading-matched donors  donors drawn as usual but also in the target's heading quadrant
                               (true heading, 90 deg bins from 0), transfer toward those donors
Every condition also reports the median change in the walk frame's predicted heading.
    python -m spatialcells.vggt_patch --controls --blocks 19 21 --out results/vggt_patch_controls.json

--crossroom R runs experiment 14d: the standard setup in the room in --data (named by --room; its anchors
at the same 8 poses), with targets and donors drawn there as usual. At each block it writes the whole
token and position PCA 10, position PCA 2 (its top two directions) and the lambda = 1e4 ridge probe for
(x, y), each fit on this room's fit frames and on room R's (the standard run's fit frames there, with
room R's anchors). "readout_r2" is a least-squares readout of (x, y) from each subspace in each room's
fit frames (held-out split), "overlap" the mean squared cosine of the principal angles between the two
rooms' subspaces of a kind, and "other_minus_own" the paired difference in transfer (room R's fit minus
this room's; the pre-registered test is whether it lies within 0.1).
    python -m spatialcells.vggt_patch --crossroom A --data data/walks-roomB --room B --blocks 19 21 \
        --out results/vggt_patch_crossroom.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .vggt_cells import anchors
from .vggt_control import RES, umeyama

BLOCKS = [2, 5, 9, 12, 15, 19, 21, 23]
LAMS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


def ridge_name(lam):
    return f"ridge 1e{round(np.log10(lam))}"


def features(xy, theta):
    x, y = xy[:, 0], xy[:, 1]
    pos = [x, y]
    for lam in (10.0, 5.0):
        w = 2 * np.pi / lam
        pos += [np.cos(w * x), np.sin(w * x), np.cos(w * y), np.sin(w * y)]
    return np.stack(pos[:2], 1), np.stack(pos, 1), np.stack([np.cos(theta), np.sin(theta)], 1)


def mean_subspace(H, labels, k):
    """Top-k principal directions (C, k) of the mean token per label (labels with >= 5 frames)."""
    labels = torch.as_tensor(labels, device=H.device)
    means = torch.stack([H[labels == u].mean(0) for u in labels.unique() if (labels == u).sum() >= 5])
    _, _, Vt = torch.linalg.svd(means - means.mean(0), full_matrices=False)
    return Vt[:k].T


def split(n, rng):
    idx = rng.permutation(n)
    return idx[: int(0.8 * n)], idx[int(0.8 * n):]


def r2(pred, Y):
    return (1 - ((pred - Y) ** 2).sum() / (Y ** 2).sum()).item()


def ridge_probes(H, Y, tr, va, lams):
    """Ridge probes H -> Y fitted on rows tr, one per lambda: [(lambda, held-out R^2 on rows va, W)].
    H and Y are centred on the tr rows."""
    G, C = H[tr].T @ H[tr], H[tr].T @ Y[tr]
    eye = torch.eye(H.shape[1], dtype=H.dtype, device=H.device)
    out = []
    for lam in lams:
        W = torch.linalg.solve(G + lam * eye, C)
        out.append((lam, r2(H[va] @ W, Y[va]), W))
    return out


def readout_r2(H, Y, tr, va, Q):
    """Held-out R^2 of a least-squares readout of Y from H's coordinates in the subspace Q."""
    Z = H @ Q
    B = torch.linalg.solve(Z[tr].T @ Z[tr], Z[tr].T @ Y[tr])
    return r2(Z[va] @ B, Y[va])


def ridge_subspace(H, Y, rng):
    """Orthonormal basis (C, k) of the directions a ridge probe H -> Y reads, plus held-out R^2."""
    tr, va = split(len(H), rng)
    Hc, Yc = H - H[tr].mean(0), Y - Y[tr].mean(0)
    _, r2_best, W = max(ridge_probes(Hc, Yc, tr, va, LAMS[:6]), key=lambda p: p[1])
    Q, _ = torch.linalg.qr(W)
    return Q, r2_best


def to_room(cams, T):
    """Camera centres (S, 2) and headings (S,) in room coordinates, from centres and optical axes cams = (C, fwd), each
    (S, 3) in VGGT's frame, under the similarity transform T = (s, R, t) (as in vggt_cells)."""
    (C, fwd), (s, R, t) = cams, T
    Cr, fr = (s * (R @ C.T)).T + t, (R @ fwd.T).T
    return Cr[:, :2], np.arctan2(fr[:, 1], fr[:, 0])


def angdiff(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def plot_probes(out):
    """Transfer against R^2 for the 2-d (x, y) subspaces, one panel per block. R^2 is that of a least-squares
    readout from each subspace, so heavily regularised probes are not penalised for their shrinkage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = json.load(open(out))
    blocks = list(R["blocks"])
    fig, axs = plt.subplots(1, len(blocks), figsize=(4.4 * len(blocks), 4.4), sharey=True, squeeze=False)
    note = dict(textcoords="offset points", fontsize=8, color="0.3")
    for ax, g in zip(axs[0], blocks):
        b = R["blocks"][g]
        pt = lambda k: (b["readout_r2"][k], b[k]["transfer_mean"], b[k]["transfer_sem"])
        lams = list(b["probe_r2"])
        ax.errorbar(*zip(*map(pt, lams)), color="#2a78d6", lw=1.5, marker="o", ms=5, capsize=2,
                    label="ridge probe, λ = 0.1 … 10⁶")
        ax.annotate(f"best-R² probe (λ = {b['best_lambda']:g})", pt(ridge_name(b["best_lambda"]))[:2],
                    xytext=(-10, -5), ha="right", va="top", **note)
        ax.annotate("λ = 10⁶", pt(lams[-1])[:2], xytext=(6, 12), ha="left", **note)
        for k, kw, label in (("pattern", dict(color="#eb6834", marker="^", ms=11, mfc="none", mew=1.6), "Haufe pattern Cov(H) W"),
                             ("cross-covariance", dict(color="#eb6834", marker="D", ms=5), "cross-covariance Hᵀ Y"),
                             ("position PCA 2", dict(color="0.6", marker="s", ms=7), "position PCA, 2 dims"),
                             ("position PCA 10", dict(color="0.2", marker="s", ms=7), "position PCA, 10 dims (reference)")):
            ax.errorbar(*pt(k), capsize=2, ls="none", label=label, zorder=3, **kw)
        ax.set_title(f"block {g}", fontsize=10)
        ax.set_xlabel("held-out R² of (x, y) read from the written directions", fontsize=8.5)
        ax.set_xlim(0.2, 1.02)
        ax.grid(color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0, 0].set_ylim(-0.09, 0.95)
    axs[0, 0].set_ylabel("transfer toward the donor's position\n(mean ± SEM, 128 targets)", fontsize=8.5)
    axs[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("VGGT: writing two (x, y) directions from a donor frame into the target's camera token", fontsize=10)
    fig.tight_layout()
    fig.savefig(out.replace(".json", ".png"), dpi=150)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/walks")
    ap.add_argument("--vggt", default="/workspace/vggt")
    ap.add_argument("--n-fit", type=int, default=2048)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--out", help="default results/vggt_patch.json (--controls: results/vggt_patch_controls.json, "
                    "--crossroom: results/vggt_patch_crossroom.json)")
    ap.add_argument("--room", default="A", help="the room in --data (its anchors are rendered there if missing)")
    ap.add_argument("--blocks", type=int, nargs="+", default=BLOCKS)
    ap.add_argument("--probes", action="store_true", help="probe-choice sweep instead of the standard subspaces")
    ap.add_argument("--controls", action="store_true", help="experiment 13c's controls instead of the standard subspaces")
    ap.add_argument("--crossroom", metavar="R", help="experiment 14d: also fit the subspaces in room R (A, B, ...)")
    ap.add_argument("--plot", action="store_true", help="only redraw the --probes figure from --out")
    args = ap.parse_args()
    args.out = args.out or ("results/vggt_patch_controls.json" if args.controls else
                            "results/vggt_patch_crossroom.json" if args.crossroom else "results/vggt_patch.json")
    if args.plot:
        return plot_probes(args.out)
    sys.path.insert(0, args.vggt)
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    rng = np.random.default_rng(0)
    frames = np.load(f"{args.data}/frames.npy", mmap_mode="r")
    meta = dict(np.load(f"{args.data}/meta.npz"))
    xy_true = np.stack([meta["x"], meta["y"]], 1)
    A = anchors(args.data, args.room)
    model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
    agg = model.aggregator
    P = agg.patch_start_idx + (RES // 14) ** 2
    n_anchor = len(A["images"])
    S = n_anchor + 1
    cam = n_anchor * P                      # walk frame's camera token in the global blocks' (B, S*P, C) layout

    record, patch = {}, {}                  # block -> recorded (B, C) / replacement (B, C); "f23" = frame block 23
    recording = [True]

    def ghook(g):
        def f(_, __, out):
            if recording[0]:
                record[g] = out[:, cam].float().cpu()
            if g in patch:
                out = out.clone()
                out[:, cam] = patch[g].to(out.device, out.dtype)
                return out
        return f
    blocks = sorted(set(args.blocks) | {23})
    for g in blocks:
        agg.global_blocks[g].register_forward_hook(ghook(g))

    def fhook(_, __, out):                  # frame block 23: (B*S, P, C)
        if "f23" in patch:
            out = out.clone()
            out[n_anchor::S, 0] = patch["f23"].to(out.device, out.dtype)
            return out
        if recording[0]:
            record["f23"] = out[n_anchor::S, 0].float().cpu()
    agg.frame_blocks[23].register_forward_hook(fhook)

    def prep(ims):
        x = torch.from_numpy(np.stack([np.asarray(a) for a in ims])).cuda().permute(0, 3, 1, 2).float() / 255
        return F.interpolate(x, size=(RES, RES), mode="bicubic", align_corners=False).clamp(0, 1)

    anchor_px = prep(A["images"])
    gt_anchor = np.c_[A["poses"][:, :2], np.ones(n_anchor)]

    def align(cams):
        """Similarity transform of a run's anchor camera centres onto the anchors' true positions."""
        return umeyama(cams[0][:n_anchor], gt_anchor)

    def walk_pose(cams, T=None):
        """The walk frame's (x, y) (n, 2) and heading (n,) in room coordinates, each run aligned by its own anchor
        fit, or by the given transforms T."""
        T = [align(c) for c in cams] if T is None else T
        xy, th = zip(*(to_room(c, t) for c, t in zip(cams, T)))
        return np.stack(xy)[:, n_anchor], np.stack(th)[:, n_anchor]

    def predict(idx, src=None):
        """The walk frame's (x, y) in room coordinates, and every camera's centre and optical axis in VGGT's frame
        (n, 2, S, 3). src: (frames, anchor pixels) of another room instead of --data's."""
        fr, apx = src or (frames, anchor_px)
        seqs = torch.stack([torch.cat([apx, prep(fr[[i]])]) for i in idx])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, _ = agg(seqs)
            pose = model.camera_head(tokens)[-1]
        ext, _ = pose_encoding_to_extri_intri(pose.float(), (RES, RES))
        ext = ext.cpu().numpy().astype(np.float64)
        R, t = ext[..., :3], ext[..., 3]
        cams = np.stack([-np.einsum("bsji,bsj->bsi", R, t), np.einsum("bsji,j->bsi", R, np.array([0.0, 0.0, 1.0]))], 1)
        return walk_pose(cams)[0], cams

    def run_clean(idx, src=None):
        global_record = {}
        preds, cams = [], []
        for b0 in range(0, len(idx), args.batch):
            record.clear()
            p, c = predict(idx[b0:b0 + args.batch], src)
            preds.append(p)
            cams.append(c)
            for k, v in record.items():
                global_record.setdefault(k, []).append(v)
        return np.concatenate(preds), {k: torch.cat(v) for k, v in global_record.items()}, np.concatenate(cams)

    def run_patched(idx, repl, cams=False):
        """repl: block -> (n, C) replacement tokens for each target in idx. cams: return the cameras (see predict)
        instead of the walk frame's position."""
        recording[0] = False
        out = []
        for b0 in range(0, len(idx), args.batch):
            patch.clear()
            patch.update({k: v[b0:b0 + args.batch] for k, v in repl.items()})
            out.append(predict(idx[b0:b0 + args.batch])[1 if cams else 0])
        patch.clear()
        recording[0] = True
        return np.concatenate(out)

    t0 = time.time()
    even = np.flatnonzero(meta["walk"] % 2 == 0)
    odd = np.flatnonzero(meta["walk"] % 2 == 1)
    fit = np.sort(rng.choice(even, args.n_fit, replace=False))
    targets = rng.choice(odd, args.n_test, replace=False)
    donors = np.array([rng.choice(odd[np.hypot(*(xy_true[odd] - xy_true[t]).T) > 3]) for t in targets])
    if args.controls:                       # heading-matched donors: also in the target's heading quadrant
        quad = (meta["theta"] % (2 * np.pi) / (np.pi / 2)).astype(int) % 4
        donors_h = np.array([rng.choice(odd[(np.hypot(*(xy_true[odd] - xy_true[t]).T) > 3) & (quad[odd] == quad[t])])
                             for t in targets])

    _, H_fit, _ = run_clean(fit)
    pred_t, H_t, cams_t = run_clean(targets)
    pred_d, H_d, cams_d = run_clean(donors)
    if args.controls:
        pred_dh, H_dh, cams_dh = run_clean(donors_h)
    if args.crossroom:                      # room R's fit frames: its standard run's (rng(0)'s first draw), its anchors
        data_o = "data/walks" if args.crossroom == "A" else f"data/walks-room{args.crossroom}"
        meta_o = dict(np.load(f"{data_o}/meta.npz"))
        A_o = anchors(data_o, args.crossroom)
        assert np.allclose(A_o["poses"], A["poses"]), "the two rooms' anchors must be at the same poses"
        fit_o = np.sort(np.random.default_rng(0).choice(np.flatnonzero(meta_o["walk"] % 2 == 0), args.n_fit, replace=False))
        xy_fit_o = np.stack([meta_o["x"], meta_o["y"]], 1)[fit_o]
        pred_fo, H_fit_o, _ = run_clean(fit_o, (np.load(f"{data_o}/frames.npy", mmap_mode="r"), prep(A_o["images"])))
    print(f"clean passes done, {time.time() - t0:.0f}s; clean target error median "
          f"{np.median(np.hypot(*(pred_t - xy_true[targets]).T)):.2f} m", flush=True)

    f2, f10, fh = features(xy_true[fit], meta["theta"][fit])
    t2, t10, th = features(xy_true[targets], meta["theta"][targets])
    gap = pred_d - pred_t
    dist2 = (gap ** 2).sum(1)
    results = {"n_test": int(args.n_test), "clean_error_m": float(np.median(np.hypot(*(pred_t - xy_true[targets]).T))),
               "target_donor_distance_m": float(np.median(np.sqrt(dist2))), "blocks": {}}
    T_t = [align(c) for c in cams_t]        # the clean target runs' similarity transforms
    head_t = walk_pose(cams_t)[1]
    deg = lambda a, b: float(np.degrees(np.median(angdiff(a, b))))
    if args.controls:
        results.update(clean_heading_error_deg=deg(head_t, meta["theta"][targets]),
                       target_donor_heading_deg=deg(walk_pose(cams_d)[1], head_t),
                       heading_matched_target_donor_distance_m=float(np.median(np.hypot(*(pred_dh - pred_t).T))),
                       heading_matched_target_donor_heading_deg=deg(walk_pose(cams_dh)[1], head_t),
                       heading_matched_target_donor_true_heading_deg=deg(meta["theta"][donors_h], meta["theta"][targets]))
        print("controls: " + ", ".join(f"{k} {v:.2f}" for k, v in results.items() if k != "blocks"), flush=True)

    def summarize(pred, head=None, erase=False, pred_d=pred_d):
        """Position moved and error; transfer toward pred_d unless erase; heading change if head is given."""
        moved = pred - pred_t
        r = {"moved_m_median": float(np.median(np.linalg.norm(moved, axis=1))),
             "error_m_median": float(np.median(np.hypot(*(pred - xy_true[targets]).T)))}
        if not erase:
            gap = pred_d - pred_t
            tr = (moved * gap).sum(1) / (gap ** 2).sum(1)
            r.update(transfer_mean=float(tr.mean()), transfer_sem=float(tr.std() / np.sqrt(len(tr))),
                     transfer_median=float(np.median(tr)))
        if head is not None:
            r["heading_change_deg_median"] = deg(head, head_t)
        return r

    def anchor_moved(cams):
        """Median movement (m) of the anchor cameras from the clean target runs, both mapped by the clean transform."""
        return float(np.median([np.linalg.norm(to_room(c, T)[0][:n_anchor] - to_room(c0, T)[0][:n_anchor], axis=1)
                                for c, c0, T in zip(cams, cams_t, T_t)]))

    def proj(Q, v):
        return (v @ Q) @ Q.T

    def share(Q, d):
        return float(((proj(Q, d) ** 2).sum(1) / (d ** 2).sum(1)).median())

    def probe_block(g, Hf, mu, pos_bin, ht, d):
        """--probes: write and erase each 2-d subspace for (x, y) at block g (fits in float64)."""
        tr, va = split(len(Hf), rng)
        H, Y = Hf.double(), torch.tensor(f2, dtype=torch.float64, device="cuda")
        H, Y = H - H[tr].mean(0), Y - Y[tr].mean(0)
        probes = ridge_probes(H, Y, tr, va, LAMS)
        lam_best, _, W_best = max(probes, key=lambda p: p[1])
        G = H[tr].T @ H[tr]
        orth = lambda M: torch.linalg.qr(M)[0]
        pca = mean_subspace(Hf, pos_bin, 10).double()
        sub = {**{ridge_name(lam): orth(W) for lam, _, W in probes},
               "pattern": orth(G @ W_best),           # Haufe et al. 2014: A = Cov(H) W, up to scale
               "cross-covariance": orth(H[tr].T @ Y[tr]),
               "position PCA 2": pca[:, :2], "position PCA 10": pca}
        eig = torch.linalg.eigvalsh(G)
        res = {"best_lambda": lam_best, "probe_r2": {ridge_name(lam): v for lam, v, _ in probes},
               "readout_r2": {k: readout_r2(H, Y, tr, va, Q) for k, Q in sub.items()},
               "gram_eigenvalue_max": eig[-1].item(), "gram_eigenvalue_median": eig.median().item(),
               "token_norm_median": float(ht.norm(dim=1).median())}
        sub = {k: Q.float() for k, Q in sub.items()}
        res["share"] = {k: share(Q, d) for k, Q in sub.items()}
        for name, Q in sub.items():
            res[name] = summarize(run_patched(targets, {g: (ht + proj(Q, d)).cpu()}))
            res[name]["erase_error_m_median"] = summarize(
                run_patched(targets, {g: (ht + proj(Q, mu - ht)).cpu()}), erase=True)["error_m_median"]
        print(f"block {g} ({time.time() - t0:.0f}s): best lambda {lam_best:g}; "
              + "; ".join(f"{k} R2 {res['probe_r2'].get(k, res['readout_r2'][k]):.3f} share {res['share'][k]:.3f} "
                          f"transfer {res[k]['transfer_mean']:.2f} erase {res[k]['erase_error_m_median']:.2f} m" for k in sub), flush=True)
        return res

    def control_block(g, Hf, mu, pos_bin, ht, hd):
        """--controls (experiment 13c) at block g."""
        d, hdh = hd - ht, H_dh[g].cuda()
        dh = hdh - ht
        Q = mean_subspace(Hf, pos_bin, 10)
        gen = torch.Generator("cuda").manual_seed(g)       # the standard run's "random 10" at this block
        Qr, _ = torch.linalg.qr(torch.randn(Hf.shape[1], 10, device="cuda", generator=gen))
        res = {"token_norm_median": float(ht.norm(dim=1).median()),
               "share": {"position PCA 10": share(Q, d), "random 10": share(Qr, d),
                         "all but position PCA 10": float((((d - proj(Q, d)) ** 2).sum(1) / (d ** 2).sum(1)).median()),
                         "position PCA 10, heading-matched donors": share(Q, dh)}}
        patched = lambda h: run_patched(targets, {g: h.cpu()}, cams=True)
        for name, h in (("full", hd), ("position PCA 10", ht + proj(Q, d))):
            cams = patched(h)
            res[name] = summarize(*walk_pose(cams))
            res[f"{name}, fixed alignment"] = {**summarize(*walk_pose(cams, T_t)), "anchor_moved_m_median": anchor_moved(cams)}
        for name, h in (("full", hdh), ("position PCA 10", ht + proj(Q, dh))):
            res[f"{name}, heading-matched donors"] = summarize(*walk_pose(patched(h)), pred_d=pred_dh)
        res["all but position PCA 10"] = summarize(*walk_pose(patched(hd - proj(Q, d))))
        for name, Qe in (("erase position PCA 10", Q), ("erase random 10", Qr)):
            res[name] = summarize(*walk_pose(patched(ht + proj(Qe, mu - ht))), erase=True)
        keys = ("transfer_mean", "error_m_median", "anchor_moved_m_median", "heading_change_deg_median")
        print(f"block {g} ({time.time() - t0:.0f}s): " + "; ".join(
            f"{k}: " + ", ".join(f"{m} {v[m]:.2f}" for m in keys if m in v) for k, v in res.items() if k not in ("token_norm_median", "share"))
            + "; share " + ", ".join(f"{k} {v:.3f}" for k, v in res["share"].items()), flush=True)
        return res

    def crossroom_block(g, ht, hd):
        """--crossroom (experiment 14d) at block g: subspaces fit in this room and in room R, written here."""
        d = hd - ht
        own, other = args.room, args.crossroom
        kinds = ("position PCA 10", "position PCA 2", "ridge 1e4")
        name = lambda k, r: f"{k}, fit in room {r}"
        sub, fits = {}, {}
        res = {"token_norm_median": float(ht.norm(dim=1).median()), "probe_r2": {}}
        for r, Hf, xy in ((own, H_fit[g], xy_true[fit]), (other, H_fit_o[g], xy_fit_o)):
            Hf = Hf.cuda()
            pos_bin = np.clip(xy[:, 0].astype(int), 0, 9) * 10 + np.clip(xy[:, 1].astype(int), 0, 9)
            pca = mean_subspace(Hf, pos_bin, 10)
            tr, va = split(len(Hf), rng)        # ridge as in --probes: fits in float64, centred on the tr rows
            H, Y = Hf.double(), torch.tensor(xy, dtype=torch.float64, device="cuda")
            H, Y = H - H[tr].mean(0), Y - Y[tr].mean(0)
            [(_, r2_ridge, W)] = ridge_probes(H, Y, tr, va, (1e4,))
            res["probe_r2"][name("ridge 1e4", r)] = r2_ridge
            fits[r] = (H, Y, tr, va)
            sub.update({name("position PCA 10", r): pca, name("position PCA 2", r): pca[:, :2],
                        name("ridge 1e4", r): torch.linalg.qr(W)[0].float()})
        res["readout_r2"] = {k: {f"room {r}": readout_r2(*fits[r], Q.double()) for r in fits} for k, Q in sub.items()}
        res["overlap"] = {k: float((sub[name(k, own)].T @ sub[name(k, other)]).pow(2).sum()
                                   / min(sub[name(k, own)].shape[1], sub[name(k, other)].shape[1])) for k in kinds}
        res["share"] = {k: share(Q, d) for k, Q in sub.items()}
        preds = {"full": run_patched(targets, {g: hd.cpu()}),
                 **{k: run_patched(targets, {g: (ht + proj(Q, d)).cpu()}) for k, Q in sub.items()}}
        res.update({k: summarize(p) for k, p in preds.items()})
        tvec = lambda p: ((p - pred_t) * gap).sum(1) / dist2
        res["other_minus_own"] = {}
        for k in kinds:
            dt = tvec(preds[name(k, other)]) - tvec(preds[name(k, own)])
            res["other_minus_own"][k] = {"transfer_mean": float(dt.mean()), "transfer_sem": float(dt.std() / np.sqrt(len(dt)))}
        print(f"block {g} ({time.time() - t0:.0f}s): transfer " + ", ".join(f"{k} {res[k]['transfer_mean']:.2f}" for k in preds)
              + f"; room {other} fit minus room {own} fit " + ", ".join(
                  f"{k} {v['transfer_mean']:+.2f} +/- {v['transfer_sem']:.2f}" for k, v in res["other_minus_own"].items())
              + "; overlap " + ", ".join(f"{k} {v:.2f}" for k, v in res["overlap"].items())
              + "; share " + ", ".join(f"{k} {v:.3f}" for k, v in res["share"].items()), flush=True)
        return res

    if args.crossroom:
        results.update(room=args.room, other_room=args.crossroom,
                       other_room_fit_clean_error_m=float(np.median(np.hypot(*(pred_fo - xy_fit_o).T))))
    for g in args.blocks:
        Hf = H_fit[g].cuda()
        mu = Hf.mean(0)
        x, y, th = xy_true[fit, 0], xy_true[fit, 1], meta["theta"][fit] % (2 * np.pi)
        pos_bin = np.clip(x.astype(int), 0, 9) * 10 + np.clip(y.astype(int), 0, 9)
        ht, hd = H_t[g].cuda(), H_d[g].cuda()
        d = hd - ht
        if args.probes or args.controls or args.crossroom:
            results["blocks"][g] = (probe_block(g, Hf, mu, pos_bin, ht, d) if args.probes
                                    else control_block(g, Hf, mu, pos_bin, ht, hd) if args.controls
                                    else crossroom_block(g, ht, hd))
            json.dump(results, open(args.out, "w"), indent=1)
            continue
        Q2, r2_2 = ridge_subspace(Hf, torch.tensor(f2, dtype=torch.float32, device="cuda"), rng)
        Q10, r2_10 = ridge_subspace(Hf, torch.tensor(f10, dtype=torch.float32, device="cuda"), rng)
        Qh, r2_h = ridge_subspace(Hf, torch.tensor(fh, dtype=torch.float32, device="cuda"), rng)
        gen = torch.Generator("cuda").manual_seed(g)
        Qr, _ = torch.linalg.qr(torch.randn(Hf.shape[1], 10, device="cuda", generator=gen))
        Qr30, _ = torch.linalg.qr(torch.randn(Hf.shape[1], 30, device="cuda", generator=gen))
        head_bin = (th / (np.pi / 6)).astype(int) % 12
        pose_bin = (np.clip((x / 2).astype(int), 0, 4) * 5 + np.clip((y / 2).astype(int), 0, 4)) * 4 + (th / (np.pi / 2)).astype(int) % 4
        sub = {"position 2": Q2, "position 10": Q10, "heading 2": Qh,
               "position PCA 10": mean_subspace(Hf, pos_bin, 10), "position PCA 30": mean_subspace(Hf, pos_bin, 30),
               "heading PCA 6": mean_subspace(Hf, head_bin, 6), "pose PCA 30": mean_subspace(Hf, pose_bin, 30),
               "random 10": Qr, "random 30": Qr30}
        conds = {"full": hd, **{k: ht + proj(Q, d) for k, Q in sub.items()},
                 "all but position": hd - proj(Q10, d), "all but pose PCA 30": hd - proj(sub["pose PCA 30"], d)}
        res = {"probe_r2": {"position 2": r2_2, "position 10": r2_10, "heading 2": r2_h},
               "token_norm_median": float(ht.norm(dim=1).median()),
               "share": {k: share(Q, d) for k, Q in sub.items()}}
        for name, h in conds.items():
            res[name] = summarize(run_patched(targets, {g: h.cpu()}))
        for name, Q in (("erase position 10", Q10), ("erase pose PCA 30", sub["pose PCA 30"]), ("erase random 30", Qr30)):
            res[name] = summarize(run_patched(targets, {g: (ht + proj(Q, mu - ht)).cpu()}), erase=True)
        results["blocks"][g] = res
        print(f"block {g} ({time.time() - t0:.0f}s): probe R2 pos2 {r2_2:.2f} pos10 {r2_10:.2f} heading {r2_h:.2f}; transfer "
              + ", ".join(f"{k} {v['transfer_mean']:.2f}" for k, v in res.items() if isinstance(v, dict) and "transfer_mean" in v)
              + f"; erase pose PCA 30 error {res['erase pose PCA 30']['error_m_median']:.2f} m vs random "
              f"{res['erase random 30']['error_m_median']:.2f} m; share " + ", ".join(f"{k} {v:.3f}" for k, v in res["share"].items()), flush=True)
        json.dump(results, open(args.out, "w"), indent=1)
    if args.probes:
        return plot_probes(args.out)
    if args.controls or args.crossroom:
        print(f"done, {time.time() - t0:.0f}s", flush=True)
        return
    results["head input"] = summarize(run_patched(targets, {23: H_d[23], "f23": H_d["f23"]}))
    print(f"head input transfer {results['head input']['transfer_mean']:.2f}; done, {time.time() - t0:.0f}s", flush=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
