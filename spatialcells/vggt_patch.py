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


def plot_probes(out):
    """Transfer against R^2 for the probe subspaces, one panel per block."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = json.load(open(out))
    blocks = list(R["blocks"])
    fig, axs = plt.subplots(1, len(blocks), figsize=(4.2 * len(blocks), 4.2), sharey=True)
    for ax, g in zip(np.atleast_1d(axs), blocks):
        b = R["blocks"][g]
        tr = lambda k: (b[k]["transfer_mean"], b[k]["transfer_sem"])
        lams = [k for k in b["probe_r2"]]
        xs = [b["probe_r2"][k] for k in lams]
        ys, es = zip(*(tr(k) for k in lams))
        ax.errorbar(xs, ys, es, color="#2a78d6", lw=1.5, marker="o", ms=5, capsize=2, label="ridge probe, lambda 0.1 to 1e6")
        for k in (lams[0], lams[-1], ridge_name(b["best_lambda"])):
            ax.annotate(k.replace("ridge ", "λ=") + (" (best R²)" if k == ridge_name(b["best_lambda"]) else ""),
                        (b["probe_r2"][k], tr(k)[0]), textcoords="offset points", xytext=(-6, 8), ha="right", fontsize=7.5, color="0.25")
        for k, color, marker, label in (("pattern", "#eb6834", "^", "pattern Cov(H)W of best probe"),
                                        ("cross-covariance", "#1baf7a", "D", "cross-covariance HᵀY"),
                                        ("position PCA 2", "0.45", "s", "position PCA, 2 dims"),
                                        ("position PCA 10", "0.15", "s", "position PCA, 10 dims")):
            ax.errorbar(b["readout_r2"][k], *tr(k), color=color, marker=marker, ms=8 if k != "pattern" else 10,
                        mfc=color if k != "pattern" else "none", mew=1.5, capsize=2, ls="none", label=label)
        ax.set_title(f"block {g}", fontsize=10)
        ax.set_xlabel("held-out R² of (x, y)", fontsize=9)
        ax.grid(color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    np.atleast_1d(axs)[0].set_ylabel("transfer toward the donor's position", fontsize=9)
    np.atleast_1d(axs)[-1].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("Writing a 2-d (x, y) subspace from donor into target: R² of the ridge probe itself, "
                 "or of a readout from the subspace (other points)", fontsize=9.5)
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
    ap.add_argument("--out", default="results/vggt_patch.json")
    ap.add_argument("--blocks", type=int, nargs="+", default=BLOCKS)
    ap.add_argument("--probes", action="store_true", help="probe-choice sweep instead of the standard subspaces")
    ap.add_argument("--plot", action="store_true", help="only redraw the --probes figure from --out")
    args = ap.parse_args()
    if args.plot:
        return plot_probes(args.out)
    sys.path.insert(0, args.vggt)
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    rng = np.random.default_rng(0)
    frames = np.load(f"{args.data}/frames.npy", mmap_mode="r")
    meta = dict(np.load(f"{args.data}/meta.npz"))
    xy_true = np.stack([meta["x"], meta["y"]], 1)
    A = anchors(args.data)
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

    def predict(idx):
        seqs = torch.stack([torch.cat([anchor_px, prep(frames[[i]])]) for i in idx])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, _ = agg(seqs)
            pose = model.camera_head(tokens)[-1]
        ext, _ = pose_encoding_to_extri_intri(pose.float(), (RES, RES))
        ext = ext.cpu().numpy().astype(np.float64)
        out = np.zeros((len(idx), 2))
        for b, E in enumerate(ext):
            R, t = E[:, :, :3], E[:, :, 3]
            C = -np.einsum("sji,sj->si", R, t)
            s, Ra, ta = umeyama(C[:n_anchor], gt_anchor)
            out[b] = ((s * (Ra @ C.T)).T + ta)[n_anchor, :2]
        return out

    def run_clean(idx):
        global_record = {}
        preds = []
        for b0 in range(0, len(idx), args.batch):
            record.clear()
            preds.append(predict(idx[b0:b0 + args.batch]))
            for k, v in record.items():
                global_record.setdefault(k, []).append(v)
        return np.concatenate(preds), {k: torch.cat(v) for k, v in global_record.items()}

    def run_patched(idx, repl):
        """repl: block -> (n, C) replacement tokens for each target in idx."""
        recording[0] = False
        preds = []
        for b0 in range(0, len(idx), args.batch):
            patch.clear()
            patch.update({k: v[b0:b0 + args.batch] for k, v in repl.items()})
            preds.append(predict(idx[b0:b0 + args.batch]))
        patch.clear()
        recording[0] = True
        return np.concatenate(preds)

    t0 = time.time()
    even = np.flatnonzero(meta["walk"] % 2 == 0)
    odd = np.flatnonzero(meta["walk"] % 2 == 1)
    fit = np.sort(rng.choice(even, args.n_fit, replace=False))
    targets = rng.choice(odd, args.n_test, replace=False)
    donors = np.array([rng.choice(odd[np.hypot(*(xy_true[odd] - xy_true[t]).T) > 3]) for t in targets])

    _, H_fit = run_clean(fit)
    pred_t, H_t = run_clean(targets)
    pred_d, H_d = run_clean(donors)
    print(f"clean passes done, {time.time() - t0:.0f}s; clean target error median "
          f"{np.median(np.hypot(*(pred_t - xy_true[targets]).T)):.2f} m", flush=True)

    f2, f10, fh = features(xy_true[fit], meta["theta"][fit])
    t2, t10, th = features(xy_true[targets], meta["theta"][targets])
    gap = pred_d - pred_t
    dist2 = (gap ** 2).sum(1)
    results = {"n_test": int(args.n_test), "clean_error_m": float(np.median(np.hypot(*(pred_t - xy_true[targets]).T))),
               "target_donor_distance_m": float(np.median(np.sqrt(dist2))), "blocks": {}}

    def summarize(pred, erase=False):
        moved = pred - pred_t
        r = {"moved_m_median": float(np.median(np.linalg.norm(moved, axis=1))),
             "error_m_median": float(np.median(np.hypot(*(pred - xy_true[targets]).T)))}
        if not erase:
            tr = (moved * gap).sum(1) / dist2
            r.update(transfer_mean=float(tr.mean()), transfer_sem=float(tr.std() / np.sqrt(len(tr))),
                     transfer_median=float(np.median(tr)))
        return r

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

    for g in args.blocks:
        Hf = H_fit[g].cuda()
        mu = Hf.mean(0)
        x, y, th = xy_true[fit, 0], xy_true[fit, 1], meta["theta"][fit] % (2 * np.pi)
        pos_bin = np.clip(x.astype(int), 0, 9) * 10 + np.clip(y.astype(int), 0, 9)
        ht, hd = H_t[g].cuda(), H_d[g].cuda()
        d = hd - ht
        if args.probes:
            results["blocks"][g] = probe_block(g, Hf, mu, pos_bin, ht, d)
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
    results["head input"] = summarize(run_patched(targets, {23: H_d[23], "f23": H_d["f23"]}))
    print(f"head input transfer {results['head input']['transfer_mean']:.2f}; done, {time.time() - t0:.0f}s", flush=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
