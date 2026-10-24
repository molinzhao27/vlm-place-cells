"""Positive control: can a 3D vision model place the new views from these same frames?

    python -m spatialcells.vggt_control --walks 16 --stride 4

VGGT (Wang et al., CVPR 2025; weights facebook/VGGT-1B, non-commercial licence) predicts a
camera pose for every image it is given. Each trial gets every 4th frame of the walk, the
two candidate frames of the two-choice question, and the probe: one probe at a time, as
the VLMs were asked. Predicted camera centres are put into room coordinates by a
similarity transform fitted on the walk frames only (the probe never enters the fit).
Same trials as behave_reason (its direct two-choice records): all novel ones and every
4th familiar one. Scores:
  probe localisation error (m) and heading error (deg)
  nearest frame  the subsampled walk frame whose predicted centre is nearest the
                 probe's; a hit if it is truly within 1 m
  two-choice     whichever of near / far has its predicted centre nearer the probe's
Needs the VGGT repo on the path (cloned to /workspace/vggt; not pip-installed, because
its requirements pin an old torch). Writes results/vggt_control.json and .npz.
--trial-set R instead runs every trial of results/export/dissimilar-room<R>/trials.json (lure-free
trials of room R, novel and familiar) on data/walks-room<R>, and writes results/vggt_control_<R>.json
and .npz (experiment 12: R = ReplicaApt0 ...).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .analyze_behave import NEAR, angdiff, geometry

RES = 518   # VGGT's training width; 37 x 14 px patches


def umeyama(src, dst):
    """Similarity (s, R, t) with dst ~ s R src + t, least squares."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((xs ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4, help="give VGGT every n-th walk frame")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--familiar-every", type=int, default=4, help="keep every n-th familiar trial (all novel ones)")
    ap.add_argument("--trials", default="results/behave/qwen38-reason.npz")
    ap.add_argument("--vggt", default="/workspace/vggt")
    ap.add_argument("--trial-set", help="room R: its lure-free trials and walks (see above)")
    args = ap.parse_args()
    sys.path.insert(0, args.vggt)
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    data = f"data/walks-room{args.trial_set}" if args.trial_set else "data/walks"
    out_base = f"results/vggt_control_{args.trial_set}" if args.trial_set else "results/vggt_control"
    meta, pmeta, G = geometry(data)
    frames = np.load(f"{data}/frames.npy", mmap_mode="r")
    probes = np.load(f"{data}/probes.npy", mmap_mode="r")
    if args.trial_set:
        trials = [(t["probe"], t["near"], t["far"]) for t in
                  json.load(open(f"results/export/dissimilar-room{args.trial_set}/trials.json"))]
    else:
        z = np.load(args.trials)
        walks = np.unique(meta["walk"])[:args.walks]
        trials = [(int(p), int(nr), int(fr)) for p, q, th, nr, fr in
                  zip(z["probe"], z["question"], z["think"], z["near"], z["far"])
                  if q == "choice" and not th and pmeta["walk"][p] in walks]
        trials = [tr for k, tr in enumerate(trials) if not G[tr[0]]["familiar"] or k % args.familiar_every == 0]
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    for head in ("depth_head", "point_head", "track_head"):      # unused here: only the aggregator and camera head run
        setattr(model, head, None)
    model = model.cuda().eval()
    print(f"VGGT: {len(trials)} trials, walk stride {args.stride}", flush=True)

    def prep(ims):
        x = torch.from_numpy(np.stack([np.asarray(a) for a in ims])).cuda().permute(0, 3, 1, 2).float() / 255
        return F.interpolate(x, size=(RES, RES), mode="bicubic", align_corners=False).clamp(0, 1)

    out = {k: [] for k in ("probe", "familiar", "loc_err", "head_err", "fit_rms", "nearest_hit",
                           "nearest_random", "choice_right")}
    t0 = time.time()
    for b0 in range(0, len(trials), args.batch):
        chunk = trials[b0:b0 + args.batch]
        seqs, cand = [], []
        for p, nr, fr in chunk:
            fi = G[p]["fi"]
            sub = list(range(0, len(fi), args.stride))                       # walk-order indices given
            order = sub + [nr - 1, fr - 1]                                    # then the two options
            seqs.append(prep([frames[fi[k]] for k in order] + [probes[p]]))  # the probe goes last
            cand.append(order)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, _ = model.aggregator(torch.stack(seqs))
            pose = model.camera_head(tokens)[-1]
        ext, _ = pose_encoding_to_extri_intri(pose.float(), (RES, RES))     # [B, S, 3, 4], camera-from-world
        ext = ext.cpu().numpy().astype(np.float64)
        for (p, nr, fr), order, E in zip(chunk, cand, ext):
            Rcw, tcw = E[:, :, :3], E[:, :, 3]
            C = -np.einsum("sji,sj->si", Rcw, tcw)                            # camera centres
            fwd = np.einsum("sji,j->si", Rcw, np.array([0.0, 0.0, 1.0]))      # optical axis in world
            fi = G[p]["fi"]
            gt = np.stack([meta["x"][fi[order]], meta["y"][fi[order]], np.full(len(order), 1.0)], 1)
            s, R, t = umeyama(C[:-1], gt)                                     # walk frames only
            Cr = (s * (R @ C.T)).T + t
            fr_ = (R @ fwd.T).T
            rms = float(np.sqrt(((Cr[:-1] - gt) ** 2).sum(1).mean()))
            ph = Cr[-1, :2]
            loc = float(np.hypot(*(ph - [pmeta["x"][p], pmeta["y"][p]])))
            head = float(np.degrees(angdiff(np.arctan2(fr_[-1, 1], fr_[-1, 0]), pmeta["theta"][p])))
            nsub = len(order) - 2
            dpred = np.hypot(*(Cr[:nsub, :2] - ph).T)
            k = order[int(np.argmin(dpred))]
            dn, df = np.hypot(*(Cr[nsub, :2] - ph)), np.hypot(*(Cr[nsub + 1, :2] - ph))
            for key, v in [("probe", p), ("familiar", G[p]["familiar"]), ("loc_err", loc), ("head_err", head),
                           ("fit_rms", rms), ("nearest_hit", G[p]["d"][k] < NEAR),
                           ("nearest_random", float((G[p]["d"][order[:nsub]] < NEAR).mean())),
                           ("choice_right", dn < df)]:
                out[key].append(v)
        if b0 == 0 or (b0 // args.batch) % 32 == 0:
            print(f"{b0 + len(chunk)}/{len(trials)} trials, {time.time() - t0:.0f}s, "
                  f"walk-fit rms {np.median(out['fit_rms']):.2f} m, probe error {np.median(out['loc_err']):.2f} m, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB", flush=True)

    o = {k: np.array(v) for k, v in out.items()}
    np.savez(f"{out_base}.npz", **o)
    res = {}
    for cat, sel in [("familiar", o["familiar"]), ("novel", ~o["familiar"])]:
        res[cat] = {"n": int(sel.sum()), "walk_fit_rms_m": float(np.median(o["fit_rms"][sel])),
                    "probe_error_m_median": float(np.median(o["loc_err"][sel])),
                    "probe_within_1m": float((o["loc_err"][sel] < NEAR).mean()),
                    "heading_error_deg_median": float(np.median(o["head_err"][sel])),
                    "nearest_hit_rate_1m": float(o["nearest_hit"][sel].mean()),
                    "nearest_random_hit_rate": float(o["nearest_random"][sel].mean()),
                    "choice_accuracy": float(o["choice_right"][sel].mean())}
        print(cat, json.dumps({k: round(v, 3) for k, v in res[cat].items()}), flush=True)
    json.dump(res, open(f"{out_base}.json", "w"), indent=1)
    print(f"vggt_control: done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
