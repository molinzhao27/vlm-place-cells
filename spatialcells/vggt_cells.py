"""Does a model that knows where it is have place cells? VGGT's internal units.

    python -m spatialcells.vggt_cells --mode alone|single|walk
    python -m spatialcells.vggt_cells --mode single --room B --data data/walks-roomB   (remapping)
    python -m spatialcells.vggt_cells --mode single --blocks 21      (add block 21; other files kept)

VGGT (Wang et al., CVPR 2025) predicts each image's camera pose in the frame of the first
image it is given. Every sequence here starts with the same 8 anchor views (the room's
centre facing each wall, and four spots 3 m in from the corners facing the centre;
rendered on first use to data/walks/anchors.npz), so that frame is the same for every
input. Modes:
  alone   each walk frame by itself: nothing to localise against
  single  each walk frame after the anchors: it can localise, but has no walk memory
  walk    64 consecutive walk frames after the anchors (each walk in 4 chunks)
Records the MLP neurons (fc2 input, 4096) at each walk frame's camera token in 7 of the
24 global-attention blocks, as data/acts/vggt-<mode>/L<block>.npy (n_frames x 4096, in
frames.npy order); room B goes to data/acts/vggt-<mode>-roomB. Also saves the predicted camera centres in room coordinates, by a
similarity fit on the anchors only, as data/acts/vggt-<mode>/pose.npz.
Needs the VGGT repo at /workspace/vggt (see vggt_control.py).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .extract import DEPTHS, save
from .vggt_control import RES, umeyama

ANCHORS = [(5.0, 5.0, 0.0), (5.0, 5.0, np.pi / 2), (5.0, 5.0, np.pi), (5.0, 5.0, 3 * np.pi / 2),
           (2.0, 2.0, np.pi / 4), (8.0, 2.0, 3 * np.pi / 4), (8.0, 8.0, -3 * np.pi / 4), (2.0, 8.0, -np.pi / 4)]
CHUNK = 64


def anchors(data, layout="A"):
    path = f"{data}/anchors.npz"
    if not os.path.exists(path):
        from .room import Room
        room = Room(layout)
        np.savez(path, images=np.stack([room.render(x, y, th) for x, y, th in ANCHORS]), poses=np.array(ANCHORS))
    return np.load(path)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["alone", "single", "walk"], required=True)
    ap.add_argument("--data", default="data/walks")
    ap.add_argument("--room", default="A", help="room layout (anchors are rendered in it)")
    ap.add_argument("--vggt", default="/workspace/vggt")
    ap.add_argument("--walks", type=int, help="only the first N walks (smoke test; not saved)")
    ap.add_argument("--blocks", type=int, nargs="+", help="record these global blocks instead (pose.npz not rewritten)")
    args = ap.parse_args()
    sys.path.insert(0, args.vggt)
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    frames = np.load(f"{args.data}/frames.npy", mmap_mode="r")
    meta = dict(np.load(f"{args.data}/meta.npz"))
    A = anchors(args.data, args.room)
    model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
    agg = model.aggregator
    P = agg.patch_start_idx + (RES // 14) ** 2                       # tokens per image: camera, registers, patches
    want = args.blocks or sorted({round(d * (len(agg.global_blocks) - 1)) for d in DEPTHS})
    rec = {}
    for g in want:   # camera token (index 0) of every image, at the input of each chosen block's fc2
        agg.global_blocks[g].mlp.fc2.register_forward_pre_hook(
            lambda _, inp, g=g: rec.__setitem__(f"L{g}", inp[0][:, ::P].float()))
    n_anchor = 0 if args.mode == "alone" else len(A["images"])
    acts = {f"L{g}": np.zeros((len(frames), agg.global_blocks[g].mlp.fc2.in_features), np.float16) for g in want}
    xy = np.full((len(frames), 2), np.nan)
    heading = np.full(len(frames), np.nan)

    def prep(ims):
        x = torch.from_numpy(np.stack([np.asarray(a) for a in ims])).cuda().permute(0, 3, 1, 2).float() / 255
        return F.interpolate(x, size=(RES, RES), mode="bicubic", align_corners=False).clamp(0, 1)

    anchor_px = prep(A["images"]) if n_anchor else None
    gt_anchor = np.c_[A["poses"][:, :2], np.ones(len(A["poses"]))]

    def run(groups):
        """groups: list of frame-index arrays, each its own sequence (after the anchors)."""
        seqs = [torch.cat([anchor_px, prep(frames[g])]) if n_anchor else prep(frames[g]) for g in groups]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, _ = agg(torch.stack(seqs))
            pose = model.camera_head(tokens)[-1] if n_anchor else None
        for k, v in rec.items():
            v = v[:, n_anchor:].cpu().numpy()                          # walk frames only
            for b, g in enumerate(groups):
                acts[k][g] = v[b].astype(np.float16)
        if pose is None:
            return
        ext, _ = pose_encoding_to_extri_intri(pose.float(), (RES, RES))
        ext = ext.cpu().numpy().astype(np.float64)
        for E, g in zip(ext, groups):
            R, t = E[:, :, :3], E[:, :, 3]
            C = -np.einsum("sji,sj->si", R, t)
            fwd = np.einsum("sji,j->si", R, np.array([0.0, 0.0, 1.0]))
            s, Ra, ta = umeyama(C[:n_anchor], gt_anchor)               # anchors only
            Cr, fr = (s * (Ra @ C.T)).T + ta, (Ra @ fwd.T).T
            xy[g] = Cr[n_anchor:, :2]
            heading[g] = np.arctan2(fr[n_anchor:, 1], fr[n_anchor:, 0])

    t0 = time.time()
    order = []
    for w in np.unique(meta["walk"])[:args.walks]:
        i = np.flatnonzero(meta["walk"] == w)
        order.append(i[np.argsort(meta["step"][i])])
    if args.mode == "walk":
        groups = [o[c:c + CHUNK] for o in order for c in range(0, len(o), CHUNK)]
        batch = 1
    else:
        groups = [np.array([f]) for o in order for f in o]
        batch = 48 if args.mode == "single" else 64
    for b0 in range(0, len(groups), batch):
        run(groups[b0:b0 + batch])
        if b0 % (batch * 50) == 0:
            err = np.nanmedian(np.hypot(xy[:, 0] - meta["x"], xy[:, 1] - meta["y"])) if n_anchor else np.nan
            print(f"{b0 + batch}/{len(groups)} sequences, {time.time() - t0:.0f}s, median position error so far "
                  f"{err:.2f} m, peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB", flush=True)
    if not args.walks:
        out = f"data/acts/vggt-{args.mode}" + ("" if args.room == "A" else f"-room{args.room}")
        save(out, acts)
        if not args.blocks:
            np.savez(f"{out}/pose.npz", x=xy[:, 0], y=xy[:, 1], theta=heading)
    if n_anchor:
        done = np.isfinite(xy[:, 0])
        err = np.hypot(xy[done, 0] - meta["x"][done], xy[done, 1] - meta["y"][done])
        dh = np.degrees(np.abs((heading[done] - meta["theta"][done] + np.pi) % (2 * np.pi) - np.pi))
        print(f"vggt-{args.mode}: camera position error median {np.median(err):.2f} m "
              f"(90th pct {np.percentile(err, 90):.2f}), heading error median {np.median(dh):.1f} deg", flush=True)
    print(f"vggt-{args.mode}: done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
