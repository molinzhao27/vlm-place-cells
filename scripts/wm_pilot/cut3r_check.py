"""Check wm_cut3r's step loop against CUT3R's own forward pass (the demo's path: model(views) over the whole sequence),
and wm_cut3r's preprocessing against CUT3R's load_images, on the first frames of room A's walk 0. Also the per-walk
aligned position error of the official path over the whole walk.
    ~/wm_models/envs/cut3r/bin/python scripts/wm_pilot/cut3r_check.py [--device mps] [--n 24]
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
from spatialcells.two_choice import walk_order                      # noqa: E402
from spatialcells.vggt_control import umeyama                       # noqa: E402
from spatialcells.wm_common import WM_ROOT                          # noqa: E402
from spatialcells.wm_cut3r import Stream, to_room_c                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="mps")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--rule", default="cut3r")
ap.add_argument("--only-compare", action="store_true")
args = ap.parse_args()
dev = torch.device(args.device)
repo = os.path.join(WM_ROOT, "CUT3R" if args.rule == "cut3r" else "TTT3R")
st = Stream(repo, os.path.join(WM_ROOT, "weights", "cut3r", "cut3r_512_dpt_4_64.pth"), args.rule, dev)
from dust3r.utils.camera import pose_encoding_to_camera            # noqa: E402
from dust3r.utils.image import load_images                         # noqa: E402

frames = np.load("data/walks/frames.npy", mmap_mode="r")
meta = dict(np.load("data/walks/meta.npz"))
fi = walk_order(meta)[0]

# preprocessing: CUT3R's load_images on PNG files of the same frames
tmp = tempfile.mkdtemp()
paths = []
for k in range(3):
    p = os.path.join(tmp, f"{k:03d}.png")
    Image.fromarray(np.asarray(frames[fi[k]])).save(p)
    paths.append(p)
theirs = torch.cat([v["img"] for v in load_images(paths, size=512, verbose=False)])
mine = st.prep(frames[fi[:3]]).cpu()
print(f"preprocessing: shapes {tuple(theirs.shape)} vs {tuple(mine.shape)}, max abs diff {(theirs - mine).abs().max():.2e}")


def official(rows):
    views = []
    for i, r in enumerate(rows):
        img = st.prep(frames[[r]])
        H, W = img.shape[-2:]
        views.append({"img": img, "ray_map": torch.full((1, H, W, 6), torch.nan, device=dev),
                      "true_shape": torch.tensor([[H, W]], device=dev), "idx": i, "instance": str(i),
                      "camera_pose": torch.eye(4, device=dev)[None], "img_mask": torch.tensor([True], device=dev),
                      "ray_mask": torch.tensor([False], device=dev), "update": torch.tensor([True], device=dev),
                      "reset": torch.tensor([False], device=dev)})
    with torch.no_grad():
        out = st.model(views)
    return np.stack([pose_encoding_to_camera(r["camera_pose"].float())[0, :3, 3].cpu().numpy() for r in out.ress])


ref = official(fi[:args.n])
S, C = st.fresh(1), []
for r in fi[:args.n]:
    (c, _), S = st.step(st.prep(frames[[r]]), S)
    C.append(c[0])
C = np.array(C)
print(f"step loop vs official forward, first {args.n} frames: max |camera center difference| {np.abs(C - ref).max():.2e} "
      f"(official path spans {np.ptp(ref, axis=0).round(2).tolist()}); per frame: "
      + " ".join(f"{d:.1e}" for d in np.linalg.norm(C - ref, axis=1)))
if args.only_compare:
    sys.exit(0)

full = official(fi)
gt = np.stack([meta["x"][fi], meta["y"][fi], np.ones(len(fi))], 1)
err = np.linalg.norm(to_room_c(full, umeyama(full, gt)) - gt, axis=1)
s = umeyama(full, gt)[0]
print(f"official path, walk 0 (256 frames): aligned position error median {np.median(err):.2f} m, first 64 "
      f"{np.median(err[:64]):.2f}, last 64 {np.median(err[-64:]):.2f}; similarity scale {s:.2f}")
np.savez(os.path.join("results", "wm_pilot", f"cut3r_check_walk0_{args.rule}.npz"), official=full, gt=gt)
