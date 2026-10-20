"""Render foraging walks through the room.

    python -m spatialcells.render_data --walks 48 --steps 256 --out data/walks
    python -m spatialcells.render_data --room B --out data/walks-roomB     (second room, for remapping)

Writes frames.npy (N, 256, 256, 3) uint8 and meta.npz (walk, step, x, y, theta),
plus preview figures in results/preview/.
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import numpy as np

from .room import IMG, L, LAYOUTS, Room, foraging_walk

_room = None


def _init(layout):
    global _room
    _room = Room(layout)


def _render(pose):
    return _room.render(*pose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks", type=int, default=48)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/walks")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--room", default="A", choices=list(LAYOUTS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pillars = LAYOUTS[args.room]["pillars"]
    poses = np.stack([foraging_walk(args.steps, rng, pillars=pillars) for _ in range(args.walks)])  # (W, S, 3)
    flat = poses.reshape(-1, 3)
    with Pool(args.workers, initializer=_init, initargs=(args.room,)) as pool:
        frames = np.stack(pool.map(_render, [tuple(p) for p in flat], chunksize=16))
    walk, step = np.meshgrid(np.arange(args.walks), np.arange(args.steps), indexing="ij")
    np.save(os.path.join(args.out, "frames.npy"), frames)
    np.savez(os.path.join(args.out, "meta.npz"), walk=walk.ravel(), step=step.ravel(),
             x=flat[:, 0], y=flat[:, 1], theta=flat[:, 2])
    print(f"rendered {len(frames)} frames {frames.shape} -> {args.out}")
    preview(frames, flat, poses, args.room)


def preview(frames, flat, poses, layout="A"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = "results/preview" + ("" if layout == "A" else f"/room{layout}")
    os.makedirs(out, exist_ok=True)
    room = Room(layout)
    spots = [(5, 5, 0), (5, 5, np.pi / 2), (5, 5, np.pi), (5, 5, 3 * np.pi / 2),
             (1.5, 1.5, np.pi / 4), (8.5, 8.5, 5 * np.pi / 4), (8.5, 1.5, 3 * np.pi / 4), (2, 8, -np.pi / 4)]
    fig, axs = plt.subplots(2, 4, figsize=(16, 8.5))
    for ax, (x, y, th) in zip(axs.ravel(), spots):
        ax.imshow(room.render(x, y, th))
        ax.set_title(f"x={x}, y={y}, facing {np.rad2deg(th) % 360:.0f}°")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{out}/views.png", dpi=80)
    fig, ax = plt.subplots(figsize=(7, 7))
    for w in poses[:6]:
        ax.plot(w[:, 0], w[:, 1], lw=0.8)
    ax.scatter(flat[:, 0], flat[:, 1], s=0.3, c="k", alpha=0.15)
    for px, py, pr, _, _ in room.pillars:
        ax.add_patch(plt.Circle((px, py), pr, color="tab:red", alpha=0.6))
    ax.set_xlim(0, L), ax.set_ylim(0, L), ax.set_aspect("equal")
    ax.set_title("Foraging walks (6 shown) and all visited positions")
    fig.savefig(f"{out}/walks.png", dpi=80)


if __name__ == "__main__":
    main()
