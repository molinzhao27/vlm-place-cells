"""Teleport probes: views at random headings from places each walk visited.

    python -m spatialcells.render_probes --per-walk 64
    python -m spatialcells.render_probes --room B --data data/walks-roomB

Appended after a walk (in random order), these test whether a model can place a
view it was "teleported" to. Consecutive probes come from unrelated places, so
memory of the last few frames cannot help (unlike within a slow walk).
Writes <data>/probes.npy and <data>/probes_meta.npz (walk, idx, x, y, theta); --room must be
the room the walks in --data were rendered in.
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np

from .render_data import _init, _render
from .room import get_layout, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-walk", type=int, default=64)
    ap.add_argument("--data", default="data/walks")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--room", default="A", help="A, B or P<seed> (room.get_layout)")
    args = ap.parse_args()
    meta = np.load(f"{args.data}/meta.npz")
    rng = np.random.default_rng(args.seed)
    pillars = get_layout(args.room)["pillars"]
    rows = []
    for w in np.unique(meta["walk"]):
        pos = np.stack([meta["x"], meta["y"]], 1)[meta["walk"] == w]
        for k in range(args.per_walk):
            while True:
                x, y = pos[rng.integers(len(pos))] + rng.normal(0, 0.15, 2)
                if valid(x, y, pillars=pillars):
                    break
            rows.append((w, k, x, y, rng.uniform(0, 2 * np.pi)))
    rows = np.array(rows)
    with Pool(args.workers, initializer=_init, initargs=(args.room,)) as pool:
        frames = np.stack(pool.map(_render, [tuple(r[2:]) for r in rows], chunksize=16))
    np.save(f"{args.data}/probes.npy", frames)
    np.savez(f"{args.data}/probes_meta.npz", walk=rows[:, 0].astype(int), idx=rows[:, 1].astype(int),
             x=rows[:, 2], y=rows[:, 3], theta=rows[:, 4])
    print(f"rendered {len(frames)} probes -> {args.data}/probes.npy")


if __name__ == "__main__":
    main()
