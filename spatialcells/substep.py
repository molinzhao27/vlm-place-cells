"""Walks with every turn split into sub-steps of at most 15 degrees (follow-up pilot WP, amendment 1 in
docs/experiments.md), for models trained on video whose camera rarely turns far between frames (CUT3R, TTT3R).

    python -m spatialcells.substep --data data/walks --out data/walks-sub15 --room A
    <habitat-sim 0.3.3 python> -m spatialcells.substep --data data/walks-roomReplicaApt0 \
        --out data/walks-roomReplicaApt0-sub15 --habitat <replica>/replica.scene_dataset_config.json \
        --scene apartment_0 --zup --seed 200

Between original steps t and t + 1 of a walk whose heading change (wrapped to (-180, 180] degrees) exceeds
--max-turn, n - 1 sub-frames are inserted, n = ceil(|dtheta| / max_turn): sub-frame j sits at step t's position
with heading theta_t + j dtheta / n. The camera turns on the spot and then steps, as room.foraging_walk moves,
and the last transition turns dtheta / n and takes the original step. Every frame, original or inserted, is
rendered at its exact pose with the scene's own renderer: room.py's ray caster, or habitat_scene.py with
experiment 12's floor rule (the floor height snapped at the frame's position; a sub-frame shares step t's).
The original poses are read from the source meta.npz, and the re-rendered originals are compared byte for byte
with the source frames (substep.json; exit status 4 if any differs).

Writes frames.npy and meta.npz: walk, step (index in the sub-stepped walk), x, y, theta, orig (an original
step), orig_step (the original step, or for a sub-frame the step it follows) and src_row (the original's row
in the source files, -1 for a sub-frame).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

_room = None


def substep(meta, walks, max_turn=np.deg2rad(15)):
    """The sub-stepped poses of `walks` (dict of arrays, one row per frame, walks in the given order)."""
    out = {k: [] for k in ("walk", "step", "x", "y", "theta", "orig", "orig_step", "src_row")}

    def add(w, s, x, y, th, orig, os_, src):
        for k, v in zip(out, (w, s, x, y, th, orig, os_, src)):
            out[k].append(v)

    for w in walks:
        rows = np.flatnonzero(meta["walk"] == w)
        rows = rows[np.argsort(meta["step"][rows])]
        x, y, th = meta["x"][rows], meta["y"][rows], meta["theta"][rows]
        s = 0
        for t, r in enumerate(rows):
            add(w, s, x[t], y[t], th[t], True, t, r)
            s += 1
            if t + 1 == len(rows):
                break
            d = (th[t + 1] - th[t] + np.pi) % (2 * np.pi) - np.pi
            n = int(np.ceil(abs(d) / max_turn - 1e-9)) if abs(d) > max_turn else 1
            for j in range(1, n):
                add(w, s, x[t], y[t], (th[t] + j * d / n) % (2 * np.pi), False, t, -1)
                s += 1
    types = dict(walk=np.int64, step=np.int64, x=np.float64, y=np.float64, theta=np.float64, orig=bool,
                 orig_step=np.int64, src_row=np.int64)
    return {k: np.array(v, dtype=types[k]) for k, v in out.items()}


def _init(layout):
    global _room
    from .room import Room
    _room = Room(layout)


def _render(pose):
    return _room.render(*pose)


def render_room(poses, layout, workers):
    from multiprocessing import Pool
    with Pool(workers, initializer=_init, initargs=(layout,)) as pool:
        return np.stack(pool.map(_render, [tuple(p) for p in poses], chunksize=16))


def render_habitat(poses, args):
    """As habitat_scene.walks(): the same simulator settings, the same reference floor height (from the scene's
    seed) and the floor snapped at each position."""
    from .habitat_scene import from_habitat, make_sim, nav, render
    sim = make_sim(args.habitat, args.scene, semantic=True, zup=args.zup)
    pf = sim.pathfinder
    pf.seed(args.seed)
    island = max(range(pf.num_islands), key=pf.island_area)
    floor = from_habitat(pf.get_random_navigable_point(island_index=island))[2]
    out = []
    for x, y, th in poses:
        f = from_habitat(pf.snap_point(nav(x, y, floor)))[2]
        out.append(render(sim, x, y, th, f)["rgb"][..., :3].copy())
    sim.close()
    return np.stack(out)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the source walks (frames.npy, meta.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--walks", type=int, default=16, help="the first N walks")
    ap.add_argument("--max-turn", type=float, default=15.0, help="degrees")
    ap.add_argument("--room", default="A", help="ray-cast room (room.get_layout)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--habitat", help="a scene_dataset_config.json: render with habitat_scene.py instead")
    ap.add_argument("--scene", help="--habitat: the scene name")
    ap.add_argument("--zup", action="store_true", help="--habitat: as habitat_scene.py --zup (Replica)")
    ap.add_argument("--seed", type=int, default=0, help="--habitat: the seed the source walks were rendered with")
    args = ap.parse_args()
    if os.path.abspath(args.out) == os.path.abspath(args.data):
        sys.exit("--out must differ from --data")
    t0 = time.time()
    meta = dict(np.load(f"{args.data}/meta.npz"))
    walks = sorted(np.unique(meta["walk"]))[:args.walks]
    sub = substep(meta, walks, np.deg2rad(args.max_turn))
    poses = np.stack([sub["x"], sub["y"], sub["theta"]], 1)
    print(f"{len(walks)} walks: {sub['orig'].sum()} original frames + {(~sub['orig']).sum()} sub-frames", flush=True)
    frames = render_habitat(poses, args) if args.habitat else render_room(poses, args.room, args.workers)
    src = np.load(f"{args.data}/frames.npy", mmap_mode="r")
    o = np.flatnonzero(sub["orig"])
    same = np.array([np.array_equal(frames[i], src[sub["src_row"][i]]) for i in o])
    os.makedirs(args.out, exist_ok=True)
    np.save(f"{args.out}/frames.npy", frames)
    np.savez(f"{args.out}/meta.npz", **sub)
    lens = np.bincount(np.searchsorted(walks, sub["walk"]))
    d = np.abs((np.diff(sub["theta"]) + np.pi) % (2 * np.pi) - np.pi)[np.diff(sub["walk"]) == 0]
    info = {"source": args.data, "renderer": f"habitat {args.scene}" if args.habitat else f"room {args.room}",
            "walks": [int(w) for w in walks], "max_turn_deg": args.max_turn, "frames": int(len(frames)),
            "original_frames": int(len(o)), "sub_frames": int(len(frames) - len(o)),
            "frames_per_walk": [int(n) for n in lens], "max_transition_turn_deg": float(np.rad2deg(d.max())),
            "originals_byte_identical": int(same.sum()), "originals_differing_rows": [int(sub["src_row"][i]) for i in o[~same]][:50],
            "md5_frames_npy": md5(f"{args.out}/frames.npy"),
            "seconds": round(time.time() - t0, 1)}
    json.dump(info, open(f"{args.out}/substep.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "frames_per_walk"}), flush=True)
    if not same.all():
        sys.exit(4)


if __name__ == "__main__":
    main()
