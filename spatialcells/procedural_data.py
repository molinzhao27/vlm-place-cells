"""Procedural rooms and lure-free two-choice trials for teaching a VLM the task (experiment 9).

    python -m spatialcells.procedural_data --preview                  # results/preview/procedural.png
    python -m spatialcells.procedural_data --rooms 1200 --out data/proc

Each room (room.random_layout(seed)) gets one 256-step foraging walk. Only the 64 frames a
64-image prompt shows are rendered (frame numbers 1, 5, 9, ..., 256, evenly spaced as in
two_choice.py --max-images), plus PROBES teleport views near visited places at random headings
(as render_probes.py). Trials follow dissimilar_trials.py, with the options taken from the shown
frames: the near option within 1 m of the probe, the far one more than 3 m away, both facing more
than 90 deg away from it and within 45 deg of each other's turn, and neither showing any poster or
pillar the probe shows. Two shortcuts are balanced: pixel similarity to the probe, and frame
order (which option is nearer in frame number to the shown frame that looks most like the
probe) each favour the near option in half the trials: each room cycles through the four
combinations in its own random order (ties excluded). Up to PER_ROOM trials per room, novel first. Writes data/proc/room_XXXX.npz (frames, frame numbers, probes) and data/proc/trials.json.
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from .analyze_behave import angdiff
from .analyze_probes import pixel_feats
from .dissimilar_trials import AWAY, FAR, NEAR, SIMILAR_TURN, landmarks
from .room import Room, foraging_walk, random_layout, valid

STEPS, SHOWN, PROBES, PER_ROOM = 256, 64, 40, 8
FRAMES = np.linspace(1, STEPS, SHOWN).round().astype(int)        # 1-based frame numbers shown


def make_room(seed, out):
    lay = random_layout(seed)
    room = Room(lay)
    posters, pillars = lay["posters"](), lay["pillars"]
    rng = np.random.default_rng(seed)
    walk = foraging_walk(STEPS, rng, pillars=pillars)
    shown = walk[FRAMES - 1]
    probes = []
    while len(probes) < PROBES:
        x, y = walk[rng.integers(STEPS), :2] + rng.normal(0, 0.15, 2)
        if valid(x, y, pillars=pillars):
            probes.append((x, y, rng.uniform(0, 2 * np.pi)))
    probes = np.array(probes)
    f_img = np.stack([room.render(*p) for p in shown])
    p_img = np.stack([room.render(*p) for p in probes])
    np.savez_compressed(f"{out}/room_{seed:04d}.npz", frames=f_img, frame_numbers=FRAMES, probes=p_img,
                        frame_poses=shown, probe_poses=probes, walk=walk)

    W, P = pixel_feats(f_img), pixel_feats(p_img)
    mu, sd = W.mean(0), W.std(0) + 1e-6
    unit = lambda X: (X - mu) / sd / np.linalg.norm((X - mu) / sd, axis=1, keepdims=True)
    sims = unit(P) @ unit(W).T                                             # (PROBES, SHOWN)
    seen_f = [landmarks(*p, pillars=pillars, posters=posters) for p in shown]
    order = sorted(range(PROBES), key=lambda p: rng.random())
    familiar = [bool(((np.hypot(*(walk[:, :2] - probes[p, :2]).T) < NEAR) & (angdiff(walk[:, 2], probes[p, 2]) < np.pi / 4)).any())
                for p in range(PROBES)]
    order = sorted(order, key=lambda p: familiar[p])                     # novel first
    combos = [(True, True), (True, False), (False, True), (False, False)]
    combos = [combos[c] for c in rng.permutation(4)]                    # cue combinations, shuffled per room
    trials, k = [], 0
    for p in order:
        d = np.hypot(*(shown[:, :2] - probes[p, :2]).T)
        dh = angdiff(shown[:, 2], probes[p, 2])
        seen_p = landmarks(*probes[p], pillars=pillars, posters=posters)
        clean = np.array([not (seen_p & s) for s in seen_f]) & (dh > AWAY)
        near, far = np.flatnonzero((d < NEAR) & clean), np.flatnonzero((d > FAR) & clean)
        sim, best_frame = sims[p], FRAMES[np.argmax(sims[p])]
        want_pix, want_order = combos[k % 4]
        best = None
        for i in near:
            for j in far:
                if abs(dh[i] - dh[j]) >= SIMILAR_TURN or (sim[i] > sim[j]) != want_pix:
                    continue
                di, dj = abs(FRAMES[i] - best_frame), abs(FRAMES[j] - best_frame)
                if di == dj or (di < dj) != want_order:
                    continue
                if best is None or abs(sim[i] - sim[j]) < best[2]:
                    best = (i, j, abs(sim[i] - sim[j]))
        if best is None:
            continue
        i, j, _ = best
        nr, fr = int(FRAMES[i]), int(FRAMES[j])
        a, b = (nr, fr) if rng.random() < 0.5 else (fr, nr)
        trials.append({"id": f"r{seed:04d}p{p:02d}", "room": seed, "probe": p, "familiar": familiar[p],
                       "near": nr, "far": fr, "a": a, "b": b, "near_dist_m": round(float(d[i]), 3),
                       "far_dist_m": round(float(d[j]), 3), "pixels_favour_near": bool(sim[i] > sim[j]),
                       "frame_order_favours_near": bool(want_order), "best_match_frame": int(best_frame),
                       "near_heading_diff_deg": round(float(np.degrees(dh[i])), 1),
                       "far_heading_diff_deg": round(float(np.degrees(dh[j])), 1)})
        k += 1
        if k == PER_ROOM:
            break
    return trials


def _job(a):
    return make_room(*a)


def preview():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(6, 4, figsize=(12, 18))
    for r in range(6):
        room = Room(random_layout(r))
        for c, th in enumerate((0, np.pi / 2, np.pi, 3 * np.pi / 2)):
            axs[r, c].imshow(room.render(5, 5, th))
            axs[r, c].set_title(f"room {r}, facing {np.degrees(th):.0f}°", fontsize=8)
            axs[r, c].axis("off")
    fig.tight_layout()
    os.makedirs("results/preview", exist_ok=True)
    fig.savefig("results/preview/procedural.png", dpi=70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=1200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="data/proc")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    if args.preview:
        return preview()
    os.makedirs(args.out, exist_ok=True)
    with Pool(args.workers) as pool:
        per_room = pool.map(_job, [(s, args.out) for s in range(args.start, args.start + args.rooms)], chunksize=1)
    path = f"{args.out}/trials.json"
    old = [t for t in json.load(open(path)) if not args.start <= t["room"] < args.start + args.rooms] if os.path.exists(path) else []
    trials = old + [t for ts in per_room for t in ts]
    json.dump(trials, open(path, "w"), indent=1)
    T = [t for ts in per_room for t in ts]
    print(f"{args.rooms} rooms -> {len(T)} trials ({sum(not t['familiar'] for t in T)} novel); per room "
          f"{np.mean([len(ts) for ts in per_room]):.1f}; pixels favour near {np.mean([t['pixels_favour_near'] for t in T]):.0%}, "
          f"frame order favours near {np.mean([t['frame_order_favours_near'] for t in T]):.0%}; near/far medians "
          f"{np.median([t['near_dist_m'] for t in T]):.2f} / {np.median([t['far_dist_m'] for t in T]):.2f} m -> {path}")


if __name__ == "__main__":
    main()
