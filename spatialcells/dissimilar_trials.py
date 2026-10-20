"""Two-choice trials in which neither option looks like the new view, so view matching gives 50%.

    python -m spatialcells.dissimilar_trials [--room B]

For each probe in the 16 test walks: the near option is a walk frame taken within 1 m of it,
the far option one taken more than 3 m away, and both face more than 90 deg away from it
(and within 45 deg of each other's angle to it), and neither shows any poster or pillar
that the probe shows (landmarks(), by ray casting the room). Among such pairs, the one whose pixel
similarity to the probe is closest is chosen, with the sign of that small difference
alternating from trial to trial, so the more similar-looking option is the nearer one
in exactly half the trials. Takes up to PER_WALK probes per walk, novel ones first.
In room A one cue is not balanced: the near option is usually closer in frame number to the walk
frame that looks most like the probe (best_match_frame), so two_choice.py --score also reports
accuracy on the trials where that cue points to the far option. Room A's trials are kept as every
model got them. In any other room (--room B: data/walks-roomB, with probes from render_probes.py
--room B) frame order is balanced too, as in procedural_data.py: the trials cycle through the four
combinations of which option pixel similarity and frame order favour (frame-order ties excluded),
separately in novel and familiar trials.
Writes results/export/dissimilar/trials.json (room B: results/export/dissimilar-roomB/; same fields
as the Claude trials) and the probe JPEGs beside it.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

from .analyze_behave import angdiff, geometry
from .room import CAM_H, H, HFOV, L, LAYOUTS, PILLARS, _posters

OUT = "results/export/dissimilar"
WALKS = 16
PER_WALK = 12
NEAR, FAR, AWAY, SIMILAR_TURN = 1.0, 3.0, np.pi / 2, np.pi / 4
MIN_RAYS = 8          # of 64 x 64 rays: a landmark this visible counts as shown
# (sign of near minus far pixel similarity, frame order favours near), cycled outside room A
COMBOS = [(1, True), (1, False), (-1, True), (-1, False)]


def landmarks(x, y, theta, n=64, pillars=PILLARS, posters=None):
    """Set of posters ("N0", "E1", ...) and pillars ("P0", "P1") visible from (x, y) facing theta
    (room A unless another room's pillars and posters dict are given)."""
    posters = _POSTERS if posters is None else posters
    a = np.tan(HFOV / 2)
    lin = (np.arange(n) + 0.5) / n * 2 - 1
    u, v = np.meshgrid(lin * a, -lin * a)
    f, r = np.array([np.cos(theta), np.sin(theta), 0.0]), np.array([np.sin(theta), -np.cos(theta), 0.0])
    d = (f[None, None] + u[..., None] * r + v[..., None] * np.array([0, 0, 1.0])).reshape(-1, 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    dx, dy, dz = d.T
    with np.errstate(divide="ignore", invalid="ignore"):
        cands = [("floor", np.where(dz < 0, -CAM_H / dz, np.inf)), ("ceil", np.where(dz > 0, (H - CAM_H) / dz, np.inf)),
                 ("W", np.where(dx < 0, -x / dx, np.inf)), ("E", np.where(dx > 0, (L - x) / dx, np.inf)),
                 ("S", np.where(dy < 0, -y / dy, np.inf)), ("N", np.where(dy > 0, (L - y) / dy, np.inf))]
        for k, (px, py, pr, ph, _) in enumerate(pillars):
            ox, oy = x - px, y - py
            a2, b, c = dx * dx + dy * dy, 2 * (dx * ox + dy * oy), ox * ox + oy * oy - pr * pr
            disc = b * b - 4 * a2 * c
            tp = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a2)
            zp = CAM_H + tp * dz
            cands.append((f"P{k}", np.where((disc > 0) & (tp > 0) & (zp >= 0) & (zp <= ph), tp, np.inf)))
    T = np.stack([t for _, t in cands])
    hit = T.argmin(0)
    t = T.min(0)
    seen = set()
    for k, (name, _) in enumerate(cands):
        m = hit == k
        if name.startswith("P") and m.sum() >= MIN_RAYS:
            seen.add(name)
        if name in posters and m.any():
            s = (y + t[m] * dy[m]) if name in "EW" else (x + t[m] * dx[m])
            z = CAM_H + t[m] * dz[m]
            for i, (s0, s1, z0, z1, _) in enumerate(posters[name]):
                if ((s >= s0) & (s <= s1) & (z >= z0) & (z <= z1)).sum() >= MIN_RAYS:
                    seen.add(f"{name}{i}")
    return seen


_POSTERS = _posters()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="A", choices=list(LAYOUTS))
    ap.add_argument("--out", help="default results/export/dissimilar (room A) or ...-room<R>")
    args = ap.parse_args()
    data = "data/walks" if args.room == "A" else f"data/walks-room{args.room}"
    out = args.out or (OUT if args.room == "A" else f"{OUT}-room{args.room}")
    pillars, posters = LAYOUTS[args.room]["pillars"], LAYOUTS[args.room]["posters"]()
    seen = lambda x, y, th: landmarks(x, y, th, pillars=pillars, posters=posters)
    meta, pmeta, G = geometry(data)
    probes = np.load(f"{data}/probes.npy", mmap_mode="r")
    rng = np.random.default_rng(11)
    trials, sign, order = [], 1, None
    for w in range(WALKS):
        pidx = np.flatnonzero(pmeta["walk"] == w)
        pidx = sorted(pidx, key=lambda p: (G[p]["familiar"], rng.random()))     # novel first, random within
        n = 0
        for p in pidx:
            if args.room != "A":                      # cycled separately in novel and familiar trials
                sign, order = COMBOS[sum(t["familiar"] == G[p]["familiar"] for t in trials) % 4]
            fi, d, sim, bm = G[p]["fi"], G[p]["d"], G[p]["sim"], G[p]["pix"]
            dh = angdiff(meta["theta"][fi], pmeta["theta"][p])
            seen_p = seen(pmeta["x"][p], pmeta["y"][p], pmeta["theta"][p])
            shares = np.array([bool(seen_p & seen(meta["x"][f], meta["y"][f], meta["theta"][f]))
                               if dh[k] > AWAY and (d[k] < NEAR or d[k] > FAR) else True for k, f in enumerate(fi)])
            near = np.flatnonzero((d < NEAR) & (dh > AWAY) & ~shares)
            far = np.flatnonzero((d > FAR) & (dh > AWAY) & ~shares)
            best = None
            for i in near:
                ok = far[(np.abs(dh[far] - dh[i]) < SIMILAR_TURN) & (np.sign(sim[i] - sim[far]) == sign)]
                if order is not None:
                    ok = ok[(np.abs(ok - bm) != abs(i - bm)) & ((abs(i - bm) < np.abs(ok - bm)) == order)]
                if len(ok):
                    j = ok[np.argmin(np.abs(sim[i] - sim[ok]))]
                    if best is None or abs(sim[i] - sim[j]) < best[2]:
                        best = (i, j, abs(sim[i] - sim[j]))
            if best is None:
                continue
            i, j, _ = best
            nr, fr = int(i) + 1, int(j) + 1
            a, b = (nr, fr) if rng.random() < 0.5 else (fr, nr)
            trials.append({
                "id": f"w{w:02d}p{p:04d}", "walk": w, "probe": int(p), "familiar": G[p]["familiar"],
                "near": nr, "far": fr, "a": a, "b": b,
                "near_dist_m": round(float(d[i]), 3), "far_dist_m": round(float(d[j]), 3),
                "pixels_favour_near": bool(sim[i] > sim[j]), "sim_near": round(float(sim[i]), 4),
                "sim_far": round(float(sim[j]), 4), "sim_best_frame": round(float(sim.max()), 4),
                "near_heading_diff_deg": round(float(np.degrees(dh[i])), 1),
                "far_heading_diff_deg": round(float(np.degrees(dh[j])), 1),
                "probe_landmarks": sorted(seen_p),
                "best_match_frame": int(bm) + 1,     # walk frame that looks most like the probe
                "frame_order_favours_near": bool(abs(i - bm) < abs(j - bm)),
            })
            sign, n = -sign, n + 1                    # room A alternates the pixel cue only
            if n == PER_WALK:
                break
    os.makedirs(out, exist_ok=True)
    json.dump(trials, open(f"{out}/trials.json", "w"), indent=1)
    for t in trials:
        Image.fromarray(np.asarray(probes[t["probe"]])).save(f"{out}/probe_{t['probe']:04d}.jpg", quality=92)
    T = trials
    print(f"{len(T)} trials ({sum(not t['familiar'] for t in T)} novel); pixels favour near in "
          f"{np.mean([t['pixels_favour_near'] for t in T]):.0%}, frame order in "
          f"{np.mean([t['frame_order_favours_near'] for t in T]):.0%}; median |sim diff| "
          f"{np.median([abs(t['sim_near'] - t['sim_far']) for t in T]):.3f} (options' sim median "
          f"{np.median([t['sim_near'] for t in T] + [t['sim_far'] for t in T]):.2f}, best walk frame "
          f"{np.median([t['sim_best_frame'] for t in T]):.2f}); near/far distance medians "
          f"{np.median([t['near_dist_m'] for t in T]):.2f} / {np.median([t['far_dist_m'] for t in T]):.2f} m; "
          f"heading diff medians {np.median([t['near_heading_diff_deg'] for t in T]):.0f} / "
          f"{np.median([t['far_heading_diff_deg'] for t in T]):.0f} deg -> {out}/trials.json")


if __name__ == "__main__":
    main()
