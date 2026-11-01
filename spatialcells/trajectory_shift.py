"""Experiment 30 (pre-registered in docs/experiments.md): the fine-tuned model on walks from other walk distributions.

    python -m spatialcells.trajectory_shift --dry-run                    # poses and geometry only: no image, no model
    python -m spatialcells.trajectory_shift --check-builder              # build_trials() against experiment 10's lists
    python -m spatialcells.trajectory_shift --prepare [--rooms ..] [--conditions ..] [--workers 8]
                                    # Mac CPU: render, pixel features, landmarks, trial lists, payload for vast-3
    python -m spatialcells.trajectory_shift --dino [--threads 6]        # Mac CPU: render again (MD5-checked), DINOv2
    python -m spatialcells.trajectory_shift --combine                    # the combined novel-trial manifest (roomE30)
    python -m spatialcells.trajectory_shift --unpack                     # vast-3: data/walks-roomE30 from the payloads
    python -m spatialcells.trajectory_shift --check-unpacked             # vast-3: exit 0 if that is current
    python -m spatialcells.trajectory_shift --parity ADAPTER             # vast-3: experiment 10's P5000 answers again

Walks (256 steps; seeded from the room and the condition, never experiment 10's seeds), in the held-out procedural
rooms P5000-P5009 without P5005 (its binary_blobs poster cannot be rendered again exactly):
  control    room.foraging_walk with its defaults: the training walk distribution (step 0.22 m x U(0.7, 1.3),
             heading drift SD 15 deg per step, on-the-spot head scans of 30-90 deg with p 0.15)
  long-step  the same with step 0.44 m: twice the step length
  low-turn   the same with heading drift SD 5 deg and no head scans: much less turning
  patrol     a 128-step foraging walk with the defaults, then a return to its start and the same route walked again in
             the same direction (foraging steps and head scans as usual, the heading pulled halfway toward the route
             point 3 ahead on each step) until step 256: most places are visited twice, about 150 steps apart
Probes as render_probes.py (64 per walk, near visited places, random headings).

Trials (build_trials):
  standard   dissimilar_trials.py's rules exactly (mode "standard" reproduces experiment 10's trial lists byte for
             byte; --check-builder): near option within 1 m, far more than 3 m away, both facing more than 90 deg away
             from the new view and within 45 deg of each other's turn, neither showing a poster or pillar the new view
             shows, pixel similarity and frame order each favouring near in half the trials; up to 192 / walks per
             walk, novel first
  nbhd       (control and patrol walks) the same option rules, with each pair typed by its options' walk
             neighbourhoods in the 64-frame prompt (frame_cut.py's near and far sets: the 4 shown frames before and the
             4 after each option, never an option). conflict: no frame of the near option's neighbourhood shows any
             poster or pillar the new view shows, and at least one frame of the far option's does. agreement: the
             reverse. Trials cycle through (conflict, agreement) x (pixels favour near, far); frame order is not
             balanced (it follows the type) and is recorded. Novel probes only, up to 192 / walks per walk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from .analyze_behave import angdiff
from .dissimilar_trials import AWAY, COMBOS, FAR, NEAR, SIMILAR_TURN, landmarks
from .room import L, MARGIN, foraging_walk, get_layout, valid

ROOMS = [f"P{s}" for s in range(5000, 5010) if s != 5005]
CONDITIONS = {"control": {}, "long-step": {"step": 0.44}, "low-turn": {"turn_sd": np.deg2rad(5), "p_scan": 0.0},
              "patrol": {"patrol": True}}
COND_ORDER = list(CONDITIONS)
N_WALKS = {"control": 48, "long-step": 16, "low-turn": 16, "patrol": 32}
NBHD_CONDITIONS = ["control", "patrol"]
STEPS, PROBES_PER_WALK, TRIALS_PER_ROOM = 256, 64, 192
SIDE = 4                          # neighbourhood: 4 shown frames each side (experiment 15's near and far sets)
SHOWN = np.linspace(1, STEPS, 61).round().astype(int)          # two_choice.present --max-images 64: 61 evenly spaced
NBHD_COMBOS = [("conflict", 1), ("agreement", 1), ("conflict", -1), ("agreement", -1)]
PATROL = dict(lap=128, gain=0.5, reach=0.5, look=3)
E30 = "data/e30"                  # per room and condition: poses, features, landmarks, frame MD5s; payload/ for vast-3
EXPORT = "results/export/e30"     # trial lists
COMBINED = "roomE30"              # vast-3: data/walks-roomE30, results/export/dissimilar-roomE30/trials.json


def scene(room, cond):
    return f"{room}-{cond}"


def walk_seed(room, cond):
    return 30_000 + 10 * (int(room[1:]) - 5000) + COND_ORDER.index(cond)


def probe_seed(room, cond):
    return 31_000 + 10 * (int(room[1:]) - 5000) + COND_ORDER.index(cond)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def patrol_walk(rng, pillars, step=0.22, turn_sd=np.deg2rad(15), p_scan=0.15):
    """A 128-step foraging walk (the route), then back to its start and along it again, until STEPS poses. After the
    route, each step is foraging_walk's (scan with p_scan, else drift and a move with wall turns), with the heading also
    pulled by `gain` toward the route point `look` ahead of the last one reached (within `reach` m)."""
    route = foraging_walk(PATROL["lap"], rng, pillars=pillars)
    tg = route[:, :2]
    poses = list(route)
    x, y, th = route[-1]
    th, j = float(th), 0
    while len(poses) < STEPS:
        poses.append(np.array([x, y, th % (2 * np.pi)]))
        if rng.random() < p_scan:
            th += rng.choice([-1, 1]) * rng.uniform(np.deg2rad(30), np.deg2rad(90))
            continue
        while j < len(tg) - 1 and np.hypot(tg[j, 0] - x, tg[j, 1] - y) < PATROL["reach"]:
            j += 1
        k = min(j + PATROL["look"], len(tg) - 1) if j > 0 else 0
        th += PATROL["gain"] * _wrap(np.arctan2(tg[k, 1] - y, tg[k, 0] - x) - th) + rng.normal(0, turn_sd)
        for _ in range(20):
            s = step * rng.uniform(0.7, 1.3)
            nx, ny = x + s * np.cos(th), y + s * np.sin(th)
            if valid(nx, ny, pillars=pillars):
                x, y = nx, ny
                break
            th += rng.choice([-1, 1]) * rng.uniform(np.deg2rad(60), np.deg2rad(180))
    return np.stack(poses[:STEPS])


def make_walks(room, cond, n_walks=None, pillars=None):
    """(n_walks, STEPS, 3) poses (x, y, theta) for one room and condition."""
    n_walks = N_WALKS[cond] if n_walks is None else n_walks
    pillars = get_layout(room)["pillars"] if pillars is None else pillars
    rng = np.random.default_rng(walk_seed(room, cond))
    if cond == "patrol":
        return np.stack([patrol_walk(rng, pillars) for _ in range(n_walks)])
    return np.stack([foraging_walk(STEPS, rng, pillars=pillars, **CONDITIONS[cond]) for _ in range(n_walks)])


def make_probes(room, cond, poses, per_walk=PROBES_PER_WALK, pillars=None):
    """render_probes.py's sampling: rows (walk, idx, x, y, theta)."""
    pillars = get_layout(room)["pillars"] if pillars is None else pillars
    rng = np.random.default_rng(probe_seed(room, cond))
    rows = []
    for w, walk in enumerate(poses):
        pos = walk[:, :2]
        for k in range(per_walk):
            while True:
                x, y = pos[rng.integers(len(pos))] + rng.normal(0, 0.15, 2)
                if valid(x, y, pillars=pillars):
                    break
            rows.append((w, k, x, y, rng.uniform(0, 2 * np.pi)))
    return np.array(rows)


def metas(poses, prows):
    W, S = poses.shape[:2]
    walk, step = np.meshgrid(np.arange(W), np.arange(S), indexing="ij")
    flat = poses.reshape(-1, 3)
    meta = {"walk": walk.ravel(), "step": step.ravel(), "x": flat[:, 0], "y": flat[:, 1], "theta": flat[:, 2]}
    pmeta = {"walk": prows[:, 0].astype(int), "idx": prows[:, 1].astype(int), "x": prows[:, 2], "y": prows[:, 3],
             "theta": prows[:, 4]}
    return meta, pmeta


def walk_stats(poses):
    """Manipulation check: per-step displacement and heading change, path length, area covered, revisits."""
    d = np.hypot(*np.diff(poses[:, :, :2], axis=1).transpose(2, 0, 1))
    turn = angdiff(poses[:, 1:, 2], poses[:, :-1, 2])
    moved = d > 1e-9
    gaps = []
    for walk in poses[:, :, :2]:
        D = np.hypot(*(walk[:, None] - walk[None]).transpose(2, 0, 1))
        for i in range(0, STEPS, 8):                    # every 8th frame: the longest time gap to a frame within 0.5 m
            gaps.append(int(np.abs(np.flatnonzero(D[i] < 0.5) - i).max()))
    cells = [len({(int(x // 1), int(y // 1)) for x, y in walk[:, :2]}) for walk in poses]
    return {"step_m_mean_when_moving": float(d[moved].mean()), "share_steps_moving": float(moved.mean()),
            "turn_deg_mean": float(np.degrees(turn.mean())), "turn_deg_median": float(np.degrees(np.median(turn))),
            "path_m": float(d.sum(1).mean()), "cells_1m_visited": float(np.mean(cells)),
            "revisit_gap_steps_median": float(np.median(gaps)),
            "share_frames_revisited_after_64_steps": float(np.mean(np.array(gaps) >= 64))}


# ------------------------------------------------------------------------------------------------ features
def pixel_sums(frames):
    """8 x 8 block sums (uint16) of each frame: analyze_probes.pixel_feats is exactly (sums / 64).astype(float32)."""
    return np.stack([np.asarray(f).reshape(32, 8, 32, 8, 3).sum((1, 3), dtype=np.uint16).ravel() for f in frames])


def pix_from_sums(s):
    return (s.astype(np.float64) / 64).astype(np.float32)


def landmark_sets(meta, pmeta, layout):
    pillars, posters = layout["pillars"], layout["posters"]()
    f = [landmarks(x, y, th, pillars=pillars, posters=posters) for x, y, th in zip(meta["x"], meta["y"], meta["theta"])]
    p = [landmarks(x, y, th, pillars=pillars, posters=posters) for x, y, th in zip(pmeta["x"], pmeta["y"], pmeta["theta"])]
    return f, p


# ------------------------------------------------------------------------------------------------ trials
def shown_order(a, b):
    """The frames a 64-frame prompt shows, in order (two_choice.present with --max-images 64, ordered walk)."""
    return sorted(set(SHOWN.tolist()) | {a, b})


def around(o, cand, pos):
    """frame_cut.frame_sets' neighbourhood of option o: the 4 candidates before and the 4 after it in the order shown
    (topped up from the other side at a walk end)."""
    before = [k for k in cand if pos[k] < pos[o]]
    after = [k for k in cand if pos[k] > pos[o]]
    nb, na = min(SIDE, len(before)), min(SIDE, len(after))
    if nb < SIDE:
        na = min(len(after), 2 * SIDE - nb)
    elif na < SIDE:
        nb = min(len(before), 2 * SIDE - na)
    return before[len(before) - nb:] + after[:na]


def neighbourhoods(near, far):
    """Both options' 8-frame neighbourhoods (real frame numbers) in the 64-frame prompt."""
    shown = shown_order(near, far)
    pos = {k: i for i, k in enumerate(shown)}
    cand = [k for k in shown if k not in (near, far)]
    return around(near, cand, pos), around(far, cand, pos)


def nbhd_type(near, far, shares):
    """'conflict', 'agreement' or None; shares[k - 1]: walk frame k shows a poster or pillar the new view shows."""
    sn, sf = neighbourhoods(near, far)
    hn, hf = any(shares[k - 1] for k in sn), any(shares[k - 1] for k in sf)
    return "conflict" if hf and not hn else "agreement" if hn and not hf else None


def geometry(meta, pmeta, Wf, Pf):
    """analyze_behave.geometry from given pixel features (z-scored over all walk frames given)."""
    mu, sd = Wf.mean(0), Wf.std(0) + 1e-6
    unit = lambda X: (X - mu) / sd / np.linalg.norm((X - mu) / sd, axis=1, keepdims=True)
    Wn, Pn = unit(Wf), unit(Pf)
    walk_frames = {}
    for w in np.unique(meta["walk"]):
        i = np.flatnonzero(meta["walk"] == w)
        walk_frames[w] = i[np.argsort(meta["step"][i])]
    G = {}
    for p in range(len(pmeta["walk"])):
        fi = walk_frames[pmeta["walk"][p]]
        d = np.hypot(meta["x"][fi] - pmeta["x"][p], meta["y"][fi] - pmeta["y"][p])
        dh = angdiff(meta["theta"][fi], pmeta["theta"][p])
        G[p] = {"d": d, "familiar": bool(((d < NEAR) & (dh < np.pi / 4)).any()),
                "sim": Wn[fi] @ Pn[p], "pix": int(np.argmax(Wn[fi] @ Pn[p])), "fi": fi}
    return G


def build_trials(meta, pmeta, G, frame_marks, probe_marks, walks, per_walk, mode="standard"):
    """dissimilar_trials.main's loop (rooms other than A, no holes) for mode "standard"; mode "nbhd" types each pair by
    its options' neighbourhoods and cycles NBHD_COMBOS over novel probes only."""
    rng = np.random.default_rng(11)
    trials, sign, order = [], 1, None
    for w in range(walks):
        pidx = np.flatnonzero(pmeta["walk"] == w)
        pidx = sorted(pidx, key=lambda p: (G[p]["familiar"], rng.random()))     # novel first, random within
        n = 0
        for p in pidx:
            if mode == "nbhd" and G[p]["familiar"]:
                break
            if mode == "standard":
                sign, order = COMBOS[sum(t["familiar"] == G[p]["familiar"] for t in trials) % 4]
            else:
                want, sign = NBHD_COMBOS[len(trials) % 4]
            fi, d, sim, bm = G[p]["fi"], G[p]["d"], G[p]["sim"], G[p]["pix"]
            dh = angdiff(meta["theta"][fi], pmeta["theta"][p])
            seen_p = probe_marks(p)
            lm = np.array([bool(seen_p & frame_marks(f)) for f in fi])
            shares = np.array([lm[k] if dh[k] > AWAY and (d[k] < NEAR or d[k] > FAR) else True for k in range(len(fi))])
            near = np.flatnonzero((d < NEAR) & (dh > AWAY) & ~shares)
            far = np.flatnonzero((d > FAR) & (dh > AWAY) & ~shares)
            best = None
            for i in near:
                ok = far[(np.abs(dh[far] - dh[i]) < SIMILAR_TURN) & (np.sign(sim[i] - sim[far]) == sign)]
                if order is not None and mode == "standard":
                    ok = ok[(np.abs(ok - bm) != abs(i - bm)) & ((abs(i - bm) < np.abs(ok - bm)) == order)]
                if mode == "nbhd":
                    ok = np.array([j for j in ok if nbhd_type(int(i) + 1, int(j) + 1, lm) == want], dtype=int)
                if len(ok):
                    j = ok[np.argmin(np.abs(sim[i] - sim[ok]))]
                    if best is None or abs(sim[i] - sim[j]) < best[2]:
                        best = (i, j, abs(sim[i] - sim[j]))
            if best is None:
                continue
            i, j, _ = best
            nr, fr = int(i) + 1, int(j) + 1
            a, b = (nr, fr) if rng.random() < 0.5 else (fr, nr)
            t = {
                "id": f"w{w:02d}p{p:04d}", "walk": w, "probe": int(p), "familiar": G[p]["familiar"],
                "near": nr, "far": fr, "a": a, "b": b,
                "near_dist_m": round(float(d[i]), 3), "far_dist_m": round(float(d[j]), 3),
                "pixels_favour_near": bool(sim[i] > sim[j]), "sim_near": round(float(sim[i]), 4),
                "sim_far": round(float(sim[j]), 4), "sim_best_frame": round(float(sim.max()), 4),
                "near_heading_diff_deg": round(float(np.degrees(dh[i])), 1),
                "far_heading_diff_deg": round(float(np.degrees(dh[j])), 1),
                "probe_landmarks": sorted(seen_p),
                "best_match_frame": int(bm) + 1,
                "frame_order_favours_near": bool(abs(i - bm) < abs(j - bm)),
            }
            if mode == "nbhd":
                sn, sf = neighbourhoods(nr, fr)
                t.update(type=want, near_nbhd=sn, far_nbhd=sf,
                         near_nbhd_sharing=int(sum(lm[k - 1] for k in sn)), far_nbhd_sharing=int(sum(lm[k - 1] for k in sf)),
                         near_nbhd_dist_m=[round(float(d[k - 1]), 3) for k in sn],
                         far_nbhd_dist_m=[round(float(d[k - 1]), 3) for k in sf])
            trials.append(t)
            if mode == "standard":
                sign = -sign
            n += 1
            if n == per_walk:
                break
    return trials


def check_builder(args):
    """Rebuild experiment 10's trial lists from its renders with build_trials and compare the JSON text."""
    from .analyze_probes import pixel_feats
    ok = True
    for room in args.rooms:
        data = f"data/walks-room{room}"
        man_path = f"results/export/dissimilar-room{room}/trials.json"
        man = json.load(open(man_path))
        walks = max(t["walk"] for t in man) + 1
        per_walk = 12 if walks == 16 else 6
        meta, pmeta = dict(np.load(f"{data}/meta.npz")), dict(np.load(f"{data}/probes_meta.npz"))
        t0 = time.time()
        Wf = pixel_feats(np.load(f"{data}/frames.npy", mmap_mode="r"))
        Pf = pixel_feats(np.load(f"{data}/probes.npy", mmap_mode="r"))
        lay = get_layout(room)
        fl, pl = landmark_sets(meta, pmeta, lay)
        G = geometry(meta, pmeta, Wf, Pf)
        trials = build_trials(meta, pmeta, G, lambda f: fl[f], lambda p: pl[p], walks, per_walk)
        same = json.dumps(trials, indent=1) == open(man_path).read()
        ok &= same
        print(f"{room}: {len(trials)} trials ({walks} walks, {per_walk} per walk) {'IDENTICAL' if same else 'DIFFERENT'} "
              f"to {man_path} ({time.time() - t0:.0f} s)", flush=True)
    sums_ok = all(np.array_equal(pix_from_sums(pixel_sums(f[None]))[0], pixel_feats(f[None])[0])
                  for f in np.load("data/walks-roomP5000/frames.npy", mmap_mode="r")[:64])
    print(f"pixel sums reproduce pixel_feats on 64 frames: {sums_ok}")
    raise SystemExit(0 if ok and sums_ok else 1)


# ------------------------------------------------------------------------------------------------ dry run
def dry_run(args):
    """Geometry-only counts per room and condition: novel probes, probes with a valid standard pair (the pixel cue and
    frame order need renders, so this is an upper bound), and probes with a valid conflict or agreement pair."""
    res = {}
    for cond in args.conditions:
        for room in args.rooms:
            t0 = time.time()
            lay = get_layout(room)
            poses = make_walks(room, cond, pillars=lay["pillars"])
            prows = make_probes(room, cond, poses, pillars=lay["pillars"])
            meta, pmeta = metas(poses, prows)
            r = count_candidates(meta, pmeta, lay, len(poses), nbhd=cond in NBHD_CONDITIONS)
            r["walks"] = len(poses)
            r["walk_stats"] = walk_stats(poses)
            res[scene(room, cond)] = r
            print(f"{room} {cond:9s} {len(poses)} walks: novel probes {r['novel_probes']:4d}, with a valid pair "
                  f"{r['standard_probes']:4d} (capped at {TRIALS_PER_ROOM // len(poses)} per walk {r['standard_capped']:4d}); "
                  f"conflict-capable {r.get('conflict_probes', '-')!s:>4s}, agreement-capable "
                  f"{r.get('agreement_probes', '-')!s:>4s}  ({time.time() - t0:.0f} s)", flush=True)
    if args.exp10:
        for room in args.rooms:
            data = f"data/walks-room{room}"
            meta, pmeta = dict(np.load(f"{data}/meta.npz")), dict(np.load(f"{data}/probes_meta.npz"))
            nw = int(meta["walk"].max()) + 1
            r = count_candidates(meta, pmeta, get_layout(room), nw, nbhd=True)
            man = json.load(open(f"results/export/dissimilar-room{room}/trials.json"))
            r["exp10_novel_trials"] = sum(not t["familiar"] for t in man)
            res[f"{room}-exp10"] = r
            print(f"{room} exp10 ({nw} walks): novel probes {r['novel_probes']:4d}, with a valid pair {r['standard_probes']:4d}"
                  f", conflict-capable {r['conflict_probes']:4d}, agreement-capable {r['agreement_probes']:4d}; "
                  f"novel trials built {r['exp10_novel_trials']}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")


def count_candidates(meta, pmeta, lay, n_walks, nbhd):
    from .analyze_behave import angdiff as ad
    order = {}
    for w in np.unique(meta["walk"]):
        i = np.flatnonzero(meta["walk"] == w)
        order[int(w)] = i[np.argsort(meta["step"][i])]
    fl, pl = landmark_sets(meta, pmeta, lay)
    out = {"novel_probes": 0, "standard_probes": 0, "standard_capped": 0}
    if nbhd:
        out.update(conflict_probes=0, agreement_probes=0)
    per_walk = {}
    for p in range(len(pmeta["walk"])):
        w = int(pmeta["walk"][p])
        if w >= n_walks:
            continue
        fi = order[w]
        d = np.hypot(meta["x"][fi] - pmeta["x"][p], meta["y"][fi] - pmeta["y"][p])
        dh = ad(meta["theta"][fi], pmeta["theta"][p])
        if ((d < NEAR) & (dh < np.pi / 4)).any():
            continue
        out["novel_probes"] += 1
        lm = np.array([bool(pl[p] & fl[f]) for f in fi])
        near = np.flatnonzero((d < NEAR) & (dh > AWAY) & ~lm)
        far = np.flatnonzero((d > FAR) & (dh > AWAY) & ~lm)
        pairs = [(i, j) for i in near for j in far[np.abs(dh[far] - dh[i]) < SIMILAR_TURN]]
        if not pairs:
            continue
        out["standard_probes"] += 1
        per_walk[w] = per_walk.get(w, 0) + 1
        if nbhd:
            types = {nbhd_type(int(i) + 1, int(j) + 1, lm) for i, j in pairs}
            out["conflict_probes"] += "conflict" in types
            out["agreement_probes"] += "agreement" in types
    out["standard_capped"] = int(sum(min(TRIALS_PER_ROOM // n_walks, v) for v in per_walk.values()))
    return out


# ------------------------------------------------------------------------------------------------ prepare (Mac)
_room = None


def _init(room):
    global _room
    from .room import Room
    _room = Room(room)


def _render(pose):
    return _room.render(*pose)


def render_all(room, poses, prows, workers):
    from multiprocessing import Pool
    with Pool(workers, initializer=_init, initargs=(room,)) as pool:
        frames = np.stack(pool.map(_render, [tuple(p) for p in poses.reshape(-1, 3)], chunksize=16))
        probes = np.stack(pool.map(_render, [tuple(r[2:]) for r in prows], chunksize=16))
    return frames, probes


def md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def prepare(args):
    """Per room and condition: poses, renders, pixel sums, landmarks, trial lists, and a payload of the images the
    64-frame prompts of the novel trials show (61 evenly spaced frames per walk, the options, the new views)."""
    for cond in args.conditions:
        for room in args.rooms:
            sc = scene(room, cond)
            out = f"{E30}/{sc}"
            if os.path.exists(f"{out}/done.json"):
                print(f"{sc}: done", flush=True)
                continue
            os.makedirs(out, exist_ok=True)
            os.makedirs(f"{EXPORT}/{sc}", exist_ok=True)
            os.makedirs(f"{E30}/payload", exist_ok=True)
            t0 = time.time()
            lay = get_layout(room)
            poses = make_walks(room, cond, pillars=lay["pillars"])
            prows = make_probes(room, cond, poses, pillars=lay["pillars"])
            meta, pmeta = metas(poses, prows)
            frames, probes = render_all(room, poses, prows, args.workers)
            t_r = time.time() - t0
            fs, ps = pixel_sums(frames), pixel_sums(probes)
            fl, pl = landmark_sets(meta, pmeta, lay)
            G = geometry(meta, pmeta, pix_from_sums(fs), pix_from_sums(ps))
            W = len(poses)
            sets = {"standard": build_trials(meta, pmeta, G, lambda f: fl[f], lambda p: pl[p], W, TRIALS_PER_ROOM // W)}
            if cond in NBHD_CONDITIONS:
                sets["nbhd"] = build_trials(meta, pmeta, G, lambda f: fl[f], lambda p: pl[p], W, TRIALS_PER_ROOM // W,
                                            mode="nbhd")
            for name, ts in sets.items():
                json.dump(ts, open(f"{EXPORT}/{sc}/{name}.json", "w"), indent=1)
            # payload: what the novel trials' 64-frame prompts show
            order = {w: np.flatnonzero(meta["walk"] == w)[np.argsort(meta["step"][meta["walk"] == w])] for w in range(W)}
            rows, prow = set(), set()
            for ts in sets.values():
                for t in ts:
                    if t["familiar"]:
                        continue
                    rows |= {int(order[t["walk"]][k - 1]) for k in shown_order(t["a"], t["b"])}
                    prow.add(t["probe"])
            rows, prow = np.array(sorted(rows)), np.array(sorted(prow))
            np.savez_compressed(f"{E30}/payload/{sc}.npz", frame_rows=rows, frames=frames[rows], probe_rows=prow,
                                probes=probes[prow])
            np.savez(f"{out}/poses.npz", **{f"frame_{k}": v for k, v in meta.items()},
                     **{f"probe_{k}": v for k, v in pmeta.items()})
            np.savez_compressed(f"{out}/pixels.npz", frame_sums=fs, probe_sums=ps)
            json.dump({"frames": [sorted(s) for s in fl], "probes": [sorted(s) for s in pl]}, open(f"{out}/landmarks.json", "w"))
            json.dump({"frames": [md5(f) for f in frames], "probes": [md5(p) for p in probes]}, open(f"{out}/md5.json", "w"))
            info = {"scene": sc, "walks": W, "walk_seed": walk_seed(room, cond), "probe_seed": probe_seed(room, cond),
                    "frames": len(frames), "probes": len(probes), "walk_stats": walk_stats(poses),
                    "trials": {k: {"all": len(v), "novel": sum(not t["familiar"] for t in v),
                                   **({"conflict": sum(t.get("type") == "conflict" for t in v),
                                       "agreement": sum(t.get("type") == "agreement" for t in v)} if k == "nbhd" else {})}
                               for k, v in sets.items()},
                    "payload": {"frames": len(rows), "probes": len(prow), "md5": md5_file(f"{E30}/payload/{sc}.npz")},
                    "render_s": round(t_r), "total_s": round(time.time() - t0)}
            json.dump(info, open(f"{out}/done.json", "w"), indent=1)
            print(f"{sc}: {info['trials']}, payload {len(rows)} frames + {len(prow)} new views, render {t_r:.0f} s, "
                  f"total {time.time() - t0:.0f} s", flush=True)
            del frames, probes


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def dino(args):
    """Per room and condition, the walks rendered again (each frame's MD5 checked against prepare's) and DINOv2-base's
    CLS token and patch-token mean of every walk frame and of every trial's new view, as retrieval_baselines.features
    computes them, into data/e30/<scene>/dino.npz."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel
    torch.set_num_threads(args.threads)
    model = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def feats(images, batch=32):
        cls, patch = [], []
        for i in range(0, len(images), batch):
            x = torch.from_numpy(np.array(images[i:i + batch])).permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, size=224, mode="bicubic", antialias=True)
            h = model(pixel_values=(x - mean) / std).last_hidden_state
            cls.append(h[:, 0].float().numpy())
            patch.append(h[:, 1:].mean(1).float().numpy())
        return np.concatenate(cls), np.concatenate(patch)

    for cond in args.conditions:
        for room in args.rooms:
            sc = scene(room, cond)
            out = f"{E30}/{sc}"
            if os.path.exists(f"{out}/dino.npz"):
                print(f"{sc}: cached", flush=True)
                continue
            t0 = time.time()
            lay = get_layout(room)
            poses = make_walks(room, cond, pillars=lay["pillars"])
            prows = make_probes(room, cond, poses, pillars=lay["pillars"])
            frames, probes = render_all(room, poses, prows, args.workers)
            ref = json.load(open(f"{out}/md5.json"))
            bad = sum(md5(f) != m for f, m in zip(frames, ref["frames"])) + sum(md5(p) != m for p, m in zip(probes, ref["probes"]))
            if bad:
                raise SystemExit(f"{sc}: {bad} frames render differently from prepare's")
            pidx = sorted({t["probe"] for name in ("standard", "nbhd") if os.path.exists(f"{EXPORT}/{sc}/{name}.json")
                           for t in json.load(open(f"{EXPORT}/{sc}/{name}.json"))})
            fc, fp = feats(frames)
            pc, pp = feats(probes[pidx])
            np.savez(f"{out}/dino.npz", probe_idx=np.array(pidx), frame_cls=fc, frame_patch=fp, probe_cls=pc, probe_patch=pp)
            print(f"{sc}: {len(frames)} frames (MD5 identical), {len(pidx)} new views, {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------ vast-3
def combine(args):
    """One manifest of every novel trial (standard and nbhd sets) with global walk and probe indices, for one
    two_choice run per adapter on vast-3 (--trials roomE30), and the index the unpack step needs."""
    trials, index, w_off, p_off = [], [], 0, 0
    for cond in COND_ORDER:
        for room in ROOMS:
            sc = scene(room, cond)
            info = json.load(open(f"{E30}/{sc}/done.json"))
            for name in ("standard", "nbhd"):
                path = f"{EXPORT}/{sc}/{name}.json"
                if not os.path.exists(path):
                    continue
                for t in json.load(open(path)):
                    if t["familiar"]:
                        continue
                    trials.append({**{k: t[k] for k in ("near", "far", "a", "b", "familiar")},
                                   "id": f"{sc}-{name}-{t['id']}", "walk": t["walk"] + w_off, "probe": t["probe"] + p_off,
                                   "scene": sc, "room": room, "condition": cond, "set": name, "local_id": t["id"],
                                   **({"type": t["type"]} if "type" in t else {})})
            index.append({"scene": sc, "walk_offset": w_off, "probe_offset": p_off, "walks": info["walks"],
                          "probes": info["probes"], "payload": f"{sc}.npz", "payload_md5": info["payload"]["md5"]})
            w_off += info["walks"]
            p_off += info["probes"]
    trials.sort(key=lambda t: (t["walk"], t["id"]))
    os.makedirs(f"results/export/dissimilar-{COMBINED}", exist_ok=True)
    json.dump(trials, open(f"results/export/dissimilar-{COMBINED}/trials.json", "w"), indent=1)
    json.dump({"walks": w_off, "probes": p_off, "scenes": index}, open(f"{E30}/payload/index.json", "w"), indent=1)
    n = {}
    for t in trials:
        k = f"{t['condition']}/{t['set']}" + (f"/{t['type']}" if "type" in t else "")
        n[k] = n.get(k, 0) + 1
    print(f"{len(trials)} novel trials, {w_off} walks, {p_off} new views: {n}")


def unpack(args):
    """vast-3: data/walks-roomE30 (frames.npy and probes.npy filled only where a prompt shows an image; the rest of the
    files are holes) and the meta files, from data/e30/payload/ and data/e30/<scene>/poses.npz; MD5s checked."""
    idx = json.load(open(f"{E30}/payload/index.json"))
    data = f"data/walks-{COMBINED}"
    os.makedirs(data, exist_ok=True)
    F = np.lib.format.open_memmap(f"{data}/frames.npy", mode="w+", dtype=np.uint8, shape=(idx["walks"] * STEPS, 256, 256, 3))
    P = np.lib.format.open_memmap(f"{data}/probes.npy", mode="w+", dtype=np.uint8, shape=(idx["probes"], 256, 256, 3))
    meta = {k: [] for k in ("walk", "step", "x", "y", "theta")}
    pmeta = {k: [] for k in ("walk", "idx", "x", "y", "theta")}
    for s in idx["scenes"]:
        path = f"{E30}/payload/{s['payload']}"
        assert md5_file(path) == s["payload_md5"], f"{path}: MD5 differs from the Mac's"
        z = np.load(path)
        ref = json.load(open(f"{E30}/{s['scene']}/md5.json"))
        fo, po = s["walk_offset"] * STEPS, s["probe_offset"]
        for r, img in zip(z["frame_rows"], z["frames"]):
            assert md5(img) == ref["frames"][r], (s["scene"], "frame", r)
            F[fo + r] = img
        for r, img in zip(z["probe_rows"], z["probes"]):
            assert md5(img) == ref["probes"][r], (s["scene"], "probe", r)
            P[po + r] = img
        q = np.load(f"{E30}/{s['scene']}/poses.npz")
        for k in meta:
            meta[k].append(q[f"frame_{k}"] + (s["walk_offset"] if k == "walk" else 0))
        for k in pmeta:
            pmeta[k].append(q[f"probe_{k}"] + (s["walk_offset"] if k == "walk" else 0))
        print(f"{s['scene']}: {len(z['frame_rows'])} frames, {len(z['probe_rows'])} new views", flush=True)
    F.flush()
    P.flush()
    np.savez(f"{data}/meta.npz", **{k: np.concatenate(v) for k, v in meta.items()})
    np.savez(f"{data}/probes_meta.npz", **{k: np.concatenate(v) for k, v in pmeta.items()})
    json.dump({"index_md5": md5_file(f"{E30}/payload/index.json")}, open(f"{data}/unpacked.json", "w"))
    print(f"-> {data}")


def check_unpacked(args):
    """Exit 0 if data/walks-roomE30 was unpacked from the current payload index."""
    path = f"data/walks-{COMBINED}/unpacked.json"
    ok = os.path.exists(path) and json.load(open(path))["index_md5"] == md5_file(f"{E30}/payload/index.json")
    print(f"{path}: {'current' if ok else 'missing or stale'}")
    raise SystemExit(0 if ok else 1)


def parity(args):
    """Experiment 10's P5000 trials answered again by this adapter on this machine (64-frame walk) against its stored
    answers: exit 1 below 98% identical."""
    name = f"Qwen3-VL-8B-Instruct+{os.path.basename(args.parity.rstrip('/'))}"
    ref = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(f"results/two_choice/{name}-roomP5000-256px-max64.jsonl")}
    new = {json.loads(l)["id"]: json.loads(l)["answer"]
           for l in open(f"results/two_choice/e30/{name}-roomP5000-256px-max64-e30parity.jsonl")}
    assert set(new) == set(ref), "the parity run does not cover experiment 10's P5000 trials"
    same = sum(new[k] == ref[k] for k in ref)
    print(f"parity {name}: {same}/{len(ref)} identical answers")
    raise SystemExit(0 if same >= 0.98 * len(ref) else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-builder", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--dino", action="store_true")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--unpack", action="store_true")
    ap.add_argument("--check-unpacked", action="store_true")
    ap.add_argument("--parity", metavar="ADAPTER")
    ap.add_argument("--rooms", nargs="+", default=ROOMS)
    ap.add_argument("--conditions", nargs="+", default=COND_ORDER)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--exp10", action="store_true", help="dry run: also count experiment 10's own walks (calibration)")
    ap.add_argument("--out", default=f"{E30}/dry_run.json")
    args = ap.parse_args()
    if args.dry_run:
        dry_run(args)
    elif args.check_builder:
        check_builder(args)
    elif args.prepare:
        prepare(args)
    elif args.dino:
        dino(args)
    elif args.combine:
        combine(args)
    elif args.unpack:
        unpack(args)
    elif args.check_unpacked:
        check_unpacked(args)
    elif args.parity:
        parity(args)


if __name__ == "__main__":
    main()
