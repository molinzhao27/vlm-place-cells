"""Experiment 29 (pre-registered in docs/experiments.md): does the fine-tuned model's decision use where the walk
frames around the near option were taken, or what they show?

    python -m spatialcells.vlm_intervention --build                     # CPU: swap poses, renders, checks -> SPECS
    python -m spatialcells.vlm_intervention --stimulus-check            # CPU: DINOv2, landmarks, the two agents
    python -m spatialcells.vlm_intervention --dry-run N [--condition here-away]   # prompts and token counts; no model
    python -m spatialcells.vlm_intervention --run clean here-toward here-away there-toward there-away random-away \
        [near-cut] --adapter data/ft/qwen8b-proc-e4-s5 --device cuda|mps
    python -m spatialcells.vlm_intervention --render-check [--per-room 2]   # renders vs SPECS' MD5s (another machine)
    python -m spatialcells.vlm_intervention --build-control | --control-stimulus   # amendment 1's frames and checks (CPU)

Trials: experiment 10's novel trials in P5000-P5009 with its exact 64-frame prompts (task_record.build_prompt; the
token count of every prompt is checked against experiment 10's run of the same trial). The "near set" is experiment
15's: the 8 shown walk frames around the near option (4 before and 4 after in the order shown, topped up at walk
ends; frame_cut.frame_sets), never an option frame. In each swap condition those 8 frames' images are replaced by
renders of the same room (room.Room, the renderer of the stored frames), with every label, the order, both options,
the new view and the question unchanged:

  here-toward    each frame at its own place, facing the new view's target point
  here-away      each frame at its own place, facing directly away from that point
  there-toward   the 8 places moved together by one translation (3 m or more from the new view), facing the point
  there-away     the moved places, facing directly away from the point
  random-away    experiment 15's random set (8 random shown frames) at their own places, facing away from the point
  near-cut       no swap: the near set cut from attention in every layer (frame_cut.AttentionCut), a reference
  clean          no change (experiment 10's prompt); runs first, and must reproduce experiment 10's answers

The target point is where the new view's central ray (horizontal, at camera height) first meets a wall or a pillar.
The translation for the "there" conditions: lengths 3.0, 3.25, ..., 9.0 m, 32 directions in an order seeded by the
trial; the first (shortest) for which every moved place is walkable (room.valid, the room's pillars), at least 3 m
from the new view's position and at least 1 m from the target point, with at least 6 of the 8 moved places seeing
the target point with no pillar in the way. Trials with no such translation are left out of every condition.

P5005 draws two textures with skimage's unseeded binary_blobs, so its frames cannot be re-rendered exactly. Each of
its swap frames is rendered with two different blob draws; a frame whose two renders are identical shows no blob
texture and equals what the original room renders. P5005 trials with any frame that shows a blob texture (in any
condition, or in the identity re-render of the near and random sets) are left out of every condition.

--build writes SPECS (per trial: the new view's pose, the target point, the translation, each condition's swapped
frames with their poses and the MD5 of each render, inclusion) after checking that every frame of the near and random
sets re-renders byte-identically at its original pose. --run renders each swap frame again (a few worker processes
ahead of the model) and stops if its MD5 differs from SPECS, so every machine shows the model the same images.
One JSONL per condition: results/vlm_intervention/<model>+<adapter>-<condition>-<device>.jsonl (rows as frame_cut's);
trials already there are skipped. When "clean" is among the conditions it runs first and is checked against
experiment 10 (>= 99% identical answers over the included trials, else the job stops before any swap).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np

from .frame_cut import AttentionCut, frame_spans, key, load, load_sets, novel_rows
from .task_record import MODEL, ROOMS, build_prompt, sites
from .two_choice import parse, walk_order

OUT = "results/vlm_intervention"
SPECS = f"{OUT}/specs.json"
STIM = f"{OUT}/stimulus_check.json"
ADAPTER = "data/ft/qwen8b-proc"
SWAPS = ("here-toward", "here-away", "there-toward", "there-away", "random-away")
CONTROL_SWAPS = ("block-there", "far-far", "far-close")              # experiment 29's amendment 1
ALL_SWAPS = SWAPS + CONTROL_SWAPS
CONDITIONS = ("clean",) + ALL_SWAPS + ("near-cut",)
CONTROL_SPECS = f"{OUT}/specs_control.json"
CONTROL_STIM = f"{OUT}/stimulus_control.json"
FAR_FROM_VIEW, MIN_TO_TARGET, MIN_SIGHT = 3.0, 1.0, 6
RADII = np.arange(3.0, 9.0001, 0.25)
CLOSE_OFFSETS = np.arange(0.0, 1.5001, 0.25)
N_DIR = 32
BLOB_ROOM, BLOB_SEEDS = "P5005", (11, 23)
MAX_NEW = 16


# ---------------------------------------------------------------- geometry

def target_point(x, y, theta, pillars, L=10.0):
    """Where the horizontal ray from (x, y) along theta (at camera height) first meets a wall or a pillar, and what it
    meets ("wall" or the pillar's index)."""
    dx, dy = math.cos(theta), math.sin(theta)
    best, what = math.inf, "wall"
    for t in [(-x / dx) if dx < 0 else (L - x) / dx if dx > 0 else math.inf,
              (-y / dy) if dy < 0 else (L - y) / dy if dy > 0 else math.inf]:
        if 0 < t < best:
            best = t
    for i, (px, py, pr, _, _) in enumerate(pillars):
        ox, oy = x - px, y - py
        b = 2 * (dx * ox + dy * oy)
        c = ox * ox + oy * oy - pr * pr
        disc = b * b - 4 * c
        if disc > 0:
            t = (-b - math.sqrt(disc)) / 2
            if 0 < t < best:
                best, what = t, i
    return (x + best * dx, y + best * dy), what


def facing(sx, sy, q):
    return math.atan2(q[1] - sy, q[0] - sx) % (2 * math.pi)


def sees(sx, sy, q, what, pillars):
    """True if no pillar (other than the one the target point is on) crosses the segment from (sx, sy) to q."""
    vx, vy = q[0] - sx, q[1] - sy
    n2 = vx * vx + vy * vy
    for i, (px, py, pr, _, _) in enumerate(pillars):
        if i == what:
            continue
        t = min(1.0, max(0.0, ((px - sx) * vx + (py - sy) * vy) / n2)) if n2 > 0 else 0.0
        if math.hypot(sx + t * vx - px, sy + t * vy - py) < pr:
            return False
    return True


def trial_seed(r):
    return int(hashlib.sha256(f"vlm_intervention/{key(r)}".encode()).hexdigest(), 16) % 2 ** 32


def translation(places, view, q, what, pillars, seed):
    """The shortest translation (RADII, N_DIR directions in a seeded order) that keeps every place walkable, at least
    FAR_FROM_VIEW from the new view and MIN_TO_TARGET from the target point, with at least MIN_SIGHT of the places in
    sight of the target (the original places see it 94% of the time)."""
    from .room import valid
    dirs = list(range(N_DIR))
    random.Random(seed).shuffle(dirs)
    for rad in RADII:
        for j in dirs:
            a = 2 * math.pi * j / N_DIR
            d = (float(rad * math.cos(a)), float(rad * math.sin(a)))
            moved = [(sx + d[0], sy + d[1]) for sx, sy in places]
            if all(valid(mx, my, pillars=pillars) and math.hypot(mx - view[0], my - view[1]) >= FAR_FROM_VIEW
                   and math.hypot(mx - q[0], my - q[1]) >= MIN_TO_TARGET for mx, my in moved) \
                    and sum(sees(mx, my, q, what, pillars) for mx, my in moved) >= MIN_SIGHT:
                return d
    return None


def swap_poses(r, st, meta, order, probes_meta, pillars):
    """Trial r's conditions: condition -> [[frame number, x, y, theta], ...] (in the set's shown order), plus the
    target point and translation. st is frame_cut's sets entry for the trial."""
    fi = order[r["walk"]]
    pose = lambda k: (float(meta["x"][fi[k - 1]]), float(meta["y"][fi[k - 1]]), float(meta["theta"][fi[k - 1]]))
    p = r["probe"]
    view = (float(probes_meta["x"][p]), float(probes_meta["y"][p]), float(probes_meta["theta"][p]))
    q, what = target_point(*view, pillars)
    near = [(k, *pose(k)[:2]) for k in st["near"]]
    rnd = [(k, *pose(k)[:2]) for k in st["random"]]
    d = translation([(x, y) for _, x, y in near], view, q, what, pillars, trial_seed(r))
    away = lambda x, y: (facing(x, y, q) + math.pi) % (2 * math.pi)
    cond = {"here-toward": [[k, x, y, facing(x, y, q)] for k, x, y in near],
            "here-away": [[k, x, y, away(x, y)] for k, x, y in near],
            "random-away": [[k, x, y, away(x, y)] for k, x, y in rnd]}
    if d is not None:
        mv = [(k, x + d[0], y + d[1]) for k, x, y in near]
        cond["there-toward"] = [[k, x, y, facing(x, y, q)] for k, x, y in mv]
        cond["there-away"] = [[k, x, y, away(x, y)] for k, x, y in mv]
    ident = {k: pose(k) for k in dict.fromkeys(st["near"] + st["random"])}
    return {"view": list(view), "target": list(q), "target_on": what if what == "wall" else f"pillar{what}",
            "translation": list(d) if d else None, "conditions": cond,
            "identity": [[k, *ident[k]] for k in ident]}


def control_poses(k, s, pose, pillars):
    """Experiment 29 amendment 1's conditions for one included trial with no near-far overlap: block-there, far-close and
    far-far (docs/experiments.md), or None with the reason. s is the trial's SPECS entry, pose(frame) its (x, y)."""
    from .room import valid
    view, q = s["view"][:2], s["target"]

    def directions(tag):
        d = list(range(N_DIR))
        random.Random(int(hashlib.sha256(f"vlm_control/{tag}/{k}".encode()).hexdigest(), 16) % 2 ** 32).shuffle(d)
        return [2 * math.pi * j / N_DIR for j in d]

    def fits(places, d, keep_far):
        return all(valid(x + d[0], y + d[1], pillars=pillars) and math.dist((x + d[0], y + d[1]), q) >= MIN_TO_TARGET
                   and (not keep_far or math.dist((x + d[0], y + d[1]), view) >= FAR_FROM_VIEW) for x, y in places)

    away = lambda x, y: (facing(x, y, q) + math.pi) % (2 * math.pi)
    swap = lambda ks, d: [[f, x + d[0], y + d[1], away(x + d[0], y + d[1])] for f, (x, y) in zip(ks, map(pose, ks))]
    cand = [f for f in s["shown"] if f not in (s["near"], s["far"])]
    used = set(s["near_set"]) | set(s["far_set"])
    eligible = []
    for i, c in enumerate(cand):
        blk = cand[max(i - 4, 0):i] + cand[i + 1:i + 5]
        if len(blk) == 8 and c not in used and not used & set(blk) \
                and all(math.dist(pose(f), view) >= FAR_FROM_VIEW for f in blk):
            eligible.append((c, blk))
    random.Random(int(hashlib.sha256(f"vlm_control/center/{k}".encode()).hexdigest(), 16) % 2 ** 32).shuffle(eligible)
    dirs, block = directions("block"), None
    for c, blk in eligible:
        block = next(((c, blk, (float(rad * math.cos(a)), float(rad * math.sin(a)))) for rad in RADII for a in dirs
                      if fits(list(map(pose, blk)), (rad * math.cos(a), rad * math.sin(a)), True)), None)
        if block:
            break
    if not block:
        return None, "no_block"
    far = list(map(pose, s["far_set"]))
    mean = np.mean(far, 0)
    close = next((d for r in CLOSE_OFFSETS for a in directions("close")
                  for d in [(float(view[0] + r * math.cos(a) - mean[0]), float(view[1] + r * math.sin(a) - mean[1]))]
                  if fits(far, d, False)), None)
    if close is None:
        return None, "no_close"
    length = math.hypot(*close)
    apart = next((d for dl in (0, 0.25, -0.25, 0.5, -0.5) for a in directions("far")
                  for d in [(float((length + dl) * math.cos(a)), float((length + dl) * math.sin(a)))]
                  if fits(far, d, True)), None)
    if apart is None:
        return None, "no_far_far"
    return {"center": block[0], "block_set": block[1], "d_block": list(block[2]), "d_close": list(close), "d_far": list(apart),
            "conditions": {"block-there": swap(block[1], block[2]), "far-close": swap(s["far_set"], close),
                           "far-far": swap(s["far_set"], apart)}}, None


# ---------------------------------------------------------------- rendering (worker processes)

_ROOMS = {}


def room_obj(room, blob_seed=None):
    """room.Room for a held-out room; with blob_seed, skimage's binary_blobs draws are seeded (P5005 only)."""
    k = (room, blob_seed)
    if k not in _ROOMS:
        from skimage import data as skdata
        from .room import Room
        if blob_seed is None:
            _ROOMS[k] = Room(room)
        else:
            orig, n = skdata.binary_blobs, [0]

            def seeded(*a, **kw):
                n[0] += 1
                return orig(*a, **{**kw, "rng": blob_seed * 100 + n[0]})
            skdata.binary_blobs = seeded
            try:
                _ROOMS[k] = Room(room)
            finally:
                skdata.binary_blobs = orig
    return _ROOMS[k]


def render_one(room, x, y, theta):
    """(image, blob_free): P5005 is rendered with two blob draws; blob_free says whether they agree."""
    if room == BLOB_ROOM:
        a, b = (room_obj(room, s).render(x, y, theta) for s in BLOB_SEEDS)
        return a, bool(np.array_equal(a, b))
    return room_obj(room).render(x, y, theta), True


def md5(img):
    return hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()


def _render_job(job):
    room, poses = job
    out = []
    for x, y, th in poses:
        img, free = render_one(room, x, y, th)
        out.append((img, free))
    return out


def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def pool(n):
    return mp.get_context("spawn").Pool(n, initializer=_worker_init)


# ---------------------------------------------------------------- build (CPU)

def build(args, rows, sets):
    from .room import get_layout
    t0 = time.time()
    specs = {"settings": {
        "trials": "novel trials of P5000-P5009 (experiment 10), experiment 15's near and random sets",
        "target": "first wall or pillar hit by the new view's central horizontal ray",
        "translation": f"lengths {RADII[0]}-{RADII[-1]} m step 0.25, {N_DIR} directions in a seeded order; moved places "
                       f"walkable (room.valid), >= {FAR_FROM_VIEW} m from the new view, >= {MIN_TO_TARGET} m from the "
                       f"target, >= {MIN_SIGHT} of 8 with the target in sight", "blob_room": BLOB_ROOM, "blob_seeds": list(BLOB_SEEDS),
        "renderer": "spatialcells.room.Room(room).render(x, y, theta), 256 px"}, "trials": {}}
    jobs, where = [], []
    for room in dict.fromkeys(r["room"] for r in rows):
        d = f"data/walks-room{room}"
        meta, pm = dict(np.load(f"{d}/meta.npz")), dict(np.load(f"{d}/probes_meta.npz"))
        order = walk_order(meta)
        pillars = get_layout(room)["pillars"]
        for r in [x for x in rows if x["room"] == room]:
            s = swap_poses(r, sets[key(r)], meta, order, pm, pillars)
            s.update(room=room, id=r["id"], walk=r["walk"], probe=r["probe"], near=r["near"], far=r["far"],
                     near_set=sets[key(r)]["near"], far_set=sets[key(r)]["far"], random_set=sets[key(r)]["random"],
                     shown=sets[key(r)]["shown"], overlap_near_far=sets[key(r)]["overlap_near_far"])
            specs["trials"][key(r)] = s
            for c, fr in s["conditions"].items():
                jobs.append((room, [tuple(f[1:]) for f in fr]))
                where.append((key(r), c))
            jobs.append((room, [tuple(f[1:]) for f in s["identity"]]))
            where.append((key(r), "identity"))
    print(f"{len(rows)} trials, {len(jobs)} render jobs ({sum(len(j[1]) for j in jobs)} frames)", flush=True)
    stored = {}
    n_ident_bad = 0
    with pool(args.workers) as pl:
        for i, ((k, c), res) in enumerate(zip(where, pl.imap(_render_job, jobs, chunksize=4))):
            s = specs["trials"][k]
            free = [f for _, f in res]
            if c == "identity":
                room = s["room"]
                if room not in stored:
                    stored.clear()
                    d = f"data/walks-room{room}"
                    stored[room] = (np.load(f"{d}/frames.npy", mmap_mode="r"),
                                    walk_order(dict(np.load(f"{d}/meta.npz"))))
                frames, order = stored[room]
                fi = order[s["walk"]]
                same = [bool(np.array_equal(img, frames[fi[f[0] - 1]])) for (img, _), f in zip(res, s["identity"])]
                s["identity_blob_free"] = all(free)
                s["identity_same"] = all(sm for sm, fr in zip(same, free) if fr)
                bad = [f[0] for (sm, fr), f in zip(zip(same, free), s["identity"]) if fr and not sm]
                if bad:
                    n_ident_bad += 1
                    print(f"IDENTITY MISMATCH {k}: frames {bad} re-render differently from the stored frames", flush=True)
            else:
                for f, (img, fr) in zip(s["conditions"][c], res):
                    f.append(md5(img))
                    f.append(fr)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)
    ex = {"no_translation": 0, "blob_texture": 0}
    for k, s in specs["trials"].items():
        reason = None
        if s["translation"] is None:
            reason = "no_translation"
        elif not s["identity_blob_free"] or not all(f[5] for fr in s["conditions"].values() for f in fr):
            reason = "blob_texture"
        s["included"], s["exclude_reason"] = reason is None, reason
        if reason:
            ex[reason] += 1
    inc = [s for s in specs["trials"].values() if s["included"]]
    per_room = {room: sum(s["room"] == room for s in inc) for room in dict.fromkeys(r["room"] for r in rows)}
    tl = np.array([math.hypot(*s["translation"]) for s in inc])
    specs["build"] = {"n_trials": len(rows), "n_included": len(inc), "excluded": ex, "included_per_room": per_room,
                      "identity_mismatch_trials": n_ident_bad,
                      "translation_m": {"median": float(np.median(tl)), "min": float(tl.min()), "max": float(tl.max())},
                      "seconds": round(time.time() - t0)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(specs, open(SPECS, "w"))
    print(json.dumps(specs["build"], indent=1))
    if n_ident_bad:
        raise SystemExit(f"BUILD FAILED: {n_ident_bad} trials have frames that do not re-render byte-identically")
    print(f"BUILD PASSED -> {SPECS}")


def load_specs():
    if not os.path.exists(SPECS):
        raise SystemExit(f"{SPECS} is missing: run python -m spatialcells.vlm_intervention --build first (CPU)")
    specs = json.load(open(SPECS))
    if os.path.exists(CONTROL_SPECS):
        for k, c in json.load(open(CONTROL_SPECS))["trials"].items():
            specs["trials"][k]["conditions"].update(c["conditions"])
    return specs


def build_control(args):
    """Experiment 29 amendment 1: poses and renders of block-there, far-close and far-far -> CONTROL_SPECS. Every other
    file stays as it is."""
    from .room import get_layout
    t0 = time.time()
    trials = json.load(open(SPECS))["trials"]
    out, why, jobs, where = {}, {}, [], []
    for room in dict.fromkeys(s["room"] for s in trials.values()):
        ss = {k: s for k, s in trials.items() if s["room"] == room and s["included"] and not s["overlap_near_far"]}
        if not ss:
            continue
        meta = dict(np.load(f"data/walks-room{room}/meta.npz"))
        order = walk_order(meta)
        pillars = get_layout(room)["pillars"]
        for k, s in ss.items():
            fi = order[s["walk"]]
            c, reason = control_poses(k, s, lambda f: (float(meta["x"][fi[f - 1]]), float(meta["y"][fi[f - 1]])), pillars)
            if c is None:
                why[reason] = why.get(reason, 0) + 1
                continue
            out[k] = c
            for cond, fr in c["conditions"].items():
                jobs.append((room, [tuple(f[1:]) for f in fr]))
                where.append((k, cond))
    print(f"{len(out)} trials, left out {why}; {sum(len(j[1]) for j in jobs)} frames", flush=True)
    with pool(args.workers) as pl:
        for i, ((k, cond), res) in enumerate(zip(where, pl.imap(_render_job, jobs, chunksize=4))):
            for f, (img, free) in zip(out[k]["conditions"][cond], res):
                assert free, (k, cond)
                f += [md5(img), free]
            if (i + 1) % 300 == 0:
                print(f"  {i + 1}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)
    tl = {n: np.array([math.hypot(*c[n]) for c in out.values()]) for n in ("d_block", "d_close", "d_far")}
    build = {"n_included": len(out), "left_out": why, "seconds": round(time.time() - t0),
             "included_per_room": {r: sum(k.startswith(r + "/") for k in out) for r in dict.fromkeys(k.split("/")[0] for k in out)},
             "translation_m": {n: {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())} for n, v in tl.items()}}
    json.dump({"build": build, "trials": out}, open(CONTROL_SPECS, "w"))
    print(json.dumps(build, indent=1))
    print(f"BUILD PASSED -> {CONTROL_SPECS}")


def control_stimulus(args):
    """Experiment 29 amendment 1's stimulus check (CPU): DINOv2 cosines of the swapped frames with the new view and with
    the moved option, the moved frames' distance to the new view, and the place and view agents' answers."""
    import importlib.util
    import torch
    from transformers import AutoModel
    from .frame_cut import DINO
    spec = importlib.util.spec_from_file_location("appearance_cue", "scripts/appearance_cue.py")
    ac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    torch.set_num_threads(args.threads)
    model = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    specs = load_specs()["trials"]
    keys = json.load(open(CONTROL_SPECS))["trials"]
    dino0 = json.load(open(DINO))["trials"]
    res, t0 = {"trials": {}}, time.time()
    with pool(args.workers) as pl:
        for room in dict.fromkeys(k.split("/")[0] for k in keys):
            kk = [k for k in keys if k.startswith(room + "/")]
            probes = np.load(f"data/walks-room{room}/probes.npy", mmap_mode="r")
            frames = np.load(f"data/walks-room{room}/frames.npy", mmap_mode="r")
            order = walk_order(dict(np.load(f"data/walks-room{room}/meta.npz")))
            jobs = [(room, [tuple(f[1:4]) for f in specs[k]["conditions"][c]]) for k in kk for c in CONTROL_SWAPS]
            imgs = [img for r in pl.imap(_render_job, jobs, chunksize=4) for img, _ in r]
            n = 8 * len(CONTROL_SWAPS)
            assert [md5(im) for im in imgs] == [f[4] for k in kk for c in CONTROL_SWAPS for f in specs[k]["conditions"][c]], room
            opt = [np.asarray(frames[order[specs[k]["walk"]][specs[k]["far"] - 1]]) for k in kk]
            emb = ac.dino(model, imgs + [np.asarray(probes[specs[k]["probe"]]) for k in kk] + opt)
            fe, pe, oe = emb[:len(imgs)], emb[len(imgs):len(imgs) + len(kk)], emb[len(imgs) + len(kk):]
            for j, k in enumerate(kk):
                s = specs[k]
                cos0 = dict(zip(dino0[k]["shown"], dino0[k]["cos"]))
                v = fe[j * n:(j + 1) * n].reshape(len(CONTROL_SWAPS), 8, -1)
                t = {}
                for ci, c in enumerate(CONTROL_SWAPS):
                    fr = s["conditions"][c]
                    t[f"cos_{c}"] = [float(x) for x in v[ci] @ pe[j]]
                    t[f"optcos_{c}"] = [float(x) for x in v[ci] @ oe[j]]
                    t[f"dist_{c}"] = [math.dist(f[1:3], s["view"][:2]) for f in fr]
                out = {"place": {}, "view": {}}
                for c in ("clean",) + CONTROL_SWAPS:
                    swapped = {f[0]: (f[1], f[2], i) for i, f in enumerate(s["conditions"][c])} if c != "clean" else {}
                    pos = lambda f: swapped[f][:2] if f in swapped else pose_of(s, f, room)[:2]
                    cos = lambda f: t[f"cos_{c}"][swapped[f][2]] if f in swapped else cos0[f]
                    d = {o: math.hypot(*(np.mean([pos(f) for f in ks], 0) - np.array(s["view"][:2])))
                         for o, ks in (("near", s["near_set"]), ("far", s["far_set"]))}
                    out["place"][c] = bool(d["near"] < d["far"])
                    out["view"][c] = bool(max(cos(f) for f in s["near_set"]) > max(cos(f) for f in s["far_set"]))
                t["agents"] = out
                res["trials"][k] = t
            print(f"{room}: {len(kk)} trials, {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(CONTROL_STIM, "w"))
    print(f"-> {CONTROL_STIM}")


# ---------------------------------------------------------------- stimulus check (CPU): DINOv2, landmarks, agents

def stimulus_check(args, rows):
    """DINOv2-base cosines of every swap frame with the new view and with the near option's frame, landmark sharing
    with the new view, and the answers of two agents whose strategy is known (the positive control for the stimuli)."""
    import importlib.util
    import torch
    from transformers import AutoModel
    from .dissimilar_trials import landmarks
    from .frame_cut import DINO
    from .room import get_layout
    spec = importlib.util.spec_from_file_location("appearance_cue", "scripts/appearance_cue.py")
    ac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    torch.set_num_threads(args.threads)
    model = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    specs = load_specs()["trials"]
    dino0 = json.load(open(DINO))["trials"]
    out, t0 = {"trials": {}}, time.time()
    with pool(args.workers) as pl:
        for room in dict.fromkeys(r["room"] for r in rows):
            rr = [r for r in rows if specs[key(r)]["included"] and r["room"] == room]
            lay = get_layout(room)                  # landmark geometry does not depend on P5005's blob textures
            posters, pillars = lay["posters"](), lay["pillars"]
            probes = np.load(f"data/walks-room{room}/probes.npy", mmap_mode="r")
            frames = np.load(f"data/walks-room{room}/frames.npy", mmap_mode="r")
            order = walk_order(dict(np.load(f"data/walks-room{room}/meta.npz")))
            jobs = [(room, [tuple(f[1:4]) for f in specs[key(r)]["conditions"][c]]) for r in rr for c in SWAPS]
            imgs = [img for res in pl.imap(_render_job, jobs, chunksize=4) for img, _ in res]
            for r, i0 in zip(rr, range(0, len(imgs), 8 * len(SWAPS))):
                s = specs[key(r)]
                got = [md5(im) for im in imgs[i0:i0 + 8 * len(SWAPS)]]
                want = [f[4] for c in SWAPS for f in s["conditions"][c]]
                assert got == want, f"{key(r)}: renders differ from {SPECS}"
            orig = [np.asarray(frames[order[r["walk"]][k - 1]]) for r in rr
                    for k in [r["near"]] + specs[key(r)]["near_set"]]      # the near option, then its 8 frames
            emb = ac.dino(model, imgs + [np.asarray(probes[r["probe"]]) for r in rr] + orig)
            fe, pe, oe = emb[:len(imgs)], emb[len(imgs):len(imgs) + len(rr)], emb[len(imgs) + len(rr):]
            for j, r in enumerate(rr):
                s = specs[key(r)]
                e0 = dino0[key(r)]
                cos0 = dict(zip(e0["shown"], e0["cos"]))
                view_lm = landmarks(*s["view"], pillars=pillars, posters=posters)
                opt = oe[9 * j]                                            # the near option's frame
                t = {"cos_original_near": [cos0[k] for k in s["near_set"]],
                     "optcos_original_near": [float(x) for x in oe[9 * j + 1:9 * j + 9] @ opt],
                     "cos_original_random": [cos0[k] for k in s["random_set"]],
                     "landmark_original_near": [bool(view_lm & landmarks(*pose_of(s, k, room), pillars=pillars,
                                                                              posters=posters)) for k in s["near_set"]]}
                for ci, c in enumerate(SWAPS):
                    v = fe[(j * len(SWAPS) + ci) * 8:(j * len(SWAPS) + ci + 1) * 8] @ pe[j]
                    t[f"cos_{c}"] = [float(x) for x in v]
                    t[f"optcos_{c}"] = [float(x) for x in fe[(j * len(SWAPS) + ci) * 8:(j * len(SWAPS) + ci + 1) * 8] @ opt]
                    t[f"landmark_{c}"] = [bool(view_lm & landmarks(*f[1:4], pillars=pillars, posters=posters))
                                          for f in s["conditions"][c]]
                t["agents"] = agents(r, s, cos0, t)
                out["trials"][key(r)] = t
            print(f"{room}: {len(rr)} trials, {len(imgs)} swap frames, {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(STIM, "w"))
    print(f"-> {STIM}")


_POSE_CACHE = {}


def pose_of(s, k, room):
    """Original (x, y, theta) of shown frame k of trial s."""
    if room not in _POSE_CACHE:
        _POSE_CACHE.clear()
        meta = dict(np.load(f"data/walks-room{room}/meta.npz"))
        _POSE_CACHE[room] = (meta, walk_order(meta))
    meta, order = _POSE_CACHE[room]
    i = order[s["walk"]][k - 1]
    return float(meta["x"][i]), float(meta["y"][i]), float(meta["theta"][i])


def agents(r, s, cos0, t):
    """Answers (near or far) of two agents in each condition, each looking only at the 8 frames around each option
    (experiment 15's near and far sets) as the condition presents them:
      place  the option whose 8 frames' mean position (as rendered) is nearer the new view's true position
      view   the option whose 8 frames include the one most similar to the new view (DINOv2 cosine)"""
    view = s["view"]
    res = {}
    for c in ("clean",) + SWAPS:
        swapped = {f[0]: (f[1], f[2], i) for i, f in enumerate(s["conditions"][c])} if c != "clean" else {}
        pos = lambda k: swapped[k][:2] if k in swapped else pose_of(s, k, s["room"])[:2]
        cos = lambda k: t[f"cos_{c}"][swapped[k][2]] if k in swapped else cos0[k]
        dist = {o: math.hypot(*(np.mean([pos(k) for k in ks], 0) - np.array(view[:2])))
                for o, ks in (("near", s["near_set"]), ("far", s["far_set"]))}
        best = {o: max(cos(k) for k in ks) for o, ks in (("near", s["near_set"]), ("far", s["far_set"]))}
        res[c] = {"place": dist["near"] < dist["far"], "view": best["near"] > best["far"]}
    return res


# ---------------------------------------------------------------- the model run

def out_path(condition, adapter, device):
    name = MODEL.split("/")[-1] + (f"+{os.path.basename(adapter.rstrip('/'))}" if adapter else "")
    return f"{OUT}/{name}-{condition}" + (f"-{device}" if device != "cuda" else "") + ".jsonl"


def prompt(r, s, condition, imgs, proc, data, adapter, device):
    """Experiment 10's prompt for trial r with the condition's frames swapped in: (inputs, shown, lab, spans)."""
    from PIL import Image
    text, images, shown, lab = build_prompt(r, proc, data, adapter, device)
    assert shown == s["shown"], key(r)
    if condition in ALL_SWAPS:
        for f, img in zip(s["conditions"][condition], imgs):
            images[shown.index(f[0])] = Image.fromarray(img)
    inputs = proc(text=[text], images=images, return_tensors="pt", padding=True)
    ids = inputs["input_ids"][0]
    assert sites(ids, proc.tokenizer)["n_images"] == len(images)
    if r["exp10_prompt_tokens"] != len(ids):
        raise SystemExit(f"{key(r)}: {len(ids)} prompt tokens, experiment 10's run had {r['exp10_prompt_tokens']}")
    return inputs, shown, lab, frame_spans(ids, proc.tokenizer, shown, lab)


def rendered(rows, specs, condition, pl, ahead=12):
    """Yield (row, images) in order, rendering each trial's swap frames in the worker pool a few trials ahead and
    checking every render against SPECS' MD5."""
    if condition not in ALL_SWAPS:
        for r in rows:
            yield r, None
        return
    pending = []
    it = iter(rows)

    def submit():
        r = next(it, None)
        if r is not None:
            fr = specs[key(r)]["conditions"][condition]
            pending.append((r, pl.apply_async(_render_job, ((r["room"], [tuple(f[1:4]) for f in fr]),))))
    for _ in range(ahead):
        submit()
    while pending:
        r, job = pending.pop(0)
        submit()
        imgs = [img for img, _ in job.get()]
        want = [f[4] for f in specs[key(r)]["conditions"][condition]]
        got = [md5(im) for im in imgs]
        if got != want:
            raise SystemExit(f"RENDER MISMATCH {key(r)} {condition}: this machine's renders differ from {SPECS}")
        yield r, imgs


def check(rows, adapter, device):
    """Clean vs experiment 10 on the included trials: >= 99% identical answers, else exit 1."""
    path = out_path("clean", adapter, device)
    got = {(x["room"], x["id"]): x for x in map(json.loads, open(path))} if os.path.exists(path) else {}
    missing = [key(r) for r in rows if (r["room"], r["id"]) not in got]
    if missing:
        print(f"CHECK FAILED: {path} lacks {len(missing)} of {len(rows)} trials, e.g. {missing[:3]}")
        raise SystemExit(1)
    bad = [(key(r), r["answer"], got[r["room"], r["id"]]["answer"]) for r in rows
           if got[r["room"], r["id"]]["answer"] != r["answer"]]
    agree = 1 - len(bad) / len(rows)
    by_room = {}
    for r in rows:
        by_room.setdefault(r["room"], []).append(got[r["room"], r["id"]]["answer"] == r["answer"])
    print(f"clean vs experiment 10: {len(rows) - len(bad)}/{len(rows)} answers agree ({agree:.2%}); per room "
          + ", ".join(f"{k} {sum(v)}/{len(v)} ({r_dev(rows, k)})" for k, v in by_room.items()))
    for b in bad[:20]:
        print(f"  {b[0]}: experiment 10 {b[1]}, clean {b[2]}")
    if agree < 0.99:
        print("CHECK FAILED: below 99%; no swap condition may run")
        raise SystemExit(1)
    print("CHECK PASSED")


def r_dev(rows, room):
    return next(r["exp10_device"] for r in rows if r["room"] == room)


def run(args, rows, specs):
    import torch
    conds = list(dict.fromkeys(args.run))
    if "clean" in conds:
        conds.remove("clean")
        conds.insert(0, "clean")
    os.makedirs(OUT, exist_ok=True)
    with pool(args.workers) as pl:            # spawned before the model is loaded
        proc, model = load(args.adapter, args.device)
        cut = AttentionCut(model)
        proc.tokenizer.padding_side = "left"
        special = {proc.tokenizer.pad_token, proc.tokenizer.eos_token, "<|im_end|>", "<|endoftext|>"} - {None}
        dev = torch.mps if args.device == "mps" else torch.cuda
        for condition in conds:
            path = out_path(condition, args.adapter, args.device)
            done = {(x["room"], x["id"]) for x in map(json.loads, open(path))} if os.path.exists(path) else set()
            todo = [r for r in rows if (r["room"], r["id"]) not in done
                    and (condition not in CONTROL_SWAPS or condition in specs[key(r)]["conditions"])]
            print(f"{os.path.basename(path)}: {len(todo)} trials to run ({len(done)} done)", flush=True)
            data, t0 = {}, time.time()
            torch.manual_seed(0)
            for j, (r, imgs) in enumerate(rendered(todo, specs, condition, pl)):
                s = specs[key(r)]
                inputs, shown, lab, spans = prompt(r, s, condition, imgs, proc, data, args.adapter, args.device)
                cut.spans = [spans[k] for k in s["near_set"]] if condition == "near-cut" else []
                inputs = inputs.to(args.device)
                P = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
                cut.spans = []
                g = out[0][P:]
                reply = proc.decode(g, skip_special_tokens=False)
                n_gen = int((g != proc.tokenizer.pad_token_id).sum())
                for sp in special:
                    reply = reply.replace(sp, "")
                ans = {lab[r["a"]]: r["a"], lab[r["b"]]: r["b"]}.get(parse(reply, lab[r["a"]], lab[r["b"]]), -1)
                swapped = s["conditions"].get(condition, [])
                rec = {"room": r["room"], **{k: r[k] for k in ("id", "walk", "probe", "familiar", "near", "far", "a", "b")}}
                rec.update(answer=ans, correct=ans == r["near"], prompt_tokens=P, gen_tokens=n_gen, text=reply[-600:],
                           condition=condition, set=[f[0] for f in swapped] or (s["near_set"] if condition == "near-cut"
                                                                                   else []),
                           poses=[f[1:4] for f in swapped], md5=[f[4] for f in swapped],
                           overlap_near_far=s["overlap_near_far"], exp10_answer=r["answer"])
                with open(path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                if j < 3 or (j + 1) % 48 == 0 or j + 1 == len(todo):
                    mem = (dev.driver_allocated_memory() if args.device == "mps" else dev.max_memory_allocated()) / 2 ** 30
                    el = time.time() - t0
                    print(f"{condition} {j + 1}/{len(todo)}, {el:.0f}s ({el / (j + 1):.2f} s/trial), {key(r)}, "
                          f"prompt {P} tokens, answer {ans} (experiment 10: {r['answer']}), "
                          f"{'driver' if args.device == 'mps' else 'peak'} {mem:.1f} GiB", flush=True)
            print(f"{os.path.basename(path)}: done, {time.time() - t0:.0f}s", flush=True)
            if condition == "clean":
                check(rows, args.adapter, args.device)
        cut.remove()


# ---------------------------------------------------------------- dry run, render check

def dry_run(args, rows, specs):
    """Prompts for the first N included trials in one condition: token counts against experiment 10, which labels
    get new images, and a preview PNG of the first trial's conditions."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.image_processor.size = {"shortest_edge": 256 * 256, "longest_edge": 256 * 256}
    data, t0 = {}, time.time()
    n = min(args.dry_run, len(rows))
    with pool(args.workers) as pl:
        for j, (r, imgs) in enumerate(rendered(rows[:n], specs, args.condition, pl)):
            s = specs[key(r)]
            inputs, shown, lab, spans = prompt(r, s, args.condition, imgs, proc, data, args.adapter, "mps")
            if j == 0:
                fr = s["conditions"].get(args.condition, [])
                print(f"{key(r)} ({args.condition}): {inputs['input_ids'].shape[1]} tokens (experiment 10: "
                      f"{r['exp10_prompt_tokens']}); near {r['near']} at shown position {shown.index(r['near']) + 1}; "
                      f"swapped labels {[lab[f[0]] for f in fr]} at positions {[shown.index(f[0]) + 1 for f in fr]}; "
                      f"translation {s['translation']}; target {s['target']} on {s['target_on']}")
    print(f"{n} prompts in {time.time() - t0:.0f}s; every token count matched experiment 10")
    preview(rows[0], specs[key(rows[0])])


def preview(r, s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = f"data/walks-room{r['room']}"
    frames, probes = np.load(f"{d}/frames.npy", mmap_mode="r"), np.load(f"{d}/probes.npy", mmap_mode="r")
    fi = walk_order(dict(np.load(f"{d}/meta.npz")))[r["walk"]]
    rows_ = [("original", [frames[fi[k - 1]] for k in s["near_set"]])]
    for c in SWAPS[:4]:
        rows_.append((c, [render_one(r["room"], *f[1:4])[0] for f in s["conditions"][c]]))
    fig, axs = plt.subplots(len(rows_) + 1, 8, figsize=(16, 2.1 * (len(rows_) + 1)))
    for ax in axs.flat:
        ax.axis("off")
    axs[0, 0].imshow(probes[r["probe"]])
    axs[0, 0].set_title("new view", fontsize=8)
    axs[0, 1].imshow(frames[fi[r["near"] - 1]])
    axs[0, 1].set_title(f"near option {r['near']}", fontsize=8)
    axs[0, 2].imshow(frames[fi[r["far"] - 1]])
    axs[0, 2].set_title(f"far option {r['far']}", fontsize=8)
    for i, (name, ims) in enumerate(rows_, start=1):
        for j, im in enumerate(ims):
            axs[i, j].imshow(im)
            axs[i, j].set_title(f"{name} {s['near_set'][j]}" if j == 0 else str(s["near_set"][j]), fontsize=8)
    fig.tight_layout()
    path = f"{OUT}/preview_{r['room']}_{r['id']}.png"
    fig.savefig(path, dpi=60)
    print(f"preview -> {path}")


def render_check(args, rows, specs):
    """Render the first --per-room included trials of every room in every swap condition and compare with SPECS."""
    pick = []
    for room in dict.fromkeys(r["room"] for r in rows):
        pick += [r for r in rows if r["room"] == room][:args.per_room]
    n = 0
    with pool(args.workers) as pl:
        for c in SWAPS:
            for r, imgs in rendered(pick, specs, c, pl):
                n += len(imgs)
    print(f"RENDER CHECK PASSED: {n} frames ({len(pick)} trials x {len(SWAPS)} conditions) match {SPECS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="swap poses, renders, identity check -> SPECS (CPU)")
    ap.add_argument("--build-control", action="store_true", help="amendment 1's conditions -> CONTROL_SPECS (CPU)")
    ap.add_argument("--control-stimulus", action="store_true", help="amendment 1's stimulus check (CPU)")
    ap.add_argument("--stimulus-check", action="store_true", help="DINOv2, landmarks and the two agents (CPU)")
    ap.add_argument("--render-check", action="store_true", help="re-render a few trials and compare with SPECS")
    ap.add_argument("--per-room", type=int, default=2)
    ap.add_argument("--dry-run", type=int, metavar="N", help="tokenise N prompts of --condition; no model")
    ap.add_argument("--condition", choices=CONDITIONS, default="here-away")
    ap.add_argument("--run", nargs="+", choices=CONDITIONS, help="conditions to run (clean first, then its check)")
    ap.add_argument("--rooms", nargs="+", default=ROOMS)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--workers", type=int, default=4, help="render processes")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads for DINOv2")
    ap.add_argument("--limit", type=int, help="first N trials only (tests)")
    args = ap.parse_args()
    if args.build_control:
        return build_control(args)
    if args.control_stimulus:
        return control_stimulus(args)
    rows = novel_rows(args.rooms, args.adapter, args.device)
    sets = load_sets(rows)
    if args.build:
        return build(args, rows[:args.limit] if args.limit else rows, sets)
    specs = load_specs()["trials"]
    rows = [r for r in rows if key(r) in specs and specs[key(r)]["included"]]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} included novel trials in {len(args.rooms)} rooms", flush=True)
    if args.stimulus_check:
        return stimulus_check(args, rows)
    if args.render_check:
        return render_check(args, rows, specs)
    if args.dry_run:
        return dry_run(args, rows, specs)
    if not args.run:
        ap.error("nothing to do: --build, --stimulus-check, --render-check, --dry-run or --run")
    run(args, rows, specs)


if __name__ == "__main__":
    main()
