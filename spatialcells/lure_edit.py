"""Experiment 14c (pre-registered in docs/experiments.md): the landmark lure, edited. Builds the rooms and
trial sets; scripts/mac/run_lure_edit.sh runs them and scripts/lure_edit_analysis.py scores them.

    python -m spatialcells.lure_edit [--workers 8]

The lure trials are room A's 50 original novel trials (results/export/claude/trials.json) where only the
far option shares a landmark with the new view (results/landmark_shares.json, checked here against
dissimilar_trials.landmarks()). A trial qualifies if its shared landmarks are all posters: anything on a
wall that landmarks() names by wall and index (the six pictures, the door S1 and the window E1), not a
pillar. Its edited room leaves the shared posters out (room.Room(remove=...): the wall's own texture shows
in their place). Its control room leaves out instead the first poster in room.py's order (N0 N1 E0 E1 S0 S1
W0 W1) that none of the trial's three views (new view, near and far option) shows (by landmarks()); a
trial whose three views show all eight has no control. Rooms are deduplicated by the set of posters left
out, so one room serves every trial whose edit or control it is. In each room the walks of its trials are
re-rendered at their stored poses (data/walks/meta.npz), and the trials' new views at theirs.

Writes, for each room R = Ano<posters> (e.g. AnoS1W1):
  data/walks-roomR/  frames.npy and meta.npz (those walks only; original walk ids and steps, orig = row in
                     data/walks), probes.npy and probes_meta.npz (the trials' new views; orig = index in
                     data/walks/probes.npy)
  results/export/dissimilar-roomR/trials.json  its trials as in the original file, plus probe = index into
                     the room's probes.npy, orig_probe, role (edited or control), removed and shared
and, for the unedited trials, results/export/dissimilar-roomAlure/trials.json (role original) with
data/walks-roomAlure a link to data/walks; so two_choice.py and claude_test.py run all three kinds as
--trials room<R>. Verification, in results/lure_edit_manifest.json, with each room's views compared with
the originals: a view in which no ray of the renderer's 512 x 512 grid hits a removed poster must be byte-
identical, and in any other view only pixels covered by such rays may change (both asserted); per trial,
the removed poster's footprint (pixels) and the pixels changed in the new view, near and far option, and
how many of its walk's frames changed. results/preview/lure_edit/<room>.png shows each edited trial's new
view and far option before and after.
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from .dissimilar_trials import landmarks
from .room import CAM_H, H, IMG, SS, L, Room, _posters

TRIALS = "results/export/claude/trials.json"
SHARES = "results/landmark_shares.json"
MANIFEST = "results/lure_edit_manifest.json"
ORDER = [f"{w}{i}" for w, ps in _posters().items() for i in range(len(ps))]     # N0 N1 E0 E1 S0 S1 W0 W1

_edit = _orig = _remove = None


def poster_mask(room, x, y, theta, names):
    """(IMG, IMG) bool: pixels with a ray of room.render's supersampling grid whose first hit is one of the
    named posters (same ray casting as Room.render)."""
    f = np.array([np.cos(theta), np.sin(theta), 0.0])
    r = np.array([np.sin(theta), -np.cos(theta), 0.0])
    d = (f[None, None] + room.u[..., None] * r + room.v[..., None] * np.array([0, 0, 1.0])).reshape(-1, 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    dx, dy, dz = d.T
    with np.errstate(divide="ignore", invalid="ignore"):
        cands = [np.where(dz < 0, -CAM_H / dz, np.inf), np.where(dz > 0, (H - CAM_H) / dz, np.inf),
                 np.where(dx < 0, -x / dx, np.inf), np.where(dx > 0, (L - x) / dx, np.inf),
                 np.where(dy < 0, -y / dy, np.inf), np.where(dy > 0, (L - y) / dy, np.inf)]
        for px, py, pr, ph, _ in room.pillars:
            ox, oy = x - px, y - py
            a2, b, c = dx * dx + dy * dy, 2 * (dx * ox + dy * oy), ox * ox + oy * oy - pr * pr
            disc = b * b - 4 * a2 * c
            tp = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a2)
            zp = CAM_H + tp * dz
            cands.append(np.where((disc > 0) & (tp > 0) & (zp >= 0) & (zp <= ph), tp, np.inf))
    T = np.stack(cands)
    first, t = T.argmin(0), T.min(0)
    hit = np.zeros(len(d), bool)
    for k, wall in [(2, "W"), (3, "E"), (4, "S"), (5, "N")]:
        m = np.flatnonzero(first == k)
        s = (y + t[m] * dy[m]) if wall in "EW" else (x + t[m] * dx[m])
        z = CAM_H + t[m] * dz[m]
        for i, (s0, s1, z0, z1, _) in enumerate(room.posters[wall]):
            if f"{wall}{i}" in names:
                hit[m[(s >= s0) & (s <= s1) & (z >= z0) & (z <= z1)]] = True
    return hit.reshape(IMG, SS, IMG, SS).any((1, 3))


def _init(remove):
    global _edit, _orig, _remove
    _edit, _orig, _remove = Room("A", remove), Room("A"), remove


def _work(pose):
    return _edit.render(*pose), poster_mask(_orig, *pose, _remove)


def walk_order(meta):
    out = {}
    for w in np.unique(meta["walk"]):
        i = np.flatnonzero(meta["walk"] == w)
        out[int(w)] = i[np.argsort(meta["step"][i])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    meta, pmeta = dict(np.load("data/walks/meta.npz")), dict(np.load("data/walks/probes_meta.npz"))
    frames = np.load("data/walks/frames.npy", mmap_mode="r")
    probes = np.load("data/walks/probes.npy", mmap_mode="r")
    order = walk_order(meta)
    shares = json.load(open(SHARES))
    trials = {t["id"]: t for t in json.load(open(TRIALS)) if not t["familiar"]}
    fpose = lambda t, k: (meta["x"][order[t["walk"]][k - 1]], meta["y"][order[t["walk"]][k - 1]],
                          meta["theta"][order[t["walk"]][k - 1]])
    ppose = lambda p: (pmeta["x"][p], pmeta["y"][p], pmeta["theta"][p])

    # the lure trials and their edits
    lure, plan = {}, {}
    for i, t in trials.items():
        seen_p, seen_n, seen_f = landmarks(*ppose(t["probe"])), landmarks(*fpose(t, t["near"])), landmarks(*fpose(t, t["far"]))
        s = shares[str(t["probe"])]
        assert sorted(seen_p & seen_n) == s["near_shares"] and sorted(seen_p & seen_f) == s["far_shares"], i
        if not (s["far_shares"] and not s["near_shares"]):
            continue
        lure[i] = s["far_shares"]
        if any(x.startswith("P") for x in s["far_shares"]):
            continue
        ctrl = next((x for x in ORDER if x not in seen_p | seen_n | seen_f), None)
        plan[i] = {"walk": t["walk"], "shared": sorted(s["far_shares"], key=ORDER.index), "control": ctrl,
                   "door_or_window": bool({"S1", "E1"} & set(s["far_shares"])), "control_room": None}
    rooms = {}
    for i, p in plan.items():
        for role, removed in (("edited", tuple(p["shared"])), ("control", (p["control"],) if p["control"] else None)):
            if removed is None:
                continue
            name = "roomAno" + "".join(removed)
            rooms.setdefault(name, {"removed": list(removed), "trials": []})["trials"].append((i, role))
            p[f"{role}_room"] = name
    n_frames = sum(len({trials[i]["walk"] for i, _ in r["trials"]}) for r in rooms.values()) * 256
    print(f"{len(lure)} lure trials; {len(plan)} qualify (shared landmarks all posters; {sum(not p['door_or_window'] for p in plan.values())} "
          f"without the door or window), {sum(p['control'] is None for p in plan.values())} of them without a control poster; "
          f"{len(rooms)} rooms, {n_frames} walk frames to render", flush=True)

    manifest = {"poster_order": ORDER, "lure_trials": len(lure), "qualifying": sorted(plan),
                "excluded_pillar": {i: s for i, s in sorted(lure.items()) if i not in plan},
                "no_control": sorted(i for i, p in plan.items() if p["control"] is None),
                "trials": dict(sorted(plan.items())), "rooms": {}}
    os.makedirs("results/preview/lure_edit", exist_ok=True)
    for name, room in sorted(rooms.items()):
        rem = room["removed"]
        tr = sorted(room["trials"], key=lambda x: (trials[x[0]]["walk"], trials[x[0]]["probe"]))
        walks = sorted({trials[i]["walk"] for i, _ in tr})
        rows = np.flatnonzero(np.isin(meta["walk"], walks))                    # original row order
        pidx = sorted({trials[i]["probe"] for i, _ in tr})
        poses = [(meta["x"][j], meta["y"][j], meta["theta"][j]) for j in rows] + [ppose(p) for p in pidx]
        with Pool(args.workers, initializer=_init, initargs=(tuple(rem),)) as pool:
            out = pool.map(_work, poses, chunksize=16)
        new = np.stack([im for im, _ in out])
        mask = np.stack([m for _, m in out])
        old = np.concatenate([frames[rows], probes[pidx]])
        changed = (new != old).any(-1)
        has_rays = mask.any((1, 2))
        assert not (changed & ~mask).any(), f"{name}: pixels changed outside the removed posters' footprint"
        assert not changed[~has_rays].any(), f"{name}: a view without the removed posters changed"
        d = f"data/walks-{name}"
        os.makedirs(d, exist_ok=True)
        np.save(f"{d}/frames.npy", new[:len(rows)])
        np.savez(f"{d}/meta.npz", walk=meta["walk"][rows], step=meta["step"][rows], x=meta["x"][rows],
                 y=meta["y"][rows], theta=meta["theta"][rows], orig=rows)
        np.save(f"{d}/probes.npy", new[len(rows):])
        np.savez(f"{d}/probes_meta.npz", walk=pmeta["walk"][pidx], idx=pmeta["idx"][pidx], x=pmeta["x"][pidx],
                 y=pmeta["y"][pidx], theta=pmeta["theta"][pidx], orig=np.array(pidx))
        local = {j: k for k, j in enumerate(rows)}
        views, at, out_trials = {}, {}, []
        for i, role in tr:
            t = trials[i]
            fi = order[t["walk"]]
            v = {"new view": len(rows) + pidx.index(t["probe"]), "near": local[fi[t["near"] - 1]],
                 "far": local[fi[t["far"] - 1]]}
            at[i] = v
            views[i] = {"role": role, **{k: {"footprint_px": int(mask[j].sum()), "changed_px": int(changed[j].sum())}
                                         for k, j in v.items()},
                        "walk_frames_changed": int(changed[[local[j] for j in fi]].any((1, 2)).sum())}
            if role == "edited":                     # the shared posters are gone from the new view and far option
                assert views[i]["new view"]["changed_px"] and views[i]["far"]["changed_px"], (name, i)
            out_trials.append({**t, "probe": pidx.index(t["probe"]), "orig_probe": t["probe"], "role": role,
                               "removed": rem, "shared": plan[i]["shared"]})
        os.makedirs(f"results/export/dissimilar-{name}", exist_ok=True)
        json.dump(out_trials, open(f"results/export/dissimilar-{name}/trials.json", "w"), indent=1)
        manifest["rooms"][name] = {
            "removed": rem, "walks": walks, "edited_trials": [i for i, r in tr if r == "edited"],
            "control_trials": [i for i, r in tr if r == "control"], "views": len(new),
            "views_with_poster_rays": int(has_rays.sum()), "of_them_changed": int(changed[has_rays].any((1, 2)).sum()),
            "views_without_rays_identical": int((~has_rays).sum()), "pixels_changed_outside_footprint": 0,
            "trial_views": views}
        ed = [i for i, r in tr if r == "edited"]
        print(f"{name}: removed {rem}; walks {walks}; {len(ed)} edited, {len(tr) - len(ed)} control trials; "
              f"{has_rays.sum()} of {len(new)} views show the removed posters ({changed[has_rays].any((1, 2)).sum()} changed), "
              f"the other {(~has_rays).sum()} byte-identical", flush=True)
        if ed:
            preview(name, ed, [old[at[i][k]] for i in ed for k in ("new view", "far")],
                    [new[at[i][k]] for i in ed for k in ("new view", "far")])

    # the unedited trials, run from data/walks itself
    orig = [{**trials[i], "role": "original", "removed": [], "shared": plan[i]["shared"]}
            for i in sorted(plan, key=lambda i: (trials[i]["walk"], trials[i]["probe"]))]
    os.makedirs("results/export/dissimilar-roomAlure", exist_ok=True)
    json.dump(orig, open("results/export/dissimilar-roomAlure/trials.json", "w"), indent=1)
    if not os.path.lexists("data/walks-roomAlure"):
        os.symlink("walks", "data/walks-roomAlure")
    json.dump(manifest, open(MANIFEST, "w"), indent=1)
    print(f"-> {MANIFEST}, results/export/dissimilar-roomA*/trials.json, data/walks-roomA*")


def preview(name, ids, before, after):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(ids), 4, figsize=(8, 2.1 * len(ids)), squeeze=False)
    for r, i in enumerate(ids):
        for c, (im, lab) in enumerate(zip((before[2 * r], after[2 * r], before[2 * r + 1], after[2 * r + 1]),
                                          ("new view", "new view, edited", "far option", "far option, edited"))):
            axs[r, c].imshow(im)
            axs[r, c].set_title(f"{i}: {lab}" if c == 0 else lab, fontsize=7)
            axs[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(f"results/preview/lure_edit/{name}.png", dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    main()
