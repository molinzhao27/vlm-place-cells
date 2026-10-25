"""Experiment 15's pre-registered analysis (docs/experiments.md): which walk frames the LoRA model reads.

The LoRA model (seed 0) on the 1,299 novel trials of P5000-P5009 with 8 shown walk frames cut from attention
(spatialcells/frame_cut.py). As experiment 13a (scripts/exp13_analysis.py): novel-trial accuracy per room
under each condition, means over rooms with 95% bootstrap CIs (10,000 resamples, seed 0), paired per-room
differences with bootstrap CIs and room-level sign tests, and trial-level pooled sign tests. Primary
contrasts: near neighbours - random, look-alikes - random, near neighbours - far neighbours; each is
repeated on the trials without the relevant overlap (near with look-alikes for the first two, near with far
for the third). Each cut minus clean is reported alongside. Secondary, on P5000-P5002: near neighbours and
look-alikes cut in layers 1-18 or 19-36 only, each against clean and against the all-layer cut of the same
set there. Also the overlap counts and the clean run's agreement with experiment 10's answers. A condition
whose file is missing or incomplete is listed under "incomplete" and left out.

    python scripts/frame_cut_analysis.py      # -> results/frame_cut_summary.json
"""
import argparse
import json
import math
import os

import numpy as np

ROOMS = [f"P{s}" for s in range(5000, 5010)]
WINDOW_ROOMS = ["P5000", "P5001", "P5002"]
NAME = "Qwen3-VL-8B-Instruct+qwen8b-proc"
EXP10 = "results/two_choice/Qwen3-VL-8B-Instruct+qwen8b-proc-room{r}-256px-max64-mps.jsonl"
PRIMARY = ["clean", "near", "far", "lookalike", "random"]
WINDOWS = [f"{s}-layers{w}" for s in ("near", "lookalike") for w in ("1-18", "19-36")]
CONTRASTS = [("near", "random", "overlap_near_lookalike"), ("lookalike", "random", "overlap_near_lookalike"),
             ("near", "far", "overlap_near_far")]
FIELDS = ("walk", "probe", "familiar", "near", "far", "a", "b")


def manifest(r):
    return {t["id"]: t for t in json.load(open(f"results/export/dissimilar-room{r}/trials.json")) if not t["familiar"]}


def load(path, trials, rooms):
    """(room, id) -> row, if the file covers every novel trial of the rooms; else None."""
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = {(x["room"], x["id"]): x for x in rows}
    bad = [k for k, x in out.items() if x["id"] not in trials[x["room"]]
           or any(x[f] != trials[x["room"]][x["id"]].get(f, x[f]) for f in FIELDS)
           or bool(x["correct"]) != (x["answer"] == x["near"])]
    assert len(out) == len(rows) and not bad, (path, len(rows), bad[:3])
    want = {(r, i) for r in rooms for i in trials[r]}
    return {k: out[k] for k in want} if want <= set(out) else None


def sign_p(up, dn):
    n = up + dn
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(max(up, dn), n + 1)) / 2 ** n) if n else 1.0


def boot(x, rng, n=10_000):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return {"mean": float(x.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}


def room_acc(d, rooms, ids):
    return [float(np.mean([d[r, i]["correct"] for i in ids[r]])) if ids[r] else float("nan") for r in rooms]


def contrast(da, db, rooms, ids, rng):
    """a - b: per-room paired accuracy differences (rooms with no trials left dropped), sign tests."""
    keep = [r for r in rooms if ids[r]]
    d = np.array(room_acc(da, keep, ids)) - np.array(room_acc(db, keep, ids))
    up = sum(da[r, i]["correct"] and not db[r, i]["correct"] for r in keep for i in ids[r])
    dn = sum(db[r, i]["correct"] and not da[r, i]["correct"] for r in keep for i in ids[r])
    return {**boot(d, rng), "n_trials": sum(len(ids[r]) for r in keep), "n_rooms": len(keep),
            "per_room": dict(zip(keep, map(float, d))),
            "rooms_up": int((d > 0).sum()), "rooms_down": int((d < 0).sum()),
            "room_sign_p": sign_p(int((d > 0).sum()), int((d < 0).sum())),
            "trials_better": int(up), "trials_worse": int(dn), "trial_sign_p": sign_p(up, dn)}


def conditions(names, data, rooms, ids, rng):
    out = {}
    for n in names:
        if data.get(n) is None:
            continue
        acc = room_acc(data[n], rooms, ids)
        ans = [data[n][r, i] for r in rooms for i in ids[r]]
        out[n] = {**boot(acc, rng), "per_room": dict(zip(rooms, acc)),
                  "pooled": float(np.mean([x["correct"] for x in ans])),
                  "first_listed": float(np.mean([x["answer"] == x["a"] for x in ans])),
                  "parsed": float(np.mean([x["answer"] in (x["a"], x["b"]) for x in ans]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/frame_cut")
    ap.add_argument("--out", default="results/frame_cut_summary.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    trials = {r: manifest(r) for r in ROOMS}
    path = lambda c: f"{args.dir}/{NAME}-{c}-mps.jsonl"
    data = {c: load(path(c), trials, ROOMS) for c in PRIMARY}
    win = {c: load(path(c), trials, WINDOW_ROOMS) for c in WINDOWS}
    win.update({c: data[c] for c in ("clean", "near", "lookalike")})
    ids = {r: sorted(trials[r]) for r in ROOMS}
    out = {"n_novel": sum(len(v) for v in ids.values()), "rooms": ROOMS,
           "incomplete": [c for c in PRIMARY if data[c] is None] + [c for c in WINDOWS if win[c] is None]}

    # overlaps (identical in every condition's rows; taken from the first complete one)
    ref = next((data[c] for c in PRIMARY if data[c] is not None), None)
    if ref is not None:
        out["overlaps"] = {}
        for o in ("overlap_near_lookalike", "overlap_near_far"):
            v = np.array([ref[r, i][o] for r in ROOMS for i in ids[r]])
            out["overlaps"][o] = {"trials_with_overlap": int((v > 0).sum()), "trials_without": int((v == 0).sum()),
                                  "mean_frames": float(v.mean()),
                                  "count_by_size": {int(k): int((v == k).sum()) for k in np.unique(v)}}

    # clean vs experiment 10
    if data["clean"] is not None:
        e10 = {(r, x["id"]): x for r in ROOMS for x in map(json.loads, open(EXP10.format(r=r)))}
        keys = [(r, i) for r in ROOMS for i in ids[r]]
        agree = [data["clean"][k]["answer"] == e10[k]["answer"] for k in keys]
        out["clean_vs_exp10"] = {"agree": int(sum(agree)), "n": len(keys), "share": float(np.mean(agree)),
                                 "per_room": {r: float(np.mean([data["clean"][r, i]["answer"] == e10[r, i]["answer"]
                                                                for i in ids[r]])) for r in ROOMS},
                                 "exp10_accuracy_pooled": float(np.mean([e10[k]["correct"] for k in keys])),
                                 "clean_accuracy_pooled": float(np.mean([data["clean"][k]["correct"] for k in keys])),
                                 "mismatches": [f"{r}/{i}" for (r, i), a in zip(keys, agree) if not a]}

    out["conditions"] = conditions(PRIMARY, data, ROOMS, ids, rng)
    out["contrasts"], out["contrasts_without_overlap"], out["cut_minus_clean"] = {}, {}, {}
    for a, b, o in CONTRASTS:
        if data[a] is None or data[b] is None:
            continue
        out["contrasts"][f"{a} - {b}"] = contrast(data[a], data[b], ROOMS, ids, rng)
        sub = {r: [i for i in ids[r] if data[a][r, i][o] == 0] for r in ROOMS}
        out["contrasts_without_overlap"][f"{a} - {b}"] = {"without": o, **contrast(data[a], data[b], ROOMS, sub, rng)}
    for c in ("near", "far", "lookalike", "random"):
        if data[c] is not None and data["clean"] is not None:
            out["cut_minus_clean"][f"{c} - clean"] = contrast(data[c], data["clean"], ROOMS, ids, rng)

    # secondary: the layer windows on P5000-P5002
    wids = {r: ids[r] for r in WINDOW_ROOMS}
    w = {"rooms": WINDOW_ROOMS, "n_novel": sum(len(v) for v in wids.values()),
         "conditions": conditions(["clean", "near", "lookalike"] + WINDOWS, win, WINDOW_ROOMS, wids, rng),
         "contrasts": {}}
    for c in WINDOWS:
        s = c.split("-")[0]
        for ref_c in ("clean", s):
            if win[c] is not None and win[ref_c] is not None:
                w["contrasts"][f"{c} - {ref_c}{' (all layers)' if ref_c == s else ''}"] = \
                    contrast(win[c], win[ref_c], WINDOW_ROOMS, wids, rng)
    out["windows"] = w
    json.dump(out, open(args.out, "w"), indent=1)

    print(f"Experiment 15: LoRA, novel trials of P5000-P5009 ({out['n_novel']}); mean over rooms [95% CI]")
    if out["incomplete"]:
        print(f"  incomplete or missing (left out): {', '.join(out['incomplete'])}")
    if "clean_vs_exp10" in out:
        cv = out["clean_vs_exp10"]
        print(f"  clean vs experiment 10: {cv['agree']}/{cv['n']} answers agree ({100 * cv['share']:.1f}%)")
    for o, v in out.get("overlaps", {}).items():
        print(f"  {o}: {v['trials_with_overlap']} trials overlap, {v['trials_without']} don't; by size {v['count_by_size']}")
    for n, v in out["conditions"].items():
        print(f"  {n:10s} {100 * v['mean']:5.1f}% [{100 * v['ci95'][0]:.1f}, {100 * v['ci95'][1]:.1f}]"
              f"   pooled {100 * v['pooled']:.1f}%   first-listed {100 * v['first_listed']:.0f}%")
    for title, block in (("contrasts", out["contrasts"]), ("without overlap", out["contrasts_without_overlap"]),
                         ("cut - clean", out["cut_minus_clean"]), ("windows, P5000-P5002", w["contrasts"])):
        if block:
            print(f"  {title}:")
        for n, v in block.items():
            print(f"    {n:34s} {100 * v['mean']:+5.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]"
                  f"  rooms {v['rooms_up']}/{v['rooms_down']} (p {v['room_sign_p']:.3g})"
                  f"  trials {v['trials_better']}/{v['trials_worse']} (p {v['trial_sign_p']:.2g}), n {v['n_trials']}")
    for n, v in w["conditions"].items():
        print(f"  P5000-P5002 {n:18s} {100 * v['mean']:5.1f}%")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
