"""Experiment 10's pre-registered analysis (docs/experiments.md): accuracy on novel trials in each
held-out room, the mean over rooms with a 95% CI from a bootstrap over rooms (10,000 resamples), and
the pre-registered contrasts, with trial-level paired sign tests pooled over rooms as a supplement.

Every run is checked against its room's trial manifest (results/export/dissimilar-room<R>/trials.json):
known ids only, the same options, no duplicates. A run that has not answered every trial is reported
as INCOMPLETE with its per-room denominators. Each contrast uses only the novel trials both runs
answered (matched trial ids). The analysis is done on all walks and again on walks 0-15 only, the
trials Opus first ran in every room (see the deviation note in Experiment 10).

    python scripts/heldout_analysis.py
        # prints the tables; writes results/heldout_summary.json when every run is complete, else
        # results/heldout_summary_incomplete.json (leaving the last complete summary untouched)
"""
import json
import math
import os

import numpy as np

ROOMS = [f"P{s}" for s in range(5000, 5010)]
MANIFEST = "results/export/dissimilar-room{r}/trials.json"
RUNS = {   # name: path pattern (room inserted)
    "base, walk": "results/two_choice/Qwen3-VL-8B-Instruct-room{r}-256px-max64-mps.jsonl",
    "base, no walk": "results/two_choice/Qwen3-VL-8B-Instruct-room{r}-256px-nowalk-mps.jsonl",
    "base, shuffled": "results/two_choice/Qwen3-VL-8B-Instruct-room{r}-256px-max64-shuffled-mps.jsonl",
    "LoRA, walk": "results/two_choice/Qwen3-VL-8B-Instruct+qwen8b-proc-room{r}-256px-max64-mps.jsonl",
    "LoRA, no walk": "results/two_choice/Qwen3-VL-8B-Instruct+qwen8b-proc-room{r}-256px-nowalk-mps.jsonl",
    "LoRA, shuffled": "results/two_choice/Qwen3-VL-8B-Instruct+qwen8b-proc-room{r}-256px-max64-shuffled-mps.jsonl",
    "Opus, walk": "results/claude/claude-opus-5-5-room{r}-low.jsonl",
    "Opus, no walk": "results/claude/claude-opus-5-5-room{r}-nowalk-low.jsonl",
}
CONTRASTS = [   # (name, a, b): a minus b; the first is the primary question
    ("Opus walk gain", "Opus, walk", "Opus, no walk"),
    ("LoRA walk gain", "LoRA, walk", "LoRA, no walk"),
    ("LoRA order effect", "LoRA, walk", "LoRA, shuffled"),
    ("LoRA vs base (walk)", "LoRA, walk", "base, walk"),
    ("LoRA vs Opus (walk)", "LoRA, walk", "Opus, walk"),
]
FIELDS = ("walk", "probe", "familiar", "near", "far", "a", "b")
# Experiment 14b, procedural rooms (registered in docs/experiments.md; run 2026-10-24 on the second key). Kept
# out of RUNS so the bootstrap draws of the experiment-10 numbers above stay exactly as they were.
OPUS_SHUFFLED = "results/claude/claude-opus-5-5-room{r}-shuffled-low.jsonl"


def novel(path):
    if not os.path.exists(path):
        return None
    return {r["id"]: bool(r["correct"]) for r in map(json.loads, open(path)) if not r["familiar"]}


def load(path, trials):
    """A run's rows by id, checked against the room's trials; None if the file does not exist."""
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = {r["id"]: r for r in rows}
    bad = [r["id"] for r in rows if r["id"] not in trials or any(r[k] != trials[r["id"]][k] for k in FIELDS)
           or bool(r["correct"]) != (r["answer"] == r["near"])]
    if len(out) != len(rows) or bad:
        raise SystemExit(f"{path}: {len(rows) - len(out)} duplicate ids; {len(bad)} rows not matching the manifest "
                         f"or with correct != (answer == near), e.g. {bad[:3]}")
    return out


def boot_mean(x, rng, n=10_000):
    x = np.asarray(x, float)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test(up, dn):
    n = up + dn
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(max(up, dn), n + 1)) / 2 ** n) if n else 1.0


def analyse(rows, trials, walks=None):
    """The pre-registered tables on the trials of walks < `walks` (all walks if None). rng seed 0 and the
    order of bootstrap calls are those of the original script, so complete data reproduce it exactly."""
    rng = np.random.default_rng(0)
    keep = {r: {i for i, t in trials[r].items() if walks is None or t["walk"] < walks} for r in ROOMS}
    nov = {r: {i for i in keep[r] if not trials[r][i]["familiar"]} for r in ROOMS}
    data = {name: {r: (None if rooms[r] is None else {i: bool(x["correct"]) for i, x in rooms[r].items() if i in nov[r]})
                   for r in ROOMS} for name, rooms in rows.items()}
    acc = {name: {r: (sum(v.values()) / len(v) if v else None) for r, v in rooms.items()} for name, rooms in data.items()}
    fam = {name: {r: (float(np.mean([bool(x["correct"]) for i, x in rooms[r].items() if i in keep[r] and x["familiar"]]))
                      if rooms[r] else None) for r in ROOMS} for name, rooms in rows.items()}
    out = {"walks": "all" if walks is None else f"0-{walks - 1}", "novel_trials": {r: len(nov[r]) for r in ROOMS},
           "novel_answered": {name: {r: len(v or {}) for r, v in rooms.items()} for name, rooms in data.items()},
           "accuracy": acc, "familiar_accuracy": fam, "summary": {}, "contrasts": {}}
    print(f"\n=== walks {out['walks']} ===\nnovel-trial accuracy per room (* = run misses some of these trials)\n"
          + " " * 16 + " ".join(f"{r:>6s}" for r in ROOMS) + "   mean [95% CI over rooms]")
    print(f"{'novel trials':16s}" + " ".join(f"{len(nov[r]):6d}" for r in ROOMS))
    for name in RUNS:
        vals = [acc[name][r] for r in ROOMS]
        if any(v is None for v in vals):
            print(f"{name:16s} missing ({sum(v is not None for v in vals)}/10 rooms)")
            continue
        full = {r: len(data[name][r]) == len(nov[r]) for r in ROOMS}
        m, lo, hi = boot_mean(vals, rng)
        out["summary"][name] = {"mean": m, "ci95": [lo, hi], "complete": all(full.values()),
                                "incomplete_rooms": [r for r in ROOMS if not full[r]]}
        print(f"{name:16s}" + " ".join(f"{v:5.0%}" + (" " if full[r] else "*") for r, v in zip(ROOMS, vals))
              + f"  {m:.1%} [{lo:.1%}, {hi:.1%}]" + ("" if all(full.values()) else "  INCOMPLETE"))
        if not all(full.values()):
            print(f"{'  answered':16s}" + " ".join(f"{len(data[name][r]):6d}" for r in ROOMS))
    print("\ncontrasts (a - b) on matched novel trials, rooms as the unit; pooled trial-level sign test as a supplement")
    for cname, a, b in CONTRASTS:
        if a not in out["summary"] or b not in out["summary"]:
            continue
        ids = {r: sorted(set(data[a][r]) & set(data[b][r])) for r in ROOMS}
        if not all(ids.values()):
            print(f"{cname:22s} skipped: no matched trials in {[r for r in ROOMS if not ids[r]]}")
            continue
        d = [np.mean([data[a][r][i] for i in ids[r]]) - np.mean([data[b][r][i] for i in ids[r]]) for r in ROOMS]
        m, lo, hi = boot_mean(d, rng)
        pos, neg = int(sum(x > 0 for x in d)), int(sum(x < 0 for x in d))
        up = sum(data[a][r][i] and not data[b][r][i] for r in ROOMS for i in ids[r])
        dn = sum(data[b][r][i] and not data[a][r][i] for r in ROOMS for i in ids[r])
        full = all(len(ids[r]) == len(nov[r]) for r in ROOMS)
        out["contrasts"][cname] = {"mean": m, "ci95": [lo, hi], "ci_excludes_0": lo > 0 or hi < 0, "rooms_positive": pos,
                                   "rooms_negative": neg, "room_sign_test_p": sign_test(pos, neg),
                                   "trials_better": up, "trials_worse": dn, "trial_sign_test_p": sign_test(up, dn),
                                   "per_room": dict(zip(ROOMS, map(float, d))),
                                   "matched_novel_trials": {r: len(ids[r]) for r in ROOMS}, "complete": full}
        print(f"{cname:22s} {m:+.1%} [{lo:+.1%}, {hi:+.1%}]  rooms +{pos}/-{neg} (p {sign_test(pos, neg):.2g}); "
              f"trials {up} better / {dn} worse (p {sign_test(up, dn):.2g}); "
              f"{sum(map(len, ids.values()))} matched novel trials" + ("" if full else "  INCOMPLETE"))
    return out


def exp14b(rows, trials):
    """Ordered minus shuffled walk for Opus on novel trials, rooms as the unit (bootstrap over rooms, own rng
    seeded 0); "does not use the order" only if the 95% CI lies inside +/-3 points (the registered margin)."""
    shuf = {r: load(OPUS_SHUFFLED.format(r=r), trials[r]) for r in ROOMS}
    if any(v is None or len(v) != len(trials[r]) for r, v in shuf.items()):
        print("\nexperiment 14b: shuffled Opus runs incomplete; skipped")
        return None
    rng = np.random.default_rng(0)
    nov = {r: [i for i, t in trials[r].items() if not t["familiar"]] for r in ROOMS}
    c = lambda run, r, i: bool(run[r][i]["correct"])
    walk, nowalk = rows["Opus, walk"], rows["Opus, no walk"]
    acc = [np.mean([c(shuf, r, i) for i in nov[r]]) for r in ROOMS]
    out = {"shuffled": dict(zip(["mean", "lo", "hi"], boot_mean(acc, rng))), "per_room_shuffled": dict(zip(ROOMS, map(float, acc)))}
    for name, a, b in (("ordered - shuffled", walk, shuf), ("shuffled - no walk", shuf, nowalk)):
        d = [np.mean([c(a, r, i) for i in nov[r]]) - np.mean([c(b, r, i) for i in nov[r]]) for r in ROOMS]
        m, lo, hi = boot_mean(d, rng)
        pos, neg = int(sum(x > 0 for x in d)), int(sum(x < 0 for x in d))
        up = sum(c(a, r, i) and not c(b, r, i) for r in ROOMS for i in nov[r])
        dn = sum(c(b, r, i) and not c(a, r, i) for r in ROOMS for i in nov[r])
        out[name] = {"mean": m, "ci95": [lo, hi], "rooms_positive": pos, "rooms_negative": neg,
                     "room_sign_test_p": sign_test(pos, neg), "trials_better": up, "trials_worse": dn,
                     "trial_sign_test_p": sign_test(up, dn), "per_room": dict(zip(ROOMS, map(float, d)))}
    o = out["ordered - shuffled"]
    o["within_3_points"] = -0.03 < o["ci95"][0] and o["ci95"][1] < 0.03
    print(f"\nexperiment 14b (procedural rooms): Opus shuffled {out['shuffled']['mean']:.1%} "
          f"[{out['shuffled']['lo']:.1%}, {out['shuffled']['hi']:.1%}]")
    for name in ("ordered - shuffled", "shuffled - no walk"):
        v = out[name]
        print(f"  {name:20s} {v['mean']:+.1%} [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]  rooms +{v['rooms_positive']}/-"
              f"{v['rooms_negative']} (p {v['room_sign_test_p']:.2g}); trials {v['trials_better']}/{v['trials_worse']} "
              f"(p {v['trial_sign_test_p']:.2g})")
    print(f"  CI of ordered - shuffled inside +/-3 points: {o['within_3_points']}")
    return out


def main():
    trials = {r: {t["id"]: t for t in json.load(open(MANIFEST.format(r=r)))} for r in ROOMS}
    rows = {name: {r: load(p.format(r=r), trials[r]) for r in ROOMS} for name, p in RUNS.items()}
    comp, incomplete = {}, []
    for name in RUNS:
        comp[name] = {}
        for r in ROOMS:
            got, tr = rows[name][r], trials[r]
            if got is None:
                comp[name][r] = {"file": False, "complete": False}
                incomplete.append(f"{name}, {r}: no file")
                continue
            miss = sorted(set(tr) - set(got))
            comp[name][r] = {"file": True, "trials": len(got), "of": len(tr),
                             "novel": sum(not x["familiar"] for x in got.values()), "novel_of": sum(not t["familiar"] for t in tr.values()),
                             "walks_answered": sorted({x["walk"] for x in got.values()}), "walks_in_manifest": len({t["walk"] for t in tr.values()}),
                             "missing_ids": len(miss), "complete": not miss}
            if miss:
                c = comp[name][r]
                incomplete.append(f"{name}, {r}: {c['trials']}/{c['of']} trials, {c['novel']}/{c['novel_of']} novel, "
                                  f"walks {min(c['walks_answered'])}-{max(c['walks_answered'])} of 0-{c['walks_in_manifest'] - 1}")
    print("completeness against the trial manifests: novel trials answered per room (* = run misses trials)")
    print(" " * 16 + " ".join(f"{r:>6s}" for r in ROOMS))
    print(f"{'manifest':16s}" + " ".join(f"{sum(not t['familiar'] for t in trials[r].values()):6d}" for r in ROOMS))
    for name in RUNS:
        print(f"{name:16s}" + " ".join(f"{comp[name][r].get('novel', 0):5d}" + (" " if comp[name][r]["complete"] else "*") for r in ROOMS))
    if incomplete:
        print("\n" + "!" * 100 + f"\nINCOMPLETE: {len(incomplete)} run-rooms have not answered every trial in the manifest.\n"
              "The all-walks numbers below use what has been answered; contrasts use matched trial ids.\n  "
              + "\n  ".join(incomplete) + "\n" + "!" * 100)
    out = {"rooms": ROOMS, "all_runs_complete": not incomplete, "incomplete_runs": incomplete, "completeness": comp}
    out.update(analyse(rows, trials))
    out["walks_0_15"] = analyse(rows, trials, walks=16)
    out["exp14b_procedural"] = exp14b(rows, trials)
    path = "results/heldout_summary.json" if not incomplete else "results/heldout_summary_incomplete.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n-> {path}" + ("" if not incomplete else "  (results/heldout_summary.json left as it was: some runs are incomplete)"))


if __name__ == "__main__":
    main()
