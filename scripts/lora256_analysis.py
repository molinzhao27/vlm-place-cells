"""Experiment 24's registered analysis (docs/experiments.md): the four-epoch LoRA runs (experiment 9c, runs 3-7)
with the full 256-frame walk that Opus 5.5 saw, on the novel trials of the ten held-out procedural rooms
(experiment 10), against Opus with that walk, and against the same runs' own 64-frame answers.

    .venv-claude/bin/python scripts/lora256_analysis.py            # -> results/lora_256frames.json
    .venv-claude/bin/python scripts/lora256_analysis.py --test     # stand-in check (below) -> scratch file

Files (results/two_choice/Qwen3-VL-8B-Instruct+qwen8b-proc-e4-s<k>-room<R>-256px...). Amendment 1 moved the runs from
the Mac to vast-3 (CUDA) before any of them started:
- 256-frame walk, vast-3: ...-256px.jsonl                    (scripts/run_lora256_cuda.sh; novel trials only)
- 64-frame walk (exp 9c): ...-256px-max64.jsonl (CUDA, vast-3) or ...-256px-max64-mps.jsonl (P5005); one must exist
- 64-frame CUDA rerun:    ...-256px-max64-parity-cuda.jsonl   (P5005 novel trials, the one room 9c ran on the Mac)
- Opus 5.5, low effort, 256-frame walk: results/claude/claude-opus-5-5-room<R>-low.jsonl
Every file is checked against its room's manifest (known ids, same options, no duplicates, correct ==
(answer == near)). Unparsed answers count as wrong. Units, as registered:
- per run and per room, accuracy on the room's novel trials; the summary over rooms is the mean with a 95%
  bootstrap CI over rooms (10,000 resamples, a fresh generator seeded 0 for each CI);
- the recipe (all runs complete in every room): per room, the mean over runs, then the same bootstrap;
- pooled trial-level sign tests are a supplement.
A run counts only once its 256-frame files cover every novel trial in all ten rooms. With fewer than five, the
summary is marked "interim".

Verdict on the primary contrast (LoRA at 256 frames minus Opus at 256 frames, recipe mean over rooms):
"survives" if the mean is at least 10 points and the CI's lower end is above 0; "reduced" if the lower end is
above 0 but the mean is under 10 points; "gone" if the CI includes 0; "reversed" if its upper end is below 0.

The 256-minus-64 contrast is made on one device: the CUDA 64-frame answers of experiment 9c in nine rooms and the
CUDA rerun in P5005 (the same contrast against 9c's files, MPS in P5005, is a supplement). Parity is the share of
P5005's novel trials where the CUDA rerun and the Mac's 64-frame answers are identical, pooled over runs.

--test runs the whole analysis with the existing 64-frame files standing in for the 256-frame ones and P5005's MPS
64-frame files for the CUDA reruns, so it must reproduce experiment 9c's numbers (results/
seed_scenes_summary.json, "10": LoRA walk 97.1% recipe mean, LoRA minus Opus +27.0 to +28.2 per run),
a 256-minus-64 difference of exactly 0 and parity of 1.0.
"""
import argparse
import json
import math
import os

import numpy as np

EXP = "24"
ROOMS = [f"P{s}" for s in range(5000, 5010)]
RUNS = (5, 7, 3, 4, 6)                      # the order the jobs run in: the two shuffle extremes first
PARITY_ROOMS = ("P5005",)                   # the room whose 64-frame answers came from the Mac (MPS)
STEM = "results/two_choice/Qwen3-VL-8B-Instruct+qwen8b-proc-e4-s{k}-room{r}-256px"
OPUS = "results/claude/claude-opus-5-5-room{r}-low.jsonl"
MANIFEST = "results/export/dissimilar-room{r}/trials.json"
OUT = "results/lora_256frames.json"
FIELDS = ("walk", "probe", "familiar", "near", "far", "a", "b")
RULE_POINTS = 0.10                          # the registered margin: 10 points
PARITY_AGREEMENT = 0.98                     # the registered parity threshold (answers identical)


def paths(k, r, test=False):
    """(256-frame file, 64-frame file and its platform, CUDA 64-frame rerun) for run k in room r."""
    stem = STEM.format(k=k, r=r)
    found = [(stem + "-max64" + t + ".jsonl", plat) for t, plat in (("", "CUDA"), ("-mps", "MPS"))
             if os.path.exists(stem + "-max64" + t + ".jsonl")]
    assert len(found) <= 1, f"both CUDA and MPS 64-frame runs of {stem}"
    p64, plat64 = found[0] if found else (None, None)
    if test:                                 # stand-ins: 64-frame files for the 256-frame files and the CUDA reruns
        return p64, (p64, plat64), (p64 if plat64 == "MPS" else None)
    return stem + ".jsonl", (p64, plat64), stem + "-max64-parity-cuda.jsonl"


def load(path, man):
    """id -> row for the novel trials in a run file, checked against the manifest; None if no file."""
    if path is None or not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = {r["id"]: r for r in rows}
    bad = [r["id"] for r in rows if r["id"] not in man or any(r[f] != man[r["id"]][f] for f in FIELDS)
           or bool(r["correct"]) != (r["answer"] == r["near"])]
    if len(out) != len(rows) or bad:
        raise SystemExit(f"{path}: {len(rows) - len(out)} duplicate ids; {len(bad)} rows not matching the manifest "
                         f"or with correct != (answer == near), e.g. {bad[:3]}")
    return {i: r for i, r in out.items() if not r["familiar"]}


def boot_mean(x, n=10_000):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(0).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def sign_p(a, b):
    n = a + b
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n) if n else 1.0


def verdict(mean, ci):
    if ci[1] < 0:
        return "reversed"
    if ci[0] <= 0:
        return "gone"
    return "survives" if mean >= RULE_POINTS else "reduced"


def contrast(a, b, ids, rooms):
    """a minus b on the same novel trials, rooms as the unit; a and b: room -> id -> row."""
    d = [float(np.mean([a[r][i]["correct"] for i in ids[r]]) - np.mean([b[r][i]["correct"] for i in ids[r]]))
         for r in rooms]
    up = sum(a[r][i]["correct"] and not b[r][i]["correct"] for r in rooms for i in ids[r])
    dn = sum(b[r][i]["correct"] and not a[r][i]["correct"] for r in rooms for i in ids[r])
    m, ci = boot_mean(d)
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    return {"mean": m, "ci95": ci, "rooms_positive": int(pos), "rooms_negative": int(neg),
            "room_sign_p": sign_p(pos, neg), "trials_better": int(up), "trials_worse": int(dn),
            "trial_sign_p": sign_p(up, dn), "n": int(sum(len(ids[r]) for r in rooms)),
            "per_room": dict(zip(rooms, d))}


def acc(run, ids, rooms):
    per = [float(np.mean([run[r][i]["correct"] for i in ids[r]])) for r in rooms]
    m, ci = boot_mean(per)
    return {"mean": m, "ci95": ci, "per_room": dict(zip(rooms, per)),
            "pooled": float(np.mean([run[r][i]["correct"] for r in rooms for i in ids[r]])),
            "unparsed": float(np.mean([run[r][i]["answer"] <= 0 for r in rooms for i in ids[r]]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="64-frame files stand in for the 256-frame ones")
    ap.add_argument("--out", help=f"output JSON (default {OUT}; with --test, required and not {OUT})")
    args = ap.parse_args()
    out_path = args.out or (None if args.test else OUT)
    if args.test and (out_path is None or os.path.abspath(out_path) == os.path.abspath(OUT)):
        raise SystemExit("--test needs --out somewhere other than " + OUT)

    man = {r: {t["id"]: t for t in json.load(open(MANIFEST.format(r=r)))} for r in ROOMS}
    nov = {r: sorted(i for i, t in man[r].items() if not t["familiar"]) for r in ROOMS}
    n_nov = sum(map(len, nov.values()))
    opus = {r: load(OPUS.format(r=r), man[r]) for r in ROOMS}
    assert all(opus[r] is not None and set(opus[r]) == set(nov[r]) for r in ROOMS), "Opus does not cover the novel trials"

    l256, l64, par, plat64, completeness = {}, {}, {}, {}, {}
    for k in RUNS:
        l256[k], l64[k], par[k], plat64[k] = {}, {}, {}, {}
        for r in ROOMS:
            p256, (p64, pl), ppar = paths(k, r, args.test)
            l256[k][r], l64[k][r], plat64[k][r] = load(p256, man[r]), load(p64, man[r]), pl
            par[k][r] = load(ppar, man[r]) if r in PARITY_ROOMS else None
            assert l64[k][r] is not None and set(l64[k][r]) == set(nov[r]), f"64-frame run {k} {r} incomplete"
        completeness[k] = {r: len(l256[k][r] or {}) for r in ROOMS}
    complete = [k for k in RUNS if all(completeness[k][r] == len(nov[r]) for r in ROOMS)]

    print(f"experiment {EXP}{' (TEST: 64-frame files stand in for 256)' if args.test else ''}: "
          f"novel trials answered with the 256-frame walk, per room")
    print(" " * 10 + " ".join(f"{r:>6s}" for r in ROOMS))
    print(f"{'manifest':10s}" + " ".join(f"{len(nov[r]):6d}" for r in ROOMS))
    for k in RUNS:
        print(f"{'run ' + str(k):10s}" + " ".join(f"{completeness[k][r]:6d}" for r in ROOMS)
              + ("  complete" if k in complete else "  INCOMPLETE"))

    res = {"experiment": EXP, "test_stand_in": args.test, "rooms": ROOMS, "novel_trials": {r: len(nov[r]) for r in ROOMS},
           "runs_order": list(RUNS), "completeness": completeness, "platform_256": "CUDA", "platform_64": plat64,
           "opus": acc(opus, nov, ROOMS), "runs": {}, "parity": {}, "recipe": None}
    o = res["opus"]
    print(f"\nOpus 5.5, 256-frame walk: {o['mean']:.1%} [{o['ci95'][0]:.1%}, {o['ci95'][1]:.1%}] (room mean)")

    # parity: the CUDA rerun of P5005's 64-frame walk against the Mac's (MPS) answers, novel trials
    agree = total = 0
    for k in RUNS:
        got = {r: par[k][r] for r in PARITY_ROOMS}
        if any(g is None or set(g) != set(nov[r]) for r, g in got.items()) or any(plat64[k][r] != "MPS" for r in PARITY_ROOMS):
            res["parity"][k] = {"status": "incomplete"}
            continue
        a = sum(got[r][i]["answer"] == l64[k][r][i]["answer"] for r in PARITY_ROOMS for i in nov[r])
        n = sum(len(nov[r]) for r in PARITY_ROOMS)
        cuda = np.mean([got[r][i]["correct"] for r in PARITY_ROOMS for i in nov[r]])
        mps = np.mean([l64[k][r][i]["correct"] for r in PARITY_ROOMS for i in nov[r]])
        res["parity"][k] = {"status": "done", "agree": int(a), "n": int(n), "agreement": a / n,
                            "accuracy_mps": float(mps), "accuracy_cuda": float(cuda)}
        agree, total = agree + a, total + n
    parity_runs = [k for k in RUNS if res["parity"][k]["status"] == "done"]
    parity_agreement = agree / total if total else None
    parity_pass = None if parity_agreement is None else parity_agreement >= PARITY_AGREEMENT
    print(f"\nparity (64-frame, {'+'.join(PARITY_ROOMS)} novel, CUDA rerun vs MPS): runs {parity_runs}, "
          + (f"answers identical {agree}/{total} = {parity_agreement:.1%} -> {'PASS' if parity_pass else 'FAIL'}"
             if total else "no parity run complete"))

    # the 64-frame answers on the 256-frame walk's device (CUDA): experiment 9c's in nine rooms, the rerun in P5005
    def cuda64(k):
        m = {r: l64[k][r] for r in ROOMS if plat64[k][r] == "CUDA"}
        if res["parity"][k]["status"] == "done":
            m.update({r: par[k][r] for r in PARITY_ROOMS})
        return m

    print("\nper run (novel trials; room mean [95% CI over rooms]; rooms +/-; trials better/worse)")
    for k in complete:
        ids = nov
        a256, a64 = acc(l256[k], ids, ROOMS), acc(l64[k], ids, ROOMS)
        lead256 = contrast(l256[k], opus, ids, ROOMS)
        lead64 = contrast(l64[k], opus, ids, ROOMS)
        same = cuda64(k)
        assert set(same) == set(ROOMS), f"run {k}: no CUDA 64-frame answers in {sorted(set(ROOMS) - set(same))}"
        d = contrast(l256[k], same, ids, ROOMS)
        d_9c = contrast(l256[k], l64[k], ids, ROOMS)
        res["runs"][k] = {"lora256": a256, "lora64": a64, "lead256_over_opus": lead256, "lead64_over_opus": lead64,
                          "lora256_minus_64": d, "lora256_minus_64_9c_files": d_9c,
                          "verdict": verdict(lead256["mean"], lead256["ci95"])}
        print(f"  run {k}: 256 {a256['mean']:.1%} [{a256['ci95'][0]:.1%}, {a256['ci95'][1]:.1%}] (unparsed {a256['unparsed']:.1%})"
              f" | 64 {a64['mean']:.1%} | 256 - Opus {100 * lead256['mean']:+.1f} [{100 * lead256['ci95'][0]:+.1f}, "
              f"{100 * lead256['ci95'][1]:+.1f}] {lead256['rooms_positive']}/{lead256['rooms_negative']} "
              f"({lead256['trials_better']}/{lead256['trials_worse']}) -> {res['runs'][k]['verdict']}"
              f" | 256 - 64 {100 * d['mean']:+.1f} [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}] "
              f"({d['trials_better']}/{d['trials_worse']}) | vs 9c's files {100 * d_9c['mean']:+.1f}")

    summary = {"status": "no complete run" if not complete else ("complete" if len(complete) == len(RUNS) else "interim"),
               "runs_complete": complete, "n_runs": len(complete), "n_novel_trials": n_nov,
               "opus256_mean": o["mean"], "opus256_ci95": o["ci95"],
               "parity_agreement": parity_agreement, "parity_pass": parity_pass, "parity_runs": parity_runs}
    if complete:
        rec = {}
        mean_over = lambda key: {r: float(np.mean([res["runs"][k][key]["per_room"][r] for k in complete])) for r in ROOMS}
        for name, key in (("lora256", "lora256"), ("lora64", "lora64"), ("lead256", "lead256_over_opus"),
                          ("lead64", "lead64_over_opus"), ("lora256_minus_64", "lora256_minus_64"),
                          ("lora256_minus_64_9c_files", "lora256_minus_64_9c_files")):
            per = mean_over(key)
            m, ci = boot_mean(list(per.values()))
            rec[name] = {"mean": m, "ci95": ci, "per_room": per,
                         "rooms_positive": int(sum(v > 0 for v in per.values())),
                         "rooms_negative": int(sum(v < 0 for v in per.values()))}
        res["recipe"] = rec
        runs256 = [res["runs"][k]["lora256"]["mean"] for k in complete]
        leads = [res["runs"][k]["lead256_over_opus"]["mean"] for k in complete]
        v = verdict(rec["lead256"]["mean"], rec["lead256"]["ci95"])
        summary.update({
            "lora256_mean": rec["lora256"]["mean"], "lora256_ci95": rec["lora256"]["ci95"],
            "lora256_min": float(min(runs256)), "lora256_max": float(max(runs256)),
            "lora256_unparsed": float(np.mean([res["runs"][k]["lora256"]["unparsed"] for k in complete])),
            "lora64_mean": rec["lora64"]["mean"],
            "lead256_mean": rec["lead256"]["mean"], "lead256_ci95": rec["lead256"]["ci95"],
            "lead256_min": float(min(leads)), "lead256_max": float(max(leads)),
            "lead256_rooms_positive": rec["lead256"]["rooms_positive"],
            "lead64_mean": rec["lead64"]["mean"],
            "lead_retained": rec["lead256"]["mean"] / rec["lead64"]["mean"] if rec["lead64"]["mean"] else None,
            "lora256_minus_64_mean": rec["lora256_minus_64"]["mean"], "lora256_minus_64_ci95": rec["lora256_minus_64"]["ci95"],
            "lora256_minus_64_rooms_negative": rec["lora256_minus_64"]["rooms_negative"],
            "lora256_minus_64_9c_files_mean": rec["lora256_minus_64_9c_files"]["mean"],
            "runs_survive": [k for k in complete if res["runs"][k]["verdict"] == "survives"],
            "verdict": v})
        print(f"\nrecipe ({len(complete)} runs: {complete}; {summary['status']}):")
        for name in ("lora256", "lora64", "lead256", "lead64", "lora256_minus_64", "lora256_minus_64_9c_files"):
            x = rec[name]
            fmt = (lambda y: f"{y:.1%}") if name.startswith("lora") and "minus" not in name else (lambda y: f"{100 * y:+.1f}")
            print(f"  {name:18s} {fmt(x['mean'])} [{fmt(x['ci95'][0])}, {fmt(x['ci95'][1])}]  rooms +{x['rooms_positive']}/-{x['rooms_negative']}")
        print(f"  lead retained at 256 frames: {summary['lead_retained']:.2f} of the 64-frame lead")
        print(f"  VERDICT (registered rule: lead >= 10 points and CI above 0): {v}")
    res = {"summary": summary, **res}
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
