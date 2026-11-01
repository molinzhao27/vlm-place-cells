"""Experiment 24, unregistered follow-ups on the 256-frame answers (no thresholds; docs/experiments.md "Exp 24 results").

    .venv-mac/bin/python scripts/lora256_exploratory.py     # -> results/lora_256frames_exploratory.json

- margin over Exp 20's best procedural rule, which had all 256 frames, with the LoRA at 256 frames (room means, bootstrap
  over rooms as in lora256_analysis.py);
- which option the runs lean to: share of answers naming the later of the two frames, against the share of trials where
  the near option is the later one;
- how the runs' errors overlap, and how often Opus is wrong on the trials most runs miss;
- the 256-minus-64 drop by the options' distance in walk time.
"""
import json

import numpy as np

import lora256_analysis as la

OUT = "results/lora_256frames_exploratory.json"
BINS = ((0, 32), (32, 64), (64, 128), (128, 256))


def main():
    man = {r: {t["id"]: t for t in json.load(open(la.MANIFEST.format(r=r)))} for r in la.ROOMS}
    l256, l64 = {}, {}
    for k in la.RUNS:
        l256[k] = {r: la.load(la.paths(k, r)[0], man[r]) for r in la.ROOMS}
        l64[k] = {r: la.load(la.paths(k, r)[1][0], man[r]) for r in la.ROOMS}
        l64[k]["P5005"] = la.load(la.paths(k, "P5005")[2], man["P5005"])     # the CUDA rerun, as the registered contrast
    opus = {r: la.load(la.OPUS.format(r=r), man[r]) for r in la.ROOMS}
    ids = [(r, i) for r in la.ROOMS for i in sorted(l256[la.RUNS[0]][r])]
    near = np.array([man[r][i]["near"] for r, i in ids])
    far = np.array([man[r][i]["far"] for r, i in ids])
    wrong256 = np.array([[not l256[k][r][i]["correct"] for r, i in ids] for k in la.RUNS])
    wrong64 = np.array([[not l64[k][r][i]["correct"] for r, i in ids] for k in la.RUNS])
    wrong_opus = np.array([not opus[r][i]["correct"] for r, i in ids])

    ret = json.load(open("results/retrieval_baselines.json"))["summary"]["procedural"]
    grp = json.load(open("results/retrieval_baselines.json"))["groups"]["procedural"]
    lora4_minus_rule = grp["LoRA four-epoch mean minus best variant (novel)"]["per_scene"]
    rec = json.load(open(la.OUT))["recipe"]
    assert ret["best_variant_chosen_in"] == "own"
    # Exp 20's lora4 per room is the five-run 64-frame mean with 9c's files, which is recipe["lora64"]
    rule = {r: rec["lora64"]["per_room"][r] - lora4_minus_rule[r] for r in la.ROOMS}
    assert abs(np.mean(list(rule.values())) - ret["novel"]["best_variant"]["mean"]) < 1e-9
    d = [rec["lora256"]["per_room"][r] - rule[r] for r in la.ROOMS]
    m, ci = la.boot_mean(d)

    later = lambda run, r, i: run[r][i]["answer"] == max(run[r][i]["near"], run[r][i]["far"])
    res = {"experiment": "24, unregistered", "n_trials": len(ids),
           "rule": {"name": ret["best_variant"], "mean": float(np.mean(list(rule.values()))),
                    "lora256_minus_rule_mean": m, "lora256_minus_rule_ci95": ci,
                    "rooms_positive": int(sum(x > 0 for x in d)), "per_room": dict(zip(la.ROOMS, d))},
           "lean": {"near_is_later_share": float((near > far).mean()),
                    "answers_later_256": {k: float(np.mean([later(l256[k], r, i) for r, i in ids])) for k in la.RUNS},
                    "answers_later_64": {k: float(np.mean([later(l64[k], r, i) for r, i in ids])) for k in la.RUNS}},
           "errors": {"per_run_256": dict(zip(la.RUNS, map(int, wrong256.sum(1)))),
                      "per_run_64": dict(zip(la.RUNS, map(int, wrong64.sum(1)))),
                      "trials_3plus_runs_wrong_256": int((wrong256.sum(0) >= 3).sum()),
                      "trials_3plus_runs_wrong_64": int((wrong64.sum(0) >= 3).sum()),
                      "trials_all_runs_wrong_256": int(wrong256.all(0).sum()),
                      "opus_wrong_overall": float(wrong_opus.mean()),
                      "opus_wrong_on_3plus_256": float(wrong_opus[wrong256.sum(0) >= 3].mean())},
           "drop_by_option_gap": [],
           "unparsed_replies": {k: [l256[k][r][i]["text"] for r, i in ids if l256[k][r][i]["answer"] <= 0] for k in la.RUNS}}
    gap = np.abs(near - far)
    a256, a64 = 1 - wrong256.mean(0), 1 - wrong64.mean(0)
    for lo, hi in BINS:
        s = (gap >= lo) & (gap < hi)
        res["drop_by_option_gap"].append({"gap_frames": [lo, hi], "n": int(s.sum()), "lora256": float(a256[s].mean()),
                                          "lora64": float(a64[s].mean()), "drop": float((a256 - a64)[s].mean())})
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "unparsed_replies"}, indent=1))


if __name__ == "__main__":
    main()
