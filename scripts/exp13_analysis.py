"""Experiments 13a and 13b's pre-registered analyses (docs/experiments.md).

13a, the LoRA model's use of order (ten held-out rooms, 64-frame setting, novel trials): accuracy per
room under each presentation, means over rooms with 95% bootstrap CIs (10,000 resamples), paired
per-room differences with bootstrap CIs and room-level sign tests. Primary contrasts: window-shuffled
minus shuffled, ordered minus reversed, ordered minus renumbered. Trial-level pooled sign tests and
first-listed answer rates are reported alongside.

13b, the base model's drop with the walk (room A, base Qwen3-VL-8B, 256 px): irrelevant walk minus no
walk and minus the 64-frame walk, all on MPS, with paired sign tests and 95% intervals from a
bootstrap over walks; novel trials (primary) and all trials, CUDA references reported too.

    python scripts/exp13_analysis.py      # -> results/exp13_summary.json
"""
import json
import math

import numpy as np

ROOMS = [f"P{s}" for s in range(5000, 5010)]
L = "results/two_choice/Qwen3-VL-8B-Instruct+qwen8b-proc-room{r}-256px-"
ORDER_RUNS = {
    "ordered": L + "max64-mps.jsonl",
    "shuffled": L + "max64-shuffled-mps.jsonl",
    "no walk": L + "nowalk-mps.jsonl",
    "reversed": L + "max64-reversed-mps.jsonl",
    "window-shuffled": L + "max64-window8-mps.jsonl",
    "renumbered": L + "max64-renumbered-mps.jsonl",
}
ORDER_CONTRASTS = [("window-shuffled", "shuffled"), ("ordered", "reversed"), ("ordered", "renumbered"),
                   ("window-shuffled", "ordered"), ("reversed", "shuffled"), ("renumbered", "shuffled")]
B = "results/two_choice/Qwen3-VL-8B-Instruct-dissimilar-256px-"
BASE_RUNS = {
    "irrelevant walk": B + "max64-irrelevant-roomB-mps.jsonl",
    "no walk": B + "nowalk-mps.jsonl",
    "64-frame walk": B + "max64-mps.jsonl",
    "no walk (CUDA)": B + "nowalk.jsonl",
    "64-frame walk (CUDA)": B + "max64.jsonl",
    "256-frame walk (CUDA)": "results/two_choice/Qwen3-VL-8B-Instruct-dissimilar-256px.jsonl",
}
BASE_CONTRASTS = [("irrelevant walk", "no walk"), ("irrelevant walk", "64-frame walk"), ("64-frame walk", "no walk")]
FIELDS = ("walk", "probe", "familiar", "near", "far")


def manifest(path):
    t = json.load(open(path))
    return {x["id"]: x for x in (t["trials"] if isinstance(t, dict) else t)}


def load(path, trials):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = {r["id"]: r for r in rows}
    bad = [r["id"] for r in rows if r["id"] not in trials or any(r[k] != trials[r["id"]][k] for k in FIELDS)
           or bool(r["correct"]) != (r["answer"] == r["near"])]
    assert len(out) == len(rows) == len(trials) and not bad, (path, len(rows), len(trials), bad[:3])
    return out


def sign_p(up, dn):
    n = up + dn
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(max(up, dn), n + 1)) / 2 ** n) if n else 1.0


def boot(x, rng, n=10_000):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return {"mean": float(x.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}


def exp13a(rng):
    trials = {r: manifest(f"results/export/dissimilar-room{r}/trials.json") for r in ROOMS}
    data = {(n, r): load(p.format(r=r), trials[r]) for n, p in ORDER_RUNS.items() for r in ROOMS}
    nov = {r: [i for i, t in trials[r].items() if not t["familiar"]] for r in ROOMS}
    acc = {n: [np.mean([data[n, r][i]["correct"] for i in nov[r]]) for r in ROOMS] for n in ORDER_RUNS}
    out = {"n_novel": sum(len(v) for v in nov.values()), "conditions": {}, "contrasts": {}}
    for n in ORDER_RUNS:
        first = [data[n, r][i]["answer"] == data[n, r][i]["a"] for r in ROOMS for i in trials[r]]
        out["conditions"][n] = {**boot(acc[n], rng), "per_room": dict(zip(ROOMS, map(float, acc[n]))),
                                "all_trials": float(np.mean([data[n, r][i]["correct"] for r in ROOMS for i in trials[r]])),
                                "first_listed": float(np.mean(first))}
    for a, b in ORDER_CONTRASTS:
        d = np.array(acc[a]) - np.array(acc[b])
        up = sum(data[a, r][i]["correct"] and not data[b, r][i]["correct"] for r in ROOMS for i in nov[r])
        dn = sum(data[b, r][i]["correct"] and not data[a, r][i]["correct"] for r in ROOMS for i in nov[r])
        out["contrasts"][f"{a} - {b}"] = {
            **boot(d, rng), "rooms_up": int((d > 0).sum()), "rooms_down": int((d < 0).sum()),
            "room_sign_p": sign_p(int((d > 0).sum()), int((d < 0).sum())),
            "trials_better": int(up), "trials_worse": int(dn), "trial_sign_p": sign_p(up, dn)}
    return out


def exp13b(rng):
    trials = manifest("results/export/dissimilar/trials.json")
    data = {n: load(p, trials) for n, p in BASE_RUNS.items()}
    out = {"conditions": {}, "contrasts": {}}
    for split in ("novel", "all"):
        ids = [i for i, t in trials.items() if split == "all" or not t["familiar"]]
        for n in BASE_RUNS:
            out["conditions"].setdefault(n, {})[split] = {
                "n": len(ids), "accuracy": float(np.mean([data[n][i]["correct"] for i in ids])),
                "first_listed": float(np.mean([data[n][i]["answer"] == data[n][i]["a"] for i in ids]))}
        walks = sorted({trials[i]["walk"] for i in ids})
        for a, b in BASE_CONTRASTS:
            d = {w: np.array([int(data[a][i]["correct"]) - int(data[b][i]["correct"]) for i in ids if trials[i]["walk"] == w])
                 for w in walks}
            bs = [np.concatenate([d[w] for w in rng.choice(walks, len(walks))]).mean() for _ in range(10_000)]
            up = sum(int(v.clip(0).sum()) for v in d.values())
            dn = sum(int((-v).clip(0).sum()) for v in d.values())
            out["contrasts"].setdefault(f"{a} - {b}", {})[split] = {
                "difference": float(np.concatenate(list(d.values())).mean()),
                "ci95_walks": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "better": up, "worse": dn, "sign_p": sign_p(up, dn), "n_walks": len(walks)}
    return out


def main():
    rng = np.random.default_rng(0)
    out = {"13a": exp13a(rng), "13b": exp13b(rng)}
    json.dump(out, open("results/exp13_summary.json", "w"), indent=1)
    a = out["13a"]
    print(f"13a: LoRA, ten held-out rooms, novel trials ({a['n_novel']}); mean over rooms [95% CI]")
    for n, v in a["conditions"].items():
        print(f"  {n:16s} {100 * v['mean']:5.1f}% [{100 * v['ci95'][0]:.1f}, {100 * v['ci95'][1]:.1f}]"
              f"   all trials {100 * v['all_trials']:.1f}%   first-listed {100 * v['first_listed']:.0f}%")
    for n, v in a["contrasts"].items():
        print(f"  {n:30s} {100 * v['mean']:+5.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]"
              f"  rooms {v['rooms_up']}/{v['rooms_down']} (p {v['room_sign_p']:.3g})"
              f"  trials {v['trials_better']}/{v['trials_worse']} (p {v['trial_sign_p']:.2g})")
    b = out["13b"]
    print("13b: base 8B, room A")
    for n, v in b["conditions"].items():
        print(f"  {n:22s} novel {100 * v['novel']['accuracy']:5.1f}% ({v['novel']['n']}), first-listed "
              f"{100 * v['novel']['first_listed']:.0f}%   all {100 * v['all']['accuracy']:5.1f}%")
    for n, v in b["contrasts"].items():
        for s in ("novel", "all"):
            x = v[s]
            print(f"  {n:34s} {s:5s} {100 * x['difference']:+5.1f} [{100 * x['ci95_walks'][0]:+.1f}, "
                  f"{100 * x['ci95_walks'][1]:+.1f}]  {x['better']}/{x['worse']} (p {x['sign_p']:.2g})")
    print("-> results/exp13_summary.json")


if __name__ == "__main__":
    main()
