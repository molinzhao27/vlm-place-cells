"""Experiment 16 (pre-registered in docs/experiments.md): behaviour in eight HSSD houses.

    .venv-mac/bin/python scripts/hssd_analysis.py      # -> results/hssd_summary.json

As experiment 12 (scripts/photoreal_analysis.py), with eight scenes: accuracy on all lure-free trials
per house with an exact (Clopper-Pearson) 95% CI and pooled; novel-trial accuracy pooled; the
registered contrasts with paired two-sided sign tests pooled over houses, a bootstrap over walks
(clusters within houses), and, as registered for eight houses, houses as the unit (mean over houses,
95% bootstrap CI with 10,000 resamples and seed 0, house-level sign test). Qwen ran on two platforms
(see the platform note in experiment 16): HSSD1-3 on the Mac (MPS, files tagged -mps), HSSD4-8 on vast-3
(CUDA, untagged). Every run is checked against its house's trial manifest. A separate, labelled block
pools these houses with experiment 12's apartments.
"""
import json
from math import comb

import numpy as np
from scipy.stats import beta

HOUSES = [f"HSSD{k}" for k in range(1, 9)]
MPS = {"HSSD1", "HSSD2", "HSSD3", "ReplicaApt0", "ReplicaApt1", "ReplicaApt2"}   # Qwen ran on the Mac here
Q = "Qwen3-VL-8B-Instruct"


def qwen(adapter, cond):
    def path(s):
        tag = "-mps" if s in MPS else ""
        a = "+qwen8b-proc" if adapter else ""
        c = {"walk": "-256px-max64", "no walk": "-256px-nowalk", "shuffled": "-256px-max64-shuffled"}[cond]
        return f"results/two_choice/{Q}{a}-room{s}{c}{tag}.jsonl"
    return path


RUNS = {
    "base, walk": qwen(False, "walk"), "base, no walk": qwen(False, "no walk"), "base, shuffled": qwen(False, "shuffled"),
    "LoRA, walk": qwen(True, "walk"), "LoRA, no walk": qwen(True, "no walk"), "LoRA, shuffled": qwen(True, "shuffled"),
    "Opus, walk": lambda s: f"results/claude/claude-opus-5-5-room{s}-low.jsonl",
    "Opus, no walk": lambda s: f"results/claude/claude-opus-5-5-room{s}-nowalk-low.jsonl",
    "Opus, shuffled": lambda s: f"results/claude/claude-opus-5-5-room{s}-shuffled-low.jsonl",
}
CONTRASTS = [("LoRA, walk", "base, walk"), ("LoRA, walk", "LoRA, no walk"), ("LoRA, walk", "LoRA, shuffled"),
             ("Opus, walk", "Opus, no walk"), ("Opus, shuffled", "Opus, walk"), ("Opus, shuffled", "Opus, no walk"),
             ("LoRA, walk", "Opus, walk"), ("base, walk", "base, no walk")]
APARTMENTS = ["ReplicaApt0", "ReplicaApt1", "ReplicaApt2"]


def cp(k, n):
    return [float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0, float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def boot_mean(x, seed=0, n=10_000):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def walk_clustered(pairs, seed=0, n_boot=10000, n_flip=20000):
    rng = np.random.default_rng(seed)
    ids = sorted({c for c, _, _ in pairs})
    d = {c: np.array([int(x) - int(y) for k, x, y in pairs if k == c]) for c in ids}
    n, diff = len(pairs), sum(int(x) - int(y) for _, x, y in pairs) / len(pairs)
    boot = [np.concatenate([d[c] for c in rng.choice(ids, len(ids))]).mean() for _ in range(n_boot)]
    sums = np.array([d[c].sum() for c in ids])
    flips = np.abs((sums * rng.choice([-1, 1], (n_flip, len(ids)))).sum(1)) / n
    return {"ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "sign_flip_p": float((flips >= abs(diff) - 1e-12).mean()), "n_walks": len(ids)}


def manifest(s):
    tr = json.load(open(f"results/export/dissimilar-room{s}/trials.json"))
    return {t["id"]: t for t in (tr["trials"] if isinstance(tr, dict) else tr)}


def load(path, man):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        t = man[r["id"]]
        assert (r["near"], r["far"]) == (t["near"], t["far"]) and r["familiar"] == t["familiar"], (path, r["id"])
        assert r["id"] not in rows, (path, "duplicate", r["id"])
        rows[r["id"]] = r
    missing = set(man) - set(rows)
    assert not missing, f"{path} misses {len(missing)} trials"
    return rows


def analyse(scenes, runs, houses_as_unit):
    man = {s: manifest(s) for s in scenes}
    data = {(name, s): load(runs[name](s), man[s]) for name in runs for s in scenes}
    out = {"scenes": {}, "pooled": {}, "contrasts": {}, "first_listed": {}}
    for s in scenes:
        ids = list(man[s]); nov = [i for i in ids if not man[s][i]["familiar"]]
        out["scenes"][s] = {}
        for name in runs:
            k, kn = sum(bool(data[name, s][i]["correct"]) for i in ids), sum(bool(data[name, s][i]["correct"]) for i in nov)
            out["scenes"][s][name] = {"all": {"n": len(ids), "accuracy": k / len(ids), "ci95": cp(k, len(ids))},
                                      "novel": {"n": len(nov), "accuracy": kn / len(nov) if nov else None}}
    for name in runs:
        for split in ("all", "novel", "familiar"):
            k = n = 0
            for s in scenes:
                ids = [i for i, t in man[s].items() if split == "all" or t["familiar"] == (split == "familiar")]
                k += sum(bool(data[name, s][i]["correct"]) for i in ids); n += len(ids)
            out["pooled"].setdefault(name, {})[split] = {"n": n, "correct": k, "accuracy": k / n, "ci95": cp(k, n)}
        first = [data[name, s][i]["answer"] == data[name, s][i]["a"] for s in scenes for i in man[s]]
        out["first_listed"][name] = sum(first) / len(first)
    for a, b in CONTRASTS:
        if a not in runs or b not in runs:
            continue
        for split in ("all", "novel"):
            better = worse = 0; pairs = []; per_scene = []
            for s in scenes:
                ids = [i for i, t in man[s].items() if split == "all" or not t["familiar"]]
                for i in ids:
                    x, y = bool(data[a, s][i]["correct"]), bool(data[b, s][i]["correct"])
                    better += x and not y; worse += y and not x
                    pairs.append((f"{s}-{man[s][i]['walk']}", x, y))
                if ids:
                    per_scene.append(np.mean([bool(data[a, s][i]["correct"]) for i in ids]) - np.mean([bool(data[b, s][i]["correct"]) for i in ids]))
            c = {"difference": out["pooled"][a][split]["accuracy"] - out["pooled"][b][split]["accuracy"],
                 "better": better, "worse": worse, "sign_p": sign_p(better, worse), "walk_clustered": walk_clustered(pairs)}
            if houses_as_unit:
                m, ci = boot_mean(per_scene)
                up, dn = int(sum(x > 0 for x in per_scene)), int(sum(x < 0 for x in per_scene))
                c["houses"] = {"mean": m, "ci95": ci, "up": up, "down": dn, "sign_p": sign_p(up, dn),
                               "per_house": [float(x) for x in per_scene]}
            out["contrasts"].setdefault(f"{a} - {b}", {})[split] = c
    return out


def main():
    out = {"platform": {h: ("Mac MPS" if h in MPS else "vast-3 CUDA (RTX 5090)") for h in HOUSES}}
    out.update(analyse(HOUSES, RUNS, houses_as_unit=True))
    out["vggt"] = {}
    for s in HOUSES:
        try:
            out["vggt"][s] = json.load(open(f"results/vggt_control_{s}.json"))
        except FileNotFoundError:
            out["vggt"][s] = None
    # Separate, labelled summary: houses and experiment 12's apartments together (apartments: Qwen on the Mac).
    out["with_apartments"] = analyse(HOUSES + APARTMENTS, RUNS, houses_as_unit=True)
    json.dump(out, open("results/hssd_summary.json", "w"), indent=1)

    print("Experiment 16: eight HSSD houses. Pooled accuracy, all lure-free trials (n) [exact 95% CI]; novel")
    for name, p in out["pooled"].items():
        per = " ".join(f"{100 * out['scenes'][s][name]['all']['accuracy']:.0f}" for s in HOUSES)
        print(f"  {name:15s} {100 * p['all']['accuracy']:5.1f}% ({p['all']['n']}) [{100 * p['all']['ci95'][0]:.1f}, "
              f"{100 * p['all']['ci95'][1]:.1f}]  novel {100 * p['novel']['accuracy']:.1f}% ({p['novel']['n']})  houses {per}")
    print("contrasts: pooled (better/worse, sign p); houses as the unit (mean [CI], up/down, p)")
    for c, v in out["contrasts"].items():
        for split in ("all", "novel"):
            x, h = v[split], v[split]["houses"]
            print(f"  {c:28s} {split:5s} {100 * x['difference']:+5.1f} ({x['better']}/{x['worse']}, p {x['sign_p']:.2g})  "
                  f"houses {100 * h['mean']:+5.1f} [{100 * h['ci95'][0]:+.1f}, {100 * h['ci95'][1]:+.1f}] {h['up']}/{h['down']} (p {h['sign_p']:.2g})")
    print("-> results/hssd_summary.json")


if __name__ == "__main__":
    main()
