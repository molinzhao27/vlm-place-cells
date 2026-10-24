"""Experiment 12 (pre-registered in docs/experiments.md): behaviour in three Replica apartments.

    .venv-mac/bin/python scripts/photoreal_analysis.py      # -> results/photoreal_summary.json

Primary outcome: accuracy on all lure-free trials per scene with an exact (Clopper-Pearson) 95% CI,
and pooled over the three scenes. Secondary: novel-trial accuracy, pooled (58 trials). Questions:
LoRA vs base and LoRA walk vs no walk (paired two-sided sign tests, pooled over scenes); where the
base model and Opus sit; VGGT's probe errors (results/vggt_control_<scene>.json). Every run is
checked against the scene's trial manifest. Opus is also reported on its first pass (the first 16
walks with trials; see the deviation note in experiment 12).
"""
import json
from math import comb

from scipy.stats import beta

SCENES = ["ReplicaApt0", "ReplicaApt1", "ReplicaApt2"]
Q = "Qwen3-VL-8B-Instruct"
RUNS = {
    "base, walk": "results/two_choice/" + Q + "-room{s}-256px-max64-mps.jsonl",
    "base, no walk": "results/two_choice/" + Q + "-room{s}-256px-nowalk-mps.jsonl",
    "base, shuffled": "results/two_choice/" + Q + "-room{s}-256px-max64-shuffled-mps.jsonl",
    "LoRA, walk": "results/two_choice/" + Q + "+qwen8b-proc-room{s}-256px-max64-mps.jsonl",
    "LoRA, no walk": "results/two_choice/" + Q + "+qwen8b-proc-room{s}-256px-nowalk-mps.jsonl",
    "LoRA, shuffled": "results/two_choice/" + Q + "+qwen8b-proc-room{s}-256px-max64-shuffled-mps.jsonl",
    "Opus, walk": "results/claude/claude-opus-5-5-room{s}-low.jsonl",
    "Opus, no walk": "results/claude/claude-opus-5-5-room{s}-nowalk-low.jsonl",
    "Opus, shuffled": "results/claude/claude-opus-5-5-room{s}-shuffled-low.jsonl",     # experiment 14b
}
CONTRASTS = [("LoRA, walk", "base, walk"), ("LoRA, walk", "LoRA, no walk"), ("LoRA, walk", "LoRA, shuffled"),
             ("Opus, walk", "Opus, no walk"), ("LoRA, walk", "Opus, walk"), ("base, walk", "base, no walk"),
             ("Opus, walk", "Opus, shuffled"), ("Opus, shuffled", "Opus, no walk")]


def cp(k, n):
    """Exact 95% binomial interval."""
    return [float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0, float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def walk_clustered(pairs, seed=0, n_boot=10000, n_flip=20000):
    """pairs: (cluster id, a correct, b correct). Bootstrap over clusters (walks within a scene) of the
    pooled accuracy difference, and a whole-cluster sign-flip test of it."""
    import numpy as np
    rng = np.random.default_rng(seed)
    ids = sorted({c for c, _, _ in pairs})
    d = {c: np.array([int(x) - int(y) for k, x, y in pairs if k == c]) for c in ids}
    n, diff = len(pairs), sum(int(x) - int(y) for _, x, y in pairs) / len(pairs)
    boot = [np.concatenate([d[c] for c in rng.choice(ids, len(ids))]).mean() for _ in range(n_boot)]
    sums = np.array([d[c].sum() for c in ids])
    flips = np.abs((sums * rng.choice([-1, 1], (n_flip, len(ids)))).sum(1)) / n
    return {"ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "sign_flip_p": float((flips >= abs(diff) - 1e-12).mean()), "n_walks": len(ids)}


def load(path, manifest):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        t = manifest[r["id"]]
        assert (r["near"], r["far"]) == (t["near"], t["far"]) and r["familiar"] == t["familiar"], (path, r["id"])
        rows[r["id"]] = r
    return rows


def acc(rows, ids):
    k = sum(bool(rows[i]["correct"]) for i in ids)
    return {"n": len(ids), "correct": k, "accuracy": k / len(ids) if ids else None, "ci95": cp(k, len(ids)) if ids else None}


def main():
    out = {"scenes": {}, "pooled": {}, "contrasts": {}, "opus_first_pass": {}, "first_listed": {}, "vggt": {}}
    data, manifests = {}, {}
    for s in SCENES:
        tr = json.load(open(f"results/export/dissimilar-room{s}/trials.json"))
        tr = tr["trials"] if isinstance(tr, dict) else tr
        manifests[s] = {t["id"]: t for t in tr}
        for name, pat in RUNS.items():
            rows = load(pat.format(s=s), manifests[s])
            missing = set(manifests[s]) - set(rows)
            assert not missing, f"{name} in {s} misses {len(missing)} trials"
            data[name, s] = rows
    for s in SCENES:
        ids = list(manifests[s])
        nov = [i for i in ids if not manifests[s][i]["familiar"]]
        out["scenes"][s] = {name: {"all": acc(data[name, s], ids), "novel": acc(data[name, s], nov)} for name in RUNS}
    for name in RUNS:
        for split in ("all", "novel", "familiar"):
            k = n = 0
            for s in SCENES:
                ids = [i for i, t in manifests[s].items() if split == "all" or t["familiar"] == (split == "familiar")]
                k += sum(bool(data[name, s][i]["correct"]) for i in ids)
                n += len(ids)
            out["pooled"].setdefault(name, {})[split] = {"n": n, "correct": k, "accuracy": k / n, "ci95": cp(k, n)}
        first = [(data[name, s][i]["answer"] == data[name, s][i]["a"]) for s in SCENES for i in manifests[s]]
        out["first_listed"][name] = sum(first) / len(first)
    for a, b in CONTRASTS:
        for split in ("all", "novel"):
            better = worse = 0
            pairs = []
            for s in SCENES:
                for i, t in manifests[s].items():
                    if split == "novel" and t["familiar"]:
                        continue
                    x, y = bool(data[a, s][i]["correct"]), bool(data[b, s][i]["correct"])
                    better += x and not y
                    worse += y and not x
                    pairs.append((f"{s}-{t['walk']}", x, y))
            out["contrasts"][f"{a} - {b}"] = out["contrasts"].get(f"{a} - {b}", {})
            out["contrasts"][f"{a} - {b}"][split] = {
                "difference": out["pooled"][a][split]["accuracy"] - out["pooled"][b][split]["accuracy"],
                "better": better, "worse": worse, "sign_p": sign_p(better, worse), "walk_clustered": walk_clustered(pairs)}
    for name in ("Opus, walk", "Opus, no walk"):          # the first pass: the first 16 walks that have trials
        for split in ("all", "novel"):
            k = n = 0
            for s in SCENES:
                first = set(sorted({t["walk"] for t in manifests[s].values()})[:16])
                ids = [i for i, t in manifests[s].items() if t["walk"] in first and (split == "all" or not t["familiar"])]
                k += sum(bool(data[name, s][i]["correct"]) for i in ids)
                n += len(ids)
            out["opus_first_pass"].setdefault(name, {})[split] = {"n": n, "correct": k, "accuracy": k / n, "ci95": cp(k, n)}
    for s in SCENES:
        out["vggt"][s] = json.load(open(f"results/vggt_control_{s}.json"))
    json.dump(out, open("results/photoreal_summary.json", "w"), indent=1)

    print("pooled accuracy, all lure-free trials (n) [exact 95% CI]; novel")
    for name in RUNS:
        p, q = out["pooled"][name]["all"], out["pooled"][name]["novel"]
        per = "  ".join(f"{100 * out['scenes'][s][name]['all']['accuracy']:.0f}%" for s in SCENES)
        print(f"  {name:16s} {100 * p['accuracy']:5.1f}% ({p['n']}) [{100 * p['ci95'][0]:.1f}, {100 * p['ci95'][1]:.1f}]"
              f"   scenes {per}   novel {100 * q['accuracy']:.0f}% ({q['n']})   first-listed {100 * out['first_listed'][name]:.0f}%")
    print("contrasts (pooled, paired two-sided sign tests)")
    for c, v in out["contrasts"].items():
        print(f"  {c:30s} all {100 * v['all']['difference']:+5.1f} ({v['all']['better']}/{v['all']['worse']}, p {v['all']['sign_p']:.2g})"
              f"   novel {100 * v['novel']['difference']:+5.1f} ({v['novel']['better']}/{v['novel']['worse']}, p {v['novel']['sign_p']:.2g})")
    for name, v in out["opus_first_pass"].items():
        print(f"  {name} first pass (16 walks): all {100 * v['all']['accuracy']:.1f}% ({v['all']['n']}), novel {100 * v['novel']['accuracy']:.0f}% ({v['novel']['n']})")
    print("-> results/photoreal_summary.json")


if __name__ == "__main__":
    main()
