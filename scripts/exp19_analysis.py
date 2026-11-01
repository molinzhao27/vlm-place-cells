"""Experiment 19 (pre-registered in docs/experiments.md): more photoreal scenes.

    .venv-mac/bin/python scripts/exp19_analysis.py      # -> results/exp19_summary.json (EXP19_SUMMARY=path overrides)

Six Replica FRL apartments (ReplicaFRL0-5, one floor plan with six furnishings) and twelve more HSSD houses
(HSSD9-20, experiment 16's draw continued); the accepted ones are listed in results/exp19_scenes.json. As in
experiments 12 and 16 (photoreal_analysis.py, hssd_analysis.py, seed_scenes_analysis.py), the primary measure is
accuracy on all lure-free trials and novel trials are secondary. Groups: the new apartments, the new houses, the
new scenes together (reported first), then each pooled with the old scenes (experiment 12's three apartments,
experiment 16's eight houses), labelled as pooled.

Per model and condition: accuracy per scene with an exact (Clopper-Pearson) 95% CI, pooled, and the mean over
scenes with a 95% bootstrap CI (10,000 resamples, seed 0). Contrasts per LoRA run (0 = data/ft/qwen8b-proc, 1-2 =
the two-epoch runs, 3-7 = experiment 9c's four-epoch runs) and as each recipe's mean and range (never mixed):
LoRA walk vs base walk, vs no walk, vs shuffled, vs Opus walk (Opus scenes only, same trials); Opus walk vs no
walk; base walk vs no walk. Each with the pooled trial-level sign test, a bootstrap over walks (with a sign-flip
test), scenes as the unit (bootstrap CI over scenes, scene-level sign test) and, where FRL scenes are in a group
with other apartments, layouts as the unit (the FRL scenes averaged into one layout; descriptive, no test).
The registered replication criteria are evaluated at the end.

Platforms: each scene's Qwen runs are on the Mac (MPS, files tagged -mps) or on CUDA (untagged); for every
(model, scene, condition) exactly one of the two files must exist. Every run is checked against its scene's
trial manifest. An Opus scene counts only if both its runs answered every trial. Runs that are missing or
incomplete are reported as such and left out, so the script also gives interim tables while runs are going.
"""
import json
import math
import os

import numpy as np
from scipy.stats import beta

OUT = os.environ.get("EXP19_SUMMARY", "results/exp19_summary.json")    # another path for interim checks
Q = "results/two_choice/Qwen3-VL-8B-Instruct"
GEM = "results/two_choice/gemma-4-12B-it"                  # experiment 18, CUDA only
RUNS = {0: "qwen8b-proc", 1: "qwen8b-proc-s1", 2: "qwen8b-proc-s2", **{k: f"qwen8b-proc-e4-s{k}" for k in range(3, 8)}}
MODELS = {"base": Q, **{f"run {k}": f"{Q}+{a}" for k, a in RUNS.items()},
          "Gemma base": GEM, **{f"Gemma run {k}": f"{GEM}+gemma12b-proc-e4-s{k}" for k in range(3, 6)}}
# Gemma stops at runs 3-5 (experiment 18 amendment 2; experiment 19 amendment 3)
RECIPES = {"two-epoch runs 0-2": ["run 0", "run 1", "run 2"], "four-epoch runs 3-7": [f"run {k}" for k in range(3, 8)],
           "Gemma four-epoch runs 3-5": [f"Gemma run {k}" for k in range(3, 6)]}     # never pooled with each other
BASE = lambda m: "Gemma base" if m.startswith("Gemma") else "base"
KINDS = ("vs base walk", "vs no walk", "vs shuffled", "vs Opus walk")
CONDS = {"walk": "-256px-max64", "no walk": "-256px-nowalk", "shuffled": "-256px-max64-shuffled"}
OLD_APTS = ["ReplicaApt0", "ReplicaApt1", "ReplicaApt2"]
OLD_HOUSES = [f"HSSD{k}" for k in range(1, 9)]
SCENES = json.load(open("results/exp19_scenes.json")) if os.path.exists("results/exp19_scenes.json") else {"accepted": []}
FRL = [s for s in SCENES["accepted"] if s.startswith("ReplicaFRL")]
NEW_HOUSES = [s for s in SCENES["accepted"] if s.startswith("HSSD")]
GROUPS = {"new apartments (FRL)": FRL, "new houses": NEW_HOUSES, "new scenes": FRL + NEW_HOUSES,
          "pooled: all apartments": OLD_APTS + FRL, "pooled: all houses": OLD_HOUSES + NEW_HOUSES,
          "pooled: all photoreal": OLD_APTS + FRL + OLD_HOUSES + NEW_HOUSES}
LAYOUT = lambda s: "FRL" if s.startswith("ReplicaFRL") else s
SPLITS = ("all", "novel")


def cp(k, n):
    return [float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0, float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]


def sign_p(a, b):
    n = a + b
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n) if n else 1.0


def boot_mean(x, seed=0, n=10_000):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def walk_clustered(pairs, seed=0, n_boot=10_000, n_flip=20_000):
    """pairs: (walk id, a correct, b correct). Bootstrap CI over walks of the pooled difference, sign-flip p."""
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
    t = json.load(open(f"results/export/dissimilar-room{s}/trials.json"))
    return {r["id"]: r for r in (t["trials"] if isinstance(t, dict) else t)}


def load(path, man):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        t = man[r["id"]]
        assert (r["near"], r["far"], r["familiar"]) == (t["near"], t["far"], t["familiar"]), (path, r["id"])
        assert r["id"] not in rows, (path, "duplicate", r["id"])
        assert bool(r["correct"]) == (r["answer"] == r["near"]), (path, r["id"])
        rows[r["id"]] = bool(r["correct"])
    return rows


def model_path(s, model, cond):
    """(path, platform) of a model (a MODELS key) in scene s, or (None, None) if neither file exists."""
    stem = f"{MODELS[model]}-room{s}{CONDS[cond]}"
    found = [(stem + t + ".jsonl", p) for t, p in (("", "CUDA"), ("-mps", "MPS")) if os.path.exists(stem + t + ".jsonl")]
    assert len(found) <= 1, f"both CUDA and MPS runs of {stem}"
    return found[0] if found else (None, None)


def run_paths():
    """{run name: {scene: path}} over every scene of every group, and the platform of each scene's Qwen runs."""
    scenes = GROUPS["pooled: all photoreal"]
    paths, plat = {}, {}
    for cond in CONDS:
        for model in MODELS:
            name = f"{model}, {cond}"
            paths[name] = {}
            for s in scenes:
                p, pl = model_path(s, model, cond)
                paths[name][s] = p
                if pl and not model.startswith("Gemma"):
                    plat.setdefault(s, set()).add(pl)
    paths["Opus, walk"] = {s: f"results/claude/claude-opus-5-5-room{s}-low.jsonl" for s in scenes}
    paths["Opus, no walk"] = {s: f"results/claude/claude-opus-5-5-room{s}-nowalk-low.jsonl" for s in scenes}
    return paths, {s: "/".join(sorted(v)) for s, v in plat.items()}


def accuracy(data, man, scenes):
    per = {}
    for s in scenes:
        per[s] = {}
        for split in SPLITS:
            ids = [i for i, t in man[s].items() if split == "all" or not t["familiar"]]
            k = sum(data[s][i] for i in ids)
            per[s][split] = {"n": len(ids), "accuracy": k / len(ids) if ids else None, "ci95": cp(k, len(ids)) if ids else None}
    pooled = {}
    for split in ("all", "novel", "familiar"):
        ids = [(s, i) for s in scenes for i, t in man[s].items() if split == "all" or t["familiar"] == (split == "familiar")]
        k = sum(data[s][i] for s, i in ids)
        pooled[split] = {"n": len(ids), "correct": k, "accuracy": k / len(ids), "ci95": cp(k, len(ids))}
    out = {"per_scene": per, "pooled": pooled}
    for split in SPLITS:
        x = [per[s][split]["accuracy"] for s in scenes if per[s][split]["accuracy"] is not None]
        if len(x) >= 2:
            m, ci = boot_mean(x)
            out.setdefault("scene_mean", {})[split] = {"mean": m, "ci95": ci, "scenes": len(x)}
        lay = {}
        for s in scenes:
            if per[s][split]["accuracy"] is not None:
                lay.setdefault(LAYOUT(s), []).append(per[s][split]["accuracy"])
        if "FRL" in lay and len(lay) > 1:                 # descriptive: the FRL scenes as one layout
            L = {k: float(np.mean(v)) for k, v in lay.items()}
            out.setdefault("layout_mean", {})[split] = {"per_layout": L, "mean": float(np.mean(list(L.values()))), "n": len(L)}
    return out


def contrast(da, db, man, scenes):
    out = {}
    for split in SPLITS:
        pairs, per, lay = [], {}, {}
        for s in scenes:
            ids = [i for i, t in man[s].items() if split == "all" or not t["familiar"]]
            pairs += [(f"{s}-{man[s][i]['walk']}", da[s][i], db[s][i]) for i in ids]
            if ids:
                per[s] = float(np.mean([da[s][i] for i in ids]) - np.mean([db[s][i] for i in ids]))
                lay.setdefault(LAYOUT(s), []).append(per[s])
        up = sum(x and not y for _, x, y in pairs); dn = sum(y and not x for _, x, y in pairs)
        c = {"n": len(pairs), "difference": float(np.mean([x for _, x, _ in pairs]) - np.mean([y for _, _, y in pairs])),
             "better": up, "worse": dn, "sign_p": sign_p(up, dn), "walk_clustered": walk_clustered(pairs), "per_scene": per}
        if len(per) >= 2:
            d = list(per.values())
            m, ci = boot_mean(d)
            u, w = int(sum(x > 0 for x in d)), int(sum(x < 0 for x in d))
            c["scenes"] = {"mean": m, "ci95": ci, "up": u, "down": w, "sign_p": sign_p(u, w), "n": len(d)}
        if "FRL" in lay and len(lay) > 1:
            L = {k: float(np.mean(v)) for k, v in lay.items()}
            c["layouts"] = {"per_layout": L, "mean": float(np.mean(list(L.values()))), "n": len(L)}
        out[split] = c
    return out


def main():
    paths, plat = run_paths()
    allsc = GROUPS["pooled: all photoreal"]
    man = {s: manifest(s) for s in allsc}
    data, status = {}, {}
    for name, v in paths.items():
        data[name] = {}
        for s, p in v.items():
            if p is None or not os.path.exists(p):
                status.setdefault(name, {})[s] = "missing"
                continue
            rows = load(p, man[s])
            if len(rows) < len(man[s]):
                status.setdefault(name, {})[s] = f"incomplete {len(rows)}/{len(man[s])}"
                continue
            data[name][s] = rows
    opus_scenes = [s for s in allsc if s in data["Opus, walk"] and s in data["Opus, no walk"]]
    res = {"scenes": SCENES, "platform": plat, "opus_scenes": opus_scenes, "not_complete": status, "groups": {}}

    def have(name, scenes):
        return all(s in data[name] for s in scenes)

    for g, scenes in GROUPS.items():
        if not scenes:
            continue
        G = {"scenes": scenes, "accuracy": {}, "contrasts": {}, "recipes": {}}
        osc = [s for s in scenes if s in opus_scenes]
        G["opus_scenes"] = osc
        for name in data:
            sc = osc if name.startswith("Opus") else scenes
            if sc and have(name, sc):
                G["accuracy"][name] = accuracy(data[name], man, sc)
        todo = []
        for r in (m for m in MODELS if m not in ("base", "Gemma base")):
            todo += [(f"{r} walk - {BASE(r)} walk", f"{r}, walk", f"{BASE(r)}, walk", scenes),
                     (f"{r} walk - no walk", f"{r}, walk", f"{r}, no walk", scenes),
                     (f"{r} walk - shuffled", f"{r}, walk", f"{r}, shuffled", scenes),
                     (f"{r} walk - Opus walk", f"{r}, walk", "Opus, walk", osc)]
        todo += [("Opus walk - no walk", "Opus, walk", "Opus, no walk", osc), ("base walk - no walk", "base, walk", "base, no walk", scenes),
                 ("Gemma base walk - no walk", "Gemma base, walk", "Gemma base, no walk", scenes)]
        for cname, a, b, sc in todo:
            if sc and have(a, sc) and have(b, sc):
                G["contrasts"][cname] = contrast(data[a], data[b], man, sc)
        kind_name = lambda m, kind: {"vs base walk": f"{m} walk - {BASE(m)} walk", "vs no walk": f"{m} walk - no walk",
                                     "vs shuffled": f"{m} walk - shuffled", "vs Opus walk": f"{m} walk - Opus walk"}[kind]
        for rec, runs in RECIPES.items():
            done = [m for m in runs if f"{m}, walk" in G["accuracy"]]
            if not done:
                continue
            R = {"runs": done}
            for split in SPLITS:
                acc = [G["accuracy"][f"{m}, walk"]["pooled"][split]["accuracy"] for m in done]
                R.setdefault(split, {})["walk accuracy, pooled"] = {"values": acc, "mean": float(np.mean(acc)), "min": min(acc), "max": max(acc)}
                sm = [G["accuracy"][f"{m}, walk"].get("scene_mean", {}).get(split, {}).get("mean") for m in done]
                if None not in sm:
                    R[split]["walk accuracy, scene mean"] = {"values": sm, "mean": float(np.mean(sm)), "min": min(sm), "max": max(sm)}
                for c in KINDS:
                    v = [G["contrasts"][kind_name(m, c)][split] for m in done if kind_name(m, c) in G["contrasts"]]
                    if len(v) == len(done):
                        for unit in ("difference", "scenes"):
                            x = [z[unit] if unit == "difference" else z[unit]["mean"] for z in v if unit == "difference" or unit in z]
                            if len(x) == len(done):
                                R[split][f"{c} ({'pooled' if unit == 'difference' else 'scene mean'})"] = {
                                    "values": x, "mean": float(np.mean(x)), "min": min(x), "max": max(x)}
            G["recipes"][rec] = R
        res["groups"][g] = G

    # Registered replication criteria.
    crit = {}
    H, A = res["groups"].get("new houses"), res["groups"].get("new apartments (FRL)")
    for m in RECIPES["four-epoch runs 3-7"]:
        for c in ("walk - base walk", "walk - no walk", "walk - shuffled"):
            name = f"{m} {c}"
            if H and name in H["contrasts"] and "scenes" in H["contrasts"][name]["all"]:
                crit.setdefault("1. LoRA transfers: new houses (scene CI above zero)", {})[name] = H["contrasts"][name]["all"]["scenes"]["ci95"][0] > 0
            if A and name in A["contrasts"]:
                x = A["contrasts"][name]["all"]
                crit.setdefault("1. LoRA transfers: FRL apartments (pooled sign p < 0.05)", {})[name] = x["difference"] > 0 and x["sign_p"] < 0.05
    N = res["groups"].get("new scenes")
    if N and "Opus walk - no walk" in N["contrasts"]:
        x = N["contrasts"]["Opus walk - no walk"]["all"]
        crit["2. Opus walk gain (new Opus scenes, pooled sign p < 0.05)"] = x["difference"] > 0 and x["sign_p"] < 0.05
    for key, v in list(crit.items()):
        if isinstance(v, dict):
            crit[key] = {"met": all(v.values()) and len(v) == 15, "checks": v}
    res["criteria"] = crit
    json.dump(res, open(OUT, "w"), indent=1)

    print(f"Experiment 19. Accepted: {SCENES['accepted']}; Opus scenes: {opus_scenes}")
    if status:
        print("not complete: " + "; ".join(f"{n}: {len(v)} scenes" for n, v in status.items()))
    for g, G in res["groups"].items():
        print(f"\n=== {g}: {len(G['scenes'])} scenes; Opus on {len(G['opus_scenes'])}")
        for name, a in G["accuracy"].items():
            p, sm = a["pooled"], a.get("scene_mean", {})
            s = f"  {name:18s} all {100 * p['all']['accuracy']:5.1f}% ({p['all']['n']}) [{100 * p['all']['ci95'][0]:.1f}, {100 * p['all']['ci95'][1]:.1f}]"
            s += f"  novel {100 * p['novel']['accuracy']:5.1f}% ({p['novel']['n']})"
            if "all" in sm:
                s += f"  scene mean {100 * sm['all']['mean']:.1f} [{100 * sm['all']['ci95'][0]:.1f}, {100 * sm['all']['ci95'][1]:.1f}]"
            print(s)
        for cname, c in G["contrasts"].items():
            for split in SPLITS:
                x = c[split]
                s = f"  {cname:26s} {split:5s} {100 * x['difference']:+5.1f} ({x['better']}/{x['worse']}, p {x['sign_p']:.2g})"
                if "scenes" in x:
                    h = x["scenes"]
                    s += f"  scenes {100 * h['mean']:+5.1f} [{100 * h['ci95'][0]:+.1f}, {100 * h['ci95'][1]:+.1f}] {h['up']}/{h['down']}"
                if "layouts" in x:
                    s += f"  layouts {100 * x['layouts']['mean']:+5.1f} (n {x['layouts']['n']})"
                print(s)
        for rec, R in G["recipes"].items():
            for split in SPLITS:
                for m, v in R.get(split, {}).items():
                    f = ".1f" if m.startswith("walk accuracy") else "+.1f"
                    print(f"  [{rec}] {split:5s} {m:36s} mean {100 * v['mean']:{f}} (range {100 * v['min']:{f}} to {100 * v['max']:{f}})")
    print("\nCriteria:", json.dumps({k: (v["met"] if isinstance(v, dict) else v) for k, v in crit.items()}))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
