"""Training seeds 1-2 in every new scene (amendment to experiments 9b, 10, 12 and 16 in docs/experiments.md).

    .venv-claude/bin/python scripts/seed_scenes_analysis.py      # -> results/seed_scenes_summary.json

For each experiment and each LoRA seed (0 = data/ft/qwen8b-proc, 1-2 = qwen8b-proc-s1/-s2): accuracy with the
64-frame walk, without it and with it shuffled, and the experiment's registered contrasts that involve the
LoRA model, then the three-seed mean and range. The units are those of each experiment's own analysis:
- experiment 10 (ten held-out procedural rooms): novel trials; mean over rooms with a 95% bootstrap CI over
  rooms (10,000 resamples, seed 0), room-level sign test, and the pooled trial-level sign test;
- experiment 12 (three Replica apartments): all lure-free trials (primary) and novel trials, pooled;
  trial-level sign test and a bootstrap over walks (as scripts/photoreal_analysis.py);
- experiment 16 (eight HSSD houses): as experiment 12, plus houses as the unit (as scripts/hssd_analysis.py).
Each contrast is on the same trials for both runs. Every run is checked against its scene's trial manifest;
a seed whose runs are missing or incomplete in an experiment is reported as such and left out of the mean.

Experiment 9c's four-epoch runs (seeds 3-7) are analysed the same way and summarised separately
("four_epoch_runs"); the two recipes are never pooled.

Platforms: Qwen ran on the Mac (MPS, files tagged -mps) or on CUDA (untagged). For base and seed 0 the
files are those of each experiment's analysis; for seeds 1-2 whichever exists (the JSON names it), and a
scene with both is an error. Seed 0 here reproduces the seed-0 accuracies, differences and counts of
heldout_summary.json, photoreal_summary.json and hssd_summary.json; experiment 10's bootstrap CIs can differ
from heldout_summary.json in the last digit, because that script draws all its resamples from one stream.
"""
import json
import math
import os

import numpy as np

Q = "results/two_choice/Qwen3-VL-8B-Instruct"
SEEDS = {0: "qwen8b-proc", 1: "qwen8b-proc-s1", 2: "qwen8b-proc-s2",
         **{k: f"qwen8b-proc-e4-s{k}" for k in range(3, 8)}}             # 3-7: experiment 9c, four epochs
RECIPES = {"three_seeds": (0, 1, 2), "four_epoch_runs": (3, 4, 5, 6, 7)}    # summary key -> runs (never mixed)
CONDS = {"walk": "-256px-max64", "no walk": "-256px-nowalk", "shuffled": "-256px-max64-shuffled"}
MAC_SEED0 = {"10": "all", "12": "all", "16": {"HSSD1", "HSSD2", "HSSD3"}}     # scenes whose base/seed-0 runs are -mps
EXPS = {
    "10": {"scenes": [f"P{s}" for s in range(5000, 5010)], "splits": ["novel"], "scene_unit": True},
    "12": {"scenes": ["ReplicaApt0", "ReplicaApt1", "ReplicaApt2"], "splits": ["all", "novel"], "scene_unit": False},
    "16": {"scenes": [f"HSSD{k}" for k in range(1, 9)], "splits": ["all", "novel"], "scene_unit": True},
}
CONTRASTS = [("LoRA walk vs base walk", "LoRA, walk", "base, walk"),
             ("LoRA walk vs no walk", "LoRA, walk", "LoRA, no walk"),
             ("LoRA walk vs shuffled", "LoRA, walk", "LoRA, shuffled"),
             ("LoRA walk vs Opus walk", "LoRA, walk", "Opus, walk")]


def sign_p(a, b):
    n = a + b
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n) if n else 1.0


def boot_mean(x, seed=0, n=10_000):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def walk_clustered(pairs, seed=0, n_boot=10000):
    """pairs: (walk id, a correct, b correct); 95% bootstrap CI over walks of the pooled difference."""
    rng = np.random.default_rng(seed)
    ids = sorted({c for c, _, _ in pairs})
    d = {c: np.array([int(x) - int(y) for k, x, y in pairs if k == c]) for c in ids}
    boot = [np.concatenate([d[c] for c in rng.choice(ids, len(ids))]).mean() for _ in range(n_boot)]
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


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


def qwen_path(exp, s, seed, cond):
    """The file for base (seed None) or a seed; returns (path or None, platform)."""
    a = "" if seed is None else "+" + SEEDS[seed]
    stem = f"{Q}{a}-room{s}{CONDS[cond]}"
    if seed in (None, 0):
        mac = MAC_SEED0[exp] == "all" or s in MAC_SEED0[exp]
        p = stem + ("-mps" if mac else "") + ".jsonl"
        return (p if os.path.exists(p) else None), ("MPS" if mac else "CUDA")
    found = [(stem + t + ".jsonl", plat) for t, plat in (("", "CUDA"), ("-mps", "MPS")) if os.path.exists(stem + t + ".jsonl")]
    assert len(found) <= 1, f"both CUDA and MPS runs of {stem}"
    return found[0] if found else (None, None)


def runs_for(exp, seed):
    """{run name: {scene: path}} for one seed, plus the platform per scene; None entries mark missing files."""
    sc = EXPS[exp]["scenes"]
    out, plat = {}, {}
    for cond in CONDS:
        out[f"base, {cond}"] = {s: qwen_path(exp, s, None, cond)[0] for s in sc}
        lora = {s: qwen_path(exp, s, seed, cond) for s in sc}
        out[f"LoRA, {cond}"] = {s: p for s, (p, _) in lora.items()}
        for s, (_, pl) in lora.items():
            plat.setdefault(s, set()).add(pl)
    out["Opus, walk"] = {s: f"results/claude/claude-opus-5-5-room{s}-low.jsonl" for s in sc}
    return out, {s: "/".join(sorted(p for p in v if p)) for s, v in plat.items()}


def analyse(exp, seed):
    cfg = EXPS[exp]
    sc = cfg["scenes"]
    man = {s: manifest(s) for s in sc}
    paths, plat = runs_for(exp, seed)
    missing = {n: [s for s, p in v.items() if p is None or not os.path.exists(p)] for n, v in paths.items()}
    missing = {n: v for n, v in missing.items() if v}
    if missing:
        return {"status": "missing", "missing": missing}
    data = {n: {s: load(p, man[s]) for s, p in v.items()} for n, v in paths.items()}
    short = {n: [s for s in sc if len(data[n][s]) != len(man[s])] for n in data if n != "Opus, walk"}
    short = {n: v for n, v in short.items() if v}
    if short:
        return {"status": "incomplete", "incomplete": short}
    out = {"status": "done", "platform": plat,
           "files": {n: v for n, v in paths.items() if n.startswith("LoRA")}, "accuracy": {}, "contrasts": {}}
    for split in cfg["splits"]:
        ids = {s: [i for i, t in man[s].items() if split == "all" or not t["familiar"]] for s in sc}
        for n in ("LoRA, walk", "LoRA, no walk", "LoRA, shuffled"):
            k = sum(data[n][s][i] for s in sc for i in ids[s]); m = sum(map(len, ids.values()))
            per = [float(np.mean([data[n][s][i] for i in ids[s]])) for s in sc]
            o = {"pooled": k / m, "n": m, "per_scene": dict(zip(sc, per))}
            if cfg["scene_unit"]:
                o["scene_mean"], o["scene_ci95"] = boot_mean(per)
            out["accuracy"].setdefault(split, {})[n] = o
        for cname, a, b in CONTRASTS:
            mid = {s: [i for i in ids[s] if i in data[a][s] and i in data[b][s]] for s in sc}
            pairs = [(f"{s}-{man[s][i]['walk']}", data[a][s][i], data[b][s][i]) for s in sc for i in mid[s]]
            up = sum(x and not y for _, x, y in pairs); dn = sum(y and not x for _, x, y in pairs)
            diff = float(np.mean([x for _, x, _ in pairs]) - np.mean([y for _, _, y in pairs]))
            c = {"pooled_difference": diff, "better": up, "worse": dn, "sign_p": sign_p(up, dn), "n": len(pairs),
                 "walk_clustered_ci95": walk_clustered(pairs)}
            if cfg["scene_unit"]:
                d = [float(np.mean([data[a][s][i] for i in mid[s]]) - np.mean([data[b][s][i] for i in mid[s]])) for s in sc]
                pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
                m, ci = boot_mean(d)
                c.update({"scene_mean": m, "scene_ci95": ci, "scenes_up": int(pos), "scenes_down": int(neg),
                          "scene_sign_p": sign_p(pos, neg), "per_scene": dict(zip(sc, d))})
            out["contrasts"].setdefault(split, {})[cname] = c
    return out


def summarise(exp, cfg, seeds, label):
    """Mean and range over the runs of one recipe that are done, with each run's contrasts printed."""
    done = [s for s, v in seeds.items() if v["status"] == "done"]
    print(f"\n=== experiment {exp} ({len(cfg['scenes'])} scenes), {label}; runs done: {done}")
    for seed, v in seeds.items():
        if v["status"] != "done":
            print(f"  seed {seed}: {v['status']} {v.get('missing') or v.get('incomplete')}"[:300])
    if not done:
        return None
    key = "scene_mean" if cfg["scene_unit"] else "pooled_difference"
    out = {}
    for split in cfg["splits"]:
        print(f"  {split} trials" + (" (scene mean [95% CI over scenes])" if cfg["scene_unit"] else " (pooled)"))
        for n in ("LoRA, walk", "LoRA, no walk", "LoRA, shuffled"):
            vals = []
            for seed in done:
                o = seeds[seed]["accuracy"][split][n]
                vals.append(o["scene_mean"] if cfg["scene_unit"] else o["pooled"])
            out.setdefault(split, {})[n] = {"seeds": done, "values": vals, "mean": float(np.mean(vals)),
                                            "min": float(min(vals)), "max": float(max(vals))}
            print(f"    {n:16s} " + "  ".join(f"s{s} {x:.1%}" for s, x in zip(done, vals))
                  + f"   mean {np.mean(vals):.1%} (range {min(vals):.1%}-{max(vals):.1%})")
        for cname, _, _ in CONTRASTS:
            vals = [seeds[s]["contrasts"][split][cname][key] for s in done]
            out[split][cname] = {"seeds": done, "values": vals, "mean": float(np.mean(vals)),
                                 "min": float(min(vals)), "max": float(max(vals))}
            cells = []
            for s in done:
                c = seeds[s]["contrasts"][split][cname]
                cell = f"s{s} {100 * c[key]:+.1f}"
                if cfg["scene_unit"]:
                    cell += f" [{100 * c['scene_ci95'][0]:+.1f}, {100 * c['scene_ci95'][1]:+.1f}] {c['scenes_up']}/{c['scenes_down']}"
                cell += f" ({c['better']}/{c['worse']}, p {c['sign_p']:.2g})"
                cells.append(cell)
            print(f"    {cname:24s} " + "; ".join(cells))
    return out


def main():
    res = {}
    for exp, cfg in EXPS.items():
        res[exp] = {"seeds": {seed: analyse(exp, seed) for seed in SEEDS}}
        for key, runs in RECIPES.items():
            label = "two-epoch runs" if key == "three_seeds" else "four-epoch runs (experiment 9c)"
            summary = summarise(exp, cfg, {s: res[exp]["seeds"][s] for s in runs}, label)
            if summary is not None:
                res[exp][key] = summary
    json.dump(res, open("results/seed_scenes_summary.json", "w"), indent=1)
    print("\n-> results/seed_scenes_summary.json")


if __name__ == "__main__":
    main()
