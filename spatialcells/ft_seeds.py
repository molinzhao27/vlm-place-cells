"""Room B and training seeds for the fine-tuned Qwen3-VL-8B (experiment 9b).

    python -m spatialcells.ft_seeds

Reads two_choice.py's per-trial results for base Qwen3-VL-8B and the LoRA adapters
(data/ft/qwen8b-proc = seed 0, qwen8b-proc-s1, -s2) on room A's and room B's lure-free trials,
and prints, per room and condition (64-frame walk, no walk, shuffled walk): accuracy on novel
and familiar trials for each seed, the 3-seed mean and range, and paired two-sided sign tests
by trial (each seed vs base with the same walk, ordered vs no walk, ordered vs shuffled).
Each run is the CUDA one (vast-1) if it exists, else the same run on Apple MPS (tag + "-mps"; the
room-B runs of base and seed 0); the JSON names the file used. Writes results/two_choice/ft_seeds.json.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .two_choice import OUT

BASE = "Qwen3-VL-8B-Instruct"
SEEDS = {0: "qwen8b-proc", 1: "qwen8b-proc-s1", 2: "qwen8b-proc-s2"}
ROOMS = {"A": "dissimilar", "B": "roomB"}
CONDS = {"walk": "-max64", "nowalk": "-nowalk", "shuffled": "-max64-shuffled"}


def load(tag):
    for t in (tag, f"{tag}-mps"):
        if os.path.exists(f"{OUT}/{t}.jsonl"):
            return t, {r["id"]: r for r in map(json.loads, open(f"{OUT}/{t}.jsonl"))}
    return None, None


def sign_test(x, y, ids):
    """x better / worse than y on the same trials, two-sided sign test."""
    b = sum(x[i]["correct"] and not y[i]["correct"] for i in ids)
    w = sum(y[i]["correct"] and not x[i]["correct"] for i in ids)
    n = b + w
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, w) + 1)) / 2 ** n) if n else 1.0
    return {"better": b, "worse": w, "p": p}


def main():
    res = {}
    for room, trials in ROOMS.items():
        runs, tags = {}, {}
        for cond, suf in CONDS.items():
            tags[("base", cond)], runs[("base", cond)] = load(f"{BASE}-{trials}-256px{suf}")
            for s, a in SEEDS.items():
                tags[(s, cond)], runs[(s, cond)] = load(f"{BASE}+{a}-{trials}-256px{suf}")
        out = {}
        for cat in ("novel", "familiar"):
            o = {}
            for (who, cond), r in runs.items():
                if r is None:
                    continue
                ids = [i for i, t in r.items() if t["familiar"] == (cat == "familiar")]
                k = sum(r[i]["correct"] for i in ids)
                o[f"{who}/{cond}"] = {"tag": tags[(who, cond)], "k": k, "n": len(ids), "acc": k / len(ids) if ids else float("nan"),
                                      "p_above_chance": sum(math.comb(len(ids), i) for i in range(k, len(ids) + 1)) / 2 ** len(ids)}
                if who != "base" and runs[("base", cond)] is not None:
                    o[f"{who}/{cond}"]["vs_base"] = sign_test(r, runs[("base", cond)], ids)
                if who != "base" and cond != "walk" and runs[(who, "walk")] is not None:
                    o[f"{who}/{cond}"]["walk_vs_this"] = sign_test(runs[(who, "walk")], r, ids)
            for cond in CONDS:
                accs = [o[f"{s}/{cond}"]["acc"] for s in SEEDS if f"{s}/{cond}" in o]
                if accs:
                    o[f"seeds/{cond}"] = {"n_seeds": len(accs), "mean": float(np.mean(accs)),
                                          "min": float(min(accs)), "max": float(max(accs))}
            out[cat] = o
        res[room] = out
        for cat, o in out.items():
            print(f"room {room}, {cat}:")
            for key, v in o.items():
                if key.startswith("seeds/"):
                    print(f"  {key:16s} {v['mean']:.0%} (range {v['min']:.0%}-{v['max']:.0%}, {v['n_seeds']} seeds)")
                else:
                    extra = "".join(f", {lab} {v[t]['better']} better / {v[t]['worse']} worse (p={v[t]['p']:.2g})"
                                    for t, lab in (("vs_base", "vs base"), ("walk_vs_this", "walk vs this")) if t in v)
                    mps = " [mps]" if v["tag"].endswith("-mps") else ""
                    print(f"  {key:16s} {v['acc']:.0%} ({v['k']}/{v['n']}, p={v['p_above_chance']:.2g}){extra}{mps}")
    json.dump(res, open(f"{OUT}/ft_seeds.json", "w"), indent=1)
    print(f"-> {OUT}/ft_seeds.json")


if __name__ == "__main__":
    main()
