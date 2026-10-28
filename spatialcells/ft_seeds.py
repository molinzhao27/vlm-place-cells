"""Room B and training seeds for the fine-tuned Qwen3-VL-8B (experiment 9b).

    python -m spatialcells.ft_seeds

Reads two_choice.py's per-trial results for base Qwen3-VL-8B and the LoRA adapters
(data/ft/qwen8b-proc = seed 0, qwen8b-proc-s1, -s2) on room A's and room B's lure-free trials,
and prints, per room and condition (64-frame walk, no walk, shuffled walk): accuracy on novel
and familiar trials for each seed, the 3-seed mean and range, and paired two-sided sign tests
by trial (each seed vs base with the same walk, ordered vs no walk, ordered vs shuffled).
Each run is the CUDA one (vast-1) if it exists, else the same run on Apple MPS (tag + "-mps"; the
room-B runs of base and seed 0); the JSON names the file used. Writes results/two_choice/ft_seeds.json.
The GPU control (a second run of the seed-0 configuration on vast-3's RTX 5090, qwen8b-proc-s0-5090; seed 0
does not fix torch's generator, so its initialisation differs from the original's) is reported as "0-5090",
with its answers compared to seed 0's, and is left out of the three-seed mean.
Experiment 9c's four-epoch runs (seeds 3-7, qwen8b-proc-e4-s<seed>) are reported per run and summarised
separately ("runs4ep/<cond>"), never pooled with the two-epoch seeds. "training" holds each run's training
checks from results/two_choice/ft_seed_logs/train_log_<run>.json, and "decision_rule_9c" evaluates
experiment 9c's registered rule once all five runs are in.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .two_choice import OUT

BASE = "Qwen3-VL-8B-Instruct"
SEEDS = {0: "qwen8b-proc", 1: "qwen8b-proc-s1", 2: "qwen8b-proc-s2"}
CONTROL = {"0-5090": "qwen8b-proc-s0-5090"}
E4 = {f"e4-s{k}": f"qwen8b-proc-e4-s{k}" for k in range(3, 8)}       # experiment 9c
LOGS = {"0": "s0", "1": "s1", "2": "s2", "0-5090": "s0-5090", **{k: k for k in E4}}
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


def training_checks():
    """Per run: first update with loss < 0.15, the last logged loss, held-out accuracy before and at the end."""
    out = {}
    for run, name in LOGS.items():
        p = f"{OUT}/ft_seed_logs/train_log_{name}.json"
        if not os.path.exists(p):
            continue
        log = json.load(open(p))
        loss = [(r["update"], r["loss"]) for r in log if "loss" in r]
        val = [(r["update"], r["val_acc"]) for r in log if "val_acc" in r]
        out[run] = {"plateau_exit_update": next((u for u, l in loss if l < 0.15), None),
                    "loss_end": loss[-1][1], "loss_end_update": loss[-1][0], "updates": val[-1][0],
                    "held_out_start": val[0][1], "held_out_end": val[-1][1]}
        print(f"training {run:7s}: loss < 0.15 at update {out[run]['plateau_exit_update']}, loss {loss[-1][1]:.3f} at "
              f"update {loss[-1][0]}, held-out {val[0][1]:.0%} -> {val[-1][1]:.0%} (of {val[-1][0]} updates)")
    return out


def decision_rule(res):
    """Experiment 9c's registered rule: every run ends >= 85% held-out, five-run room-A novel mean >= 85%, none < 80%."""
    tr = res["training"]
    if not all(k in tr for k in E4) or not all(f"{k}/walk" in res["A"]["novel"] for k in E4):
        return None
    held = {k: tr[k]["held_out_end"] for k in E4}
    room_a = {k: res["A"]["novel"][f"{k}/walk"]["acc"] for k in E4}
    c = {"all_held_out_at_least_85": all(v >= 0.85 for v in held.values()),
         "room_a_novel_mean_at_least_85": float(np.mean(list(room_a.values()))) >= 0.85,
         "no_run_below_80_room_a": min(room_a.values()) >= 0.80}
    c.update({"held_out_end": held, "room_a_novel_walk": room_a, "met": all(c.values())})
    print(f"experiment 9c decision rule met: {c['met']} ({', '.join(k for k, v in c.items() if v is True or v is False)})")
    return c


def main():
    res = {}
    for room, trials in ROOMS.items():
        runs, tags = {}, {}
        for cond, suf in CONDS.items():
            tags[("base", cond)], runs[("base", cond)] = load(f"{BASE}-{trials}-256px{suf}")
            for s, a in {**SEEDS, **CONTROL, **E4}.items():
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
                if who in CONTROL and runs[(0, cond)] is not None:
                    o[f"{who}/{cond}"]["vs_seed0"] = sign_test(r, runs[(0, cond)], ids)
                    o[f"{who}/{cond}"]["same_answer_as_seed0"] = sum(r[i]["answer"] == runs[(0, cond)][i]["answer"] for i in ids)
            for cond in CONDS:
                accs = [o[f"{s}/{cond}"]["acc"] for s in SEEDS if f"{s}/{cond}" in o]
                if accs:
                    o[f"seeds/{cond}"] = {"n_seeds": len(accs), "mean": float(np.mean(accs)),
                                          "min": float(min(accs)), "max": float(max(accs))}
                accs = [o[f"{s}/{cond}"]["acc"] for s in E4 if f"{s}/{cond}" in o]
                if accs:
                    o[f"runs4ep/{cond}"] = {"n_seeds": len(accs), "mean": float(np.mean(accs)),
                                            "min": float(min(accs)), "max": float(max(accs))}
            out[cat] = o
        res[room] = out
        for cat, o in out.items():
            print(f"room {room}, {cat}:")
            for key, v in o.items():
                if key.startswith(("seeds/", "runs4ep/")):
                    print(f"  {key:16s} {v['mean']:.0%} (range {v['min']:.0%}-{v['max']:.0%}, {v['n_seeds']} seeds)")
                else:
                    extra = "".join(f", {lab} {v[t]['better']} better / {v[t]['worse']} worse (p={v[t]['p']:.2g})"
                                    for t, lab in (("vs_base", "vs base"), ("walk_vs_this", "walk vs this"), ("vs_seed0", "vs seed 0")) if t in v)
                    if "same_answer_as_seed0" in v:
                        extra += f", same answer as seed 0 on {v['same_answer_as_seed0']}/{v['n']}"
                    mps = " [mps]" if v["tag"].endswith("-mps") else ""
                    print(f"  {key:16s} {v['acc']:.0%} ({v['k']}/{v['n']}, p={v['p_above_chance']:.2g}){extra}{mps}")
    res["training"] = training_checks()
    rule = decision_rule(res)
    if rule:
        res["decision_rule_9c"] = rule
    json.dump(res, open(f"{OUT}/ft_seeds.json", "w"), indent=1)
    print(f"-> {OUT}/ft_seeds.json")


if __name__ == "__main__":
    main()
