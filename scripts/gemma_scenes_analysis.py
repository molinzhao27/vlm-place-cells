"""Experiment 18 (docs/experiments.md): Gemma-4-12B-it's four-epoch LoRA runs 3-5 in every new scene.

    .venv-claude/bin/python scripts/gemma_scenes_analysis.py      # -> results/gemma_scenes_summary.json

The same analysis as scripts/seed_scenes_analysis.py (whose functions it reuses), for base Gemma-4-12B-it
and the runs gemma12b-proc-e4-s<seed>, seeds 3-5 (amendment 2 stopped the experiment before runs 6-7):
per experiment (10: ten held-out procedural rooms, novel trials, room means; 12: three Replica apartments, all and novel trials, pooled; 16: eight HSSD houses, all
and novel trials, house means and pooled) and per run, accuracy with the 64-frame walk, without it and
shuffled, and each run vs base Gemma with the walk, vs no walk, vs shuffled and vs Opus 5.5 with the walk;
then the three-run mean and range. Base Gemma's files, like the runs', are whichever platform exists (CUDA
untagged, Apple MPS tagged -mps; both is an error). The layout is seed_scenes_summary.json's ("seeds",
"four_epoch_runs"), plus "base": base Gemma's own accuracy with the walk, without it and shuffled, on the
same units (descriptive; base enters the tests only through each run's "LoRA walk vs base walk"). Qwen's
four-epoch summary (experiment 9c, from results/seed_scenes_summary.json) is copied in as
"qwen_four_epoch_runs" for the side-by-side; the two families are never pooled and not tested against
each other.
"""
import importlib.util
import json
import os

import numpy as np

spec = importlib.util.spec_from_file_location("ssa", os.path.join(os.path.dirname(__file__), "seed_scenes_analysis.py"))
ssa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssa)

ssa.Q = "results/two_choice/gemma-4-12B-it"
ssa.SEEDS = {k: f"gemma12b-proc-e4-s{k}" for k in range(3, 6)}
RUNS = (3, 4, 5)                                     # amendment 2: runs 6-7 never trained


def gemma_path(exp, s, seed, cond):
    """Base (seed None) or a run: whichever of the CUDA and MPS files exists."""
    a = "" if seed is None else "+" + ssa.SEEDS[seed]
    stem = f"{ssa.Q}{a}-room{s}{ssa.CONDS[cond]}"
    found = [(stem + t + ".jsonl", plat) for t, plat in (("", "CUDA"), ("-mps", "MPS")) if os.path.exists(stem + t + ".jsonl")]
    assert len(found) <= 1, f"both CUDA and MPS runs of {stem}"
    return found[0] if found else (None, None)


ssa.qwen_path = gemma_path


def base_accuracy(exp):
    """Base Gemma per condition and split, on the units of ssa.analyse's LoRA accuracies."""
    cfg = ssa.EXPS[exp]
    sc = cfg["scenes"]
    man = {s: ssa.manifest(s) for s in sc}
    paths = {c: {s: gemma_path(exp, s, None, c) for s in sc} for c in ssa.CONDS}
    missing = {c: [s for s, (p, _) in v.items() if p is None] for c, v in paths.items()}
    if any(missing.values()):
        return {"status": "missing", "missing": {c: v for c, v in missing.items() if v}}
    data = {c: {s: ssa.load(p, man[s]) for s, (p, _) in v.items()} for c, v in paths.items()}
    short = {c: [s for s in sc if len(data[c][s]) != len(man[s])] for c in data}
    if any(short.values()):
        return {"status": "incomplete", "incomplete": {c: v for c, v in short.items() if v}}
    out = {"status": "done", "platform": {s: "/".join(sorted({paths[c][s][1] for c in data})) for s in sc}, "accuracy": {}}
    for split in cfg["splits"]:
        ids = {s: [i for i, t in man[s].items() if split == "all" or not t["familiar"]] for s in sc}
        for c in ssa.CONDS:
            k = sum(data[c][s][i] for s in sc for i in ids[s]); m = sum(map(len, ids.values()))
            per = [float(np.mean([data[c][s][i] for i in ids[s]])) for s in sc]
            o = {"pooled": k / m, "n": m, "per_scene": dict(zip(sc, per))}
            if cfg["scene_unit"]:
                o["scene_mean"], o["scene_ci95"] = ssa.boot_mean(per)
            out["accuracy"].setdefault(split, {})[f"base, {c}"] = o
    print(f"  base Gemma: " + "; ".join(f"{split} {n} {o.get('scene_mean', o['pooled']):.1%}"
                                         for split, v in out["accuracy"].items() for n, o in v.items()))
    return out


def main():
    qwen = json.load(open("results/seed_scenes_summary.json"))
    res = {}
    for exp, cfg in ssa.EXPS.items():
        res[exp] = {"seeds": {seed: ssa.analyse(exp, seed) for seed in RUNS}}
        summary = ssa.summarise(exp, cfg, res[exp]["seeds"], "Gemma-4-12B-it four-epoch runs (experiment 18)")
        if summary is not None:
            res[exp]["four_epoch_runs"] = summary
        res[exp]["base"] = base_accuracy(exp)
        if "four_epoch_runs" in qwen.get(exp, {}):
            res[exp]["qwen_four_epoch_runs"] = qwen[exp]["four_epoch_runs"]
    json.dump(res, open("results/gemma_scenes_summary.json", "w"), indent=1)
    print("\n-> results/gemma_scenes_summary.json")


if __name__ == "__main__":
    main()
