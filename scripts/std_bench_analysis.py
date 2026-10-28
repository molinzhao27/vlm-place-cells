"""Experiment 17: standard spatial benchmarks (VSI-Bench, MindCube-Tiny) for the base model and the LoRA runs.

    .venv-mac/bin/python scripts/std_bench_analysis.py [--boot 10000]
    .venv/bin/python scripts/std_bench_analysis.py --gate      # the registered check of the base model's VSI-Bench run

Reads results/std_bench/<bench>/<model>[-blind].jsonl (spatialcells/std_bench.py) and writes
results/std_bench_summary.json (docs/experiments.md, experiment 17): every model's official scores
(overall and per task), how many replies the official parser could read, and the registered paired
contrasts with 95% bootstrap CIs and their labels.

A contrast is a linear combination of models' official scores (a model minus the base, or a recipe's
mean over runs minus the base) on the questions every model in it answered. Its main CI resamples scenes
(VSI-Bench: videos; MindCube: image groups) with replacement, keeping every question of a drawn scene; a
second CI resamples questions within each question type. 10,000 resamples, seed 0; the runs of a recipe
are held fixed. Scores are in points (0-100). VSI-Bench's official score is the mean of its eight task
scores, relative direction being the mean of its three levels; MindCube's is accuracy over questions.
--gate exits 1 unless the base model answered all 5,130 VSI-Bench questions, at least 95% of its
replies can be read, and its score is at least 49.9 (8 points below the published 57.9).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

OUT = os.environ.get("STD_BENCH_OUT", "results/std_bench")
SUMMARY = os.environ.get("STD_BENCH_SUMMARY", "results/std_bench_summary.json")
BASE = "base"
RUNS_2EP = ["run0", "run1", "run2"]
RUNS_4EP = ["e4s3", "e4s4", "e4s5", "e4s6", "e4s7"]
ADAPTER = {"run0": "qwen8b-proc", "run1": "qwen8b-proc-s1", "run2": "qwen8b-proc-s2",
           **{f"e4s{s}": f"qwen8b-proc-e4-s{s}" for s in range(3, 8)}}
MARGIN = {"vsibench": 2.0, "vsibench_debiased": 3.0, "mindcube": 5.0}      # "no change" margins, points
N_FULL = {"vsibench": 5130, "vsibench_debiased": 2362, "mindcube": 1050}
VSI_DIR = ["object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"]
VSI_TASKS = ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation",
             "object_rel_distance", "object_rel_direction", "route_planning", "obj_appearance_order"]
VSI_VIEW = ["object_rel_direction", "object_rel_distance", "route_planning", "obj_appearance_order"]
VSI_METRIC = ["object_counting", "object_size_estimation", "room_size_estimation", "object_abs_distance"]
MC_SETTINGS = ["among", "around", "rotation"]


def load(bench):
    """{file tag: {question id: record}}."""
    out = {}
    for path in sorted(glob.glob(f"{OUT}/{bench}/*.jsonl")):
        recs = {}
        for line in open(path):
            r = json.loads(line)
            recs[r["id"]] = r
        out[os.path.basename(path)[:-6]] = recs
    return out


def mindcube_yes_no():
    """Ids of MindCube-Tiny questions whose two options are Yes and No."""
    ids = set()
    for line in open("data/bench/mindcube/data/raw/MindCube_tinybench.jsonl"):
        q = json.loads(line)
        opts = re.findall(r"\b([A-E])\.\s*(.*?)(?=\s+[A-E]\.\s|$)", q["question"])
        if sorted(v.strip().rstrip(".").strip().lower() for _, v in opts) == ["no", "yes"]:
            ids.add(q["id"])
    return ids


class Bench:
    """One benchmark's records, with the official score of any set of models, under any question weights."""

    def __init__(self, name, data):
        self.name = name
        self.vsi = name.startswith("vsibench")
        self.data = data

    def ids(self, models, subset=None):
        ids = set.intersection(*[set(self.data[m]) for m in models])
        if self.name == "vsibench_debiased":
            ids = {i for i in ids if not self.data[models[0]][i]["pruned"]}
        if subset is not None:
            ids &= subset
        return sorted(ids)

    def qtype(self, r):
        return r["question_type"] if self.vsi else r["setting"]

    def task(self, r):
        t = self.qtype(r)
        return "object_rel_direction" if t in VSI_DIR else t

    def setup(self, models, ids, key="score"):
        """Arrays for scoring: scores (models x questions), question-type one-hot, cluster index."""
        ref = self.data[models[0]]
        types = sorted({self.qtype(ref[i]) for i in ids})
        T = np.zeros((len(ids), len(types)))
        T[np.arange(len(ids)), [types.index(self.qtype(ref[i])) for i in ids]] = 1
        S = np.array([[self.data[m][i][key] for i in ids] for m in models], dtype=float)
        clusters = sorted({ref[i]["cluster"] for i in ids})
        c_idx = np.array([clusters.index(ref[i]["cluster"]) for i in ids])
        return dict(S=S, T=T, types=types, c_idx=c_idx, n_cl=len(clusters))

    def score(self, A, W, tasks=None):
        """Official score of each model (rows of A['S']) under question weights W (resamples x questions),
        over all tasks or only `tasks`: resamples x models, in points."""
        S, T, types = A["S"], A["T"], A["types"]
        den = W @ T                                                  # resamples x types
        num = np.stack([(W * s) @ T for s in S], 1)                  # resamples x models x types
        keep = np.array([tasks is None or ("object_rel_direction" if t in VSI_DIR else t) in tasks for t in types])
        if not self.vsi:
            return 100 * num[..., keep].sum(-1) / den[:, None, keep].sum(-1)
        n_tasks = len({"object_rel_direction" if t in VSI_DIR else t for t, k in zip(types, keep) if k})
        w = np.array([(1 / 3 if t in VSI_DIR else 1.0) / n_tasks if k else 0.0 for t, k in zip(types, keep)])
        return 100 * (num / np.maximum(den[:, None, :], 1e-12)) @ w


def weights(A, B, seed=0, chunk=1000):
    """Bootstrap question weights in chunks: (by scene, by question within type), each chunk x questions."""
    rng = np.random.default_rng(seed)
    n = A["S"].shape[1]
    t_of = A["T"].argmax(1)
    groups = [np.flatnonzero(t_of == k) for k in range(A["T"].shape[1])]
    for b0 in range(0, B, chunk):
        b = min(chunk, B - b0)
        draws = rng.integers(0, A["n_cl"], (b, A["n_cl"]))
        counts = np.stack([np.bincount(d, minlength=A["n_cl"]) for d in draws])
        Ws = counts[:, A["c_idx"]].astype(float)
        Wq = np.zeros((b, n))
        for g in groups:
            pick = g[rng.integers(0, len(g), (b, len(g)))]
            np.add.at(Wq, (np.repeat(np.arange(b), len(g)), pick.ravel()), 1)
        yield Ws, Wq


def contrast(bench, coef, B, tasks=None, subset=None, key="score", second=None):
    """sum_m coef[m] * score(m) (minus the same over `second`'s tasks, for a difference of differences)."""
    models = list(coef)
    if not all(m in bench.data for m in models):
        return None
    ids = bench.ids(models, subset)
    if tasks is not None and second is None:          # resample only the scenes that hold these tasks
        ids = [i for i in ids if bench.task(bench.data[models[0]][i]) in tasks]
    if not ids:
        return None
    A = bench.setup(models, ids, key)
    c = np.array([coef[m] for m in models])

    def stat(W):
        v = bench.score(A, W, tasks) @ c
        return v - bench.score(A, W, second) @ c if second is not None else v

    est = float(stat(np.ones((1, len(ids))))[0])
    s_all, q_all = [], []
    for Ws, Wq in weights(A, B):
        s_all.append(stat(Ws))
        q_all.append(stat(Wq))
    s_all, q_all = np.concatenate(s_all), np.concatenate(q_all)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"diff": est, "ci_scene": ci(s_all), "ci_question": ci(q_all), "n": len(ids)}


def decide(ci, margin):
    lo, hi = ci
    if lo > 0:
        return "helps"
    if hi < 0:
        return "hurts"
    if lo >= -margin and hi <= margin:
        return "no change"
    return "inconclusive"


def mean_minus_base(runs):
    return {**{r: 1 / len(runs) for r in runs}, BASE: -1.0}


def model_summary(bench, m):
    recs = bench.data[m]
    ids = bench.ids([m])
    one = np.ones((1, len(ids)))
    A = bench.setup([m], ids)
    out = {"n": len(ids), "complete": len(ids) == N_FULL[bench.name], "overall": float(bench.score(A, one)[0, 0])}
    tasks = VSI_TASKS if bench.vsi else MC_SETTINGS
    out["per_task"] = {t: float(bench.score(A, one, [t])[0, 0]) for t in tasks
                       if any(bench.task(recs[i]) == t for i in ids)}
    readable = [recs[i]["parsed"] is not None for i in ids]
    out["readable"] = float(np.mean(readable))
    out["readable_per_task"] = {t: float(np.mean([recs[i]["parsed"] is not None for i in ids if bench.task(recs[i]) == t]))
                                for t in out["per_task"]}
    ok = [i for i in ids if recs[i]["parsed"] is not None]
    if ok and len({bench.qtype(recs[i]) for i in ok}) == len({bench.qtype(recs[i]) for i in ids}):
        out["overall_on_readable"] = float(bench.score(bench.setup([m], ok), np.ones((1, len(ok))))[0, 0])
    if bench.vsi:
        out["overall_lenient_parser"] = float(bench.score(bench.setup([m], ids, "score_lenient"), one)[0, 0])
    return out


def analyse(name, B, yes_no):
    bench = Bench(name, load("vsibench" if name.startswith("vsibench") else name))
    models = [m for m in bench.data if not m.endswith("-blind")]
    blind = [m for m in bench.data if m.endswith("-blind")]
    res = {"models": {m: model_summary(bench, m) for m in models},
           "blind": {m[:-6]: model_summary(bench, m) for m in blind}}
    if BASE not in bench.data:
        return res
    # only complete runs enter the contrasts; a run still in progress is listed with its count
    full = lambda m: len(bench.ids([m])) == N_FULL[name]
    res["incomplete"] = [m for m in bench.data if not full(m)]
    bench.data = {m: v for m, v in bench.data.items() if full(m)}
    if BASE not in bench.data:
        return res
    runs_2 = [r for r in RUNS_2EP if r in bench.data]
    runs_4 = [r for r in RUNS_4EP if r in bench.data]
    res["runs_present"] = {"2-epoch": runs_2, "4-epoch": runs_4}

    def add(store, label, coef, decision=False, **kw):
        c = contrast(bench, coef, B, **kw)
        if c is None:
            return
        c["models"] = coef
        if decision:
            c["label"] = decide(c["ci_scene"], MARGIN[name])
        store[label] = c

    prim = res["primary"] = {}
    add(prim, "C1 run0 - base", {"run0": 1.0, BASE: -1.0}, decision=True)
    if len(runs_2) == len(RUNS_2EP):
        add(prim, "C2 2-epoch mean - base", mean_minus_base(RUNS_2EP), decision=True)
    if len(runs_4) == len(RUNS_4EP):
        add(prim, "C3 4-epoch mean - base", mean_minus_base(RUNS_4EP), decision=True)
    labels = [prim[k]["label"] for k in sorted(prim)]
    if len(labels) == 3:
        if labels[1:].count("helps") and "hurts" not in labels:
            verdict = "transfers"
        elif "hurts" in labels and "helps" not in labels:
            verdict = "hurts"
        elif labels.count("no change") == 3:
            verdict = "no change"
        else:
            verdict = "mixed"
        res["verdict"] = verdict
    for k in prim:
        prim[k]["range_over_runs"] = None
    each = res["each_run_minus_base"] = {}
    for r in runs_2 + runs_4:
        add(each, r, {r: 1.0, BASE: -1.0}, decision=True)
    for k, runs in [("C2 2-epoch mean - base", RUNS_2EP), ("C3 4-epoch mean - base", RUNS_4EP)]:
        if k in prim:
            d = [each[r]["diff"] for r in runs]
            prim[k]["range_over_runs"] = [min(d), max(d)]

    sec = res["secondary"] = {}
    groups = [("run0", {"run0": 1.0, BASE: -1.0})]
    if "C2 2-epoch mean - base" in prim:
        groups.append(("2-epoch mean", mean_minus_base(RUNS_2EP)))
    if "C3 4-epoch mean - base" in prim:
        groups.append(("4-epoch mean", mean_minus_base(RUNS_4EP)))
    # 1. per-task contrasts
    tasks = VSI_TASKS if bench.vsi else MC_SETTINGS
    sec["per_task"] = {}
    for t in tasks:
        sec["per_task"][t] = {}
        for g, coef in groups:
            add(sec["per_task"][t], g, coef, tasks=[t])
    # 2. view-relating tasks against metric estimates (VSI-Bench)
    if bench.vsi:
        sec["view_vs_metric"] = {}
        for g, coef in groups:
            d = sec["view_vs_metric"][g] = {}
            add(d, "view-relating tasks", coef, tasks=VSI_VIEW)
            add(d, "metric tasks", coef, tasks=VSI_METRIC)
            add(d, "difference (view - metric)", coef, tasks=VSI_VIEW, second=VSI_METRIC)
    # 3. format: the lenient number parser (VSI-Bench); readable shares are in res["models"]
    if bench.vsi:
        sec["lenient_parser"] = {}
        for g, coef in groups:
            add(sec["lenient_parser"], g, coef, key="score_lenient")
    # 5. without the visual input
    if f"{BASE}-blind" in bench.data and "run0-blind" in bench.data:
        sec["blind"] = {}
        add(sec["blind"], "run0 - base, no visual input", {"run0-blind": 1.0, f"{BASE}-blind": -1.0})
        add(sec["blind"], "base: with - without visual input", {BASE: 1.0, f"{BASE}-blind": -1.0})
        add(sec["blind"], "run0: with - without visual input", {"run0": 1.0, "run0-blind": -1.0})
        add(sec["blind"], "run0 - base: change in the gain from the visual input",
            {"run0": 1.0, "run0-blind": -1.0, BASE: -1.0, f"{BASE}-blind": 1.0})
    for recipe, runs in [("2-epoch", RUNS_2EP), ("4-epoch", RUNS_4EP)]:
        if all(f"{r}-blind" in bench.data for r in runs) and f"{BASE}-blind" in bench.data:
            sec.setdefault("blind", {})
            add(sec["blind"], f"{recipe} mean - base, no visual input",
                {**{f"{r}-blind": 1 / len(runs) for r in runs}, f"{BASE}-blind": -1.0})
            add(sec["blind"], f"{recipe} mean - base: change in the gain from the visual input",
                {**{r: 1 / len(runs) for r in runs}, **{f"{r}-blind": -1 / len(runs) for r in runs},
                 BASE: -1.0, f"{BASE}-blind": 1.0})
            if not bench.vsi:
                for t in MC_SETTINGS:
                    add(sec["blind"], f"{recipe} mean - base, no visual input, {t}",
                        {**{f"{r}-blind": 1 / len(runs) for r in runs}, f"{BASE}-blind": -1.0}, tasks=[t])
    # 6. MindCube without its yes/no questions
    if not bench.vsi:
        sec["without_yes_no"] = {"n_yes_no": len(yes_no)}
        other = set(bench.data[BASE]) - yes_no
        for m in [m for m in models if m in bench.data]:
            ids = bench.ids([m], other)
            sec["without_yes_no"].setdefault("scores", {})[m] = float(
                bench.score(bench.setup([m], ids), np.ones((1, len(ids))))[0, 0])
        for g, coef in groups:
            add(sec["without_yes_no"], g, coef, subset=other)
    # 7. the four-epoch mean minus the two-epoch mean
    if "C2 2-epoch mean - base" in prim and "C3 4-epoch mean - base" in prim:
        add(sec, "4-epoch mean - 2-epoch mean", {**{r: 1 / 5 for r in RUNS_4EP}, **{r: -1 / 3 for r in RUNS_2EP}})
    # 8. exploratory: two-choice accuracy (room A novel) against the benchmark change, across runs
    tc = two_choice_room_a()
    pts = [(tc[r], each[r]["diff"]) for r in runs_2 + runs_4 if r in tc and r in each]
    if len(pts) >= 4:
        x, y = np.array(pts).T
        rank = lambda a: np.argsort(np.argsort(a)).astype(float)
        sec["two_choice_vs_change"] = {"runs": [r for r in runs_2 + runs_4 if r in tc and r in each],
                                       "room_a_novel": x.tolist(), "change": y.tolist(),
                                       "spearman": float(np.corrcoef(rank(x), rank(y))[0, 1])}
    return res


def mc_options(question):
    return {k: v.strip().rstrip(".").strip().lower()
            for k, v in re.findall(r"\b([A-E])\.\s*(.*?)(?=\s+[A-E]\.\s|$)", question)}


def exploratory():
    """Checks chosen after seeing the results (labeled exploratory in docs/experiments.md): MindCube chance
    levels and accuracy by setting without the yes/no questions, how often each model answers "Yes" on the
    yes/no "around" questions, and VSI-Bench's counting and distance answers."""
    out = {"note": "chosen after seeing the results; not registered"}
    Q = {q["id"]: q for q in map(json.loads, open("data/bench/mindcube/data/raw/MindCube_tinybench.jsonl"))}
    opts = {i: mc_options(q["question"]) for i, q in Q.items()}
    yn = {i for i, o in opts.items() if sorted(o.values()) == ["no", "yes"]}
    setting = lambda i: i.split("_")[0]
    mc = out["mindcube"] = {"chance": {}, "accuracy_without_yes_no": {}, "around_yes_no_says_yes": {}}
    for s in MC_SETTINGS + ["all"]:
        ids = [i for i in Q if (s == "all" or setting(i) == s) and i not in yn]
        mc["chance"][f"{s} without yes/no"] = 100 * float(np.mean([1 / len(opts[i]) for i in ids]))
    mc["chance"]["all"] = 100 * float(np.mean([1 / len(o) for o in opts.values()]))
    for m in [BASE] + RUNS_2EP + RUNS_4EP:
        p = f"{OUT}/mindcube/{m}.jsonl"
        if not os.path.exists(p):
            continue
        R = {r["id"]: r for r in map(json.loads, open(p))}
        mc["accuracy_without_yes_no"][m] = {
            s: 100 * float(np.mean([R[i]["score"] for i in R if setting(i) == s and i not in yn])) for s in MC_SETTINGS}
        around_yn = [i for i in yn if setting(i) == "around"]
        mc["around_yes_no_says_yes"][m] = sum(opts[i].get(R[i]["parsed"] or "", "") == "yes" for i in around_yn)
    mc["around_yes_no_n"] = sum(setting(i) == "around" for i in yn)
    vsi = out["vsibench"] = {}
    for m in [BASE, "run0", "e4s6", f"{BASE}-blind", "run0-blind"]:
        p = f"{OUT}/vsibench/{m}.jsonl"
        if not os.path.exists(p):
            continue
        R = list(map(json.loads, open(p)))
        cnt = [r["parsed"] for r in R if r["question_type"] == "object_counting" and r["parsed"] is not None]
        dist = [r["parsed"] for r in R if r["question_type"] == "object_abs_distance" and r["parsed"] is not None]
        vsi[m] = {"counting_answers_2": int(sum(x == 2 for x in cnt)), "counting_n": len(cnt),
                  "abs_distance_median_answer": float(np.median(dist))}
    vsi["abs_distance_median_truth"] = float(np.median(
        [float(r["ground_truth"]) for r in map(json.loads, open(f"{OUT}/vsibench/{BASE}.jsonl"))
         if r["question_type"] == "object_abs_distance"]))
    return out


def two_choice_room_a():
    """Room A novel-trial accuracy with the 64-frame walk, per run (results/two_choice, CUDA files)."""
    out = {}
    for r, a in ADAPTER.items():
        p = f"results/two_choice/Qwen3-VL-8B-Instruct+{a}-dissimilar-256px-max64.jsonl"
        if os.path.exists(p):
            R = [json.loads(l) for l in open(p)]
            nov = [x["correct"] for x in R if not x["familiar"]]
            if nov:
                out[r] = 100 * float(np.mean(nov))
    return out


def gate():
    bench = Bench("vsibench", load("vsibench"))
    if BASE not in bench.data:
        sys.exit("gate: no base run")
    s = model_summary(bench, BASE)
    print(f"gate: base VSI-Bench n={s['n']} overall {s['overall']:.1f} readable {100 * s['readable']:.1f}%")
    ok = s["n"] == N_FULL["vsibench"] and s["readable"] >= 0.95 and s["overall"] >= 57.9 - 8
    print("gate:", "passed" if ok else "FAILED (check the pipeline before the LoRA runs are scored)")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    if args.gate:
        gate()
    yes_no = mindcube_yes_no() if os.path.exists("data/bench/mindcube/data/raw/MindCube_tinybench.jsonl") else set()
    summary = {"note": "Experiment 17 (docs/experiments.md). Points (0-100). Contrasts are linear combinations of "
                       "official scores on the questions all their models answered; ci_scene resamples scenes "
                       "(the decision rule), ci_question resamples questions within question types; runs held "
                       "fixed. Labels: helps / hurts (CI excludes 0), no change (CI within the margin), "
                       "inconclusive.",
               "margins": MARGIN}
    for name in ["vsibench", "vsibench_debiased", "mindcube"]:
        summary[name] = analyse(name, args.boot, yes_no)
    if yes_no and os.path.exists(f"{OUT}/vsibench/{BASE}.jsonl"):
        summary["exploratory"] = exploratory()
    with open(SUMMARY, "w") as fh:
        json.dump(summary, fh, indent=1)
    for name in ["vsibench", "vsibench_debiased", "mindcube"]:
        s = summary[name]
        print(f"== {name}  verdict: {s.get('verdict', 'pending')}")
        for m, v in s["models"].items():
            extra = f" lenient {v['overall_lenient_parser']:.1f}" if "overall_lenient_parser" in v else ""
            print(f"  {m:6s} n={v['n']:5d} overall {v['overall']:5.1f}  readable {100 * v['readable']:5.1f}%{extra}")
        for m, v in s["blind"].items():
            print(f"  {m:6s} no visual input: n={v['n']:5d} overall {v['overall']:5.1f}")
        for k, c in {**s.get("primary", {}), **s.get("each_run_minus_base", {})}.items():
            print(f"  {k:24s} {c['diff']:+6.2f} scene CI [{c['ci_scene'][0]:+.2f}, {c['ci_scene'][1]:+.2f}] "
                  f"question CI [{c['ci_question'][0]:+.2f}, {c['ci_question'][1]:+.2f}]  {c.get('label', '')}")


if __name__ == "__main__":
    main()
