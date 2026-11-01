"""Experiment 30's descriptive item: experiment 28's five chosen matchers on experiment 30's trial lists, no threshold.

    .venv-mac/bin/python scripts/trajectory_shift_matcher.py        # -> results/trajectory_shift_matcher.json

Features are experiment 30's DINOv2 CLS tokens (data/e30/<scene>/dino.npz); z-scoring, the model and the scoring are
learned_matcher.py's, imported unchanged. A first step rebuilds P5000's novel trials from experiment 20's cache by this
script's own route and must give the scores stored in data/learned_matcher/test_scores.json.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import learned_matcher as lm  # noqa: E402
import trajectory_shift_analysis as ta  # noqa: E402
from spatialcells import trajectory_shift as ts  # noqa: E402

OUT = "results/trajectory_shift_matcher.json"


def inputs(trials, F, P, prow, order):
    """Trials as the LoRA's 64-frame prompt shows them: padded arrays, as learned_matcher.scene_inputs."""
    items = []
    for t in trials:
        S = np.array(ts.shown_order(t["a"], t["b"]))
        Zf, q = lm.zscore(F[order[t["walk"]][S - 1]], P[prow[t["probe"]]], "cls")
        items.append((Zf, S, int(np.flatnonzero(S == t["a"])[0]), int(np.flatnonzero(S == t["b"])[0]), q,
                      0 if t["near"] == t["a"] else 1))
    T, N, D = len(items), max(len(it[1]) for it in items), items[0][0].shape[1]
    W, pos, keep = np.zeros((T, N, D), np.float32), np.zeros((T, N), np.float32), np.zeros((T, N), bool)
    for i, (Zf, S, *_r) in enumerate(items):
        W[i, :len(S)], pos[i, :len(S)], keep[i, :len(S)] = Zf, S, True
    return {"W": W, "pos": pos, "keep": keep, "ia": np.array([it[2] for it in items]), "ib": np.array([it[3] for it in items]),
            "Q": np.stack([it[4] for it in items]), "y": np.array([it[5] for it in items])}


def check(models):
    """P5000's novel trials through inputs() against the stored scores."""
    from spatialcells.two_choice import walk_order
    z = np.load(f"{lm.CACHE}/P5000.npz")
    meta = dict(np.load(f"{lm.data_dir('P5000')}/meta.npz"))
    trials = [t for t in lm.trials_of("P5000") if not t["familiar"]]
    x = inputs(trials, z["frame_cls"], z["probe_cls"], {int(p): i for i, p in enumerate(z["probe_idx"])}, walk_order(meta))
    rec = json.load(open(f"{lm.OUTDIR}/test_scores.json"))["scenes"]["P5000"]
    row = {i: j for j, i in enumerate(rec["ids"])}
    res = {}
    for v, mods in models.items():
        stored = np.array(rec[v])[[row[t["id"]] for t in trials]]
        res[v] = float((np.stack([lm.score_scene(m, x, x["pos"]) for m in mods], 1) == stored).mean())
    return {"trials": len(trials), "identical_share": res}


def main():
    import torch
    torch.set_num_threads(4)
    models = {v: [lm.load_model(p) for p in lm.chosen_paths("cls", v)] for v in lm.VARIANTS}
    chk = check(models)
    print("check:", chk, flush=True)
    assert all(v == 1 for v in chk["identical_share"].values()), chk
    man = ta.combined_manifest()
    answers = ta.load_answers(man)
    done = json.load(open(ta.OUT))
    sc_data = {}
    per = {}                                    # (cond, set, type) -> method -> per-room accuracy
    for cond in ts.COND_ORDER:
        for room in ts.ROOMS:
            sc = ts.scene(room, cond)
            sd = sc_data[sc] = ta.scene_data(sc, need_dino=True)
            for name, trials in ta.load_trials(sc).items():
                x = inputs(trials, sd["z"]["frame_cls"], sd["z"]["probe_cls"], sd["prow"], sd["order"])
                scores = {v: np.stack([lm.score_scene(m, x, x["pos"]) for m in mods]).mean(0) for v, mods in models.items()}
                lora = np.mean([[answers[k][f"{sc}-{name}-{t['id']}"] for t in trials] for k in ta.RUNS], 0)
                for typ in ([None] if name == "standard" else ["conflict", "agreement"]):
                    sel = np.array([typ is None or t["type"] == typ for t in trials])
                    d = per.setdefault((cond, name, typ), {"lora": {}, **{v: {} for v in models}})
                    d["lora"][room] = float(lora[sel].mean())
                    for v in models:
                        d[v][room] = float(scores[v][sel].mean())
            print(sc, "done", flush=True)
    out = {"settings": {"registration": "docs/experiments.md, Experiment 30, Visual difficulty; descriptive, no threshold",
                        "checkpoints": {v: lm.chosen_paths("cls", v) for v in models},
                        "units": "fractions; each model's accuracy is the mean over its five seeds",
                        "ties": "0.5, as experiment 28"},
           "check": chk, "standard": {}, "separation": {}}
    for (cond, name, typ), d in per.items():
        rooms = ts.ROOMS if name == "standard" else done["separation"][cond]["rooms_used"]
        vec = lambda m: np.array([d[m][r] for r in rooms])
        e = {v: {**ta.ci_block(vec(v)), "lora_minus": ta.ci_block(vec("lora") - vec(v))} for v in models}
        e["lora"] = ta.ci_block(vec("lora"))
        if name == "standard":
            out["standard"][cond] = e
        else:
            out["separation"].setdefault(cond, {})[typ] = e
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    for cond, e in out["standard"].items():
        print(cond, {v: f"{e[v]['mean']:.3f} LoRA {e[v]['lora_minus']['mean']:+.3f} {np.round(e[v]['lora_minus']['ci95'], 3).tolist()}" for v in models})
    for cond, s in out["separation"].items():
        for typ, e in s.items():
            print(cond, typ, {v: f"{e[v]['mean']:.3f} LoRA {e[v]['lora_minus']['mean']:+.3f}" for v in models})
    print("->", OUT)


if __name__ == "__main__":
    main()
