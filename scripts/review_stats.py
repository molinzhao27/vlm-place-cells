"""Sensitivity re-analyses of existing data (Experiment 13, "Re-analyses of existing data"; not new tests).

1. Walk-clustered intervals. For each paired contrast the paper reports on room A (102 novel lure-free
   trials, 16 walks) or room B (106 novel, 16 walks): the difference in novel-trial accuracy (a minus b,
   pooled over trials), a 95% interval from a bootstrap over walks (whole walks resampled, 10,000
   resamples, rng seed 0 for each contrast) and an exact whole-walk sign-flip p (the two conditions
   swapped for whole walks, all 2^16 patterns), next to the trial-level two-sided sign test the paper
   uses. Unparsed answers count as wrong. A contrast over several efforts averages each trial's
   correctness over them first, as the paper's pooled sign test does. Contrasts that overlap the
   external review's audit (results/review/preprint-audit-2026-10-23.json, 100,000 resamples) are
   compared with it.
2. First-listed answer rates for every run in results/claude, results/two_choice and results/api: the
   share of parsed answers naming the first-listed option (a), the share of trials whose near option is
   listed first, and the unparsed rate, for novel and familiar trials. Rates above 70% are flagged
   (and below 30%, i.e. second-listed above 70%).

    .venv-mac/bin/python scripts/review_stats.py     # prints the tables, writes results/review_stats.json
"""
import glob
import json
import math
import os
import re

import numpy as np

NBOOT = 10_000
MANIFESTS = {"original": "results/export/claude/trials.json", "dissimilar": "results/export/dissimilar/trials.json"}
ROOM = {"A": "dissimilar", "B": "roomB"}
OP, SO, HA = "claude/claude-opus-5-5-dissimilar", "claude/claude-sonnet-5-5-dissimilar", "claude/claude-haiku-5-5-dissimilar"
FT, BASE = "two_choice/Qwen3-VL-8B-Instruct+qwen8b-proc-dissimilar-256px", "two_choice/Qwen3-VL-8B-Instruct-dissimilar-256px"
FTB, BASEB = "two_choice/Qwen3-VL-8B-Instruct+qwen8b-proc-roomB-256px", "two_choice/Qwen3-VL-8B-Instruct-roomB-256px"
EFF = {"none": "-none", "low": "-low", "medium": "-medium", "high": ""}   # high effort has no suffix
CONTRASTS = [   # (room, name, a runs, b runs, review audit key): a minus b; several runs = per-trial mean over them
    ("A", "Opus low: walk - no walk", [OP + "-low"], [OP + "-nowalk-low"], "Opus low, ordered minus no walk"),
    ("A", "Opus medium: walk - no walk", [OP + "-medium"], [OP + "-nowalk-medium"], "Opus medium, walk gain"),
    ("A", "Opus high: walk - no walk", [OP], [OP + "-nowalk"], "Opus high, walk gain"),
    ("A", "Opus, mean over low/medium/high: walk - no walk", [OP + "-low", OP + "-medium", OP],
     [OP + "-nowalk-low", OP + "-nowalk-medium", OP + "-nowalk"], "Opus, mean over three efforts, walk gain"),
    ("A", "Opus low: ordered - shuffled", [OP + "-low"], [OP + "-shuffled-low"], "Opus low, ordered minus shuffled"),
    ("A", "Opus low: ordered - shuffled relabelled", [OP + "-low"], [OP + "-shuffled-relabel-low"], "Opus low, ordered minus shuffled relabelled"),
    ("A", "Opus low: shuffled - no walk", [OP + "-shuffled-low"], [OP + "-nowalk-low"], None),
    ("A", "Opus low: shuffled relabelled - no walk", [OP + "-shuffled-relabel-low"], [OP + "-nowalk-low"], None),
    ("A", "Opus low: 64-frame walk - 256-frame walk", [OP + "-f64-low"], [OP + "-low"], None),
    ("A", "Opus low: 64-frame walk - no walk", [OP + "-f64-low"], [OP + "-nowalk-low"], None),
    *[("A", f"{m} {e}: walk - no walk", [p + s], [p + "-nowalk" + s], None)
      for m, p in (("Sonnet", SO), ("Haiku", HA)) for e, s in EFF.items()],
    *[("A", f"{m}, mean over none/low/medium/high: walk - no walk", [p + s for s in EFF.values()],
       [p + "-nowalk" + s for s in EFF.values()], None) for m, p in (("Sonnet", SO), ("Haiku", HA))],
    ("A", "LoRA: walk (64) - no walk", [FT + "-max64"], [FT + "-nowalk"], "LoRA, walk gain"),
    ("A", "LoRA: ordered - shuffled (64)", [FT + "-max64"], [FT + "-max64-shuffled"], "LoRA, order effect"),
    ("A", "LoRA: ordered - shuffled relabelled (64)", [FT + "-max64"], [FT + "-max64-shuffled-relabel"], None),
    ("A", "LoRA - base (64-frame walk)", [FT + "-max64"], [BASE + "-max64"], None),
    ("A", "LoRA: 256-frame - 64-frame walk", [FT], [FT + "-max64"], None),
    ("A", "LoRA (64) - Opus low (walk)", [FT + "-max64"], [OP + "-low"], None),
    ("A", "LoRA (64) - Opus high (walk)", [FT + "-max64"], [OP], None),
    ("A", "base Qwen3-VL-8B: walk (64) - no walk", [BASE + "-max64"], [BASE + "-nowalk"], None),
    ("A", "base Qwen3-VL-8B: walk (256) - no walk", [BASE], [BASE + "-nowalk"], None),
    ("A", "Qwen3.8-27B direct: walk - no walk", ["two_choice/Qwen3.8-27B-dissimilar-256px"], ["two_choice/Qwen3.8-27B-dissimilar-256px-nowalk"], None),
    ("A", "Qwen3.8-27B thinking: walk - no walk", ["two_choice/Qwen3.8-27B-dissimilar-256px-think-medium-cached"],
     ["two_choice/Qwen3.8-27B-dissimilar-256px-nowalk-think-medium"], None),
    ("A", "Qwen3.6-35B-A3B direct: walk - no walk", ["two_choice/Qwen3.6-35B-A3B-dissimilar-256px"], ["two_choice/Qwen3.6-35B-A3B-dissimilar-256px-nowalk"], None),
    ("A", "Qwen3.6-35B-A3B thinking: walk - no walk", ["two_choice/Qwen3.6-35B-A3B-dissimilar-256px-think-medium-cached"],
     ["two_choice/Qwen3.6-35B-A3B-dissimilar-256px-nowalk-think-medium"], None),
    ("A", "Gemma-4-31B: walk - no walk", ["two_choice/gemma-4-31B-it-dissimilar-256px"], ["two_choice/gemma-4-31B-it-dissimilar-256px-nowalk"], None),
    ("A", "Kimi K3: walk - no walk", ["api/kimi-k3-dissimilar-low"], ["api/kimi-k3-dissimilar-nowalk-low"], None),
    ("A", "DeepSeek V4.1 Flash: walk - no walk", ["api/deepseek-flash-dissimilar-low"], ["api/deepseek-flash-dissimilar-nowalk-low"], None),
    ("A", "GPT-6 Astra: walk - no walk", ["api/gpt-6-astra-dissimilar-low"], ["api/gpt-6-astra-dissimilar-nowalk-low"], None),
    ("A", "GPT-6 Astra: walk - shuffled", ["api/gpt-6-astra-dissimilar-low"], ["api/gpt-6-astra-dissimilar-shuffled-low"], None),
    ("A", "GPT-6 Astra: shuffled - no walk", ["api/gpt-6-astra-dissimilar-shuffled-low"], ["api/gpt-6-astra-dissimilar-nowalk-low"], None),
    ("B", "Opus low: walk - no walk", ["claude/claude-opus-5-5-roomB-low"], ["claude/claude-opus-5-5-roomB-nowalk-low"], None),
    ("B", "LoRA: walk (64) - no walk", [FTB + "-max64-mps"], [FTB + "-nowalk-mps"], None),
    ("B", "LoRA: ordered - shuffled (64)", [FTB + "-max64-mps"], [FTB + "-max64-shuffled-mps"], None),
    ("B", "LoRA - base (64-frame walk)", [FTB + "-max64-mps"], [BASEB + "-max64-mps"], None),
    ("B", "LoRA (64) - Opus low (walk)", [FTB + "-max64-mps"], ["claude/claude-opus-5-5-roomB-low"], None),
    ("B", "base Qwen3-VL-8B: walk (64) - no walk", [BASEB + "-max64-mps"], [BASEB + "-nowalk-mps"], None),
]


def manifest(trial_set):
    return {t["id"]: t for t in json.load(open(MANIFESTS.get(trial_set) or f"results/export/dissimilar-{trial_set}/trials.json"))}


def load(path, trials):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = {r["id"]: r for r in rows}
    assert len(out) == len(rows), f"{path}: duplicate ids"
    assert all(r["id"] in trials and all(r[k] == trials[r["id"]][k] for k in ("walk", "familiar", "near", "far", "a", "b"))
               for r in rows), f"{path}: rows do not match the trial manifest"
    assert all(bool(r["correct"]) == (r["answer"] == r["near"]) for r in rows), f"{path}: correct != (answer == near)"
    return out


def sign_test(up, dn):
    n = up + dn
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(max(up, dn), n + 1)) / 2 ** n) if n else 1.0


def paired(a_runs, b_runs, trials):
    A = [load(f"results/{p}.jsonl", trials) for p in a_runs]
    B = [load(f"results/{p}.jsonl", trials) for p in b_runs]
    novel = sorted(i for i, t in trials.items() if not t["familiar"])
    ids = [i for i in novel if all(i in r for r in A + B)]
    ca = np.array([np.mean([bool(r[i]["correct"]) for r in A]) for i in ids])
    cb = np.array([np.mean([bool(r[i]["correct"]) for r in B]) for i in ids])
    d = ca - cb
    walks = np.array([trials[i]["walk"] for i in ids])
    u = np.unique(walks)
    sums, counts = np.array([d[walks == w].sum() for w in u]), np.array([(walks == w).sum() for w in u])
    rng = np.random.default_rng(0)
    ix = rng.integers(0, len(u), (NBOOT, len(u)))
    bs = sums[ix].sum(1) / counts[ix].sum(1)                     # whole walks resampled; pooled per-trial estimand
    signs = 2 * ((np.arange(2 ** len(u))[:, None] >> np.arange(len(u))) & 1) - 1   # every whole-walk swap
    null = signs @ sums / len(ids)
    up, dn = int((d > 0).sum()), int((d < 0).sum())
    return {"a": a_runs, "b": b_runs, "n_trials": len(ids), "novel_in_manifest": len(novel), "complete": len(ids) == len(novel),
            "n_walks": len(u), "accuracy_a": float(ca.mean()), "accuracy_b": float(cb.mean()), "difference": float(d.mean()),
            "walk_bootstrap_ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
            "walk_sign_flip_p": float(np.mean(np.abs(null) >= abs(d.mean()) - 1e-12)),
            "walks_positive": int((sums > 0).sum()), "walks_negative": int((sums < 0).sum()),
            "trials_better": up, "trials_worse": dn, "trial_sign_test_p": sign_test(up, dn)}


def trial_set(name):
    m = re.search(r"-(original|dissimilar|room[A-Z][A-Za-z0-9]*)(?=-|$)", name)
    return m.group(1) if m else "original"                       # claude-opus-5-5[-nowalk]: the original trials


def listing(rows):
    out = {}
    for part, fam in (("novel", False), ("familiar", True)):
        rs = [r for r in rows if r["familiar"] == fam]
        parsed = [r for r in rs if r["answer"] in (r["a"], r["b"])]
        out[part] = {"n": len(rs), "unparsed": (len(rs) - len(parsed)) / len(rs),
                     "first_listed": sum(r["answer"] == r["a"] for r in parsed) / len(parsed) if parsed else None,
                     "near_first": sum(r["a"] == r["near"] for r in rs) / len(rs),
                     "accuracy": sum(bool(r["correct"]) for r in rs) / len(rs)} if rs else None
    return out


def main():
    out = {"settings": {"bootstrap_resamples": NBOOT, "bootstrap_seed": 0, "unit": "walk",
                        "estimand": "pooled novel-trial accuracy difference, a minus b; unparsed = wrong",
                        "sign_flip": "exact, all 2^n_walks whole-walk swaps; two-sided",
                        "trial_sign_test": "two-sided exact sign test on trials where a and b differ"},
           "walk_clustered": {"A": {}, "B": {}}}
    trials = {room: manifest(ts) for room, ts in ROOM.items()}
    print("walk-clustered contrasts on novel trials (a - b): difference [95% bootstrap CI over walks]; "
          "whole-walk sign-flip p; trial-level sign test")
    for room, name, a, b, _ in CONTRASTS:
        r = paired(a, b, trials[room])
        out["walk_clustered"][room][name] = r
        lo, hi = r["walk_bootstrap_ci95"]
        print(f"room {room}  {name:52s} {r['difference']:+6.1%} [{lo:+6.1%}, {hi:+6.1%}]  flip p {r['walk_sign_flip_p']:.2g}  "
              f"walks +{r['walks_positive']}/-{r['walks_negative']}  trials {r['trials_better']}/{r['trials_worse']} "
              f"p {r['trial_sign_test_p']:.2g}  (n {r['n_trials']}" + ("" if r["complete"] else f" of {r['novel_in_manifest']}, INCOMPLETE") + ")")

    audit = json.load(open("results/review/preprint-audit-2026-10-23.json"))["room_A_walk_cluster_sensitivity"]
    out["review_audit_comparison"] = {}
    print("\nagainst the review's audit (100,000 resamples there, 10,000 here; intervals agree to Monte Carlo error)")
    for room, name, a, b, key in CONTRASTS:
        if key is None:
            continue
        mine, theirs = out["walk_clustered"][room][name], audit[key]
        exact = (abs(mine["difference"] - theirs["difference"]) < 1e-12 and abs(mine["walk_sign_flip_p"] - theirs["walk_sign_flip_p"]) < 1e-12
                 and mine["n_trials"] == theirs["n_trials"] and (mine["trials_better"], mine["trials_worse"]) == (theirs["better"], theirs["worse"]))
        ci_dev = max(abs(x - y) for x, y in zip(mine["walk_bootstrap_ci95"], theirs["walk_bootstrap_ci95"]))
        out["review_audit_comparison"][name] = {"audit_key": key, "difference_p_counts_identical": exact,
                                                "audit_ci95": theirs["walk_bootstrap_ci95"], "max_ci_endpoint_deviation": ci_dev,
                                                "agree": exact and ci_dev < 0.01}
        print(f"  {name:52s} difference/p/counts {'identical' if exact else 'DIFFER'}; CI endpoints within {ci_dev:.4f}"
              + ("" if exact and ci_dev < 0.01 else "  DISAGREES"))

    out["first_listed"], out["first_listed_flags"], out["near_first_by_trial_set"] = {}, [], {}
    print("\nfirst-listed answers (share of parsed answers naming option a); near-first = share of trials listing the near "
          "option first\n" + f"{'run':72s} {'set':>16s} {'n':>4s}  novel first  fam first  unparsed  near-first nov/fam")
    files = sorted(glob.glob("results/claude/*.jsonl") + glob.glob("results/two_choice/*.jsonl") + glob.glob("results/api/*.jsonl"))
    for path in files:
        stem = path[len("results/"):-len(".jsonl")]
        ts = trial_set(os.path.basename(stem))
        tr = manifest(ts)
        rows = list(load(path, tr).values())
        res = {"trial_set": ts, "trials": len(rows), "manifest_trials": len(tr), **listing(rows)}
        res["unparsed_all"] = sum(r["answer"] not in (r["a"], r["b"]) for r in rows) / len(rows)
        out["first_listed"][stem] = res
        flags = []
        for part in ("novel", "familiar"):
            x = res[part]
            if x and x["first_listed"] is not None and x["n"] >= 20:
                if x["first_listed"] > 0.7:
                    flags.append(f"{part} first-listed {x['first_listed']:.0%}")
                elif x["first_listed"] < 0.3:
                    flags.append(f"{part} second-listed {1 - x['first_listed']:.0%}")
        if flags:
            out["first_listed_flags"].append({"run": stem, "flags": flags})
        f = lambda part, k: f"{res[part][k]:5.0%}" if res[part] and res[part][k] is not None else "    -"
        print(f"{stem:72s} {ts:>16s} {len(rows):4d}  {f('novel', 'first_listed'):>11s}  {f('familiar', 'first_listed'):>9s}  "
              f"{res['unparsed_all']:8.1%}  {f('novel', 'near_first')}/{f('familiar', 'near_first')}"
              + (f"  <- {'; '.join(flags)}" if flags else "") + ("" if len(rows) == len(tr) else f"  ({len(rows)} of {len(tr)} trials)"))
    for ts in sorted({v["trial_set"] for v in out["first_listed"].values()}):
        tr = list(manifest(ts).values())
        out["near_first_by_trial_set"][ts] = {part: float(np.mean([t["a"] == t["near"] for t in tr if t["familiar"] == fam]))
                                              for part, fam in (("novel", False), ("familiar", True)) if any(t["familiar"] == fam for t in tr)}
    print("\nnear option listed first, by trial set (manifest):", json.dumps(out["near_first_by_trial_set"]))
    json.dump(out, open("results/review_stats.json", "w"), indent=1)
    print("-> results/review_stats.json")


if __name__ == "__main__":
    main()
