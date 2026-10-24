"""Experiment 14c's pre-registered analysis (docs/experiments.md): the landmark lure, edited. On room A's
lure trials whose shared landmarks are all posters (spatialcells/lure_edit.py; results/lure_edit_manifest.json),
accuracy on the original, edited and control renders for each model and condition, and paired two-sided
sign tests on the same trials: edited minus original and edited minus control (the registered tests), plus
control minus original (the control edit by itself). An unparsed answer counts as wrong. Repeated on the
qualifying trials whose shared posters are pictures only (no door S1 or window E1).

Every run is checked against the trial manifests (known ids, the same options, no duplicates); a run that
has not answered every trial is reported with complete = false and the counts it has. Opus rows appear once
any of their edited or control runs exists (low effort: original = the roomAlure run; high effort, as in
figure 2b: original = the existing results/claude/claude-opus-5-5[-nowalk].jsonl). For the base model,
device_check compares the MPS original answers with the CUDA runs of two_choice.py on the original trials.

    .venv-mac/bin/python scripts/lure_edit_analysis.py     # prints the tables, writes results/lure_edit_summary.json
"""
import json
import math
import os

MANIFEST = "results/lure_edit_manifest.json"
OUT = "results/lure_edit_summary.json"
TC, CL = "results/two_choice/", "results/claude/"
RUNS = {   # name: (original run, edited/control run with the room inserted)
    "Qwen3-VL-8B, walk": (TC + "Qwen3-VL-8B-Instruct-roomAlure-256px-mps.jsonl", TC + "Qwen3-VL-8B-Instruct-{room}-256px-mps.jsonl"),
    "Qwen3-VL-8B, no walk": (TC + "Qwen3-VL-8B-Instruct-roomAlure-256px-nowalk-mps.jsonl",
                             TC + "Qwen3-VL-8B-Instruct-{room}-256px-nowalk-mps.jsonl"),
    "Opus 5.5 low, walk": (CL + "claude-opus-5-5-roomAlure-low.jsonl", CL + "claude-opus-5-5-{room}-low.jsonl"),
    "Opus 5.5 low, no walk": (CL + "claude-opus-5-5-roomAlure-nowalk-low.jsonl", CL + "claude-opus-5-5-{room}-nowalk-low.jsonl"),
    "Opus 5.5 high, walk": (CL + "claude-opus-5-5.jsonl", CL + "claude-opus-5-5-{room}.jsonl"),
    "Opus 5.5 high, no walk": (CL + "claude-opus-5-5-nowalk.jsonl", CL + "claude-opus-5-5-{room}-nowalk.jsonl"),
}
CUDA = {"Qwen3-VL-8B, walk": TC + "Qwen3-VL-8B-Instruct-original-256px.jsonl",
        "Qwen3-VL-8B, no walk": TC + "Qwen3-VL-8B-Instruct-original-256px-nowalk.jsonl"}
OPTS = ("walk", "near", "far", "a", "b")


def sign_test(up, dn):
    n = up + dn
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(max(up, dn), n + 1)) / 2 ** n) if n else 1.0


def load(path, trials):
    """id -> record for the ids in trials, after checking options and duplicates; None if no file."""
    if not os.path.exists(path):
        return None
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows = [r for r in rows if r["id"] in trials]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), f"{path}: duplicate trials"
    for r in rows:
        assert all(r[k] == trials[r["id"]][k] for k in OPTS), f"{path}: {r['id']} options differ from the manifest"
    return {r["id"]: r for r in rows}


def acc(d):
    k, n = sum(d.values()), len(d)
    return {"k": k, "n": n, "accuracy": k / n if n else None}


def paired(x, y):
    """x minus y on the trials both answered."""
    ids = sorted(set(x) & set(y))
    up = sum(x[i] and not y[i] for i in ids)
    dn = sum(y[i] and not x[i] for i in ids)
    return {"n": len(ids), "both_right": sum(x[i] and y[i] for i in ids), "only_first_right": up, "only_second_right": dn,
            "difference": (up - dn) / len(ids) if ids else None, "p_two_sided": sign_test(up, dn)}


def tables(orig, edit, ctrl, ids):
    o, e, c = ({i: v for i, v in d.items() if i in ids} for d in (orig, edit, ctrl))
    return {"accuracy": {"original": acc(o), "edited": acc(e), "control": acc(c)},
            "edited_minus_original": paired(e, o), "edited_minus_control": paired(e, c),
            "control_minus_original": paired(c, o)}


def main():
    m = json.load(open(MANIFEST))
    plan = m["trials"]
    orig_trials = {t["id"]: t for t in json.load(open("results/export/dissimilar-roomAlure/trials.json"))}
    room_trials = {r: {t["id"]: t for t in json.load(open(f"results/export/dissimilar-{r}/trials.json"))} for r in m["rooms"]}
    pictures = {i for i, p in plan.items() if not p["door_or_window"]}
    expected = {"original": len(plan), "edited": len(plan), "control": sum(p["control"] is not None for p in plan.values())}
    out = {"design": {"lure_trials": m["lure_trials"], "qualifying": len(plan), "picture_posters_only": len(pictures),
                      "excluded_pillar": len(m["excluded_pillar"]), "without_control": m["no_control"],
                      "poster_order": m["poster_order"], "rooms": len(m["rooms"])}, "runs": {}}
    for name, (opath, pattern) in RUNS.items():
        rooms = {r: load(pattern.format(room=r), room_trials[r]) for r in m["rooms"]}
        if name.startswith("Opus") and not any(rooms.values()):
            continue
        orig = load(opath, orig_trials) or {}
        edit, ctrl, parsed = {}, {}, []
        for r, recs in rooms.items():
            for i, rec in (recs or {}).items():
                (edit if room_trials[r][i]["role"] == "edited" else ctrl)[i] = bool(rec["correct"])
                parsed.append(rec["answer"] > 0)
        parsed += [rec["answer"] > 0 for rec in orig.values()]
        orig = {i: bool(rec["correct"]) for i, rec in orig.items()}
        got = {"original": len(orig), "edited": len(edit), "control": len(ctrl)}
        res = {"complete": got == expected, "answered": got, "expected": expected,
               "parsed": sum(parsed) / len(parsed) if parsed else None,
               "missing_runs": sorted([opath] * (not os.path.exists(opath)) + [pattern.format(room=r) for r, v in rooms.items() if v is None]),
               **tables(orig, edit, ctrl, set(plan)),
               "picture_posters_only": tables(orig, edit, ctrl, pictures),
               "per_trial": {i: [orig.get(i), edit.get(i), ctrl.get(i)] for i in sorted(plan)}}
        if name in CUDA and os.path.exists(CUDA[name]):
            cuda = load(CUDA[name], orig_trials)
            mps = load(opath, orig_trials) or {}
            both = sorted(set(cuda) & set(mps))
            res["device_check"] = {"cuda_run": CUDA[name], "trials": len(both),
                                   "same_answer": sum(cuda[i]["answer"] == mps[i]["answer"] for i in both),
                                   "cuda_accuracy": acc({i: bool(r["correct"]) for i, r in cuda.items()})}
        out["runs"][name] = res

    print(f"{m['lure_trials']} lure trials; {len(plan)} qualify ({len(pictures)} picture posters only); "
          f"{len(m['no_control'])} without a control poster")
    f = lambda a: f"{a['accuracy']:.0%} ({a['k']}/{a['n']})" if a["n"] else "-"
    g = lambda p: f"{p['difference']:+.0%} ({p['only_first_right']}/{p['only_second_right']}, n {p['n']}, p {p['p_two_sided']:.2g})" if p["n"] else "-"
    print(f"\n{'run':38s} {'original':>12s} {'edited':>12s} {'control':>12s}   {'edited - original':32s} {'edited - control':32s} control - original")
    for name, res in out["runs"].items():
        for sub, t in (("", res), ("  pictures only", res["picture_posters_only"])):
            a = t["accuracy"]
            print(f"{name + sub:38s} {f(a['original']):>12s} {f(a['edited']):>12s} {f(a['control']):>12s}   "
                  f"{g(t['edited_minus_original']):32s} {g(t['edited_minus_control']):32s} {g(t['control_minus_original'])}"
                  + ("" if sub or res["complete"] else f"  INCOMPLETE {res['answered']}"))
        if "device_check" in res:
            d = res["device_check"]
            print(f"{'':38s} MPS vs CUDA original answers: {d['same_answer']}/{d['trials']} the same; CUDA accuracy {f(d['cuda_accuracy'])}")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
