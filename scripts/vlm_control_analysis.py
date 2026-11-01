"""Experiment 29 amendment 1's registered analysis (docs/experiments.md): is the harm of moved frames a penalty for a
broken walk?

Inputs: results/vlm_intervention/specs_control.json (the trials), stimulus_control.json (the agents), experiment 29's
clean and there-away answers and the three new conditions per run. For each run and agent, on the same trials: accuracy
per room (rooms with at least 20 trials), means over rooms with 95% bootstrap CIs (10,000 resamples, seed 0), paired
differences, room and trial sign tests. With A(c) the mean accuracy in condition c:
  G = A(clean) - A(there-away)        the harm to be explained
  W = A(clean) - A(block-there)       the same break, away from both options
  F = A(clean) - A(far-far)           the same break, at the far option
  P = A(far-far) - A(far-close)       the pull: same break, far set moved to the new view
Not a generic penalty: W <= G/4 and W's upper CI end < G/2. Generic penalty: W >= G/2. Otherwise partial. Directional:
P >= 0.05 with its CI above 0. "place confirmed": not generic and directional; "broken walk": generic and not directional;
otherwise "mixed". The verdict is the one runs 5 and 7 share, else "mixed". Fractions throughout (0.05 = 5 points).

    .venv-claude/bin/python scripts/vlm_control_analysis.py [--fetch]   # -> results/vlm_control.json
    python scripts/vlm_control_analysis.py --check-stimulus             # exit 1 unless the stimuli pass (job gate)
"""
import argparse
import json
import math
import os
import subprocess

import numpy as np

OUT_JSON = "results/vlm_control.json"
DIR = "results/vlm_intervention"
NAME = "Qwen3-VL-8B-Instruct+"
RUNS = {"run5": "qwen8b-proc-e4-s5", "run7": "qwen8b-proc-e4-s7"}
NEW = ["block-there", "far-far", "far-close"]
CONDS = ["clean", "there-away"] + NEW
MIN_ROOM_TRIALS = 20


def sign_p(up, dn):
    n = up + dn
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(max(up, dn), n + 1)) / 2 ** n) if n else 1.0


def ci(b):
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def stats(acc, keys, rng, nboot=10_000):
    rooms = sorted({k.split("/")[0] for k in keys})
    by = {r: [k for k in keys if k.startswith(r + "/")] for r in rooms}
    rooms = [r for r in rooms if len(by[r]) >= MIN_ROOM_TRIALS]
    M = {c: np.array([np.mean([acc[c][k] for k in by[r]]) for r in rooms]) for c in CONDS}
    idx = rng.integers(0, len(rooms), (nboot, len(rooms)))
    B = {c: M[c][idx].mean(1) for c in CONDS}
    tk = [k for r in rooms for k in by[r]]

    def diff(a, b):
        d = M[a] - M[b]
        up = sum(acc[a][k] and not acc[b][k] for k in tk)
        dn = sum(acc[b][k] and not acc[a][k] for k in tk)
        return {"mean": float(d.mean()), "ci95": ci(B[a] - B[b]), "rooms_up_down": [int((d > 0).sum()), int((d < 0).sum())],
                "room_sign_p": sign_p(int((d > 0).sum()), int((d < 0).sum())), "trials_up_down": [int(up), int(dn)],
                "trial_sign_p": sign_p(int(up), int(dn))}
    C = {"G": diff("clean", "there-away"), "W": diff("clean", "block-there"), "F": diff("clean", "far-far"),
         "P": diff("far-far", "far-close")}
    G, W, P = C["G"]["mean"], C["W"], C["P"]
    generic = "not generic" if W["mean"] <= G / 4 and W["ci95"][1] < G / 2 else "generic" if W["mean"] >= G / 2 else "partial"
    directional = P["mean"] >= 0.05 and P["ci95"][0] > 0
    verdict = ("place confirmed" if generic == "not generic" and directional else
               "broken walk" if generic == "generic" and not directional else "mixed")
    return {"rooms": rooms, "n_trials": len(tk), "contrasts": C, "generic": generic, "directional": bool(directional),
            "verdict": verdict, "accuracy": {c: {"mean": float(M[c].mean()), "ci95": ci(B[c]),
                                                 "per_room": dict(zip(rooms, M[c].round(4).tolist()))} for c in CONDS}}


def load_run(adapter, keys, conds):
    out = {}
    for c in conds:
        rows = {f"{x['room']}/{x['id']}": x for x in map(json.loads, open(f"{DIR}/{NAME}{adapter}-{c}.jsonl"))}
        missing = set(keys) - set(rows)
        assert not missing, (c, len(missing))
        out[c] = {k: bool(rows[k]["correct"]) for k in keys}
    return out


def agents(keys, stim):
    """The place and view agents (near chosen or not, per condition) and the synthetic broken-walk agent."""
    res = {}
    for name in ("place", "view"):
        acc = {c: {k: stim[k]["agents"][name][c] for k in keys} for c in ["clean"] + NEW}
        acc["there-away"] = acc["clean"]            # replaced by check_stimulus with experiment 29's answers
        res[name] = acc
    return res


def check_stimulus(keys, rng):
    stim = json.load(open(f"{DIR}/stimulus_control.json"))["trials"]
    s29 = json.load(open(f"{DIR}/stimulus_check.json"))["trials"]
    A = agents(keys, stim)
    for name in A:
        A[name]["there-away"] = {k: s29[k]["agents"]["there-away"][name] for k in keys}
    place, view = stats(A["place"], keys, rng), stats(A["view"], keys, rng)
    base = load_run(RUNS["run5"], keys, ["clean", "there-away"])
    Gs = stats({**base, **{c: base["clean"] for c in NEW}}, keys, rng)["contrasts"]["G"]["mean"]
    broken = {"clean": base["clean"]}
    for c in ["there-away"] + NEW:
        hit = dict(zip(keys, rng.random(len(keys)) < Gs))
        broken[c] = {k: base["clean"][k] and not hit[k] for k in keys}
    walk = stats(broken, keys, rng)
    ok = place["verdict"] == "place confirmed" and view["contrasts"]["P"]["mean"] < 0.05 and walk["verdict"] == "broken walk"
    out = {"place_agent": place["verdict"], "place_P": place["contrasts"]["P"]["mean"], "view_agent_P": view["contrasts"]["P"]["mean"],
           "view_agent": view, "broken_walk_agent": walk["verdict"], "broken_walk_P": walk["contrasts"]["P"]["mean"],
           "broken_walk_W": walk["contrasts"]["W"]["mean"], "passed": bool(ok)}
    print({k: v for k, v in out.items() if k != "view_agent"})
    return out


def manipulation(keys):
    stim = json.load(open(f"{DIR}/stimulus_control.json"))["trials"]
    build = json.load(open(f"{DIR}/specs_control.json"))["build"]
    T = [stim[k] for k in keys]
    return {"build": build, **{c: {"dist_to_view_mean": float(np.mean([np.mean(t[f"dist_{c}"]) for t in T])),
                                   "cos_new_view_mean": float(np.mean([np.mean(t[f"cos_{c}"]) for t in T])),
                                   "cos_new_view_best": float(np.mean([max(t[f"cos_{c}"]) for t in T])),
                                   "cos_far_option_mean": float(np.mean([np.mean(t[f"optcos_{c}"]) for t in T]))} for c in NEW}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="copy the control's files from vast-3 first")
    ap.add_argument("--check-stimulus", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        cmd = ('source scripts/vast1/common.sh && rsync -a -e "$MI_RSYNC_SHELL" --include="*e4-s[57]-block-there.jsonl" '
               '--include="*e4-s[57]-far-*.jsonl" --include="specs_control.json" --include="stimulus_control.json" '
               f'--exclude="*" "$MI_CONTAINER:$MI_REMOTE_ROOT/{DIR}/" {DIR}/')
        subprocess.run(["bash", "-c", cmd], check=True, env={**os.environ, "MI_HOST": "vast3"})
    keys = sorted(json.load(open(f"{DIR}/specs_control.json"))["trials"])
    rng = np.random.default_rng(0)
    stim = check_stimulus(keys, rng)
    if args.check_stimulus:
        if not stim["passed"]:
            raise SystemExit("STIMULUS CHECK FAILED: no model job may run; amend the design")
        print("STIMULUS CHECK PASSED")
        return
    runs = {r: stats(load_run(a, keys, CONDS), keys, rng) for r, a in RUNS.items()}
    vs = {v["verdict"] for v in runs.values()}
    res = {"summary": {"n_trials": len(keys), "verdict": vs.pop() if len(vs) == 1 else "mixed", "stimulus": stim,
                       "manipulation": manipulation(keys),
                       **{r: {"verdict": v["verdict"], "generic": v["generic"], "directional": v["directional"],
                              **{c: v["accuracy"][c]["mean"] for c in CONDS},
                              **{n: {"mean": v["contrasts"][n]["mean"], "ci95": v["contrasts"][n]["ci95"]} for n in "GWFP"}}
                          for r, v in runs.items()}}, "runs": runs}
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    print(json.dumps({k: v for k, v in res["summary"].items() if k not in ("stimulus", "manipulation")}, indent=1))
    print(f"-> {OUT_JSON}")


if __name__ == "__main__":
    main()
