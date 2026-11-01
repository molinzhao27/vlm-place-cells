"""Experiment 29's pre-registered analysis (docs/experiments.md): does the fine-tuned model's decision use where the
walk frames around the near option were taken, or what they show?

Inputs: results/vlm_intervention/specs.json (included trials), each run's JSONL per condition
(spatialcells/vlm_intervention.py), experiment 15's near cut for run 0 (results/frame_cut/), and
results/vlm_intervention/stimulus_check.json (DINOv2 cosines, landmark sharing and the two agents).

For each run and each agent: novel-trial accuracy per room in every condition (unparsed answers wrong), the mean over
rooms with a 95% bootstrap CI over rooms (10,000 resamples, seed 0), paired differences with bootstrap CIs, room-level
sign tests and pooled trial-level sign tests. With A(c) the mean accuracy in condition c:
  swap gap  G = A(clean) - A(there-away)        must be >= 5 points (0.05) with its CI above 0, else "inconclusive: no gap"
  place     A(here-away) - A(there-away)        R_place = place / G
  view      A(there-toward) - A(there-away)     R_view = view / G
  conflict  A(here-away) - A(there-toward)
  both      A(here-toward) - A(there-away)      R_both = both / G
  random    A(clean) - A(random-away)           more than G / 2 -> "inconclusive: the swaps disrupt wherever they are"
Verdicts: "place" (R_place >= 0.5 with place's CI above 0, R_view < 0.5, conflict's CI above 0); "view plus walk"
(R_view >= 0.5 with view's CI above 0, R_place < 0.5, conflict's CI below 0); "both" (R_place and R_view >= 0.5, each
CI above 0); otherwise inconclusive, labelled "conjunction" if R_both >= 0.5 with both's CI above 0, else "not
separated". The fine-tuned model's verdict is the one runs 0, 5 and 7 share; "mixed" if they differ. The stimuli pass
their check if the place agent's verdict is "place" and the view agent's is "view plus walk". The same rule is
repeated on the trials where the near and far sets share no frame (sensitivity). Accuracies and differences are
fractions (0.05 = 5 points) everywhere in the output.

    .venv-claude/bin/python scripts/vlm_intervention_analysis.py [--fetch]   # -> results/vlm_intervention.json
    .venv-claude/bin/python scripts/vlm_intervention_analysis.py --test   # synthetic run files from the agents
    python scripts/vlm_intervention_analysis.py --check-stimulus          # exit 1 unless the stimuli pass (job gate)
"""
import argparse
import json
import math
import os
import tempfile

import numpy as np

OUT_JSON = "results/vlm_intervention.json"
DIR = "results/vlm_intervention"
SPECS = f"{DIR}/specs.json"
STIM = f"{DIR}/stimulus_check.json"
NAME = "Qwen3-VL-8B-Instruct+"
RUNS = {"run0": ("qwen8b-proc", "mps"), "run5": ("qwen8b-proc-e4-s5", "cuda"), "run7": ("qwen8b-proc-e4-s7", "cuda")}
EXP15_NEAR = "results/frame_cut/Qwen3-VL-8B-Instruct+qwen8b-proc-near-mps.jsonl"
SWAPS = ["here-toward", "here-away", "there-toward", "there-away", "random-away"]
CONDS = ["clean"] + SWAPS
MIN_ROOM_TRIALS = 20
FIELDS = ("walk", "probe", "near", "far", "a", "b")


def sign_p(up, dn):
    n = up + dn
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(max(up, dn), n + 1)) / 2 ** n) if n else 1.0


def path(run, cond, files):
    adapter, dev = RUNS[run]
    return f"{files}/{NAME}{adapter}-{cond}" + (f"-{dev}" if dev != "cuda" else "") + ".jsonl"


def load_run(run, specs, files):
    """condition -> {trial key: correct}, for files covering every included trial; plus the clean check."""
    inc = {k: s for k, s in specs.items() if s["included"]}
    out, incomplete = {}, []
    for cond in CONDS + ["near-cut"]:
        p = path(run, cond, files)
        if cond == "near-cut" and run == "run0":
            p = EXP15_NEAR                                  # experiment 15's cut of the same set, same device
        if not os.path.exists(p):
            incomplete.append(cond)
            continue
        rows = {f"{x['room']}/{x['id']}": x for x in map(json.loads, open(p)) if x.get("room")}
        for k, x in rows.items():
            if k in inc:
                s = inc[k]
                assert all(x[f] == s[f] for f in ("walk", "probe", "near", "far")), (p, k)
                assert bool(x["correct"]) == (x["answer"] == x["near"]), (p, k)
                if cond in SWAPS:
                    assert x["md5"] == [f[4] for f in s["conditions"][cond]], (p, k)
        if set(inc) - set(rows):
            incomplete.append(cond)
            continue
        out[cond] = {k: bool(rows[k]["correct"]) for k in inc}
        if cond == "clean":
            agree = [rows[k]["answer"] == rows[k].get("exp10_answer") for k in inc]
            out["_clean_agreement"] = float(np.mean(agree))
    return out, incomplete


def stats(acc, keys, rooms_of, rng, nboot=10_000):
    """Room means, bootstrap CIs, contrasts, ratios and the verdict for one run or agent (acc: cond -> key -> bool)."""
    rooms = sorted({rooms_of[k] for k in keys})
    by_room = {r: [k for k in keys if rooms_of[k] == r] for r in rooms}
    rooms = [r for r in rooms if len(by_room[r]) >= MIN_ROOM_TRIALS]
    conds = [c for c in CONDS + ["near-cut"] if c in acc]
    M = {c: np.array([np.mean([acc[c][k] for k in by_room[r]]) for r in rooms]) for c in conds}   # fractions
    idx = rng.integers(0, len(rooms), (nboot, len(rooms)))
    B = {c: M[c][idx].mean(1) for c in conds}
    res = {"rooms": rooms, "n_trials": sum(len(by_room[r]) for r in rooms),
           "accuracy": {c: {"mean": float(M[c].mean()), "ci95": ci(B[c]),
                            "per_room": dict(zip(rooms, M[c].round(4).tolist()))} for c in conds}}

    def diff(a, b):
        if a not in M or b not in M:
            return None
        d = M[a] - M[b]
        tk = [k for r in rooms for k in by_room[r]]
        up = sum(acc[a][k] and not acc[b][k] for k in tk)
        dn = sum(acc[b][k] and not acc[a][k] for k in tk)
        return {"mean": float(d.mean()), "ci95": ci(B[a] - B[b]), "rooms_up_down": [int((d > 0).sum()), int((d < 0).sum())],
                "room_sign_p": sign_p(int((d > 0).sum()), int((d < 0).sum())), "trials_up_down": [int(up), int(dn)],
                "trial_sign_p": sign_p(int(up), int(dn))}
    C = {"swap_gap": diff("clean", "there-away"), "place": diff("here-away", "there-away"),
         "view": diff("there-toward", "there-away"), "conflict": diff("here-away", "there-toward"),
         "both": diff("here-toward", "there-away"), "random_cost": diff("clean", "random-away"),
         "near_cut_cost": diff("clean", "near-cut"), "there_away_minus_near_cut": diff("there-away", "near-cut"),
         "here_toward_minus_clean": diff("here-toward", "clean")}
    res["contrasts"] = {k: v for k, v in C.items() if v is not None}
    if all(c in M for c in CONDS):
        G, Gb = C["swap_gap"]["mean"], B["clean"] - B["there-away"]
        for name, a in (("r_place", "here-away"), ("r_view", "there-toward"), ("r_both", "here-toward")):
            num, numb = M[a].mean() - M["there-away"].mean(), B[a] - B["there-away"]
            with np.errstate(divide="ignore", invalid="ignore"):
                res[name] = {"value": float(num / G) if G else None, "ci95": ci(numb / Gb)}
        res["verdict"] = verdict(C, res)
    return res


def ci(b):
    b = np.asarray(b)[np.isfinite(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else [None, None]


def above(c):
    return c["ci95"][0] > 0


def below(c):
    return c["ci95"][1] < 0


def verdict(C, res):
    G = C["swap_gap"]
    if not (G["mean"] >= 0.05 and above(G)):
        return "inconclusive: no gap"
    if C["random_cost"]["mean"] > G["mean"] / 2:
        return "inconclusive: the swaps disrupt wherever they are"
    rp, rv, rb = res["r_place"]["value"], res["r_view"]["value"], res["r_both"]["value"]
    if rp >= 0.5 and rv >= 0.5 and above(C["place"]) and above(C["view"]):
        return "both"
    if rp >= 0.5 and above(C["place"]) and rv < 0.5 and above(C["conflict"]):
        return "place"
    if rv >= 0.5 and above(C["view"]) and rp < 0.5 and below(C["conflict"]):
        return "view plus walk"
    if rb >= 0.5 and above(C["both"]) and rp < 0.5 and rv < 0.5:
        return "inconclusive: conjunction"
    return "inconclusive: not separated"


def manipulation(stim, keys):
    """Mean DINOv2 cosine with the new view (mean over a trial's 8 frames, and the trial's best frame), with the near
    option's frame, and the share of frames that show a landmark the new view shows, per condition, on the included
    trials."""
    T = [stim["trials"][k] for k in keys if k in stim["trials"]]
    out = {}
    for c in ["original_near", "original_random"] + SWAPS:
        cos = [t[f"cos_{c}"] for t in T]
        o = {"cos_mean": float(np.mean([np.mean(x) for x in cos])), "cos_best": float(np.mean([max(x) for x in cos]))}
        if f"optcos_{c}" in T[0]:
            o["cos_with_near_option_mean"] = float(np.mean([np.mean(t[f"optcos_{c}"]) for t in T]))
        if f"landmark_{c}" in T[0]:
            o["landmark_share"] = float(np.mean([np.mean(t[f"landmark_{c}"]) for t in T]))
        out[c] = o
    return out


def analyse(specs, stim, files, rng):
    inc = sorted(k for k, s in specs.items() if s["included"])
    rooms_of = {k: specs[k]["room"] for k in inc}
    no_ov = [k for k in inc if specs[k]["overlap_near_far"] == 0]
    summary = {"n_included": len(inc), "n_included_no_near_far_overlap": len(no_ov),
               "included_per_room": {r: sum(rooms_of[k] == r for k in inc) for r in sorted(set(rooms_of.values()))}}
    full = {"agents": {}, "runs": {}}
    if stim:
        keys = [k for k in inc if k in stim["trials"]]
        for agent in ("place", "view"):
            acc = {c: {k: bool(stim["trials"][k]["agents"][c][agent]) for k in keys} for c in CONDS}
            full["agents"][agent] = stats(acc, keys, rooms_of, rng)
        summary["manipulation"] = manipulation(stim, keys)
        pa, va = full["agents"]["place"].get("verdict"), full["agents"]["view"].get("verdict")
        summary["stimulus_check"] = {"place_agent_verdict": pa, "view_agent_verdict": va,
                                     "passed": pa == "place" and va == "view plus walk"}
    verdicts = {}
    for run in RUNS:
        acc, incomplete = load_run(run, specs, files)
        agree = acc.pop("_clean_agreement", None)
        r = {"incomplete": incomplete, "clean_agreement_with_exp10": agree}
        if all(c in acc for c in CONDS):
            r.update(stats(acc, inc, rooms_of, rng))
            r["no_near_far_overlap"] = stats(acc, no_ov, rooms_of, rng)
            verdicts[run] = r["verdict"]
        full["runs"][run] = r
        s = {"incomplete": incomplete, "clean_agreement_with_exp10": agree}
        if "verdict" in r:
            s.update({c: r["accuracy"][c]["mean"] for c in r["accuracy"]})
            s.update({k: r["contrasts"][k]["mean"] for k in r["contrasts"]})
            s.update({f"{k}_ci95": r["contrasts"][k]["ci95"] for k in ("swap_gap", "place", "view", "conflict")})
            s.update(r_place=r["r_place"]["value"], r_view=r["r_view"]["value"], r_both=r["r_both"]["value"],
                     verdict=r["verdict"], verdict_no_near_far_overlap=r["no_near_far_overlap"].get("verdict"))
        summary[run] = s
    done = list(verdicts)
    summary["runs_complete"] = done
    summary["status"] = "complete" if len(done) == len(RUNS) else "interim" if done else "no model data"
    vs = set(verdicts.values())
    summary["verdict"] = (verdicts[done[0]] if len(vs) == 1 and len(done) == len(RUNS) else
                          "mixed" if len(vs) > 1 else "pending")
    return summary, full


def synthetic(specs, stim, d):
    """--test: run files made from the agents' answers (run0 = place agent, run5 = view agent, run7 = place agent with
    its here-away answers taken from there-away, which should not come out as "place")."""
    src = {"run0": ("place", {}), "run5": ("view", {}), "run7": ("place", {"here-away": "there-away"})}
    for run, (agent, remap) in src.items():
        for c in CONDS:
            with open(path(run, c, d), "w") as fh:
                for k, s in specs.items():
                    if not s["included"] or k not in stim["trials"]:
                        continue
                    ok = stim["trials"][k]["agents"][remap.get(c, c)][agent]
                    ans = s["near"] if ok else s["far"]
                    rec = {"room": s["room"], "id": s["id"], "walk": s["walk"], "probe": s["probe"], "near": s["near"],
                           "far": s["far"], "answer": ans, "correct": ok, "exp10_answer": ans if c == "clean" else None,
                           "md5": [f[4] for f in s["conditions"][c]] if c in SWAPS else []}
                    fh.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="synthetic run files from the agents (pipeline test)")
    ap.add_argument("--files", default=DIR, help="folder of the run files")
    ap.add_argument("--test-dir", help="where --test writes its synthetic files (default: the system's temp folder)")
    ap.add_argument("--fetch", action="store_true",
                    help="first copy runs 5 and 7's JSONL files (only those) from the vast-3 container")
    ap.add_argument("--check-stimulus", action="store_true",
                    help="only the stimulus check: print it, write nothing, exit 1 unless it passed")
    args = ap.parse_args()
    if args.fetch:
        import subprocess
        subprocess.run(["bash", "-c", 'source scripts/vast1/common.sh && rsync -a -e "$MI_RSYNC_SHELL" '
                        '--include="*e4-s5-*.jsonl" --include="*e4-s7-*.jsonl" --exclude="*" '
                        '"$MI_CONTAINER:$MI_REMOTE_ROOT/results/vlm_intervention/" results/vlm_intervention/'],
                       check=True, env={**os.environ, "MI_HOST": "vast3"})
    specs = json.load(open(SPECS))["trials"]
    stim = json.load(open(STIM)) if os.path.exists(STIM) else None
    rng = np.random.default_rng(0)
    out = OUT_JSON
    if args.test:
        assert stim, f"{STIM} is needed for --test"
        specs = {k: s for k, s in specs.items() if k in stim["trials"]}
        args.files = tempfile.mkdtemp(prefix="vlm_intervention_test_", dir=args.test_dir)
        synthetic(specs, stim, args.files)
        out = os.path.join(args.files, "vlm_intervention_test.json")
        global EXP15_NEAR
        EXP15_NEAR = "/nonexistent"
    if args.check_stimulus:
        assert stim, f"{STIM} is missing"
        missing = [k for k, x in specs.items() if x["included"] and k not in stim["trials"]]
        if missing:
            raise SystemExit(f"STIMULUS CHECK FAILED: {STIM} lacks {len(missing)} included trials, e.g. {missing[:3]}")
        summary, _ = analyse(specs, stim, "/nonexistent", rng)
        print(json.dumps({k: summary[k] for k in ("n_included", "stimulus_check", "manipulation")}, indent=1))
        ok = summary["stimulus_check"]["passed"]
        print("STIMULUS CHECK PASSED" if ok else "STIMULUS CHECK FAILED: the model runs must not start")
        raise SystemExit(0 if ok else 1)
    summary, full = analyse(specs, stim, args.files, rng)
    json.dump({"summary": summary, **full}, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
