"""Experiment 31's pre-registered analysis (docs/experiments.md): experiment 13a's walk-order controls, a relabeled
shuffle, and experiment 15's attention cuts on four-epoch runs 5 and 7, with run 0 (the original two-epoch LoRA) as
the reference.

Per run, on the novel trials of P5000-P5009 (1,299): accuracy per room in each condition (unparsed answers wrong),
means over rooms with 95% bootstrap CIs over rooms (10,000 resamples, seed 0; ratios as ratios of resampled means;
every measure of a run uses the same resamples), paired per-room differences, room-level and pooled trial-level sign
tests, as scripts/exp13_analysis.py and scripts/frame_cut_analysis.py do for run 0. Every measure is also computed on
the log-odds scale, from each room's empirical logit log((k + 0.5) / (n - k + 0.5)). Accuracies and differences are
fractions (0.925 = 92.5%); log-odds are in nats.

Conditions. 31a (experiment 13a): O ordered and S shuffled with real frame numbers (experiment 9c's answers; P5005's
rerun on CUDA from results/two_choice/exp31/), NW no walk (experiment 9c), V reversed, W window-shuffled, N renumbered,
R shuffled and relabeled (the 31c condition and 31a's floor). 31b (experiment 15): clean, near, far, lookalike, random,
and the layer windows on P5000-P5002. Run 0's files are the Mac (MPS) ones of experiments 10, 13a and 15; run 0 has no
R, so its floor is S (its registered definition).

The registered decision rule (each criterion: replicates if the CI lies wholly on run 0's side of the threshold,
differs if wholly on the other side, inconclusive otherwise; verdicts from points, "scale_dependent" when the log-odds
verdict differs):
  31a  gate D = O - floor >= 0.10 with CI above 0; C1 O - N < 0.03 (log-odds 0.4); C2 (W - floor)/D > 0.5;
       C3 (O - V)/D > 0.25; C4 V - floor, W - V, O - W each > 0 (differs if any CI is below 0)
  31b  gate: clean reproduces the stored answers on >= 99% of trials; e_X = logit(random) - logit(X) per room;
       F1 near - random < 0 (points); differs if > 0, or if e_near's CI lies within +-0.20; F2 near - far < 0
       (points); differs if > 0, or if (e_near - e_far)'s CI lies within +-0.14; F1/F2 replicating in direction and
       inside the margin count as replicates, flagged weaker; F3 e_lookalike - 0.5 e_near < 0 (differs if > 0)
  31c  L = S - R; first match of: L > 0 and R - NW not above 0 -> labels carry the shuffled walk's accuracy;
       L > 0 -> labels carry part of it; L within +-0.03 -> labels carry none of it; L < 0 -> relabeling helps;
       else inconclusive
  recipe (31a, 31b): the verdict runs 5 and 7 share, else "mixed"

    .venv-claude/bin/python scripts/exp31_analysis.py [--fetch]   # -> results/exp31_summary.json (top-level summary)
    .venv-claude/bin/python scripts/exp31_analysis.py --test      # run 0 only: reproduces exps 13a and 15; no file
    .venv/bin/python scripts/exp31_analysis.py --parity data/ft/qwen8b-proc-e4-s5    # job gate on vast-3 (>= 98%)
"""
import argparse
import json
import math
import os
import subprocess

import numpy as np

ROOMS = [f"P{s}" for s in range(5000, 5010)]
WINDOW_ROOMS = ["P5000", "P5001", "P5002"]
TC, FC = "results/two_choice", "results/frame_cut"
M = "Qwen3-VL-8B-Instruct+"
RUN0 = "qwen8b-proc"
RUNS = {"run5": "qwen8b-proc-e4-s5", "run7": "qwen8b-proc-e4-s7"}
TC_CONDS = {"O": "max64", "S": "max64-shuffled", "NW": "nowalk", "V": "max64-reversed", "W": "max64-window8",
            "N": "max64-renumbered", "R": "max64-shuffled-relabel"}
LABELS = {"O": "ordered", "S": "shuffled", "R": "relabeled shuffle", "W": "window-shuffled", "V": "reversed",
          "N": "renumbered", "NW": "no walk"}
ORDER_CONTRASTS = [("W", "S"), ("O", "V"), ("O", "N"), ("W", "O"), ("V", "S"), ("N", "S"),        # run 0's table
                   ("W", "R"), ("V", "R"), ("O", "R"), ("S", "R"), ("R", "NW")]
CUTS = ["clean", "near", "far", "lookalike", "random"]
WINDOWS = [f"{s}-layers{w}" for s in ("near", "lookalike") for w in ("1-18", "19-36")]
CUT_CONTRASTS = [("near", "random", "overlap_near_lookalike"), ("lookalike", "random", "overlap_near_lookalike"),
                 ("near", "far", "overlap_near_far")]
TC_FIELDS = ("walk", "probe", "familiar", "near", "far")
FC_FIELDS = ("walk", "probe", "familiar", "near", "far", "a", "b")
EXP29 = "results/vlm_intervention/" + M + "{a}-near-cut.jsonl"
PARITY_MIN, CHECK_MIN = 0.98, 0.99


# ---------------------------------------------------------------- files

def tc_path(adapter, room, cond):
    """Run 0: the Mac's files. Four-epoch runs: experiment 9c's CUDA files, except P5005's ordered and shuffled walks
    (this experiment's CUDA reruns) and its no walk (experiment 9c's MPS record)."""
    if adapter == RUN0:
        return f"{TC}/{M}{adapter}-room{room}-256px-{cond}-mps.jsonl"
    if room == "P5005" and cond in ("max64", "max64-shuffled"):
        return f"{TC}/exp31/{M}{adapter}-room{room}-256px-{cond}.jsonl"
    return f"{TC}/{M}{adapter}-room{room}-256px-{cond}" + ("-mps" if room == "P5005" and cond == "nowalk" else "") + ".jsonl"


def fc_path(adapter, cond):
    return f"{FC}/{M}{adapter}-{cond}" + ("-mps" if adapter == RUN0 else "") + ".jsonl"


def stored_ordered(adapter, room):
    """The answers experiment 15's check compares with: experiment 10 / 9c's ordered walk as stored."""
    if adapter == RUN0 or room == "P5005":
        return f"{TC}/{M}{adapter}-room{room}-256px-max64-mps.jsonl"
    return f"{TC}/{M}{adapter}-room{room}-256px-max64.jsonl"


def manifest(room):
    return {t["id"]: t for t in json.load(open(f"results/export/dissimilar-room{room}/trials.json"))}


def read(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def load_tc(path, trials):
    """id -> row if the file answers every trial of the room's manifest; None if missing or incomplete."""
    if not os.path.exists(path):
        return None
    rows = read(path)
    out = {r["id"]: r for r in rows}
    bad = [r["id"] for r in rows if r["id"] not in trials or any(r[k] != trials[r["id"]][k] for k in TC_FIELDS)
           or bool(r["correct"]) != (r["answer"] == r["near"])]
    assert len(out) == len(rows) and not bad, (path, len(rows), bad[:3])
    return out if set(trials) <= set(out) else None


def load_fc(path, novel, rooms):
    """(room, id) -> row if the file covers every novel trial of the rooms; None if missing or incomplete."""
    if not os.path.exists(path):
        return None
    rows = read(path)
    out = {(x["room"], x["id"]): x for x in rows}
    bad = [k for k, x in out.items() if x["id"] not in novel[x["room"]]
           or any(x[f] != novel[x["room"]][x["id"]].get(f, x[f]) for f in FC_FIELDS)
           or bool(x["correct"]) != (x["answer"] == x["near"])]
    assert len(out) == len(rows) and not bad, (path, len(rows), bad[:3])
    want = {(r, i) for r in rooms for i in novel[r]}
    return {k: out[k] for k in want} if want <= set(out) else None


# ---------------------------------------------------------------- statistics

def sign_p(up, dn):
    n = up + dn
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(max(up, dn), n + 1)) / 2 ** n) if n else 1.0


def est(g, I):
    """g(M) builds a measure from M(x), the mean over rooms of a per-room array x. Point estimate with M = mean; CI
    from the same g with M = the mean over each bootstrap resample of rooms (rows of I)."""
    pt = g(lambda x: float(np.mean(x)))
    bs = g(lambda x: np.asarray(x, float)[I].mean(1))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"mean": float(pt), "ci95": [float(lo), float(hi)]}


def side(e, t, above):
    """replicates / differs / inconclusive: is the CI wholly on run 0's side of t (above it if `above`)?"""
    lo, hi = e["ci95"]
    if above:
        return "replicates" if lo > t else "differs" if hi < t else "inconclusive"
    return "replicates" if hi < t else "differs" if lo > t else "inconclusive"


def within(e, m):
    return -m < e["ci95"][0] and e["ci95"][1] < m


def combine(verdicts):
    v = list(verdicts.values())
    if all(x == "replicates" for x in v):
        return "replicates"
    if any(x == "differs" for x in v):
        return "differs"
    return "inconclusive (" + ", ".join(k for k, x in verdicts.items() if x != "replicates") + ")"


class Cond:
    """One condition's novel-trial answers in some rooms: correct by (room, id), per-room accuracy and log-odds."""

    def __init__(self, rows, rooms, ids):
        self.rows, self.rooms, self.ids = rows, rooms, ids
        k = np.array([sum(bool(rows[r, i]["correct"]) for i in ids[r]) for r in rooms], float)
        n = np.array([len(ids[r]) for r in rooms], float)
        self.acc, self.lo = k / n, np.log((k + 0.5) / (n - k + 0.5))

    def table(self, I):
        ans = [self.rows[r, i] for r in self.rooms for i in self.ids[r]]
        return {**est(lambda M: M(self.acc), I), "per_room": dict(zip(self.rooms, map(float, self.acc))),
                "log_odds": est(lambda M: M(self.lo), I), "pooled": float(np.mean([x["correct"] for x in ans])),
                "first_listed": float(np.mean([x["answer"] == x["a"] for x in ans])),
                "parsed": float(np.mean([x["answer"] in (x["a"], x["b"]) for x in ans]))}


def contrast(a, b, I, keep=None):
    """a - b per room (a, b: Cond on the same rooms), with sign tests; keep: (room, id) -> bool subset of trials."""
    ids = {r: [i for i in a.ids[r] if keep is None or keep(r, i)] for r in a.rooms}
    rooms = [r for r in a.rooms if ids[r]]
    if keep is not None:
        a, b = Cond(a.rows, rooms, ids), Cond(b.rows, rooms, ids)
        I = np.random.default_rng(1).integers(0, len(rooms), (I.shape[0], len(rooms)))
    d = a.acc - b.acc
    up = sum(bool(a.rows[r, i]["correct"]) and not b.rows[r, i]["correct"] for r in rooms for i in ids[r])
    dn = sum(bool(b.rows[r, i]["correct"]) and not a.rows[r, i]["correct"] for r in rooms for i in ids[r])
    return {**est(lambda M: M(d), I), "log_odds": est(lambda M: M(a.lo - b.lo), I),
            "n_trials": sum(len(ids[r]) for r in rooms), "n_rooms": len(rooms),
            "per_room": dict(zip(rooms, map(float, d))), "rooms_up": int((d > 0).sum()),
            "rooms_down": int((d < 0).sum()), "room_sign_p": sign_p(int((d > 0).sum()), int((d < 0).sum())),
            "trials_better": int(up), "trials_worse": int(dn), "trial_sign_p": sign_p(up, dn)}


def agreement(a, b, keys):
    same = [a[k]["answer"] == b[k]["answer"] for k in keys]
    return {"agree": int(sum(same)), "n": len(same), "share": float(np.mean(same)) if same else float("nan")}


# ---------------------------------------------------------------- one run

def load_run(adapter, trials, novel):
    """Every condition this run has; missing or incomplete ones listed."""
    ids = {r: sorted(novel[r]) for r in ROOMS}
    c, missing = {}, []
    for key, cond in TC_CONDS.items():
        if adapter == RUN0 and key == "R":
            continue
        d = {r: load_tc(tc_path(adapter, r, cond), trials[r]) for r in ROOMS}
        if any(v is None for v in d.values()):
            missing.append(LABELS[key])
            continue
        c[key] = Cond({(r, i): d[r][i] for r in ROOMS for i in ids[r]}, ROOMS, ids)
    for cond in CUTS:
        d = load_fc(fc_path(adapter, cond), novel, ROOMS)
        if d is None:
            missing.append(cond)
            continue
        c[cond] = Cond(d, ROOMS, ids)
    w = {}
    wids = {r: ids[r] for r in WINDOW_ROOMS}
    for cond in WINDOWS:
        d = load_fc(fc_path(adapter, cond), novel, WINDOW_ROOMS)
        if d is None:
            missing.append(cond)
            continue
        w[cond] = Cond(d, WINDOW_ROOMS, wids)
    for cond in ("clean", "near", "lookalike"):
        if cond in c:
            w[cond] = Cond(c[cond].rows, WINDOW_ROOMS, wids)
    return c, w, ids, missing


def analyse_31a(c, floor, I):
    """Gate and C1-C4 against the given floor (R for the four-epoch runs, S for run 0's definition)."""
    if not all(k in c for k in ("O", "N", "W", "V", floor)):
        return {"verdict": "pending"}
    A = {k: c[k].acc for k in ("O", "N", "W", "V", floor)}
    G = {k: c[k].lo for k in A}
    f = floor
    out = {"floor": LABELS[f], "order_effect": est(lambda M: M(A["O"] - A[f]), I),
           "order_effect_log_odds": est(lambda M: M(G["O"] - G[f]), I), "criteria": {}}
    D = out["order_effect"]
    out["gate_passed"] = D["mean"] >= 0.10 and D["ci95"][0] > 0
    for scale, X, t1 in (("points", A, 0.03), ("log_odds", G, 0.4)):
        m = {"C1_renumber_cost": est(lambda M: M(X["O"] - X["N"]), I),
             "C2_local_share": est(lambda M: M(X["W"] - X[f]) / M(X["O"] - X[f]), I),
             "C3_reversal_share": est(lambda M: M(X["O"] - X["V"]) / M(X["O"] - X[f]), I),
             "C4_reversed_minus_floor": est(lambda M: M(X["V"] - X[f]), I),
             "C4_window_minus_reversed": est(lambda M: M(X["W"] - X["V"]), I),
             "C4_ordered_minus_window": est(lambda M: M(X["O"] - X["W"]), I)}
        c4 = [m[k] for k in m if k.startswith("C4")]
        v = {"C1": side(m["C1_renumber_cost"], t1, above=False), "C2": side(m["C2_local_share"], 0.5, above=True),
             "C3": side(m["C3_reversal_share"], 0.25, above=True),
             "C4": "replicates" if all(e["ci95"][0] > 0 for e in c4)
             else "differs" if any(e["ci95"][1] < 0 for e in c4) else "inconclusive"}
        out[scale] = {"measures": m, "verdicts": v}
    for k, v in out["points"]["verdicts"].items():
        out["criteria"][k] = v + (" (scale-dependent)" if out["log_odds"]["verdicts"][k] != v else "")
    if not out["gate_passed"]:
        out["verdict"] = "inconclusive (gate: order effect under 10 points)"
    else:
        out["verdict"] = combine(out["points"]["verdicts"])
        if any("scale-dependent" in v for v in out["criteria"].values()):
            out["verdict"] += "; scale-dependent: " + ", ".join(k for k, v in out["criteria"].items()
                                                               if "scale-dependent" in v)
    return out


def analyse_31b(c, check, I):
    if not all(k in c for k in ("near", "far", "lookalike", "random")):
        return {"verdict": "pending", "clean_check": check}
    A = {k: c[k].acc for k in ("near", "far", "lookalike", "random")}
    G = {k: c[k].lo for k in A}
    m = {"F1_near_minus_random": est(lambda M: M(A["near"] - A["random"]), I),
         "F1_e_near": est(lambda M: M(G["random"] - G["near"]), I),
         "F2_near_minus_far": est(lambda M: M(A["near"] - A["far"]), I),
         "F2_e_near_minus_e_far": est(lambda M: M(G["far"] - G["near"]), I),
         "F3_e_lookalike_minus_half_e_near": est(lambda M: M((G["random"] - G["lookalike"])
                                                            - 0.5 * (G["random"] - G["near"])), I),
         "e_lookalike": est(lambda M: M(G["random"] - G["lookalike"]), I),
         "e_far": est(lambda M: M(G["random"] - G["far"]), I),
         "lookalike_minus_random": est(lambda M: M(A["lookalike"] - A["random"]), I)}
    v, weaker = {}, {}
    for k, pts, lo, margin in (("F1", "F1_near_minus_random", "F1_e_near", 0.20),
                               ("F2", "F2_near_minus_far", "F2_e_near_minus_e_far", 0.14)):
        eq = within(m[lo], margin)
        if m[pts]["ci95"][1] < 0:
            v[k], weaker[k] = "replicates", eq
        else:
            v[k], weaker[k] = ("differs" if m[pts]["ci95"][0] > 0 or eq else "inconclusive"), False
    v["F3"] = side(m["F3_e_lookalike_minus_half_e_near"], 0.0, above=False)
    crit = {k: x + (" (weaker)" if weaker.get(k) else "") for k, x in v.items()}
    gate = check is not None and check["share"] >= CHECK_MIN
    verdict = combine(v) if gate else "inconclusive (gate: clean check)"
    if gate and any(weaker.values()):
        verdict += "; weaker: " + ", ".join(k for k, w in weaker.items() if w)
    return {"measures": m, "criteria": crit, "clean_check": check, "gate_passed": gate, "verdict": verdict}


def analyse_31c(c, I):
    if not all(k in c for k in ("S", "R", "NW")):
        return {"reading": "pending"}
    A = {k: c[k].acc for k in ("S", "R", "NW")}
    m = {"label_effect_S_minus_R": est(lambda M: M(A["S"] - A["R"]), I),
         "label_effect_log_odds": est(lambda M: M(c["S"].lo - c["R"].lo), I),
         "relabeled_minus_no_walk": est(lambda M: M(A["R"] - A["NW"]), I),
         "relabeled_minus_chance": est(lambda M: M(A["R"] - 0.5), I),
         "shuffled_minus_no_walk": est(lambda M: M(A["S"] - A["NW"]), I)}
    L, RN = m["label_effect_S_minus_R"]["ci95"], m["relabeled_minus_no_walk"]["ci95"]
    small = within(m["label_effect_S_minus_R"], 0.03)
    if L[0] > 0:
        reading = ("the labels carry part of it" if RN[0] > 0 else "the labels carry the shuffled walk's accuracy")
        if small:
            reading += " (the effect also lies within +-3 points)"
    elif small:
        reading = "the labels carry none of it"
    elif L[1] < 0:
        reading = "relabeling helps"
    else:
        reading = "inconclusive"
    return {"measures": m, "reading": reading}


def analyse_run(adapter, trials, novel, rng):
    c, w, ids, missing = load_run(adapter, trials, novel)
    I = rng.integers(0, len(ROOMS), (10_000, len(ROOMS)))
    out = {"adapter": f"data/ft/{adapter}", "incomplete": missing,
           "conditions": {LABELS.get(k, k): v.table(I) for k, v in c.items()}, "contrasts": {}}
    for a, b in ORDER_CONTRASTS:
        if a in c and b in c:
            out["contrasts"][f"{LABELS[a]} - {LABELS[b]}"] = contrast(c[a], c[b], I)
    for a, b, o in CUT_CONTRASTS:
        if a in c and b in c:
            out["contrasts"][f"{a} - {b}"] = contrast(c[a], c[b], I)
            out["contrasts"][f"{a} - {b}, without {o}"] = {
                "without": o, **contrast(c[a], c[b], I, keep=lambda r, i, a=a: c[a].rows[r, i][o] == 0)}
    for x in ("near", "far", "lookalike", "random"):
        if x in c and "clean" in c:
            out["contrasts"][f"{x} - clean"] = contrast(c[x], c["clean"], I)
    WI = rng.integers(0, len(WINDOW_ROOMS), (10_000, len(WINDOW_ROOMS)))
    out["windows"] = {"rooms": WINDOW_ROOMS, "conditions": {k: v.table(WI) for k, v in w.items()}, "contrasts": {}}
    for x in WINDOWS:
        s = x.split("-")[0]
        for ref in ("clean", s):
            if x in w and ref in w:
                out["windows"]["contrasts"][f"{x} - {ref}{' (all layers)' if ref == s else ''}"] = contrast(w[x], w[ref], WI)
    # experiment 15's check: clean against the stored ordered answers (experiment 10 / 9c)
    check = None
    if "clean" in c:
        ref = {(r, x["id"]): x for r in ROOMS for x in read(stored_ordered(adapter, r))}
        keys = [(r, i) for r in ROOMS for i in ids[r]]
        check = {**agreement(c["clean"].rows, ref, keys),
                 "per_room": {r: agreement(c["clean"].rows, ref, [(r, i) for i in ids[r]])["share"] for r in ROOMS},
                 "mismatches": [f"{r}/{i}" for r, i in keys if c["clean"].rows[r, i]["answer"] != ref[r, i]["answer"]]}
    out["31a"] = analyse_31a(c, "S" if adapter == RUN0 else "R", I)
    if adapter != RUN0:
        out["31a_with_S"] = analyse_31a(c, "S", I)
        out["31c"] = analyse_31c(c, I)
    out["31b"] = analyse_31b(c, check, I)
    return out, c


def cross_device(adapter, trials):
    """P5005: this experiment's CUDA reruns against experiment 9c's MPS answers; the parity rerun of P5000."""
    out = {}
    for cond in ("max64", "max64-shuffled"):
        a = load_tc(f"{TC}/exp31/{M}{adapter}-roomP5005-256px-{cond}.jsonl", trials["P5005"])
        b = load_tc(f"{TC}/{M}{adapter}-roomP5005-256px-{cond}-mps.jsonl", trials["P5005"])
        if a and b:
            ids = sorted(trials["P5005"])
            nov = [i for i in ids if not trials["P5005"][i]["familiar"]]
            out[f"P5005 {cond} CUDA vs MPS"] = {
                "all": agreement(a, b, ids), "novel": agreement(a, b, nov),
                "accuracy_novel_cuda": float(np.mean([a[i]["correct"] for i in nov])),
                "accuracy_novel_mps": float(np.mean([b[i]["correct"] for i in nov]))}
    p = load_tc(f"{TC}/exp31/{M}{adapter}-roomP5000-256px-max64-parity.jsonl", trials["P5000"])
    if p:
        out["parity P5000 ordered"] = agreement(p, load_tc(stored_ordered(adapter, "P5000"), trials["P5000"]),
                                                sorted(trials["P5000"]))
    e29 = EXP29.format(a=adapter)
    if os.path.exists(e29):
        x = {(r["room"], r["id"]): r for r in read(e29)}
        y = {(r["room"], r["id"]): r for r in read(fc_path(adapter, "near"))} if os.path.exists(fc_path(adapter, "near")) else {}
        keys = sorted(set(x) & set(y))
        if keys:
            out["near cut vs experiment 29 near-cut"] = agreement(x, y, keys)
    return out


def vs_run0(c, c0, rng):
    """Paired per-room differences from run 0 (same rooms, same resamples)."""
    I = rng.integers(0, len(ROOMS), (10_000, len(ROOMS)))
    out = {}
    if all(k in c0 for k in ("O", "S", "W", "V")):
        s0 = lambda M, X: M(X["W"] - X["S"]) / M(X["O"] - X["S"])
        for scale in ("acc", "lo"):
            X0 = {k: getattr(c0[k], scale) for k in ("O", "S", "W", "V")}
            for floor in ("R", "S"):
                if all(k in c for k in ("O", "W", "V", floor)):
                    X = {k: getattr(c[k], scale) for k in ("O", "W", "V")}
                    X[floor] = getattr(c[floor], scale)
                    out[f"local_share_{floor}_minus_run0_S" + ("_log_odds" if scale == "lo" else "")] = est(
                        lambda M: M(X["W"] - X[floor]) / M(X["O"] - X[floor]) - s0(M, X0), I)
                    out[f"reversal_share_{floor}_minus_run0_S" + ("_log_odds" if scale == "lo" else "")] = est(
                        lambda M: M(X["O"] - X["V"]) / M(X["O"] - X[floor]) - M(X0["O"] - X0["V"]) / M(X0["O"] - X0["S"]), I)
    if all(k in c and k in c0 for k in ("near", "random")):
        out["e_near_minus_run0"] = est(lambda M: M(c["random"].lo - c["near"].lo) - M(c0["random"].lo - c0["near"].lo), I)
    for k in ("O", "clean"):
        if k in c and k in c0:
            out[f"{LABELS.get(k, k)}_minus_run0"] = est(lambda M: M(c[k].acc - c0[k].acc), I)
    return out


# ---------------------------------------------------------------- summary, checks, entry point

def short(run):
    """The numbers the paper needs, per run."""
    s = {"incomplete": run["incomplete"]}
    cond = run["conditions"]
    pick = lambda e: {"mean": e["mean"], "ci95": e["ci95"]}
    s["accuracy"] = {k: pick(v) for k, v in cond.items()}
    a = run["31a"]
    if "points" in a:
        s["31a"] = {"verdict": a["verdict"], "criteria": a["criteria"], "floor": a["floor"],
                    "order_effect": pick(a["order_effect"]),
                    **{k: pick(v) for k, v in a["points"]["measures"].items()},
                    "C2_local_share_log_odds": pick(a["log_odds"]["measures"]["C2_local_share"]),
                    "C3_reversal_share_log_odds": pick(a["log_odds"]["measures"]["C3_reversal_share"])}
        if "31a_with_S" in run and "points" in run["31a_with_S"]:
            w = run["31a_with_S"]
            s["31a"]["with_S"] = {"verdict": w["verdict"], "order_effect": pick(w["order_effect"]),
                                  "C2_local_share": pick(w["points"]["measures"]["C2_local_share"]),
                                  "C3_reversal_share": pick(w["points"]["measures"]["C3_reversal_share"])}
    else:
        s["31a"] = {"verdict": a["verdict"]}
    b = run["31b"]
    s["31b"] = {"verdict": b["verdict"], "clean_check": b["clean_check"] and
                {k: b["clean_check"][k] for k in ("agree", "n", "share")}}
    if "measures" in b:
        s["31b"].update(criteria=b["criteria"], **{k: pick(v) for k, v in b["measures"].items()})
        for k in ("near - random, without overlap_near_lookalike", "near - far, without overlap_near_far"):
            if k in run["contrasts"]:
                s["31b"][k] = pick(run["contrasts"][k])
    if "31c" in run:
        s["31c"] = {"reading": run["31c"]["reading"],
                    **{k: pick(v) for k, v in run["31c"].get("measures", {}).items()}}
    if "vs_run0" in run:
        s["vs_run0"] = run["vs_run0"]
    return s


def recipe(runs, part):
    v = {r: runs[r][part]["verdict"] for r in runs}
    base = {r: x.split(";")[0] for r, x in v.items()}
    return base["run5"] if base["run5"] == base["run7"] else "mixed"


def test_run0(run0):
    """Run 0's numbers against results/exp13_summary.json (13a) and results/frame_cut_summary.json (15): means and
    sign tests exactly, CIs within 1 point (different bootstrap draws)."""
    e13 = json.load(open("results/exp13_summary.json"))["13a"]
    e15 = json.load(open("results/frame_cut_summary.json"))
    names = {"ordered": "ordered", "shuffled": "shuffled", "no walk": "no walk", "reversed": "reversed",
             "window-shuffled": "window-shuffled", "renumbered": "renumbered"}
    bad, n = [], 0

    def cmp(tag, mine, ref, keys=("rooms_up", "rooms_down", "trials_better", "trials_worse")):
        nonlocal n
        n += 1
        if abs(mine["mean"] - ref["mean"]) > 1e-12:
            bad.append(f"{tag}: mean {mine['mean']} vs {ref['mean']}")
        if max(abs(x - y) for x, y in zip(mine["ci95"], ref["ci95"])) > 0.01:
            bad.append(f"{tag}: CI {mine['ci95']} vs {ref['ci95']}")
        for k in keys:
            if k in ref and mine[k] != ref[k]:
                bad.append(f"{tag}: {k} {mine[k]} vs {ref[k]}")
    for k, ref in e13["conditions"].items():       # (its first-listed rates are over all trials, ours novel only)
        cmp(f"13a {k}", run0["conditions"][names[k]], ref, keys=())
    for k, ref in e13["contrasts"].items():
        cmp(f"13a {k}", run0["contrasts"][k], ref)
    for k, ref in e15["conditions"].items():
        cmp(f"15 {k}", run0["conditions"][k], ref, keys=("pooled", "first_listed", "parsed"))
    for k, ref in e15["contrasts"].items():
        cmp(f"15 {k}", run0["contrasts"][k], ref)
    for k, ref in e15["contrasts_without_overlap"].items():
        cmp(f"15 {k} without overlap", run0["contrasts"][f"{k}, without {ref['without']}"], ref,
            keys=("n_trials", "rooms_up", "rooms_down", "trials_better", "trials_worse"))
    for k, ref in e15["cut_minus_clean"].items():
        cmp(f"15 {k}", run0["contrasts"][k], ref)
    for k, ref in e15["windows"]["contrasts"].items():
        cmp(f"15 window {k}", run0["windows"]["contrasts"][k], ref)
    ref_check = e15["clean_vs_exp10"]
    if (run0["31b"]["clean_check"]["agree"], run0["31b"]["clean_check"]["n"]) != (ref_check["agree"], ref_check["n"]):
        bad.append(f"15 clean check {run0['31b']['clean_check']['agree']} vs {ref_check['agree']}")
    return n, bad


def parity(adapter_dir):
    """Job gate on vast-3: the P5000 ordered rerun against experiment 9c's stored answers (>= 98% identical)."""
    a = os.path.basename(adapter_dir.rstrip("/"))
    trials = manifest("P5000")
    new = load_tc(f"{TC}/exp31/{M}{a}-roomP5000-256px-max64-parity.jsonl", trials)
    old = load_tc(stored_ordered(a, "P5000"), trials)
    if new is None or old is None:
        raise SystemExit(f"PARITY FAILED: rerun or stored answers incomplete for {a}")
    g = agreement(new, old, sorted(trials))
    acc = lambda d: np.mean([d[i]["correct"] for i in trials if not trials[i]["familiar"]])
    print(f"parity {a} P5000 ordered: {g['agree']}/{g['n']} answers identical ({g['share']:.1%}); novel accuracy "
          f"{acc(new):.1%} vs stored {acc(old):.1%}")
    if g["share"] < PARITY_MIN:
        print("PARITY FAILED: below 98%; nothing else may run")
        raise SystemExit(1)
    print("PARITY PASSED")


def fetch():
    """Only this experiment's answer files from the vast-3 container, never its whole results/."""
    inc = " ".join(f'--include="*e4-s[57]-roomP500[0-9]-256px-{c}.jsonl"'
                   for c in ("max64-reversed", "max64-window8", "max64-renumbered", "max64-shuffled-relabel"))
    cmd = ('source scripts/vast1/common.sh && '
           f'rsync -a -e "$MI_RSYNC_SHELL" {inc} --exclude="*" "$MI_CONTAINER:$MI_REMOTE_ROOT/{TC}/" {TC}/ && '
           f'mkdir -p {TC}/exp31 && rsync -a -e "$MI_RSYNC_SHELL" --include="*e4-s[57]-*.jsonl" --exclude="*" '
           f'"$MI_CONTAINER:$MI_REMOTE_ROOT/{TC}/exp31/" {TC}/exp31/ && '
           f'rsync -a -e "$MI_RSYNC_SHELL" --include="*e4-s[57]-*.jsonl" --exclude="*" '
           f'"$MI_CONTAINER:$MI_REMOTE_ROOT/{FC}/" {FC}/')
    subprocess.run(["bash", "-c", cmd], check=True, env={**os.environ, "MI_HOST": "vast3"})


def fmt(e, pct=True):
    f = (lambda x: f"{100 * x:+.1f}") if pct else (lambda x: f"{x:+.2f}")
    return f"{f(e['mean'])} [{f(e['ci95'][0])}, {f(e['ci95'][1])}]"


def report(name, run):
    print(f"{name} ({run['adapter']}); incomplete: {', '.join(run['incomplete']) or 'none'}")
    for k, v in run["conditions"].items():
        print(f"  {k:18s} {100 * v['mean']:5.1f}% [{100 * v['ci95'][0]:.1f}, {100 * v['ci95'][1]:.1f}]  "
              f"first-listed {100 * v['first_listed']:.0f}%")
    for part in ("31a", "31a_with_S", "31b", "31c"):
        x = run.get(part)
        if not x:
            continue
        print(f"  {part}: {x.get('verdict', x.get('reading'))}")
        if "points" in x:
            print(f"    order effect {fmt(x['order_effect'])}, criteria {x['criteria']}")
            for k, v in x["points"]["measures"].items():
                print(f"    {k:28s} {fmt(v, 'share' not in k)}   log-odds {fmt(x['log_odds']['measures'][k], False)}")
        elif "measures" in x:
            if "criteria" in x:
                cc = x["clean_check"]
                print(f"    criteria {x['criteria']}; clean check "
                      + (f"{cc['agree']}/{cc['n']} ({cc['share']:.2%})" if cc else "missing"))
            for k, v in x["measures"].items():
                print(f"    {k:34s} {fmt(v, not k.startswith(('F1_e', 'F2_e', 'F3', 'e_', 'label_effect_log')))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="first copy this experiment's answer files from vast-3")
    ap.add_argument("--test", action="store_true", help="run 0 only: reproduce experiments 13a and 15; write nothing")
    ap.add_argument("--parity", metavar="ADAPTER", help="job gate: P5000 ordered rerun vs stored (exit 1 below 98%%)")
    ap.add_argument("--out", default="results/exp31_summary.json")
    args = ap.parse_args()
    if args.parity:
        return parity(args.parity)
    if args.fetch:
        fetch()
    trials = {r: manifest(r) for r in ROOMS}
    novel = {r: {i: t for i, t in trials[r].items() if not t["familiar"]} for r in ROOMS}
    rng = np.random.default_rng(0)
    run0, c0 = analyse_run(RUN0, trials, novel, rng)
    if args.test:
        n, bad = test_run0(run0)
        report("run 0", run0)
        print(f"reproduction of experiments 13a and 15 from run 0's files: {n - len(bad)}/{n} entries match")
        for b in bad:
            print("  MISMATCH " + b)
        raise SystemExit(1 if bad else 0)
    runs = {}
    for name, adapter in RUNS.items():
        runs[name], c = analyse_run(adapter, trials, novel, rng)
        runs[name]["vs_run0"] = vs_run0(c, c0, rng)
        runs[name]["cross_device_and_parity"] = cross_device(adapter, trials)
    summary = {"verdicts": {"31a": {**{r: runs[r]["31a"]["verdict"] for r in runs}, "recipe": recipe(runs, "31a")},
                            "31b": {**{r: runs[r]["31b"]["verdict"] for r in runs}, "recipe": recipe(runs, "31b")},
                            "31c": {r: runs[r]["31c"]["reading"] for r in runs}},
               "n_novel": sum(len(v) for v in novel.values()),
               **{r: short(runs[r]) for r in runs},
               "run0": short(run0),
               "checks": {r: runs[r]["cross_device_and_parity"] for r in runs}}
    json.dump({"summary": summary, "runs": runs, "run0": run0}, open(args.out, "w"), indent=1)
    for name in runs:
        report(name, runs[name])
    print(json.dumps(summary["verdicts"], indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
