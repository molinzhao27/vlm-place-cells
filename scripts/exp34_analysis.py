"""Experiment 34's pre-registered analysis (docs/experiments.md): does the cost of reversing the walk come from the
direction inside short stretches of it, or from the direction of the longer sequence? Four-epoch runs 5 and 7.

Experiment 31 reversed the whole 64-frame walk (V) and shuffled its blocks of 8 shown frames (W). Here the same blocks
(two_choice.py's --window-shuffle cut, counted from the walk's start, so the last, shorter block holds its final frames)
are reversed at one level only: B, the blocks last to first, each in time order (--reverse-blocks 8, -revblocks8); I,
the blocks first to last, each in reverse time order (--reverse-within-blocks 8, -revwithin8). Both are labelled 1, 2,
... as shown, as V and W are. B read backward is I, as V read backward is O. B, I and W break the walk at the same seven
block edges (W keeps about one of them by chance); O and V break it nowhere.

O, R, W and V are experiment 31's answers on the same trials (O: experiment 9c's, P5005 experiment 31's CUDA rerun),
read with scripts/exp31_analysis.py's loaders; only two of experiment 31's conditions are rerun, as the parity check.

Per run, on the 1,299 novel trials of P5000-P5009, with experiment 31's definitions: per-room accuracy (unparsed
answers wrong) and empirical log-odds log((k + 0.5) / (n - k + 0.5)), means over rooms, 95% bootstrap CIs over rooms
(10,000 resamples; ratios as ratios of resampled means), paired per-room differences, room-level and pooled trial-level
sign tests. Each run uses the room resamples experiment 31 drew for it (its seed-0 sequence replayed), so O, R, W and
V's stored values reproduce exactly and every new measure shares their resamples. Order effect D = O - R; as shares of D:
  retained share of X   (X - R) / D, for W, V, B, I (experiment 31's local share is W's; its reversal share 1 - V's)
  BI = (B - I) / D      which level's reversal costs more, at the same broken block edges (the reviewer's comparison)
  BW = (B - W) / D      the block sequence reversed against shuffled, each block forward in both, same block edges
  IV = (I - V) / D      the blocks put back in forward order while the frames inside them stay reversed
Reported, not part of the verdict: the 2 x 2 main effects m_within = ((O + B) - (I + V)) / 2D and
m_blocks = ((O + I) - (B + V)) / 2D (they sum to experiment 31's reversal share; an edge cost cancels from them only if
it adds to B and I alike), the edge term ((O + V) - (B + I)) / 2D, every pairwise difference in points, parse rates,
and how often each condition's answer is the option shown later in the prompt.

The registered decision rule (per run; points decide, a log-odds reading that differs is flagged "scale-dependent"):
  gate     D >= 0.10 with its CI above 0 (experiment 31's gate), and the job's parity reruns >= 98% identical
  each contrast, T = 0.10: "above" if its CI lies above +T, "below" if below -T, "within" if within +-T,
           else "unresolved"
  short-range direction    BI above, BW within (or above: flagged "a reversed block sequence keeps more than a
                           shuffled one"), IV's CI upper end below +T
  longer-range direction   BI below, BW below, IV above
  both                     BW below and BI not below (CI upper end at or above -T); "mostly short-range" if BI above,
                           else "comparable"
  inconclusive             otherwise, naming each contrast's state
  recipe                   the reading runs 5 and 7 share (without the qualifier), else "mixed"

    .venv-claude/bin/python scripts/exp34_analysis.py [--fetch]   # -> results/exp34_summary.json (top-level summary)
    .venv-claude/bin/python scripts/exp34_analysis.py --smoke short|long   # stand-in answer files; writes nothing
    .venv/bin/python scripts/exp34_analysis.py --parity data/ft/qwen8b-proc-e4-s5    # job gate on vast-3 (>= 98%)
--smoke short reads W's files for B and V's for I (a model that ignores the block sequence and pays no edge cost once
the frames are reversed); --smoke long reads V's for B and W's for I (the mirror image). They check the code path, the
reproduction of experiment 31's stored values and the readings; no new answer file is needed.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import exp31_analysis as e31  # noqa: E402
from spatialcells import two_choice as tc  # noqa: E402

ROOMS, TC, M, RUNS = e31.ROOMS, e31.TC, e31.M, e31.RUNS
CONDS = {"O": "max64", "R": "max64-shuffled-relabel", "W": "max64-window8", "V": "max64-reversed",
         "B": "max64-revblocks8", "I": "max64-revwithin8"}
LABELS = {"O": "ordered", "R": "relabeled shuffle", "W": "window-shuffled", "V": "reversed",
          "B": "blocks reversed", "I": "reversed within blocks"}
FLAGS = {"O": {}, "R": {"shuffle_walk": True, "relabel": True}, "W": {"window_shuffle": 8}, "V": {"reverse_walk": True},
         "B": {"reverse_blocks": 8}, "I": {"reverse_within_blocks": 8}}
SMOKE = {"short": {"B": "W", "I": "V"}, "long": {"B": "V", "I": "W"}}
PARITY = ("max64-window8", "max64-reversed")
PARITY_MIN, T = 0.98, 0.10
CONTRASTS = [("B", "I"), ("B", "W"), ("I", "V"), ("O", "B"), ("O", "I"), ("B", "V"), ("I", "W"), ("B", "R"),
             ("I", "R"), ("O", "V"), ("O", "W"), ("W", "V")]
OUT = "results/exp34_summary.json"


def parity_path(adapter, cond):
    return f"{TC}/exp34/{M}{adapter}-roomP5000-256px-{cond}-parity.jsonl"


def present_args(key):
    a = dict(no_walk=False, max_images=64, shuffle_walk=False, relabel=False, reverse_walk=False, window_shuffle=None,
             reverse_blocks=None, reverse_within_blocks=None, renumber=False, irrelevant_walk=None)
    return argparse.Namespace(**{**a, **FLAGS[key]})


def shown_later(novel):
    """(room, id) -> {condition: is the near option shown after the far one}, from two_choice.present() itself."""
    out = {}
    for r in ROOMS:
        n = {w: len(i) for w, i in tc.walk_order(dict(np.load(f"data/walks-room{r}/meta.npz"))).items()}
        for i, t in novel[r].items():
            out[r, i] = {}
            for k in CONDS:
                shown, _ = tc.present(t, n[t["walk"]], present_args(k))
                out[r, i][k] = shown.index(t["near"]) > shown.index(t["far"])
    return out


def exp31_resamples():
    """The room resamples experiment 31 drew for runs 5 and 7: its rng(0) sequence (run 0's rooms and layer-window
    rooms, then per run its rooms, layer-window rooms and paired differences from run 0), replayed."""
    rng = np.random.default_rng(0)
    draw = lambda n: rng.integers(0, n, (10_000, n))
    draw(len(ROOMS)), draw(len(e31.WINDOW_ROOMS))
    out = {}
    for name in RUNS:
        out[name] = draw(len(ROOMS))
        draw(len(e31.WINDOW_ROOMS)), draw(len(ROOMS))
    return out


# ---------------------------------------------------------------- one run

def load_run(adapter, trials, novel, smoke=None):
    ids = {r: sorted(novel[r]) for r in ROOMS}
    c, missing = {}, []
    for key in CONDS:
        cond = CONDS[SMOKE[smoke].get(key, key)] if smoke else CONDS[key]
        d = {r: e31.load_tc(e31.tc_path(adapter, r, cond), trials[r]) for r in ROOMS}
        if any(v is None for v in d.values()):
            missing.append(LABELS[key])
            continue
        c[key] = e31.Cond({(r, i): d[r][i] for r in ROOMS for i in ids[r]}, ROOMS, ids)
    return c, ids, missing


def measures(X, I):
    """The registered shares and the reported 2 x 2 terms on one scale (X: condition -> per-room accuracy or log-odds)."""
    D = lambda M: M(X["O"] - X["R"])
    share = lambda f: e31.est(lambda M: f(M) / D(M), I)
    m = {"order_effect": e31.est(D, I)}
    for k in ("W", "V", "B", "I"):
        m[f"retained_{k}"] = share(lambda M, k=k: M(X[k] - X["R"]))
    m["BI"] = share(lambda M: M(X["B"] - X["I"]))
    m["BW"] = share(lambda M: M(X["B"] - X["W"]))
    m["IV"] = share(lambda M: M(X["I"] - X["V"]))
    mw = lambda M: M(X["O"] + X["B"] - X["I"] - X["V"]) / 2
    mb = lambda M: M(X["O"] + X["I"] - X["B"] - X["V"]) / 2
    rv = lambda M: M(X["O"] - X["V"])
    m["m_within"], m["m_blocks"], m["reversal_share"] = share(mw), share(mb), share(rv)
    m["edge_term"] = share(lambda M: M(X["O"] + X["V"] - X["B"] - X["I"]) / 2)
    # m_within + m_blocks = the reversal share, in the point estimate and in every resample
    Mb = lambda x: np.asarray(x, float)[I].mean(1)
    m["identity_error"] = float(max(abs(mw(np.mean) + mb(np.mean) - rv(np.mean)),
                                    np.nanmax(np.abs(mw(Mb) + mb(Mb) - rv(Mb)))))
    return m


def state(e):
    lo, hi = e["ci95"]
    return "above" if lo > T else "below" if hi < -T else "within" if -T < lo and hi < T else "unresolved"


def reading(m):
    """The registered reading from BI, BW and IV (shares of D) on one scale."""
    s = {k: state(m[k]) for k in ("BI", "BW", "IV")}
    if s["BI"] == "above" and s["BW"] in ("within", "above") and m["IV"]["ci95"][1] < T:
        return "short-range direction" + (" (a reversed block sequence keeps more than a shuffled one)"
                                          if s["BW"] == "above" else ""), s
    if s["BI"] == "below" and s["BW"] == "below" and s["IV"] == "above":
        return "longer-range direction", s
    if s["BW"] == "below" and m["BI"]["ci95"][1] >= -T:
        return "both (" + ("mostly short-range" if s["BI"] == "above" else "comparable") + ")", s
    return "inconclusive (" + "; ".join(f"{k} {v}" for k, v in s.items()) + ")", s


def analyse_run(adapter, trials, novel, I, smoke=None, later=None):
    c, ids, missing = load_run(adapter, trials, novel, smoke)
    out = {"adapter": f"data/ft/{adapter}", "incomplete": missing, "conditions": {}, "contrasts": {}}
    for k, v in c.items():
        out["conditions"][LABELS[k]] = v.table(I)
        if later is not None:
            ans = [(v.rows[r, i], later[r, i][k]) for r in ROOMS for i in ids[r]]
            parsed = [(x, s) for x, s in ans if x["answer"] in (x["a"], x["b"])]
            out["conditions"][LABELS[k]]["near_shown_later"] = float(np.mean([s for _, s in ans]))
            out["conditions"][LABELS[k]]["picked_shown_later"] = float(np.mean(
                [(x["answer"] == x["near"]) == s for x, s in parsed]))
    for a, b in CONTRASTS:
        if a in c and b in c:
            out["contrasts"][f"{LABELS[a]} - {LABELS[b]}"] = e31.contrast(c[a], c[b], I)
    if not all(k in c for k in CONDS):
        return out
    out["points"] = measures({k: v.acc for k, v in c.items()}, I)
    out["log_odds"] = measures({k: v.lo for k, v in c.items()}, I)
    D = out["points"]["order_effect"]
    out["gate_D"] = D["mean"] >= 0.10 and D["ci95"][0] > 0
    for s in ("points", "log_odds"):
        out[f"reading_{s}"], out[f"states_{s}"] = reading(out[s])
    return out


def verdict(run, parity_ok):
    if "points" not in run:
        return "pending"
    if not run["gate_D"]:
        return "inconclusive (gate: order effect under 10 points)"
    if not parity_ok:
        return "inconclusive (gate: parity)"
    v = run["reading_points"]
    if run["reading_log_odds"] != v:
        v += f"; scale-dependent: log-odds gives {run['reading_log_odds']}"
    return v


# ---------------------------------------------------------------- checks

def parity_check(adapter, trials):
    """The job's reruns of P5000's window-shuffled and reversed walks against experiment 31's stored answers."""
    out = {}
    for cond in PARITY:
        new = e31.load_tc(parity_path(adapter, cond), trials["P5000"])
        old = e31.load_tc(e31.tc_path(adapter, "P5000", cond), trials["P5000"])
        out[cond] = e31.agreement(new, old, sorted(trials["P5000"])) if new and old else None
    out["passed"] = all(out[c] is not None and out[c]["share"] >= PARITY_MIN for c in PARITY)
    return out


def exp31_check(run, name):
    """Experiment 31's stored values for O, R, W and V, recomputed here with its resamples: identical."""
    s = json.load(open("results/exp31_summary.json"))["runs"][name]
    a31 = lambda scale, k: s["31a"][scale]["measures"][k]
    pairs = [(run["conditions"][LABELS[k]], s["conditions"][LABELS[k]]) for k in ("O", "R", "W", "V")]
    pairs += [(run["points"]["order_effect"], s["31a"]["order_effect"]),
              (run["points"]["retained_W"], a31("points", "C2_local_share")),
              (run["points"]["reversal_share"], a31("points", "C3_reversal_share")),
              (run["log_odds"]["retained_W"], a31("log_odds", "C2_local_share")),
              (run["log_odds"]["reversal_share"], a31("log_odds", "C3_reversal_share"))]
    bad = [f"{a['mean']} vs {b['mean']}, CI {a['ci95']} vs {b['ci95']}" for a, b in pairs
           if max(abs(x - y) for x, y in zip([a["mean"], *a["ci95"]], [b["mean"], *b["ci95"]])) > 1e-12]
    return {"n": len(pairs), "matched": len(pairs) - len(bad), "mismatches": bad}


def parity(adapter_dir):
    """Job gate on vast-3: both parity reruns >= 98% identical to experiment 31's stored answers, else exit 1."""
    a = os.path.basename(adapter_dir.rstrip("/"))
    p = parity_check(a, {"P5000": e31.manifest("P5000")})
    for cond in PARITY:
        g = p[cond]
        print(f"parity {a} P5000 {cond}: " + (f"{g['agree']}/{g['n']} answers identical ({g['share']:.1%})" if g
                                              else "rerun or stored answers incomplete"))
    if not p["passed"]:
        print("PARITY FAILED: below 98% (or incomplete); nothing else may run")
        raise SystemExit(1)
    print("PARITY PASSED")


def fetch():
    """Only this experiment's answer files from the vast-3 container, never its whole results/."""
    cmd = ('source scripts/vast1/common.sh && '
           f'rsync -a -e "$MI_RSYNC_SHELL" --include="*e4-s[57]-roomP500[0-9]-256px-max64-rev*8.jsonl" --exclude="*" '
           f'"$MI_CONTAINER:$MI_REMOTE_ROOT/{TC}/" {TC}/ && '
           f'mkdir -p {TC}/exp34 && rsync -a -e "$MI_RSYNC_SHELL" --include="*e4-s[57]-*.jsonl" --exclude="*" '
           f'"$MI_CONTAINER:$MI_REMOTE_ROOT/{TC}/exp34/" {TC}/exp34/')
    subprocess.run(["bash", "-c", cmd], check=True, env={**os.environ, "MI_HOST": "vast3"})


# ---------------------------------------------------------------- summary and entry point

KEYS = ("order_effect", "retained_W", "retained_V", "retained_B", "retained_I", "BI", "BW", "IV", "m_within",
        "m_blocks", "reversal_share", "edge_term")
DIFFS = ("blocks reversed - reversed within blocks", "blocks reversed - window-shuffled",
         "reversed within blocks - reversed", "ordered - blocks reversed", "ordered - reversed within blocks")


def short(run):
    """The numbers the paper needs, per run (fractions: 0.10 = 10 points, or a tenth of the order effect)."""
    pick = lambda e: {"mean": e["mean"], "ci95": e["ci95"]}
    s = {"verdict": run["verdict"], "incomplete": run["incomplete"],
         "accuracy": {k: pick(v) for k, v in run["conditions"].items()},
         "parsed": {k: v["parsed"] for k, v in run["conditions"].items()},
         "picked_shown_later": {k: v.get("picked_shown_later") for k, v in run["conditions"].items()}}
    if "points" in run:
        for scale in ("points", "log_odds"):
            s[scale] = {k: pick(run[scale][k]) for k in KEYS}
            s[f"reading_{scale}"], s[f"states_{scale}"] = run[f"reading_{scale}"], run[f"states_{scale}"]
        s["differences_in_points"] = {k: pick(run["contrasts"][k]) for k in DIFFS}
    return s


def report(name, run):
    print(f"{name} ({run['adapter']}); incomplete: {', '.join(run['incomplete']) or 'none'}; verdict: {run['verdict']}")
    for k, v in run["conditions"].items():
        print(f"  {k:24s} {100 * v['mean']:5.1f}% [{100 * v['ci95'][0]:.1f}, {100 * v['ci95'][1]:.1f}]  "
              f"parsed {100 * v['parsed']:.1f}%" + (f", picked the option shown later {100 * v['picked_shown_later']:.0f}%"
                                                     if "picked_shown_later" in v else ""))
    if "points" in run:
        for k in KEYS:
            print(f"  {k:16s} {e31.fmt(run['points'][k], k == 'order_effect')}   log-odds "
                  f"{e31.fmt(run['log_odds'][k], False)}")
        print(f"  states {run['states_points']} (log-odds {run['states_log_odds']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="first copy this experiment's answer files from vast-3")
    ap.add_argument("--smoke", choices=sorted(SMOKE), help="stand-in answer files for B and I; write nothing")
    ap.add_argument("--parity", metavar="ADAPTER", help="job gate: P5000 parity reruns vs stored (exit 1 below 98%%)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.parity:
        return parity(args.parity)
    if args.fetch:
        fetch()
    trials = {r: e31.manifest(r) for r in ROOMS}
    novel = {r: {i: t for i, t in trials[r].items() if not t["familiar"]} for r in ROOMS}
    later = shown_later(novel)
    resamples = exp31_resamples()
    runs, checks = {}, {}
    for name, adapter in RUNS.items():
        runs[name] = analyse_run(adapter, trials, novel, resamples[name], args.smoke, later)
        p = parity_check(adapter, trials)
        runs[name]["verdict"] = verdict(runs[name], p["passed"] or bool(args.smoke))
        checks[name] = {"parity": p}
        if "points" in runs[name]:
            checks[name]["exp31_reproduced"] = exp31_check(runs[name], name)
            checks[name]["identity_error"] = {s: runs[name][s]["identity_error"] for s in ("points", "log_odds")}
    base = {r: runs[r]["verdict"].split(" (")[0].split(";")[0] for r in runs}
    summary = {"verdicts": {**{r: runs[r]["verdict"] for r in runs},
                            "recipe": base["run5"] if base["run5"] == base["run7"] else "mixed"},
               "n_novel": sum(len(v) for v in novel.values()), "threshold_share_of_D": T,
               **{r: short(runs[r]) for r in runs}, "checks": checks}
    for name in runs:
        report(name, runs[name])
    print(json.dumps({"verdicts": summary["verdicts"], "checks": checks}, indent=1))
    if args.smoke:
        print(f"smoke ({args.smoke}): B read from {LABELS[SMOKE[args.smoke]['B']]}'s files, I from "
              f"{LABELS[SMOKE[args.smoke]['I']]}'s; nothing written")
        return
    json.dump({"summary": summary, "runs": runs}, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
