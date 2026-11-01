"""Four-epoch internals experiment: apply the registered decision rules (docs/experiments.md).

    .venv-mac/bin/python scripts/e4_internals_analysis.py --fetch

--fetch first copies results/e4_internals/ from vast-3's container (only that folder; never the whole
results/). If results/e4_internals/task_extras_run0_mps.json is missing, it is computed from run 0's MPS
recording on this machine (data/acts_task/qwen8b+qwen8b-proc, under a minute). Reads:
  results/e4_internals/summary_quadrant.json   analyze.py --quadrant for <model>-walk / -single / -single+ema8
  results/e4_internals/probes_quadrant.json    e4_internals probes
  results/e4_internals/task_probe.json         task_probe.py on the CUDA recordings
  results/e4_internals/task_extras.json        e4_internals task-extras on the CUDA recordings
and the references: run 0's original scores (results/summary_quadrant.json, probes_quadrant.json,
task_probe.json), the path-integrating RNN (grid_rnn_quadrant.json), and each run's sensitivity to a shuffled
walk (seed_scenes_summary.json, two_choice/ft_seeds.json). Writes results/e4_internals_summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
D = "results/e4_internals"
RUNS = [3, 4, 5, 6, 7]
UNIT = {"run 0": "qwen8b-ft", "run 0 (vast-3 rerun)": "qwen8b-ft-rerun", **{f"run {k}": f"qwen8b-ft-e4-s{k}" for k in RUNS}}
TASK = {"run 0": "qwen8b+qwen8b-proc", "run 0 (CUDA rerun)": "qwen8b+qwen8b-proc-cuda",
        **{f"run {k}": f"qwen8b+qwen8b-proc-e4-s{k}" for k in RUNS}}
E4 = [f"run {k}" for k in RUNS]

# registered thresholds
WALK_FEW, SINGLE_FEW, WALK_PLACE = 0.25, 0.06, 0.85
XY_NONE, XY_CODE = 0.10, 0.30
DEC_LAYERS = (18, 22)
NEARER_TOL, RESID_MAX = 0.03, 0.10
PAR_SHARE, PAR_PROBE, PAR_NEAR, PAR_DD = 0.02, 0.10, 0.02, 0.05


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fetch():
    cmd = ('source scripts/vast1/common.sh && mkdir -p results/e4_internals && '
           'rsync -a -e "$MI_RSYNC_SHELL" "$MI_CONTAINER:$MI_REMOTE_ROOT/results/e4_internals/" results/e4_internals/')
    subprocess.run(["bash", "-c", cmd], check=True, env={**os.environ, "MI_HOST": "vast3"})


def lnum(L):
    return int(L[1:])


def share(v, strict=True):
    k = "frac_place_heading_invariant_quadrant" if strict else "frac_place_heading_invariant"
    return v[k] / v["frac_place"] if v.get("frac_place") else float("nan")


def unit_measures(S, P, tag):
    """The paper's fine-tuned-8B unit numbers for one model (make_facts.py's selection rules)."""
    if not S or f"{tag}-walk" not in S:
        return None
    w, s, e = S[f"{tag}-walk"], S.get(f"{tag}-single", {}), S.get(f"{tag}-single+ema8", {})
    mx = lambda c, strict=True: max(share(v, strict) for v in c.values()) if c else None
    arg = lambda c: max(c, key=lambda L: share(c[L])) if c else None
    late = [v["frac_place_heading_invariant_quadrant"] for L, v in w.items() if lnum(L) >= 23]
    o = {"walk_share_max": mx(w), "walk_share_layer": arg(w), "single_share_max": mx(s), "ema8_share_max": mx(e),
         "walk_share_by_layer": {L: share(v) for L, v in w.items()},
         "single_share_by_layer": {L: share(v) for L, v in s.items()},
         "ema8_share_by_layer": {L: share(v) for L, v in e.items()},
         "walk_share_180_max": mx(w, False), "single_share_180_max": mx(s, False),
         "late_frac_invariant_lo": min(late), "late_frac_invariant_hi": max(late),
         "walk_place_frac_by_layer": {L: v["frac_place"] for L, v in w.items()},
         "walk_cross_sector_decoding_best_m": min(v["decode_cross_quadrant"] for v in w.values()),
         "ema8_cross_sector_decoding_best_m": min(v["decode_cross_quadrant"] for v in e.values()) if e else None}
    o["moving_average_at_least_walk"] = o["ema8_share_max"] is not None and o["ema8_share_max"] >= o["walk_share_max"]
    if P:
        for c in ("walk", "shuffled", "only"):
            t = f"{tag}-probes-{c}"
            if t in P and not t.startswith("_"):
                o[f"probe_{c}_cross_best_m"] = min(v["cross_quadrant"] for v in P[t].values())
                o[f"probe_{c}_cross_by_layer"] = {L: v["cross_quadrant"] for L, v in P[t].items()}
        if "probe_only_cross_best_m" in o and "probe_walk_cross_best_m" in o:
            o["probe_walk_gain_m"] = o["probe_only_cross_best_m"] - o["probe_walk_cross_best_m"]
    return o


def task_measures(T, X, name):
    if not T or name not in T:
        return None
    m, beh = T[name], T["_behaviour"][name]
    best = lambda site, ro, k, f=max: f(((v[ro][k], L) for L, v in m[site].items()), key=lambda t: t[0])
    dd_last = {L: v["ridge"]["d_diff_r2"] for L, v in m["last"].items()}
    half = 0.5 * max(dd_last.values())
    dec = min(lnum(L) for L, v in dd_last.items() if v >= half)
    allxy = [(v[ro]["xy_r2"], v[ro]["xy_median_m"], s, L, ro) for s in m for L, v in m[s].items() for ro in v]
    o = {"behaviour_accuracy": beh["accuracy"], "behaviour_accuracy_novel": beh["accuracy_novel"], "n": beh["n"],
         "chance": T["_chance"][name],
         "nearer_last_ridge_best": best("last", "ridge", "a_nearer_acc")[0], "nearer_last_ridge_layer": best("last", "ridge", "a_nearer_acc")[1],
         "nearer_last_ridge_by_layer": {L: v["ridge"]["a_nearer_acc"] for L, v in m["last"].items()},
         "d_diff_last_ridge_by_layer": dd_last, "decision_layer": dec,
         "d_diff_max_question_last_ridge": max(best(s, "ridge", "d_diff_r2")[0] for s in ("question", "last")),
         "d_diff_max_question_last_mlp": max(best(s, "mlp", "d_diff_r2")[0] for s in ("question", "last") if "mlp" in next(iter(m[s].values()))),
         "xy_r2_best": max(allxy)[0], "xy_r2_best_where": list(max(allxy)[2:]),
         "xy_median_best_m": min(allxy, key=lambda t: t[1])[1],
         "sector_within_image_best_m": min(best(s, "ridge", "within_sector_m", min)[0] for s in ("img_end", "img_mean")),
         "sector_cross_image_best_m": min(best(s, ro, "cross_sector_m", min)[0] for s in ("img_end", "img_mean") for ro in ("ridge", "mlp"))}
    o["pipeline_check_passes"] = abs(o["nearer_last_ridge_best"] - o["behaviour_accuracy"]) <= NEARER_TOL
    if X and name in X:
        x = X[name]
        res = [v["residual_ridge_r2"] for L, v in x["last"].items() if 20 <= lnum(L) <= 30]
        o.update({"answer_only_r2": x["answer_only_r2"], "true_sign_r2": x["true_sign_r2"],
                  "residual_after_answer_last_L20_30_lo": min(res), "residual_after_answer_last_L20_30_hi": max(res),
                  "within_room_img_mean_best_m": min(v["within_room_m"] for v in x["img_mean"].values()),
                  "within_room_last_best_m": min(v["within_room_m"] for v in x["last"].values()),
                  "within_room_chance_m": x["within_room_chance_m"], "recording_device": x["device"]})
    return o


def shuffle_sensitivity():
    ss, fs = load("results/seed_scenes_summary.json"), load("results/two_choice/ft_seeds.json")
    out = {}
    fe = ss["10"]["four_epoch_runs"]["novel"]
    for i, k in enumerate(fe["LoRA, walk"]["seeds"]):
        w, s = fe["LoRA, walk"]["values"][i], fe["LoRA, shuffled"]["values"][i]
        out[f"run {k}"] = {"heldout_rooms_ordered_minus_shuffled": w - s, "heldout_rooms_shuffled": s,
                           "roomA_novel_shuffled": fs["A"]["novel"][f"e4-s{k}/shuffled"]["acc"]}
    return out


def rank_corr(a, b):
    """Spearman correlation over the runs both dicts share (ties get average ranks); descriptive only."""
    from scipy.stats import spearmanr
    k = sorted(set(a) & set(b))
    return float(spearmanr([a[x] for x in k], [b[x] for x in k]).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="copy results/e4_internals/ from vast-3 first")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    if not os.path.exists(f"{D}/task_extras_run0_mps.json") and os.path.exists("data/acts_task/qwen8b+qwen8b-proc/done.jsonl"):
        subprocess.run([sys.executable, "-m", "spatialcells.e4_internals", "task-extras", "--models", "qwen8b+qwen8b-proc",
                        "--out", f"{D}/task_extras_run0_mps.json"], check=True)
    S0, P0, T0 = load("results/summary_quadrant.json"), load("results/probes_quadrant.json"), load("results/task_probe.json")
    S1, P1, T1 = load(f"{D}/summary_quadrant.json"), load(f"{D}/probes_quadrant.json"), load(f"{D}/task_probe.json")
    X0, X1 = load(f"{D}/task_extras_run0_mps.json"), load(f"{D}/task_extras.json")
    rnn = load("results/grid_rnn_quadrant.json")

    units = {r: unit_measures(S0 if r == "run 0" else S1, P0 if r == "run 0" else P1, t) for r, t in UNIT.items()}
    task = {r: task_measures(T0 if r == "run 0" else T1, X0 if r == "run 0" else X1, n) for r, n in TASK.items()}
    out = {"units": units, "task": task, "shuffle_sensitivity": shuffle_sensitivity(),
           "references": {"rnn_strict_share": {f"{k} {m}": share(v) for k, d in (rnn or {}).items() for m, v in d.items() if isinstance(v, dict) and "frac_place" in v},
                          "single_frames_all_models": "0-6% (exp. 13e)", "moving_average": "88-97% (exp. 13e-f)"},
           "thresholds": {"walk_few": WALK_FEW, "single_few": SINGLE_FEW, "walk_place_like": WALK_PLACE, "xy_none": XY_NONE,
                          "xy_code": XY_CODE, "decision_layers": DEC_LAYERS, "nearer_tolerance": NEARER_TOL,
                          "residual_max": RESID_MAX, "parity": {"share": PAR_SHARE, "probe_m": PAR_PROBE, "nearer": PAR_NEAR, "d_diff": PAR_DD}}}

    # ---- parity: run 0 re-recorded (vast-3 CUDA) against its original recordings
    par = {}
    u0, ur = units["run 0"], units["run 0 (vast-3 rerun)"]
    if u0 and ur:
        d = max(abs(u0[f"{c}_share_by_layer"][L] - ur[f"{c}_share_by_layer"][L]) for c in ("walk", "single", "ema8") for L in u0[f"{c}_share_by_layer"])
        dp = max((abs(u0[f"probe_{c}_cross_by_layer"][L] - ur[f"probe_{c}_cross_by_layer"][L]) for c in ("walk", "shuffled", "only")
                  for L in u0.get(f"probe_{c}_cross_by_layer", {}) if f"probe_{c}_cross_by_layer" in ur), default=None)
        par["units"] = {"max_share_difference": d, "max_probe_difference_m": dp,
                        "passes": d <= PAR_SHARE and (dp is None or dp <= PAR_PROBE)}
    t0, tr = task["run 0"], task["run 0 (CUDA rerun)"]
    if t0 and tr:
        dn = max(abs(t0["nearer_last_ridge_by_layer"][L] - tr["nearer_last_ridge_by_layer"][L]) for L in t0["nearer_last_ridge_by_layer"] if lnum(L) >= 20)
        dd = max(abs(t0["d_diff_last_ridge_by_layer"][L] - tr["d_diff_last_ridge_by_layer"][L]) for L in t0["d_diff_last_ridge_by_layer"])
        par["task"] = {"max_nearer_difference_L20_on": dn, "max_d_diff_difference": dd,
                       "decision_layer": [t0["decision_layer"], tr["decision_layer"]],
                       "passes": dn <= PAR_NEAR and dd <= PAR_DD and t0["decision_layer"] == tr["decision_layer"]}
    out["parity"] = par

    # ---- decision rules (registered)
    verdict = {}
    ue = {r: units[r] for r in E4 if units[r]}
    if ue:
        W = {r: v["walk_share_max"] for r, v in ue.items()}
        Sg = {r: v["single_share_max"] for r, v in ue.items()}
        if any(w >= WALK_PLACE for w in W.values()):
            v_ = "does not carry over"
        elif len(ue) == len(E4) and all(W[r] <= WALK_FEW and Sg[r] <= SINGLE_FEW for r in ue):
            v_ = "carries over"
        elif len(ue) < len(E4):
            v_ = f"incomplete ({len(ue)} of {len(E4)} runs)"
        else:
            v_ = "partly"
        verdict["units"] = {"verdict": v_, "walk_share_max": W, "single_share_max": Sg,
                            "moving_average_at_least_walk": {r: v["moving_average_at_least_walk"] for r, v in ue.items()}}
        sens = {r: out["shuffle_sensitivity"][r]["heldout_rooms_ordered_minus_shuffled"] for r in ue}
        if len(ue) >= 3:
            verdict["units"]["rank_correlation_walk_share_vs_shuffle_sensitivity_descriptive"] = rank_corr(W, sens)
    te = {r: task[r] for r in E4 if task[r]}
    if te:
        xy = {r: v["xy_r2_best"] for r, v in te.items()}
        if any(x >= XY_CODE for x in xy.values()):
            v1 = "does not carry over"
        elif len(te) == len(E4) and all(x <= XY_NONE for x in xy.values()):
            v1 = "carries over"
        elif len(te) < len(E4):
            v1 = f"incomplete ({len(te)} of {len(E4)} runs)"
        else:
            v1 = "partly"
        dec = {r: v["decision_layer"] for r, v in te.items()}
        v2 = ("carries over" if len(te) == len(E4) and all(DEC_LAYERS[0] <= d <= DEC_LAYERS[1] for d in dec.values())
              else f"incomplete ({len(te)} of {len(E4)} runs)" if len(te) < len(E4) else "differs")
        res = {r: v.get("residual_after_answer_last_L20_30_hi") for r, v in te.items()}
        verdict["task"] = {"pipeline_check": {r: v["pipeline_check_passes"] for r, v in te.items()},
                           "no_cross_room_position_code": {"verdict": v1, "xy_r2_best": xy},
                           "decision_forms_layers_18_22": {"verdict": v2, "decision_layer": dec},
                           "readout_mostly_the_answer": {"holds": all(x is not None and x <= RESID_MAX for x in res.values()),
                                                         "residual_r2_hi": res}}
    out["verdicts"] = verdict
    json.dump(out, open("results/e4_internals_summary.json", "w"), indent=1)

    # ---- print
    pc = lambda x: "  n/a" if x is None else f"{100 * x:5.1f}"
    print("Strictly heading-invariant share of place units, max over the 7 layers (exp. 13e measure):")
    print(f"  {'':22s} {'walk':>6s} {'single':>6s} {'ema8':>6s}  late frac  probe cross walk/shuf/only (m)  shuffle sens.")
    for r, u in units.items():
        if not u:
            print(f"  {r:22s} (not yet scored)")
            continue
        sh = out["shuffle_sensitivity"].get(r, {}).get("heldout_rooms_ordered_minus_shuffled")
        print(f"  {r:22s} {pc(u['walk_share_max'])} {pc(u['single_share_max'])} {pc(u['ema8_share_max'])}  "
              f"{100 * u['late_frac_invariant_lo']:.1f}-{100 * u['late_frac_invariant_hi']:.1f}%  "
              + "/".join(f"{u.get(f'probe_{c}_cross_best_m', float('nan')):.2f}" for c in ("walk", "shuffled", "only"))
              + (f"   {100 * sh:+.1f}" if sh is not None else ""))
    print("Task readout (P5000-P5009, five rooms -> five rooms):")
    for r, t in task.items():
        if not t:
            print(f"  {r:22s} (not yet read out)")
            continue
        print(f"  {r:22s} behaviour {pc(t['behaviour_accuracy'])}  nearer@last {pc(t['nearer_last_ridge_best'])} ({t['nearer_last_ridge_layer']})"
              f"  decision layer {t['decision_layer']}  d_diff max {t['d_diff_max_question_last_ridge']:.2f}  xy R2 best {t['xy_r2_best']:.2f}"
              f" ({t['xy_median_best_m']:.2f} m)" + (f"  residual {t['residual_after_answer_last_L20_30_lo']:.2f}-{t['residual_after_answer_last_L20_30_hi']:.2f}"
                                                     if "answer_only_r2" in t else ""))
    print("Parity:", json.dumps(par))
    print("Verdicts:", json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
