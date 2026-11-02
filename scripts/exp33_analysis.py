"""Experiment 33 (docs/experiments.md): experiment 25's population assay on teleport views shown one at a time, as the
new view of the two-choice prompt, scored on novel views; and on the walk frames of the same prompts.

    # vast-3 CPU, after the recordings (scripts/exp33_analysis.sh)
    python scripts/exp33_analysis.py assay           # per-view errors of every readout -> data/exp33/errors/
    python scripts/exp33_analysis.py summarize       # results/exp33_single_query.json (top-level `summary`)
    # no model, no recording: synthetic populations through the whole pipeline, in a scratch folder
    python scripts/exp33_analysis.py selftest --root <scratch folder>

The readout is population_assay.py's primary assay (guard45), unchanged and imported from it: units z-scored over all
rows of the source, kernel ridge of (x, y) trained on the training walks' rows whose heading is at least 90 deg from
the test sector's centre (unseen) or within 90 deg of it (seen), penalty picked on half of the training walks, every
row tested once by its fold (even / odd walks) and its own sector. errors() below is heldout_decoding()'s loop for
guard45, returning each row's four errors instead of their medians, so that HG can be taken over any test subset of
the same fits: all views (as experiment 25), novel views (the primary), familiar views, views with a valid trial. The
readouts are the same for every subset (the test rows never touch a fit). On all rows, HG and its interval are
bit-identical to population_assay's (`assay` checks this on its first matrix and stops otherwise). Intervals: 1,000
resamples of the 48 walks (seed 0, population_assay's), the same resamples for every source, so differences between
two sources on the same rows (ordered minus shuffled, fine-tuned minus base, residual minus MLP, single-query minus
experiment 25's multi-query recording) get paired intervals.

Sources (each at layers 3, 23, 29, 35):
  sq/<model>/<cond>/<site>/<kind>  the single-query recordings (single_query.py): cond walk, shuffled, nowalk; site
                                   img_end (the view's <|vision_end|>) or last (the last prompt token); kind mlp or resid
  sqw/<model>/<cond>/<kind>        the walk frames of the walk and shuffled prompts (61 per walk, 2,928)
  sqA/<model>/<cond>/<kind>        img_end with room A's 192 existing trials' prompts in place of their views' new trials
  mq/<model>/<cond>                experiment 25's teleport views (extract.py: all 64 after the walk, after the shuffled
                                   walk, or alone; data/acts/<model>-probes-<cond>), MLP only
  mqw/<model>/<cond>               experiment 25's walk frames (data/acts/<model>-walk or -shuffled), the same 2,928 frames
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
import population_assay as pa                                        # noqa: E402  (experiment 25's assay)
from spatialcells import analyze                                     # noqa: E402
from spatialcells import single_query as sq                          # noqa: E402

RUNS = {str(k): f"qwen8b-ft-e4-s{k}" for k in range(3, 8)}
BASE = "qwen8b"
MODELS = list(RUNS.values()) + [BASE]
LAYERS = list(sq.LAYERS)                                             # L3, L23, L29, L35
PRIMARY_LAYER = "L35"
VARIANT = "guard45"
OUT = "results/exp33_single_query.json"
EXP25 = "results/population_assay.json"
AGREE_MIN = 0.98                                                     # answers equal to the stored run's
REPRO_TOL = 0.005                                                    # experiment 25's stored HG, recomputed here
ORDER_D = 0.10                                                       # order readings (ordered minus shuffled)
WALK_ORDER_D = 0.25                                                  # walk frames: experiment 25 amendment 1's rule
QUERY_SUBSETS = ("all", "novel", "familiar", "valid", "novel_valid")


class Paths:
    def __init__(self, root="."):
        self.sq = os.path.join(root, sq.OUT)
        self.acts = os.path.join(root, "data/acts")
        self.err = os.path.join(root, "data/exp33/errors")
        self.out = os.path.join(root, OUT)
        self.trials = sq.TRIALS
        self.exp25 = EXP25


# ---------------------------------------------------------------- inputs
def probe_meta():
    m = dict(np.load("data/walks/probes_meta.npz"))
    return {k: m[k] for k in ("x", "y", "theta", "walk")}


def walkframe_meta():
    m = dict(np.load("data/walks/meta.npz"))
    rows = sq.walkframe_rows()
    return {k: m[k][rows] for k in ("x", "y", "theta", "walk")}, rows


def masks(T):
    nov = np.array([not t["familiar"] for t in T])
    val = np.array([t["valid"] for t in T])
    return {"all": np.ones(len(T), bool), "novel": nov, "familiar": ~nov, "valid": val, "novel_valid": nov & val}


def recorded(P, model, tag, n):
    """Rows done in data/acts_sq/<model>/done_<tag>.jsonl (walk frames: walks), compared with n."""
    f = f"{P.sq}/{model}/done_{tag}.jsonl"
    if not os.path.exists(f):
        return 0
    return len({json.loads(l).get("row", json.loads(l).get("walk")) for l in open(f)})


def sources(P, models):
    """name -> (loader(layer) -> float32 rows x units or None, 'probes' or 'walkframes'), for the data that exist."""
    PT = json.load(open(sq.PARITY_TRIALS))
    _, wrows = walkframe_meta()
    out = {}

    def sq_arr(model, tag, site, kind, n, done_n, done_tag=None):
        f = f"{P.sq}/{model}/{tag}_{site}_{kind}.npy"
        if not os.path.exists(f) or recorded(P, model, done_tag or tag, n) < done_n:
            return None
        lay = json.load(open(f"{P.sq}/{model}/meta.json"))["layers"]
        return lambda layer: np.asarray(np.load(f, mmap_mode="r")[:, lay.index(layer), :], dtype=np.float32)

    for m in models:
        for cond in sq.CONDS:
            for site in sq.SITES:
                for kind in sq.KINDS:
                    ld = sq_arr(m, cond, site, kind, 3072, 3072)
                    if ld:
                        out[f"sq/{m}/{cond}/{site}/{kind}"] = (ld, "probes")
            for kind in sq.KINDS:
                base = sq_arr(m, cond, "img_end", kind, 3072, 3072)
                par = sq_arr(m, f"parity_{cond}", "img_end", kind, 192, 192)
                if base and par:
                    def ldA(layer, base=base, par=par):
                        X = base(layer)
                        X[[t["probe"] for t in PT]] = par(layer)
                        return X
                    out[f"sqA/{m}/{cond}/{kind}"] = (ldA, "probes")
        for cond in ("walk", "shuffled"):
            for kind in sq.KINDS:
                ld = sq_arr(m, cond, "walkframes", kind, 48, 48, f"walkframes_{cond}")
                if ld:
                    out[f"sqw/{m}/{cond}/{kind}"] = (ld, "walkframes")
        for cond in ("walk", "shuffled", "only"):
            d = f"{P.acts}/{m}-probes-{cond}"
            if all(os.path.exists(f"{d}/{l}.npy") for l in LAYERS):
                out[f"mq/{m}/{cond}"] = (lambda layer, d=d: np.load(f"{d}/{layer}.npy").astype(np.float32), "probes")
        for cond in ("walk", "shuffled"):
            d = f"{P.acts}/{m}-{cond}"
            if all(os.path.exists(f"{d}/{l}.npy") for l in LAYERS):
                out[f"mqw/{m}/{cond}"] = (lambda layer, d=d: np.asarray(np.load(f"{d}/{layer}.npy", mmap_mode="r")[wrows],
                                                                        dtype=np.float32), "walkframes")
    return out


# ---------------------------------------------------------------- the readout (population_assay's, per row)
def errors(Z, meta, variant=VARIANT):
    """population_assay.heldout_decoding's loop for one variant: each row's four errors, and the penalties."""
    G = pa.gram(Z)
    Y = torch.tensor(np.stack([meta["x"], meta["y"]], 1), dtype=torch.float64)
    walk, th = meta["walk"], meta["theta"]
    sec = analyze.sectors(th)
    N = len(walk)
    ho_deg, seen_deg = pa.VARIANTS[variant]
    err = {k: np.full(N, np.nan) for k in ("ho", "seen", "chance_ho", "chance_seen")}
    lams = {"ho": [], "seen": []}
    for par in (0, 1):
        trw, tew = walk % 2 == par, walk % 2 == 1 - par
        a_w, b_w = walk % 4 == par, walk % 4 == par + 2
        for s in range(4):
            d = np.abs(pa.wrap(th - pa.CENTERS[s]))
            te_m = tew & (sec == s)
            te = torch.from_numpy(np.flatnonzero(te_m))
            for kind, arc in (("ho", d >= np.deg2rad(ho_deg)), ("seen", d < np.deg2rad(seen_deg))):
                t = lambda m: torch.from_numpy(np.flatnonzero(m))
                pred, ym, lam = pa.ridge(G, Y, t(trw & arc), te, t(a_w & arc), t(b_w & arc))
                err[kind][te_m] = (pred - Y[te]).norm(dim=1).numpy()
                err[f"chance_{kind}"][te_m] = (ym[None] - Y[te]).norm(dim=1).numpy()
                lams[kind].append(float(lam))
    del G
    return err, lams


def resamples(n_walks=48):
    rng = np.random.default_rng(0)                                    # population_assay.summarize_errors' resamples
    return [rng.integers(0, n_walks, n_walks) for _ in range(pa.N_BOOT)]


def stats(err, walk, sel, R):
    """HG and its parts over the rows in sel, its interval over walk resamples, and the resampled HGs (for pairs)."""
    e = {k: v[sel] for k, v in err.items()}
    m, gain_ho, gain_seen, hg = pa.hg_from(e)
    idx = [np.flatnonzero((walk == w) & sel) for w in range(48)]
    hgs = []
    for r in R:
        j = np.concatenate([idx[w] for w in r])
        hgs.append(pa.hg_from({k: v[j] for k, v in err.items()})[3])
    hgs = np.array(hgs)
    res = {"HG": hg, "HG_ci95": [float(np.nanpercentile(hgs, 2.5)), float(np.nanpercentile(hgs, 97.5))],
           "err_ho_m": m["ho"], "err_seen_m": m["seen"], "chance_ho_m": m["chance_ho"], "chance_seen_m": m["chance_seen"],
           "gain_ho_m": gain_ho, "gain_seen_m": gain_seen, "n": int(sel.sum())}
    res["call"] = pa.call(res)
    return res, hgs


def paired(a, b):
    """A minus B with its interval from the same walk resamples."""
    (ra, ha), (rb, hb) = a, b
    d = ha - hb
    return {"D": ra["HG"] - rb["HG"], "D_ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}


# ---------------------------------------------------------------- assay mode
def run_assay(P, models, layers, threads):
    torch.set_num_threads(threads)
    pmeta = probe_meta()
    wmeta, _ = walkframe_meta()
    os.makedirs(P.err, exist_ok=True)
    src = sources(P, models)
    print(f"{len(src)} sources: " + ", ".join(sorted(src)), flush=True)
    parity_file = f"{P.err}/_parity.json"
    for name, (load, kind) in sorted(src.items()):
        meta = pmeta if kind == "probes" else wmeta
        for layer in layers:
            f = f"{P.err}/{name.replace('/', '__')}__{layer}.npz"
            if os.path.exists(f):
                continue
            t0 = time.time()
            Z = pa.standardize(torch.from_numpy(load(layer)))
            err, lams = errors(Z, meta)
            if not os.path.exists(parity_file):                     # this path against population_assay's own
                ref = pa.heldout_decoding(Z, meta, {VARIANT: pa.VARIANTS[VARIANT]})[VARIANT]
                mine = pa.summarize_errors(err, meta["walk"], analyze.sectors(meta["theta"]))
                keys = ("HG", "HG_ci95", "err_ho_m", "err_seen_m", "chance_ho_m", "chance_seen_m", "call")
                same = all(ref[k] == mine[k] for k in keys)
                R = resamples()
                own, _ = stats(err, meta["walk"], np.ones(len(meta["walk"]), bool), R)
                same_stats = own["HG"] == ref["HG"] and own["HG_ci95"] == ref["HG_ci95"]
                json.dump({"source": name, "layer": layer, "identical": bool(same and same_stats),
                           "population_assay": {k: ref[k] for k in keys}, "this": {k: mine[k] for k in keys},
                           "stats_HG_ci95": own["HG_ci95"]}, open(parity_file, "w"), indent=1)
                print(f"assay parity on {name} {layer}: HG {ref['HG']:.4f} [{ref['HG_ci95'][0]:.4f}, "
                      f"{ref['HG_ci95'][1]:.4f}] both ways, identical {same and same_stats}", flush=True)
                if not (same and same_stats):
                    raise SystemExit("this script's per-row path differs from population_assay.heldout_decoding")
            np.savez(f, **err, lam_ho=np.array(lams["ho"]), lam_seen=np.array(lams["seen"]), units=Z.shape[1])
            del Z
            print(f"{name} {layer}: {len(err['ho'])} rows, {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------- summary
def load_err(P, name, layer):
    f = f"{P.err}/{name.replace('/', '__')}__{layer}.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return {k: z[k] for k in ("ho", "seen", "chance_ho", "chance_seen")}


def readings_primary(calls):
    """Registered rule over runs 3-7: holds if every run is partial or better, does not hold if every run is below
    partial (negative or no position code), mixed otherwise."""
    if len(calls) < len(RUNS):
        return "incomplete"
    ok = [c in ("partial", "positive") for c in calls.values()]
    return "holds" if all(ok) else ("does not hold" if not any(ok) else "mixed")


def combined(mlp, resid, calls_mlp, calls_resid):
    """The registered joint reading of the two primary verdicts."""
    if "incomplete" in (mlp, resid):
        return "incomplete"
    if all(c == "positive" for c in calls_resid.values()) and all(c != "positive" for c in calls_mlp.values()):
        return "residual positive, MLP partial or below"
    return {("holds", "holds"): "holds in both", ("holds", "does not hold"): "holds in the MLP only",
            ("does not hold", "holds"): "holds in the residual stream only",
            ("does not hold", "does not hold"): "holds in neither"}.get((mlp, resid), "mixed")


def order_reading(ds, d_min=ORDER_D):
    if len(ds) < len(RUNS):
        return "incomplete"
    if all(x["D"] >= d_min and x["D_ci95"][0] > 0 for x in ds.values()):
        return "needs order"
    if all(x["D"] < d_min for x in ds.values()):
        return "little order dependence"
    return "mixed"


def behavior(P, model, T, PT):
    out = {}
    nov = {t["id"]: not t["familiar"] for t in T}
    val = {t["id"]: t["valid"] for t in T}
    favours_far = {t["id"]: not t["frame_order_favours_near"] for t in T}
    for cond in sq.CONDS:
        f = f"{P.sq}/{model}/done_{cond}.jsonl"
        if not os.path.exists(f):
            continue
        R = [r for r in map(json.loads, open(f)) if val[r["id"]]]
        o = {}
        for cat, sel in (("all", R), ("novel", [r for r in R if nov[r["id"]]]),
                         ("familiar", [r for r in R if not nov[r["id"]]]),
                         ("frame_order_favours_far", [r for r in R if favours_far[r["id"]]])):
            if sel:
                o[cat] = {"n": len(sel), "accuracy": float(np.mean([r["correct"] for r in sel])),
                          "parsed": float(np.mean([r["answer"] > 0 for r in sel]))}
        out[cond] = o
        g = f"{P.sq}/{model}/done_parity_{cond}.jsonl"
        if os.path.exists(g):
            Q = list(map(json.loads, open(g)))
            ref, path = sq.stored(cond, sq.model_adapter(model))
            withref = [r for r in Q if r["id"] in ref]
            pnov = {t["id"]: not t["familiar"] for t in PT}
            out[f"parity_{cond}"] = {
                "n": len(Q), "accuracy": float(np.mean([r["correct"] for r in Q])) if Q else None,
                "accuracy_novel": float(np.mean([r["correct"] for r in Q if pnov[r["id"]]])) if Q else None,
                "stored_run": os.path.basename(path) if ref else None, "n_compared": len(withref),
                "agreement": float(np.mean([r["answer"] == ref[r["id"]]["answer"] for r in withref])) if withref else None,
                "token_count_mismatches": sum(r["prompt_tokens"] != ref[r["id"]]["prompt_tokens"] for r in withref)}
    return out


def gates(P, models, res, PT):
    g = {"complete": {}, "answer_agreement": {}, "token_count_mismatches": 0, "cache_checks": None,
         "assay_parity_identical": None, "exp25_reproduction_max_abs_diff": None, "nonfinite_or_clipped": {}}
    for m in models:
        g["complete"][m] = {c: recorded(P, m, c, 3072) == 3072 for c in sq.CONDS}
        g["complete"][m].update({f"walkframes_{c}": recorded(P, m, f"walkframes_{c}", 48) == 48 for c in ("walk", "shuffled")})
        g["complete"][m].update({f"parity_{c}": recorded(P, m, f"parity_{c}", 192) == 192 for c in sq.CONDS})
        b = res["behavior"].get(m, {})
        for c in sq.CONDS:
            p = b.get(f"parity_{c}")
            if p and p["n_compared"]:
                g["answer_agreement"].setdefault(m, {})[c] = {"agreement": p["agreement"], "n": p["n_compared"],
                                                              "meets": p["agreement"] >= AGREE_MIN}
                g["token_count_mismatches"] += p["token_count_mismatches"]
        nb = 0
        for f in glob.glob(f"{P.sq}/{m}/done_*.jsonl"):
            nb += sum(json.loads(l).get("nonfinite_or_clipped", 0) for l in open(f))
        g["nonfinite_or_clipped"][m] = nb
    ch = [json.loads(l) for m in models for l in (open(f"{P.sq}/{m}/checks.jsonl") if os.path.exists(f"{P.sq}/{m}/checks.jsonl") else [])]
    if ch:
        g["cache_checks"] = {"n": len(ch), "min_cos_view": min(c["min_cos_view"] for c in ch),
                             "min_cos_walkframes": min(c["min_cos_walkframes"] for c in ch),
                             "all_logits_within_tolerance": all(c["max_dlogit"] < 0.05 * c["logit_scale"]
                                                                and c["gap"] <= 2 * c["max_dlogit"] for c in ch)}
    pf = f"{P.err}/_parity.json"
    if os.path.exists(pf):
        g["assay_parity_identical"] = json.load(open(pf))["identical"]
    if os.path.exists(P.exp25):
        ref = json.load(open(P.exp25))
        diffs = {}
        for m in models:
            for cond in ("walk", "shuffled", "only"):
                r25 = ref.get("conditions_e4", {}).get(f"{m}-probes-{cond}") or ref.get("conditions", {}).get(f"{m}-probes-{cond}")
                mine = res["conditions"].get(f"mq/{m}/{cond}", {})
                for l in LAYERS:
                    if r25 and l in r25 and l in mine:
                        diffs[f"{m}-probes-{cond}:{l}"] = abs(r25[l][VARIANT]["HG"] - mine[l]["all"]["HG"])
        if diffs:
            g["exp25_reproduction_max_abs_diff"] = max(diffs.values())
            g["exp25_reproduction_n"] = len(diffs)
            g["exp25_reproduction_meets"] = max(diffs.values()) <= REPRO_TOL
    ag = [x["meets"] for v in g["answer_agreement"].values() for x in v.values()]
    walk_ag = [g["answer_agreement"].get(m, {}).get(c, {}).get("meets") for m in RUNS.values() for c in ("walk", "shuffled")]
    g["pipeline_ok"] = bool(all(all(v.values()) for v in g["complete"].values()) and g["token_count_mismatches"] == 0
                            and all(ag) and all(x is True for x in walk_ag)
                            and (g["cache_checks"] or {}).get("min_cos_view", 0) >= sq.COS_MIN
                            and (g["cache_checks"] or {}).get("min_cos_walkframes", 0) >= sq.COS_MIN
                            and (g["cache_checks"] or {}).get("all_logits_within_tolerance", False)
                            and g["assay_parity_identical"] is True
                            and g.get("exp25_reproduction_meets") is not False)
    return g


def summarize(P, models):
    T = json.load(open(P.trials))
    PT = json.load(open(sq.PARITY_TRIALS))
    pmeta = probe_meta()
    wmeta, _ = walkframe_meta()
    M = masks(T)
    R = resamples()
    res = json.load(open(P.out)) if os.path.exists(P.out) else {}
    res["_about"] = {"experiment": "33 (docs/experiments.md)", "variant": VARIANT, "layers": LAYERS,
                     "HG_pos": pa.HG_POS, "HG_neg": pa.HG_NEG, "min_code_m": pa.MIN_CODE_M, "n_boot": pa.N_BOOT}
    res["trials"] = sq.trials_report(T)
    names = sorted({os.path.basename(f).rsplit("__", 1)[0].replace("__", "/") for f in glob.glob(f"{P.err}/*__L*.npz")})
    cond_stats, boots = {}, {}
    for name in names:
        kind = "walkframes" if name.split("/")[0] in ("sqw", "mqw") else "probes"
        walk = (wmeta if kind == "walkframes" else pmeta)["walk"]
        subsets = {"all": np.ones(len(walk), bool)} if kind == "walkframes" else M
        for layer in LAYERS:
            err = load_err(P, name, layer)
            if err is None:
                continue
            for sub, sel in subsets.items():
                st, hgs = stats(err, walk, sel, R)
                cond_stats.setdefault(name, {}).setdefault(layer, {})[sub] = st
                boots[name, layer, sub] = (st, hgs)
    res["conditions"] = cond_stats
    res["behavior"] = {m: behavior(P, m, T, PT) for m in models}
    res["gates"] = gates(P, models, res, PT)

    def get(name, layer=PRIMARY_LAYER, sub="novel"):
        return boots.get((name, layer, sub))

    def brief(x):
        r = x[0]
        return {k: r[k] for k in ("HG", "HG_ci95", "call", "err_ho_m", "err_seen_m", "chance_ho_m", "gain_seen_m", "n")}

    def best(name, sub):
        """The highest HG among layers whose seen-heading readout beats chance by MIN_CODE_M (experiment 25)."""
        rows = [(l, cond_stats[name][l][sub]) for l in LAYERS if l in cond_stats.get(name, {})]
        coded = [(l, r) for l, r in rows if r["gain_seen_m"] >= pa.MIN_CODE_M and np.isfinite(r["HG"])]
        if not coded:
            return {"best_layer": None, "call": "no position code"} if rows else None
        l, r = max(coded, key=lambda lr: lr[1]["HG"])
        return {"best_layer": l, "HG": r["HG"], "HG_ci95": r["HG_ci95"], "call": r["call"]}

    s = {"_about": "Experiment 33. HG = experiment 25's heading-held-out gain (guard45), read from the same readouts on "
                   "a test subset of rows. Primary: the view's image-end token, single-query prompt with the ordered "
                   "walk, novel views, layer 35, runs 3-7, for MLP neurons and the residual stream separately. Calls: "
                   "positive >= 0.5, partial 0.25-0.5, negative < 0.25, no position code if the seen-heading readout "
                   "beats chance by < 0.5 m. Differences are A minus B with paired walk-bootstrap intervals.",
         "gates": res["gates"], "trials": res["trials"]}
    cal = json.load(open(CAL)) if os.path.exists(CAL) else None
    s["calibration"] = None if cal is None else {"thresholds": cal["thresholds"], "curve": cal["curve"], "seeds": cal["seeds"]}
    eq = (lambda hg, sample: pa.equivalent_p(hg, list(zip(cal["curve"][sample]["p"], cal["curve"][sample]["HG_mean"])))) \
        if cal else (lambda hg, sample: None)
    for kind in sq.KINDS:
        runs = {k: brief(get(f"sq/{m}/walk/img_end/{kind}")) for k, m in RUNS.items() if get(f"sq/{m}/walk/img_end/{kind}")}
        for k, v in runs.items():
            a = get(f"sq/{RUNS[k]}/walk/img_end/{kind}", sub="all")
            v["HG_all"] = a[0]["HG"] if a else None
            if cal:
                v["call_calibrated"] = calibrated_call(v, cal["thresholds"])
                v["equivalent_invariant_share_novel"] = eq(v["HG"], "probes:novel") if v["call"] != "no position code" else None
                v["equivalent_invariant_share_all"] = eq(v["HG_all"], "probes:all") if a and a[0]["call"] != "no position code" else None
        calls = {k: v["call"] for k, v in runs.items()}
        hg = [v["HG"] for v in runs.values()]
        mean = None
        if len(runs) == len(RUNS):                                   # five-run mean, same walk resamples
            b = np.mean([get(f"sq/{m}/walk/img_end/{kind}")[1] for m in RUNS.values()], 0)
            mean = {"HG": float(np.mean(hg)), "HG_ci95": [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]}
        s[f"primary_{kind}"] = {"runs": runs, "verdict": readings_primary(calls),
                                "verdict_calibrated": readings_primary({k: v.get("call_calibrated") for k, v in runs.items()})
                                if cal else None,
                                "HG_range": [min(hg), max(hg)] if hg else None, "five_run_mean": mean}
        bl = {k: best(f"sq/{m}/walk/img_end/{kind}", "novel") for k, m in RUNS.items() if f"sq/{m}/walk/img_end/{kind}" in cond_stats}
        s[f"primary_best_layer_{kind}"] = {"runs": bl, "verdict": readings_primary({k: v["call"] for k, v in bl.items() if v})}
        s[f"base_{kind}"] = {c: {sub: brief(get(f"sq/{BASE}/{c}/img_end/{kind}", sub=sub))
                                 for sub in ("novel", "all") if get(f"sq/{BASE}/{c}/img_end/{kind}", sub=sub)}
                             for c in sq.CONDS}
        for label, other in (("order", "shuffled"), ("nowalk", "nowalk")):
            d = {}
            for k, m in RUNS.items():
                a, b = get(f"sq/{m}/walk/img_end/{kind}"), get(f"sq/{m}/{other}/img_end/{kind}")
                if a and b:
                    d[k] = {**paired(a, b), f"HG_{other}": b[0]["HG"], f"call_{other}": b[0]["call"],
                            f"HG_{other}_ci95": b[0]["HG_ci95"]}
            s[f"{label}_{kind}"] = {"runs": d}
        s[f"order_{kind}"]["reading"] = order_reading(s[f"order_{kind}"]["runs"])
        tr = {}
        for k, m in RUNS.items():
            a, b = get(f"sq/{m}/walk/img_end/{kind}"), get(f"sq/{BASE}/walk/img_end/{kind}")
            if a and b:
                tr[k] = paired(a, b)
        s[f"training_{kind}"] = {"runs": tr}
        s[f"subsets_{kind}"] = {k: {sub: get(f"sq/{m}/walk/img_end/{kind}", sub=sub)[0]["HG"]
                                    for sub in QUERY_SUBSETS if get(f"sq/{m}/walk/img_end/{kind}", sub=sub)}
                                for k, m in RUNS.items()}
        nf = {}
        for k, m in RUNS.items():
            a, b = get(f"sq/{m}/walk/img_end/{kind}"), get(f"sq/{m}/walk/img_end/{kind}", sub="familiar")
            if a and b:
                nf[k] = paired(a, b)
        s[f"novel_minus_familiar_{kind}"] = {"runs": nf}
        s[f"last_token_{kind}"] = {k: {l: {sub: cond_stats[f"sq/{m}/walk/last/{kind}"][l][sub]["HG"]
                                          for sub in ("novel_valid", "valid")}
                                      for l in LAYERS if l in cond_stats.get(f"sq/{m}/walk/last/{kind}", {})}
                                   for k, m in {**RUNS, "base": BASE}.items()}
        s[f"option_a_{kind}"] = {k: get(f"sqA/{m}/walk/{kind}")[0]["HG"] for k, m in RUNS.items() if get(f"sqA/{m}/walk/{kind}")}
        s[f"by_layer_{kind}"] = {k: {c: {sub: {l: cond_stats[f"sq/{m}/{c}/img_end/{kind}"][l][sub]["HG"] for l in LAYERS
                                               if l in cond_stats.get(f"sq/{m}/{c}/img_end/{kind}", {})}
                                         for sub in ("novel", "all")} for c in sq.CONDS}
                                 for k, m in {**RUNS, "base": BASE}.items()}
        # walk frames of the task prompt
        wf = {}
        for k, m in {**RUNS, "base": BASE}.items():
            a, b = get(f"sqw/{m}/walk/{kind}", sub="all"), get(f"sqw/{m}/shuffled/{kind}", sub="all")
            if a:
                wf[k] = {"walk": brief(a), "best_layer_walk": best(f"sqw/{m}/walk/{kind}", "all")}
                if b:
                    wf[k].update(shuffled=brief(b), order=paired(a, b))
                c = get(f"mqw/{m}/walk", sub="all") if kind == "mlp" else None
                if c:
                    wf[k]["exp25_same_frames_walk"] = brief(c)
                    wf[k]["task_minus_exp25"] = paired(a, c)
        wruns = {k: v for k, v in wf.items() if k in RUNS}
        s[f"walkframes_{kind}"] = {"runs": wf, "reading_walk": (
            "incomplete" if len(wruns) < len(RUNS) else
            "positive in every run" if all(v["walk"]["call"] == "positive" for v in wruns.values()) else
            "below positive in every run" if all(v["walk"]["call"] != "positive" for v in wruns.values()) else "mixed"),
            "reading_order": (
            "incomplete" if len(wruns) < len(RUNS) or any("order" not in v for v in wruns.values()) else
            "depends on order" if all(v["order"]["D"] >= WALK_ORDER_D and v["shuffled"]["call"] != "positive"
                                      for v in wruns.values()) else
            "does not depend on order" if all(v["order"]["D"] < ORDER_D and v["shuffled"]["call"] == "positive"
                                              for v in wruns.values()) else "mixed")}
    s["joint_reading"] = combined(s["primary_mlp"]["verdict"], s["primary_resid"]["verdict"],
                                  {k: v["call"] for k, v in s["primary_mlp"]["runs"].items()},
                                  {k: v["call"] for k, v in s["primary_resid"]["runs"].items()})
    kd = {}
    for k, m in RUNS.items():
        a, b = get(f"sq/{m}/walk/img_end/resid"), get(f"sq/{m}/walk/img_end/mlp")
        if a and b:
            kd[k] = paired(a, b)
    s["resid_minus_mlp"] = {"runs": kd}
    # protocol: single query against experiment 25's multi-query recording of the same views (MLP)
    pr = {}
    for k, m in {**RUNS, "base": BASE}.items():
        row = {}
        for c_sq, c_mq in (("walk", "walk"), ("shuffled", "shuffled"), ("nowalk", "only")):
            for sub in ("all", "novel"):
                a, b = get(f"sq/{m}/{c_sq}/img_end/mlp", sub=sub), get(f"mq/{m}/{c_mq}", sub=sub)
                if a:
                    row[f"single_{c_sq}_{sub}"] = a[0]["HG"]
                if b:
                    row[f"multi_{c_mq}_{sub}"] = b[0]["HG"]
                    row[f"multi_{c_mq}_{sub}_ci95"] = b[0]["HG_ci95"]
                if a and b:
                    row[f"single_minus_multi_{c_sq}_{sub}"] = paired(a, b)
        pr[k] = row
    s["protocol_mlp"] = {"runs": pr}
    s["behavior"] = res["behavior"]
    s["joint_reading_calibrated"] = combined(
        s["primary_mlp"]["verdict_calibrated"] or "incomplete", s["primary_resid"]["verdict_calibrated"] or "incomplete",
        {k: v.get("call_calibrated") for k, v in s["primary_mlp"]["runs"].items()},
        {k: v.get("call_calibrated") for k, v in s["primary_resid"]["runs"].items()})
    res["summary"] = s
    os.makedirs(os.path.dirname(P.out), exist_ok=True)
    pa.save(res, P.out)
    print(f"wrote {P.out}: pipeline_ok {res['gates']['pipeline_ok']}; primary MLP {s['primary_mlp']['verdict']} "
          f"(calibrated {s['primary_mlp']['verdict_calibrated']}) {s['primary_mlp']['HG_range']}, residual "
          f"{s['primary_resid']['verdict']} (calibrated {s['primary_resid']['verdict_calibrated']}) "
          f"{s['primary_resid']['HG_range']}; joint: {s['joint_reading']} (calibrated {s['joint_reading_calibrated']})",
          flush=True)
    return res


# ---------------------------------------------------------------- calibration (synthetic only)
CAL = "results/exp33_calibration.json"
CAL_SEEDS = (2400, 2401, 2402, 2403, 2404)                           # 2400 = experiment 25's validation seed
CAL_K = 3.0


def calibrate(out=CAL):
    """What the novel subset does to HG for a population whose code is known: experiment 25's synthetic mixtures
    (population_assay.population: a share p of heading-invariant place units, the rest conjunctive, view and
    head-direction units; 2,000 units; noise k = 3 times each unit's SD, independent per row, since every view is its
    own prompt), built on the teleport views' poses and on the 2,928 walk frames, scored with the same readout, five
    seeds. Gives, per p, HG on all views, novel views and walk frames, and the novel-view thresholds that match
    experiment 25's 0.25 and 0.5 on all views."""
    T = json.load(open(sq.TRIALS))
    M = masks(T)
    R = resamples()
    pmeta = probe_meta()
    wmeta, _ = walkframe_meta()
    rows = []
    for seed in CAL_SEEDS:
        for p in pa.SYN_P:
            for name, meta in (("probes", pmeta), ("walkframes", wmeta)):
                rng = np.random.default_rng([seed, int(p * 1000), 0])
                S = pa.population(meta, p, rng)
                eps = np.random.default_rng([seed, int(p * 1000), 99]).standard_normal(S.shape)
                X = torch.from_numpy((S + CAL_K * S.std(0) * eps).astype(np.float32))
                err, _ = errors(pa.standardize(X), meta)
                subs = {"all": M["all"], "novel": M["novel"]} if name == "probes" else {"all": np.ones(len(S), bool)}
                for sub, sel in subs.items():
                    st, _ = stats(err, meta["walk"], sel, R)
                    rows.append({"seed": seed, "p": p, "sample": f"{name}:{sub}", **{k: st[k] for k in
                                 ("HG", "HG_ci95", "err_ho_m", "err_seen_m", "chance_ho_m", "chance_seen_m", "call")}})
                    print(f"seed {seed} p={p:g} {name}:{sub}: HG {st['HG']:.3f} [{st['HG_ci95'][0]:.2f}, "
                          f"{st['HG_ci95'][1]:.2f}] seen {st['err_seen_m']:.2f} m, unseen {st['err_ho_m']:.2f} m, chance "
                          f"{st['chance_seen_m']:.2f} m", flush=True)
    curve = {}
    for sample in ("probes:all", "probes:novel", "walkframes:all"):
        h = np.array([[r["HG"] for r in rows if r["sample"] == sample and r["p"] == p] for p in pa.SYN_P])
        curve[sample] = {"p": list(pa.SYN_P), "HG_mean": h.mean(1).tolist(), "HG_sd_over_seeds": h.std(1).tolist()}
    pts = lambda s: list(zip(pa.SYN_P, curve[s]["HG_mean"]))
    at = lambda s, p: float(np.interp(p, pa.SYN_P, curve[s]["HG_mean"]))
    th = {}
    for t in (pa.HG_NEG, pa.HG_POS):
        p_t = pa.equivalent_p(t, pts("probes:all"))
        th[f"{t:g}"] = {"p_all": p_t, "HG_novel": at("probes:novel", p_t), "HG_walkframes": at("walkframes:all", p_t)}
    res = {"_about": calibrate.__doc__, "seeds": list(CAL_SEEDS), "k": CAL_K, "units": pa.SYN_U, "rows": rows,
           "curve": curve, "thresholds": th,
           "inputs_md5": {f: hashlib.md5(open(f, "rb").read()).hexdigest() for f in
                          ("scripts/exp33_analysis.py", "scripts/population_assay.py", "scripts/heading_stat_validation.py",
                           "data/walks/meta.npz", "data/walks/probes_meta.npz", sq.TRIALS)},
           "time": time.strftime("%Y-%m-%d %H:%M %Z")}
    pa.save(res, out)
    print(json.dumps({"curve": curve, "thresholds": th}, indent=1))
    return res


def calibrated_call(st, th):
    """A novel-view result against the novel-view thresholds that match 0.25 and 0.5 on all views (the calibration);
    no position code stays no position code."""
    if st["call"] == "no position code" or not np.isfinite(st["HG"]):
        return st["call"]
    hg = st["HG"]
    return "positive" if hg >= th["0.5"]["HG_novel"] else ("partial" if hg >= th["0.25"]["HG_novel"] else "negative")


# ---------------------------------------------------------------- selftest (no model, no recording)
def selftest(args):
    """Synthetic place-and-view populations (population_assay.population) written in single_query.py's layout,
    with fake done files, through assay and summarize in a scratch folder."""
    root = os.path.abspath(args.root)
    P = Paths(root)
    P.exp25 = os.path.join(root, "none.json")
    T = json.load(open(sq.TRIALS))
    PT = json.load(open(sq.PARITY_TRIALS))
    pmeta = probe_meta()
    wmeta, wrows = walkframe_meta()
    models = args.models or [RUNS["5"], BASE]
    rng = np.random.default_rng(7)
    U = {"mlp": 96, "resid": 64}

    def pop(meta, p, n):
        X = pa.population(meta, p, rng)[:, rng.choice(pa.SYN_U, n, replace=False)]
        return X + 3 * X.std(0) * rng.standard_normal(X.shape)

    for i, m in enumerate(models):
        d = f"{P.sq}/{m}"
        os.makedirs(d, exist_ok=True)
        json.dump({"layers": LAYERS}, open(f"{d}/meta.json", "w"))
        for cond, p in (("walk", 0.4 - 0.3 * (m == BASE)), ("shuffled", 0.2), ("nowalk", 0.1)):
            for site in sq.SITES:
                for kind in sq.KINDS:
                    A = np.stack([pop(pmeta, p, U[kind]) for _ in LAYERS], 1).astype(np.float16)
                    np.save(f"{d}/{cond}_{site}_{kind}.npy", A)
                    np.save(f"{d}/parity_{cond}_{site}_{kind}.npy", A[[t["probe"] for t in PT]])
            with open(f"{d}/done_{cond}.jsonl", "w") as fh:
                for r, t in enumerate(T):
                    a = t["near"] if rng.random() < 0.8 else t["far"]
                    fh.write(json.dumps({"row": r, "id": t["id"], "answer": a, "correct": a == t["near"]}) + "\n")
            ref, _ = sq.stored(cond, sq.model_adapter(m))
            with open(f"{d}/done_parity_{cond}.jsonl", "w") as fh:
                for r, t in enumerate(PT):
                    a = ref[t["id"]]["answer"] if t["id"] in ref else t["near"]
                    fh.write(json.dumps({"row": r, "id": t["id"], "answer": a, "correct": a == t["near"],
                                         "prompt_tokens": ref[t["id"]]["prompt_tokens"] if t["id"] in ref else 0}) + "\n")
            if cond != "nowalk":
                for kind in sq.KINDS:
                    np.save(f"{d}/{cond}_walkframes_{kind}.npy",
                            np.stack([pop(wmeta, 0.6 if cond == "walk" else 0.2, U[kind]) for _ in LAYERS], 1).astype(np.float16))
                with open(f"{d}/done_walkframes_{cond}.jsonl", "w") as fh:
                    for w in range(48):
                        fh.write(json.dumps({"walk": w}) + "\n")
        with open(f"{d}/checks.jsonl", "w") as fh:
            fh.write(json.dumps({"min_cos_view": 0.9999, "min_cos_walkframes": 0.9999, "max_dlogit": 0.01,
                                 "logit_scale": 30.0, "gap": 0.0}) + "\n")
        for cond in ("walk", "shuffled", "only"):
            os.makedirs(f"{P.acts}/{m}-probes-{cond}", exist_ok=True)
            for l in LAYERS:
                np.save(f"{P.acts}/{m}-probes-{cond}/{l}.npy", pop(pmeta, 0.3, U["mlp"]).astype(np.float16))
        meta_all = dict(np.load("data/walks/meta.npz"))
        for cond in ("walk", "shuffled"):
            os.makedirs(f"{P.acts}/{m}-{cond}", exist_ok=True)
            for l in LAYERS:                                        # only the 2,928 shown frames carry signal
                X = np.zeros((len(meta_all["walk"]), U["mlp"]), np.float16)
                X[wrows] = pop(wmeta, 0.5, U["mlp"])
                np.save(f"{P.acts}/{m}-{cond}/{l}.npy", X)
    run_assay(P, models, LAYERS, args.threads)
    res = summarize(P, models)
    s = res["summary"]
    print(json.dumps({k: s[k] for k in ("primary_mlp", "primary_resid", "joint_reading")}, indent=1, default=str)[:3000])
    print("gates:", json.dumps(res["gates"], default=str)[:1500])
    print("summary keys:", sorted(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["assay", "summarize", "selftest", "calibrate"])
    ap.add_argument("--models", nargs="*", help="default: runs 3-7 and the base model")
    ap.add_argument("--layers", nargs="*", default=LAYERS)
    ap.add_argument("--root", default=".", help="selftest: a scratch folder")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("POP_THREADS", torch.get_num_threads())))
    args = ap.parse_args()
    if args.mode == "selftest":
        selftest(args)
        return
    if args.mode == "calibrate":
        torch.set_num_threads(args.threads)
        calibrate()
        return
    P = Paths()
    models = args.models or MODELS
    if args.mode == "assay":
        files = ["scripts/exp33_analysis.py", "scripts/population_assay.py", "spatialcells/single_query.py",
                 "spatialcells/analyze.py", "data/walks/meta.npz", "data/walks/probes_meta.npz", sq.TRIALS]
        md5 = {f: hashlib.md5(open(f, "rb").read()).hexdigest() for f in files if os.path.exists(f)}
        os.makedirs(P.err, exist_ok=True)
        json.dump({"host": os.uname().nodename, "time": time.strftime("%Y-%m-%d %H:%M %Z"), **md5},
                  open(f"{P.err}/_inputs_md5.json", "w"), indent=1)
        run_assay(P, models, args.layers, args.threads)
    else:
        summarize(P, models)


if __name__ == "__main__":
    main()
