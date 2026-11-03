"""Experiment 35 (docs/experiments.md): what copying activations between trials does to the fine-tuned VLM's decision,
from spatialcells/vlm_patch.py's done files. No model is run.

    .venv-claude/bin/python scripts/exp35_analysis.py --fetch     # copy the small output files from vast-3 first
    .venv-claude/bin/python scripts/exp35_analysis.py             # -> results/exp35_summary.json (top-level summary)
    .venv-claude/bin/python scripts/exp35_analysis.py selftest    # synthetic done files through the whole analysis

Per pair (recipient r, place donor p): S_r and S_p are the clean scores (log P(near label) - log P(far label)), the
donor shown under the recipient's question; G = S_r - S_p. A pair is included if S_r > 0 and S_p < 0 (both views
placed by the clean model). For any copy X, m_X = (S_r - S_X) / G: for the place donor the fraction of the way to the
donor (F, experiment 27's transfer), for the same-place and random-place donors the move toward the far option in
units of the same pair's place gap (signed, and |m| for the same-place donor, whose copy should move the decision
in neither direction). Means over included pairs; 95% intervals from 10,000 resamples of the walks (seed 0), the same
resamples for every statistic, so differences on the same pairs are paired. The rule (registration) is applied at
each run's primary layer (layers.json, recomputed here from the sweep).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from spatialcells import vlm_patch as vp                              # noqa: E402

RUNS = {str(k): f"qwen8b-ft-e4-s{k}" for k in range(3, 8)}
BASE = "qwen8b"
IN = "results/exp35"
OUT = "results/exp35_summary.json"
N_BOOT = 10000
AGREE_MIN = 0.99
MARGIN, SHARE, C2_MARGIN = 0.2, 0.5, 0.1
FILES = ("done_sweep.jsonl", "done_subspace.jsonl", "layers.json", "fit_report.json", "checks.jsonl", "meta.json")


# ---------------------------------------------------------------- bootstrap over walks
class Boot:
    def __init__(self, walks, n=N_BOOT, seed=0):
        self.W = np.unique(walks)
        self.idx = np.searchsorted(self.W, walks)
        draws = np.random.default_rng(seed).integers(0, len(self.W), (n, len(self.W)))
        self.C = np.zeros((n, len(self.W)))
        np.add.at(self.C, (np.arange(n)[:, None], draws), 1)

    def means(self, v):
        v = np.asarray(v, float)
        ok = ~np.isnan(v)
        s = np.bincount(self.idx[ok], weights=v[ok], minlength=len(self.W))
        c = np.bincount(self.idx[ok], minlength=len(self.W)).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.C @ s) / (self.C @ c)

    def stat(self, v):
        v = np.asarray(v, float)
        b = self.means(v)
        return {"mean": float(np.nanmean(v)) if (~np.isnan(v)).any() else float("nan"),
                "ci95": [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))], "n": int((~np.isnan(v)).sum())}

    def ratio(self, a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = self.means(a) / self.means(b)
        return {"ratio": float(np.nanmean(a) / np.nanmean(b)), "ci95": [float(np.nanpercentile(r, 2.5)),
                                                                         float(np.nanpercentile(r, 97.5))]}


def above(s, margin):
    return s["mean"] >= margin and s["ci95"][0] > 0


# ---------------------------------------------------------------- loading and gates
def load(model, root=IN):
    d = f"{root}/{model}"
    rd = lambda f: [json.loads(l) for l in open(f"{d}/{f}")] if os.path.exists(f"{d}/{f}") else []
    js = lambda f: json.load(open(f"{d}/{f}")) if os.path.exists(f"{d}/{f}") else None
    return {"sweep": rd("done_sweep.jsonl"), "sub": rd("done_subspace.jsonl"), "layers": js("layers.json"),
            "fit": js("fit_report.json"), "checks": rd("checks.jsonl"), "meta": js("meta.json")}


def tol(G):
    return max(vp.TOL_ABS, vp.TOL_REL * abs(G)) + vp.TOL_EPS   # amendment 1


def gates(D, run_ids, finetuned):
    R = D["sweep"]
    n_layers = D["meta"]["n_layers"] if D["meta"] else 36
    g = {"sweep_complete": {r["id"] for r in R} == set(run_ids) and len(R) == len(run_ids)}
    ref = [r for r in R if r.get("stored_answer") is not None]
    g["exp33_agreement"] = float(np.mean([r["answer"] == r["stored_answer"] for r in ref])) if ref else float("nan")
    g["exp33_compared"] = len(ref)
    g["score_sign_matches_greedy"] = float(np.mean([(r["S"]["recipient"] > 0) == (r["answer"] == r["near"])
                                                    for r in R if r["answer"] > 0])) if R and "near" in R[0] else None
    G = lambda r: r["S"]["recipient"] - r["S"]["place"]
    g["none_within_tol"] = float(np.mean([abs(r["S_none"] - r["S"]["recipient"]) <= tol(G(r)) for r in R])) if R else 0
    last = [abs(v[site][n_layers - 1] - r["S"]["recipient"]) <= tol(G(r))
            for r in R for v in r["sweep"].values() for site in ("view", "view_end") if site in v]
    g["last_layer_copy_changes_nothing"] = float(np.mean(last)) if last else 0
    g["stop_checks"] = len(D["checks"])
    g["stop_checks_failed"] = sum(bool(c.get("fails")) for c in D["checks"])
    ok = (g["sweep_complete"] and g["none_within_tol"] >= AGREE_MIN and g["last_layer_copy_changes_nothing"] >= AGREE_MIN
          and g["stop_checks"] > 0 and g["stop_checks_failed"] == 0)
    if finetuned:
        ok &= bool(g["exp33_agreement"] >= AGREE_MIN) and bool((g["score_sign_matches_greedy"] or 0) >= AGREE_MIN)
        lay = vp.choose_layers(R, n_layers) if g["sweep_complete"] else None
        g["layers_recomputed_match"] = bool(lay and D["layers"] and lay["layers"] == D["layers"]["layers"]
                                            and lay["primary"] == D["layers"]["primary"])
        S = {s["id"]: s for s in D["sub"]}
        g["subspace_complete"] = set(S) == set(run_ids)
        dv, ds = [], []
        for r in R:
            s = S.get(r["id"])
            if not s or not vp.included(r):
                continue
            ds.append(abs(s["S"]["recipient"] - r["S"]["recipient"]) <= tol(G(r)))
            for L, d in s["layers"].items():
                for kind in d:
                    dv.append(abs(d[kind]["view"] - r["sweep"][kind]["view"][int(L)]) / G(r))
        g["subspace_clean_within_tol"] = float(np.mean(ds)) if ds else 0
        g["subspace_view_vs_sweep_mean_abs_dF"] = float(np.mean(dv)) if dv else float("nan")
        ok &= (g["layers_recomputed_match"] and g["subspace_complete"] and g["subspace_clean_within_tol"] >= AGREE_MIN
               and g["subspace_view_vs_sweep_mean_abs_dF"] <= 0.02)
    else:
        g["exp33_agreement_note"] = "base model: reported only (experiment 33's base parity was 96-97% by device)"
    g["ok"] = bool(ok)
    return g


# ---------------------------------------------------------------- per run
def frac(r, s):
    return (r["S"]["recipient"] - s) / (r["S"]["recipient"] - r["S"]["place"])


def sweep_profiles(R, B, n_layers):
    out = {}
    for kind, sites in vp.SWEEP_SITES.items():
        for site in sites:
            if not any(kind in r["sweep"] and site in r["sweep"][kind] for r in R):
                continue
            prof = []
            for L in range(n_layers):
                m = np.array([frac(r, r["sweep"][kind][site][L]) if kind in r["sweep"] else np.nan for r in R])
                flip = np.array([float(r["sweep"][kind][site][L] < 0) if kind in r["sweep"] else np.nan for r in R])
                e = {"m": B.stat(m), "flips": B.stat(flip)}
                if kind == "same":
                    e["abs_m"] = B.stat(np.abs(m))
                prof.append(e)
            out[f"{kind}/{site}"] = prof
    return out


def first_or_last(prof, mark, last):
    v = [p["m"]["mean"] for p in prof]
    L = [i for i, x in enumerate(v) if x >= mark and 2 <= i <= len(v) - 2] if last else \
        [i for i, x in enumerate(v) if x >= mark]
    return (max(L) if last else min(L)) if L else None


def layer_table(R, S, L, B, ks):
    """Every subspace condition at layer L: fractions, contrasts and shares."""
    ids = [r["id"] for r in R]
    sub = [S.get(i) for i in ids]

    def col(kind, cond):
        return np.array([frac(r, s["layers"][str(L)][kind][cond]) if s and kind in s["layers"].get(str(L), {})
                         and cond in s["layers"][str(L)][kind] else np.nan for r, s in zip(R, sub)])

    def share(kind, cond):
        v = [s["share"].get(f"{L}/{kind}/{cond}") for s in sub if s]
        v = [x for x in v if x is not None]
        return float(np.median(v)) if v else None

    t = {"view": {k: B.stat(col(k, "view")) for k in vp.KINDS}}
    t["view"]["same_abs"] = B.stat(np.abs(col("same", "view")))
    Fv = col("place", "view")
    rows = {}
    for k in ks:
        Fp, Fq = col("place", f"pos{k}"), col("place", f"pca{k}")
        Fr = np.nanmean(np.stack([col("place", f"rand{k}s{s}") for s in (0, 1, 2)]), 0)
        ms, mr = col("same", f"pos{k}"), col("random", f"pos{k}")
        e = {"F_pos": B.stat(Fp), "F_pca": B.stat(Fq), "F_rand_mean3": B.stat(Fr),
             "R_pos_of_view": B.ratio(Fp, Fv), "R_pca_of_view": B.ratio(Fq, Fv),
             "E_rand": B.stat(Fp - Fr), "E_pca": B.stat(Fp - Fq),
             "same_pos_m": B.stat(ms), "same_pos_abs_m": B.stat(np.abs(ms)), "C1": B.stat(Fp - np.abs(ms)),
             "random_pos_m": B.stat(mr), "C2": B.stat(Fp - mr),
             "same_pca_abs_m": B.stat(np.abs(col("same", f"pca{k}"))),
             "same_rand_abs_m": B.stat(np.abs(col("same", f"rand{k}s0"))),
             "flips_pos": B.stat(np.array([float(s["layers"][str(L)]["place"][f"pos{k}"] < 0) if s else np.nan
                                           for s in sub])),
             "share_median": {c: share("place", c) for c in (f"pos{k}", f"pca{k}", f"rand{k}s0")}}
        if k >= 10:
            Fn = col("place", f"notpos{k}")
            e["F_view_without_pos"] = B.stat(Fn)
            e["loss_without_pos"] = B.ratio(Fv - Fn, Fv)
        if k == 10:
            e["random_pca_m"] = B.stat(col("random", "pca10"))
        rows[str(k)] = e
    t["k"] = rows
    t["readout2"] = {"F": B.stat(col("place", "readout2")), "share_median": share("place", "readout2")}
    return t


def verdict(t, F_view, ks, no_primary):
    """The registered rule at the primary layer."""
    st = {}
    for k in ks:
        e = t["k"][str(k)]
        big = e["R_pos_of_view"]["ratio"] >= SHARE
        moves = big and above(e["E_rand"], MARGIN)
        loc = moves and above(e["C1"], MARGIN) and above(e["C2"], C2_MARGIN)
        st[str(k)] = {"half_of_view": bool(big), "beats_random": bool(above(e["E_rand"], MARGIN)), "moves": bool(moves),
                      "C1": bool(above(e["C1"], MARGIN)), "C2": bool(above(e["C2"], C2_MARGIN)), "location": bool(loc)}
    loc = [k for k in ks if st[str(k)]["location"]]
    out = {"states": st, "F_view_primary": F_view}
    if no_primary:
        out["verdict"] = "unclear: no layer from 2 on where the view copy moves the decision half way"
    elif loc:
        k = min(loc)
        out.update(verdict="low-dimensional location code", k_star=k,
                   specificity="position-specific" if above(t["k"][str(k)]["E_pca"], MARGIN)
                   else "not specific to position labels (top-variance directions do about as well)")
    elif not any(st[str(k)]["half_of_view"] for k in ks):
        out["verdict"] = "not low-dimensional"
    elif not any(st[str(k)]["moves"] for k in ks):
        out["verdict"] = "unclear: subspaces that move the decision are not beaten by random ones"
    elif not any(st[str(k)]["C1"] for k in ks if st[str(k)]["moves"]):
        out["verdict"] = "moves the decision, not specifically by location"
    else:
        out["verdict"] = "unclear: place beats same place but not random place"
    return out


def familiarity_split(R, S, L, B, novel):
    """Place donor, full view and position-10 copies at layer L, by whether the place donor is novel (described)."""
    out = {}
    for name, want in (("novel_donor", True), ("familiar_donor", False)):
        keep = np.array([novel.get(r["id"]) == want for r in R])
        for c in ("view", "pos10"):
            v = np.array([frac(r, S[r["id"]]["layers"][str(L)]["place"][c]) if r["id"] in S
                          and c in S[r["id"]]["layers"][str(L)]["place"] else np.nan for r in R])
            out[f"{name}/{c}"] = B.stat(np.where(keep, v, np.nan))
    return out


def analyze_run(D, ks=vp.KS, novel=None):
    R = [r for r in D["sweep"] if vp.included(r)]
    n_layers = D["meta"]["n_layers"]
    if not R:
        return {"n_recipients": len(D["sweep"]), "n_included": 0}
    B = Boot(np.array([r["walk"] for r in R]))
    prof = sweep_profiles(R, B, n_layers)
    lay = D["layers"]
    S = {s["id"]: s for s in D["sub"]}
    pv = prof["place/view"]
    out = {"n_recipients": len(D["sweep"]), "n_included": len(R),
           "recipient_correct": float(np.mean([r["S"]["recipient"] > 0 for r in D["sweep"]])),
           "place_donor_placed": float(np.mean([r["S"]["place"] < 0 for r in D["sweep"]])),
           "same_donor_placed": float(np.mean([r["S"]["same"] > 0 for r in D["sweep"] if "same" in r["S"]])),
           "G_median": float(np.median([r["S"]["recipient"] - r["S"]["place"] for r in R])) if R else None,
           "layers": lay,
           "profile_marks": {"view_last_ge_0.5": first_or_last(pv, 0.5, True),
                             "view_last_ge_0.9": first_or_last(pv, 0.9, True),
                             "question_first_ge_0.5": first_or_last(prof["place/question"], 0.5, False)
                             if "place/question" in prof else None,
                             "last_first_ge_0.5": first_or_last(prof["place/last"], 0.5, False)
                             if "place/last" in prof else None,
                             "view_end_peak": max(((p["m"]["mean"], L) for L, p in enumerate(prof["place/view_end"])),
                                                  default=None) if "place/view_end" in prof else None},
           "profiles": prof}
    if not D["sub"]:
        return out
    ks = tuple(k for k in ks if all(f"pos{k}" in s["layers"][L]["place"] for s in D["sub"] for L in s["layers"]
                                    if "place" in s["layers"][L]))
    out["tables"] = {str(L): layer_table(R, S, L, B, ks) for L in lay["layers"]}
    Lp = lay["primary"]
    out["primary_layer"] = Lp
    out["rule"] = verdict(out["tables"][str(Lp)], pv[Lp]["m"], ks, lay.get("no_primary", False))
    if novel:
        out["by_place_donor_familiarity_at_primary"] = familiarity_split(R, S, Lp, B, novel)
    return out


def analyze_base(D):
    """No fractions (the base model is near chance): the raw move toward the far option, S_r - S_X, in nats."""
    R = D["sweep"]
    if not R:
        return None
    n_layers = D["meta"]["n_layers"]
    B = Boot(np.array([r["walk"] for r in R]))
    d = lambda r, k, L: (r["S"]["recipient"] - r["sweep"][k]["view"][L]) if k in r["sweep"] else np.nan
    prof = []
    for L in range(n_layers):
        p, s, q = (np.array([d(r, k, L) for r in R]) for k in ("place", "same", "random"))
        prof.append({"place": B.stat(p), "same": B.stat(s), "same_abs": B.stat(np.abs(s)), "random": B.stat(q),
                     "place_minus_same": B.stat(p - s), "place_minus_random": B.stat(p - q)})
    return {"n": len(R), "recipient_correct": float(np.mean([r["S"]["recipient"] > 0 for r in R])),
            "place_donor_placed": float(np.mean([r["S"]["place"] < 0 for r in R])),
            "n_would_be_included": sum(vp.included(r) for r in R), "view_move_nats": prof, "fit": D["fit"]}


def summarize(root=IN, out=OUT, runs=RUNS, base=BASE, quiet=False):
    Pr = [p for p in json.load(open(vp.PAIRS)) if "place" in p["donors"]]
    run_ids = [p["id"] for p in Pr]
    near = {p["id"]: p["near"] for p in Pr}
    novel = {p["id"]: p["donors"]["place"]["novel"] for p in Pr}
    res, gs = {}, {}
    for run, m in runs.items():
        D = load(m, root)
        for r in D["sweep"]:
            r["near"] = near[r["id"]]
        gs[m] = gates(D, run_ids, True)
        res[run] = analyze_run(D, novel=novel) if D["sweep"] else None
        if res[run]:
            res[run]["fit"] = D["fit"]
    Db = load(base, root)
    for r in Db["sweep"]:
        r["near"] = near[r["id"]]
    gs[base] = gates(Db, run_ids, False) if Db["sweep"] else {"ok": False, "missing": True}
    vs = {run: r["rule"] for run, r in res.items() if r and "rule" in r}
    if not vs:
        print("no run has subspace results yet")
    names = {v["verdict"] for v in vs.values()}
    recipe = (names.pop() if len(names) == 1 and len(vs) == len(runs) else "mixed")
    # amendment 2: the base model's arm is dropped (its cached path differs from a full recompute by up to 0.75 nats);
    # it has no part in the verdict rule, so pipeline_ok covers the five four-epoch runs and the base gates are reported only
    summ = {"pipeline_ok": all(gs[m]["ok"] for m in runs.values()),
            "base_arm": "dropped (amendment 2); its gates are reported only",
            "gates": gs, "recipe_verdict": recipe,
            "k_star_by_run": {run: v.get("k_star") for run, v in vs.items()},
            "verdict_by_run": {run: v["verdict"] + (f" (k = {v['k_star']}, {v['specificity']})" if "k_star" in v else "")
                               for run, v in vs.items()},
            "primary_layer_by_run": {run: r.get("primary_layer") for run, r in res.items() if r},
            "F_view_at_primary_by_run": {run: v["F_view_primary"]["mean"] for run, v in vs.items()}}
    for run, r in res.items():
        if not r or "tables" not in r:
            continue
        t = r["tables"][str(r["primary_layer"])]
        summ.setdefault("at_primary", {})[run] = {k: {x: (e[x]["mean"] if "mean" in e[x] else e[x]["ratio"])
                                                    for x in ("F_pos", "F_pca", "F_rand_mean3", "R_pos_of_view", "C1",
                                                              "C2", "E_pca", "same_pos_abs_m")}
                                                for k, e in t["k"].items()}
    full = {"summary": summ, "runs": res, "base": analyze_base(Db) if Db["sweep"] else None}
    json.dump(full, open(out, "w"), indent=1, default=float)
    if not quiet:
        print(json.dumps(summ, indent=1, default=float)[:6000])
        print(f"-> {out}")
    return full


# ---------------------------------------------------------------- fetch and selftest
def fetch():
    """Only this experiment's small files from the vast-3 container (never its whole results/ or the arrays)."""
    models = list(RUNS.values()) + [BASE]
    inc = " ".join(f'--include="{m}/" ' + " ".join(f'--include="{m}/{f}"' for f in FILES) for m in models)
    cmd = ('source scripts/vast1/common.sh && '
           f'mkdir -p {IN} && rsync -a -e "$MI_RSYNC_SHELL" {inc} --exclude="*" '
           f'"$MI_CONTAINER:$MI_REMOTE_ROOT/{vp.OUT}/" {IN}/ && '
           f'rsync -a -e "$MI_RSYNC_SHELL" --include="run_inputs*.md5" --exclude="*" '
           f'"$MI_CONTAINER:$MI_REMOTE_ROOT/{IN}/" {IN}/')
    subprocess.run(["bash", "-c", cmd], check=True, env={**os.environ, "MI_HOST": "vast3"})


def synth(root, scenario, seed):
    """Synthetic done files in vlm_patch's format for the five runs and the base model. scenario: location (pos-10
    carries the move and same-place donors do not move it), view (pos-10 moves it, and same-place donors as much,
    either way), distributed (no subspace up to 50 moves it)."""
    rng = np.random.default_rng(seed)
    Pr = [p for p in json.load(open(vp.PAIRS)) if "place" in p["donors"]]
    n_layers, ks = 36, vp.KS
    drop = 1 / (1 + np.exp((np.arange(n_layers) - 19.0) / 1.5))          # view copy: 1 early, 0 late
    drop[-1] = 0
    conds = vp.subspace_conds(ks)
    for m in list(RUNS.values()) + [BASE]:
        d = f"{root}/{m}"
        os.makedirs(d, exist_ok=True)
        recs, subs = [], []
        for p in Pr:
            Sr, Sp = rng.normal(10, 2), rng.normal(-10, 2)
            if rng.random() < 0.02:
                Sr = -abs(Sr)
            G = Sr - Sp
            S = {"recipient": Sr, "place": Sp}
            if "same" in p["donors"]:
                S["same"] = rng.normal(9, 2)
            if "random" in p["donors"]:
                S["random"] = rng.normal(0, 8)
            sw = {}
            for kind in [k for k in vp.KINDS if k in p["donors"]]:
                tgt = S[kind]
                sw[kind] = {"view": list(Sr - drop * (Sr - tgt) + rng.normal(0, 0.05, n_layers)),
                            "view_end": list(Sr - 0.1 * drop * (Sr - tgt) + rng.normal(0, 0.05, n_layers))}
                if kind == "place":
                    sw[kind]["question"] = list(Sr - (1 - drop) * G + rng.normal(0, 0.05, n_layers))
                    sw[kind]["last"] = list(Sr - (1 - drop) ** 2 * G + rng.normal(0, 0.05, n_layers))
                for site in sw[kind]:
                    sw[kind][site][-1] = Sr if site in ("view", "view_end") else sw[kind][site][-1]
            ans = p["near"] if Sr > 0 else p["far"]
            recs.append({"id": p["id"], "walk": p["walk"], "probe": p["probe"], "answer": ans, "stored_answer": ans,
                         "S": S, "S_none": Sr + rng.normal(0, 0.01), "sweep": sw})
        lay = vp.choose_layers(recs, n_layers)
        for p, r in zip(Pr, recs):
            S, Sr = r["S"], r["S"]["recipient"]
            G = Sr - S["place"]
            res, share = {}, {}
            for L in lay["layers"]:
                f = drop[L]
                for kind in [k for k in vp.KINDS if k in p["donors"]]:
                    out = {}
                    for c in conds[kind]:
                        if c == "view":
                            v = r["sweep"][kind]["view"][L]
                        else:
                            k = int(re.search(r"\d+", c).group())
                            if kind == "place":
                                a = {"location": {2: 0.3, 10: 0.85, 50: 0.95}, "view": {2: 0.3, 10: 0.85, 50: 0.95},
                                     "distributed": {2: 0.02, 10: 0.05, 50: 0.15}}[scenario][k]
                                a = a if c.startswith("pos") or c == "readout2" else (
                                    1 - a if c.startswith("notpos") else (0.1 if c.startswith("pca") else 0.0))
                                v = Sr - f * a * G + rng.normal(0, 0.05)
                            elif kind == "same":
                                a = 0.0 if c.startswith("rand") else (
                                    0.03 if scenario != "view" else 0.8 * {2: 0.3, 10: 0.85, 50: 0.95}[k])
                                v = Sr - f * a * G * rng.choice([-1, 1]) + rng.normal(0, 0.05)
                            else:
                                a = {"location": 0.85, "view": 0.85, "distributed": 0.05}[scenario]
                                v = Sr - f * a * (Sr - S["random"]) + rng.normal(0, 0.05)
                            share[f"{L}/{kind}/{c}"] = float(rng.uniform(0, 1))
                        out[c] = float(v)
                    res.setdefault(str(L), {})[kind] = out
            subs.append({"id": p["id"], "walk": p["walk"], "probe": p["probe"], "half_of_U": 1 - p["walk"] % 2,
                         "S": dict(S), "layers": res, "share": share})
        with open(f"{d}/done_sweep.jsonl", "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in recs)
        if m != BASE:
            with open(f"{d}/done_subspace.jsonl", "w") as fh:
                fh.writelines(json.dumps(s) + "\n" for s in subs)
            json.dump(lay, open(f"{d}/layers.json", "w"))
        json.dump({"n_layers": n_layers}, open(f"{d}/meta.json", "w"))
        with open(f"{d}/checks.jsonl", "w") as fh:
            fh.write(json.dumps({"fails": []}) + "\n")


def selftest(root):
    want = {"location": ("low-dimensional location code", 10), "view": ("moves the decision, not specifically by "
                                                                         "location", None),
            "distributed": ("not low-dimensional", None)}
    for i, (sc, (v, k)) in enumerate(want.items()):
        r = f"{root}/{sc}"
        synth(r, sc, i)
        S = summarize(r, f"{r}/summary.json", quiet=True)["summary"]
        assert S["pipeline_ok"], S["gates"]
        assert S["recipe_verdict"] == v, (sc, S["recipe_verdict"], S["verdict_by_run"])
        assert all(x == k for x in S["k_star_by_run"].values()), (sc, S["k_star_by_run"])
        print(f"selftest {sc}: pipeline_ok, primary layers {sorted(set(S['primary_layer_by_run'].values()))}, "
              f"verdict '{S['recipe_verdict']}' (k* {set(S['k_star_by_run'].values())}) as built")
    print(f"selftest passed -> {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="summarize", choices=["summarize", "selftest"])
    ap.add_argument("--fetch", action="store_true", help="first copy this experiment's small files from vast-3")
    ap.add_argument("--root", help="selftest: scratch folder; summarize: the done files' folder (default results/exp35; "
                                   "on vast-3, data/vlm_patch)")
    args = ap.parse_args()
    if args.mode == "selftest":
        import tempfile
        selftest(args.root or tempfile.mkdtemp(prefix="exp35_selftest_"))
        return
    if args.fetch:
        fetch()
    summarize(args.root or IN)


if __name__ == "__main__":
    main()
