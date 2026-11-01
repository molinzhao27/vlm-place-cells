"""Experiment 30 (pre-registered in docs/experiments.md): the fine-tuned model on walks it was not trained on.

    .venv-mac/bin/python scripts/trajectory_shift_analysis.py --stimulus   # before any model run: counts, walk statistics,
                                                                          # the pixel window-maximum check -> results/trajectory_shift_stimulus.json
    .venv-mac/bin/python scripts/trajectory_shift_analysis.py --fetch      # copy the five runs' answers (and parity files) from vast-3
    .venv-mac/bin/python scripts/trajectory_shift_analysis.py              # -> results/trajectory_shift.json
    .venv-mac/bin/python scripts/trajectory_shift_analysis.py --test       # stand-in data and synthetic answers, checks the verdicts

Per run (four-epoch runs 3-7) and room: accuracy on the novel trials of each condition's standard trials and of the
neighborhood (conflict, agreement) trials; the five-run mean per room; means over the nine rooms with 95% bootstrap CIs
over rooms (10,000 resamples, seed 0); per-room differences the same way. Beside them, experiment 20's 149 retrieval
rules on the same trials (scripts/retrieval_baselines.py's score_trial and unit_feats, imported unchanged; the combined
regression trained leave-one-room-out on the condition's standard novel trials; b* chosen per condition on its own
trials). Then the registered verdicts. The paper's numbers are in the JSON's "summary" (fractions; 0.15 = 15 points).
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import retrieval_baselines as rb  # noqa: E402
from spatialcells import trajectory_shift as ts  # noqa: E402

OUT, STIM = "results/trajectory_shift.json", "results/trajectory_shift_stimulus.json"
RUNS = [3, 4, 5, 6, 7]
SHIFTS = ["long-step", "low-turn", "patrol"]
ANS = "results/two_choice"
MARGIN, DROP, LOW_MARGIN, MAX_COST, MIN_CONFLICT = 0.15, -0.10, 0.05, 0.10, 20
PIX_RULE = "window maximum | pixels | 64 shown | k 16"
PATHS = {"e30": ts.E30, "export": ts.EXPORT, "answers": ANS}


def answer_file(k):
    return f"{PATHS['answers']}/Qwen3-VL-8B-Instruct+qwen8b-proc-e4-s{k}-{ts.COMBINED}-256px-max64.jsonl"


def scenes():
    return [(room, cond) for cond in ts.COND_ORDER for room in ts.ROOMS]


def load_trials(sc):
    out = {}
    for name in ("standard", "nbhd"):
        path = f"{PATHS['export']}/{sc}/{name}.json"
        if os.path.exists(path):
            out[name] = [t for t in json.load(open(path)) if not t["familiar"]]
    return out


def boot(x):
    return rb.boot_scene_mean(x)


def sign_test(k, n):
    """Two-sided binomial sign test."""
    if n == 0:
        return float("nan")
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * p)


# ------------------------------------------------------------------------------------------------ scene data
def scene_data(sc, need_dino):
    """Poses (frame rows in walk order), pixel features and (if present) DINOv2 features of one scene."""
    d = f"{PATHS['e30']}/{sc}"
    q = np.load(f"{d}/poses.npz")
    meta = {k[6:]: q[k] for k in q.files if k.startswith("frame_")}
    pmeta = {k[6:]: q[k] for k in q.files if k.startswith("probe_")}
    px = np.load(f"{d}/pixels.npz")
    z = {"frame_pixels": ts.pix_from_sums(px["frame_sums"])}
    probe_pix = ts.pix_from_sums(px["probe_sums"])
    if os.path.exists(f"{d}/dino.npz"):
        dz = np.load(f"{d}/dino.npz")
        pidx = dz["probe_idx"]
        z.update(frame_cls=dz["frame_cls"], frame_patch=dz["frame_patch"], probe_cls=dz["probe_cls"],
                 probe_patch=dz["probe_patch"], probe_pixels=probe_pix[pidx])
    elif need_dino:
        return None
    else:
        pidx = np.arange(len(probe_pix))
        z["probe_pixels"] = probe_pix
    order = {int(w): np.flatnonzero(meta["walk"] == w)[np.argsort(meta["step"][meta["walk"] == w])]
             for w in np.unique(meta["walk"])}
    return {"meta": meta, "pmeta": pmeta, "z": z, "prow": {int(p): i for i, p in enumerate(pidx)}, "order": order}


def pixel_rule(t, sd):
    """Experiment 20's window maximum with pixels over the 64 shown frames, k = 16: 1 near, 0 far, 0.5 tie."""
    W = sd["z"]["frame_pixels"]
    mu, s = W.mean(0), W.std(0) + 1e-6
    rows = sd["order"][t["walk"]]
    S = np.array(ts.shown_order(t["a"], t["b"]))
    U = (W[rows[S - 1]] - mu) / s
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    P = (sd["z"]["probe_pixels"][sd["prow"][t["probe"]]] - mu) / s
    sim = U @ (P / np.linalg.norm(P))
    v = {o: sim[np.abs(S - o) <= 16].max() for o in (t["near"], t["far"])}
    return rb.pick(v[t["near"]], v[t["far"]])


def rule_scores(sc, trials, sd):
    """Per trial: the 148 hand rules' scores and signed values (experiment 20's order), and its breakdown values."""
    U = rb.unit_feats(sd["z"])
    out = []
    for t in trials:
        rows, p = sd["order"][t["walk"]], t["probe"]

        def sims_of(f, S, rows=rows, p=p):
            W, P = U[f]
            return W[rows[S - 1]] @ P[sd["prow"][p]]
        sc_, sg, shown = rb.score_trial(t, rows, sims_of, len(rows))
        raw = U["dino-cls-raw"]
        meta, pmeta = sd["meta"], sd["pmeta"]
        d = np.hypot(meta["x"][rows] - pmeta["x"][p], meta["y"][rows] - pmeta["y"][p])
        dh = np.abs((meta["theta"][rows] - pmeta["theta"][p] + np.pi) % (2 * np.pi) - np.pi)
        same = dh < np.pi / 4
        out.append({"scores": sc_, "signed": sg, "a_is_near": t["a"] == t["near"],
                    "heading gap (deg)": float(t["near_heading_diff_deg"]),
                    "distance margin (m)": float(t["far_dist_m"] - t["near_dist_m"]),
                    "novelty margin (m)": float(d[same].min()) if same.any() else float("inf"),
                    "visual overlap": float(np.max(raw[0][rows[np.array(sorted(shown)) - 1]] @ raw[1][sd["prow"][p]]))})
    return out


def combined(train_rows, test_rows):
    from sklearn.linear_model import LogisticRegression
    X = np.array([r["signed"] for r in train_rows], float)
    y = np.array([r["a_is_near"] for r in train_rows], float)
    X, y = np.concatenate([X, -X]), np.concatenate([y, 1 - y])
    sd = X.std(0)
    sd[sd == 0] = 1
    clf = LogisticRegression(C=1.0, fit_intercept=False, max_iter=10_000).fit(X / sd, y)
    z = clf.decision_function(np.array([r["signed"] for r in test_rows], float) / sd)
    pick_a = np.where(z > 0, 1.0, np.where(z < 0, 0.0, 0.5))
    return np.array([pa if r["a_is_near"] else 1 - pa for pa, r in zip(pick_a, test_rows)])


# ------------------------------------------------------------------------------------------------ answers
def load_answers(manifest):
    """run -> combined id -> 1/0 (unparsed counts as wrong); None for a run whose file is missing or incomplete."""
    out = {}
    for k in RUNS:
        path = answer_file(k)
        if not os.path.exists(path):
            out[k] = None
            continue
        rows = {}
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            t = manifest[r["id"]]
            assert (r["near"], r["far"], r["a"], r["b"]) == (t["near"], t["far"], t["a"], t["b"]), (path, r["id"])
            assert r["id"] not in rows, (path, "duplicate", r["id"])
            rows[r["id"]] = float(r["answer"] == r["near"])
        out[k] = rows if set(rows) == set(manifest) else None
        if out[k] is None:
            print(f"run {k}: {len(rows)} of {len(manifest)} trials answered (incomplete)")
    return out


def combined_manifest():
    """Combined id -> trial, from the per-scene lists (the ids --combine gives them)."""
    man = {}
    for room, cond in scenes():
        sc = ts.scene(room, cond)
        for name, trials in load_trials(sc).items():
            for t in trials:
                man[f"{sc}-{name}-{t['id']}"] = {**t, "scene": sc, "room": room, "condition": cond, "set": name}
    return man


# ------------------------------------------------------------------------------------------------ verdicts
def shift_verdict(margin, margin_ci, delta):
    if margin < LOW_MARGIN or margin_ci[0] <= 0:
        return "fails"
    if margin >= MARGIN and margin_ci[0] > 0 and delta >= DROP:
        return "survives"
    return "reduced"


def separation_verdict(conflict, agreement):
    if agreement - conflict <= MAX_COST:
        return "geometric proximity wins"
    if conflict <= (agreement + 0.5) / 2:
        return "local association"
    return "partial"


def room_acc(values, rooms_of, rooms):
    """Mean of per-trial values in each room (nan if none)."""
    return np.array([np.mean([v for v, r in zip(values, rooms_of) if r == room]) if any(r == room for r in rooms_of)
                     else np.nan for room in rooms])


def ci_block(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return None
    m, ci = boot(x)
    return {"mean": m, "ci95": ci, "rooms": int(len(x)), "rooms_positive": int((x > 0).sum()),
            "rooms_negative": int((x < 0).sum()), "sign_p": sign_test(int((x > 0).sum()), int((x != 0).sum()))}


# ------------------------------------------------------------------------------------------------ stimulus check
def stimulus(args):
    res = {"scenes": {}, "counts": {}, "walk_stats": {}, "pixel_rule_on_nbhd": {}}
    for room, cond in scenes():
        sc = ts.scene(room, cond)
        info = json.load(open(f"{PATHS['e30']}/{sc}/done.json"))
        res["scenes"][sc] = info
        res["counts"].setdefault(cond, {})[room] = info["trials"]
        res["walk_stats"].setdefault(cond, []).append(info["walk_stats"])
    res["walk_stats"] = {c: {k: float(np.mean([w[k] for w in v])) for k in v[0]} for c, v in res["walk_stats"].items()}
    for cond in ts.NBHD_CONDITIONS:
        per = {"conflict": [], "agreement": []}
        low = []
        for room in ts.ROOMS:
            sc = ts.scene(room, cond)
            nb = load_trials(sc).get("nbhd", [])
            sd = scene_data(sc, need_dino=False)
            for typ in per:
                tt = [t for t in nb if t["type"] == typ]
                per[typ].append(np.mean([pixel_rule(t, sd) for t in tt]) if tt else np.nan)
            n_conf = sum(t["type"] == "conflict" for t in nb)
            if n_conf < MIN_CONFLICT:
                low.append({"room": room, "conflict_trials": n_conf})
        r = {typ: {"room_mean": float(np.nanmean(v)), "per_room": [float(x) for x in v]} for typ, v in per.items()}
        r["opposes_pixels_as_intended"] = bool(r["conflict"]["room_mean"] < 0.5 < r["agreement"]["room_mean"])
        r["rooms_below_min_conflict"] = low
        res["pixel_rule_on_nbhd"][cond] = r
    tot = {}
    for c, rooms in res["counts"].items():
        tot[c] = {"standard_novel": sum(v["standard"]["novel"] for v in rooms.values())}
        if all("nbhd" in v for v in rooms.values()):
            tot[c].update({k: sum(v["nbhd"][k] for v in rooms.values()) for k in ("conflict", "agreement")})
    res["totals"] = tot
    json.dump(res, open(args.stimulus_out, "w"), indent=1)
    print(json.dumps({"totals": tot, "walk_stats": res["walk_stats"],
                      "pixel_rule_on_nbhd": {c: {k: (v[k]["room_mean"] if isinstance(v[k], dict) else v[k]) for k in v}
                                             for c, v in res["pixel_rule_on_nbhd"].items()}}, indent=1))
    print(f"-> {args.stimulus_out}")


# ------------------------------------------------------------------------------------------------ analysis
def exp10_by_type():
    """Experiment 10's novel trials in all ten rooms typed by the neighborhood rule, with the four-epoch runs' accuracy."""
    from spatialcells.dissimilar_trials import landmarks
    from spatialcells.room import get_layout
    res, rows_all = {}, []
    for s in range(5000, 5010):
        room = f"P{s}"
        man = {t["id"]: t for t in rb.trials_of(room) if not t["familiar"]}
        files = rb.model_files(room)
        ans = [rb.load_answers(files[f"LoRA run {k}"], {t["id"]: t for t in rb.trials_of(room)}) for k in RUNS]
        if any(a is None for a in ans):
            return {"status": "missing experiment-10 answers"}
        meta = dict(np.load(f"{rb.data_dir(room)}/meta.npz"))
        pmeta = dict(np.load(f"{rb.data_dir(room)}/probes_meta.npz"))
        order = rb.walk_order(meta)
        lay = get_layout(room)
        posters, pillars = lay["posters"](), lay["pillars"]
        cache = {}

        def marks(f):
            if f not in cache:
                cache[f] = landmarks(meta["x"][f], meta["y"][f], meta["theta"][f], pillars=pillars, posters=posters)
            return cache[f]
        for tid, t in man.items():
            rows = order[t["walk"]]
            seen = landmarks(pmeta["x"][t["probe"]], pmeta["y"][t["probe"]], pmeta["theta"][t["probe"]],
                             pillars=pillars, posters=posters)
            sn, sf = ts.neighbourhoods(t["near"], t["far"])
            share = {k: bool(seen & marks(rows[k - 1])) for k in sn + sf}
            hn, hf = any(share[k] for k in sn), any(share[k] for k in sf)
            typ = "conflict" if hf and not hn else "agreement" if hn and not hf else "neither"
            rows_all.append({"room": room, "type": typ, "acc": float(np.mean([a[tid] for a in ans]))})
    for typ in ("conflict", "agreement", "neither"):
        sel = [r for r in rows_all if r["type"] == typ]
        per_room = [np.mean([r["acc"] for r in sel if r["room"] == f"P{s}"]) for s in range(5000, 5010)
                    if any(r["room"] == f"P{s}" for r in sel)]
        res[typ] = {"n": len(sel), "pooled_acc": float(np.mean([r["acc"] for r in sel])) if sel else None,
                    "room_mean": boot(per_room)[0] if per_room else None, "room_ci95": boot(per_room)[1] if per_room else None}
    return res


def exp10_reference():
    """Experiment 10's novel trials in the nine rooms: the four-epoch runs' accuracy, the reference for A(control)."""
    per_run = {k: [] for k in RUNS}
    for room in ts.ROOMS:
        man = {t["id"]: t for t in rb.trials_of(room)}
        files = rb.model_files(room)
        nov = [i for i, t in man.items() if not t["familiar"]]
        for k in RUNS:
            ans = rb.load_answers(files[f"LoRA run {k}"], man)
            per_run[k].append(float(np.mean([ans[i] for i in nov])))
    per_room = np.mean([per_run[k] for k in RUNS], 0)
    return {"lora": ci_block(per_room), "per_room": per_room.tolist(),
            "per_run": {f"run {k}": float(np.mean(v)) for k, v in per_run.items()}}


def unparsed_share():
    """Per run, the share of answers that name neither option (they count as wrong)."""
    out = {}
    for k in RUNS:
        if os.path.exists(answer_file(k)):
            rows = [json.loads(line) for line in open(answer_file(k)) if line.strip()]
            out[f"run {k}"] = float(np.mean([r["answer"] not in (r["near"], r["far"]) for r in rows]))
    return out


def parity_summary():
    out = {}
    for k in RUNS:
        name = f"Qwen3-VL-8B-Instruct+qwen8b-proc-e4-s{k}"
        new = f"{PATHS['answers']}/e30/{name}-roomP5000-256px-max64-e30parity.jsonl"
        ref = f"{ANS}/{name}-roomP5000-256px-max64.jsonl"
        if os.path.exists(new) and os.path.exists(ref):
            a = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(ref)}
            b = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(new)}
            out[f"run {k}"] = {"identical": sum(b.get(i) == v for i, v in a.items()), "of": len(a)}
    return out


def analyze(args):
    man = combined_manifest()
    answers = load_answers(man)
    done = [k for k in RUNS if answers[k] is not None]
    rooms = ts.ROOMS
    out = {"runs_complete": done, "conditions": {}, "separation": {}, "rules": {}, "trials": {}}
    # rule scores per scene and set
    tables = {}
    for room, cond in scenes():
        sc = ts.scene(room, cond)
        sd = scene_data(sc, need_dino=True)
        if sd is None:
            out["rules"] = {"status": "DINOv2 features missing (trajectory_shift.py --dino)"}
            tables = None
            break
        for name, trials in load_trials(sc).items():
            tables[sc, name] = (trials, rule_scores(sc, trials, sd))
    names = rb.variant_names() + ["combined"]

    def lora_vec(trials, sc, name, runs):
        return np.array([[answers[k][f"{sc}-{name}-{t['id']}"] for t in trials] for k in runs])

    def group(cond, name, typ=None):
        """Per room (aligned with rooms, nan where a room has no such trial): the five-run mean and each run's
        accuracy; the pooled per-trial five-run means."""
        per_room, per_run, pooled = [], {k: [] for k in done}, []
        for room in rooms:
            sc = ts.scene(room, cond)
            tr = [t for t in load_trials(sc).get(name, []) if typ is None or t.get("type") == typ]
            if not tr or not done:
                per_room.append(np.nan)
                for k in done:
                    per_run[k].append(np.nan)
                continue
            v = lora_vec(tr, sc, name, done)
            per_room.append(float(v.mean()))
            for j, k in enumerate(done):
                per_run[k].append(float(v[j].mean()))
            pooled.extend(v.mean(0).tolist())
        return np.array(per_room), {k: np.array(v) for k, v in per_run.items()}, None, pooled

    # combined regression: leave one room out within each condition, trained on standard novel trials
    comb = {}
    if tables:
        for cond in ts.COND_ORDER:
            for room in rooms:
                train = [r for rr in rooms if rr != room for r in tables[ts.scene(rr, cond), "standard"][1]]
                for name in ("standard", "nbhd"):
                    key = (ts.scene(room, cond), name)
                    if key in tables:
                        comb[key] = combined(train, tables[key][1])

    def rule_matrix(cond, name, typ=None):
        """rooms x 149 rule accuracies (nan where a room has no trials)."""
        M = np.full((len(rooms), len(names)), np.nan)
        for i, room in enumerate(rooms):
            key = (ts.scene(room, cond), name)
            if key not in tables:
                continue
            trials, rows = tables[key]
            idx = [j for j, t in enumerate(trials) if typ is None or t.get("type") == typ]
            if not idx:
                continue
            M[i, :-1] = np.array([[rows[j]["scores"][q] for j in idx] for q in range(len(names) - 1)]).mean(1)
            M[i, -1] = comb[key][idx].mean()
        return M

    for cond in ts.COND_ORDER:
        per_room, per_run, _, pooled = group(cond, "standard")
        c = {"n_novel": int(sum(len(load_trials(ts.scene(r, cond)).get("standard", [])) for r in rooms)),
             "lora_per_room": per_room.tolist()}
        if done:
            c["lora"] = ci_block(per_room)
            c["per_run"] = {f"run {k}": float(np.nanmean(v)) for k, v in per_run.items()}
        if tables:
            M = rule_matrix(cond, "standard")
            best = int(np.nanargmax(np.nanmean(M, 0)))
            c["best_rule"] = names[best]
            c["best_rule_acc"] = float(np.nanmean(M[:, best]))
            c["best_rule_per_room"] = M[:, best].tolist()
            c["best_rule_by_family"] = {fam: names[max((q for q in range(len(names)) if names[q].startswith(fam)),
                                                       key=lambda q: np.nanmean(M[:, q]))]
                                        for fam in ("option similarity", "look-alike then time", "window maximum",
                                                    "window vote", "combined")}
            if done:
                c["margin"] = ci_block(per_room - M[:, best])
                c["runs_margin"] = {f"run {k}": float(np.nanmean(v - M[:, best])) for k, v in per_run.items()}
        out["conditions"][cond] = c
    ctrl = out["conditions"]["control"]
    for cond in SHIFTS:
        c = out["conditions"][cond]
        if done:
            c["delta_vs_control"] = ci_block(np.array(c["lora_per_room"]) - np.array(ctrl["lora_per_room"]))
            if tables:
                c["verdict"] = shift_verdict(c["margin"]["mean"], c["margin"]["ci95"], c["delta_vs_control"]["mean"])
                runs_ok = 0
                g_c, g_0 = group(cond, "standard")[1], group("control", "standard")[1]
                for k in done:
                    m = boot(g_c[k] - np.array(c["best_rule_per_room"]))
                    runs_ok += shift_verdict(m[0], m[1], float(np.mean(g_c[k] - g_0[k]))) == "survives"
                c["runs_survive"] = runs_ok
    for cond in ts.NBHD_CONDITIONS:
        s = {}
        conf, conf_run, _, conf_pool = group(cond, "nbhd", "conflict")
        agr, agr_run, _, agr_pool = group(cond, "nbhd", "agreement")
        n_conf = [sum(t.get("type") == "conflict" for t in load_trials(ts.scene(r, cond)).get("nbhd", [])) for r in rooms]
        keep = np.array(n_conf) >= MIN_CONFLICT
        s["n_conflict"], s["n_agreement"] = int(sum(n_conf)), int(sum(
            t.get("type") == "agreement" for r in rooms for t in load_trials(ts.scene(r, cond)).get("nbhd", [])))
        s["rooms_used"] = [r for r, k in zip(rooms, keep) if k]
        if done:
            s["conflict"], s["agreement"] = ci_block(conf[keep]), ci_block(agr[keep])
            s["cost"] = ci_block((agr - conf)[keep])
            s["midpoint"] = (s["agreement"]["mean"] + 0.5) / 2
            s["verdict"] = separation_verdict(s["conflict"]["mean"], s["agreement"]["mean"])
            s["runs_verdicts"] = {f"run {k}": separation_verdict(float(np.nanmean(conf_run[k][keep])),
                                                                 float(np.nanmean(agr_run[k][keep]))) for k in done}
            s["per_run"] = {f"run {k}": {"conflict": float(np.nanmean(conf_run[k][keep])),
                                         "agreement": float(np.nanmean(agr_run[k][keep]))} for k in done}
            # descriptive: accuracy split by whether frame order favors near
            for typ in ("conflict", "agreement"):
                split = {}
                for fo in (True, False):
                    vals = [float(np.mean([answers[k][f"{ts.scene(r, cond)}-nbhd-{t['id']}"] for k in done]))
                            for r in rooms for t in load_trials(ts.scene(r, cond)).get("nbhd", [])
                            if t["type"] == typ and t["frame_order_favours_near"] == fo]
                    split["frame order favors near" if fo else "frame order favors far"] = {
                        "n": len(vals), "pooled_acc": float(np.mean(vals)) if vals else None}
                s[f"{typ}_by_frame_order"] = split
        if tables:
            for typ in ("conflict", "agreement"):
                M = rule_matrix(cond, "nbhd", typ)[keep]
                best = int(np.nanargmax(np.nanmean(M, 0)))
                s[f"{typ}_best_rule"] = {"rule": names[best], "acc": float(np.nanmean(M[:, best]))}
                s[f"{typ}_pixel_window_max_k16"] = float(np.nanmean(M[:, names.index(PIX_RULE)]))
                if done:
                    s[f"{typ}_margin_over_best_rule"] = ci_block((conf if typ == "conflict" else agr)[keep] - M[:, best])
        out["separation"][cond] = s
    # descriptive: standard trials typed by the neighborhood rule; geometry per condition
    for cond in ts.COND_ORDER:
        typed = {"conflict": [], "agreement": [], "neither": []}
        geo = {}
        for room in rooms:
            sc = ts.scene(room, cond)
            lm = json.load(open(f"{PATHS['e30']}/{sc}/landmarks.json"))
            q = np.load(f"{PATHS['e30']}/{sc}/poses.npz")
            walk, step = q["frame_walk"], q["frame_step"]
            order = {int(w): np.flatnonzero(walk == w)[np.argsort(step[walk == w])] for w in np.unique(walk)}
            for t in load_trials(sc).get("standard", []):
                rows = order[t["walk"]]
                seen = set(lm["probes"][t["probe"]])
                shares = [bool(seen & set(lm["frames"][f])) for f in rows]
                typ = ts.nbhd_type(t["near"], t["far"], shares) or "neither"
                v = float(np.mean([answers[k][f"{sc}-standard-{t['id']}"] for k in done])) if done else np.nan
                typed[typ].append(v)
            if tables:
                for r in tables[sc, "standard"][1]:
                    for b in ("heading gap (deg)", "distance margin (m)", "novelty margin (m)", "visual overlap"):
                        geo.setdefault(b, []).append(r[b])
        n_all = sum(len(v) for v in typed.values())
        out["trials"][cond] = {"standard_by_type": {k: {"n": len(v), "share": len(v) / max(n_all, 1),
                                                        "pooled_acc": float(np.mean(v)) if v and done else None}
                                                    for k, v in typed.items()},
                               "geometry_medians": {b: float(np.median(v)) for b, v in geo.items()}}
    # descriptive: experiment 20's geometry bins per condition, trials pooled, against the condition's own b*
    if tables and done:
        settings = json.load(open(rb.OUT))["settings"]
        edges = settings["visual overlap tertile edges"]["procedural"]
        for cond in ts.COND_ORDER:
            best = names.index(out["conditions"][cond]["best_rule"])
            rows, lora, rule = [], [], []
            for room in rooms:
                sc = ts.scene(room, cond)
                trials, rr = tables[sc, "standard"]
                rows += rr
                lora += lora_vec(trials, sc, "standard", done).mean(0).tolist()
                rule += (comb[sc, "standard"].tolist() if names[best] == "combined" else [r["scores"][best] for r in rr])
            lora, rule = np.array(lora), np.array(rule, float)
            bins = {}
            for key, e in {**rb.BINS, "visual overlap": edges}.items():
                v = np.array([r[key] for r in rows])
                bins[key] = {}
                for lo, hi in zip(e[:-1], e[1:]):
                    sel = (v >= lo) & (v < hi)
                    if sel.any():
                        bins[key][f"{lo:g} to {hi:g}"] = {"n": int(sel.sum()), "lora": float(lora[sel].mean()),
                                                          "best_rule": float(rule[sel].mean())}
            out["trials"][cond]["geometry_bins"] = bins
    out["exp10_by_type"] = exp10_by_type() if not args.test else {"status": "not run in --test"}
    if not args.test:
        out["exp10_reference"] = exp10_reference()
    out["unparsed"] = unparsed_share()
    out["parity"] = parity_summary()
    out["summary"] = summarize(out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    report(out)
    return out


def summarize(out):
    s = {"status": "complete" if len(out["runs_complete"]) == len(RUNS) else f"interim: runs {out['runs_complete']}",
         "rules_status": out["rules"].get("status", "complete"), "parity": out["parity"], "conditions": {},
         "separation": {}, "exp10_by_type": out["exp10_by_type"], "exp10_reference": out.get("exp10_reference"),
         "unparsed": out["unparsed"]}
    for cond, c in out["conditions"].items():
        s["conditions"][cond] = {k: c.get(k) for k in ("n_novel", "lora", "per_run", "best_rule", "best_rule_acc", "margin",
                                                        "delta_vs_control", "verdict", "runs_survive")
                                 if c.get(k) is not None}
    for cond, v in out["separation"].items():
        s["separation"][cond] = {k: v.get(k) for k in ("n_conflict", "n_agreement", "rooms_used", "conflict", "agreement",
                                                        "cost", "midpoint", "verdict", "runs_verdicts", "per_run",
                                                        "conflict_best_rule", "conflict_pixel_window_max_k16",
                                                        "conflict_margin_over_best_rule", "agreement_best_rule",
                                                        "agreement_pixel_window_max_k16",
                                                        "agreement_margin_over_best_rule",
                                                        "conflict_by_frame_order", "agreement_by_frame_order")
                                 if v.get(k) is not None}
    return s


def report(out):
    s = out["summary"]
    print(f"status: {s['status']}; rules: {s['rules_status']}")
    for cond, c in s["conditions"].items():
        line = f"{cond:10s} n={c['n_novel']:5d}"
        if "lora" in c:
            line += f"  LoRA {c['lora']['mean']:.1%} {np.round(c['lora']['ci95'], 3).tolist()}"
        if "best_rule_acc" in c:
            line += f"  b* {c['best_rule_acc']:.1%} ({c['best_rule']})"
        if "margin" in c:
            line += f"  M {c['margin']['mean']:+.3f} {np.round(c['margin']['ci95'], 3).tolist()}"
        if "delta_vs_control" in c:
            line += f"  delta {c['delta_vs_control']['mean']:+.3f}"
        if "verdict" in c:
            line += f"  -> {c['verdict']} ({c['runs_survive']}/{len(out['runs_complete'])} runs)"
        print(line)
    for cond, v in s["separation"].items():
        if "verdict" in v:
            print(f"separation {cond}: conflict {v['conflict']['mean']:.1%} (n={v['n_conflict']}), agreement "
                  f"{v['agreement']['mean']:.1%}, cost {v['cost']['mean']:+.3f} {np.round(v['cost']['ci95'], 3).tolist()} "
                  f"-> {v['verdict']} {v['runs_verdicts']}")


def fetch(args):
    """Copy only the five runs' experiment-30 answer files and their parity files from the vast-3 container."""
    files = [answer_file(k) for k in RUNS] + [f"{ANS}/e30/Qwen3-VL-8B-Instruct+qwen8b-proc-e4-s{k}-roomP5000-256px-max64-e30parity.jsonl"
                                              for k in RUNS]
    cmd = ("export MI_HOST=vast3; source scripts/vast1/common.sh; for f in " + " ".join(files) + "; do "
           "rsync -a -R -e \"$MI_RSYNC_SHELL\" \"$MI_CONTAINER:$MI_REMOTE_ROOT/./$f\" . || echo \"missing $f\"; done")
    subprocess.run(["bash", "-c", cmd], check=True)


# ------------------------------------------------------------------------------------------------ test
def make_test_data(root):
    """Stand-in data: experiment 10's renders, trials and DINOv2 features of the nine rooms used as every condition's
    scene; neighborhood sets = experiment 10's novel trials typed conflict or agreement by the registered rule; synthetic
    answers per run with known accuracies, so that each verdict branch is exercised."""
    from spatialcells.analyze_probes import pixel_feats  # noqa: F401
    rng = np.random.default_rng(0)
    PATHS.update(e30=f"{root}/data/e30", export=f"{root}/export", answers=f"{root}/answers")
    os.makedirs(f"{PATHS['answers']}/e30", exist_ok=True)
    acc = {"control": 0.96, "long-step": 0.92, "low-turn": 0.66, "patrol": 0.80}            # survives / fails / reduced
    nb_acc = {"control": {"conflict": 0.93, "agreement": 0.96}, "patrol": {"conflict": 0.30, "agreement": 0.95}}
    answers = {k: {} for k in RUNS}
    for room in ts.ROOMS:
        data = rb.data_dir(room)
        meta, pmeta = dict(np.load(f"{data}/meta.npz")), dict(np.load(f"{data}/probes_meta.npz"))
        frames = np.load(f"{data}/frames.npy", mmap_mode="r")
        probes = np.load(f"{data}/probes.npy", mmap_mode="r")
        fz = np.load(f"{rb.FEATS}/{room}.npz")
        from spatialcells.room import get_layout
        fl, pl = ts.landmark_sets(meta, pmeta, get_layout(room))
        man = [t for t in rb.trials_of(room) if not t["familiar"]]
        order = rb.walk_order(meta)
        nb = []
        for t in man:
            shares = [bool(pl[t["probe"]] & fl[f]) for f in order[t["walk"]]]
            typ = ts.nbhd_type(t["near"], t["far"], shares)
            if typ:
                nb.append({**t, "type": typ})
        fsums = ts.pixel_sums(frames)
        psums = ts.pixel_sums(probes)
        first = None
        for cond in ts.COND_ORDER:
            sc = ts.scene(room, cond)
            d = f"{PATHS['e30']}/{sc}"
            os.makedirs(d, exist_ok=True)
            os.makedirs(f"{PATHS['export']}/{sc}", exist_ok=True)
            if first is None:
                first = d
                np.savez(f"{d}/poses.npz", **{f"frame_{k}": v for k, v in meta.items()},
                         **{f"probe_{k}": v for k, v in pmeta.items()})
                np.savez(f"{d}/pixels.npz", frame_sums=fsums, probe_sums=psums)
                np.savez(f"{d}/dino.npz", probe_idx=fz["probe_idx"], frame_cls=fz["frame_cls"], frame_patch=fz["frame_patch"],
                         probe_cls=fz["probe_cls"], probe_patch=fz["probe_patch"])
                json.dump({"frames": [sorted(s) for s in fl], "probes": [sorted(s) for s in pl]},
                          open(f"{d}/landmarks.json", "w"))
            else:                                   # the same stand-in scene for every condition
                for f in ("poses.npz", "pixels.npz", "dino.npz", "landmarks.json"):
                    if not os.path.exists(f"{d}/{f}"):
                        os.symlink(os.path.abspath(f"{first}/{f}"), f"{d}/{f}")
            json.dump(man, open(f"{PATHS['export']}/{sc}/standard.json", "w"))
            sets = {"standard": man}
            if cond in ts.NBHD_CONDITIONS:
                json.dump(nb, open(f"{PATHS['export']}/{sc}/nbhd.json", "w"))
                sets["nbhd"] = nb
            for name, trials in sets.items():
                for t in trials:
                    p = acc[cond] if name == "standard" else nb_acc[cond][t["type"]]
                    for k in RUNS:
                        right = rng.random() < p
                        answers[k][f"{sc}-{name}-{t['id']}"] = (t, t["near"] if right else t["far"])
        print(f"{room}: stand-in data written", flush=True)
    for k in RUNS:
        with open(answer_file(k), "w") as f:
            for cid, (t, a) in answers[k].items():
                f.write(json.dumps({"id": cid, "near": t["near"], "far": t["far"], "a": t["a"], "b": t["b"], "answer": a,
                                    "correct": a == t["near"]}) + "\n")


def test(args):
    root = args.test_root
    if not os.path.exists(f"{root}/answers"):
        make_test_data(root)
    else:
        PATHS.update(e30=f"{root}/data/e30", export=f"{root}/export", answers=f"{root}/answers")
    args.out = f"{root}/trajectory_shift_test.json"
    out = analyze(args)
    s = out["summary"]
    want = {"long-step": "survives", "low-turn": "fails", "patrol": "reduced"}
    got = {c: s["conditions"][c].get("verdict") for c in want}
    sep = {c: s["separation"][c].get("verdict") for c in ts.NBHD_CONDITIONS}
    ok = got == want and sep == {"control": "geometric proximity wins", "patrol": "local association"}
    print(f"test verdicts {got} {sep}: {'as expected' if ok else 'NOT as expected'}")
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimulus", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--test-root", default="data/e30_test")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--stimulus-out", default=STIM)
    args = ap.parse_args()
    if args.stimulus:
        stimulus(args)
    elif args.fetch:
        fetch(args)
    elif args.test:
        test(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
