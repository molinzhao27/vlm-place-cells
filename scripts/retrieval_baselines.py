"""Experiment 20 (pre-registered in docs/experiments.md): retrieval baselines on the same two-choice trials.

    .venv-mac/bin/python scripts/retrieval_baselines.py features --scenes primary [--threads 6]
    .venv-mac/bin/python scripts/retrieval_baselines.py features --scenes exp19
    .venv-mac/bin/python scripts/retrieval_baselines.py analyze      # -> results/retrieval_baselines.json

`features` caches, per scene, the 32 x 32 pixel features of every walk frame and of every trial's new view, and
DINOv2-base's CLS token and patch-token mean (224 px, bicubic with antialiasing, ImageNet normalization, as
adjacency_baseline.py and appearance_cue.py; CPU, float32) in data/retrieval_feats/<scene>.npz.

`analyze` scores 149 retrieval rules on every lure-free trial: option similarity, "look-alike, then time"
(adjacency_baseline.py's rule, top 1/3/5), the window maximum and the window vote (k = 2, 4, 8, 16, 32 frames),
each with four features (pixels, DINOv2 CLS z-scored or raw, DINOv2 patch mean) and two frame sets (the 64 the
LoRA prompt shows, as two_choice.present with --max-images 64, or all 256), plus a logistic regression on all of
them trained leave-one-scene-out. Beside them: the four-epoch LoRA runs 3-7 (64-frame walk; per trial, the share
of the five runs that answered correctly) and Opus 5.5 (256-frame walk, low effort). Novel-trial accuracy per
scene, the scene mean with a 95% bootstrap CI over scenes, and the same split by heading gap, distance margin,
visual overlap, novelty margin and (HSSD houses) annotated rooms; then the registered decision rule.
The paper's values are under the JSON's "summary" key (fractions, 0.15 = 15 points). Also written, outside the
registered rule: the same comparison against the best rule that sees only the 64 shown frames, and the window
maximum with k = 64 and 128. Experiment 19's scenes enter automatically; the rule is applied to them once every
four-epoch run has answered them.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from spatialcells.two_choice import present, walk_order  # noqa: E402

FEATS = "data/retrieval_feats"
OUT = "results/retrieval_baselines.json"
P_ROOMS = [f"P{s}" for s in range(5000, 5010)]
APTS = [f"ReplicaApt{i}" for i in range(3)]
HOUSES = [f"HSSD{i}" for i in range(1, 9)]
X19_HOUSES = [f"HSSD{i}" for i in range(9, 27)]
X19_FRL = [f"ReplicaFRL{i}" for i in (0, 1, 3, 4, 5)]
SCENE_SETS = {"primary": ["A", "B"] + P_ROOMS + APTS + HOUSES, "exp19": X19_HOUSES + X19_FRL}
GROUPS = {"procedural": P_ROOMS, "photoreal": APTS + HOUSES, "apartments": APTS, "houses": HOUSES,
          "room A": ["A"], "room B": ["B"], "exp19 all": X19_HOUSES + X19_FRL, "exp19 houses": X19_HOUSES,
          "exp19 FRL apartments": X19_FRL}
RULE_GROUPS = ["procedural", "photoreal"]                 # the decision rule's two groups
COMBINED_TRAIN = {"procedural": P_ROOMS, "photoreal": APTS + HOUSES}    # leave-one-scene-out within these
FEATURES = ["pixels", "dino-cls", "dino-cls-raw", "dino-patch"]
FRAME_SETS = ["64 shown", "all 256"]
KS = [2, 4, 8, 16, 32]
N_BOOT, SEED, MARGIN, MIN_BIN = 10_000, 0, 0.15, 30
LORA4 = [3, 4, 5, 6, 7]
EXPLORE_KS = [64, 128]                                    # unregistered, reported apart from the verdict
BINS = {"heading gap (deg)": [90, 120, 150, 180.0001], "distance margin (m)": [2, 3, 4, np.inf],
        "novelty margin (m)": [1, 1.25, 1.5, 2, np.inf]}


def trial_set(scene):
    return "dissimilar" if scene == "A" else f"room{scene}"


def data_dir(scene):
    return "data/walks" if scene == "A" else f"data/walks-room{scene}"


def trials_of(scene):
    path = "results/export/dissimilar/trials.json" if scene == "A" else f"results/export/dissimilar-room{scene}/trials.json"
    t = json.load(open(path))
    return t["trials"] if isinstance(t, dict) else t


# ----------------------------------------------------------------------------------------------- features
def features(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel
    from spatialcells.analyze_probes import pixel_feats
    torch.set_num_threads(args.threads)
    model = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def dino(images, batch=32):
        cls, patch = [], []
        for i in range(0, len(images), batch):
            x = torch.from_numpy(np.array(images[i:i + batch])).permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, size=224, mode="bicubic", antialias=True)
            h = model(pixel_values=(x - mean) / std).last_hidden_state
            cls.append(h[:, 0].float().numpy())
            patch.append(h[:, 1:].mean(1).float().numpy())
        return np.concatenate(cls), np.concatenate(patch)

    outdir = args.out or FEATS
    os.makedirs(outdir, exist_ok=True)
    scenes = [s for name in args.scenes for s in SCENE_SETS.get(name, [name])]
    for scene in scenes:
        path = f"{outdir}/{scene}.npz"
        if os.path.exists(path):
            print(f"{scene}: cached", flush=True)
            continue
        t0 = time.time()
        frames = np.load(f"{data_dir(scene)}/frames.npy", mmap_mode="r")[:args.limit]
        probes = np.load(f"{data_dir(scene)}/probes.npy", mmap_mode="r")
        pidx = np.array(sorted({t["probe"] for t in trials_of(scene)}))[:args.limit]
        wcls, wpatch = [], []
        for i in range(0, len(frames), 1024):
            c, p = dino(frames[i:i + 1024])
            wcls.append(c)
            wpatch.append(p)
            print(f"  {scene}: {min(i + 1024, len(frames))}/{len(frames)} frames, {time.time() - t0:.0f} s", flush=True)
        pcls, ppatch = dino(probes[pidx])
        np.savez(path + ".tmp.npz", probe_idx=pidx, frame_pixels=pixel_feats(frames), probe_pixels=pixel_feats(probes[pidx]),
                 frame_cls=np.concatenate(wcls), frame_patch=np.concatenate(wpatch), probe_cls=pcls, probe_patch=ppatch)
        os.replace(path + ".tmp.npz", path)
        print(f"{scene}: {len(frames)} frames, {len(pidx)} probes in {time.time() - t0:.0f} s -> {path}", flush=True)


# ----------------------------------------------------------------------------------------------- rules
def variant_names():
    names = [f"option similarity | {f}" for f in FEATURES]
    names += [f"look-alike then time | {f} | {s} | top {m}" for f in FEATURES for s in FRAME_SETS for m in (1, 3, 5)]
    names += [f"window maximum | {f} | {s} | k {k}" for f in FEATURES for s in FRAME_SETS for k in KS]
    names += [f"window vote | {f} | {s} | top {m} | k {k}" for f in FEATURES for s in FRAME_SETS for m in (1, 5) for k in KS]
    return names


def unit_feats(z):
    """Per feature: (walk-frame matrix, probe matrix), unit rows; z-scored over the scene's walk frames except -raw."""
    out = {}
    for name, (W, P) in {"pixels": (z["frame_pixels"], z["probe_pixels"]), "dino-cls": (z["frame_cls"], z["probe_cls"]),
                         "dino-cls-raw": (z["frame_cls"], z["probe_cls"]), "dino-patch": (z["frame_patch"], z["probe_patch"])}.items():
        if name.endswith("-raw"):
            unit = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
        else:                                       # as adjacency_baseline.py
            mu, sd = W.mean(0), W.std(0) + 1e-6
            unit = lambda X, mu=mu, sd=sd: (X - mu) / sd / np.linalg.norm((X - mu) / sd, axis=1, keepdims=True)
        out[name] = (unit(W), unit(P))
    return out


def pick(near_score, far_score):
    return 1.0 if near_score > far_score else 0.0 if near_score < far_score else 0.5


def score_trial(t, rows, sims_of, n):
    """All 148 hand rules for trial t: (score for near, signed a-minus-b value), in variant_names() order.
    rows: the walk's frame rows in step order; sims_of(feature, frame numbers) -> similarities to the new view."""
    args64 = argparse.Namespace(no_walk=False, max_images=64, shuffle_walk=False, relabel=False, reverse_walk=False,
                                window_shuffle=None, renumber=False)
    shown, lab = present(t, n, args64)
    assert all(lab[f] == f for f in shown)          # ordered: labels are the real frame numbers
    sets = {"64 shown": np.array(sorted(shown)), "all 256": np.arange(1, n + 1)}
    near, far, a, b = t["near"], t["far"], t["a"], t["b"]
    sc, sg = [], []
    cache = {(f, s): sims_of(f, S) for f in FEATURES for s, S in sets.items()}
    for f in FEATURES:                              # option similarity
        S = sets["all 256"]
        sim = cache[f, "all 256"]
        s = {o: sim[o - 1] for o in (a, b)}
        sc.append(pick(s[near], s[far]))
        sg.append(s[a] - s[b])
    for f in FEATURES:                              # look-alike, then time (adjacency_baseline.py; ties go to far)
        for sname in FRAME_SETS:
            S, sim = sets[sname], cache[f, sname]
            order = np.argsort(-sim)
            for m in (1, 3, 5):
                ref = np.median(S[order[:m]])
                sc.append(1.0 if abs(near - ref) < abs(far - ref) else 0.0)
                sg.append(abs(b - ref) - abs(a - ref))
    for f in FEATURES:                              # window maximum
        for sname in FRAME_SETS:
            S, sim = sets[sname], cache[f, sname]
            for k in KS:
                s = {o: sim[np.abs(S - o) <= k].max() for o in (a, b)}
                sc.append(pick(s[near], s[far]))
                sg.append(s[a] - s[b])
    for f in FEATURES:                              # window vote
        for sname in FRAME_SETS:
            S, sim = sets[sname], cache[f, sname]
            order = np.argsort(-sim)
            for m in (1, 5):
                top = S[order[:m]]
                for k in KS:
                    c = {o: int(np.sum(np.abs(top - o) <= k)) for o in (a, b)}
                    sc.append(pick(c[near], c[far]))
                    sg.append(c[a] - c[b])
    return sc, sg, shown


# ----------------------------------------------------------------------------------------------- models
def model_files(scene):
    ts = trial_set(scene)
    files = {}
    for k in LORA4:
        base = f"results/two_choice/Qwen3-VL-8B-Instruct+qwen8b-proc-e4-s{k}-{ts}-256px-max64"
        found = [p for p in (base + ".jsonl", base + "-mps.jsonl") if os.path.exists(p)]
        assert len(found) <= 1, found
        files[f"LoRA run {k}"] = found[0] if found else None
    p = f"results/claude/claude-opus-5-5-{ts}-low.jsonl"
    files["Opus"] = p if os.path.exists(p) else None
    return files


def load_answers(path, man):
    """id -> 1/0 (unparsed counts as wrong); None if the file does not cover the manifest."""
    rows = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        t = man[r["id"]]
        assert (r["near"], r["far"], r["familiar"]) == (t["near"], t["far"], t["familiar"]), (path, r["id"])
        assert r["id"] not in rows, (path, "duplicate", r["id"])
        assert bool(r["correct"]) == (r["answer"] == r["near"]), (path, r["id"])
        rows[r["id"]] = float(bool(r["correct"]))
    return rows if set(rows) == set(man) else None


# ----------------------------------------------------------------------------------------------- per scene
def hssd_rooms(scene):
    """Point (x, y) -> index of the annotated region containing (x, -y) (habitat_scene.to_habitat), else None."""
    from matplotlib.path import Path
    sid = json.load(open(f"{data_dir(scene)}/scene.json"))["scene"]
    regions = json.load(open(f"data/hssd_regions/{sid}.semantic_config.json"))["region_annotations"]
    paths = [Path(np.array(r["poly_loop"])[:, [0, 2]]) for r in regions]

    def room(x, y):
        hit = [j for j, p in enumerate(paths) if p.contains_point((x, -y))]
        assert len(hit) <= 1, (scene, x, y, hit)
        return hit[0] if hit else None
    return room, regions, paths


def scene_table(scene, names):
    z = np.load(f"{FEATS}/{scene}.npz")
    U = unit_feats(z)
    prow = {int(p): i for i, p in enumerate(z["probe_idx"])}
    meta = dict(np.load(f"{data_dir(scene)}/meta.npz"))
    pmeta = dict(np.load(f"{data_dir(scene)}/probes_meta.npz"))
    order = walk_order(meta)
    trials = trials_of(scene)
    man = {t["id"]: t for t in trials}
    answers = {}
    for name, path in model_files(scene).items():
        answers[name] = load_answers(path, man) if path else None
    room = None
    if scene.startswith("HSSD"):
        room, regions, paths = hssd_rooms(scene)
        inside = np.array([sum(p.contains_point((x, -y)) for p in paths) for x, y in zip(meta["x"], meta["y"])])
        room_check = {"walk_frames_in_one_room": float(np.mean(inside == 1)), "in_two_or_more": float(np.mean(inside >= 2))}
    rows_out = []
    for t in trials:
        w, p = t["walk"], t["probe"]
        rows = order[w]
        n = len(rows)
        assert pmeta["walk"][p] == w

        def sims_of(f, S, rows=rows, p=p):
            W, P = U[f]
            return W[rows[S - 1]] @ P[prow[p]]
        sc, sg, shown = score_trial(t, rows, sims_of, n)
        explore = []                                # unregistered sensitivity: window maximum with k = 64, 128
        for f in FEATURES:
            for sname in FRAME_SETS:
                S = np.array(sorted(shown)) if sname == "64 shown" else np.arange(1, n + 1)
                sim = sims_of(f, S)
                for k in EXPLORE_KS:
                    v = {o: sim[np.abs(S - o) <= k].max() for o in (t["near"], t["far"])}
                    explore.append(pick(v[t["near"]], v[t["far"]]))
        raw = U["dino-cls-raw"]
        overlap = float(np.max(raw[0][rows[np.array(sorted(shown)) - 1]] @ raw[1][prow[p]]))
        d = np.hypot(meta["x"][rows] - pmeta["x"][p], meta["y"][rows] - pmeta["y"][p])
        dh = np.abs((meta["theta"][rows] - pmeta["theta"][p] + np.pi) % (2 * np.pi) - np.pi)
        same = dh < np.pi / 4
        nov_margin = float(d[same].min()) if same.any() else float("inf")
        fn, ff = rows[t["near"] - 1], rows[t["far"] - 1]
        for fi, dist in ((fn, t["near_dist_m"]), (ff, t["far_dist_m"])):          # the manifest's distances
            assert abs(np.hypot(meta["x"][fi] - pmeta["x"][p], meta["y"][fi] - pmeta["y"][p]) - dist) < 2e-3, (scene, t["id"])
        assert (nov_margin >= 1.0 - 1e-6) == (not t["familiar"]), (scene, t["id"], nov_margin)
        rec = {"scene": scene, "id": t["id"], "familiar": bool(t["familiar"]), "scores": sc, "signed": sg, "explore": explore,
               "a_is_near": t["a"] == t["near"],
               "heading gap (deg)": float(t["near_heading_diff_deg"]),
               "distance margin (m)": float(t["far_dist_m"] - t["near_dist_m"]),
               "novelty margin (m)": nov_margin, "visual overlap": overlap}
        if room is not None:
            rp, rn, rf = room(pmeta["x"][p], pmeta["y"][p]), room(meta["x"][fn], meta["y"][fn]), room(meta["x"][ff], meta["y"][ff])
            rec["rooms"] = ("unlabeled" if None in (rp, rn, rf) else "options in the same room" if rn == rf
                            else "different rooms, new view in the near option's room" if rp == rn
                            else "different rooms, new view elsewhere")
        rows_out.append(rec)
    model = {}
    for name, ans in answers.items():
        model[name] = None if ans is None else np.array([ans[r["id"]] for r in rows_out])
    runs = [model[f"LoRA run {k}"] for k in LORA4]
    model["LoRA four-epoch mean"] = np.mean(runs, 0) if all(r is not None for r in runs) else None
    info = {"trials": len(trials), "novel": sum(not t["familiar"] for t in trials),
            "files": {k: v for k, v in model_files(scene).items()},
            "complete": {k: v is not None for k, v in model.items()}}
    if room is not None:
        info["room_annotation_check"] = room_check
    return rows_out, model, info


# ----------------------------------------------------------------------------------------------- statistics
def boot_scene_mean(per_scene, rng_seed=SEED):
    x = np.asarray(per_scene, float)
    rng = np.random.default_rng(rng_seed)
    m = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def pooled_cluster(scene_ids, values_by_method, mask, rng_seed=SEED):
    """Accuracy pooled over the masked trials, with a 95% CI from resampling scenes (the same draws for every
    method); values_by_method: name -> per-trial score array aligned with scene_ids."""
    scenes = sorted(set(scene_ids[mask]))
    rng = np.random.default_rng(rng_seed)
    idx = {s: np.flatnonzero(mask & (scene_ids == s)) for s in scenes}
    draws = rng.integers(0, len(scenes), (N_BOOT, len(scenes)))
    cnts = np.array([len(idx[s]) for s in scenes])
    out = {}
    for name, v in values_by_method.items():
        sums = np.array([v[idx[s]].sum() for s in scenes])
        boot = sums[draws].sum(1) / np.maximum(cnts[draws].sum(1), 1)
        out[name] = {"acc": float(v[mask].mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    return out, len(scenes)


def combined_scores(tables, names, train_scenes, test_scene):
    from sklearn.linear_model import LogisticRegression
    X = np.array([r["signed"] for s in train_scenes for r in tables[s] if not r["familiar"]], float)
    y = np.array([r["a_is_near"] for s in train_scenes for r in tables[s] if not r["familiar"]], float)
    X, y = np.concatenate([X, -X]), np.concatenate([y, 1 - y])
    sd = X.std(0)
    sd[sd == 0] = 1
    clf = LogisticRegression(C=1.0, fit_intercept=False, max_iter=10_000).fit(X / sd, y)
    out = []
    for r in tables[test_scene]:
        z = float(clf.decision_function((np.array(r["signed"], float) / sd)[None])[0])
        pick_a = 1.0 if z > 0 else 0.0 if z < 0 else 0.5
        out.append(pick_a if r["a_is_near"] else 1 - pick_a)
    return np.array(out)


# ----------------------------------------------------------------------------------------------- checks
def checks(tables, names):
    res = {}
    ix = {n: i for i, n in enumerate(names)}
    pres = json.load(open("results/adjacency_presentations.json"))
    for feat, key_feat, exact in (("pixels", "pixels", True), ("dino-cls", "dinov2", False)):
        for m in (1, 3):
            key = f"{key_feat}, ordered, time from labels, top {m}"
            j = ix[f"look-alike then time | {feat} | 64 shown | top {m}"]
            worst = 0
            for s in P_ROOMS:
                for c, fam in (("novel", False), ("familiar", True)):
                    rs = [r for r in tables[s] if r["familiar"] == fam]
                    mine = sum(r["scores"][j] for r in rs)
                    theirs = pres[key]["per_room"][f"room{s}"][c] * len(rs)
                    worst = max(worst, abs(mine - theirs))
            res[f"P rooms, {key} (trials off, worst room)"] = {"max_trials_off": float(round(worst, 6)), "exact_required": exact,
                                                                "pass": worst < 1e-6 if exact else worst <= 2}
    adj = json.load(open("results/adjacency_baseline.json"))
    for feat, key_feat, exact in (("pixels", "pixels", True), ("dino-cls", "dinov2", False)):
        for sname in FRAME_SETS:
            for m in (1, 3):
                key = f"{key_feat}, {sname}, top {m}"
                j = ix[f"look-alike then time | {feat} | {sname} | top {m}"]
                worst = 0
                for c, fam in (("novel", False), ("familiar", True)):
                    rs = [r for r in tables["A"] if r["familiar"] == fam]
                    worst = max(worst, abs(sum(r["scores"][j] for r in rs) - adj[key][c] * len(rs)))
                res[f"room A, {key} (trials off)"] = {"max_trials_off": float(round(worst, 6)), "exact_required": exact,
                                                      "pass": worst < 1e-6 if exact else worst <= 2}
    cue = json.load(open("results/appearance_cue.json"))
    j = ix["option similarity | dino-cls-raw"]
    worst = 0
    for s in ["A", "B"] + P_ROOMS:
        for c, fam in (("novel", False), ("familiar", True)):
            rs = [r for r in tables[s] if r["familiar"] == fam]
            worst = max(worst, abs(sum(r["scores"][j] for r in rs) - cue["rooms"][s][c]["dino_favours_near"] * len(rs)))
    res["experiment 13d DINOv2 option cue, rooms A, B, P (trials off, worst room)"] = {"max_trials_off": float(round(worst, 6)),
                                                                                       "pass": worst <= 1}
    return res


# ----------------------------------------------------------------------------------------------- analysis
def analyze(args):
    names = variant_names()
    assert len(names) == 148
    scenes = [s for g in GROUPS.values() for s in g]
    scenes = list(dict.fromkeys(scenes))
    missing = [s for s in scenes if not os.path.exists(f"{FEATS}/{s}.npz")]
    scenes = [s for s in scenes if s not in missing]
    if missing:
        print("features missing (left out):", missing, flush=True)
    tables, models, infos = {}, {}, {}
    for s in scenes:
        t0 = time.time()
        tables[s], models[s], infos[s] = scene_table(s, names)
        print(f"{s}: {len(tables[s])} trials, models complete: "
              f"{', '.join(k for k, v in infos[s]['complete'].items() if v) or 'none'} ({time.time() - t0:.0f} s)", flush=True)
    chk = checks(tables, names) if all(s in tables for s in ["A", "B"] + P_ROOMS) else {}
    for k, v in chk.items():
        print(f"check {'PASS' if v['pass'] else 'FAIL'}: {k}: {v['max_trials_off']}", flush=True)
    if chk and not all(v["pass"] for v in chk.values()) and not args.ignore_checks:
        raise SystemExit("a reproduction check failed; fix the code before reading any result")

    # the combined rule, leave-one-scene-out within the rule groups; other scenes use all of the matching group
    comb = {}
    for s in scenes:
        if s in COMBINED_TRAIN["procedural"] or s in ("A", "B"):
            train = [x for x in COMBINED_TRAIN["procedural"] if x != s and x in tables]
        else:
            train = [x for x in COMBINED_TRAIN["photoreal"] if x != s and x in tables]
        comb[s] = combined_scores(tables, names, train, s)
    all_names = names + ["combined (logistic regression, leave-one-scene-out)"]
    score = {s: np.column_stack([np.array([r["scores"] for r in tables[s]]), comb[s]]) for s in scenes}

    out = {"settings": {"registration": "docs/experiments.md, Experiment 20", "variants": len(all_names),
                        "features": FEATURES, "frame_sets": FRAME_SETS, "window_k": KS,
                        "models": {"LoRA four-epoch mean": "runs 3-7 (qwen8b-proc-e4-s3..s7), 64-frame walk; per trial the "
                                                           "share of the five runs that answered correctly",
                                   "Opus": "claude-opus-5-5, 256-frame walk, low effort"},
                        "ties": "0.5, except look-alike then time (ties go to the far option, as adjacency_baseline.py)",
                        "unparsed": "wrong", "bootstrap": f"{N_BOOT} resamples over scenes, seed {SEED}",
                        "bins": {k: [float(e) for e in v] for k, v in BINS.items()},
                        "visual overlap": "raw DINOv2 CLS cosine of the new view with the most similar of the 64 shown "
                                          "frames; tertiles of the group's novel trials (rooms A and B: procedural edges; "
                                          "experiment 19: photoreal edges)",
                        "rooms": "HSSD region annotations (data/hssd_regions), point (x, -y); Replica has no room labels",
                        "decision": f"best variant b* per group by scene-mean novel accuracy; LoRA four-epoch mean minus b* "
                                    f">= {MARGIN * 100:.0f} points overall (scene mean of differences) and in b*'s best bin "
                                    f"(>= {MIN_BIN} novel trials) of every breakdown; both groups"},
           "checks": chk, "scenes": infos, "features_missing": missing, "groups": {}, "trials": {}}

    overlap_edges = {}
    for g in ("procedural", "photoreal"):
        v = [r["visual overlap"] for s in GROUPS[g] if s in tables for r in tables[s] if not r["familiar"]]
        overlap_edges[g] = [float(-np.inf)] + [float(e) for e in np.percentile(v, [100 / 3, 200 / 3])] + [float(np.inf)]
    out["settings"]["visual overlap tertile edges"] = overlap_edges

    best_of, best64_of = {}, {}
    for g, members in GROUPS.items():
        members = [s for s in members if s in tables]
        if not members:
            continue
        res = {"scenes": members, "trials": sum(len(tables[s]) for s in members),
               "novel": sum(sum(not r["familiar"] for r in tables[s]) for s in members)}
        nov_scenes = [s for s in members if any(not r["familiar"] for r in tables[s])]
        res["scenes_with_novel_trials"] = nov_scenes
        # rules: per scene novel accuracy, scene mean, pooled
        V = {}
        for j, name in enumerate(all_names):
            per = {s: float(np.mean([score[s][i, j] for i, r in enumerate(tables[s]) if not r["familiar"]])) for s in nov_scenes}
            m, ci = boot_scene_mean(list(per.values())) if per else (None, None)
            allv = np.concatenate([score[s][:, j] for s in members])
            fam = np.array([r["familiar"] for s in members for r in tables[s]])
            V[name] = {"novel_scene_mean": m, "ci95": ci,
                       "novel_pooled": float(allv[~fam].mean()) if (~fam).any() else None,
                       "familiar_pooled": float(allv[fam].mean()) if fam.any() else None,
                       "all_pooled": float(allv.mean()), "per_scene_novel": per}
        ranked = sorted((n for n in all_names if V[n]["novel_scene_mean"] is not None), key=lambda n: -V[n]["novel_scene_mean"])
        res["rules"] = V
        res["ranking_top10"] = [(n, round(V[n]["novel_scene_mean"], 4)) for n in ranked[:10]]
        res["best_variant_own"] = ranked[0] if ranked else None
        res["best_per_family"] = {fam: max((n for n in ranked if n.startswith(fam)), key=lambda n: V[n]["novel_scene_mean"])
                                  for fam in ("option similarity", "look-alike then time", "window maximum", "window vote", "combined")
                                  if any(n.startswith(fam) for n in ranked)}
        shown64 = [n for n in ranked if "64 shown" in n or n.startswith("option similarity")]
        res["best_variant_64_shown"] = shown64[0] if shown64 else None
        if nov_scenes:
            ex_names = [f"window maximum | {f} | {sn} | k {k} (unregistered)" for f in FEATURES for sn in FRAME_SETS for k in EXPLORE_KS]
            ex = {}
            for j, name in enumerate(ex_names):
                per = [float(np.mean([r["explore"][j] for r in tables[s] if not r["familiar"]])) for s in nov_scenes]
                ex[name] = boot_scene_mean(per)[0]
            top = max(ex, key=ex.get)
            res["exploratory larger windows"] = {"note": "not registered; not part of the verdict", "best": top,
                                                 "best_novel_scene_mean": ex[top], "all": ex}
        # models
        M = {}
        for name in ["LoRA four-epoch mean", "Opus"] + [f"LoRA run {k}" for k in LORA4]:
            have = [s for s in members if models[s].get(name) is not None]
            per = {s: float(models[s][name][[not r["familiar"] for r in tables[s]]].mean()) for s in have if s in nov_scenes}
            m, ci = boot_scene_mean(list(per.values())) if per else (None, None)
            allv = np.concatenate([models[s][name] for s in have]) if have else np.array([])
            fam = np.array([r["familiar"] for s in have for r in tables[s]])
            M[name] = {"scenes": have, "complete_in_group": len(have) == len(members), "novel_scene_mean": m, "ci95": ci,
                       "novel_pooled": float(allv[~fam].mean()) if len(allv) and (~fam).any() else None,
                       "familiar_pooled": float(allv[fam].mean()) if len(allv) and fam.any() else None,
                       "all_pooled": float(allv.mean()) if len(allv) else None, "per_scene_novel": per}
        res["models"] = M
        # the reference best variant: the rule group's own b*; subgroups and other sets use their parent's
        parent = {"apartments": "photoreal", "houses": "photoreal", "room A": "procedural", "room B": "procedural",
                  "exp19 all": "photoreal", "exp19 houses": "photoreal", "exp19 FRL apartments": "photoreal"}.get(g, g)
        b = best_of.get(parent, res["best_variant_own"]) if g not in RULE_GROUPS else res["best_variant_own"]
        if g in RULE_GROUPS:
            best_of[g] = b
        res["best_variant"] = b
        res["best_variant_from"] = parent if g not in RULE_GROUPS else "own"
        jb = all_names.index(b)

        def against(jb, members=members, nov_scenes=nov_scenes, parent=parent):
            r2 = {}
            # differences per scene (novel): LoRA mean and Opus minus b*
            for mname in ("LoRA four-epoch mean", "Opus"):
                have = [s for s in nov_scenes if models[s].get(mname) is not None]
                if not have:
                    continue
                diffs = {s: float(np.mean([models[s][mname][i] - score[s][i, jb] for i, r in enumerate(tables[s]) if not r["familiar"]]))
                         for s in have}
                m, ci = boot_scene_mean(list(diffs.values()))
                r2[f"{mname} minus best variant (novel)"] = {"scene_mean": m, "ci95": ci, "per_scene": diffs,
                                                              "scenes_up": sum(d > 0 for d in diffs.values()),
                                                              "scenes_down": sum(d < 0 for d in diffs.values())}
            # bins (novel trials)
            nov = [(s, i, r) for s in members for i, r in enumerate(tables[s]) if not r["familiar"]]
            if nov:
                sid = np.array([s for s, _, _ in nov])
                vals = {"best variant": np.array([score[s][i, jb] for s, i, _ in nov])}
                for mname in ("LoRA four-epoch mean", "Opus"):
                    if all(models[s].get(mname) is not None for s in members):
                        vals[mname] = np.array([models[s][mname][i] for s, i, _ in nov])
                allrules = np.array([score[s][i] for s, i, _ in nov])
                edges = overlap_edges["procedural" if parent == "procedural" else "photoreal"]
                breakdowns = {k: (np.array([r[k] for _, _, r in nov]), BINS[k]) for k in BINS}
                breakdowns["visual overlap"] = (np.array([r["visual overlap"] for _, _, r in nov]), edges)
                bins_out = {}
                for bk, (x, e) in breakdowns.items():
                    bins_out[bk] = {}
                    for lo, hi in zip(e[:-1], e[1:]):
                        mask = (x >= lo) & (x < hi)
                        label = f"[{lo:.3g}, {hi:.3g})"
                        bins_out[bk][label] = bin_stats(sid, vals, allrules, all_names, mask)
                if any("rooms" in r for _, _, r in nov):
                    cats = np.array([r.get("rooms", "no room labels (Replica)") for _, _, r in nov])
                    bins_out["rooms"] = {}
                    for c in ["options in the same room", "different rooms, new view in the near option's room",
                              "different rooms, new view elsewhere", "unlabeled", "no room labels (Replica)"]:
                        if (cats == c).any():
                            bins_out["rooms"][c] = bin_stats(sid, vals, allrules, all_names, cats == c)
                    diff = np.isin(cats, ["different rooms, new view in the near option's room", "different rooms, new view elsewhere"])
                    if diff.any():
                        bins_out["rooms"]["options in different rooms (both kinds)"] = bin_stats(sid, vals, allrules, all_names, diff)
                r2["bins"] = bins_out
                # models on the trials b* gets right vs wrong
                sb = vals["best variant"]
                r2["models by best variant right or wrong"] = {
                    mname: {"best right": {"n": int((sb == 1).sum()), "acc": float(v[sb == 1].mean()) if (sb == 1).any() else None},
                            "best wrong": {"n": int((sb == 0).sum()), "acc": float(v[sb == 0].mean()) if (sb == 0).any() else None},
                            "tie": int((sb == 0.5).sum())}
                    for mname, v in vals.items() if mname != "best variant"}
            return r2
        res.update(against(jb))
        # secondary, not registered: the same comparison against the best rule that sees only the 64 frames the
        # LoRA model saw (the parent group's, as for b*)
        b64 = res["best_variant_64_shown"] if g in RULE_GROUPS else best64_of.get(parent)
        if g in RULE_GROUPS:
            best64_of[g] = b64
        if b64 is not None:
            sec = against(all_names.index(b64))
            sec["best_variant"] = b64
            sec["rule"] = decide({**sec, "models": res["models"]}, g)
            res["secondary: against the best 64-shown rule (not registered)"] = sec
        if g in RULE_GROUPS or g.startswith("exp19") or g in ("room A", "room B", "apartments", "houses"):
            res["rule"] = decide(res, g)
        out["groups"][g] = res

    # verdict over the two rule groups
    rg = [out["groups"][g]["rule"] for g in RULE_GROUPS if g in out["groups"]]
    out["verdict"] = {
        "groups": {g: out["groups"][g]["rule"]["holds"] for g in RULE_GROUPS if g in out["groups"]},
        "retrieval account rejected": bool(len(rg) == 2 and all(r["holds"] for r in rg))}
    out = {"summary": summary(out), **out}
    # per-trial table (covariates, b*'s score, models)
    for s in scenes:
        g = "procedural" if s in P_ROOMS or s in ("A", "B") else "photoreal"
        jb = all_names.index(best_of[g]) if g in best_of else None
        out["trials"][s] = [{"id": r["id"], "familiar": r["familiar"], **{k: (round(r[k], 4) if isinstance(r[k], float) else r[k])
                             for k in list(BINS) + ["visual overlap"] + (["rooms"] if "rooms" in r else [])},
                             "best_variant": None if jb is None else float(score[s][i, jb]),
                             **{m: (None if models[s].get(m) is None else round(float(models[s][m][i]), 3))
                                for m in ("LoRA four-epoch mean", "Opus")}}
                            for i, r in enumerate(tables[s])]
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    report(out)
    print(f"-> {OUT}")


GROUP_KEYS = {"procedural": "procedural", "photoreal": "photoreal", "apartments": "apartments", "houses": "houses",
              "room A": "room_A", "room B": "room_B", "exp19 all": "exp19_all", "exp19 houses": "exp19_houses",
              "exp19 FRL apartments": "exp19_frl"}
BREAKDOWN_KEYS = {"heading gap (deg)": "heading_gap", "distance margin (m)": "distance_margin",
                  "novelty margin (m)": "novelty_margin", "visual overlap": "visual_overlap", "rooms": "rooms"}
ROOM_KEYS = {"options in the same room": "same_room", "different rooms, new view in the near option's room":
             "different_rooms_view_with_near", "different rooms, new view elsewhere": "different_rooms_view_elsewhere",
             "options in different rooms (both kinds)": "different_rooms_all", "unlabeled": "unlabeled",
             "no room labels (Replica)": "replica_no_labels"}


def bin_key(breakdown, label, i):
    if breakdown == "visual overlap":
        return ["low", "mid", "high"][i]
    if breakdown == "rooms":
        return ROOM_KEYS[label]
    lo, hi = label.strip("[)").split(", ")
    return f"{float(lo):g}+" if hi == "inf" else f"{float(lo):g}-{float(hi):g}"


def summary(out):
    """The values a paper would cite, under short stable keys. Every accuracy and difference is a fraction (0-1);
    0.15 = 15 points. mean = mean over scenes (novel trials); ci95 = bootstrap over scenes; bins pool trials."""
    ci = lambda m, c: None if m is None else {"mean": m, "ci95": c}
    asdiff = lambda x: None if x is None else {"diff": x["acc"], "ci95": x["ci95"]}
    s = {"units": "fractions of trials (0-1); differences in fractions (0.15 = 15 points)",
         "verdict": {"procedural": out["verdict"]["groups"].get("procedural"), "photoreal": out["verdict"]["groups"].get("photoreal"),
                     "retrieval_account_rejected": out["verdict"]["retrieval account rejected"]}}
    for g, res in out["groups"].items():
        b = res["best_variant"]
        M = res["models"]
        dl = res.get("LoRA four-epoch mean minus best variant (novel)")
        do = res.get("Opus minus best variant (novel)")
        e = {"best_variant": b, "best_variant_chosen_in": res["best_variant_from"], "n_scenes": len(res["scenes"]),
             "n_novel": res["novel"],
             "novel": {"best_variant": ci(res["rules"][b]["novel_scene_mean"], res["rules"][b]["ci95"]),
                       "lora4": ci(M["LoRA four-epoch mean"]["novel_scene_mean"], M["LoRA four-epoch mean"]["ci95"]),
                       "opus": ci(M["Opus"]["novel_scene_mean"], M["Opus"]["ci95"]),
                       "lora4_minus_best": ci(dl["scene_mean"], dl["ci95"]) if dl else None,
                       "opus_minus_best": ci(do["scene_mean"], do["ci95"]) if do else None},
             "all_trials_pooled": {"best_variant": res["rules"][b]["all_pooled"], "lora4": M["LoRA four-epoch mean"]["all_pooled"],
                                   "opus": M["Opus"]["all_pooled"]},
             "best_per_family": {fam.replace(" ", "_").replace(",", ""): {"variant": n, "novel_mean": res["rules"][n]["novel_scene_mean"]}
                                 for fam, n in res["best_per_family"].items()},
             "best_variant_64_shown": {"variant": res["best_variant_64_shown"],
                                       "novel": ci(res["rules"][res["best_variant_64_shown"]]["novel_scene_mean"],
                                                   res["rules"][res["best_variant_64_shown"]]["ci95"])} if res.get("best_variant_64_shown") else None,
             "models_by_best_variant_right_or_wrong": res.get("models by best variant right or wrong"),
             "exploratory_larger_windows": None if "exploratory larger windows" not in res else
                 {"note": "not registered", "best": res["exploratory larger windows"]["best"],
                  "novel_mean": res["exploratory larger windows"]["best_novel_scene_mean"]},
             "lora4_complete": M["LoRA four-epoch mean"]["complete_in_group"], "opus_complete": M["Opus"]["complete_in_group"],
             "bins": {}}
        for bk, bins in res.get("bins", {}).items():
            e["bins"][BREAKDOWN_KEYS[bk]] = {}
            for i, (lab, st) in enumerate(bins.items()):
                if st["n"] == 0:
                    continue
                e["bins"][BREAKDOWN_KEYS[bk]][bin_key(bk, lab, i)] = {
                    "n": st["n"], "n_scenes": st["scenes"], "range": lab,
                    "best_variant": st["best variant"], "lora4": st.get("LoRA four-epoch mean"), "opus": st.get("Opus"),
                    "lora4_minus_best": asdiff(st.get("LoRA four-epoch mean minus best variant")),
                    "opus_minus_best": asdiff(st.get("Opus minus best variant")),
                    "best_variant_for_this_bin_alone": st["best variant for this bin alone"]}
        r = res.get("rule", {})
        if r.get("holds") is not None:
            e["rule"] = {"holds": r["holds"], "counts_for_verdict": r["counts_for_verdict"],
                         "overall": {"lora4_minus_best": r["overall difference"], "ci95": r["overall_ci95"], "holds": r["overall holds"]},
                         "bins": {BREAKDOWN_KEYS[bk]: (None if v.get("holds") is None else
                                  {"bin": bin_key(bk, v["best bin"], list(res["bins"][bk]).index(v["best bin"])), "n": v["n"],
                                   "best_variant": v["best variant acc"], "lora4": v["LoRA four-epoch mean"],
                                   "lora4_minus_best": v["difference"], "ci95": v["difference_ci95"], "holds": v["holds"]})
                                  for bk, v in r["bins"].items()}}
        else:
            e["rule"] = {"holds": None, "reason": r.get("reason")}
        sec = res.get("secondary: against the best 64-shown rule (not registered)")
        if sec is not None:
            d2 = sec.get("LoRA four-epoch mean minus best variant (novel)")
            r2 = sec.get("rule", {})
            e["secondary_vs_best_64_shown"] = {
                "note": "not registered; the same rule against the best variant that sees only the 64 frames the LoRA saw",
                "variant": sec["best_variant"], "lora4_minus_best": ci(d2["scene_mean"], d2["ci95"]) if d2 else None,
                "holds": r2.get("holds"),
                "bins": {BREAKDOWN_KEYS[bk]: (None if v.get("holds") is None else
                         {"bin": bin_key(bk, v["best bin"], list(sec["bins"][bk]).index(v["best bin"])), "n": v["n"],
                          "best_variant": v["best variant acc"], "lora4": v["LoRA four-epoch mean"],
                          "lora4_minus_best": v["difference"], "ci95": v["difference_ci95"], "holds": v["holds"]})
                         for bk, v in r2.get("bins", {}).items()}}
        s[GROUP_KEYS[g]] = e
    return s


def bin_stats(sid, vals, allrules, all_names, mask):
    n = int(mask.sum())
    st = {"n": n, "scenes": int(len(set(sid[mask])))}
    if n == 0:
        return st
    acc, _ = pooled_cluster(sid, vals, mask)
    st.update(acc)
    if "LoRA four-epoch mean" in vals:
        d = {"diff": vals["LoRA four-epoch mean"] - vals["best variant"]}
        st["LoRA four-epoch mean minus best variant"] = pooled_cluster(sid, d, mask)[0]["diff"]
    if "Opus" in vals:
        d = {"diff": vals["Opus"] - vals["best variant"]}
        st["Opus minus best variant"] = pooled_cluster(sid, d, mask)[0]["diff"]
    j = int(np.argmax(allrules[mask].mean(0)))
    st["best variant for this bin alone"] = {"name": all_names[j], "acc": float(allrules[mask, j].mean())}
    return st


def decide(res, g):
    """The registered rule for one group (descriptive for groups outside the verdict)."""
    d = res.get("LoRA four-epoch mean minus best variant (novel)")
    if d is None or not res["models"]["LoRA four-epoch mean"]["complete_in_group"]:
        return {"holds": None, "reason": "four-epoch runs not complete in this group"}
    overall = d["scene_mean"] >= MARGIN
    per_bin = {}
    for bk, bins in res.get("bins", {}).items():
        ok = {lab: b for lab, b in bins.items() if b["n"] >= MIN_BIN and "best variant" in b
              and lab not in ("options in different rooms (both kinds)", "no room labels (Replica)", "unlabeled")}
        if not ok:
            per_bin[bk] = {"holds": None, "reason": f"no bin with >= {MIN_BIN} novel trials"}
            continue
        lab = max(ok, key=lambda k: ok[k]["best variant"]["acc"])
        diff = ok[lab]["LoRA four-epoch mean minus best variant"]["acc"]
        per_bin[bk] = {"best bin": lab, "n": ok[lab]["n"], "best variant acc": ok[lab]["best variant"]["acc"],
                       "LoRA four-epoch mean": ok[lab]["LoRA four-epoch mean"]["acc"], "difference": diff,
                       "difference_ci95": ok[lab]["LoRA four-epoch mean minus best variant"]["ci95"], "holds": diff >= MARGIN}
    holds = overall and all(v["holds"] for v in per_bin.values() if v["holds"] is not None)
    return {"overall difference": d["scene_mean"], "overall_ci95": d["ci95"], "overall holds": overall,
            "bins": per_bin, "holds": bool(holds), "counts_for_verdict": g in RULE_GROUPS}


def report(out):
    pct = lambda x: "  -  " if x is None else f"{100 * x:5.1f}"
    for g, res in out["groups"].items():
        M = res["models"]
        b = res["best_variant"]
        print(f"\n== {g}: {len(res['scenes'])} scenes, {res['trials']} trials, {res['novel']} novel")
        print(f"  LoRA four-epoch mean {pct(M['LoRA four-epoch mean']['novel_scene_mean'])}   Opus {pct(M['Opus']['novel_scene_mean'])}"
              f"   (novel, scene mean)")
        print(f"  best variant ({res['best_variant_from']}): {b}: {pct(res['rules'][b]['novel_scene_mean'])}")
        for fam, n in res["best_per_family"].items():
            print(f"    best {fam:22s} {pct(res['rules'][n]['novel_scene_mean'])}  {n}")
        if "rule" in res and res["rule"].get("holds") is not None:
            r = res["rule"]
            print(f"  rule: overall diff {100 * r['overall difference']:.1f} {['fails', 'holds'][r['overall holds']]}; "
                  + "; ".join(f"{bk}: best bin {v.get('best bin')} diff {100 * v['difference']:.1f}" if v.get("holds") is not None
                              else f"{bk}: n/a" for bk, v in r["bins"].items()) + f" -> {'HOLDS' if r['holds'] else 'FAILS'}")
    print("\nverdict:", out["verdict"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--scenes", nargs="+", default=["primary"], help="primary, exp19, or scene names (A, B, P5000, HSSD1, ...)")
    f.add_argument("--threads", type=int, default=6)
    f.add_argument("--limit", type=int, default=None, help="test only: the first N frames and probes")
    f.add_argument("--out", default=None, help="test only: another folder than data/retrieval_feats")
    a = sub.add_parser("analyze")
    a.add_argument("--ignore-checks", action="store_true")
    args = ap.parse_args()
    {"features": features, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
