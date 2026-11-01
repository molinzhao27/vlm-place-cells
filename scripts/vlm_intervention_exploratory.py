"""Experiment 29, exploratory cuts decided after the registered verdict (docs/experiments.md, "Exp 29 results").

Is the cost of moving the near set's frames graded by where they were moved? Each trial's near option stays at its
own place; "moved distance" dm is the mean distance from the 8 moved frames to the new view, and df is the far
option's distance to it. Reported: accuracy by dm - df, compared with here-away on the same trials, and a logistic
regression of "chose the near option" on df, dm, the near option's distance dn and the frames' DINOv2 similarity to
the new view, near option (stimulus_check.json), per condition, with a bootstrap over rooms. here-away and clean never
show the moved frames, so dm is a negative control there.

    .venv-claude/bin/python scripts/vlm_intervention_exploratory.py   # -> results/vlm_intervention_exploratory.json
"""
import json

import numpy as np

DIR = "results/vlm_intervention"
NAME = "Qwen3-VL-8B-Instruct+"
RUNS = {"run0": ("qwen8b-proc", "-mps"), "run5": ("qwen8b-proc-e4-s5", ""), "run7": ("qwen8b-proc-e4-s7", "")}
CONDS = ["clean", "here-toward", "here-away", "there-toward", "there-away"]
BINS = [-20, -2, -1, 0, 1, 20]


def logit(X, y, iters=25):
    X = np.c_[np.ones(len(X)), X]
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        b += np.linalg.solve((X.T * (p * (1 - p) + 1e-9)) @ X + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
    return b[1:]


def answers(run, cond, keys):
    adapter, dev = RUNS[run]
    rows = {}
    for line in open(f"{DIR}/{NAME}{adapter}-{cond}{dev}.jsonl"):
        r = json.loads(line)
        rows[f"{r['room']}/{r['id']}"] = float(r["answer"] == r["near"])
    return np.array([rows[k] for k in keys])


def main():
    trials = json.load(open(f"{DIR}/specs.json"))["trials"]
    stim = json.load(open(f"{DIR}/stimulus_check.json"))["trials"]
    metas, rows = {}, []
    for k, t in trials.items():
        if not t["included"]:
            continue
        m = metas.setdefault(t["room"], dict(np.load(f"data/walks-room{t['room']}/meta.npz")))
        idx = np.where(m["walk"] == t["walk"])[0]
        pos = lambda n: np.array([m["x"][idx[n - 1]], m["y"][idx[n - 1]]])   # prompt frame numbers start at 1
        v = np.array(t["view"][:2])
        moved = np.array([[c[1], c[2]] for c in t["conditions"]["there-away"]])
        rows.append((k, t["room"], np.linalg.norm(v - pos(t["near"])), np.linalg.norm(v - pos(t["far"])),
                     np.linalg.norm(moved - v, axis=1).mean(), np.linalg.norm(t["translation"])))
    keys = [r[0] for r in rows]
    rooms = np.array([r[1] for r in rows])
    dn, df, dm, length = (np.array([r[i] for r in rows]) for i in (2, 3, 4, 5))
    z = lambda a: (a - a.mean()) / a.std()
    near = {(run, c): answers(run, c, keys) for run in RUNS for c in CONDS}
    bins = np.digitize(dm - df, BINS)
    out = {"bins_dm_minus_df_m": BINS, "n_per_bin": [int((bins == i).sum()) for i in range(1, len(BINS))],
           "corr_df_dm": float(np.corrcoef(df, dm)[0, 1]), "runs": {}}
    rng = np.random.default_rng(0)
    ur = np.unique(rooms)
    for run in RUNS:
        res = {"chose_near_by_bin": {c: [float(near[run, c][bins == i].mean()) for i in range(1, len(BINS))] for c in CONDS}}
        for c in CONDS:
            cos = {g: np.array([np.mean(stim[k][f"cos_{g}"]) for k in keys]) for g in ("there-away", "there-toward", "here-away", "here-toward")}
            opt = {g: np.array([np.mean(stim[k][f"optcos_{g}"]) for k in keys]) for g in cos}
            g = c if c in cos else "there-away"
            for name, X in (("df_dm_dn", np.c_[z(df), z(dm), z(dn)]), ("plus_dinov2", np.c_[z(df), z(dm), z(dn), z(cos[g]), z(opt[g])]), ("plus_length", np.c_[z(df), z(dm), z(dn), z(length)])):
                y = near[run, c]
                b0 = logit(X, y)
                bs = []
                for _ in range(1000):
                    ii = np.concatenate([np.where(rooms == r)[0] for r in rng.choice(ur, len(ur))])
                    bs.append(logit(X[ii], y[ii], 12))
                lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
                res.setdefault("logit", {}).setdefault(c, {})[name] = {"coef": b0.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
        out["runs"][run] = res
    json.dump(out, open("results/vlm_intervention_exploratory.json", "w"), indent=1)
    print("n per bin", out["n_per_bin"], "corr(df, dm)", round(out["corr_df_dm"], 3))
    for run, res in out["runs"].items():
        for c in ("here-away", "there-away", "there-toward"):
            print(run, f"{c:13s} chose near", [round(x, 3) for x in res["chose_near_by_bin"][c]])
            for name, r in res["logit"][c].items():
                print("   ", f"{name:12s} dm {r['coef'][1]:.2f} [{r['lo'][1]:.2f}, {r['hi'][1]:.2f}]")
    return out


if __name__ == "__main__":
    main()
