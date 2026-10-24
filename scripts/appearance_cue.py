"""Experiment 13d (pre-registered in docs/experiments.md): an appearance cue beyond pixels. No model
behaviour is run. For every lure-free trial in rooms A, B and P5000-P5009: the cosine similarity of
DINOv2-base (ViT-B/14) CLS features between the new view and each option; the share of trials where it
favours the near option (novel and familiar); and every run's accuracy on those trials split by which
option DINOv2 favours (with a two-sided Fisher exact p), plus how often the run picks the DINOv2-favoured
option. Frames go in as adjacency_baseline.py and extract.py feed DINOv2: resized to 224 px (bicubic,
antialiased), ImageNet normalisation; the CLS token of the last hidden state. CPU only.

    .venv-mac/bin/python scripts/appearance_cue.py     # prints the tables, writes results/appearance_cue.json
"""
import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import fisher_exact

ROOMS = {"A": ("dissimilar", "data/walks"), "B": ("roomB", "data/walks-roomB"),
         **{f"P{s}": (f"roomP{s}", f"data/walks-roomP{s}") for s in range(5000, 5010)}}


@torch.no_grad()
def dino(model, images, batch=32):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(np.stack(images[i:i + batch])).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=224, mode="bicubic", antialias=True)
        out.append(model(pixel_values=(x - mean) / std).last_hidden_state[:, 0].float())
    return F.normalize(torch.cat(out), dim=1).numpy()


def trial_set(name):
    m = re.search(r"-(original|dissimilar|room[A-Z][A-Za-z0-9]*)(?=-|$)", name)
    return m.group(1) if m else "original"


def split(rows, cue, familiar):
    """Accuracy where DINOv2 favours the near option vs the far one; unparsed answers count as wrong."""
    rs = [r for r in rows if r["familiar"] == familiar]
    k = {fav: [bool(r["correct"]) for r in rs if cue[r["id"]] == fav] for fav in (True, False)}
    parsed = [r for r in rs if r["answer"] in (r["a"], r["b"])]
    picks = [(r["answer"] == r["near"]) == cue[r["id"]] for r in parsed]
    res = {"n_dino_near": len(k[True]), "n_dino_far": len(k[False]),
           "acc_dino_near": float(np.mean(k[True])) if k[True] else None,
           "acc_dino_far": float(np.mean(k[False])) if k[False] else None,
           "picks_dino_favoured": float(np.mean(picks)) if picks else None}
    if k[True] and k[False]:
        res["difference"] = res["acc_dino_near"] - res["acc_dino_far"]
        res["fisher_p"] = float(fisher_exact([[sum(k[True]), len(k[True]) - sum(k[True])],
                                              [sum(k[False]), len(k[False]) - sum(k[False])]])[1])
    return res


def main():
    from transformers import AutoModel
    torch.set_num_threads(8)
    model = AutoModel.from_pretrained("facebook/dinov2-base").eval()     # CPU
    out = {"settings": {"model": "facebook/dinov2-base", "feature": "CLS token of last_hidden_state, cosine similarity",
                        "input": "256 px frame resized to 224 (bicubic, antialias), ImageNet mean/std", "device": "cpu",
                        "unparsed": "count as wrong in accuracy; excluded from picks_dino_favoured"},
           "rooms": {}, "trials": {}, "runs": {}, "pooled_P5000_P5009": {}}
    cue = {}
    for room, (ts, data) in ROOMS.items():
        trials = json.load(open("results/export/dissimilar/trials.json" if ts == "dissimilar" else f"results/export/dissimilar-{ts}/trials.json"))
        frames = np.load(f"{data}/frames.npy", mmap_mode="r")
        probes = np.load(f"{data}/probes.npy", mmap_mode="r")
        meta, pmeta = dict(np.load(f"{data}/meta.npz")), dict(np.load(f"{data}/probes_meta.npz"))
        order = {int(w): np.flatnonzero(meta["walk"] == w)[np.argsort(meta["step"][meta["walk"] == w])] for w in np.unique(meta["walk"])}
        fidx = sorted({int(order[t["walk"]][k - 1]) for t in trials for k in (t["near"], t["far"])})
        pidx = sorted({t["probe"] for t in trials})
        for t in trials:                                  # indexing check: the manifest's distances from the walk's own metadata
            p, (n, f) = t["probe"], (order[t["walk"]][t["near"] - 1], order[t["walk"]][t["far"] - 1])
            assert pmeta["walk"][p] == t["walk"]
            for fi, dist in ((n, t["near_dist_m"]), (f, t["far_dist_m"])):
                assert abs(np.hypot(meta["x"][fi] - pmeta["x"][p], meta["y"][fi] - pmeta["y"][p]) - dist) < 2e-3, (room, t["id"])
        emb = dino(model, [np.asarray(frames[i]) for i in fidx] + [np.asarray(probes[i]) for i in pidx])
        fe, pe = dict(zip(fidx, emb[:len(fidx)])), dict(zip(pidx, emb[len(fidx):]))
        cue[room], out["trials"][room] = {}, []
        for t in trials:
            p = pe[t["probe"]]
            sn, sf = (float(p @ fe[int(order[t["walk"]][k - 1])]) for k in (t["near"], t["far"]))
            cue[room][t["id"]] = sn > sf
            out["trials"][room].append({"id": t["id"], "familiar": t["familiar"], "dino_near": round(sn, 4), "dino_far": round(sf, 4),
                                        "dino_favours_near": sn > sf, "pixels_favour_near": t["pixels_favour_near"],
                                        "frame_order_favours_near": t.get("frame_order_favours_near")})
        summ = {}
        for part, fam in (("novel", False), ("familiar", True)):
            rs = [x for x in out["trials"][room] if x["familiar"] == fam]
            summ[part] = {"n": len(rs), "dino_favours_near": float(np.mean([x["dino_favours_near"] for x in rs])),
                          "pixels_favour_near": float(np.mean([x["pixels_favour_near"] for x in rs])),
                          "dino_agrees_with_pixels": float(np.mean([x["dino_favours_near"] == x["pixels_favour_near"] for x in rs])),
                          "median_margin": float(np.median([abs(x["dino_near"] - x["dino_far"]) for x in rs]))}
        out["rooms"][room] = summ
        print(f"room {room:6s} DINOv2 favours near: novel {summ['novel']['dino_favours_near']:5.1%} of {summ['novel']['n']}, "
              f"familiar {summ['familiar']['dino_favours_near']:5.1%} of {summ['familiar']['n']}  (pixels: novel "
              f"{summ['novel']['pixels_favour_near']:.0%}; DINOv2 agrees with pixels on {summ['novel']['dino_agrees_with_pixels']:.0%} of novel)", flush=True)
    pooled = {}
    for part, fam in (("novel", False), ("familiar", True)):
        for name, rooms in (("P5000-P5009", [r for r in ROOMS if r.startswith("P")]), ("all 12 rooms", list(ROOMS))):
            xs = [x["dino_favours_near"] for r in rooms for x in out["trials"][r] if x["familiar"] == fam]
            pooled.setdefault(name, {})[part] = {"n": len(xs), "dino_favours_near": float(np.mean(xs))}
    out["rooms_pooled"] = pooled
    print("pooled:", json.dumps(pooled))

    print("\nruns: novel accuracy where DINOv2 favours near | far (n), difference, Fisher p; picks DINOv2-favoured option "
          "(* = Fisher p < 0.05)")
    room_of = {ts: room for room, (ts, _) in ROOMS.items()}
    files = sorted(glob.glob("results/claude/*.jsonl") + glob.glob("results/two_choice/*.jsonl") + glob.glob("results/api/*.jsonl"))
    groups = {}
    for path in files:
        stem = path[len("results/"):-len(".jsonl")]
        room = room_of.get(trial_set(os.path.basename(stem)))
        if room is None:
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        assert all(r["id"] in cue[room] for r in rows), path
        res = {"room": room, "trials": len(rows), "manifest_trials": len(cue[room]),
               "novel": split(rows, cue[room], False), "familiar": split(rows, cue[room], True)}
        out["runs"][stem] = res
        if room.startswith("P"):
            groups.setdefault(stem.replace(f"room{room}", "roomP*"), []).append((room, rows))
        nv = res["novel"]
        print(f"{stem:72s} {nv['acc_dino_near']:5.0%} | {nv['acc_dino_far']:5.0%} ({nv['n_dino_near']}/{nv['n_dino_far']})  "
              f"{nv['difference']:+6.1%}  p {nv['fisher_p']:.2g}{'*' if nv['fisher_p'] < 0.05 else ' '}  picks {nv['picks_dino_favoured']:4.0%}"
              + ("" if len(rows) == len(cue[room]) else f"  ({len(rows)} of {len(cue[room])} trials)"))
    print("\npooled over P5000-P5009 (trials pooled; rooms with incomplete runs included as answered)")
    for g, members in sorted(groups.items()):
        allcue = {r["id"] + room: cue[room][r["id"]] for room, rs in members for r in rs}
        rows = [{**r, "id": r["id"] + room} for room, rs in members for r in rs]
        res = {"rooms": [m for m, _ in members], "trials": len(rows),
               "complete": all(len(rs) == len(cue[m]) for m, rs in members) and len(members) == 10,
               "novel": split(rows, allcue, False), "familiar": split(rows, allcue, True)}
        out["pooled_P5000_P5009"][g] = res
        nv = res["novel"]
        print(f"{g:72s} {nv['acc_dino_near']:5.0%} | {nv['acc_dino_far']:5.0%} ({nv['n_dino_near']}/{nv['n_dino_far']})  "
              f"{nv['difference']:+6.1%}  p {nv['fisher_p']:.2g}{'*' if nv['fisher_p'] < 0.05 else ' '}  picks {nv['picks_dino_favoured']:4.0%}"
              + ("" if res["complete"] else "  (INCOMPLETE runs)"))
    json.dump(out, open("results/appearance_cue.json", "w"), indent=1)
    print("-> results/appearance_cue.json")


if __name__ == "__main__":
    main()
