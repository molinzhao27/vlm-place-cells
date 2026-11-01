"""Experiment 28 (pre-registered in docs/experiments.md): a learned sequence matcher on the same trials.

    .venv-mac/bin/python scripts/learned_matcher.py features [--threads 6]   # check against experiment 20's cache, then
                                                                            # the training rooms -> data/learned_matcher/train_feats.npz
    <python> scripts/learned_matcher.py grid --variant ordered|orderfree [--device cuda]    # 16 configurations, seed 0
    <python> scripts/learned_matcher.py seeds --variant ordered|orderfree [--device cuda]   # the chosen one, seeds 1-4
    <python> scripts/learned_matcher.py pixels [--device cuda]              # optional: the chosen ordered config on pixels
    .venv-mac/bin/python scripts/learned_matcher.py evaluate                # test scenes, the chosen checkpoints only
    .venv-mac/bin/python scripts/learned_matcher.py analyze                 # -> results/learned_matcher.json

Training data are exactly the LoRA's (finetune.py): data/hf_release/training/proc (md5-matched to vast-3's data/proc),
rooms sorted, the last 64 held out for validation. A training prompt is the room's 64 frames with their real frame
numbers, the new view and the two options; the target is the near option. Features are DINOv2-base's CLS token, computed
as experiment 20 (scripts/retrieval_baselines.py: 224 px, bicubic with antialiasing, ImageNet normalization, CPU,
float32); the test scenes reuse experiment 20's cache (data/retrieval_feats/<scene>.npz) unchanged. Within each prompt,
every feature dimension is z-scored over the prompt's walk frames and the new view is transformed the same way.

The matcher is a pre-LayerNorm transformer over one token per shown walk frame (feature, cosine similarity to the new
view, an option marker shared by both options, and in the ordered variant sinusoids of the frame number) and one token
for the new view; a linear score on each option token's final state, softmax over the two options, cross-entropy against
the near one. The order-free variant has no position input. Grid (per variant): layers {2, 4} x width {128, 256} x
dropout {0.1, 0.3} x lr {3e-4, 1e-3}, seed 0; each run keeps its best epoch on the 512 validation trials; the best
configuration is retrained with seeds 1-4. Only those five checkpoints per variant are ever run on a test scene.
The `grid`, `seeds` and `pixels` steps need only numpy and torch and data/learned_matcher/train_feats.npz, so they can
run on the RTX 4070 (gq's pc queue).
"""
import argparse
import json
import math
import os
import sys
import time
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

TRAIN_DATA = "data/hf_release/training/proc"
OUTDIR = "data/learned_matcher"
TRAIN_FEATS = f"{OUTDIR}/train_feats.npz"
RUNS = f"{OUTDIR}/runs"
CACHE = "data/retrieval_feats"                  # experiment 20's per-scene features
OUT = "results/learned_matcher.json"
N_VAL = 64
GRID = [dict(layers=L, width=d, dropout=p, lr=lr) for L in (2, 4) for d in (128, 256) for p in (0.1, 0.3) for lr in (3e-4, 1e-3)]
EPOCHS, BATCH, WD, WARMUP, CLIP, FRAME_DROP, JITTER = 60, 64, 0.05, 2, 1.0, 0.1, 2.0
PERIODS = 16 * (1024 / 16) ** (np.arange(16) / 15)          # frames; geometric from 16 to 1,024
SEEDS = [0, 1, 2, 3, 4]
P_ROOMS = [f"P{s}" for s in range(5000, 5010)]
APTS = [f"ReplicaApt{i}" for i in range(3)]
HOUSES = [f"HSSD{i}" for i in range(1, 9)]
X19_HOUSES = [f"HSSD{i}" for i in range(9, 27)]
X19_FRL = [f"ReplicaFRL{i}" for i in (0, 1, 3, 4, 5)]
GROUPS = {"procedural": P_ROOMS, "room A": ["A"], "room B": ["B"], "photoreal": APTS + HOUSES, "apartments": APTS,
          "houses": HOUSES, "exp19 houses": X19_HOUSES, "exp19 FRL apartments": X19_FRL}
N_BOOT, BOOT_SEED, SUFFICES, FALLS_SHORT = 10_000, 0, 0.05, 0.15


# ----------------------------------------------------------------------------------------------- features
def dino_model(threads):
    import torch
    from transformers import AutoModel
    torch.set_num_threads(threads)
    return AutoModel.from_pretrained("facebook/dinov2-base").eval()


def dino_cls(model, images, batch=32):
    """Experiment 20's DINOv2 CLS token (retrieval_baselines.features), same resize, normalization and batch size."""
    import torch
    import torch.nn.functional as F
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(np.array(images[i:i + batch])).permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, size=224, mode="bicubic", antialias=True)
            out.append(model(pixel_values=(x - mean) / std).last_hidden_state[:, 0].float().numpy())
    return np.concatenate(out)


def check_features(model):
    """The registered check: P5000's first 256 walk frames and the views of its first 16 trials against the cache."""
    from spatialcells.analyze_probes import pixel_feats
    z = np.load(f"{CACHE}/P5000.npz")
    frames = np.load("data/walks-roomP5000/frames.npy", mmap_mode="r")[:256]
    probes = np.load("data/walks-roomP5000/probes.npy", mmap_mode="r")
    trials = json.load(open("results/export/dissimilar-roomP5000/trials.json"))
    trials = trials["trials"] if isinstance(trials, dict) else trials
    pid = sorted({t["probe"] for t in trials[:16]})
    prow = {int(p): i for i, p in enumerate(z["probe_idx"])}
    rows = [prow[p] for p in pid]
    res = {"frames": 256, "views": len(pid),
           "frame_cls_max_abs_diff": float(np.abs(dino_cls(model, frames) - z["frame_cls"][:256]).max()),
           "view_cls_max_abs_diff": float(np.abs(dino_cls(model, probes[pid]) - z["probe_cls"][rows]).max()),
           "frame_pixels_max_abs_diff": float(np.abs(pixel_feats(frames) - z["frame_pixels"][:256]).max()),
           "view_pixels_max_abs_diff": float(np.abs(pixel_feats(probes[pid]) - z["probe_pixels"][rows]).max())}
    res["pass"] = bool(max(v for k, v in res.items() if k.endswith("diff")) < 1e-3)
    return res


def features(args):
    from spatialcells.analyze_probes import pixel_feats
    os.makedirs(OUTDIR, exist_ok=True)
    model = dino_model(args.threads)
    chk = check_features(model)
    print("feature check:", chk, flush=True)
    json.dump(chk, open(f"{OUTDIR}/feature_check.json", "w"), indent=1)
    if not chk["pass"]:
        raise SystemExit("feature check failed: the training features would not match experiment 20's cache")
    trials = json.load(open(f"{TRAIN_DATA}/trials.json"))
    rooms = sorted({t["room"] for t in trials})
    probes_of = {r: sorted({t["probe"] for t in trials if t["room"] == r}) for r in rooms}
    P = max(len(v) for v in probes_of.values())
    part = f"{OUTDIR}/train_feats_parts"
    os.makedirs(part, exist_ok=True)
    t0 = time.time()
    starts = list(range(0, len(rooms), 50))                 # resumable chunks of 50 rooms; a second job can take
    for c0 in (starts[::-1] if args.reverse else starts):   # them from the end (--reverse); finished chunks are skipped
        path = f"{part}/{c0:04d}.npz"
        if os.path.exists(path):
            continue
        chunk = rooms[c0:c0 + 50]
        fc, fp, pc, pp, fn = [], [], [], [], []
        for r in chunk:
            z = np.load(f"{TRAIN_DATA}/room_{r:04d}.npz")
            frames, pr = z["frames"], z["probes"][probes_of[r]]
            fc.append(dino_cls(model, frames))
            pc.append(np.pad(dino_cls(model, pr), ((0, P - len(pr)), (0, 0))))
            fp.append(pixel_feats(frames))
            pp.append(np.pad(pixel_feats(pr), ((0, P - len(pr)), (0, 0))))
            fn.append(z["frame_numbers"])
        tmp = f"{path}.{os.getpid()}.tmp.npz"
        np.savez(tmp, frame_cls=np.stack(fc), frame_pixels=np.stack(fp), probe_cls=np.stack(pc),
                 probe_pixels=np.stack(pp), frame_numbers=np.stack(fn))
        os.replace(tmp, path)
        print(f"rooms {chunk[0]}-{chunk[-1]} done, {time.time() - t0:.0f} s", flush=True)
    parts = [np.load(f"{part}/{c0:04d}.npz") for c0 in range(0, len(rooms), 50)]
    cat = lambda k: np.concatenate([p[k] for p in parts])
    room_ix = {r: i for i, r in enumerate(rooms)}
    probe_ids = np.full((len(rooms), P), -1)
    for r in rooms:
        probe_ids[room_ix[r], :len(probes_of[r])] = probes_of[r]
    fnum = cat("frame_numbers")
    tr = {"trial_id": np.array([t["id"] for t in trials]), "trial_room": np.array([room_ix[t["room"]] for t in trials]),
          "trial_probe_row": np.array([probes_of[t["room"]].index(t["probe"]) for t in trials]),
          "trial_a": np.array([t["a"] for t in trials]), "trial_b": np.array([t["b"] for t in trials]),
          "trial_near": np.array([t["near"] for t in trials]), "trial_familiar": np.array([t["familiar"] for t in trials])}
    for i, t in enumerate(trials):                          # every option is one of the room's 64 frames
        assert t["a"] in fnum[tr["trial_room"][i]] and t["b"] in fnum[tr["trial_room"][i]], t["id"]
    tmp = f"{TRAIN_FEATS}.{os.getpid()}.tmp.npz"
    np.savez(tmp, rooms=np.array(rooms), frame_numbers=fnum, probe_ids=probe_ids,
             frame_cls=cat("frame_cls"), probe_cls=cat("probe_cls"), frame_pixels=cat("frame_pixels"),
             probe_pixels=cat("probe_pixels"), **tr)
    os.replace(tmp, TRAIN_FEATS)
    print(f"{len(rooms)} rooms, {len(trials)} trials -> {TRAIN_FEATS} ({time.time() - t0:.0f} s)", flush=True)


# ----------------------------------------------------------------------------------------------- model
SD_FLOOR = {"cls": 1e-6, "pixels": 1.0}      # pixels: one grey level, so a dimension constant over a prompt stays finite


def zscore(F, P, feature):
    """F (N, D) walk frames of one prompt, P (D,) the new view: z-scored with the frames' mean and SD (float32)."""
    F, P = F.astype(np.float32), P.astype(np.float32)
    mu, sd = F.mean(0), F.std(0) + SD_FLOOR[feature]
    return (F - mu) / sd, (P - mu) / sd


def build_model(feat_dim, layers, width, dropout, ordered):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Matcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.ordered = ordered
            self.frame_in = nn.Linear(feat_dim, width)
            self.query_in = nn.Linear(feat_dim, width)
            self.sim_in = nn.Linear(1, width)
            self.pos_in = nn.Linear(2 * len(PERIODS), width) if ordered else None
            self.option = nn.Parameter(0.02 * torch.randn(width))
            self.qtype = nn.Parameter(0.02 * torch.randn(width))
            layer = nn.TransformerEncoderLayer(width, width // 32, 4 * width, dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, 1)
            self.register_buffer("periods", torch.tensor(PERIODS, dtype=torch.float32))

        def forward(self, W, pos, keep, ia, ib, q):
            """W (B, N, D) z-scored walk frames, pos (B, N) frame numbers, keep (B, N) bool, ia/ib (B,) option
            indices, q (B, D) z-scored new view -> (B, 2) scores for options a and b."""
            B, N, _ = W.shape
            ar = torch.arange(B, device=W.device)
            x = self.frame_in(W) + self.sim_in(F.cosine_similarity(W, q[:, None], dim=-1)[..., None])
            if self.ordered:
                ang = 2 * math.pi * pos[..., None] / self.periods
                x = x + self.pos_in(torch.cat([ang.sin(), ang.cos()], -1))
            mark = torch.zeros(B, N, device=W.device)
            mark[ar, ia] = 1
            mark[ar, ib] = 1
            x = x + mark[..., None] * self.option
            h = torch.cat([(self.query_in(q) + self.qtype)[:, None], x], 1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=W.device), ~keep], 1)
            s = self.head(self.norm(self.encoder(h, src_key_padding_mask=pad))[:, 1:])[..., 0]
            return torch.stack([s[ar, ia], s[ar, ib]], 1)

    return Matcher()


def run_dir(feature, variant, cfg, seed):
    return f"{RUNS}/{feature}-{variant}/L{cfg['layers']}-d{cfg['width']}-p{cfg['dropout']}-lr{cfg['lr']:g}-s{seed}"


def load_train(feature, device):
    """Tensors for the training and validation trials (the LoRA's split: the last 64 sorted rooms held out)."""
    import torch
    z = np.load(TRAIN_FEATS)
    Fr, Pr = z[f"frame_{feature}"].astype(np.float32), z[f"probe_{feature}"].astype(np.float32)   # (R, 64, D), (R, P, D)
    mu, sd = Fr.mean(1, keepdims=True), Fr.std(1, keepdims=True) + SD_FLOOR[feature]           # per room: its 64 frames
    Zf, Zp = (Fr - mu) / sd, (Pr - mu) / sd
    del Fr, Pr
    fnum = z["frame_numbers"]
    room, prow = z["trial_room"], z["trial_probe_row"]
    ia = np.array([np.flatnonzero(fnum[r] == a)[0] for r, a in zip(room, z["trial_a"])])
    ib = np.array([np.flatnonzero(fnum[r] == b)[0] for r, b in zip(room, z["trial_b"])])
    y = (z["trial_near"] == z["trial_b"]).astype(np.int64)            # 0: a is near, 1: b is near
    assert ((z["trial_near"] == z["trial_a"]) | (z["trial_near"] == z["trial_b"])).all()
    val = room >= len(z["rooms"]) - N_VAL
    assert sorted(z["rooms"]) == list(z["rooms"])
    T = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=device)
    d = {"Zf": T(Zf), "fnum": T(fnum), "Q": T(Zp[room, prow]), "room": T(room, torch.long), "ia": T(ia, torch.long),
         "ib": T(ib, torch.long), "y": T(y, torch.long), "familiar": z["trial_familiar"],
         "train": np.flatnonzero(~val), "val": np.flatnonzero(val)}
    return d


def batch_scores(model, d, idx, device, bs=512):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = torch.as_tensor(idx[i:i + bs], device=device)
            r = d["room"][b]
            keep = torch.ones(len(b), d["Zf"].shape[1], dtype=torch.bool, device=device)
            out.append(model(d["Zf"][r], d["fnum"][r], keep, d["ia"][b], d["ib"][b], d["Q"][b]).float().cpu().numpy())
    return np.concatenate(out)


def correct(scores, y):
    """1 if the near option scores higher, 0 if lower, 0.5 on an exact tie."""
    near, far = scores[np.arange(len(y)), y], scores[np.arange(len(y)), 1 - y]
    return np.where(near > far, 1.0, np.where(near < far, 0.0, 0.5))


def train_run(cfg, variant, seed, feature, device, d=None):
    import torch
    import torch.nn.functional as F
    out = run_dir(feature, variant, cfg, seed)
    if os.path.exists(f"{out}/log.json"):
        return json.load(open(f"{out}/log.json"))
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    d = d or load_train(feature, device)
    ordered = variant == "ordered"
    model = build_model(d["Zf"].shape[-1], cfg["layers"], cfg["width"], cfg["dropout"], ordered).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": rest, "weight_decay": 0.0}], lr=cfg["lr"])
    tr = torch.as_tensor(d["train"], device=device)
    per_epoch = math.ceil(len(tr) / BATCH)
    total, warm = EPOCHS * per_epoch, WARMUP * per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    yv = d["y"][torch.as_tensor(d["val"], device=device)].cpu().numpy()
    fam_v = d["familiar"][d["val"]]
    log = {"config": cfg, "variant": variant, "seed": seed, "feature": feature, "params": n_params, "epochs": [],
           "device": str(device)}
    best, t0 = (-1, None, None), time.time()
    N = d["Zf"].shape[1]
    for ep in range(EPOCHS):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=g, device=device)]
        tot, n = 0.0, 0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            B = len(b)
            ar = torch.arange(B, device=device)
            r = d["room"][b]
            pos = d["fnum"][r]
            if ordered:
                pos = pos + (2 * torch.rand(pos.shape, generator=g, device=device) - 1) * JITTER
            keep = torch.rand(B, N, generator=g, device=device) >= FRAME_DROP
            keep[ar, d["ia"][b]] = True
            keep[ar, d["ib"][b]] = True
            loss = F.cross_entropy(model(d["Zf"][r], pos, keep, d["ia"][b], d["ib"][b], d["Q"][b]), d["y"][b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            sched.step()
            tot, n = tot + loss.item() * B, n + B
        model.eval()
        c = correct(batch_scores(model, d, d["val"], device), yv)
        acc, acc_nov = float(c.mean()), float(c[~fam_v].mean())
        log["epochs"].append({"epoch": ep + 1, "train_loss": tot / n, "val_acc": acc, "val_novel_acc": acc_nov})
        if acc > best[0]:
            best = (acc, ep + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if (ep + 1) % 10 == 0:
            print(f"  {os.path.basename(out)} epoch {ep + 1}: loss {tot / n:.3f}, val {acc:.3f} (novel {acc_nov:.3f}), "
                  f"best {best[0]:.3f} at {best[1]}, {time.time() - t0:.0f} s", flush=True)
    be = log["epochs"][best[1] - 1]
    log.update(best_epoch=best[1], val_acc=be["val_acc"], val_novel_acc=be["val_novel_acc"], seconds=time.time() - t0)
    torch.save({"state": best[2], "config": cfg, "variant": variant, "seed": seed, "feature": feature,
                "feat_dim": int(d["Zf"].shape[-1]), "best_epoch": best[1]}, f"{out}/model.pt")
    json.dump(log, open(f"{out}/log.json.tmp", "w"), indent=1)
    os.replace(f"{out}/log.json.tmp", f"{out}/log.json")
    print(f"{out}: best epoch {best[1]}, val {be['val_acc']:.4f} (novel {be['val_novel_acc']:.4f}), "
          f"{n_params} parameters, {time.time() - t0:.0f} s", flush=True)
    return log


def select(variant, feature="cls"):
    """The registered choice: highest best-epoch validation accuracy of the seed-0 grid; ties go to fewer parameters,
    then the lower learning rate (then grid order)."""
    logs = []
    for k, cfg in enumerate(GRID):
        p = f"{run_dir(feature, variant, cfg, 0)}/log.json"
        assert os.path.exists(p), f"grid incomplete: {p}"
        logs.append((k, json.load(open(p))))
    k, best = min(logs, key=lambda kl: (-kl[1]["val_acc"], kl[1]["params"], kl[1]["config"]["lr"], kl[0]))
    sel = {"variant": variant, "feature": feature, "config": best["config"], "params": best["params"],
           "val_acc": best["val_acc"], "val_novel_acc": best["val_novel_acc"], "best_epoch": best["best_epoch"],
           "tied_at_top": [l["config"] for _, l in logs if l["val_acc"] == best["val_acc"]],
           "grid": [{**l["config"], "params": l["params"], "val_acc": l["val_acc"], "val_novel_acc": l["val_novel_acc"],
                     "best_epoch": l["best_epoch"]} for _, l in logs]}
    json.dump(sel, open(f"{RUNS}/{feature}-{variant}/selection.json", "w"), indent=1)
    return sel


def device_of(args):
    """Device setup; --feats, --runs-dir and --epochs exist only for smoke tests (the registered values are the defaults)."""
    import torch
    global TRAIN_FEATS, RUNS, EPOCHS
    TRAIN_FEATS, RUNS, EPOCHS = args.feats or TRAIN_FEATS, args.runs_dir or RUNS, args.epochs or EPOCHS
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return torch.device(args.device)


def grid(args):
    dev = device_of(args)
    d = load_train("cls", dev)
    print(f"{len(d['train'])} training trials, {len(d['val'])} validation trials, {args.variant}", flush=True)
    for cfg in GRID:
        train_run(cfg, args.variant, 0, "cls", dev, d)
    sel = select(args.variant)
    print("chosen:", sel["config"], f"val {sel['val_acc']:.4f}", flush=True)


def seeds(args):
    dev = device_of(args)
    sel = select(args.variant)
    print("chosen:", sel["config"], f"val {sel['val_acc']:.4f}", flush=True)
    d = load_train("cls", dev)
    for s in SEEDS[1:]:
        train_run(sel["config"], args.variant, s, "cls", dev, d)


def pixels(args):
    dev = device_of(args)
    sel = json.load(open(f"{RUNS}/cls-ordered/selection.json"))
    d = load_train("pixels", dev)
    for s in SEEDS:
        train_run(sel["config"], "ordered", s, "pixels", dev, d)


# ----------------------------------------------------------------------------------------------- test scenes
VARIANTS = ["ordered", "orderfree"]
ARGS64 = argparse.Namespace(no_walk=False, max_images=64, shuffle_walk=False, relabel=False, reverse_walk=False,
                            window_shuffle=None, renumber=False)


def data_dir(scene):                            # as retrieval_baselines.py
    return "data/walks" if scene == "A" else f"data/walks-room{scene}"


def trials_of(scene):
    path = "results/export/dissimilar/trials.json" if scene == "A" else f"results/export/dissimilar-room{scene}/trials.json"
    t = json.load(open(path))
    return t["trials"] if isinstance(t, dict) else t


def scene_inputs(scene, feature):
    """Every trial of a scene as the LoRA's prompt shows it (two_choice.present, --max-images 64, ordered): padded arrays."""
    from spatialcells.two_choice import present, walk_order
    z = np.load(f"{CACHE}/{scene}.npz")
    Fall, Pall = z[f"frame_{feature}"], z[f"probe_{feature}"]
    prow = {int(p): i for i, p in enumerate(z["probe_idx"])}
    meta = dict(np.load(f"{data_dir(scene)}/meta.npz"))
    pmeta = dict(np.load(f"{data_dir(scene)}/probes_meta.npz"))
    order = walk_order(meta)
    trials = trials_of(scene)
    items = []
    for t in trials:
        rows = order[t["walk"]]
        assert pmeta["walk"][t["probe"]] == t["walk"]
        shown, lab = present(t, len(rows), ARGS64)
        assert shown == sorted(shown) and all(lab[f] == f for f in shown)
        S = np.array(shown)
        Zf, q = zscore(Fall[rows[S - 1]], Pall[prow[t["probe"]]], feature)
        perm = S[np.random.default_rng(zlib.crc32(f"{scene}/{t['id']}".encode())).permutation(len(S))]
        items.append((Zf, S, perm, int(np.flatnonzero(S == t["a"])[0]), int(np.flatnonzero(S == t["b"])[0]), q,
                      0 if t["near"] == t["a"] else 1))
    T, N, D = len(items), max(len(it[1]) for it in items), items[0][0].shape[1]
    W, pos, perm, keep = np.zeros((T, N, D), np.float32), np.zeros((T, N), np.float32), np.zeros((T, N), np.float32), np.zeros((T, N), bool)
    for i, (Zf, S, pm, *_r) in enumerate(items):
        W[i, :len(S)], pos[i, :len(S)], perm[i, :len(S)], keep[i, :len(S)] = Zf, S, pm, True
    return {"ids": [t["id"] for t in trials], "familiar": [bool(t["familiar"]) for t in trials], "W": W, "pos": pos,
            "perm": perm, "keep": keep, "ia": np.array([it[3] for it in items]), "ib": np.array([it[4] for it in items]),
            "Q": np.stack([it[5] for it in items]), "y": np.array([it[6] for it in items]), "n_shown": [len(it[1]) for it in items]}


def load_model(path):
    import torch
    ck = torch.load(path, map_location="cpu")
    c = ck["config"]
    m = build_model(ck["feat_dim"], c["layers"], c["width"], c["dropout"], ck["variant"] == "ordered")
    m.load_state_dict(ck["state"])
    return m.eval()


def score_scene(model, x, positions):
    import torch
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
    out = []
    with torch.no_grad():
        for i in range(0, len(x["y"]), 256):
            s = slice(i, i + 256)
            out.append(model(T(x["W"][s]), T(positions[s]), T(x["keep"][s], torch.bool), T(x["ia"][s], torch.long),
                             T(x["ib"][s], torch.long), T(x["Q"][s])).numpy())
    return correct(np.concatenate(out), x["y"])


def chosen_paths(feature, variant):
    sel = json.load(open(f"{RUNS}/cls-{variant}/selection.json"))     # pixels reuse the ordered DINOv2 choice
    paths = [f"{run_dir(feature, variant, sel['config'], s)}/model.pt" for s in SEEDS]
    return paths if all(os.path.exists(p) for p in paths) else None


def evaluate(args):
    """Only the chosen configuration's five seeds per variant ever see a test scene."""
    import torch
    torch.set_num_threads(args.threads)
    methods = {}
    for v in VARIANTS:
        paths = chosen_paths("cls", v)
        assert paths, f"the {v} variant's five chosen runs are not all trained"
        methods[v] = ("cls", paths)
    if chosen_paths("pixels", "ordered"):
        methods["pixels"] = ("pixels", chosen_paths("pixels", "ordered"))
    models = {m: [load_model(p) for p in paths] for m, (_, paths) in methods.items()}
    scenes = list(dict.fromkeys(s for g in GROUPS.values() for s in g))
    out = {"methods": {m: paths for m, (_, paths) in methods.items()}, "scenes": {}}
    t0 = time.time()
    for scene in scenes:
        if not os.path.exists(f"{CACHE}/{scene}.npz"):
            print(f"{scene}: no cached features, skipped", flush=True)
            continue
        xs = {f: scene_inputs(scene, f) for f in {"cls"} | ({"pixels"} if "pixels" in methods else set())}
        x = xs["cls"]
        rec = {"ids": x["ids"], "familiar": x["familiar"], "n_shown": x["n_shown"]}
        for m, (f, _) in methods.items():
            rec[m] = np.stack([score_scene(mod, xs[f], xs[f]["pos"]) for mod in models[m]], 1).tolist()
        rec["ordered_permuted"] = np.stack([score_scene(mod, x, x["perm"]) for mod in models["ordered"]], 1).tolist()
        out["scenes"][scene] = rec
        print(f"{scene}: {len(x['ids'])} trials scored ({time.time() - t0:.0f} s)", flush=True)
    json.dump(out, open(f"{OUTDIR}/test_scores.json", "w"))
    print(f"-> {OUTDIR}/test_scores.json")


# ----------------------------------------------------------------------------------------------- analysis
LORA4 = [3, 4, 5, 6, 7]
GROUP_KEYS = {"procedural": "procedural", "room A": "room_A", "room B": "room_B", "photoreal": "photoreal",
              "apartments": "apartments", "houses": "houses", "exp19 houses": "exp19_houses",
              "exp19 FRL apartments": "exp19_frl"}


def lora_shuffled(scene, ids, man):
    """Per trial, the share of the four-epoch runs 3-7 that answered correctly with the shuffled walk (real frame
    numbers); None unless all five runs cover the scene."""
    ts = "dissimilar" if scene == "A" else f"room{scene}"
    runs = []
    for k in LORA4:
        base = f"results/two_choice/Qwen3-VL-8B-Instruct+qwen8b-proc-e4-s{k}-{ts}-256px-max64-shuffled"
        found = [p for p in (base + ".jsonl", base + "-mps.jsonl") if os.path.exists(p)]
        if len(found) != 1:
            return None
        rows = {}
        for line in open(found[0]):
            if line.strip():
                r = json.loads(line)
                t = man[r["id"]]
                assert (r["near"], r["far"], r["familiar"]) == (t["near"], t["far"], t["familiar"]), (found[0], r["id"])
                assert r["id"] not in rows
                rows[r["id"]] = float(bool(r["correct"]))
        if set(rows) != set(ids):
            return None
        runs.append([rows[i] for i in ids])
    return np.mean(runs, 0)


def boot_mean(x, seed=BOOT_SEED):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def analyze(args):
    ts = json.load(open(f"{OUTDIR}/test_scores.json"))
    rb = json.load(open("results/retrieval_baselines.json"))
    sel = {v: json.load(open(f"{RUNS}/cls-{v}/selection.json")) for v in VARIANTS}
    vlogs = {v: [json.load(open(f"{run_dir('cls', v, sel[v]['config'], s)}/log.json")) for s in SEEDS] for v in VARIANTS}
    val_mean = {v: float(np.mean([l["val_acc"] for l in vlogs[v]])) for v in VARIANTS}
    main_v = max(VARIANTS, key=lambda v: (val_mean[v], v == "ordered"))
    has_pix = "pixels" in ts["methods"]
    # per scene, per trial: every method's score
    V = {}
    for scene, s in ts["scenes"].items():
        ids = s["ids"]
        man = {t["id"]: t for t in trials_of(scene)}
        assert list(man) == ids
        rbt = {r["id"]: r for r in rb["trials"][scene]}
        assert set(rbt) == set(ids), scene
        col = lambda k: None if any(rbt[i][k] is None for i in ids) else np.array([rbt[i][k] for i in ids], float)
        v = {"familiar": np.array(s["familiar"]), "matcher": np.mean(s[main_v], 1),
             "ordered": np.mean(s["ordered"], 1), "orderfree": np.mean(s["orderfree"], 1),
             "ordered_permuted": np.mean(s["ordered_permuted"], 1),
             "lora4": col("LoRA four-epoch mean"), "opus": col("Opus"), "best_rule": col("best_variant"),
             "lora4_shuffled": lora_shuffled(scene, ids, man)}
        if has_pix:
            v["pixels"] = np.mean(s["pixels"], 1)
        for k in SEEDS:
            v[f"{main_v} seed {k}"] = np.array(s[main_v])[:, k]
        V[scene] = v
    methods = ["matcher", "ordered", "orderfree", "ordered_permuted"] + (["pixels"] if has_pix else []) + \
              ["lora4", "lora4_shuffled", "opus", "best_rule"]
    diffs = {"lora4_minus_matcher": ("lora4", "matcher"), "matcher_minus_opus": ("matcher", "opus"),
             "matcher_minus_best_rule": ("matcher", "best_rule"), "ordered_minus_orderfree": ("ordered", "orderfree"),
             "ordered_minus_ordered_permuted": ("ordered", "ordered_permuted"),
             "lora4_ordered_minus_shuffled": ("lora4", "lora4_shuffled"),
             "lora4_minus_ordered": ("lora4", "ordered"),            # descriptive: the other variant
             "lora4_minus_pixels": ("lora4", "pixels")}              # descriptive: the optional pixel variant
    out = {"settings": {"registration": "docs/experiments.md, Experiment 28", "matcher_variant": main_v,
                        "matcher_per_trial": "share of the five seeds (0-4) of the chosen configuration that answered correctly",
                        "lora4": "four-epoch runs 3-7, 64-frame walk, share correct (results/retrieval_baselines.json trials)",
                        "opus": "claude-opus-5-5, 256-frame walk, low effort", "best_rule": "experiment 20's best rule of the "
                        "group (procedural: combined logistic regression; photoreal: window maximum, DINOv2 CLS, all 256, k 32)",
                        "lora4_shuffled": "four-epoch runs 3-7 with the walk shuffled (real frame numbers kept)",
                        "ordered_permuted": "the ordered matcher with the frame numbers permuted at random among the shown "
                                            "frames (seeded per trial); its counterpart of the shuffled walk",
                        "bootstrap": f"{N_BOOT} resamples over scenes, seed {BOOT_SEED}; one-scene groups resample trials",
                        "grid": GRID, "epochs": EPOCHS, "batch": BATCH, "weight_decay": WD, "warmup_epochs": WARMUP,
                        "frame_drop": FRAME_DROP, "jitter_frames": JITTER, "periods": PERIODS.tolist(),
                        "decision": f"D = mean over P5000-P5009 of (LoRA four-epoch mean - matcher), novel trials: "
                                    f"<= {SUFFICES:.2f} suffices, >= {FALLS_SHORT:.2f} falls short, else partial"},
           "validation": {v: {"selection": sel[v], "seeds": [{"seed": l["seed"], "val_acc": l["val_acc"],
                                                                "val_novel_acc": l["val_novel_acc"], "best_epoch": l["best_epoch"],
                                                                "device": l.get("device")} for l in vlogs[v]],
                              "val_acc_mean": val_mean[v],
                              "val_novel_acc_mean": float(np.mean([l["val_novel_acc"] for l in vlogs[v]]))} for v in VARIANTS},
           "groups": {}}
    if has_pix:
        plogs = [json.load(open(f"{run_dir('pixels', 'ordered', sel['ordered']['config'], s)}/log.json")) for s in SEEDS]
        out["validation"]["pixels"] = {"seeds": [{"seed": l["seed"], "val_acc": l["val_acc"], "val_novel_acc": l["val_novel_acc"],
                                                  "best_epoch": l["best_epoch"]} for l in plogs],
                                       "val_acc_mean": float(np.mean([l["val_acc"] for l in plogs]))}
    for g, members in GROUPS.items():
        members = [s for s in members if s in V]
        if not members:
            continue
        nov = {s: ~V[s]["familiar"] for s in members}
        res = {"scenes": members, "n_trials": int(sum(len(V[s]["familiar"]) for s in members)),
               "n_novel": int(sum(nov[s].sum() for s in members)), "novel": {}, "per_scene_novel": {}, "familiar_pooled": {},
               "all_pooled": {}}
        for m in methods + [f"{main_v} seed {k}" for k in SEEDS]:
            have = [s for s in members if V[s].get(m) is not None and nov[s].any()]
            if not have:
                continue
            per = {s: float(V[s][m][nov[s]].mean()) for s in have}
            if len(members) == 1:
                mean, ci = boot_mean(V[have[0]][m][nov[have[0]]])
            else:
                mean, ci = boot_mean(list(per.values()))
            res["novel"][m] = {"mean": mean, "ci95": ci, "n_scenes": len(have)}
            res["per_scene_novel"][m] = per
            allv = np.concatenate([V[s][m] for s in have])
            fam = np.concatenate([V[s]["familiar"] for s in have])
            res["familiar_pooled"][m] = float(allv[fam].mean()) if fam.any() else None
            res["all_pooled"][m] = float(allv.mean())
        for name, (a, b) in diffs.items():
            have = [s for s in members if V[s].get(a) is not None and V[s].get(b) is not None and nov[s].any()]
            if not have:
                continue
            per = {s: float((V[s][a] - V[s][b])[nov[s]].mean()) for s in have}
            if len(members) == 1:
                mean, ci = boot_mean((V[have[0]][a] - V[have[0]][b])[nov[have[0]]])
            else:
                mean, ci = boot_mean(list(per.values()))
            res["novel"][name] = {"mean": mean, "ci95": ci, "n_scenes": len(have), "scenes_up": sum(x > 0 for x in per.values()),
                                  "scenes_down": sum(x < 0 for x in per.values())}
            res["per_scene_novel"][name] = per
        have = [s for s in members if V[s]["best_rule"] is not None]
        if have:                                    # descriptive: the matcher where the best rule is right or wrong
            br = np.concatenate([V[s]["best_rule"][nov[s]] for s in have])
            res["by_best_rule_novel"] = {}
            for m in ("matcher", "lora4", "opus"):
                if all(V[s][m] is not None for s in have):
                    mv = np.concatenate([V[s][m][nov[s]] for s in have])
                    res["by_best_rule_novel"][m] = {"rule_right": {"n": int((br == 1).sum()), "acc": float(mv[br == 1].mean()) if (br == 1).any() else None},
                                                    "rule_wrong": {"n": int((br == 0).sum()), "acc": float(mv[br == 0].mean()) if (br == 0).any() else None}}
        out["groups"][g] = res
    D = out["groups"]["procedural"]["novel"]["lora4_minus_matcher"]
    assert D["n_scenes"] == 10
    label = "learned matching suffices" if D["mean"] <= SUFFICES else "learned matching falls short" if D["mean"] >= FALLS_SHORT else "partial"
    out["verdict"] = {"label": label, "D_lora4_minus_matcher": D["mean"], "ci95": D["ci95"], "matcher_variant": main_v}
    out = {"summary": summary(out, sel, vlogs, val_mean), **out}
    json.dump(out, open(OUT, "w"), indent=1)
    report(out)
    print(f"-> {OUT}")


def summary(out, sel, vlogs, val_mean):
    """The values the paper cites, as fractions (0.15 = 15 points); novel trials, mean over scenes with a 95% bootstrap
    CI over scenes (one-scene groups: over trials)."""
    s = {"units": "fractions of trials (0-1); differences in fractions (0.15 = 15 points)",
         "verdict": out["verdict"]["label"], "matcher_variant": out["verdict"]["matcher_variant"],
         "lora4_minus_matcher_heldout": {"mean": out["verdict"]["D_lora4_minus_matcher"], "ci95": out["verdict"]["ci95"]},
         "thresholds": {"suffices_at_most": SUFFICES, "falls_short_at_least": FALLS_SHORT},
         "chosen": {v: {**sel[v]["config"], "params": sel[v]["params"], "val_acc_seed0": sel[v]["val_acc"],
                        "val_acc_seeds": [l["val_acc"] for l in vlogs[v]], "val_acc_mean": val_mean[v],
                        "val_novel_acc_mean": out["validation"][v]["val_novel_acc_mean"],
                        "best_epochs": [l["best_epoch"] for l in vlogs[v]]} for v in VARIANTS}}
    if "pixels" in out["validation"]:
        s["chosen"]["pixels"] = {"val_acc_mean": out["validation"]["pixels"]["val_acc_mean"], "note": "descriptive only"}
    keep = ["matcher", "ordered", "orderfree", "ordered_permuted", "pixels", "lora4", "lora4_shuffled", "opus", "best_rule",
            "lora4_minus_matcher", "matcher_minus_opus", "matcher_minus_best_rule", "ordered_minus_orderfree",
            "ordered_minus_ordered_permuted", "lora4_ordered_minus_shuffled", "lora4_minus_ordered", "lora4_minus_pixels"]
    for g, res in out["groups"].items():
        s[GROUP_KEYS[g]] = {"n_scenes": len(res["scenes"]), "n_novel": res["n_novel"],
                            "novel": {k: {kk: res["novel"][k][kk] for kk in ("mean", "ci95", "n_scenes")}
                                      for k in keep if k in res["novel"]},
                            "all_trials_pooled": {k: res["all_pooled"][k] for k in keep if k in res["all_pooled"]},
                            "familiar_pooled": {k: res["familiar_pooled"][k] for k in keep if k in res["familiar_pooled"]}}
        if "by_best_rule_novel" in res:
            s[GROUP_KEYS[g]]["by_best_rule_novel"] = res["by_best_rule_novel"]
        if g in ("procedural", "photoreal"):
            s[GROUP_KEYS[g]]["per_scene_novel"] = {k: res["per_scene_novel"][k] for k in
                                                   ("matcher", "ordered", "orderfree", "pixels", "lora4", "opus", "best_rule",
                                                    "lora4_minus_matcher") if k in res["per_scene_novel"]}
    return s


def report(out):
    pct = lambda x: "  -  " if x is None else f"{100 * x:5.1f}"
    print(f"\nchosen: " + "; ".join(f"{v}: {out['validation'][v]['selection']['config']} val mean "
                                    f"{100 * out['validation'][v]['val_acc_mean']:.1f}" for v in VARIANTS))
    for g, res in out["groups"].items():
        n = res["novel"]
        line = "  ".join(f"{k} {pct(n[k]['mean'])}" for k in ("matcher", "ordered", "orderfree", "ordered_permuted", "pixels",
                                                              "lora4", "lora4_shuffled", "opus", "best_rule") if k in n)
        print(f"== {g} ({len(res['scenes'])} scenes, {res['n_novel']} novel): {line}")
        for k in ("lora4_minus_matcher", "matcher_minus_opus", "matcher_minus_best_rule", "ordered_minus_orderfree",
                  "ordered_minus_ordered_permuted", "lora4_ordered_minus_shuffled"):
            if k in n:
                print(f"     {k}: {100 * n[k]['mean']:+.1f} [{100 * n[k]['ci95'][0]:+.1f}, {100 * n[k]['ci95'][1]:+.1f}]")
    print("verdict:", out["verdict"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("evaluate", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--threads", type=int, default=6)
    f = sub.add_parser("features")
    f.add_argument("--threads", type=int, default=6)
    f.add_argument("--reverse", action="store_true", help="take the chunks from the end (a second, parallel job)")
    for name in ("grid", "seeds", "pixels"):
        p = sub.add_parser(name)
        if name != "pixels":
            p.add_argument("--variant", choices=["ordered", "orderfree"], required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=6)
        p.add_argument("--feats", help="smoke tests only")
        p.add_argument("--runs-dir", help="smoke tests only")
        p.add_argument("--epochs", type=int, help="smoke tests only")
    args = ap.parse_args()
    {"features": features, "grid": grid, "seeds": seeds, "pixels": pixels, "evaluate": evaluate,
     "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
