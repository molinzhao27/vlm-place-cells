"""Navigation policies on our walks: ViNT, NoMaD and GNM (follow-up pilot WP, docs/experiments.md).

    python -m spatialcells.wm_vint gate --model vint [--data data/walks-roomReplicaApt0 --scene ReplicaApt0]
    python -m spatialcells.wm_vint acts --model vint --cond walk|single|shuffled
    python -m spatialcells.wm_vint acts --model vint --cond probes-single --data data/walks-p256
    python -m spatialcells.wm_vint twochoice --model vint

The Berkeley models (Shah et al.; robodhruv/visualnav-transformer, MIT) stay outside this repository: --repo is
the checkout (its train/ package is imported) and --weights holds vint.pth, nomad.pth and gnm.pth from the
README's Google Drive folder. Frames get the training preprocessing (center crop to 4:3, resize to the
config's image_size, ImageNet normalization). The context is the current frame and the context_size frames
before it (ViNT, GNM 5; NoMaD 3), oldest first; one walk step counts as one model step.

  gate       WP0: predicted temporal distance to the frame k = 0..20 steps ahead, for observations at steps
             5, 15, ..., 235 of every walk; Spearman rho with k, 95% CI over walks
             -> results/wm_pilot/gate_<model>_<scene>.json
  acts       WP1 units at every recorded layer -> data/acts/<model>-<cond>/L<k>.npy (frames.npy or probes.npy
             order). Conditions: walk (past walk frames), single (the frame repeated), shuffled (frames drawn
             at random from the same walk, seeded by walk and step), probes-single (--data with probes.npy;
             the probe repeated). ViNT and GNM take the current frame as the goal; NoMaD masks the goal.
             Layers: ViNT L0 current frame's EfficientNet-B0 feature (1280), L1-L4 MLP hidden units of
             transformer layers 1-4 at the current frame's token (2048), L5-L9 the output layers after the
             transformer (512/256/128/64/32, after ReLU). NoMaD L0 as ViNT, L1-L4 MLP hidden (1024) at the
             current token, L5 the pooled observation encoding (256). GNM L0 pooled observation-encoder
             features (1280), L1 compressed observation encoding (1024), L2-L3 joint layers (256, 32).
  twochoice  WP5: every registered lure-free trial. For each option X the observation is X with the walk
             frames before it, the goal the new view; the smaller predicted distance is the choice. Also
             no walk (X repeated) and reverse (the new view repeated as observation, X as goal)
             -> results/wm_pilot/twochoice_<model>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .extract import save
from .two_choice import trials_file, walk_order
from .wm_common import WM_ROOT, add_path, bootstrap_ci, device, safe_load

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ALLOW = ("torch.nn.modules.", "torch.optim.", "torch._C._nn.", "torchvision.models.", "torchvision.ops.",
         "efficientnet_pytorch.", "vint_train.models.", "numpy.core.multiarray.scalar", "numpy.dtype",
         "collections.defaultdict", "builtins.dict")
STUB = ("warmup_scheduler.",)
SETS = (["dissimilar", "roomB"] + [f"roomP{5000 + i}" for i in range(10)]
        + [f"roomReplicaApt{i}" for i in range(3)] + [f"roomHSSD{i}" for i in range(1, 9)])


def prep(frames, size):
    """uint8 frames (N, h, w, 3) -> float16 (N, 3, H, W): vint_train's resize_and_aspect_crop + normalization."""
    W, H = size
    out = np.empty((len(frames), 3, H, W), np.float16)
    for i, f in enumerate(frames):
        f = np.asarray(f)
        h, w = f.shape[:2]
        if w / h < 4 / 3:                              # TF.center_crop(img, (int(w / (4/3)), w))
            ch = int(w / (4 / 3))
            top = int(round((h - ch) / 2.0))
            f = f[top:top + ch]
        else:
            cw = int(h * 4 / 3)
            left = int(round((w - cw) / 2.0))
            f = f[:, left:left + cw]
        im = np.asarray(Image.fromarray(f).resize((W, H)), np.float32) / 255
        out[i] = ((im - MEAN) / STD).transpose(2, 0, 1)
    return torch.from_numpy(out)


class Policy:
    """One of the three models, with distance prediction and (optionally) unit recording."""

    def __init__(self, name, repo, weights, dev, record=False):
        add_path(os.path.join(repo, "train"))
        import yaml
        torch.backends.mha.set_fastpath_enabled(False)   # the fused encoder layer would bypass the hooks
        cfg = yaml.safe_load(open(os.path.join(repo, "train", "config", f"{name}.yaml")))
        self.name, self.cfg, self.dev = name, cfg, dev
        self.ctx = cfg["context_size"]
        self.size = tuple(cfg["image_size"])            # (width, height)
        ck = safe_load(os.path.join(weights, f"{name}.pth"), ALLOW, STUB)
        if name == "nomad":
            from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
            from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
            enc = replace_bn_with_gn(NoMaD_ViNT(obs_encoding_size=cfg["encoding_size"], context_size=self.ctx,
                                                mha_num_attention_heads=cfg["mha_num_attention_heads"],
                                                mha_num_attention_layers=cfg["mha_num_attention_layers"],
                                                mha_ff_dim_factor=cfg["mha_ff_dim_factor"]))
            model = NoMaD(vision_encoder=enc, noise_pred_net=torch.nn.Identity(),
                          dist_pred_net=DenseNetwork(embedding_dim=cfg["encoding_size"]))
            sd = {k: v for k, v in ck.items() if not k.startswith("noise_pred_net.")}   # the diffusion policy is unused
        else:
            if name == "vint":
                from vint_train.models.vint.vint import ViNT
                model = ViNT(context_size=self.ctx, len_traj_pred=cfg["len_traj_pred"], learn_angle=cfg["learn_angle"],
                             obs_encoder=cfg["obs_encoder"], obs_encoding_size=cfg["obs_encoding_size"],
                             late_fusion=cfg["late_fusion"], mha_num_attention_heads=cfg["mha_num_attention_heads"],
                             mha_num_attention_layers=cfg["mha_num_attention_layers"],
                             mha_ff_dim_factor=cfg["mha_ff_dim_factor"])
            else:
                from vint_train.models.gnm.gnm import GNM
                model = GNM(self.ctx, cfg["len_traj_pred"], cfg["learn_angle"], cfg["obs_encoding_size"],
                            cfg["goal_encoding_size"])
            src = ck["model"]
            sd = (src.module if hasattr(src, "module") else src).state_dict()
        model.load_state_dict(sd, strict=True)
        self.model = model.to(dev).eval()
        self.rec = {}
        if record:
            self._hooks()

    def _hooks(self):
        m, rec, ctx = self.model, self.rec, self.ctx

        def current_rows(key):                          # pooled encoder output, (ctx+1)*B rows, current frame last
            def f(_, __, out):
                B = len(out) // (ctx + 1)
                rec[key] = out.flatten(1)[ctx * B:]
            return f

        def token(key):
            def f(_, inp):
                rec[key] = inp[0][:, ctx]
            return f

        def output(key, relu=False):
            def f(_, __, out):
                rec[key] = F.relu(out) if relu else out
            return f
        if self.name == "vint":
            m.obs_encoder._avg_pooling.register_forward_hook(current_rows("L0"))
            for i, lay in enumerate(m.decoder.sa_decoder.layers):
                lay.linear2.register_forward_pre_hook(token(f"L{1 + i}"))
            for j, lin in enumerate(m.decoder.output_layers):
                lin.register_forward_hook(output(f"L{5 + j}", relu=True))
        elif self.name == "nomad":
            ve = m.vision_encoder
            ve.obs_encoder._avg_pooling.register_forward_hook(current_rows("L0"))
            for i, lay in enumerate(ve.sa_encoder.layers):
                lay.linear2.register_forward_pre_hook(token(f"L{1 + i}"))
            ve.register_forward_hook(output("L5"))
        else:
            m.obs_mobilenet.register_forward_hook(lambda _, __, out: rec.__setitem__(
                "L0", F.adaptive_avg_pool2d(out, 1).flatten(1)))
            m.compress_observation.register_forward_hook(output("L1"))
            m.linear_layers[1].register_forward_hook(output("L2"))
            m.linear_layers[3].register_forward_hook(output("L3"))

    @torch.no_grad()
    def __call__(self, obs, goal, mask_goal=False):
        """obs (B, ctx+1, 3, H, W), oldest first; goal (B, 3, H, W). Returns the predicted temporal distance (B,)."""
        self.rec.clear()
        obs = obs.to(self.dev, torch.float32).flatten(1, 2)
        goal = goal.to(self.dev, torch.float32)
        if self.name == "nomad":
            mask = torch.full((len(obs),), int(mask_goal), dtype=torch.long, device=self.dev)
            enc = self.model("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=mask)
            dist = self.model("dist_pred_net", obsgoal_cond=enc)
        else:
            dist, _ = self.model(obs, goal)
        return dist.flatten().float().cpu()


def run(pol, X, obs_idx, goal_idx, G=None, mask_goal=False, batch=512, record=False):
    """Distances for rows of index arrays into the preprocessed images X (and G for goals, default X)."""
    G = X if G is None else G
    dists, rec = [], {}
    for b0 in range(0, len(obs_idx), batch):
        o = torch.from_numpy(obs_idx[b0:b0 + batch])
        d = pol(X[o], G[torch.from_numpy(goal_idx[b0:b0 + batch])], mask_goal)
        dists.append(d)
        if record:
            assert pol.rec, "no units recorded"
            for k, v in pol.rec.items():
                rec.setdefault(k, []).append(v.float().cpu().numpy().astype(np.float16))
    return torch.cat(dists).numpy(), {k: np.concatenate(v) for k, v in rec.items()}


def context(fi, s, ctx):
    """Frame rows of the context ending at walk step s (earliest frame repeated before the walk's start)."""
    return [fi[max(s - j, 0)] for j in range(ctx, -1, -1)]


def gate(pol, args):
    from scipy.stats import spearmanr
    frames = np.load(f"{args.data}/frames.npy", mmap_mode="r")
    meta = dict(np.load(f"{args.data}/meta.npz"))
    order = walk_order(meta)
    t0 = time.time()
    X = prep(frames, pol.size)
    obs, goal, ks, walks = [], [], [], []
    for w, fi in order.items():
        for t in range(5, len(fi) - 20, 10):
            for k in range(21):
                obs.append(context(fi, t, pol.ctx))
                goal.append(fi[t + k])
                ks.append(k)
                walks.append(w)
    obs, goal, ks, walks = np.array(obs), np.array(goal), np.array(ks), np.array(walks)
    d, _ = run(pol, X, obs, goal)
    rho = float(spearmanr(d, ks).statistic)
    by_walk = {w: np.flatnonzero(walks == w) for w in order}
    ci = bootstrap_ci(lambda ws: spearmanr(d[np.concatenate([by_walk[w] for w in ws])],
                                           ks[np.concatenate([by_walk[w] for w in ws])]).statistic, list(order))
    per_walk = [float(spearmanr(d[i], ks[i]).statistic) for i in by_walk.values()]
    res = {"model": pol.name, "scene": args.scene, "n_pairs": int(len(d)), "n_walks": len(order),
           "spearman_rho": rho, "rho_ci95": ci, "rho_per_walk_median": float(np.median(per_walk)),
           "pass": bool(rho >= 0.5),
           "mean_pred_by_k": [float(d[ks == k].mean()) for k in range(21)],
           "sd_pred_by_k": [float(d[ks == k].std()) for k in range(21)]}
    os.makedirs("results/wm_pilot", exist_ok=True)
    json.dump(res, open(f"results/wm_pilot/gate_{pol.name}_{args.scene}.json", "w"), indent=1)
    print(f"gate {pol.name} {args.scene}: rho {rho:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], per-walk median "
          f"{res['rho_per_walk_median']:.3f}; mean prediction k=0 {res['mean_pred_by_k'][0]:.2f}, k=10 "
          f"{res['mean_pred_by_k'][10]:.2f}, k=20 {res['mean_pred_by_k'][20]:.2f}; {time.time() - t0:.0f}s", flush=True)


def acts(pol, args):
    probes = args.cond.startswith("probes")
    frames = np.load(f"{args.data}/{'probes' if probes else 'frames'}.npy", mmap_mode="r")
    meta = dict(np.load(f"{args.data}/{'probes_meta' if probes else 'meta'}.npz"))
    t0 = time.time()
    X = prep(frames, pol.size)
    n, ctx = len(frames), pol.ctx
    if args.cond in ("single", "probes-single"):
        obs = np.repeat(np.arange(n)[:, None], ctx + 1, 1)
    else:
        order = walk_order(meta)
        obs = np.zeros((n, ctx + 1), int)
        for w, fi in order.items():
            for s, r in enumerate(fi):
                if args.cond == "walk":
                    obs[r] = context(fi, s, ctx)
                else:                                   # shuffled: random other frames of the same walk
                    rng = np.random.default_rng(int(w) * 1000 + s)
                    others = rng.choice(np.delete(np.arange(len(fi)), s), ctx, replace=False)
                    obs[r] = list(fi[others]) + [r]
    _, rec = run(pol, X, obs, np.arange(n), mask_goal=pol.name == "nomad", record=True)
    tag = f"{pol.name}-{args.cond}" + (f"-{args.scene}" if args.scene != "A" else "")
    save(f"data/acts/{tag}", rec)
    print(f"acts {tag}: {n} rows, " + ", ".join(f"{k} {v.shape[1]}" for k, v in sorted(rec.items(), key=lambda kv: int(kv[0][1:])))
          + f"; {time.time() - t0:.0f}s", flush=True)


def twochoice(pol, args):
    os.makedirs("results/wm_pilot", exist_ok=True)
    path = f"results/wm_pilot/twochoice_{pol.name}.jsonl"
    done = {(r["set"], r["id"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    out = open(path, "a")
    for name in args.sets:
        data = "data/walks" if name == "dissimilar" else f"data/walks-{name}"
        trials = [t for t in json.load(open(trials_file(name))) if (name, t["id"]) not in done]
        if not trials:
            continue
        t0 = time.time()
        frames = np.load(f"{data}/frames.npy", mmap_mode="r")
        meta = dict(np.load(f"{data}/meta.npz"))
        order = walk_order(meta)
        need = sorted({int(order[t["walk"]][max(o - 1 - j, 0)]) for t in trials for o in (t["near"], t["far"])
                       for j in range(pol.ctx + 1)})
        X = torch.zeros((len(frames), 3, pol.size[1], pol.size[0]), dtype=torch.float16)
        X[need] = prep(frames[need], pol.size)         # only the frames the trials use
        pids = sorted({t["probe"] for t in trials})
        P = prep(np.load(f"{data}/probes.npy", mmap_mode="r")[pids], pol.size)
        prow = {p: i for i, p in enumerate(pids)}
        n_ctx = pol.ctx + 1
        rows = {"walk": ([], [], False), "nowalk": ([], [], False), "reverse": ([], [], True)}
        for t in trials:
            fi = order[t["walk"]]
            for opt in (t["near"], t["far"]):
                s = opt - 1                             # 1-based frame numbers in the trial files
                rows["walk"][0].append(context(fi, s, pol.ctx))
                rows["nowalk"][0].append([fi[s]] * n_ctx)
                rows["reverse"][0].append([prow[t["probe"]]] * n_ctx)
                rows["walk"][1].append(prow[t["probe"]])
                rows["nowalk"][1].append(prow[t["probe"]])
                rows["reverse"][1].append(fi[s])
        d = {}
        for cond, (obs, goal, rev) in rows.items():
            obs, goal = np.array(obs), np.array(goal)
            d[cond] = (run(pol, P, obs, goal, G=X)[0] if rev else run(pol, X, obs, goal, G=P)[0]).reshape(-1, 2)
        for i, t in enumerate(trials):
            rec = {"set": name, "id": t["id"], "walk": t["walk"], "familiar": t["familiar"]}
            for cond, dd in d.items():
                dn, df = float(dd[i, 0]), float(dd[i, 1])
                rec[cond] = {"d_near": dn, "d_far": df, "correct": 1.0 if dn < df else 0.5 if dn == df else 0.0}
            out.write(json.dumps(rec) + "\n")
        out.flush()
        acc = {c: np.mean([1.0 if a < b else 0.5 if a == b else 0.0 for a, b in dd]) for c, dd in d.items()}
        print(f"twochoice {pol.name} {name}: {len(trials)} trials, " + ", ".join(f"{c} {v:.1%}" for c, v in acc.items())
              + f"; {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gate", "acts", "twochoice"])
    ap.add_argument("--model", choices=["vint", "nomad", "gnm"], required=True)
    ap.add_argument("--repo", default=os.path.join(WM_ROOT, "visualnav-transformer"))
    ap.add_argument("--weights", default=os.path.join(WM_ROOT, "weights", "vint"))
    ap.add_argument("--data", default="data/walks")
    ap.add_argument("--scene", default="A", help="scene label for output names")
    ap.add_argument("--cond", choices=["walk", "single", "shuffled", "probes-single"])
    ap.add_argument("--sets", nargs="+", default=SETS, help="twochoice: trial sets (two_choice.py names)")
    ap.add_argument("--device")
    args = ap.parse_args()
    dev = device(args.device)
    pol = Policy(args.model, args.repo, args.weights, dev, record=args.what == "acts")
    print(f"{args.model}: context {pol.ctx}, image {pol.size}, device {dev}", flush=True)
    {"gate": gate, "acts": acts, "twochoice": twochoice}[args.what](pol, args)


if __name__ == "__main__":
    main()
