"""Positive control for the grid-cell scorer: a path-integrating RNN (Sorscher et al. 2019, 2023).

    python -m spatialcells.grid_rnn --name s093 --step 0.093 --sub 3 --train --score --dense 1000
    python -m spatialcells.grid_rnn --synthetic                   # the scorer on ideal grid cells

--train (GPU, ~10-20 min), --score (analyze.py's scorer on its units along data/walks), --dense N
(same scorer, N new walks), --synthetic (calibration on known grid cells, no RNN). Runs are named
(--name): data/grid_rnn_<name>/, results/grid_rnn.json[<name>], results/grid_rnn_<name>_*.png.

The model follows ganguli-lab/grid-pattern-formation: 4096 ReLU units with no bias, a linear
encoder that sets the first hidden state from place-cell activity at the start position, then
only the step-by-step displacement (dx, dy) as input. A linear decoder is trained to output the
current place-cell activity (softmax cross-entropy). 512 place cells with difference-of-Gaussians
tuning (centre width 0.12, surround variance x2, made non-negative and summing to 1), weight
decay 1e-4 on the recurrent weights, Adam lr 1e-4, batch 200, 20-step sequences. One change: the
weight decay starts after 5,000 steps. From the first step it shrank the recurrent weights to zero
before the task gradient grew (loss stuck at ln 512); without it the error falls within 3,000 steps.

Their box is 2.2 m and our room is 10 m, so positions are divided by 10/2.2 inside the network:
the place cells are 0.55 m wide in the room. Training walks are room.foraging_walk with other
seeds than data/walks, with --step metres per step: the "base" run used the default 0.22 m (and
TF32 matmuls); 0.093 m matches Sorscher's trajectories (2 cm steps in the 2.2 m box). --traj
sorscher trains on their own trajectory generator instead, scaled to the room. The test walks are data/walks' own 48 x 256 trajectories, so the occupancy
is the same one every model was scored on, and the hidden units go through analyze.analyze_layer
unchanged (rate maps, autocorrelogram, gridness, shuffle threshold, stability).

Two test modes: "anchored" re-sets the state from the true position every 20 network steps (the
training regime; real grid cells are corrected by landmarks too) and "free" path-integrates all
255 steps. With --sub k each data/walks step is fed as k equal sub-steps (so the inputs match the
training step size) and the state is read at the frames. Step 0 of each walk only sets the start
state, so the scored walks have 255 steps.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import analyze
from .room import L, foraging_walk

OUT = "data/grid_rnn"
SCALE = L / 2.2          # room metres per network unit
NP, NG, SIGMA, SURROUND, T = 512, 4096, 0.12, 2.0, 20


class PlaceCells:
    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.c = (torch.rand(NP, 2, generator=g) * L / SCALE).to(analyze.dev)    # centres, network units

    def __call__(self, pos):
        """pos (..., 2) in network units -> (..., NP) DoG activity, non-negative, sums to 1."""
        d2 = ((pos[..., None, :] - self.c) ** 2).sum(-1)
        out = F.softmax(-d2 / (2 * SIGMA ** 2), -1) - F.softmax(-d2 / (2 * SURROUND * SIGMA ** 2), -1)
        out = out + out.min(-1, keepdim=True).values.abs()
        return out / out.sum(-1, keepdim=True)

    def decode(self, logits, k=3):
        """Position estimate: mean centre of the k most active predicted place cells."""
        return self.c[logits.topk(k, -1).indices].mean(-2)


class PathRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(NP, NG, bias=False)
        self.rnn = nn.RNN(2, NG, nonlinearity="relu", bias=False, batch_first=True)
        self.decoder = nn.Linear(NG, NP, bias=False)

    def g(self, v, p0):
        """v (B, T, 2) displacements, p0 (B, NP) start place-cell activity -> hidden states (B, T, NG)."""
        return self.rnn(v, self.encoder(p0)[None])[0]


def sorscher_walks(n, steps, rng):
    """Their trajectory generator (ganguli-lab trajectory_generator.py: Rayleigh speeds, Gaussian turning,
    slowing and turning at walls), run in their 2.2 m box and returned in room metres. No pillars."""
    dt, sigma, b, border, half = 0.02, 5.76 * 2, 0.13 * 2 * np.pi, 0.03, 1.1
    pos = np.zeros((n, steps, 2))
    pos[:, 0] = rng.uniform(-half, half, (n, 2))
    hd = rng.uniform(0, 2 * np.pi, n)
    for t in range(steps - 1):
        x, y = pos[:, t].T
        dists = np.stack([half - x, half - y, half + x, half + y])
        theta = np.argmin(dists, 0) * np.pi / 2
        a_wall = np.mod(np.mod(hd, 2 * np.pi) - theta + np.pi, 2 * np.pi) - np.pi
        near = (dists.min(0) < border) & (np.abs(a_wall) < np.pi / 2)
        turn = np.where(near, np.sign(a_wall) * (np.pi / 2 - np.abs(a_wall)), 0.0)
        v = rng.rayleigh(b, n) * np.where(near, 0.25, 1.0)
        pos[:, t + 1] = pos[:, t] + (v * dt)[:, None] * np.stack([np.cos(hd), np.sin(hd)], 1)
        hd = hd + turn + dt * rng.normal(0, sigma, n)
    return (pos + half) * SCALE


def train_walks(step, n=2000, steps=1000, seed=1000, traj="foraging"):
    path = f"{OUT}/train_walks.npy"
    if os.path.exists(path):
        return np.load(path)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    if traj == "sorscher":
        w = sorscher_walks(n, steps, rng).astype(np.float32)
    else:
        w = np.stack([foraging_walk(steps, rng, step=step)[:, :2] for _ in range(n)]).astype(np.float32)
    np.save(path, w)
    print(f"{n} training walks x {steps} steps in {time.time() - t0:.0f}s", flush=True)
    return w


def train(args):
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    walks = torch.from_numpy(train_walks(args.step, traj=args.traj)).to(analyze.dev) / SCALE
    pc, model = PlaceCells(), PathRNN().to(analyze.dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log = []
    t0 = time.time()
    for it in range(1, args.steps + 1):
        w = torch.randint(len(walks), (args.batch,), device=analyze.dev)
        s = torch.randint(walks.shape[1] - T, (args.batch,), device=analyze.dev)
        pos = walks[w[:, None], s[:, None] + torch.arange(T + 1, device=analyze.dev)]      # (B, T+1, 2)
        logits = model.decoder(model.g(pos[:, 1:] - pos[:, :-1], pc(pos[:, 0])))
        ce = -(pc(pos[:, 1:]) * F.log_softmax(logits, -1)).sum(-1).mean()
        loss = ce + (args.weight_decay if it > args.wd_start else 0.0) * (model.rnn.weight_hh_l0 ** 2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 1000 == 0 or it == 1:
            err = ((pc.decode(logits) - pos[:, 1:]).norm(dim=-1).mean() * SCALE).item()
            log.append(dict(step=it, ce=ce.item(), loss=loss.item(), err_m=err))
            print(f"step {it}: ce {ce.item():.3f} loss {loss.item():.3f} decoded error {err:.2f} m "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if it % 10000 == 0 or it == args.steps:
            torch.save(model.state_dict(), f"{OUT}/model.pt")
            json.dump(log, open(f"{OUT}/train_log.json", "w"))


@torch.no_grad()
def hidden(model, pc, xy, chunk, sub=1):
    """xy (W, S, 2) network units -> hidden states (W, S-1, NG) at steps 1..S-1, re-anchored every chunk
    steps; each step is fed as sub equal sub-steps."""
    W, S, _ = xy.shape
    f = torch.arange(1, sub + 1, device=xy.device) / sub
    xs = torch.cat([xy[:, :1], (xy[:, :-1, None] + f[:, None] * (xy[:, 1:, None] - xy[:, :-1, None])).reshape(W, -1, 2)], 1)
    out, logits = [], []
    for a in range(0, S - 1, chunk):
        A, B = a * sub, min(a + chunk, S - 1) * sub
        g = model.g(xs[:, A + 1:B + 1] - xs[:, A:B], pc(xs[:, A]))[:, sub - 1::sub]
        out.append(g)
        logits.append(model.decoder(g))
    return torch.cat(out, 1), torch.cat(logits, 1)


def score(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = dict(np.load("data/walks/meta.npz"))
    W, S = int(meta["walk"].max()) + 1, int(meta["step"].max()) + 1
    xy = torch.tensor(np.stack([meta["x"], meta["y"]], 1).reshape(W, S, 2), dtype=torch.float32, device=analyze.dev) / SCALE
    keep = meta["step"] >= 1
    sub = {k: v[keep] for k, v in meta.items()}
    sub["step"] = sub["step"] - 1
    pc, model = PlaceCells(), PathRNN().to(analyze.dev)
    model.load_state_dict(torch.load(f"{OUT}/model.pt", map_location=analyze.dev))
    results = {}
    for mode, chunk in (("anchored", max(1, round(T / args.sub))), ("free", S)):
        g, logits = hidden(model, pc, xy, chunk, args.sub)
        err = (pc.decode(logits) - xy[:, 1:]).norm(dim=-1) * SCALE
        X = g.reshape(-1, NG)
        stats, pu = analyze.analyze_layer(X, sub, np.random.default_rng(0))
        grid = pu["grid"][torch.isfinite(pu["grid"])]
        stats.update(decoded_error_median_m=err.median().item(), decoded_error_last_step_m=err[:, -1].median().item(),
                     n_grid=int(pu["gridc"].sum()), gridness_quantiles={q: float(grid.quantile(q)) for q in (0.5, 0.9, 0.99)},
                     frac_gridness_above_threshold=float((grid > stats["grid_threshold"]).float().mean()),
                     frac_silent=float(1 - stats["n_poles_active"] / NG))
        results[mode] = stats
        print(mode, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()}), flush=True)

        # top units by gridness: rate map + autocorrelogram; and the gridness histogram against the threshold
        top = torch.argsort(torch.nan_to_num(pu["grid"], -2), descending=True)[:8].tolist()
        fig, axs = plt.subplots(3, 8, figsize=(20, 7.5))
        v = pu["valid"].cpu().numpy()
        for j, u in enumerate(top):
            m = pu["maps"][u].cpu().numpy().copy()
            m[~v] = np.nan
            axs[0, j].imshow(m, origin="lower", extent=(0, L, 0, L), cmap="jet")
            axs[0, j].set_title(f"unit {int(pu['pole'][u])}: grid {pu['grid'][u]:.2f}\nstab {pu['stab'][u]:.2f}", fontsize=8)
            axs[1, j].imshow(analyze.autocorr(pu["maps"][u:u + 1], pu["valid"])[0].cpu().numpy(), origin="lower", cmap="jet")
            for a in axs[:2, j]:
                a.set_xticks([]), a.set_yticks([])
        for a in axs[2]:
            a.axis("off")
        ax = fig.add_subplot(3, 1, 3)
        ax.hist(grid.cpu().numpy(), bins=80, color="0.4")
        ax.axvline(stats["grid_threshold"], color="r", label=f"99th shuffle percentile ({stats['grid_threshold']:.2f})")
        ax.set_xlabel("gridness"), ax.set_ylabel("units"), ax.legend()
        fig.suptitle(f"Path-integrating RNN, {mode}: {stats['frac_grid']:.0%} of active units pass the grid criteria "
                     f"(decoded error {stats['decoded_error_median_m']:.2f} m)")
        fig.tight_layout()
        fig.savefig(f"results/grid_rnn_{args.name}_{mode}.png", dpi=80)
        plt.close(fig)
    results["train_log_last"] = json.load(open(f"{OUT}/train_log.json"))[-1]
    res = json.load(open("results/grid_rnn.json")) if os.path.exists("results/grid_rnn.json") else {}
    res.setdefault(args.name, {}).update(results, step_m=args.step, sub=args.sub)
    json.dump(res, open("results/grid_rnn.json", "w"), indent=1)


def dense(args):
    """The same rate maps, autocorrelograms and gridness (analyze.py), but from --dense new walks
    (256 steps, anchored) instead of data/walks' 48, with the same circular-shift null (10 shuffles,
    pooled 99th percentile). Separates "the RNN has no grid cells" from "48 walks are too few to see them"."""
    rng = np.random.default_rng(7)
    walks = np.stack([foraging_walk(256, rng, step=args.step)[:, :2] for _ in range(args.dense)]).astype(np.float32)
    xy = torch.from_numpy(walks).to(analyze.dev) / SCALE
    pc, model = PlaceCells(), PathRNN().to(analyze.dev)
    model.load_state_dict(torch.load(f"{OUT}/model.pt", map_location=analyze.dev))
    pos = walks[:, 1:].reshape(-1, 2)
    ix = np.clip((pos[:, 0] / L * analyze.NB).astype(int), 0, analyze.NB - 1)
    iy = np.clip((pos[:, 1] / L * analyze.NB).astype(int), 0, analyze.NB - 1)
    bins = torch.from_numpy(iy * analyze.NB + ix).to(analyze.dev)
    W, S = len(walks), walks.shape[1] - 1
    offs = [rng.integers(32, S - 32, size=W) for _ in range(10)]
    perms = [torch.from_numpy((np.arange(W)[:, None] * S + (np.arange(S)[None] + o[:, None]) % S).ravel()).to(analyze.dev)
             for o in offs]
    nb2 = analyze.NB ** 2
    sums = torch.zeros(1 + len(perms), nb2, NG, device=analyze.dev)
    occ = torch.bincount(bins, minlength=nb2).float()
    for w0 in range(0, W, 50):                    # accumulate activity per bin, real and shuffled positions
        g, _ = hidden(model, pc, xy[w0:w0 + 50], T)
        R = g.reshape(-1, NG)
        sel = torch.arange(w0 * S, (w0 + len(g)) * S, device=analyze.dev)
        sums[0].index_add_(0, bins[sel], R)
        for k, p in enumerate(perms):
            sums[1 + k].index_add_(0, bins[p[sel]], R)
    sm = lambda x: F.conv2d(x.reshape(-1, 1, analyze.NB, analyze.NB), analyze.KER, padding=3)[:, 0]
    den = sm(occ[None]).clamp_min(1e-6)
    valid = occ.reshape(analyze.NB, analyze.NB) >= analyze.MIN_OCC
    grid_of = lambda s: torch.cat([analyze.gridness(analyze.autocorr(sm(s.T[i:i + 1024].contiguous()) / den, valid))
                                   for i in range(0, NG, 1024)])
    active = sums[0].sum(0) > 0
    grid = grid_of(sums[0])[active]
    null = torch.cat([grid_of(sums[1 + k])[active] for k in range(len(perms))])
    thr = torch.quantile(null[torch.isfinite(null)], 0.99).item()
    out = {"walks": W, "samples_per_bin_median": float(occ.median()), "grid_threshold": thr,
           "frac_grid_above_threshold": float((grid > thr).float().mean()),
           **{f"frac_gridness_above_{c}": float((grid > c).float().mean()) for c in (0.3, 0.5)},
           "gridness_quantiles": {q: float(grid[torch.isfinite(grid)].quantile(q)) for q in (0.5, 0.9, 0.99)}}
    print("dense", json.dumps(out), flush=True)
    res = json.load(open("results/grid_rnn.json")) if os.path.exists("results/grid_rnn.json") else {}
    res.setdefault(args.name, {})[f"dense_{W}"] = out
    json.dump(res, open("results/grid_rnn.json", "w"), indent=1)
    maps = sm(sums[0].T.contiguous()) / den
    top = torch.nonzero(active).squeeze(1)[torch.argsort(torch.nan_to_num(grid, -2), descending=True)[:16]]
    torch.save({"maps": maps.cpu(), "valid": valid.cpu(), "grid": grid_of(sums[0]).cpu(), "top": top.cpu(), "threshold": thr},
               f"{OUT}/dense_maps_{W}.pt")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 8, figsize=(20, 10))
    allg = grid_of(sums[0])
    for j, u in enumerate(top.tolist()):
        r, c = 2 * (j // 8), j % 8
        m = maps[u].cpu().numpy().copy()
        m[~valid.cpu().numpy()] = np.nan
        axs[r, c].imshow(m, origin="lower", extent=(0, L, 0, L), cmap="jet")
        axs[r, c].set_title(f"unit {u}: grid {allg[u]:.2f}", fontsize=8)
        axs[r + 1, c].imshow(analyze.autocorr(maps[u:u + 1], valid)[0].cpu().numpy(), origin="lower", cmap="jet")
        for a in (axs[r, c], axs[r + 1, c]):
            a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"Path-integrating RNN, {W} walks ({out['samples_per_bin_median']:.0f} samples per bin): "
                 f"{out['frac_grid_above_threshold']:.0%} of units above the shuffle threshold {thr:.2f}")
    fig.tight_layout()
    fig.savefig(f"results/grid_rnn_{args.name}_dense{W}.png", dpi=80)


def synthetic(args):
    """Calibrate the scorer with known grid cells: ideal hexagonal grids (Gaussian fields of width
    0.16 x spacing on a triangular lattice, random orientation and phase) at spacings 1-5 m, read
    out along data/walks' own positions with Gaussian noise (SD = 0, 0.5 or 1 x the peak rate),
    mixed with as many place-like distractors (1-3 random fields). analyze.analyze_layer scores
    them all unchanged; reports the fraction of grid units it flags per spacing and noise level."""
    rng = np.random.default_rng(3)
    meta = dict(np.load("data/walks/meta.npz"))
    pos = np.stack([meta["x"], meta["y"]], 1)
    spacings, noises, per = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0), (0.0, 0.5, 1.0), 40
    cols, labels = [], []
    for lam in spacings:
        for nz in noises:
            for _ in range(per):
                ori, phase = rng.uniform(0, np.pi / 3), rng.uniform(0, L, 2)
                a1 = lam * np.array([np.cos(ori), np.sin(ori)])
                a2 = lam * np.array([np.cos(ori + np.pi / 3), np.sin(ori + np.pi / 3)])
                k = np.arange(-12, 13)
                pts = phase + k[:, None, None] * a1 + k[None, :, None] * a2
                pts = pts.reshape(-1, 2)
                pts = pts[(np.abs(pts - L / 2) < L / 2 + 2 * lam).all(1)]
                d2 = ((pos[:, None] - pts[None]) ** 2).sum(-1)
                r = np.exp(-d2 / (2 * (0.16 * lam) ** 2)).sum(1)
                cols.append(r + nz * r.max() * rng.normal(size=len(r)))
                labels.append((lam, nz))
    for _ in range(len(cols)):                             # place-like distractors
        r = np.zeros(len(pos))
        for _ in range(rng.integers(1, 4)):
            c, s = rng.uniform(0, L, 2), rng.uniform(0.5, 2.0)
            r += np.exp(-((pos - c) ** 2).sum(1) / (2 * s ** 2))
        cols.append(r + rng.uniform(0, 1) * r.max() * rng.normal(size=len(r)))
        labels.append((0.0, -1.0))
    X = torch.tensor(np.stack(cols, 1), dtype=torch.float32, device=analyze.dev)
    stats, pu = analyze.analyze_layer(X, meta, np.random.default_rng(0))
    pole = pu["pole"].cpu().numpy()
    gridc, grid = pu["gridc"].cpu().numpy(), pu["grid"].cpu().numpy()
    unit = pole % X.shape[1]
    out = {"grid_threshold": stats["grid_threshold"], "rows": []}
    for lam in spacings:
        for nz in noises:
            ids = [i for i, l in enumerate(labels) if l == (lam, nz)]
            m = np.isin(unit, ids) & (pole < X.shape[1])         # positive poles
            out["rows"].append({"spacing_m": lam, "noise": nz, "frac_detected": float(gridc[m].mean()),
                                "gridness_median": float(np.nanmedian(grid[m]))})
            print(f"spacing {lam} m, noise {nz}: {gridc[m].mean():.0%} flagged as grid cells, "
                  f"median gridness {np.nanmedian(grid[m]):.2f}", flush=True)
    m = np.isin(unit, [i for i, l in enumerate(labels) if l[1] < 0])
    out["distractors_flagged"] = float(gridc[m].mean())
    print(f"threshold {stats['grid_threshold']:.2f}; place-like distractors flagged: {gridc[m].mean():.1%}", flush=True)
    res = json.load(open("results/grid_rnn.json")) if os.path.exists("results/grid_rnn.json") else {}
    res["synthetic"] = out
    json.dump(res, open("results/grid_rnn.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="calibrate the scorer on ideal grid cells")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--dense", type=int, help="score with this many new walks instead (see dense())")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--wd-start", type=int, default=5000, help="weight decay only after this many steps")
    ap.add_argument("--name", default="base")
    ap.add_argument("--step", type=float, default=0.22, help="training (and --dense) walk step, metres")
    ap.add_argument("--traj", default="foraging", choices=["foraging", "sorscher"], help="training trajectories")
    ap.add_argument("--sub", type=int, default=1, help="sub-steps per data/walks step when scoring")
    ap.add_argument("--tf32", action="store_true", help="TF32 matmuls (the base run used them)")
    args = ap.parse_args()
    global OUT
    OUT = "data/grid_rnn" + ("" if args.name == "base" else f"_{args.name}")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = args.tf32
    if args.train:
        train(args)
    if args.score:
        score(args)
    if args.dense:
        dense(args)
    if args.synthetic:
        synthetic(args)


if __name__ == "__main__":
    main()
