"""Score every unit for place, grid and head-direction tuning, and decode position.

    python -m spatialcells.analyze --tags pixels dinov2 dinov2-random qwen-single qwen-walk ...

Units are neuron "poles": relu(+a) and relu(-a) of each MLP neuron are scored
separately as non-negative firing rates (SwiGLU neurons are signed, and each sign
can carry its own job). Poles that are silent in >99% of frames are dropped.

Single-unit criteria follow rodent practice, with nulls from circularly shifting
each walk's activity against its positions (100 shuffles):
  place  Skaggs spatial information > the unit's own 99th shuffle percentile,
         split-half (even vs odd walks) map stability r > 0.5, 1-3 fields
         (>= 3 bins at >50% of the map's range) covering <= 40% of the arena
  grid   gridness (60/120 vs 30/90/150 deg rotational symmetry of the map's
         autocorrelogram, max over annuli) > the layer's pooled 99th shuffle
         percentile, and stability r > 0.5
  HD     mean vector length of the heading tuning curve > own 99th shuffle
         percentile, and tuning-curve stability r > 0.5
Heading invariance: correlation between the maps built only from frames facing
one half of the compass and from frames facing the opposite half (N vs S, E vs W).
A real place cell fires at its spot whichever way the animal faces (invariance
close to its stability); a "view cell" driven by what is on screen does not.

Population decoding: ridge regression from a layer's neurons to (x, y), trained
on even walks and tested on odd walks, either within the same facing half or
across to the opposite facing half (the allocentric test).

--quadrant (experiment 13e) adds the same invariance and decoding measures on
opposite 90-degree sectors centred on E, N, W, S (N vs S, E vs W; no shared view),
as in heading_matrix, and writes results/summary_quadrant.json without figures.

--arena (follow-up pilot WP, amendment 2) is for scenes other than room A's 10 m box (the
photoreal apartments): the grid is a square of 0.5 m bins laid over the bounding box of the
positions in --data's meta.npz, its corner on the 0.5 m lattice through (0, 0), so walk and
probe arenas of one scene share bin edges. Bins the walk never visits (walls, rooms outside
the walks, the square's padding) drop out of every score through the occupancy mask that room
A already uses (valid = at least MIN_OCC frames). In room A the box gives the default grid.
Without --arena nothing changes.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

NB = 20           # spatial bins per side (0.5 m)
NH = 36           # heading bins (10 deg)
L = 10.0
X0 = Y0 = 0.0     # the grid's corner (room A's); set_arena moves it, NB and L to another scene's box
MIN_OCC = 3
N_SHUF = 100
N_SHUF_GRID = 10
dev = "cuda" if torch.cuda.is_available() else "cpu"


def _kernel(sigma=1.0, r=3):
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g[:, None] * g[None, :]
    return (g / g.sum())[None, None].to(dev)


KER = _kernel()


def set_arena(meta, bin_m=0.5):
    """--arena: a square grid of bin_m bins over the bounding box of meta's positions (the corner on the bin_m
    lattice through the origin). Returns the grid, which is recorded with every score."""
    global NB, L, X0, Y0
    X0, Y0 = (float(np.floor(meta[k].min() / bin_m) * bin_m) for k in ("x", "y"))
    NB = int(np.ceil(max(meta["x"].max() - X0, meta["y"].max() - Y0) / bin_m))
    L = NB * bin_m
    return {"x0": X0, "y0": Y0, "bins_per_side": NB, "bin_m": bin_m}


def spatial_bins(meta):
    """Spatial bin id of every frame (row = y, col = x)."""
    ix = np.clip(((meta["x"] - X0) / L * NB).astype(int), 0, NB - 1)
    iy = np.clip(((meta["y"] - Y0) / L * NB).astype(int), 0, NB - 1)
    return torch.from_numpy(iy * NB + ix).to(dev)


# ---------------------------------------------------------------- maps and scores
def rate_maps(R, bins, sel=None):
    """R (N, U) non-negative rates, bins (N,) spatial bin ids -> smoothed maps (U, NB, NB), occupancy (NB, NB)."""
    if sel is not None:
        R, bins = R[sel], bins[sel]
    oh = F.one_hot(bins, NB * NB).float()
    occ = oh.sum(0).reshape(1, 1, NB, NB)
    s = (oh.T @ R).T.reshape(-1, 1, NB, NB)
    maps = F.conv2d(s, KER, padding=3) / F.conv2d(occ, KER, padding=3).clamp_min(1e-6)
    return maps[:, 0], occ[0, 0]


def tuning_curves(R, hbins, sel=None):
    if sel is not None:
        R, hbins = R[sel], hbins[sel]
    oh = F.one_hot(hbins, NH).float()
    return (oh.T @ R).T / oh.sum(0).clamp_min(1)


def spatial_info(maps, occ, valid):
    p = occ[valid] / occ[valid].sum()
    r = maps[:, valid].clamp_min(0)
    ratio = r / (r * p).sum(1, keepdim=True).clamp_min(1e-12)
    return (p * ratio * torch.log2(ratio.clamp_min(1e-12))).sum(1)


def mvl(tc):
    ang = (torch.arange(NH, device=tc.device) + 0.5) * 2 * np.pi / NH
    z = (tc * torch.exp(1j * ang)).sum(1)
    return z.abs() / tc.sum(1).clamp_min(1e-12)


def corr_rows(a, b):
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)


def map_corr(m1, m2, valid):
    return corr_rows(m1[:, valid], m2[:, valid])


def autocorr(maps, valid):
    """Normalised spatial autocorrelogram (U, 2NB-1, 2NB-1), NaN where overlap < 20 bins."""
    v = valid.float()
    m = maps * v
    mean = m.sum((1, 2), keepdim=True) / v.sum()
    m0 = (maps - mean) * v
    S = 2 * NB - 1
    fm = torch.fft.rfft2(m0, s=(S, S))
    fv = torch.fft.rfft2(v, s=(S, S))
    num = torch.fft.fftshift(torch.fft.irfft2(fm * fm.conj(), s=(S, S)), dim=(-2, -1))
    cnt = torch.fft.fftshift(torch.fft.irfft2(fv * fv.conj(), s=(S, S)), dim=(-2, -1)).round()
    var = (m0 ** 2).sum((1, 2), keepdim=True) / v.sum()
    ac = num / cnt.clamp_min(1) / var.clamp_min(1e-12)
    return torch.where(cnt >= 20, ac, torch.nan)


def _rotate(ac, deg):
    th = np.deg2rad(deg)
    A = torch.tensor([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0]], dtype=torch.float32, device=ac.device)
    grid = F.affine_grid(A[None].expand(len(ac), 2, 3), (len(ac), 1, *ac.shape[1:]), align_corners=True)
    nanmask = torch.isnan(ac)
    out = F.grid_sample(torch.nan_to_num(ac)[:, None], grid, align_corners=True)[:, 0]
    bad = F.grid_sample(nanmask.float()[:, None], grid, align_corners=True)[:, 0] > 0
    return torch.where(bad, torch.nan, out)


def gridness(ac):
    """Max over annuli of min(r60, r120) - max(r30, r90, r150) (Sargolini 2006 / Banino 2018)."""
    S = ac.shape[-1]
    c = (S - 1) / 2
    yy, xx = torch.meshgrid(torch.arange(S, device=ac.device), torch.arange(S, device=ac.device), indexing="ij")
    rad = torch.hypot(yy - c, xx - c)
    rots = {d: _rotate(ac, d) for d in (30, 60, 90, 120, 150)}
    best = torch.full((len(ac),), -2.0, device=ac.device)
    for r_in in (2, 3, 4):
        for r_out in (7, 9, 11, 13, 15, 17):
            ann = (rad >= r_in) & (rad <= r_out)
            cs = {}
            for d, rot in rots.items():
                a, b = ac[:, ann], rot[:, ann]
                ok = ~(torch.isnan(a) | torch.isnan(b))
                a, b = torch.where(ok, a, 0), torch.where(ok, b, 0)
                n = ok.sum(1).clamp_min(1)
                am, bm = a.sum(1) / n, b.sum(1) / n
                a, b = torch.where(ok, a - am[:, None], 0), torch.where(ok, b - bm[:, None], 0)
                cs[d] = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)
            g = torch.minimum(cs[60], cs[120]) - torch.maximum(torch.maximum(cs[30], cs[90]), cs[150])
            best = torch.maximum(best, g)
    return best


def field_stats(maps, valid):
    """Number of fields and fraction of the arena they cover, per map (CPU, scipy)."""
    n_fields, cover = np.zeros(len(maps), int), np.zeros(len(maps))
    v = valid.cpu().numpy()
    for i, m in enumerate(maps.cpu().numpy()):
        lo, hi = m[v].min(), m[v].max()
        above = (m > lo + 0.5 * (hi - lo)) & v
        lab, n = ndimage.label(above)
        sizes = np.bincount(lab.ravel())[1:]
        big = sizes >= 3
        n_fields[i], cover[i] = big.sum(), sizes[big].sum() / v.sum()
    return n_fields, cover


# ---------------------------------------------------------------- decoding
def _ridge(X, Y, lam):
    n, d = X.shape
    if d <= n:
        return torch.linalg.solve(X.T @ X + lam * torch.eye(d, device=X.device), X.T @ Y)
    return X.T @ torch.linalg.solve(X @ X.T + lam * torch.eye(n, device=X.device), Y)


def decode(X, Y, tr, te, val_split):
    """Ridge with lambda picked on a walk-level validation split inside the training set."""
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
    Z = (X - mu) / sd
    ym = Y[tr].mean(0)
    a, b = val_split
    best, best_err = None, np.inf
    for lam in (1e1, 1e2, 1e3, 1e4, 1e5):
        W = _ridge(Z[a], Y[a] - ym, lam)
        err = (Z[b] @ W + ym - Y[b]).norm(dim=1).median().item()
        if err < best_err:
            best, best_err = lam, err
    W = _ridge(Z[tr], Y[tr] - ym, best)
    return (Z[te] @ W + ym - Y[te]).norm(dim=1).median().item()


def sectors(theta, n=4):
    """Facing sector ids: n equal slices of the compass, 0..3 = E, N, W, S for n = 4."""
    return np.floor(((theta + np.pi / n) % (2 * np.pi)) / (2 * np.pi / n)).astype(int)


def decoding_suite(X, meta, quadrant=False):
    xy = torch.tensor(np.stack([meta["x"], meta["y"]], 1), dtype=torch.float32, device=dev)
    walk, th = meta["walk"], meta["theta"]
    even, odd = walk % 2 == 0, walk % 2 == 1
    halves = {"N": np.sin(th) > 0, "S": np.sin(th) <= 0, "E": np.cos(th) > 0, "W": np.cos(th) <= 0}
    opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}
    t = lambda m: torch.from_numpy(np.flatnonzero(m)).to(dev)
    def run(train_mask, test_mask):
        va = train_mask & (walk % 4 == 0)
        vb = train_mask & (walk % 4 == 2)
        return decode(X, xy, t(train_mask), t(test_mask), (t(va), t(vb)))
    out = {"all": run(even, odd)}
    out["within_heading"] = float(np.mean([run(even & halves[h], odd & halves[h]) for h in halves]))
    out["cross_heading"] = float(np.mean([run(even & halves[h], odd & halves[opposite[h]]) for h in halves]))
    out["chance"] = (xy[t(odd)] - xy[t(even)].mean(0)).norm(dim=1).median().item()
    if quadrant:
        # train on one 90-degree sector, test within it or on the opposite sector (heading_matrix's
        # diagonal and off-diagonal-by-two cells); chance predicts the training sector's mean position
        sec = sectors(th)
        chance = lambda a, b: (xy[t(odd & b)] - xy[t(even & a)].mean(0)).norm(dim=1).median().item()
        out["cross_heading_chance"] = float(np.mean([chance(halves[h], halves[opposite[h]]) for h in halves]))
        out["within_quadrant"] = float(np.mean([run(even & (sec == i), odd & (sec == i)) for i in range(4)]))
        out["cross_quadrant"] = float(np.mean([run(even & (sec == i), odd & (sec == (i + 2) % 4)) for i in range(4)]))
        out["cross_quadrant_chance"] = float(np.mean([chance(sec == i, sec == (i + 2) % 4) for i in range(4)]))
    return out


def heading_matrix(X, meta, n=4):
    """Median position error (m) training on facing sector i (even walks) and testing on sector j
    (odd walks); sectors are n equal slices of the compass centred on E, N, W, S for n = 4."""
    xy = torch.tensor(np.stack([meta["x"], meta["y"]], 1), dtype=torch.float32, device=dev)
    walk = meta["walk"]
    sector = sectors(meta["theta"], n)
    t = lambda m: torch.from_numpy(np.flatnonzero(m)).to(dev)
    M = np.zeros((n, n))
    for i in range(n):
        tr = (walk % 2 == 0) & (sector == i)
        val = (t(tr & (walk % 4 == 0)), t(tr & (walk % 4 == 2)))
        for j in range(n):
            M[i, j] = decode(X, xy, t(tr), t((walk % 2 == 1) & (sector == j)), val)
    return M.tolist()


# ---------------------------------------------------------------- per layer
def causal_ema(X, meta, half_life):
    """Exponential moving average over each walk's past frames (memory-of-recent-views control)."""
    W, S = int(meta["walk"].max()) + 1, int(meta["step"].max()) + 1
    x = X.reshape(W, S, -1)
    a = 0.5 ** (1 / half_life)
    out = torch.empty_like(x)
    y = x[:, 0]
    for s in range(S):
        y = a * y + (1 - a) * x[:, s] if s else y
        out[:, s] = y
    return out.reshape(W * S, -1)


def load_units(tag, layer, data_dir, meta):
    """Tags: pixels, <condition>, or <base>+ema<h> (base activations smoothed over the walk)."""
    base, _, ema = tag.partition("+ema")
    if base == "pixels":
        frames = np.load(os.path.join(data_dir, "frames.npy"), mmap_mode="r")[:len(meta["walk"])]
        X = np.stack([f.reshape(32, 8, 32, 8, 3).mean((1, 3)).ravel() for f in frames]).astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    else:
        X = np.load(f"data/acts/{base}/{layer}.npy").astype(np.float32)
    X = torch.from_numpy(X).to(dev)
    return causal_ema(X, meta, float(ema)) if ema else X


def analyze_layer(X, meta, rng, quadrant=False):
    N = len(X)
    walk, step = meta["walk"], meta["step"]
    bins = spatial_bins(meta)
    hb =torch.from_numpy(np.clip((meta["theta"] / (2 * np.pi) * NH).astype(int), 0, NH - 1)).to(dev)
    R = torch.cat([X.clamp_min(0), (-X).clamp_min(0)], 1)
    active = (R > 0).float().mean(0) >= 0.01
    pole_ids = torch.nonzero(active).squeeze(1)
    R = R[:, active]
    U = R.shape[1]

    maps, occ = rate_maps(R, bins)
    valid = occ >= MIN_OCC
    si = spatial_info(maps, occ, valid)
    tc = tuning_curves(R, hb)
    hd = mvl(tc)

    # circular-shift nulls: pair each walk's activity with its own positions shifted in time
    n_steps = int(step.max()) + 1
    order = np.lexsort((step, walk))
    assert (order == np.arange(N)).all(), "frames must be stored walk-major, step-minor"
    si_null, hd_null, grid_null = [], [], []
    for k in range(N_SHUF):
        off = rng.integers(32, n_steps - 32, size=walk.max() + 1)
        perm = torch.from_numpy(walk * n_steps + (step + off[walk]) % n_steps).to(dev)
        m_s, _ = rate_maps(R, bins[perm])
        si_null.append(spatial_info(m_s, occ, valid))
        hd_null.append(mvl(tuning_curves(R, hb[perm])))
        if k < N_SHUF_GRID:
            grid_null.append(torch.cat([gridness(autocorr(m_s[i:i + 2048], valid)) for i in range(0, U, 2048)]))
    si_thr = torch.quantile(torch.stack(si_null), 0.99, dim=0)
    hd_thr = torch.quantile(torch.stack(hd_null), 0.99, dim=0)
    grid_null = torch.cat(grid_null)
    grid_thr = torch.quantile(grid_null[torch.isfinite(grid_null)], 0.99).item()

    even = torch.from_numpy(walk % 2 == 0).to(dev)
    m_even, _ = rate_maps(R, bins, even)
    m_odd, _ = rate_maps(R, bins, ~even)
    stab = map_corr(m_even, m_odd, valid)
    hd_stab = corr_rows(tuning_curves(R, hb, even), tuning_curves(R, hb, ~even))
    th = meta["theta"]
    hi = []
    for a in (np.sin(th) > 0, np.cos(th) > 0):
        a = torch.from_numpy(a).to(dev)
        ma, oa = rate_maps(R, bins, a)
        mb, ob = rate_maps(R, bins, ~a)
        hi.append(map_corr(ma, mb, (oa >= MIN_OCC) & (ob >= MIN_OCC)))
    heading_inv = (hi[0] + hi[1]) / 2
    grid = torch.cat([gridness(autocorr(maps[i:i + 2048], valid)) for i in range(0, U, 2048)])

    cand = torch.nonzero((si > si_thr) & (stab > 0.5)).squeeze(1)
    nf, cover = field_stats(maps[cand], valid)
    keep = torch.from_numpy((nf >= 1) & (nf <= 3) & (cover <= 0.4)).to(dev)
    place = torch.zeros(U, dtype=torch.bool, device=dev)
    place[cand[keep]] = True
    gridc = (grid > grid_thr) & (stab > 0.5)
    hdc = (hd > hd_thr) & (hd_stab > 0.5)

    def med(x, m):
        return float(x[m].median()) if m.any() else float("nan")
    stats = {
        "n_neurons": X.shape[1], "n_poles_active": U,
        "frac_place": place.float().mean().item(), "frac_grid": gridc.float().mean().item(),
        "frac_hd": hdc.float().mean().item(), "grid_threshold": grid_thr,
        "place_stability_median": med(stab, place), "place_heading_inv_median": med(heading_inv, place),
        "place_inv_over_stab_median": med(heading_inv / stab.clamp_min(1e-3), place),
        "all_stable_heading_inv_median": med(heading_inv, stab > 0.5),
        "frac_place_heading_invariant": (place & (heading_inv > 0.5)).float().mean().item(),
        "max_gridness": float(grid[torch.isfinite(grid)].max()),
    }
    if quadrant:
        # maps from opposite 90-degree sectors only (N vs S, E vs W); each has about half the frames
        # of a 180-degree half. The ratio still divides by stab (even vs odd walks, all headings).
        sec = sectors(th)
        hq = []
        for i in (1, 0):
            ma, oa = rate_maps(R, bins, torch.from_numpy(sec == i).to(dev))
            mb, ob = rate_maps(R, bins, torch.from_numpy(sec == i + 2).to(dev))
            hq.append(map_corr(ma, mb, (oa >= MIN_OCC) & (ob >= MIN_OCC)))
        inv_q = (hq[0] + hq[1]) / 2
        stats.update({
            "place_heading_inv_quadrant_median": med(inv_q, place),
            "place_inv_quadrant_over_stab_median": med(inv_q / stab.clamp_min(1e-3), place),
            "all_stable_heading_inv_quadrant_median": med(inv_q, stab > 0.5),
            "frac_place_heading_invariant_quadrant": (place & (inv_q > 0.5)).float().mean().item(),
        })
    per_unit = dict(pole=pole_ids, si=si, stab=stab, hi=heading_inv, grid=grid, hd=hd,
                    place=place, gridc=gridc, hdc=hdc, maps=maps, valid=valid)
    return stats, per_unit


# ---------------------------------------------------------------- figures
def plot_examples(tag, layer, pu, R_maps_by_half, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    place = torch.nonzero(pu["place"]).squeeze(1)
    order = place[torch.argsort(pu["si"][place], descending=True)]
    # top place units by information, plus the most heading-invariant ones
    inv = place[torch.argsort(pu["hi"][place], descending=True)]
    picks = list(dict.fromkeys(order[:4].tolist() + inv[:4].tolist()))[:8]
    gpick = torch.argsort(torch.nan_to_num(pu["grid"], -2), descending=True)[:4].tolist()
    rows = len(picks) + len(gpick)
    if rows == 0:
        return
    fig, axs = plt.subplots(rows, 3, figsize=(7.5, 2.4 * rows), squeeze=False)
    v = pu["valid"].cpu().numpy()
    for r, u in enumerate(picks + gpick):
        m = pu["maps"][u].cpu().numpy().copy()
        m[~v] = np.nan
        mn, ms = R_maps_by_half(u)
        kind = "place" if r < len(picks) else "grid?"
        ext = (X0, X0 + L, Y0, Y0 + L)
        axs[r, 0].imshow(m, origin="lower", extent=ext, cmap="jet")
        axs[r, 0].set_title(f"{kind} pole {int(pu['pole'][u])}: SI {pu['si'][u]:.2f}, stab {pu['stab'][u]:.2f}", fontsize=8)
        if kind == "place":
            axs[r, 1].imshow(mn, origin="lower", extent=ext, cmap="jet")
            axs[r, 1].set_title("facing north half", fontsize=8)
            axs[r, 2].imshow(ms, origin="lower", extent=ext, cmap="jet")
            axs[r, 2].set_title(f"facing south half (inv {pu['hi'][u]:.2f})", fontsize=8)
        else:
            ac = autocorr(pu["maps"][u:u + 1], pu["valid"])[0].cpu().numpy()
            axs[r, 1].imshow(ac, origin="lower", cmap="jet")
            axs[r, 1].set_title(f"autocorrelogram, gridness {pu['grid'][u]:.2f}", fontsize=8)
            axs[r, 2].axis("off")
        for a in axs[r]:
            a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"{tag} {layer}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--data", default="data/walks")
    ap.add_argument("--layers", nargs="*", help="only these layer names, e.g. L6 L14")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-walks", type=int, help="match activations extracted with --max-walks")
    ap.add_argument("--quadrant", action="store_true",
                    help="experiment 13e: add opposite-90-degree-sector measures, write summary_quadrant.json, no figures")
    ap.add_argument("--arena", action="store_true",
                    help="0.5 m bins over the scene's bounding box (meta.npz) instead of room A's 10 m box; recorded as 'arena'")
    args = ap.parse_args()
    meta = dict(np.load(os.path.join(args.data, "meta.npz")))
    if args.max_walks:
        n = int((meta["walk"] < args.max_walks).sum())
        meta = {k: v[:n] for k, v in meta.items()}
    grid = set_arena(meta) if args.arena else None
    os.makedirs(os.path.join(args.out, "figs"), exist_ok=True)
    summary_path = os.path.join(args.out, "summary_quadrant.json" if args.quadrant else "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    if args.quadrant:
        summary["_about"] = {
            "split": "*_quadrant keys use four 90-degree facing sectors centred on E, N, W, S (heading_matrix) "
                     "and compare only opposite sectors (N vs S, E vs W); the other keys are recomputed here "
                     "(on CPU) as in summary.json, with heading invariance and decoding on 180-degree halves",
            "stability": "place_inv_quadrant_over_stab_median divides by the same split-half stability as "
                         "summary.json: even vs odd walks, all headings",
            "decoding": "decode_cross_quadrant / decode_within_quadrant: ridge trained on one sector (even walks), "
                        "tested on the opposite / same sector (odd walks), mean over the 4 sectors",
            "chance": "decode_cross_heading_chance / decode_cross_quadrant_chance: median error of predicting the "
                      "training half's / sector's mean position on the opposite one; decode_chance as in summary.json",
        }
    for tag in args.tags:
        base = tag.partition("+ema")[0]
        if base == "pixels":
            layers = ["pix"]
        else:
            files = sorted(f[:-4] for f in os.listdir(f"data/acts/{base}") if f.endswith(".npy"))
            layers = sorted(files, key=lambda s: int(s[1:]))
        if args.layers:
            layers = [l for l in layers if l in args.layers or base == "pixels"]
        for layer in layers:
            rng = np.random.default_rng(0)
            X = load_units(tag, layer, args.data, meta)
            stats, pu = analyze_layer(X, meta, rng, args.quadrant)
            stats.update({f"decode_{k}": v for k, v in decoding_suite(X, meta, args.quadrant).items()})
            if grid:
                stats["arena"] = grid
            summary.setdefault(tag, {})[layer] = stats
            print(tag, layer, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()}), flush=True)
            if args.quadrant:
                json.dump(summary, open(summary_path, "w"), indent=1)
                del X, pu
                continue

            north = torch.from_numpy(np.sin(meta["theta"]) > 0).to(dev)
            bins = spatial_bins(meta)
            Rfull = torch.cat([X.clamp_min(0), (-X).clamp_min(0)], 1)
            def by_half(u, Rfull=Rfull):
                col = Rfull[:, int(pu["pole"][u])][:, None]
                mn, on = rate_maps(col, bins, north)
                ms, os_ = rate_maps(col, bins, ~north)
                mn, ms = mn[0].cpu().numpy(), ms[0].cpu().numpy()
                mn[(on < MIN_OCC).cpu().numpy()] = np.nan
                ms[(os_ < MIN_OCC).cpu().numpy()] = np.nan
                return mn, ms
            plot_examples(tag, layer, pu, by_half, os.path.join(args.out, "figs", f"{tag}_{layer}.png"))
            json.dump(summary, open(summary_path, "w"), indent=1)
            del X, pu, Rfull
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
