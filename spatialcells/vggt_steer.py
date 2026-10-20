"""Can writing into VGGT's place neurons move its predicted camera? Three steering tests.

    python -m spatialcells.vggt_steer [--n-test 96] [--parts a b c]
    python -m spatialcells.vggt_steer --plot          # results/vggt_steer.png from the saved results

Single mode (8 anchors + 1 walk frame, room A), as in vggt_patch.py. Place neurons come from
analyze.analyze_layer on data/acts/vggt-single, blocks 19 and 21 (block 21 is added with
vggt_cells --mode single --blocks 21). A neuron is "invariant" if one of its poles is a place
unit with heading invariance > 0.5, and "heading-specific" if it has place poles but none is
invariant (poles map to neurons by pole % 4096). Test frames come from odd walks; target
activity and directions are measured on even walks. Positions are read from the camera head
in room coordinates by a similarity fit on the anchors (as in vggt_patch).

(a) Population. Each test frame gets a target location L: the centre of a 1 m bin more than
    3 m from the frame. The chosen neurons' fc2 input at the walk frame's camera token is set
    to their mean over even-walk frames in that bin (all headings). Transfer is how far the
    prediction moves toward L, as a fraction of the distance (0 = no effect, 1 = placed at L).
    Sets: invariant place neurons (all, and the top k by spatial information), invariant and
    heading-specific at equal count (top by spatial information), random neurons (as many as
    the invariant set), and all 4096; in block 19, 21 or both. "x4" pushes 4 times as far from
    each neuron's overall mean. Reference: the whole residual token after the block set to its
    mean over the fit frames in L's bin.
(b) Single units. Each invariant place neuron with a positive place pole (up to --n-units, by
    spatial information) is set to 4x its largest activation on any walk frame, on 16 test
    frames. The displacement of the prediction is compared with the direction from the clean
    prediction to the unit's field centre (peak of its rate map): cosine, against the same with
    another unit's field centre. Random neurons with no place pole, driven the same way, give
    the size of a non-specific push.
(c) Joystick. The 10-d position-PCA subspace of the residual token after the block
    (vggt_patch.mean_subspace on 1 m bins) and, inside it, the directions that move the token
    1 m east and 1 m north (least squares of the projected token on x, y). alpha times one
    direction is added to the residual at the camera token: does the prediction move alpha
    metres that way, with heading unchanged? A random direction of the same norm is the control.
Writes results/vggt_steer.json and results/vggt_steer.npz.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import analyze
from .vggt_cells import anchors
from .vggt_control import RES, umeyama
from .vggt_patch import mean_subspace

NN = 4096


def unit_info(b, meta, even, bin1):
    """Place-neuron sets and the per-neuron statistics the steering needs, for global block b."""
    X = analyze.load_units("vggt-single", f"L{b}", "data/walks", meta)
    _, pu = analyze.analyze_layer(X, meta, np.random.default_rng(0))
    pole, place, hi, si = (pu[k].cpu().numpy() for k in ("pole", "place", "hi", "si"))
    neuron = pole % NN

    def by_si(mask):
        best = {}
        for n, s in zip(neuron[mask], si[mask]):
            best[n] = max(best.get(n, -np.inf), s)
        return np.array(sorted(best, key=lambda n: -best[n]), dtype=int)
    inv = by_si(place & (hi > 0.5))
    spec = by_si(place & (hi <= 0.5))
    spec = spec[~np.isin(spec, inv)]
    # single units: positive invariant place poles, with the peak of their rate map as field centre
    sp = np.flatnonzero(place & (hi > 0.5) & (pole < NN))
    sp = sp[np.argsort(-si[sp])]
    maps = pu["maps"][torch.from_numpy(sp).to(analyze.dev)].clone()
    maps[:, ~pu["valid"]] = -torch.inf
    k = maps.flatten(1).argmax(1).cpu().numpy()
    centre = np.stack([(k % analyze.NB + 0.5), (k // analyze.NB + 0.5)], 1) * analyze.L / analyze.NB
    single = dict(neuron=pole[sp], centre=centre, si=si[sp])
    no_place = np.setdiff1d(np.arange(NN), neuron[place])
    ev = torch.from_numpy(np.flatnonzero(even)).to(analyze.dev)
    lab = torch.from_numpy(bin1).to(analyze.dev)[ev]
    sums = torch.zeros(100, NN, device=analyze.dev).index_add_(0, lab, X[ev])
    cnt = torch.bincount(lab, minlength=100).float()
    out = dict(inv=inv, spec=spec, single=single, no_place=no_place, n_place_poles=int(place.sum()),
               target=(sums / cnt.clamp_min(1)[:, None]).cpu(), mean=X[ev].mean(0).cpu(), peak=X.max(0).values.cpu())
    del X, pu
    torch.cuda.empty_cache()
    return out


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = json.load(open(f"{out}.json"))
    raw = np.load(f"{out}.npz")
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    conds = R["a"]["conditions"]
    names = list(conds)
    y = np.arange(len(names))
    axs[0].barh(y, [conds[n]["transfer_mean"] for n in names], xerr=[conds[n]["transfer_sem"] for n in names], color="0.5")
    axs[0].set_yticks(y, names, fontsize=6)
    axs[0].invert_yaxis()
    axs[0].axvline(0, color="k", lw=0.5)
    axs[0].set_xlabel("transfer toward target location L (1 = placed at L)")
    axs[0].set_title("(a) set neurons to their mean at L")
    for b in R["blocks"]:
        if f"b{b}_invariant_disp" in raw:
            d = np.linalg.norm(raw[f"b{b}_invariant_disp"], axis=2).mean(1)
            axs[1].hist(d * 100, bins=30, alpha=0.5, label=f"block {b}: invariant place unit")
            axs[1].hist(np.linalg.norm(raw[f"b{b}_no_disp"], axis=2).mean(1) * 100, bins=30, alpha=0.5,
                        histtype="step", lw=2, label=f"block {b}: no-place unit")
    axs[1].set_xlabel("mean shift of the predicted position (cm)")
    axs[1].set_title("(b) one unit at 4x its peak")
    axs[1].legend(fontsize=7)
    for b in R["blocks"]:
        rows = R["c"][f"block {b}"]["rows"]
        for name, key, ls in (("east", "dx_mean", "-"), ("north", "dy_mean", "--"), ("random", "dx_mean", ":")):
            r = sorted((x for x in rows if x["dir"] == name), key=lambda x: x["alpha"])
            axs[2].plot([x["alpha"] for x in r], [x[key] for x in r], ls, marker="o", label=f"block {b} {name}")
    axs[2].plot([-3, 3], [-3, 3], color="0.7", lw=0.8)
    axs[2].set_xlabel("alpha (metres of the east / north direction added)")
    axs[2].set_ylabel("predicted position moved along it (m)")
    axs[2].set_title("(c) joystick in the position subspace")
    axs[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("results/vggt_steer.png", dpi=90)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/walks")
    ap.add_argument("--vggt", default="/workspace/vggt")
    ap.add_argument("--blocks", type=int, nargs="+", default=[19, 21])
    ap.add_argument("--n-fit", type=int, default=2048)
    ap.add_argument("--n-test", type=int, default=96)
    ap.add_argument("--n-units", type=int, default=100)
    ap.add_argument("--n-null", type=int, default=50)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--parts", nargs="+", default=["a", "b", "c"])
    ap.add_argument("--out", default="results/vggt_steer")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        return plot(args.out)
    sys.path.insert(0, args.vggt)
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    rng = np.random.default_rng(0)
    t0 = time.time()
    frames = np.load(f"{args.data}/frames.npy", mmap_mode="r")
    meta = dict(np.load(f"{args.data}/meta.npz"))
    xy_true = np.stack([meta["x"], meta["y"]], 1)
    even = meta["walk"] % 2 == 0
    bin1 = np.clip(meta["x"].astype(int), 0, 9) * 10 + np.clip(meta["y"].astype(int), 0, 9)   # 1 m bins, x-major
    centres = np.stack(np.meshgrid(np.arange(10) + 0.5, np.arange(10) + 0.5, indexing="ij"), -1).reshape(100, 2)

    units = {b: unit_info(b, meta, even, bin1) for b in args.blocks}
    for b, u in units.items():
        print(f"block {b}: {len(u['inv'])} invariant and {len(u['spec'])} heading-specific place neurons, "
              f"{len(u['single']['neuron'])} positive invariant place poles ({time.time() - t0:.0f}s)", flush=True)

    model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
    agg = model.aggregator
    P = agg.patch_start_idx + (RES // 14) ** 2
    A = anchors(args.data)
    n_anchor = len(A["images"])
    cam = n_anchor * P                      # walk frame's camera token in the global blocks' (B, S*P, C) layout
    edit, record = {}, {}                   # ("fc2"|"set"|"add", block) -> per-batch tensors; block -> residual

    def fc2_hook(b):
        def f(_, inp):
            if ("fc2", b) not in edit:
                return None
            x = inp[0].clone()
            m, v = edit[("fc2", b)]
            x[:, cam] = torch.where(m, v.to(x.dtype), x[:, cam])
            return (x,)
        return f

    def out_hook(b):
        def f(_, __, out):
            if not edit:
                record[b] = out[:, cam].float().cpu()
            if ("set", b) in edit or ("add", b) in edit:
                out = out.clone()
                if ("set", b) in edit:
                    out[:, cam] = edit[("set", b)].to(out.dtype)
                if ("add", b) in edit:
                    out[:, cam] = out[:, cam] + edit[("add", b)].to(out.dtype)
                return out
        return f
    for b in args.blocks:
        agg.global_blocks[b].mlp.fc2.register_forward_pre_hook(fc2_hook(b))
        agg.global_blocks[b].register_forward_hook(out_hook(b))

    def prep(ims):
        x = torch.from_numpy(np.stack([np.asarray(a) for a in ims])).cuda().permute(0, 3, 1, 2).float() / 255
        return F.interpolate(x, size=(RES, RES), mode="bicubic", align_corners=False).clamp(0, 1)
    anchor_px = prep(A["images"])
    gt_anchor = np.c_[A["poses"][:, :2], np.ones(n_anchor)]

    def predict(idx, edits=None):
        """idx: frame per sequence; edits: key -> tensor(s) with leading dim len(idx). Returns xy (n, 2),
        heading (n,), and the clean residuals at the hooked blocks when there are no edits."""
        xy, th, rec = np.zeros((len(idx), 2)), np.zeros(len(idx)), {}
        for b0 in range(0, len(idx), args.batch):
            sl = slice(b0, b0 + args.batch)
            edit.clear()
            for k, v in (edits or {}).items():
                edit[k] = tuple(x[sl].cuda() for x in v) if isinstance(v, tuple) else v[sl].cuda()
            seqs = torch.stack([torch.cat([anchor_px, prep(frames[[i]])]) for i in idx[sl]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens, _ = agg(seqs)
                pose = model.camera_head(tokens)[-1]
            ext, _ = pose_encoding_to_extri_intri(pose.float(), (RES, RES))
            for j, E in enumerate(ext.cpu().numpy().astype(np.float64)):
                R, t = E[:, :, :3], E[:, :, 3]
                C = -np.einsum("sji,sj->si", R, t)
                fwd = np.einsum("sji,j->si", R, np.array([0.0, 0.0, 1.0]))
                s, Ra, ta = umeyama(C[:n_anchor], gt_anchor)
                c, fr = s * (Ra @ C[-1]) + ta, Ra @ fwd[-1]
                xy[b0 + j], th[b0 + j] = c[:2], np.arctan2(fr[1], fr[0])
            if not edits:
                for k, v in record.items():
                    rec.setdefault(k, []).append(v)
        edit.clear()
        return xy, th, {k: torch.cat(v) for k, v in rec.items()}

    def dheading(a, b):
        return np.degrees(np.abs((a - b + np.pi) % (2 * np.pi) - np.pi))

    results = {"blocks": args.blocks, "units": {b: {"invariant": len(u["inv"]), "heading_specific": len(u["spec"]),
                                                    "positive_invariant_poles": len(u["single"]["neuron"])}
                                                for b, u in units.items()}}
    raw = {}
    fit = np.sort(rng.choice(np.flatnonzero(even), args.n_fit, replace=False))
    test = np.sort(rng.choice(np.flatnonzero(~even), args.n_test, replace=False))
    _, _, H_fit = predict(fit)
    clean_xy, clean_th, _ = predict(test)
    results["clean_error_m"] = float(np.median(np.hypot(*(clean_xy - xy_true[test]).T)))
    print(f"clean passes done ({time.time() - t0:.0f}s): error {results['clean_error_m']:.2f} m", flush=True)

    # ------------------------------------------------------------ (a) population steering toward L
    if "a" in args.parts:
        cnt_even = np.bincount(bin1[even], minlength=100)
        cnt_fit = np.bincount(bin1[fit], minlength=100)
        ok = np.flatnonzero((cnt_even >= 20) & (cnt_fit >= max(1, round(8 * args.n_fit / 2048))))
        Lbin = np.array([rng.choice(ok[np.hypot(*(centres[ok] - xy_true[t]).T) > 3]) for t in test])
        Lxy = centres[Lbin]
        gap = Lxy - clean_xy
        dist2 = (gap ** 2).sum(1)

        def summarize(xy, th):
            moved = xy - clean_xy
            tr = (moved * gap).sum(1) / dist2
            return {"transfer_mean": float(tr.mean()), "transfer_sem": float(tr.std() / np.sqrt(len(tr))),
                    "moved_m_median": float(np.median(np.linalg.norm(moved, axis=1))),
                    "heading_change_deg_median": float(np.median(dheading(th, clean_th)))}

        res_a = {"target_distance_m_median": float(np.median(np.sqrt(dist2))), "conditions": {}}
        for b in args.blocks:
            Hf = H_fit[b]
            lab = torch.from_numpy(bin1[fit])
            Hbin = torch.zeros(100, Hf.shape[1]).index_add_(0, lab, Hf) / torch.bincount(lab, minlength=100).clamp_min(1)[:, None].float()
            xy, th, _ = predict(test, {("set", b): Hbin[Lbin]})
            res_a["conditions"][f"block {b}: residual token -> mean at L"] = summarize(xy, th)
        sets = {}                           # block -> key -> (label, neurons, gain)
        for b, u in units.items():
            inv, spec = u["inv"], u["spec"]
            K = min(len(inv), len(spec))
            s = {"inv": (f"invariant, all {len(inv)}", inv, 1),
                 "invK": (f"invariant, top {K} (= heading-specific count)", inv[:K], 1),
                 "spec": (f"heading-specific, top {K}", spec[:K], 1),
                 "rand": (f"random {len(inv)}", rng.choice(NN, len(inv), replace=False), 1),
                 "all": ("all 4096", np.arange(NN), 1),
                 "inv x4": (f"invariant, all {len(inv)}, x4", inv, 4),
                 "all x4": ("all 4096, x4", np.arange(NN), 4)}
            for k in (8, 32, 128):
                if k < len(inv):
                    s[f"top{k}"] = (f"invariant, top {k}", inv[:k], 1)
            sets[b] = s
        for bs in [[b] for b in args.blocks] + ([args.blocks] if len(args.blocks) > 1 else []):
            for key in [k for k in sets[bs[0]] if all(k in sets[b] for b in bs)]:
                edits = {}
                for b in bs:
                    _, nset, gain = sets[b][key]
                    m = torch.zeros(len(test), NN, dtype=torch.bool)
                    m[:, torch.from_numpy(nset)] = True
                    edits[("fc2", b)] = (m, units[b]["mean"] + gain * (units[b]["target"][Lbin] - units[b]["mean"]))
                xy, th, _ = predict(test, edits)
                label = "blocks " + "+".join(map(str, bs)) + ": " + " / ".join(sets[b][key][0] for b in bs)
                res_a["conditions"][label] = r = summarize(xy, th)
                print(f"(a) {label:70s} transfer {r['transfer_mean']:+.3f} ± {r['transfer_sem']:.3f}, "
                      f"moved {r['moved_m_median']:.3f} m ({time.time() - t0:.0f}s)", flush=True)
        results["a"] = res_a
        json.dump(results, open(f"{args.out}.json", "w"), indent=1)

    # ------------------------------------------------------------ (b) one neuron at a time
    if "b" in args.parts:
        tb = np.arange(min(16, len(test)))
        res_b = {}
        for b, u in units.items():
            sg = u["single"]
            n_u = min(args.n_units, len(sg["neuron"]))
            null = rng.choice(u["no_place"], args.n_null, replace=False)
            for kind, neurons in (("invariant place", sg["neuron"][:n_u]), ("no place pole (null)", null)):
                n = len(neurons) * len(tb)
                m = torch.zeros(n, NN, dtype=torch.bool)
                v = torch.zeros(n, NN)
                rows = torch.arange(n)
                nr = torch.from_numpy(np.repeat(neurons, len(tb)))
                m[rows, nr] = True
                v[rows, nr] = 4 * u["peak"][nr]
                fi = np.tile(tb, len(neurons))
                xy, th, _ = predict(test[fi], {("fc2", b): (m, v)})
                disp = (xy - clean_xy[fi]).reshape(len(neurons), len(tb), 2)
                r = {"n_units": len(neurons), "moved_m_median": float(np.median(np.linalg.norm(disp, axis=2))),
                     "moved_m_90th": float(np.percentile(np.linalg.norm(disp, axis=2), 90)),
                     "heading_change_deg_median": float(np.median(dheading(th, clean_th[fi])))}
                raw[f"b{b}_{kind.split()[0]}_disp"] = disp
                if kind == "invariant place":
                    def cosines(c):
                        to = c[:, None] - clean_xy[tb][None]
                        return ((disp * to).sum(2) / (np.linalg.norm(disp, axis=2) * np.linalg.norm(to, axis=2)).clip(1e-9)).mean(1)
                    cos = cosines(sg["centre"][:n_u])
                    perm = rng.permutation(n_u)
                    other = np.empty_like(perm)
                    other[perm] = np.roll(perm, 1)          # each unit gets a different unit's centre
                    cos_null = cosines(sg["centre"][:n_u][other])
                    k = int((cos > 0).sum())
                    r.update(cos_mean=float(cos.mean()), cos_sem=float(cos.std() / np.sqrt(n_u)), frac_units_cos_pos=k / n_u,
                             sign_p=float(sum(math.comb(n_u, i) for i in range(k, n_u + 1)) / 2 ** n_u),
                             cos_shuffled_centre_mean=float(cos_null.mean()))
                    raw[f"b{b}_centre"] = sg["centre"][:n_u]
                res_b[f"block {b}: {kind}"] = r
                print(f"(b) block {b} {kind}: " + json.dumps({k: round(x, 4) if isinstance(x, float) else x for k, x in r.items()})
                      + f" ({time.time() - t0:.0f}s)", flush=True)
        results["b"] = res_b
        json.dump(results, open(f"{args.out}.json", "w"), indent=1)

    # ------------------------------------------------------------ (c) joystick in the position subspace
    if "c" in args.parts:
        tc = np.arange(min(64, len(test)))
        res_c = {}
        for b in args.blocks:
            Hf = H_fit[b].cuda()
            Q = mean_subspace(Hf, bin1[fit], 10)
            Z = (Hf - Hf.mean(0)) @ Q
            D = torch.tensor(np.c_[xy_true[fit], np.ones(len(fit))], dtype=torch.float32, device="cuda")
            B = torch.linalg.lstsq(D, Z).solution                       # (3, 10): per metre east, north; offset
            r2 = 1 - ((D @ B - Z) ** 2).sum() / ((Z - Z.mean(0)) ** 2).sum()
            dirs = {"east": (Q @ B[0]).cpu(), "north": (Q @ B[1]).cpu()}
            g = torch.Generator().manual_seed(b)
            rnd = torch.randn(Hf.shape[1], generator=g)
            dirs["random"] = rnd / rnd.norm() * dirs["east"].norm()
            out = {"subspace_r2_from_xy": float(r2), "east_norm": float(dirs["east"].norm()),
                   "north_norm": float(dirs["north"].norm()), "token_norm_median": float(Hf.norm(dim=1).median()),
                   "east_north_cos": float(F.cosine_similarity(dirs["east"], dirs["north"], 0)), "rows": []}
            for name, d in dirs.items():
                alphas = (-3, -2, -1, -0.5, 0.5, 1, 2, 3) if name != "random" else (-3, -1, 1, 3)
                for a in alphas:
                    xy, th, _ = predict(test[tc], {("add", b): (a * d)[None].expand(len(tc), -1).contiguous()})
                    mv = xy - clean_xy[tc]
                    out["rows"].append({"dir": name, "alpha": a, "dx_mean": float(mv[:, 0].mean()), "dy_mean": float(mv[:, 1].mean()),
                                        "dx_sd": float(mv[:, 0].std()), "dy_sd": float(mv[:, 1].std()),
                                        "heading_change_deg_median": float(np.median(dheading(th, clean_th[tc])))})
                    raw[f"c{b}_{name}_{a}"] = mv
                print(f"(c) block {b} {name}: " + ", ".join(f"a={r['alpha']}: ({r['dx_mean']:+.2f}, {r['dy_mean']:+.2f}) m, "
                      f"dθ {r['heading_change_deg_median']:.1f}°" for r in out["rows"] if r["dir"] == name)
                      + f" ({time.time() - t0:.0f}s)", flush=True)
            for name, axis in (("east", 0), ("north", 1)):
                rows = [r for r in out["rows"] if r["dir"] == name]
                a = np.array([r["alpha"] for r in rows])
                along = np.array([r["dx_mean" if axis == 0 else "dy_mean"] for r in rows])
                across = np.array([r["dy_mean" if axis == 0 else "dx_mean"] for r in rows])
                slope = float((a * along).sum() / (a * a).sum())
                out[f"{name}_gain_m_per_m"] = slope
                out[f"{name}_linear_r2"] = float(1 - ((along - slope * a) ** 2).sum() / ((along - along.mean()) ** 2).sum())
                out[f"{name}_cross_gain"] = float((a * across).sum() / (a * a).sum())
            res_c[f"block {b}"] = out
        results["c"] = res_c
    json.dump(results, open(f"{args.out}.json", "w"), indent=1)
    np.savez(f"{args.out}.npz", **raw)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
