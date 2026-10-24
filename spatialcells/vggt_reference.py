"""Experiment 14e: do VGGT's block-19 place units move with the reference views?

    python -m spatialcells.vggt_reference

VGGT returns poses in the frame of the first image it is given. Room A's single-mode units at block 19
were recorded with three anchor sets (vggt_cells --mode single --blocks 19 --anchor-set SET, all on the
same machine):
  (i)   standard   the 8 anchors of experiment 4 (centre facing east first)
  (ii)  reordered  the same anchors in another order, a corner view first ((2, 2) facing the centre)
  (iii) rot45      every anchor pose rotated 45 deg anticlockwise about the room centre, rendered there
                   (amended 2026-10-23: a 90 deg rotation maps the 8 poses onto each other)
For (i)'s place poles (and those with heading invariance > 0.5; criteria as in vggt_remap), each pole's
map under (ii) and (iii) is correlated with its map under (i), and with (i)'s map rotated 45 deg about the
room centre: anticlockwise ("+45", the way the anchors turned) and clockwise ("-45", a control). Rotated
maps are bilinearly interpolated on the 0.5 m grid, and only bins valid in both maps (every bin used by
the interpolation occupied) are compared. For reference: (i)'s split-half stability, random pairings of
poles, and (i)'s maps against their own rotations. Units that code position in the first anchor's frame
or in the anchors' frame move ((ii), (iii) less like (i); (iii) like (i) rotated +45); units that code
the room's frame stay. Also: the share of (i)'s place poles that are place poles under each set,
mean-rate ratios, position decoding trained under one set and tested under another, and VGGT's camera
error under each set. Writes results/vggt_remap_reference.json.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from .analyze import MIN_OCC, NB, analyze_layer, load_units, map_corr, rate_maps
from .vggt_remap import bins_of, cross_decoding

DATA = "data/walks"
LAYER = "L19"
SETS = {"standard": "(i)", "reordered": "(ii)", "rot45": "(iii)"}


def rotate(maps, valid, deg):
    """Maps (U, NB, NB) (row = y, col = x, as in analyze.rate_maps) rotated deg anticlockwise about the room
    centre by bilinear interpolation (the rotated map at p is the map at R(-deg) p), and the bins whose
    interpolation used only valid bins of the original."""
    a = np.deg2rad(deg)
    theta = torch.tensor([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0]], dtype=torch.float32, device=maps.device)
    grid = F.affine_grid(theta[None], (1, 1, NB, NB), align_corners=False)
    out = F.grid_sample(maps[:, None], grid.expand(len(maps), -1, -1, -1), align_corners=False)[:, 0]
    ok = F.grid_sample(valid.float()[None, None], grid, align_corners=False)[0, 0] > 1 - 1e-4
    return out, ok


def main():
    rng = np.random.default_rng(0)
    meta = dict(np.load(f"{DATA}/meta.npz"))
    bins = bins_of(meta)
    A, Ar = np.load(f"{DATA}/anchors.npz"), np.load(f"{DATA}/anchors-rot45.npz")
    diff = np.abs(Ar["images"][:, None].astype(int) - A["images"][None].astype(int)).mean((2, 3, 4)).min(1)
    out = {"layer": LAYER, "anchor_sets": SETS, "rot45_poses": Ar["poses"].tolist(),
           "rot45_views_mean_abs_pixel_diff_to_closest_standard_view": diff.tolist()}
    X, pu, maps, rate = {}, {}, {}, {}
    for s in SETS:
        tag = f"vggt-single-anchors-{s}"
        p = np.load(f"data/acts/{tag}/pose.npz")
        out.setdefault("camera_error_m", {})[s] = float(np.nanmedian(np.hypot(p["x"] - meta["x"], p["y"] - meta["y"])))
        X[s] = load_units(tag, LAYER, DATA, meta)
        stats, pu[s] = analyze_layer(X[s], meta, rng)
        out.setdefault("frac_place", {})[s] = stats["frac_place"]
        R = torch.cat([X[s].clamp_min(0), (-X[s]).clamp_min(0)], 1)
        maps[s], occ = rate_maps(R, bins)
        rate[s] = R.mean(0)
        del R
    print(json.dumps(out), flush=True)
    valid = occ >= MIN_OCC                      # the same frames in every set, so the same occupancy

    def corr(a, b, deg=0):
        if not deg:
            return map_corr(a, b, valid)
        ar, ok = rotate(a, valid, deg)
        return map_corr(ar, b, ok & valid)

    p0 = pu["standard"]
    res = {"rotation_bins_compared": {f"{d:+d}": int((rotate(maps["standard"][:1], valid, d)[1] & valid).sum())
                                      for d in (45, -45)} | {"0": int(valid.sum())}}
    for name, mask in (("place", p0["place"]), ("invariant_place", p0["place"] & (p0["hi"] > 0.5))):
        poles = p0["pole"][mask]
        if len(poles) < 2:
            continue
        m0 = maps["standard"][poles]
        r = {"n_poles": len(poles),
             "stability_in_i_median": p0["stab"][mask].median().item(),
             "i_vs_own_rotation_+45_median": corr(m0, m0, 45).median().item(),
             "i_vs_own_rotation_-45_median": corr(m0, m0, -45).median().item()}
        for s in ("reordered", "rot45"):
            ms = maps[s][poles]
            same, plus, minus = corr(m0, ms), corr(m0, ms, 45), corr(m0, ms, -45)
            shuf = torch.cat([corr(m0, maps[s][poles[torch.from_numpy(rng.permutation(len(poles))).to(poles.device)]])
                              for _ in range(20)])
            ratio = rate[s][poles] / rate["standard"][poles].clamp_min(1e-9)
            place_s = set(pu[s]["pole"][pu[s]["place"]].tolist())
            r[s] = {"map_corr_vs_i_median": same.median().item(),
                    "map_corr_vs_i_frac_over_0.5": (same > 0.5).float().mean().item(),
                    "map_corr_vs_i_rotated_+45_median": plus.median().item(),
                    "map_corr_vs_i_rotated_+45_frac_over_0.5": (plus > 0.5).float().mean().item(),
                    "map_corr_vs_i_rotated_-45_median": minus.median().item(),
                    "frac_closer_to_i_rotated_+45_than_to_i": (plus > same).float().mean().item(),
                    "map_corr_random_pairs_median": shuf.median().item(),
                    "also_place": float(np.mean([q in place_s for q in poles.tolist()])),
                    "chance_place": pu[s]["place"].float().mean().item(),
                    "mean_rate_over_i_median": ratio.median().item(),
                    "frac_nearly_silent": (ratio < 0.1).float().mean().item()}
        res[name] = r
    res["decoding"] = {}                        # trained on even walks under one set, tested on odd walks under another
    for s in ("reordered", "rot45"):
        dec = cross_decoding({"A": X["standard"], "B": X[s]}, {"A": meta, "B": meta})
        res["decoding"].update({k.replace("A", "standard").replace("B", s): v for k, v in dec.items()})
    out[LAYER] = res
    print(json.dumps(res, indent=1), flush=True)
    json.dump(out, open("results/vggt_remap_reference.json", "w"), indent=1)


if __name__ == "__main__":
    main()
