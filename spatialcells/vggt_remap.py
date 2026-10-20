"""Does VGGT's place code remap in a second room?

    python -m spatialcells.vggt_remap

Hippocampal place cells remap between different rooms: a cell's field moves somewhere
unrelated, or the cell falls silent. VGGT localises against anchor views taken from the
same 8 poses in both rooms, so a code for "where the camera is in the anchors' frame"
would keep its fields. This compares VGGT single-mode units in room A
(data/acts/vggt-single) and room B (data/acts/vggt-single-roomB; render_data --room B,
then vggt_cells --mode single --room B):
  - each room-A place pole's map in A against its map in B, next to its split-half
    stability in A and to random pairings of place poles across the rooms
  - how many room-A place poles are place poles in B, against chance
  - position decoding trained in one room and tested in the other, raw and after
    removing each room's mean activity
  - population-vector correlation at the same spot in both rooms, against spots more
    than 3 m apart
Writes results/vggt_remap.json.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .analyze import L, MIN_OCC, NB, analyze_layer, corr_rows, decode, dev, load_units, map_corr, rate_maps

LAYERS = ["L9", "L15", "L19", "L23"]
ROOMS = {"A": ("vggt-single", "data/walks"), "B": ("vggt-single-roomB", "data/walks-roomB")}


def bins_of(meta):
    ix = np.clip((meta["x"] / L * NB).astype(int), 0, NB - 1)
    iy = np.clip((meta["y"] / L * NB).astype(int), 0, NB - 1)
    return torch.from_numpy(iy * NB + ix).to(dev)


def xy_of(meta):
    return torch.tensor(np.stack([meta["x"], meta["y"]], 1), dtype=torch.float32, device=dev)


def cross_decoding(X, meta):
    """Median error (m) training on even walks of one room and testing on odd walks of either."""
    n = len(X["A"])
    Z = torch.cat([X["A"], X["B"]])
    Y = torch.cat([xy_of(meta["A"]), xy_of(meta["B"])])
    walk = np.r_[meta["A"]["walk"], meta["B"]["walk"]]
    room = np.r_[np.zeros(n, bool), np.ones(len(X["B"]), bool)]
    t = lambda m: torch.from_numpy(np.flatnonzero(m)).to(dev)
    out = {}
    for tr_room, tr_mask in (("A", ~room), ("B", room)):
        tr = tr_mask & (walk % 2 == 0)
        val = (t(tr & (walk % 4 == 0)), t(tr & (walk % 4 == 2)))
        for te_room, te_mask in (("A", ~room), ("B", room)):
            out[f"{tr_room}->{te_room}"] = decode(Z, Y, t(tr), t(te_mask & (walk % 2 == 1)), val)
    return out


def pv_corr(X, meta):
    """Population vectors per 0.5 m bin (each room centred on its own mean), correlated across rooms."""
    pv, ok = {}, {}
    for r in ("A", "B"):
        Z = X[r] - X[r].mean(0)
        b = bins_of(meta[r])
        oh = torch.nn.functional.one_hot(b, NB * NB).float()
        cnt = oh.sum(0)
        pv[r] = (oh.T @ Z) / cnt.clamp_min(1)[:, None]
        ok[r] = cnt >= MIN_OCC
        w = torch.from_numpy(meta[r]["walk"] % 2 == 0).to(dev)
        if r == "A":   # within-room reference: even vs odd walks
            pe = (oh[w].T @ Z[w]) / oh[w].sum(0).clamp_min(1)[:, None]
            po = (oh[~w].T @ Z[~w]) / oh[~w].sum(0).clamp_min(1)[:, None]
            ok_eo = (oh[w].sum(0) >= MIN_OCC) & (oh[~w].sum(0) >= MIN_OCC)
    both = ok["A"] & ok["B"]
    same = corr_rows(pv["A"][both], pv["B"][both])
    k = torch.nonzero(both).squeeze(1)
    cx, cy = (k % NB).float() * L / NB, (k // NB).float() * L / NB
    far = torch.hypot(cx[:, None] - cx[None], cy[:, None] - cy[None]) > 3
    A = pv["A"][both] / pv["A"][both].norm(dim=1, keepdim=True).clamp_min(1e-9)
    B = pv["B"][both] / pv["B"][both].norm(dim=1, keepdim=True).clamp_min(1e-9)
    C = A @ B.T
    return {"same_spot_across_rooms": same.median().item(),
            "spots_over_3m_apart_across_rooms": C[far].median().item(),
            "same_spot_within_A_even_vs_odd": corr_rows(pe[ok_eo], po[ok_eo]).median().item()}


def main():
    rng = np.random.default_rng(0)
    meta = {r: dict(np.load(f"{d}/meta.npz")) for r, (_, d) in ROOMS.items()}
    out = {}
    for r, (tag, _) in ROOMS.items():
        p = np.load(f"data/acts/{tag}/pose.npz")
        out[f"camera_error_m_room{r}"] = float(np.nanmedian(np.hypot(p["x"] - meta[r]["x"], p["y"] - meta[r]["y"])))
    print(json.dumps(out), flush=True)
    for layer in LAYERS:
        X = {r: load_units(tag, layer, d, meta[r]) for r, (tag, d) in ROOMS.items()}
        pu = {r: analyze_layer(X[r], meta[r], rng)[1] for r in ROOMS}
        maps, occ = {}, {}
        for r in ROOMS:
            R = torch.cat([X[r].clamp_min(0), (-X[r]).clamp_min(0)], 1)
            maps[r], occ[r] = rate_maps(R, bins_of(meta[r]))
            mean_rate = R.mean(0)
            if r == "A":
                rate_A = mean_rate
            else:
                rate_B = mean_rate
        valid = (occ["A"] >= MIN_OCC) & (occ["B"] >= MIN_OCC)
        res = {}
        place_B = set(pu["B"]["pole"][pu["B"]["place"]].tolist())
        for name, mask in (("place", pu["A"]["place"]), ("invariant_place", pu["A"]["place"] & (pu["A"]["hi"] > 0.5))):
            poles = pu["A"]["pole"][mask]
            if len(poles) < 2:
                continue
            cross = map_corr(maps["A"][poles], maps["B"][poles], valid)
            shuf = torch.cat([map_corr(maps["A"][poles], maps["B"][poles[torch.from_numpy(rng.permutation(len(poles))).to(dev)]], valid)
                              for _ in range(20)])
            ratio = rate_B[poles] / rate_A[poles].clamp_min(1e-9)
            res[name] = {
                "n_poles": len(poles),
                "stability_in_A_median": pu["A"]["stab"][mask].median().item(),
                "map_corr_A_vs_B_median": cross.median().item(),
                "map_corr_A_vs_B_frac_over_0.5": (cross > 0.5).float().mean().item(),
                "map_corr_random_pairs_median": shuf.median().item(),
                "also_place_in_B": float(np.mean([p in place_B for p in poles.tolist()])),
                "chance_place_in_B": pu["B"]["place"].float().mean().item(),
                "mean_rate_B_over_A_median": ratio.median().item(),
                "frac_nearly_silent_in_B": (ratio < 0.1).float().mean().item(),
            }
        res["decoding_raw"] = cross_decoding(X, meta)
        res["decoding_room_mean_removed"] = cross_decoding({r: X[r] - X[r].mean(0) for r in X}, meta)
        res["population_vectors"] = pv_corr(X, meta)
        out[layer] = res
        print(layer, json.dumps(res, indent=1), flush=True)
        json.dump(out, open("results/vggt_remap.json", "w"), indent=1)


if __name__ == "__main__":
    main()
