"""Experiment 32, unregistered: where the feature-fit directions sit among the PCA families (no model run, no thresholds).

    PYTHONPATH=. .venv-mac/bin/python scripts/vggt_pair_patch_mix.py     # -> results/vggt_pair_patch_mix.json

For each feature family, the share of its fitted directions' energy (orthonormalized) inside each family of Qord30. The
label-free arm leaves out PCA directions, the feature-fit arm the fitted ones, so this says how much of a fitted family a
label-free condition leaves in place.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
import vggt_pair_patch as v

OUT = "results/vggt_pair_patch_mix.json"


def main():
    R27, Z27 = np.load(f"{v.E27}/raw.npz"), np.load(f"{v.E27}/resid.npz")
    fit = R27["fit"]
    m = dict(np.load("data/walks/meta.npz"))
    F = v.features(m["x"][fit], m["y"][fit], m["theta"][fit])
    res = {"experiment": "32, unregistered"}
    for g in (21, 19):
        A, _ = v.fit_directions(Z27[f"fit_{g}"].astype(np.float64), F)
        Q = R27[f"b{g}/Qord30"].astype(np.float64)
        res[f"block{g}"] = {}
        for name, keep in v.FEATS.items():
            U = np.linalg.svd(A[:, keep], full_matrices=False)[0]
            sh = {k: float(((Q[:, ix].T @ U) ** 2).sum() / U.shape[1]) for k, ix in v.PAIRS[g].items()}
            res[f"block{g}"][name] = sh
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
