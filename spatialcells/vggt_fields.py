"""Field size and spatial information of heading-invariant vs heading-specific place units
(VGGT block 19 and 23, with Qwen3-VL-8B + 8-step average for comparison). Needs the
activations from vggt_cells.py and the Qwen runs.

    python -m spatialcells.vggt_fields
"""
import numpy as np, torch
from spatialcells.analyze import load_units, analyze_layer, field_stats
meta = dict(np.load("data/walks/meta.npz"))
for tag, layer in [("vggt-single", "L19"), ("vggt-walk", "L19"), ("vggt-single", "L23"), ("qwen8b-single+ema8", "L13")]:
    X = load_units(tag, layer, "data/walks", meta)
    _, pu = analyze_layer(X, meta, np.random.default_rng(0))
    p = pu["place"]
    _, cover = field_stats(pu["maps"][p], pu["valid"])
    si, hi = pu["si"][p].cpu().numpy(), pu["hi"][p].cpu().numpy()
    inv = hi > 0.5
    q = si >= np.quantile(si, 0.9)
    print(f"{tag} {layer}: {p.sum().item()} place poles; invariant {inv.mean():.0%}")
    print(f"  SI median: invariant {np.median(si[inv]):.2f} vs not {np.median(si[~inv]):.2f}")
    print(f"  field coverage median: invariant {np.median(cover[inv]):.2f} vs not {np.median(cover[~inv]):.2f}")
    print(f"  top-10% SI place poles: invariant {inv[q].mean():.0%}; bottom-50% SI: {inv[si <= np.median(si)].mean():.0%}")
    print(f"  corr(SI, heading inv) = {np.corrcoef(si, hi)[0,1]:.2f}", flush=True)
