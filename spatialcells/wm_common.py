"""Shared helpers for the follow-up pilot's model adapters (wm_vint, wm_cut3r; docs/experiments.md, pilot WP).

The models' code and weights stay outside this repository (~/wm_models/, each family in its own virtual
environment); adapters put a checkout on sys.path and import from it. Checkpoints are unpickled only with
weights_only=True and an allowlist of the classes they hold (safe_load), never with arbitrary unpickling.
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import torch

WM_ROOT = os.path.expanduser("~/wm_models")


def device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def add_path(path):
    path = os.path.expanduser(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def _resolve(name):
    mod, _, attr = name.rpartition(".")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def safe_load(path, allow_prefixes, stub_prefixes=()):
    """Load a checkpoint without arbitrary unpickling. A plain state dict goes through torch.load(weights_only=True).
    Otherwise only the globals the file itself lists may be resolved, each only if it comes from a module in
    allow_prefixes, plus torch's own weights-only set; globals from stub_prefixes (training-only objects such as LR
    schedulers whose package is not installed) become empty stand-in classes, and anything else is refused. This
    uses a restricted Unpickler rather than weights_only=True because torch's weights-only unpickler cannot rebuild
    the defaultdict in a pickled optimizer state."""
    import pickle
    import types
    from torch._weights_only_unpickler import _get_allowed_globals
    names = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    if not names:
        return torch.load(path, map_location="cpu", weights_only=True)
    allowed = dict(_get_allowed_globals())
    for n in names:
        if any(n.startswith(p) for p in stub_prefixes):
            allowed[n] = type(n.rpartition(".")[2], (), {"__init__": lambda self, *a, **k: None,
                                                         "__setstate__": lambda self, s: setattr(self, "state", s)})
        elif any(n.startswith(p) for p in allow_prefixes):
            allowed[n] = _resolve(n)
        else:
            raise RuntimeError(f"{path}: refusing to unpickle {n}")

    class Restricted(pickle.Unpickler):
        def find_class(self, mod, name):
            import _compat_pickle                       # protocol-2 names (__builtin__, copy_reg), as fix_imports does
            mod, name = _compat_pickle.NAME_MAPPING.get((mod, name), (_compat_pickle.IMPORT_MAPPING.get(mod, mod), name))
            full = f"{mod}.{name}"
            if full not in allowed:
                raise pickle.UnpicklingError(f"{path}: refusing to unpickle {full}")
            return allowed[full]

    restricted = types.SimpleNamespace(__name__="restricted_pickle", Unpickler=Restricted,
                                       load=lambda f, **kw: Restricted(f, **kw).load())
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=restricted)


def bootstrap_ci(stat, groups, n=10000, seed=0):
    """95% CI of stat(selected groups) over a bootstrap of groups (walks or rooms)."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    vals = [stat(groups[rng.integers(len(groups), size=len(groups))]) for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
