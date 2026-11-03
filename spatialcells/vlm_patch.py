"""Experiment 35 (pre-registered in docs/experiments.md): are there activations inside the fine-tuned VLM whose copy
into another trial moves the decision toward where the copied view was taken, and in how many dimensions? The VLM
counterpart of VGGT's camera-token patching (experiments 27 and 32).

    python -m spatialcells.vlm_patch pairs                             # results/export/vlm_patch/pairs.json, no model
    python -m spatialcells.vlm_patch dry-run [-n 24]                   # tokenise recipients and donors, no model
    python -m spatialcells.vlm_patch selftest                          # a tiny random Qwen3-VL on the CPU: every check
    python -m spatialcells.vlm_patch run --model qwen8b-ft-e4-s5 [--stages sweep fit subspace] [--batch 8] [--smoke]

Prompts. Experiment 33's walk condition (single_query.py), unchanged: the 61 evenly spaced frames that two_choice.py
--max-images 64 always shows, in time order with their real frame numbers, then "New view:", one teleport view and the
two-choice question (256 px, the chat template with the generation prompt). Both options are among the 61 frames, so
every prompt of a walk starts with the same 4,416-token walk prefix. It is prefilled once per walk; every forward
continues from that cache and is rewound after it (behave.py's Rewind). The 126-token suffix is "New view:",
<|vision_start|>, the view's 64 image tokens, <|vision_end|> and the question. A donor is shown under the recipient's
question, so its prompt differs from the recipient's only in the pixels behind the 64 image tokens, and every position
lines up token for token.

Pairs (`pairs`). Recipients: experiment 33's valid novel views (368), each with its own trial (near option i within
1 m, far option j more than 3 m away). Donors are other teleport views of the recipient's walk:
  place   within 1 m of j's position and more than 3 m from i's
  same    within 1 m of the recipient's position, facing more than 90 deg away from it (same place, another view)
  random  more than 3 m from both options
One of each, drawn uniformly (numpy default_rng([35, probe, kind])): the place donor among the novel candidates if
there are any, each control among the candidates as familiar or novel as the place donor if there are any (else among
all candidates). A recipient without a place donor is not run.

Score. S = log P(near label) - log P(far label): each label is "%03d", which Qwen's tokenizer splits into three digit
tokens, all teacher-forced after the generation prompt. S > 0 picks the near option. The near label's first two digits
are appended to the suffix in the same forward; the far label's are run after cropping the cache back to the suffix.

Sites. Layer L (0-35) is the residual stream after decoder layer L, including DeepStack's addition of the vision
tower's features to the image tokens after layers 0-2: the hidden state entering layer L + 1 (for L = 35, the final
norm). A copy replaces the recipient's hidden state at some suffix positions with the donor's, recorded in the donor's
own clean run under the recipient's question:
  view      the 64 image tokens and <|vision_end|> (<|vision_start|> precedes the image, so it is the same in both)
  view_end  <|vision_end|> only
  question  every token after <|vision_end|>, through the last prompt token
  last      the last prompt token only
A view copy at L = 0 or 1 also gives the image tokens the donor's DeepStack features for the later layers, so that from
L on they are exactly the donor's. A subspace copy replaces only the part of each view token's difference inside a
k-dim subspace U: h_r + (h_d - h_r) U U^T (its complement: h_r + (h_d - h_r)(I - U U^T)).

Stages (`run`; each resumes from its done file):
  sweep     per recipient: the clean runs (recipient, and each donor under the recipient's question, batched; their
            suffix states at every site are recorded), the recipient's greedy answer decoded exactly as experiment 33
            did, then the copies: place donor view / view_end / question / last at L = 0..35; same-place and
            random-place donors view / view_end at L = 0..35 (--sweep-sites narrows this); one row with the hooks on
            and nothing copied. -> done_sweep.jsonl
  fit       the subspace layers (choose_layers on the sweep, or --fit-layers): the last layers, from 2 to 34, at which
            copying the view still moves the place donor's decision at least 0.9, 0.5 (the primary) and 0.25 of the
            way. Records the 65 view tokens of all 3,072 teleport views at those layers and fits, separately on even
            and on odd walks: position PCA (the top directions of the per-1-m-bin means of each view's token mean, bins
            with >= 5 views, as vggt_patch.py's position subspace), ordinary PCA (the top directions of the view tokens'
            variance across views, each of the 65 token slots centred on its own mean) and readout (the 2-d span of a
            ridge readout of (x, y) from the token mean). -> layers.json, subspaces.npz, fit_report.json, fit_tokmean
            .npy (token means and <|vision_end|> per view and layer, float16)
  subspace  at those layers, with U fit on the other half's walks and k = 2, 10, 50: place donor position, ordinary,
            random (seeds 0-2), readout, and the view copy without position-U (k = 10, 50); same-place donor position,
            ordinary, random (seed 0); random-place donor position and ordinary 10; and the full view copy again for
            each donor. Random U: QR of a Gaussian (numpy default_rng([35, L, k, seed])), the same for every model and
            half. Also the share of each pair's view-token difference inside each U. -> done_subspace.jsonl
Checks that stop the run (checks.jsonl), on the first two recipients of the first two walks (every recipient with
--smoke): the cached path against one forward of the whole prompt without the cache (clean, and the place donor's view
copied at L = 17); the place donor's image fed whole into the recipient's prompt, equal to the place donor's clean S;
the view and question copied together at L = 0, 2, 17, 34, whose first label digit's log-odds must equal the place
donor's (the label tokens themselves still read the recipient's states at layers <= L, so S need not); the recipient's
own states copied into itself at those layers, equal to its clean S. For every recipient the hooks-on, nothing-copied row is compared with the clean S
and the greedy answer with experiment 33's stored one (the run stops if fewer than 90% of the first 20 match, or if more
than 1% of the hooks-on rows differ, after 50 recipients).
Tolerance: |dS| <= max(0.5, 0.03 G) + 1e-4 nats (amendment 1), G = |S_recipient - S_place donor| (selftest, in float32: 1e-3).
--smoke writes to data/vlm_patch_smoke/: two recipients from each of the first two walks that have one, every check,
and the fit on walks 0-3 only (bins with >= 2 views).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np

from .single_query import (DATA, MAX_NEW, MODEL, SHOWN, TRIALS, Data, model_adapter, processor, prompt, sites, split)
from .two_choice import parse, walk_order

PAIRS = "results/export/vlm_patch/pairs.json"
OUT = "data/vlm_patch"
EXP33 = ("results/exp33/{m}/done_walk.jsonl", "data/acts_sq/{m}/done_walk.jsonl")    # experiment 33's answers
SEED = 35
NEAR, FAR, AWAY = 1.0, 3.0, np.pi / 2
KINDS = ("place", "same", "random")
SWEEP_SITES = {"place": ("view", "view_end", "question", "last"), "same": ("view", "view_end"),
               "random": ("view", "view_end")}
KS = (2, 10, 50)
UNITS_KMAX = 50
MIN_BIN = 5
LAYER_MARKS = {"hi": 0.9, "primary": 0.5, "lo": 0.25}
TOL_ABS, TOL_REL = 0.5, 0.03
TOL_EPS = 1e-4   # amendment 1: bf16 logits put S on a 0.25-nat grid, so a two-step difference is 0.5 plus float32 rounding


def subspace_conds(ks=KS):
    """Conditions of the subspace stage, by donor kind."""
    place = (["view"] + [f"pos{k}" for k in ks] + [f"pca{k}" for k in ks]
             + [f"rand{k}s{s}" for k in ks for s in (0, 1, 2)] + ["readout2"] + [f"notpos{k}" for k in ks if k >= 10])
    same = ["view"] + [f"pos{k}" for k in ks] + [f"pca{k}" for k in ks] + [f"rand{k}s0" for k in ks]
    rand = ["view"] + [f"pos{k}" for k in ks] + [f"pca{k}" for k in ks if k == 10]
    return {"place": place, "same": same, "random": rand}


# ---------------------------------------------------------------- pairs (no model)
def build_pairs(data=DATA, seed=SEED):
    from .analyze_behave import angdiff
    T = json.load(open(TRIALS))
    assert [t["probe"] for t in T] == list(range(len(T)))
    meta, pm = dict(np.load(f"{data}/meta.npz")), dict(np.load(f"{data}/probes_meta.npz"))
    order = walk_order(meta)
    xy, th = np.c_[pm["x"], pm["y"]], pm["theta"]
    novel = np.array([not t["familiar"] for t in T])
    by_walk = {}
    for t in T:
        by_walk.setdefault(t["walk"], []).append(t["probe"])

    def frame_xy(w, k):                                       # frame number k (1-based) of walk w
        r = order[w][k - 1]
        return np.array([meta["x"][r], meta["y"][r]])

    pairs = []
    for t in T:
        if not t["valid"] or t["familiar"]:
            continue
        w, p = t["walk"], t["probe"]
        xi, xj = frame_xy(w, t["near"]), frame_xy(w, t["far"])
        c = np.array([q for q in by_walk[w] if q != p])
        di, dj = np.linalg.norm(xy[c] - xi, axis=1), np.linalg.norm(xy[c] - xj, axis=1)
        dr, dh = np.linalg.norm(xy[c] - xy[p], axis=1), angdiff(th[c], th[p])
        cand = {"place": (dj < NEAR) & (di > FAR), "same": (dr < NEAR) & (dh > AWAY), "random": (di > FAR) & (dj > FAR)}
        donors = {}
        for n, kind in enumerate(KINDS):
            pool = c[cand[kind]]
            if not len(pool):
                continue
            # place: novel if possible; the controls: the place donor's familiarity if possible (novel without one)
            want = donors["place"]["novel"] if "place" in donors else True
            pool_n = pool[novel[pool] == want]
            q = int(np.random.default_rng([seed, p, n]).choice(pool_n if len(pool_n) else pool))
            k = int(np.flatnonzero(c == q)[0])
            donors[kind] = {"probe": q, "novel": bool(novel[q]), "candidates": int(len(pool)),
                            "preferred_candidates": int(len(pool_n)), "dist_near_m": round(float(di[k]), 3),
                            "dist_far_m": round(float(dj[k]), 3), "dist_recipient_m": round(float(dr[k]), 3),
                            "heading_diff_deg": round(float(np.degrees(dh[k])), 1)}
        pairs.append({"id": t["id"], "walk": w, "probe": p, "near": t["near"], "far": t["far"], "a": t["a"],
                      "b": t["b"], "near_dist_m": t["near_dist_m"], "far_dist_m": t["far_dist_m"], "donors": donors})
    return pairs


def pairs_report(Pr):
    has = {k: [k in p["donors"] for p in Pr] for k in KINDS}
    run = [p for p in Pr if "place" in p["donors"]]
    rep = {"recipients": len(Pr), "with_place_donor (run)": len(run),
           "walks_with_a_run_recipient": len({p["walk"] for p in run}),
           "run_per_half_even_odd": [sum(p["walk"] % 2 == h for p in run) for h in (0, 1)]}
    for k in KINDS:
        d = [p["donors"][k] for p in run if k in p["donors"]]
        rep[k] = {"recipients_with_one": int(sum(has[k])), "among_run": len(d), "novel": sum(x["novel"] for x in d),
                  "median_candidates": float(np.median([x["candidates"] for x in d])),
                  "median_dist_recipient_m": float(np.median([x["dist_recipient_m"] for x in d])),
                  "median_dist_near_m": float(np.median([x["dist_near_m"] for x in d])),
                  "median_dist_far_m": float(np.median([x["dist_far_m"] for x in d])),
                  "median_heading_diff_deg": float(np.median([x["heading_diff_deg"] for x in d]))}
    rd = [p["donors"]["random"] for p in run if "random" in p["donors"]]
    rep["random"]["nearer_far_option"] = float(np.mean([x["dist_far_m"] < x["dist_near_m"] for x in rd]))
    for k in ("same", "random"):
        d = [p for p in run if k in p["donors"]]
        rep[k]["familiarity_as_place_donor"] = float(np.mean([p["donors"][k]["novel"] == p["donors"]["place"]["novel"]
                                                              for p in d]))
    rep["run_with_all_three"] = sum(all(k in p["donors"] for k in KINDS) for p in run)
    rep["run_with_place_and_same"] = sum("same" in p["donors"] for p in run)
    return rep


# ---------------------------------------------------------------- layer choice (shared with scripts/exp35_analysis.py)
def included(r):
    """A pair enters the fractions if the clean recipient picks the near option and the clean place donor, under the
    recipient's question, the far one."""
    S = r["S"]
    return "place" in S and S["recipient"] > 0 and S["place"] < 0


def view_profile(records, n_layers):
    """Mean fraction of the way to the place donor moved by copying the view, per layer, over included pairs."""
    F = [[(r["S"]["recipient"] - s) / (r["S"]["recipient"] - r["S"]["place"]) for s in r["sweep"]["place"]["view"]]
         for r in records if included(r)]
    return np.mean(F, 0) if F else np.full(n_layers, np.nan), len(F)


def choose_layers(records, n_layers):
    """The subspace layers: for each mark, the last layer from 2 to n - 2 at which the view copy moves the place
    donor's decision at least that far (mean fraction); the primary is the 0.5 mark's."""
    F, n = view_profile(records, n_layers)
    cand = np.arange(2, n_layers - 1)
    out = {"profile": [round(float(x), 4) for x in F], "n_pairs": n}
    for name, mark in LAYER_MARKS.items():
        ok = cand[F[cand] >= mark]
        out[name] = int(ok.max()) if len(ok) else None
    out["no_primary"] = out["primary"] is None
    if out["primary"] is None:                                 # registered fallback: layer 2, flagged
        out["primary"] = 2
    out["layers"] = sorted({x for x in (out["hi"], out["primary"], out["lo"]) if x is not None})
    return out


# ---------------------------------------------------------------- model side
class Engine:
    """The model, one walk's prefilled cache, and batched forwards of a suffix in which hidden states at chosen
    sites and positions are replaced (and others recorded). Hooks are forward pre-hooks on decoder layer L + 1 (the
    final norm for the last layer), so site L is the residual stream after layer L and its DeepStack addition."""

    def __init__(self, name, batch=8, device="cuda", model=None, proc=None, adapter=None, tol=(TOL_ABS, TOL_REL)):
        import torch
        self.torch, self.dev, self.batch, self.tol = torch, device, batch, tol
        if model is None:
            from .extract import load_qwen
            adapter = model_adapter(name)
            proc, model, _ = load_qwen(MODEL, False, adapter)
        self.adapter, self.proc, self.model, self.tok = adapter, proc, model, proc.tokenizer
        self.lm = model.model.language_model
        self.n = len(self.lm.layers)
        self.units = model.config.text_config.hidden_size
        self.plan, self.rec, self.rec_at, self.rec_n = {}, None, set(), 0
        for L in range(self.n):
            (self.lm.layers[L + 1] if L + 1 < self.n else self.lm.norm).register_forward_pre_hook(self._hook(L))
        eos = getattr(model.generation_config, "eos_token_id", None)
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        if not self.eos:                                       # the selftest's random model has no generation config
            self.eos = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id}
        self.D = Data()
        self.cache = self.w = None

    def _hook(self, L):
        def hook(_, args):
            h = args[0]
            todo = self.plan.get(L)
            if todo:
                h = h.clone()
                for row, idx, val in todo:
                    h[row, idx] = val
            if self.rec is not None and L in self.rec_at:
                self.rec[L] = h[:, :self.rec_n].clone()
            return (h,) + tuple(args[1:]) if todo else None
        return hook

    def rope(self, ids, mm, grid):
        torch = self.torch
        ext = torch.cat([ids, ids.new_zeros(1, MAX_NEW)], 1)
        mme = torch.cat([mm, mm.new_zeros(1, MAX_NEW)], 1)
        pos, _ = self.model.model.get_rope_index(ext, mm_token_type_ids=mme, image_grid_thw=grid)
        L = ids.shape[1]
        assert (pos[:, 0, L:] == pos[:, 0, L - 1].max() + 1 + torch.arange(MAX_NEW, device=pos.device)).all()
        return pos

    def prefill(self, w, t):
        """The walk prefix of trial t's prompt (every trial of walk w has the same one), into a rewindable cache."""
        torch = self.torch
        self.cache = self.rewind = None
        text, shown, _ = prompt(t, "walk", self.adapter, self.proc)
        pre, suf = split(text)
        self.imgs = self.D.walk_imgs(w, shown)
        pin = self.proc(text=[pre], images=self.imgs, return_tensors="pt").to(self.dev)
        sin = self.proc(text=[suf], images=[self.D.view(t["probe"])], return_tensors="pt").to(self.dev)
        P = pin["input_ids"].shape[1]
        pos = self.rope(torch.cat([pin["input_ids"], sin["input_ids"]], 1),
                        torch.cat([pin["mm_token_type_ids"], sin["mm_token_type_ids"]], 1),
                        torch.cat([pin["image_grid_thw"], sin["image_grid_thw"]]))
        o = self.model.model(input_ids=pin["input_ids"], position_ids=pos[:, :, :P], pixel_values=pin["pixel_values"],
                             image_grid_thw=pin["image_grid_thw"], use_cache=True)
        from .behave import Rewind
        self.cache, self.w, self.P, self.pre, self.shown, self.pin = o.past_key_values, w, P, pre, shown, pin
        self.pos_prefix = pos[:, :, :P]
        self.rewind = Rewind(self.cache)

    def inputs(self, t):
        """Trial t's suffix with its own view: ids, positions (suffix + MAX_NEW), sites, label digits."""
        torch = self.torch
        text, shown, lab = prompt(t, "walk", self.adapter, self.proc)
        pre, suf = split(text)
        assert pre == self.pre and shown == self.shown, f"{t['id']}: walk prefix differs from its walk's"
        sin = self.proc(text=[suf], images=[self.D.view(t["probe"])], return_tensors="pt").to(self.dev)
        ids = torch.cat([self.pin["input_ids"], sin["input_ids"]], 1)
        pos = self.rope(ids, torch.cat([self.pin["mm_token_type_ids"], sin["mm_token_type_ids"]], 1),
                        torch.cat([self.pin["image_grid_thw"], sin["image_grid_thw"]]))
        assert torch.equal(pos[:, :, :self.P], self.pos_prefix), f"{t['id']}: prefix positions differ"
        return self._pack(sin, pos[:, :, self.P:], lab, t, text)

    def _pack(self, sin, pos, lab, t, text):
        torch = self.torch
        s = sites(sin["input_ids"][0], self.tok)
        v1, Ls = s["img_end"], sin["input_ids"].shape[1]
        v0 = v1 - 64
        ar = lambda a, b: torch.arange(a, b, device=self.dev)
        dig = {}
        for k in ("near", "far"):
            d = self.tok.encode(f"{lab[t[k]]:03d}", add_special_tokens=False)
            assert len(d) == 3 and self.tok.decode(d) == f"{lab[t[k]]:03d}", (lab[t[k]], d)
            dig[k] = torch.tensor([d], device=self.dev)
        return {"ids": sin["input_ids"], "pos": pos, "Ls": Ls, "v0": v0, "v1": v1, "sin": sin, "text": text,
                "lab": lab, "near": dig["near"], "far": dig["far"],
                "idx": {"view": ar(v0, v1 + 1), "view_end": ar(v1, v1 + 1), "question": ar(v1 + 1, Ls),
                        "last": ar(Ls - 1, Ls), "suffix": ar(v0 - 1, Ls)}}

    def vision(self, img):
        """The image's merged features (64 x units) and its DeepStack features (one per level)."""
        x = self.proc.image_processor(images=[img], return_tensors="pt")
        v = self.model.model.get_image_features(x["pixel_values"].to(self.dev), x["image_grid_thw"].to(self.dev))
        return v.pooler_output[0], [f[0] for f in v.deepstack_features]

    def run_rows(self, rows, X, labels=True, record=(), n_tok=None):
        """One batched forward of the suffix X for each row, continuing the walk's cache, then (labels) the far
        label's digits after cropping back to the suffix. rows: {"feats": (merged, deepstack), "patch": [(L, idx,
        values)]}. Returns (rows x 2: S and the first digit's log-odds, or None; recordings {L: rows x n_tok x
        units})."""
        torch = self.torch
        B, n_tok = len(rows), n_tok or X["Ls"]
        ids = X["ids"][:, :n_tok]
        if labels:
            ids = torch.cat([ids, X["near"][:, :2]], 1)
        pos = X["pos"][:, :, :ids.shape[1]]
        self.rewind()
        self.rewind.expand(B)
        n_ds = len(rows[0]["feats"][1])
        mm = {"image": SimpleNamespace(pooler_output=tuple(r["feats"][0] for r in rows),
                                       deepstack_features=[tuple(r["feats"][1][l] for r in rows) for l in range(n_ds)])}
        self.plan = {}
        for b, r in enumerate(rows):
            for L, idx, val in r.get("patch", ()):
                self.plan.setdefault(L, []).append((b, idx, val))
        self.rec, self.rec_at, self.rec_n = ({} if record else None), set(record), n_tok
        try:
            o = self.model.model(input_ids=ids.expand(B, -1), position_ids=pos.expand(-1, B, -1),
                                 past_key_values=self.cache, use_cache=True, mm_encoder_outputs=mm)
            rec, S = self.rec, None
        finally:
            self.plan, self.rec = {}, None
        if labels:
            near, far = X["near"][0], X["far"][0]
            lp = torch.log_softmax(self.model.lm_head(o.last_hidden_state[:, n_tok - 1:n_tok + 2]).float(), -1)
            s_near = lp[:, 0, near[0]] + lp[:, 1, near[1]] + lp[:, 2, near[2]]
            s_far = lp[:, 0, far[0]]
            for layer in self.cache.layers:                    # back to the end of the suffix
                layer.keys, layer.values = layer.keys[..., :self.P + n_tok, :], layer.values[..., :self.P + n_tok, :]
            o2 = self.model.model(input_ids=X["far"][:, :2].expand(B, -1),
                                  position_ids=X["pos"][:, :, n_tok:n_tok + 2].expand(-1, B, -1),
                                  past_key_values=self.cache, use_cache=True)
            lp2 = torch.log_softmax(self.model.lm_head(o2.last_hidden_state).float(), -1)
            S = torch.stack([s_near - s_far - lp2[:, 0, far[1]] - lp2[:, 1, far[2]],
                             lp[:, 0, near[0]] - lp[:, 0, far[0]]], 1).cpu().numpy().astype(float)
        self.rewind()
        return S, rec

    def run_many(self, rows, X, **kw):
        out = []
        for b0 in range(0, len(rows), self.batch):
            out.append(self.run_rows(rows[b0:b0 + self.batch], X, **kw)[0])
        return np.concatenate(out) if out else np.zeros((0, 2))

    def greedy(self, X):
        """Experiment 33's decoding, step for step (batch 1, the view's pixels in the forward): (answer, reply)."""
        torch = self.torch
        self.rewind()
        sin, Ls = X["sin"], X["Ls"]
        o = self.model.model(input_ids=X["ids"], position_ids=X["pos"][:, :, :Ls], past_key_values=self.cache,
                             use_cache=True, pixel_values=sin["pixel_values"], image_grid_thw=sin["image_grid_thw"])
        logits, toks = self.model.lm_head(o.last_hidden_state[:, -1]).float(), []
        for k in range(MAX_NEW):
            t = int(logits.argmax(-1))
            if t in self.eos:
                break
            toks.append(t)
            if k + 1 == MAX_NEW:
                break
            o = self.model.model(input_ids=torch.tensor([[t]], device=self.dev),
                                 position_ids=X["pos"][:, :, Ls + k:Ls + k + 1], past_key_values=self.cache,
                                 use_cache=True)
            logits = self.model.lm_head(o.last_hidden_state[:, -1]).float()
        self.rewind()
        reply = self.tok.decode(toks, skip_special_tokens=False)
        for sp in {self.tok.pad_token, self.tok.eos_token, "<|im_end|>", "<|endoftext|>"} - {None}:
            reply = reply.replace(sp, "")
        t, lab = X["trial"], X["lab"]
        return {lab[t["a"]]: t["a"], lab[t["b"]]: t["b"]}.get(parse(reply, lab[t["a"]], lab[t["b"]]), -1), reply

    def full(self, X, img, patch=()):
        """(S, first digit's log-odds) from one forward of the whole prompt per label, without the cache (batch 1).
        patch: (L, suffix idx, values) as in run_rows."""
        torch = self.torch
        inp = self.proc(text=[X["text"]], images=self.imgs + [img], return_tensors="pt").to(self.dev)
        ids, L = inp["input_ids"], inp["input_ids"].shape[1]
        assert L == self.P + X["Ls"]
        pos = torch.cat([self.pos_prefix, X["pos"]], 2)
        logp, first = {}, {}
        for k in ("near", "far"):
            self.plan = {}
            for Lp, idx, val in patch:
                self.plan.setdefault(Lp, []).append((0, idx + self.P, val))
            try:
                o = self.model.model(input_ids=torch.cat([ids, X[k][:, :2]], 1), position_ids=pos[:, :, :L + 2],
                                     pixel_values=inp["pixel_values"], image_grid_thw=inp["image_grid_thw"],
                                     use_cache=False)
            finally:
                self.plan = {}
            lp = torch.log_softmax(self.model.lm_head(o.last_hidden_state[:, L - 1:L + 2]).float(), -1)[0]
            logp[k] = float(sum(lp[j, X[k][0, j]] for j in range(3)))
            first[k] = lp[0]
        return logp["near"] - logp["far"], float(first["near"][X["near"][0, 0]] - first["near"][X["far"][0, 0]])

    def within(self, a, b, G):
        return abs(a - b) <= max(self.tol[0], self.tol[1] * abs(G)) + (TOL_EPS if self.tol[0] >= TOL_ABS else 0.0)


def ds_mix(rec_feats, don_feats, L):
    """The recipient's merged features, with the donor's DeepStack features for the levels added after layer L."""
    return rec_feats[0], [rec_feats[1][l] if l <= L else don_feats[1][l] for l in range(len(rec_feats[1]))]


def exp33_answers(model):
    for pat in EXP33:
        path = pat.format(m=model)
        if os.path.exists(path):
            return {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(path)}, path
    return {}, None


def check_layers(n):
    return sorted({0, 2, n // 2 - 1, n - 2})


# ---------------------------------------------------------------- stage: sweep
def sweep(E, out, Pr, T, args, log):
    torch = E.torch
    path = f"{out}/done_sweep.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    stored, spath = exp33_answers(args.model)
    if not stored and not args.selftest:
        raise SystemExit(f"no experiment-33 answers for {args.model} ({EXP33}); stopping")
    sites_ = {k: tuple(s for s in v if s in args.sweep_sites) for k, v in SWEEP_SITES.items()}
    todo = [p for p in Pr if p["id"] not in done]
    print(f"sweep: {len(todo)} recipients to run ({len(done)} done); experiment 33's answers from {spath}", flush=True)
    agree = n_ref = n_rows = n_seen = none_bad = 0
    t0 = time.time()
    walks = sorted({p["walk"] for p in Pr})
    checked_walks = [w for w in walks][:2]
    for w in sorted({p["walk"] for p in todo}):
        rw = [p for p in todo if p["walk"] == w]
        E.prefill(w, T[rw[0]["probe"]])
        for j, pr in enumerate(rw):
            t = T[pr["probe"]]
            X = E.inputs(t)
            X["trial"] = t
            kinds = ["recipient"] + [k for k in KINDS if k in pr["donors"]]
            probe = {"recipient": pr["probe"], **{k: pr["donors"][k]["probe"] for k in kinds[1:]}}
            feats = {k: E.vision(E.D.view(probe[k])) for k in kinds}
            ans, reply = E.greedy(X)
            S_clean, rec = E.run_rows([{"feats": feats[k]} for k in kinds], X, record=range(E.n))
            S0 = dict(zip(kinds, map(float, S_clean[:, 0])))
            S01 = dict(zip(kinds, map(float, S_clean[:, 1])))
            H = {k: {L: rec[L][i] for L in rec} for i, k in enumerate(kinds)}
            del rec
            rows, keys = [{"feats": feats["recipient"]}], [("none",)]
            for kind in KINDS:
                if kind not in pr["donors"]:
                    continue
                for site in sites_[kind]:
                    idx = X["idx"][site]
                    for L in range(E.n):
                        f = ds_mix(feats["recipient"], feats[kind], L) if site == "view" else feats["recipient"]
                        rows.append({"feats": f, "patch": [(L, idx, H[kind][L][idx])]})
                        keys.append((kind, site, L))
            check = args.smoke or args.selftest or (w in checked_walks and j < 2)
            if check:
                rows.append({"feats": feats["place"]})
                keys.append(("check", "input", -1))
                for L in check_layers(E.n):
                    for name, src, site in (("suffix", "place", "suffix"), ("self", "recipient", "view")):
                        idx = X["idx"][site]
                        f = ds_mix(feats["recipient"], feats[src], L)
                        rows.append({"feats": f, "patch": [(L, idx, H[src][L][idx])]})
                        keys.append(("check", name, L))
            S = E.run_many(rows, X)
            n_rows += len(rows)
            res, res1 = dict(zip(keys, map(float, S[:, 0]))), dict(zip(keys, map(float, S[:, 1])))
            G = S0["recipient"] - S0["place"]
            rec_out = {"id": pr["id"], "walk": w, "probe": pr["probe"], "donors": {k: probe[k] for k in kinds[1:]},
                       "answer": ans, "reply": reply[-60:], "stored_answer": stored.get(pr["id"]),
                       "S": S0, "S_first_digit": S01, "S_none": res[("none",)],
                       "sweep": {k: {s: [res[(k, s, L)] for L in range(E.n)] for s in sites_[k]}
                                 for k in KINDS if k in pr["donors"]}}
            if pr["id"] in stored:
                n_ref += 1
                agree += ans == stored[pr["id"]]
            n_seen += 1
            none_bad += not E.within(res[("none",)], S0["recipient"], G)
            if check:
                # the view and question copied together make the last prompt token the donor's, hence the first
                # digit; the label tokens themselves still read the recipient's states at layers <= L
                c = {"input": res[("check", "input", -1)],
                     **{f"suffix_L{L}_first_digit": res1[("check", "suffix", L)] for L in check_layers(E.n)},
                     **{f"self_L{L}": res[("check", "self", L)] for L in check_layers(E.n)}}
                Lm = E.n // 2 - 1
                img_r, img_p = E.D.view(pr["probe"]), E.D.view(probe["place"])
                c["full_clean"] = E.full(X, img_r)[0]
                c["full_view_place_L%d" % Lm] = E.full(X, img_r, [(Lm, X["idx"]["view"],
                                                                   H["place"][Lm][X["idx"]["view"]])])[0]
                c["full_place"] = E.full(X, img_p)[0]
                want = {"input": S0["place"], "full_clean": S0["recipient"], "full_place": S0["place"],
                        "full_view_place_L%d" % Lm: res[("place", "view", Lm)], "none": S0["recipient"],
                        **{f"suffix_L{L}_first_digit": S01["place"] for L in check_layers(E.n)},
                        **{f"self_L{L}": S0["recipient"] for L in check_layers(E.n)}}
                c["none"] = res[("none",)]
                fails = [k for k, v in c.items() if not E.within(v, want[k], G)]
                rec_out["checks"] = c
                log({"stage": "sweep", "id": pr["id"], "S": S0, "G": G, "checks": c, "none": res[("none",)],
                     "fails": fails, "tol": list(E.tol)})
                print(f"  check {pr['id']}: S recipient {S0['recipient']:.3f}, place {S0['place']:.3f}; full recompute "
                      f"{c['full_clean']:.3f}; donor fed whole {c['input']:.3f}; view+question copied "
                      + " ".join(f"{c[f'suffix_L{L}_first_digit']:.3f}" for L in check_layers(E.n))
                      + f" (first digit; donor's {S01['place']:.3f}); self "
                      + " ".join(f"{c[f'self_L{L}']:.3f}" for L in check_layers(E.n))
                      + f"; fails {fails or 'none'}", flush=True)
                if fails:
                    raise SystemExit(f"sweep {pr['id']}: checks {fails} outside tolerance; stopping")
            with open(path, "a") as fh:
                fh.write(json.dumps(rec_out) + "\n")
            del H
            if n_ref >= 20 and agree < 0.9 * n_ref and not args.selftest:
                raise SystemExit(f"only {agree} of {n_ref} greedy answers match experiment 33's; stopping")
            if n_seen >= 50 and none_bad > 0.01 * n_seen:
                raise SystemExit(f"{none_bad} of {n_seen} hooks-on rows differ from the clean S; stopping")
        print(f"  walk {w}: {len(rw)} recipients, {time.time() - t0:.0f}s, {n_rows} copied rows "
              f"({n_rows / max(time.time() - t0, 1e-9):.1f}/s), answers matching experiment 33 {agree}/{n_ref}"
              + (f"; peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB" if E.dev == "cuda" else ""), flush=True)


# ---------------------------------------------------------------- stage: fit
def readout(X, Y, groups, folds=4, rel=np.logspace(-4, 2, 13)):
    """Ridge readout of Y from X, penalty (relative to the mean squared singular value) by grouped CV: W (d x 2)."""
    g = np.unique(groups)
    f = np.array([dict(zip(g, np.arange(len(g)) % folds))[x] for x in groups])

    def fit(Xa, Ya):
        mx, my = Xa.mean(0), Ya.mean(0)
        U, s, Vt = np.linalg.svd(Xa - mx, full_matrices=False)
        return mx, my, U, s, Vt, (s ** 2).mean()

    err = np.zeros(len(rel))
    for k in range(folds):
        tr, te = f != k, f == k
        mx, my, U, s, Vt, sc = fit(X[tr], Y[tr])
        UY = U.T @ (Y[tr] - my)
        for i, c in enumerate(rel):
            W = Vt.T @ ((s / (s ** 2 + c * sc))[:, None] * UY)
            err[i] += ((((X[te] - mx) @ W + my) - Y[te]) ** 2).sum()
    mx, my, U, s, Vt, sc = fit(X, Y)
    c = rel[int(np.argmin(err))]
    return Vt.T @ ((s / (s ** 2 + c * sc))[:, None] * (U.T @ (Y - my))), mx, my, float(c)


def pos_bins(xy):
    return np.clip(xy[:, 0].astype(int), 0, 9) * 10 + np.clip(xy[:, 1].astype(int), 0, 9)


def overlap(Q1, Q2):
    """Mean squared cosine of the principal angles (experiment 27's overlap)."""
    return float(((Q1.T @ Q2) ** 2).sum() / min(Q1.shape[1], Q2.shape[1]))


def fit_stage(E, out, T, layers, args):
    torch = E.torch
    if os.path.exists(f"{out}/subspaces.npz") and os.path.exists(f"{out}/fit_report.json"):
        rep = json.load(open(f"{out}/fit_report.json"))
        assert rep["layers"] == layers, (rep["layers"], layers)
        print(f"fit: done already ({out}/subspaces.npz)", flush=True)
        return
    pm = dict(np.load(f"{DATA}/probes_meta.npz"))
    walks = sorted({t["walk"] for t in T})
    if args.smoke or args.selftest:
        walks = walks[:4]
    min_bin = 2 if (args.smoke or args.selftest) else MIN_BIN
    V = [t for t in T if t["walk"] in walks]
    d = E.units
    stats = {(L, h): {"n": 0, "S1": torch.zeros(65, d, dtype=torch.float64, device=E.dev),
                      "XtX": torch.zeros(d, d, dtype=torch.float64, device=E.dev)} for L in layers for h in (0, 1)}
    tokmean = {L: np.zeros((len(T), d), np.float32) for L in layers}
    vend = {L: np.zeros((len(T), d), np.float32) for L in layers}
    t0 = time.time()
    for w in walks:
        tw = [t for t in V if t["walk"] == w]
        E.prefill(w, tw[0])
        X = E.inputs(tw[0])
        vi = X["idx"]["view"]
        for b0 in range(0, len(tw), E.batch):
            tb = tw[b0:b0 + E.batch]
            _, rec = E.run_rows([{"feats": E.vision(E.D.view(t["probe"]))} for t in tb], X, labels=False,
                                record=layers, n_tok=X["v1"] + 1)
            for L in layers:
                A = rec[L][:, vi].double()                         # views x 65 x units
                st = stats[L, w % 2]
                st["n"] += len(tb)
                st["S1"] += A.sum(0)
                Af = A.reshape(-1, d)
                st["XtX"] += Af.T @ Af
                p = [t["probe"] for t in tb]
                tokmean[L][p] = A.mean(1).float().cpu().numpy()
                vend[L][p] = A[:, -1].float().cpu().numpy()
            del rec
        print(f"  fit: walk {w}, {time.time() - t0:.0f}s", flush=True)
    rows = np.array(sorted(t["probe"] for t in V))
    xy = np.c_[pm["x"], pm["y"]][rows]
    half = pm["walk"][rows] % 2
    U, rep = {}, {"layers": layers, "min_bin": min_bin, "walks": walks, "per_layer": {}}
    for L in layers:
        R = {}
        for h in (0, 1):
            st, sel = stats[L, h], half == h
            C = (st["XtX"] - st["S1"].T @ st["S1"] / st["n"]) / (st["n"] * 65)
            ev, evec = torch.linalg.eigh(C)
            pca = evec[:, torch.argsort(ev, descending=True)[:UNITS_KMAX]].float().cpu().numpy()
            M = tokmean[L][rows[sel]].astype(np.float64)
            bins = pos_bins(xy[sel])
            ub, cnt = np.unique(bins, return_counts=True)
            means = np.stack([M[bins == b].mean(0) for b in ub[cnt >= min_bin]])
            pos = np.linalg.svd(means - means.mean(0), full_matrices=False)[2][:UNITS_KMAX].T.astype(np.float32)
            W, mx, my, c = readout(M, xy[sel], pm["walk"][rows[sel]])
            ro = np.linalg.qr(W)[0].astype(np.float32)
            U[L, h] = {"pos": pos, "pca": pca, "readout": ro}
            Ct = C.float()
            tr = float(torch.trace(Ct))
            vf = lambda Q: float(torch.trace(torch.from_numpy(Q).to(E.dev).T @ Ct @ torch.from_numpy(Q).to(E.dev)) / tr)
            Mo = tokmean[L][rows[~sel]].astype(np.float64)
            pred = (Mo - mx) @ W + my
            R[f"h{h}"] = {
                "views": int(st["n"]), "bins": int((cnt >= min_bin).sum()), "pos_dims": int(pos.shape[1]),
                "readout_penalty_rel": c,
                "readout_test_median_error_m": float(np.median(np.linalg.norm(pred - xy[~sel], axis=1))),
                "readout_test_chance_m": float(np.median(np.linalg.norm(xy[sel].mean(0) - xy[~sel], axis=1))),
                "var_frac": {**{f"pos{k}": vf(pos[:, :k]) for k in KS if k <= pos.shape[1]},
                             **{f"pca{k}": vf(pca[:, :k]) for k in KS}, "readout2": vf(ro)},
                "overlap_pos_pca": {str(k): overlap(pos[:, :k], pca[:, :k]) for k in KS if k <= pos.shape[1]},
                "overlap_readout_pos": {str(k): overlap(ro, pos[:, :k]) for k in KS if k <= pos.shape[1]},
                "overlap_readout_pca": {str(k): overlap(ro, pca[:, :k]) for k in KS}}
        R["overlap_halves"] = {f"{kind}{k}": overlap(U[L, 0][kind][:, :k], U[L, 1][kind][:, :k])
                               for kind in ("pos", "pca") for k in KS
                               if k <= min(U[L, 0][kind].shape[1], U[L, 1][kind].shape[1])}
        R["overlap_halves"]["readout2"] = overlap(U[L, 0]["readout"], U[L, 1]["readout"])
        rep["per_layer"][str(L)] = R
    np.savez(f"{out}/subspaces.npz", **{f"L{L}/h{h}/{kind}": U[L, h][kind] for (L, h) in U for kind in U[L, h]})
    np.savez(f"{out}/fit_tokmean.npz", rows=rows, **{f"tokmean_L{L}": tokmean[L][rows].astype(np.float16)
                                                     for L in layers},
             **{f"view_end_L{L}": vend[L][rows].astype(np.float16) for L in layers})
    json.dump(rep, open(f"{out}/fit_report.json", "w"), indent=1)
    print(f"fit: done, {time.time() - t0:.0f}s; " + "; ".join(
        f"L{L}: pos10-pca10 overlap {rep['per_layer'][str(L)]['h0']['overlap_pos_pca'].get('10', float('nan')):.2f}, "
        f"readout error {rep['per_layer'][str(L)]['h0']['readout_test_median_error_m']:.2f} m "
        f"(chance {rep['per_layer'][str(L)]['h0']['readout_test_chance_m']:.2f})" for L in layers), flush=True)


# ---------------------------------------------------------------- stage: subspace
def random_U(units, L, k, s):
    return np.linalg.qr(np.random.default_rng([SEED, L, k, s]).standard_normal((units, k)))[0].astype(np.float32)


def subspace_stage(E, out, Pr, T, layers, args, log):
    torch = E.torch
    path = f"{out}/done_subspace.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    Z = np.load(f"{out}/subspaces.npz")
    ks = tuple(k for k in KS if all(Z[f"L{L}/h{h}/pos"].shape[1] >= k for L in layers for h in (0, 1)))
    conds = subspace_conds(ks)
    dev = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(E.dev)
    Q = {}
    for L in layers:
        for h in (0, 1):
            for k in ks:
                Q[L, h, f"pos{k}"] = dev(Z[f"L{L}/h{h}/pos"][:, :k])
                Q[L, h, f"pca{k}"] = dev(Z[f"L{L}/h{h}/pca"][:, :k])
                for s in (0, 1, 2):
                    Q[L, h, f"rand{k}s{s}"] = dev(random_U(E.units, L, k, s))
            Q[L, h, "readout2"] = dev(Z[f"L{L}/h{h}/readout"])
    todo = [p for p in Pr if p["id"] not in done]
    print(f"subspace: layers {layers}, k {ks}, {len(todo)} recipients to run ({len(done)} done)", flush=True)
    t0, n_rows = time.time(), 0
    for w in sorted({p["walk"] for p in todo}):
        rw = [p for p in todo if p["walk"] == w]
        E.prefill(w, T[rw[0]["probe"]])
        h = 1 - w % 2                                          # U fit on the other half's walks
        for pr in rw:
            t = T[pr["probe"]]
            X = E.inputs(t)
            X["trial"] = t
            kinds = ["recipient"] + [k for k in KINDS if k in pr["donors"]]
            probe = {"recipient": pr["probe"], **{k: pr["donors"][k]["probe"] for k in kinds[1:]}}
            feats = {k: E.vision(E.D.view(probe[k])) for k in kinds}
            S_clean, rec = E.run_rows([{"feats": feats[k]} for k in kinds], X, record=layers)
            S0 = dict(zip(kinds, map(float, S_clean[:, 0])))
            vi = X["idx"]["view"]
            rows, keys, share = [], [], {}
            for L in layers:
                hr = rec[L][0, vi].float()
                for i, kind in enumerate(kinds[1:], 1):
                    hd = rec[L][i, vi]
                    dlt = hd.float() - hr
                    n2 = float((dlt ** 2).sum())
                    for c in conds[kind]:
                        if c == "view":
                            val = hd
                        else:
                            q = Q[L, h, "pos" + c[6:] if c.startswith("notpos") else c]
                            proj = (dlt @ q) @ q.T
                            share[f"{L}/{kind}/{c}"] = float((proj ** 2).sum()) / n2 if not c.startswith("notpos") \
                                else 1 - float((proj ** 2).sum()) / n2
                            val = (hr + (dlt - proj if c.startswith("notpos") else proj)).to(hd.dtype)
                        rows.append({"feats": feats["recipient"], "patch": [(L, vi, val)]})
                        keys.append((L, kind, c))
            del rec
            S = E.run_many(rows, X)
            n_rows += len(rows)
            res = {}
            for (L, kind, c), s in zip(keys, S[:, 0]):
                res.setdefault(str(L), {}).setdefault(kind, {})[c] = float(s)
            with open(path, "a") as fh:
                fh.write(json.dumps({"id": pr["id"], "walk": w, "probe": pr["probe"], "half_of_U": h, "S": S0,
                                     "layers": res, "share": share}) + "\n")
        print(f"  walk {w}: {len(rw)} recipients, {time.time() - t0:.0f}s, {n_rows} rows "
              f"({n_rows / max(time.time() - t0, 1e-9):.1f}/s)", flush=True)


# ---------------------------------------------------------------- run
def run(args, E=None):
    torch_ = __import__("torch")
    Pr = [p for p in json.load(open(PAIRS)) if "place" in p["donors"]]
    T = json.load(open(TRIALS))
    if args.smoke or args.selftest:
        ws = sorted({p["walk"] for p in Pr})[:2]
        Pr = [p for w in ws for p in [q for q in Pr if q["walk"] == w][:2]]
    base = args.out or (f"{OUT}_smoke" if args.smoke else OUT)
    out = f"{base}/{args.model}"
    os.makedirs(out, exist_ok=True)
    E = E or Engine(args.model, batch=args.batch)
    meta = {"model": MODEL, "adapter": E.adapter, "pairs": PAIRS, "recipients": [p["id"] for p in Pr],
            "sweep_sites": list(args.sweep_sites), "n_layers": E.n, "smoke": bool(args.smoke), "tol": list(E.tol)}
    if os.path.exists(f"{out}/meta.json"):
        old = json.load(open(f"{out}/meta.json"))
        assert all(old[k] == meta[k] for k in ("adapter", "recipients", "sweep_sites")), "meta.json differs"
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)

    def log(r):
        with open(f"{out}/checks.jsonl", "a") as fh:
            fh.write(json.dumps(r) + "\n")

    print(f"{args.model} ({E.adapter or 'base'}): {len(Pr)} recipients, stages {args.stages}, batch {E.batch} -> {out}",
          flush=True)
    with torch_.inference_mode():
        for stage in args.stages:
            if stage == "sweep":
                sweep(E, out, Pr, T, args, log)
                continue
            if args.fit_layers:
                lay = {"layers": sorted(args.fit_layers), "primary": None, "given": True}
            else:
                recs = [json.loads(l) for l in open(f"{out}/done_sweep.jsonl")]
                assert {r["id"] for r in recs} == {p["id"] for p in Pr}, "the sweep is not complete"
                lay = choose_layers(recs, E.n)
            if os.path.exists(f"{out}/layers.json"):
                assert json.load(open(f"{out}/layers.json"))["layers"] == lay["layers"], "layers.json differs"
            json.dump(lay, open(f"{out}/layers.json", "w"), indent=1)
            print(f"layers: {lay['layers']} (primary {lay['primary']})", flush=True)
            if stage == "fit":
                fit_stage(E, out, T, lay["layers"], args)
            elif stage == "subspace":
                subspace_stage(E, out, Pr, T, lay["layers"], args, log)
    print(f"{args.model}: all done -> {out}", flush=True)


# ---------------------------------------------------------------- dry run and selftest (no GPU)
def dry_run(args):
    import torch
    proc, D = processor(), Data()
    tok = proc.tokenizer
    T = json.load(open(TRIALS))
    Pr = [p for p in json.load(open(PAIRS)) if "place" in p["donors"]][:args.n]
    t0 = time.time()
    for n, pr in enumerate(Pr):
        t = T[pr["probe"]]
        text, shown, lab = prompt(t, "walk", None, proc)
        pre, suf = split(text)
        assert shown == list(SHOWN) or set(SHOWN) <= set(shown)
        ids = {}
        for kind, p in [("recipient", pr["probe"])] + [(k, d["probe"]) for k, d in pr["donors"].items()]:
            assert T[p]["walk"] == t["walk"]
            sin = proc(text=[suf], images=[D.view(p)], return_tensors="pt")
            ids[kind] = sin["input_ids"][0]
            px = proc.image_processor(images=[D.view(p)], return_tensors="pt")["pixel_values"]
            assert torch.equal(px, sin["pixel_values"]), "image_processor and processor pixels differ"
        assert all(torch.equal(v, ids["recipient"]) for v in ids.values()), f"{pr['id']}: donor suffix ids differ"
        s = sites(ids["recipient"], tok)
        v1, Ls = s["img_end"], len(ids["recipient"])
        digits = {k: tok.encode(f"{lab[t[k]]:03d}", add_special_tokens=False) for k in ("near", "far")}
        assert all(len(d) == 3 for d in digits.values()), digits
        assert tok.decode(ids["recipient"][v1 - 65:v1 - 64]) == "<|vision_start|>"
        if n == 0:
            show = lambda a, b: tok.convert_ids_to_tokens(ids["recipient"][a:b].tolist())
            print(f"{pr['id']} (walk {t['walk']}, near {t['near']}, far {t['far']}; donors "
                  f"{ {k: d['probe'] for k, d in pr['donors'].items()} }): suffix {Ls} tokens; vision_start at "
                  f"{v1 - 65}, image tokens {v1 - 64}-{v1 - 1}, vision_end {v1}, question {v1 + 1}-{Ls - 1} "
                  f"({Ls - v1 - 1} tokens)")
            print(f"  suffix head {show(0, v1 - 63)}")
            print(f"  after the view {show(v1, v1 + 6)} ... tail {show(Ls - 6, Ls)}")
            print(f"  labels: near {lab[t['near']]:03d} -> {tok.convert_ids_to_tokens(digits['near'])}, far "
                  f"{lab[t['far']]:03d} -> {tok.convert_ids_to_tokens(digits['far'])}")
    print(f"dry run: {len(Pr)} recipients; every donor's suffix ids equal the recipient's (only pixels differ); "
          f"image_processor pixels equal the processor's; labels are 3 digit tokens ({time.time() - t0:.0f}s)")


def tiny_model():
    """A small random Qwen3-VL with the real processor's token ids and image geometry (CPU, float32)."""
    import torch
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    torch.manual_seed(0)
    cfg = Qwen3VLConfig(
        text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, vocab_size=151936, rms_norm_eps=1e-6,
                         rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True},
                         rope_theta=5e6, max_position_embeddings=32768, tie_word_embeddings=False),
        vision_config=dict(depth=4, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=16,
                           spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64, in_channels=3,
                           deepstack_visual_indexes=[1, 2, 3], num_position_embeddings=2304),
        image_token_id=151655, video_token_id=151656, vision_start_token_id=151652, vision_end_token_id=151653,
        tie_word_embeddings=False)
    cfg._attn_implementation = "sdpa"
    model = Qwen3VLForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():                         # larger logits, so that labels' log-probabilities differ
        model.lm_head.weight.mul_(30)
    return model


def selftest(args):
    """The whole run path on a tiny random model on the CPU, in float32, with every check at tolerance 1e-3:
    two recipients from each of two walks, the sweep (with its checks on every recipient), the fit on walks 0-3 and
    the subspace stage, into a scratch folder."""
    import tempfile
    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    model, proc = tiny_model(), processor()
    E = Engine("tiny", batch=args.batch, device="cpu", model=model, proc=proc, tol=(1e-3, 0.0))
    args.out = args.out or tempfile.mkdtemp(prefix="vlm_patch_selftest_")
    args.model, args.smoke, args.selftest = "tiny", False, True
    t0 = time.time()
    run(args, E)
    out = f"{args.out}/tiny"
    R = [json.loads(l) for l in open(f"{out}/done_sweep.jsonl")]
    Lq = E.n - 1
    for r in R:                                   # structural: copies of non-final tokens after the last layer
        S = r["S"]["recipient"]
        for kind, sw in r["sweep"].items():
            for site in ("view", "view_end"):
                assert abs(sw[site][Lq] - S) < 1e-3, (r["id"], kind, site, sw[site][Lq], S)
    sub = [json.loads(l) for l in open(f"{out}/done_subspace.jsonl")]
    for s, r in zip(sub, R):                      # the subspace stage's full view copy equals the sweep's
        for L, d in s["layers"].items():
            for kind in d:
                assert abs(d[kind]["view"] - r["sweep"][kind]["view"][int(L)]) < 1e-3, (s["id"], L, kind)
    print(f"selftest passed ({time.time() - t0:.0f}s): {len(R)} recipients, {E.n} layers, every check within 1e-3; "
          f"copies after the last layer change nothing; the subspace stage's view copy equals the sweep's -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pairs", "dry-run", "selftest", "run"])
    ap.add_argument("--model", default="qwen8b-ft-e4-s5", help="extract.py's name: qwen8b or qwen8b-ft-e4-s3 ... -s7")
    ap.add_argument("--stages", nargs="+", default=["sweep", "fit", "subspace"], choices=["sweep", "fit", "subspace"])
    ap.add_argument("--sweep-sites", nargs="+", default=["view", "view_end", "question", "last"],
                    choices=["view", "view_end", "question", "last"], help="sweep only these sites (base model: view)")
    ap.add_argument("--fit-layers", nargs="+", type=int, help="fit these layers instead of choose_layers' (base model)")
    ap.add_argument("--batch", type=int, default=8, help="copied rows per forward (the cache is expanded to this)")
    ap.add_argument("--smoke", action="store_true", help="two recipients from each of two walks, every check")
    ap.add_argument("--out", help="output folder (default data/vlm_patch, or _smoke)")
    ap.add_argument("--pairs-out", default=PAIRS, help="pairs: where to write")
    ap.add_argument("-n", type=int, default=24, help="dry run: recipients to tokenise")
    args = ap.parse_args()
    args.selftest = False
    if args.mode == "pairs":
        Pr = build_pairs()
        os.makedirs(os.path.dirname(args.pairs_out), exist_ok=True)
        json.dump(Pr, open(args.pairs_out, "w"), indent=1)
        print(json.dumps(pairs_report(Pr), indent=1))
        print(f"{len(Pr)} recipients -> {args.pairs_out}")
    elif args.mode == "dry-run":
        dry_run(args)
    elif args.mode == "selftest":
        selftest(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
