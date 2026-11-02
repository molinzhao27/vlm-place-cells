"""Experiment 33 (pre-registered in docs/experiments.md): every teleport view of room A on its own, as the new view
of the two-choice prompt.

    python -m spatialcells.single_query trials                  # results/export/single_query/trials.json (no model)
    python -m spatialcells.single_query dry-run [--model M] [-n 130] [--parity-all]   # tokenise only, no model
    python -m spatialcells.single_query record --model qwen8b-ft-e4-s5 [--conds parity walk shuffled nowalk] [--smoke]

Experiment 25's teleport-view activations (extract.py) put a walk's 64 views in one prompt after the walk, in
random order, with no question, so each view could attend to the views before it; and it scored all 3,072 views,
of which only 382 are novel. Here each view is shown alone, as "New view:" in exactly the prompt two_choice.py gives
the fine-tuned model with --max-images 64 (WALK_PROMPT, the shown walk frames as "Frame 001:" ... with their real
frame numbers, "New view:", the view, the question; 256 px; the chat template with the generation prompt).

Trials (`trials`). One per view (3,072), built with dissimilar_trials.py's rules for room A, with the options taken
from the 61 evenly spaced frames that --max-images 64 always shows (as procedural_data.py did for training): the
near option within 1 m of the view, the far one more than 3 m away, both facing more than 90 deg away from it and
within 45 deg of each other's turn, neither showing any poster or pillar the view shows; among such pairs the one
whose pixel similarity to the view is closest, with the sign of that difference alternating from trial to trial
(the other sign when only that one exists). Options a and b are ordered by a coin (seed 33). Because both options are
among the 61 frames, every view of a walk has the same walk context (61 frames, in time order or in two_choice's
--shuffle-walk order, which is seeded by the walk), so the view's own tokens are the only thing that differs before
its <|vision_end|>. A view with no valid pair gets a fallback pair (the nearest shown frame, and the frame more than
3 m away whose turn is closest to it) and valid = false: its image-end activation in the walk conditions is unaffected
(the question comes after it), but its answer and last-token activation are not used. Novel and familiar are
analyze_behave.geometry's (familiar: some frame of the view's walk within 1 m faced within 45 deg of it).

Conditions (`record`):
  walk      the 61 shown frames in time order (two_choice --max-images 64)
  shuffled  the same frames in two_choice's --shuffle-walk order (random.Random(walk), real frame numbers)
  nowalk    two_choice --no-walk: only the two options, then the view and the question
  parity    room A's 192 existing trials (results/export/dissimilar/trials.json) in each of the three conditions,
            exactly the prompts of the stored two_choice runs: token counts must match them (else the run stops),
            and the answers are compared with theirs
The walk is prefilled once per walk and condition; each view continues from that cache, which is rewound after it
(behave.py's Rewind), so views never see each other. Position ids are M-RoPE's for the whole prompt (get_rope_index).
The prefill is also where the walk frames are recorded, once per walk: under causal attention a walk frame's tokens
cannot depend on the view and question that follow them.
Checks that stop the run: for the first view of every walk, the prefix and suffix token ids equal the processor's ids
for the whole prompt; for the first two views of the first two walks of each condition, a full recompute without the
cache gives the same activations (cosine >= 0.99 for every kept layer, both kinds, at the view's two sites and at
all 61 walk frames, the walk frames compared with the prefill's, which saw no view) and the same first-token logits
(behave.py's tolerance). Answers: greedy, at most 16 new tokens, parsed as two_choice.py does.

Recorded, at layers 3, 23, 29 and 35 (of extract.py's 3, 8, 13, 18, 23, 29, 35; 8, 13 and 18 were never the best
layer of any four-epoch or base-model condition in experiment 25), two kinds of unit:
  mlp    the SwiGLU MLP neurons (the input to down_proj, 12,288), as extract.py; values beyond float16 set to 0
  resid  the residual stream, the decoder layer's output (4,096; for layer 35 before the final norm); values beyond
         float16 clipped, as task_record.py
at the view's <|vision_end|> token (img_end, the token experiment 25's probes used) and the last prompt token (last),
and, in walk and shuffled, at each shown walk frame's <|vision_end|> (walkframes; 48 x 61 rows, row = walk * 61 + the
frame's place among the 61 in time order, the same rows in both conditions). Writes data/acts_sq/<model>/
<tag>_<site>_<kind>.npy (float16, rows x layers x units; tag = walk, shuffled, nowalk or parity_<cond>), meta.json,
done_<tag>.jsonl (per view: answer, reply, prompt tokens, sites, whether the cache was used), done_walkframes_<cond>
.jsonl and checks.jsonl. A stopped run resumes from the done files. --smoke writes to data/acts_sq_smoke/ instead and
records only 8 parity trials per condition, the first 8 views of walks 0 and 1 in walk and shuffled, and 8 views
without the walk.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from types import SimpleNamespace

import numpy as np

from .two_choice import NOWALK_PROMPT, OUT as TC_OUT, QUESTION, WALK_PROMPT, default_tag, parse, present, walk_order

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DATA = "data/walks"
TRIALS = "results/export/single_query/trials.json"
PARITY_TRIALS = "results/export/dissimilar/trials.json"
OUT = "data/acts_sq"
SEED = 33
STEPS = 256
CONDS = ("walk", "shuffled", "nowalk")
SITES = ("img_end", "last")
KINDS = ("mlp", "resid")
LAYERS = ("L3", "L23", "L29", "L35")
MAX_NEW = 16
COS_MIN = 0.99
END = "<|vision_end|>"


class _Quiet(logging.Filter):
    """apply_chat_template warns on every call that enable_thinking / reasoning_effort are not template variables
    (two_choice.py passes them; they do not change the text)."""

    def filter(self, r):
        return "processor_kwargs" not in r.getMessage()


logging.getLogger("transformers.processing_utils").addFilter(_Quiet())


class _Off(SimpleNamespace):
    """two_choice's arguments; any option not set here (including ones two_choice gains later) is off."""

    def __getattr__(self, name):
        return None


def tc_args(cond, adapter):
    """two_choice's arguments for the stored room-A run of this condition."""
    return _Off(model=MODEL, adapter=adapter, trials="dissimilar", size=256, no_walk=cond == "nowalk", think=False,
                effort="medium", max_images=64, shuffle_walk=cond == "shuffled", relabel=False, device="cuda")


SHOWN = present({"a": 1, "b": STEPS, "walk": 0}, STEPS, tc_args("walk", None))[0]     # the 61 frames always shown


# ---------------------------------------------------------------- trials
def build_trials(data=DATA, seed=SEED):
    from .analyze_behave import angdiff, geometry
    from .dissimilar_trials import AWAY, FAR, NEAR, SIMILAR_TURN, landmarks
    from .room import get_layout
    meta, pmeta, G = geometry(data)
    lay = get_layout("A")
    pillars, posters = lay["pillars"], lay["posters"]()
    marks = {}

    def frame_marks(f):
        if f not in marks:
            marks[f] = landmarks(meta["x"][f], meta["y"][f], meta["theta"][f], pillars=pillars, posters=posters)
        return marks[f]

    rng = np.random.default_rng(seed)
    cand = np.asarray(SHOWN) - 1                                   # 0-based positions in the walk
    trials, sign = [], 1
    for p in range(len(pmeta["walk"])):
        w = int(pmeta["walk"][p])
        fi, d, sim, bm = G[p]["fi"], G[p]["d"], G[p]["sim"], G[p]["pix"]
        dh = angdiff(meta["theta"][fi], pmeta["theta"][p])
        seen_p = landmarks(pmeta["x"][p], pmeta["y"][p], pmeta["theta"][p], pillars=pillars, posters=posters)
        clean = np.array([dh[k] > AWAY and not (seen_p & frame_marks(fi[k])) if (d[k] < NEAR or d[k] > FAR) else False
                          for k in cand])
        near, far = cand[(d[cand] < NEAR) & clean], cand[(d[cand] > FAR) & clean]
        best, used = None, None
        for sgn in (sign, -sign):
            for i in near:
                ok = far[(np.abs(dh[far] - dh[i]) < SIMILAR_TURN) & (np.sign(sim[i] - sim[far]) == sgn)]
                if len(ok):
                    j = ok[np.argmin(np.abs(sim[i] - sim[ok]))]
                    if best is None or abs(sim[i] - sim[j]) < best[2]:
                        best = (i, j, abs(sim[i] - sim[j]))
            if best is not None:
                used = sgn
                break
        if best is not None:
            i, j, _ = best
            sign = -used
            valid = True
        else:                                                      # fallback: image-end activation only
            i = cand[np.argmin(d[cand])]
            fr = cand[d[cand] > FAR] if (d[cand] > FAR).any() else cand[[np.argmax(d[cand])]]
            j = fr[np.argmin(np.abs(dh[fr] - dh[i]))]
            valid = False
        nr, fr_ = int(i) + 1, int(j) + 1
        a, b = (nr, fr_) if rng.random() < 0.5 else (fr_, nr)
        trials.append({
            "id": f"w{w:02d}p{p:04d}", "walk": w, "probe": p, "familiar": G[p]["familiar"], "valid": valid,
            "near": nr, "far": fr_, "a": a, "b": b,
            "near_dist_m": round(float(d[i]), 3), "far_dist_m": round(float(d[j]), 3),
            "pixels_favour_near": bool(sim[i] > sim[j]), "sim_near": round(float(sim[i]), 4),
            "sim_far": round(float(sim[j]), 4), "near_heading_diff_deg": round(float(np.degrees(dh[i])), 1),
            "far_heading_diff_deg": round(float(np.degrees(dh[j])), 1), "probe_landmarks": sorted(seen_p),
            "best_match_frame": int(bm) + 1, "frame_order_favours_near": bool(abs(i - bm) < abs(j - bm))})
    return trials


def trials_report(T):
    nov = [t for t in T if not t["familiar"]]
    v = [t for t in T if t["valid"]]
    per_walk = np.bincount([t["walk"] for t in nov], minlength=48)
    return {"views": len(T), "novel": len(nov), "familiar": len(T) - len(nov), "valid": len(v),
            "valid_novel": sum(t["valid"] for t in nov), "valid_familiar": sum(t["valid"] and t["familiar"] for t in T),
            "pixels_favour_near_valid": float(np.mean([t["pixels_favour_near"] for t in v])),
            "frame_order_favours_near_valid": float(np.mean([t["frame_order_favours_near"] for t in v])),
            "a_is_near_valid": float(np.mean([t["a"] == t["near"] for t in v])),
            "novel_per_walk": {"min": int(per_walk.min()), "median": float(np.median(per_walk)),
                               "max": int(per_walk.max()), "walks_with_none": int((per_walk == 0).sum())},
            "near_far_dist_median_valid": [float(np.median([t["near_dist_m"] for t in v])),
                                           float(np.median([t["far_dist_m"] for t in v]))],
            "options_among_shown": all(t["a"] in SHOWN and t["b"] in SHOWN for t in T)}


def walkframe_rows(data=DATA):
    """The walk frames' rows in meta.npz, in walkframes order (walk * 61 + place among the 61 in time order)."""
    order = walk_order(dict(np.load(f"{data}/meta.npz")))
    return np.array([order[w][k - 1] for w in sorted(order) for k in SHOWN])


# ---------------------------------------------------------------- prompts
def processor():
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.image_processor.size = {"shortest_edge": 256 * 256, "longest_edge": 256 * 256}   # as two_choice --size 256
    return proc


def prompt(t, cond, adapter, proc):
    """two_choice's prompt for trial t in this condition: (text, shown, lab)."""
    shown, lab = present(t, STEPS, tc_args(cond, adapter))
    content = [{"type": "text", "text": NOWALK_PROMPT if cond == "nowalk" else WALK_PROMPT}]
    for k in shown:
        content += [{"type": "text", "text": f"Frame {lab[k]:03d}:"}, {"type": "image"}]
    content += [{"type": "text", "text": "New view:"}, {"type": "image"},
                {"type": "text", "text": QUESTION.format(a=lab[t["a"]], b=lab[t["b"]])}]
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False, reasoning_effort="medium")
    return text, shown, lab


def split(text):
    """(prefix, suffix) of a walk prompt's text: the prefix ends with the last walk frame's <|vision_end|>."""
    k = text.rfind(END, 0, text.rfind(END)) + len(END)
    return text[:k], text[k:]


def sites(ids, tok):
    """The view's <|vision_end|> (the last one), the last prompt token, and every earlier <|vision_end|> (the shown
    frames', in the order shown), in a 1-d id tensor."""
    ve = (ids == tok.convert_tokens_to_ids(END)).nonzero()[:, 0]
    vs = (ids == tok.convert_tokens_to_ids("<|vision_start|>")).nonzero()[:, 0]
    v0, v1 = int(vs[-1]) + 1, int(ve[-1])
    assert v1 - v0 == 64 and (ids[v0:v1] == tok.convert_tokens_to_ids("<|image_pad|>")).all(), (v0, v1)
    return {"img_end": v1, "last": len(ids) - 1, "n_images": len(ve), "frames": ve[:-1].tolist()}


def stored(cond, adapter):
    """The stored two_choice run of room A's 192 trials in this condition, by id (empty if there is none)."""
    path = f"{TC_OUT}/{default_tag(tc_args(cond, adapter))}.jsonl"
    return ({r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}), path


class Data:
    def __init__(self, data=DATA):
        from PIL import Image
        self.frames = np.load(f"{data}/frames.npy", mmap_mode="r")
        self.probes = np.load(f"{data}/probes.npy", mmap_mode="r")
        self.order = walk_order(dict(np.load(f"{data}/meta.npz")))
        self.img = lambda a: Image.fromarray(np.asarray(a))

    def walk_imgs(self, w, shown):
        return [self.img(self.frames[self.order[w][k - 1]]) for k in shown]

    def view(self, p):
        return self.img(self.probes[p])


def model_adapter(name):
    from .extract import FINETUNED, QWEN_IDS
    model_id, adapter = FINETUNED.get(name, (QWEN_IDS.get(name), None))
    assert model_id == MODEL, (name, model_id)
    return adapter


# ---------------------------------------------------------------- dry run (no model)
def dry_run(args):
    import torch
    proc = processor()
    tok = proc.tokenizer
    D = Data()
    T = json.load(open(TRIALS))
    adapter = model_adapter(args.model)
    t0 = time.time()
    for cond in ("walk", "shuffled"):
        prefix, n = {}, 0
        for t in T[:args.n]:
            text, shown, lab = prompt(t, cond, adapter, proc)
            assert shown == present({"a": SHOWN[0], "b": SHOWN[1], "walk": t["walk"]}, STEPS, tc_args(cond, adapter))[0]
            pre, suf = split(text)
            imgs = D.walk_imgs(t["walk"], shown)
            if t["walk"] not in prefix:
                prefix[t["walk"]] = (pre, proc(text=[pre], images=imgs, return_tensors="pt"))
            assert prefix[t["walk"]][0] == pre, "walk prefix differs between views"
            pin = prefix[t["walk"]][1]
            sin = proc(text=[suf], images=[D.view(t["probe"])], return_tensors="pt")
            ids = torch.cat([pin["input_ids"], sin["input_ids"]], 1)[0]
            mm = torch.cat([pin["mm_token_type_ids"], sin["mm_token_type_ids"]], 1)[0]
            full = proc(text=[text], images=imgs + [D.view(t["probe"])], return_tensors="pt")
            assert torch.equal(ids, full["input_ids"][0]) and torch.equal(mm, full["mm_token_type_ids"][0]), t["id"]
            assert torch.equal(torch.cat([pin["pixel_values"], sin["pixel_values"]]), full["pixel_values"])
            s = sites(ids, tok)
            P = pin["input_ids"].shape[1]
            pf = sites_prefix(pin["input_ids"][0], tok)
            assert pf == s["frames"] and len(pf) == len(SHOWN) and max(pf) == P - 1, "walk-frame sites"
            n += 1
            if n == 1:
                print(f"{cond} {t['id']} (walk {t['walk']}, options {t['a']} {t['b']}, valid {t['valid']}): "
                      f"{len(ids)} tokens = prefix {P} + suffix {len(ids) - P}, {s['n_images']} images; shown "
                      f"{' '.join(map(str, shown[:8]))} ...; img_end {s['img_end']} (suffix {s['img_end'] - P}), "
                      f"last {s['last']} (suffix {s['last'] - P}); walk-frame <|vision_end|> at "
                      f"{pf[:3]} ... {pf[-1]} ({len(pf)})")
                show = lambda a, b: tok.convert_ids_to_tokens(ids[a:b].tolist())
                print(f"  around img_end: {show(s['img_end'] - 2, s['img_end'] + 4)}")
                print(f"  around the first walk frame's end: {show(pf[0] - 2, pf[0] + 6)}")
                print(f"  prompt tail: {tok.decode(ids[s['img_end']:])!r}")
                print(f"  suffix head: {tok.decode(ids[P:P + 6])!r}")
        print(f"{cond}: {n} views ({len(prefix)} walks), prefix + suffix ids = whole-prompt ids in every one, walk "
              f"prefix the same for every view of a walk ({time.time() - t0:.0f}s)")
    for t in T[:3]:
        text, shown, lab = prompt(t, "nowalk", adapter, proc)
        full = proc(text=[text], images=D.walk_imgs(t["walk"], shown) + [D.view(t["probe"])], return_tensors="pt")
        s = sites(full["input_ids"][0], tok)
        print(f"nowalk {t['id']}: {full['input_ids'].shape[1]} tokens, shown {shown}, img_end {s['img_end']}, "
              f"last {s['last']}")
    PT = json.load(open(PARITY_TRIALS))
    PT = PT if args.parity_all else PT[:args.n]
    for cond in CONDS:
        ref, path = stored(cond, adapter)
        bad = checked = 0
        for t in PT:
            text, shown, lab = prompt(t, cond, adapter, proc)
            n = proc(text=[text], images=D.walk_imgs(t["walk"], shown) + [D.view(t["probe"])],
                     return_tensors="pt")["input_ids"].shape[1]
            if t["id"] in ref:
                checked += 1
                bad += ref[t["id"]]["prompt_tokens"] != n
        print(f"parity {cond}: {checked} of {len(PT)} prompts checked against {os.path.basename(path)}"
              f"{'' if ref else ' (no stored run)'}; {bad} token-count mismatches")
    wr = walkframe_rows()
    print(f"walkframes: {len(wr)} rows (48 walks x {len(SHOWN)} frames); first walk's frames {SHOWN[:5]} ... "
          f"-> meta rows {wr[:3].tolist()} ...")
    print(f"dry run done in {time.time() - t0:.0f}s")


def sites_prefix(ids, tok):
    return (ids == tok.convert_tokens_to_ids(END)).nonzero()[:, 0].tolist()


# ---------------------------------------------------------------- recording
class Hooks:
    """MLP neurons (the input to down_proj, as extract.py) and the residual stream (the decoder layer's output) of
    the kept layers at the positions in .at (a 1-d tensor), during one forward only."""

    def __init__(self, model, mlp, keep):
        self.at, self.acts = None, {}
        blocks = model.model.language_model.layers
        for n in keep:
            mlp[n].register_forward_pre_hook(self._mlp(n))
            blocks[int(n[1:])].register_forward_hook(self._resid(n))

    def _mlp(self, n):
        def hook(_, inp):
            if self.at is not None:
                self.acts["mlp", n] = inp[0][0, self.at].float().cpu().numpy()
        return hook

    def _resid(self, n):
        def hook(_, inp, out):
            if self.at is not None:
                h = out[0] if isinstance(out, tuple) else out
                self.acts["resid", n] = h[0, self.at].float().cpu().numpy()
        return hook


def record(args):
    import torch
    from .behave import Rewind
    from .extract import load_qwen

    adapter = model_adapter(args.model)
    proc, model, mlp = load_qwen(MODEL, False, adapter)
    tok = proc.tokenizer
    keep = list(args.layers)
    assert all(n in mlp for n in keep), (keep, list(mlp))
    hooks = Hooks(model, mlp, keep)
    units = {"mlp": mlp[keep[0]].in_features, "resid": model.config.text_config.hidden_size}
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    D = Data()
    T = json.load(open(TRIALS))
    PT = json.load(open(PARITY_TRIALS))
    out = f"{OUT}_smoke/{args.model}" if args.smoke else f"{OUT}/{args.model}"
    os.makedirs(out, exist_ok=True)
    meta = {"model": MODEL, "adapter": adapter, "layers": keep, "units": units, "kinds": list(KINDS),
            "trials": TRIALS, "parity_trials": PARITY_TRIALS, "views": [t["id"] for t in T],
            "parity_views": [t["id"] for t in PT], "walkframes_meta_rows": walkframe_rows().tolist(),
            "smoke": bool(args.smoke)}
    if os.path.exists(f"{out}/meta.json"):
        old = json.load(open(f"{out}/meta.json"))
        assert all(old[k] == meta[k] for k in ("adapter", "layers", "views", "parity_views")), "meta.json differs"
    json.dump(meta, open(f"{out}/meta.json", "w"))
    print(f"{args.model} ({adapter or 'base'}): recording {keep}, MLP ({units['mlp']}) and residual ({units['resid']}), "
          f"-> {out}{' (smoke)' if args.smoke else ''}", flush=True)

    def rope(ids, mm, grid):
        ext = torch.cat([ids, ids.new_zeros(1, MAX_NEW)], 1)
        mme = torch.cat([mm, mm.new_zeros(1, MAX_NEW)], 1)
        pos, _ = model.model.get_rope_index(ext, mm_token_type_ids=mme, image_grid_thw=grid)
        L = ids.shape[1]
        assert (pos[:, 0, L:] == pos[:, 0, L - 1].max() + 1 + torch.arange(MAX_NEW, device=pos.device)).all()
        return pos

    def forward(ids, pos, a, b, at, cache=None, **img):
        """One forward over ids[:, a:b] (continuing the cache if given), recording at the positions `at` (inside the
        slice); returns (activations by (kind, layer), first-token logits, cache)."""
        hooks.at, hooks.acts = torch.tensor(at, device="cuda"), {}
        o = model.model(input_ids=ids[:, a:b], position_ids=pos[:, :, a:b], past_key_values=cache, use_cache=True,
                        **img)
        hooks.at = None
        acts = dict(hooks.acts)
        assert len(acts) == 2 * len(keep), sorted(acts)
        return acts, model.lm_head(o.last_hidden_state[:, -1]).float(), o.past_key_values

    def decode(first, cache, pos, L):
        logits, toks = first, []
        for k in range(MAX_NEW):
            t = int(logits.argmax(-1))
            if t in eos:
                break
            toks.append(t)
            if k + 1 == MAX_NEW:
                break
            o = model.model(input_ids=torch.tensor([[t]], device="cuda"), position_ids=pos[:, :, L + k:L + k + 1],
                            past_key_values=cache, use_cache=True)
            logits = model.lm_head(o.last_hidden_state[:, -1]).float()
        return tok.decode(toks, skip_special_tokens=False)

    def answer(t, lab, reply):
        for sp in {tok.pad_token, tok.eos_token, "<|im_end|>", "<|endoftext|>"} - {None}:
            reply = reply.replace(sp, "")
        ans = {lab[t["a"]]: t["a"], lab[t["b"]]: t["b"]}.get(parse(reply, lab[t["a"]], lab[t["b"]]), -1)
        return ans, reply

    def arrays(tag, n, sites_=SITES):
        arr = {}
        for s in sites_:
            for kind in KINDS:
                shape = (n, len(keep), units[kind])
                f = f"{out}/{tag}_{s}_{kind}.npy"
                arr[s, kind] = np.lib.format.open_memmap(f, mode="r+" if os.path.exists(f) else "w+",
                                                         dtype=np.float16, shape=shape)
                assert arr[s, kind].shape == shape, (f, arr[s, kind].shape, shape)
        return arr

    def store(arr, site, rows, acts, sel):
        """Write the activations at positions sel (indices into the recorded positions) to rows of arr[site, *]."""
        bad = 0
        for kind in KINDS:
            A = np.stack([acts[kind, n][sel] for n in keep], 1)              # rows x layers x units
            if kind == "mlp":                                                 # extract.py: cast, non-finite -> 0
                A = A.astype(np.float16)
                nf = ~np.isfinite(A)
                A[nf] = 0
            else:                                                             # task_record.py: clip to float16
                nf = ~np.isfinite(A) | (np.abs(A) > 65504)
                A = np.clip(np.nan_to_num(A), -65504, 65504).astype(np.float16)
            bad += int(nf.sum())
            arr[site, kind][rows] = A
        return bad

    def flush(arr):
        for a in arr.values():
            a.flush()

    def full_inputs(t, cond):
        text, shown, lab = prompt(t, cond, adapter, proc)            # nowalk: shown is the two options
        inp = proc(text=[text], images=D.walk_imgs(t["walk"], shown) + [D.view(t["probe"])],
                   return_tensors="pt").to("cuda")
        return inp, shown, lab

    def run_full(inp):
        ids = inp["input_ids"]
        s = sites(ids[0], tok)
        pos = rope(ids, inp["mm_token_type_ids"], inp["image_grid_thw"])
        L = ids.shape[1]
        acts, first, cache = forward(ids, pos, 0, L, [s["img_end"], s["last"]], pixel_values=inp["pixel_values"],
                                     image_grid_thw=inp["image_grid_thw"])
        return acts, first, cache, pos, s, L

    def done_rows(path):
        return {json.loads(l)["row"] for l in open(path)} if os.path.exists(path) else set()

    def log(rec):
        with open(f"{out}/checks.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def cosines(a, b, sel_a, sel_b):
        """Smallest cosine over kinds, layers and positions between a[sel_a] and b[sel_b]."""
        c = []
        for kind in KINDS:
            for n in keep:
                x, y = torch.from_numpy(a[kind, n][sel_a]), torch.from_numpy(b[kind, n][sel_b])
                c.append(float(torch.nn.functional.cosine_similarity(x, y, dim=1).min()))
        return min(c)

    t_all = time.time()
    for cond in args.conds:
        if cond == "parity":
            PTc = PT[:8] if args.smoke else PT
            for c in CONDS:
                ref, path = stored(c, adapter)
                tag = f"parity_{c}"
                arr = arrays(tag, len(PT))
                done_path = f"{out}/done_{tag}.jsonl"
                done = done_rows(done_path)
                agree = n_ref = 0
                t0 = time.time()
                for row, t in enumerate(PTc):
                    if row in done:
                        continue
                    inp, shown, lab = full_inputs(t, c)
                    L = inp["input_ids"].shape[1]
                    if t["id"] in ref and ref[t["id"]]["prompt_tokens"] != L:
                        raise SystemExit(f"parity {c} {t['id']}: {L} prompt tokens, {os.path.basename(path)} has "
                                         f"{ref[t['id']]['prompt_tokens']}; the prompt differs")
                    with torch.no_grad():
                        acts, first, cache, pos, s, L = run_full(inp)
                        ans, reply = answer(t, lab, decode(first, cache, pos, L))
                    del cache
                    bad = store(arr, "img_end", [row], acts, [0]) + store(arr, "last", [row], acts, [1])
                    if t["id"] in ref:
                        n_ref += 1
                        agree += ans == ref[t["id"]]["answer"]
                    with open(done_path, "a") as fh:
                        fh.write(json.dumps({"row": row, "id": t["id"], "answer": ans, "correct": ans == t["near"],
                                             "stored_answer": ref.get(t["id"], {}).get("answer"), "prompt_tokens": L,
                                             "sites": {k: s[k] for k in SITES}, "text": reply[-200:],
                                             "nonfinite_or_clipped": bad}) + "\n")
                flush(arr)
                print(f"parity {c}: {n_ref} answers compared with {os.path.basename(path) if ref else 'no stored run'}, "
                      f"{agree} the same, {time.time() - t0:.0f}s", flush=True)
                if n_ref >= 20 and agree < 0.9 * n_ref:
                    raise SystemExit(f"parity {c}: only {agree} of {n_ref} answers match the stored run; stopping")
            continue

        arr = arrays(cond, len(T))
        done_path = f"{out}/done_{cond}.jsonl"
        done = done_rows(done_path)
        rows_all = [r for r in range(len(T)) if not args.smoke or (T[r]["walk"] < 2 and r % 64 < 8)]
        if cond == "nowalk" and args.smoke:
            rows_all = list(range(8))
        todo = [r for r in rows_all if r not in done]
        print(f"{cond}: {len(todo)} views to record ({len(done)} done)", flush=True)
        t0, n_checked = time.time(), 0
        if cond == "nowalk":
            for j, row in enumerate(todo):
                t = T[row]
                inp, shown, lab = full_inputs(t, cond)
                with torch.no_grad():
                    acts, first, cache, pos, s, L = run_full(inp)
                    ans, reply = answer(t, lab, decode(first, cache, pos, L))
                del cache
                bad = store(arr, "img_end", [row], acts, [0]) + store(arr, "last", [row], acts, [1])
                with open(done_path, "a") as fh:
                    fh.write(json.dumps({"row": row, "id": t["id"], "answer": ans, "correct": ans == t["near"],
                                         "prompt_tokens": L, "sites": {k: s[k] for k in SITES}, "cached": False,
                                         "text": reply[-200:], "nonfinite_or_clipped": bad}) + "\n")
                if j == 0 or (j + 1) % 512 == 0 or j + 1 == len(todo):
                    flush(arr)
                    print(f"  {j + 1}/{len(todo)}, {time.time() - t0:.0f}s, {t['id']} {L} tokens, answer {reply!r}",
                          flush=True)
            continue

        wf = arrays(cond, len(SHOWN) * 48, ("walkframes",))
        wf_path = f"{out}/done_walkframes_{cond}.jsonl"
        wf_done = {json.loads(l)["walk"] for l in open(wf_path)} if os.path.exists(wf_path) else set()
        walks = sorted({T[r]["walk"] for r in rows_all})
        walks_checked = 0
        for w in walks:
            rows = [r for r in todo if T[r]["walk"] == w]
            if not rows and w in wf_done:
                continue
            t_first = T[rows[0] if rows else 64 * w]
            text, shown, lab = prompt(t_first, cond, adapter, proc)
            pre, _ = split(text)
            imgs = D.walk_imgs(w, shown)
            pin = proc(text=[pre], images=imgs, return_tensors="pt").to("cuda")
            P = pin["input_ids"].shape[1]
            fpos = sites_prefix(pin["input_ids"][0], tok)                # the shown frames' <|vision_end|>, as shown
            assert len(fpos) == len(SHOWN) == len(shown) and fpos[-1] == P - 1, (len(fpos), fpos[-1], P)
            wrows = [w * len(SHOWN) + SHOWN.index(k) for k in shown]   # walkframes rows, in the order shown
            pos_prefix = rope(pin["input_ids"], pin["mm_token_type_ids"], pin["image_grid_thw"])[:, :, :P]
            with torch.no_grad():                                    # prefill: records the walk frames
                pacts, _, cache = forward(pin["input_ids"], pos_prefix, 0, P, fpos, pixel_values=pin["pixel_values"],
                                          image_grid_thw=pin["image_grid_thw"])
            if w not in wf_done:
                bad = store(wf, "walkframes", wrows, pacts, list(range(len(fpos))))
                flush(wf)
                with open(wf_path, "a") as fh:
                    fh.write(json.dumps({"walk": w, "prefix_tokens": P, "shown": shown, "rows": wrows,
                                         "nonfinite_or_clipped": bad}) + "\n")
            rewind = Rewind(cache)
            reply = ""
            for k, row in enumerate(rows):
                t = T[row]
                text, shown_t, lab = prompt(t, cond, adapter, proc)
                pre_t, suf = split(text)
                assert shown_t == shown and pre_t == pre, f"{t['id']}: walk context differs from its walk's"
                sin = proc(text=[suf], images=[D.view(t["probe"])], return_tensors="pt").to("cuda")
                ids = torch.cat([pin["input_ids"], sin["input_ids"]], 1)
                mm = torch.cat([pin["mm_token_type_ids"], sin["mm_token_type_ids"]], 1)
                grid = torch.cat([pin["image_grid_thw"], sin["image_grid_thw"]])
                if k == 0:                                           # the split tokenises like the whole prompt
                    full = proc(text=[text], images=imgs + [D.view(t["probe"])], return_tensors="pt")
                    assert torch.equal(ids.cpu(), full["input_ids"]) and torch.equal(mm.cpu(), full["mm_token_type_ids"]), \
                        f"{t['id']}: prefix + suffix ids differ from the whole prompt's"
                L = ids.shape[1]
                s = sites(ids[0], tok)
                assert s["frames"] == fpos, f"{t['id']}: walk-frame sites"
                pos = rope(ids, mm, grid)
                assert torch.equal(pos[:, :, :P], pos_prefix), f"{t['id']}: prefix positions differ"
                with torch.no_grad():
                    acts, first, cache = forward(ids, pos, P, L, [s["img_end"] - P, s["last"] - P], cache,
                                                 pixel_values=sin["pixel_values"], image_grid_thw=sin["image_grid_thw"])
                    ans, reply = answer(t, lab, decode(first, cache, pos, L))
                    rewind()
                    assert cache.get_seq_length() == P, (cache.get_seq_length(), P)
                    if walks_checked < 2 and k < 2:                  # against a full recompute without the cache
                        facts, ffirst, fcache = forward(ids, pos, 0, L, fpos + [s["img_end"], s["last"]], None,
                                                        pixel_values=torch.cat([pin["pixel_values"], sin["pixel_values"]]),
                                                        image_grid_thw=grid)
                        del fcache
                        nf = len(fpos)
                        c_view = cosines(acts, facts, [0, 1], [nf, nf + 1])
                        c_walk = cosines(pacts, facts, list(range(nf)), list(range(nf)))
                        dmax, scale = float((first - ffirst).abs().max()), float(ffirst.abs().max())
                        gap = float(ffirst.max() - ffirst[0, first.argmax()])
                        log({"cond": cond, "id": t["id"], "min_cos_view": c_view, "min_cos_walkframes": c_walk,
                             "max_dlogit": dmax, "logit_scale": scale, "gap": gap,
                             "argmax_same": int(first.argmax()) == int(ffirst.argmax())})
                        print(f"  check {t['id']}: min cosine {c_view:.5f} at the view's sites, {c_walk:.5f} at the "
                              f"{nf} walk frames (prefill without the view vs whole prompt), over {len(keep)} layers x "
                              f"2 kinds; max |dlogit| {dmax:.3f} (scale {scale:.1f}), gap {gap:.3f}", flush=True)
                        if min(c_view, c_walk) < COS_MIN or not (dmax < 0.05 * scale and gap <= 2 * dmax):
                            raise SystemExit(f"{cond} {t['id']}: cached activations or logits differ from a full "
                                             "recompute; stopping")
                        n_checked += 1
                bad = store(arr, "img_end", [row], acts, [0]) + store(arr, "last", [row], acts, [1])
                with open(done_path, "a") as fh:
                    fh.write(json.dumps({"row": row, "id": t["id"], "answer": ans, "correct": ans == t["near"],
                                         "prompt_tokens": L, "sites": {kk: s[kk] for kk in SITES}, "prefix_tokens": P,
                                         "cached": True, "text": reply[-200:], "nonfinite_or_clipped": bad}) + "\n")
            walks_checked += 1
            del cache, rewind, pacts
            torch.cuda.empty_cache()
            flush(arr)
            print(f"  walk {w}: {len(rows)} views, {len(done_rows(done_path))} done, {time.time() - t0:.0f}s, prefix "
                  f"{P} tokens, last answer {reply!r}; peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB",
                  flush=True)
        print(f"{cond}: done, {time.time() - t0:.0f}s ({n_checked} views checked against a full recompute)", flush=True)
    print(f"{args.model}: all done, {time.time() - t_all:.0f}s -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["trials", "dry-run", "record"])
    ap.add_argument("--model", default="qwen8b-ft-e4-s5", help="extract.py's name: qwen8b or qwen8b-ft-e4-s3 ... -s7")
    ap.add_argument("--conds", nargs="+", default=["parity", *CONDS], choices=["parity", *CONDS])
    ap.add_argument("--layers", nargs="+", default=list(LAYERS), help="extract.py's layer names to keep")
    ap.add_argument("--smoke", action="store_true", help="a few views per condition, into data/acts_sq_smoke/")
    ap.add_argument("-n", type=int, default=130, help="dry run: views (and parity trials) to tokenise")
    ap.add_argument("--parity-all", action="store_true", help="dry run: all 192 parity trials")
    args = ap.parse_args()
    if args.mode == "trials":
        T = build_trials()
        os.makedirs(os.path.dirname(TRIALS), exist_ok=True)
        json.dump(T, open(TRIALS, "w"), indent=1)
        print(json.dumps(trials_report(T), indent=1))
        print(f"{len(T)} trials -> {TRIALS}")
    elif args.mode == "dry-run":
        dry_run(args)
    else:
        record(args)


if __name__ == "__main__":
    main()
