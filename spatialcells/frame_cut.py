"""Experiment 15 (pre-registered in docs/experiments.md): which walk frames the fine-tuned model reads.

    python -m spatialcells.frame_cut --dino                                  # once, CPU, before any Qwen run
    python -m spatialcells.frame_cut --dry-run N [--condition near]          # sets, spans, token counts; no model
    python -m spatialcells.frame_cut --condition clean --device mps          # LoRA (seed 0) by default
    python -m spatialcells.frame_cut --check                                 # clean vs experiment 10: >= 99% or exit 1
    python -m spatialcells.frame_cut --condition near|lookalike|random|far --device mps
    python -m spatialcells.frame_cut --condition near --layers 1-18 --rooms P5000 P5001 P5002 --device mps
    python -m spatialcells.frame_cut --mask-check --device mps               # the mask itself, on a short prompt

The LoRA model (data/ft/qwen8b-proc, merged as task_record.py does; sdpa, bf16) on every novel trial of the
ten held-out rooms (1,299 in P5000-P5009), with experiment 10's exact 64-frame prompt (task_record's
build_prompt, every prompt's token count checked against experiment 10's run of the same trial), answered by
greedy generation and parsed exactly as two_choice.py does (its generation lines are repeated here, as
task_record repeats its prompt lines, because they are local to two_choice's run()).

Frame sets, 8 shown walk frames each, never an option frame (candidates: the shown frames minus the two
options, in the order shown):
  near       the 4 candidates just before and the 4 just after the near option in the order shown; at a walk
             end the missing ones come from the other side
  far        the same around the far option
  lookalike  the 8 candidates whose DINOv2-base CLS features (experiment 13d's dino(), from
             scripts/appearance_cue.py) are most similar (cosine) to the new view
  random     8 candidates drawn with random.Random(seed), seed from SHA-256 of "<room>/<trial id>"
--dino computes the cosines once (CPU) into data/frame_cut/dino_lookalike.json (per trial: the shown frames in
order and the new view's cosine to each); every run builds the sets from it.

The cut: for the set S, every token outside S's frames cannot attend to S's frames' tokens (each frame from
<|vision_start|> to <|vision_end|> inclusive, located from the token ids; the "Frame NNN:" labels stay
visible). Tokens of S's frames still attend as usual (to each other too). It is an additive -inf mask that
replaces the mask each decoder layer's self-attention gets (or is added to it, if the model built one), in
the prefill and in every cached generation step (one query: S's key columns blocked too); causal masking is
part of it, since with no padding the model otherwise passes no mask and relies on is_causal. --layers LO-HI
(1-based) cuts in those decoder layers only. Same tokens, positions and images; with an empty set the hooks
change nothing (clean runs with them installed).

One JSONL per condition, results/frame_cut/<model>+<adapter>-<condition>[-layersLO-HI]-<device>.jsonl, rows as
two_choice's plus room, condition, layers, the set (real frame numbers, shown order), its size, the tokens cut
and the per-trial overlaps (near with lookalike, near with far); trials already there are skipped.
--check compares the clean run with experiment 10's answers (all 1,299 trials needed; exit 1 below 99%).
--dry-run N prints, for the first trial, each set's frames with their labels and token spans (and the tokens at
the span edges), and checks N prompts' token counts against experiment 10. --mask-check, on a five-image
prompt: hooks with an empty set give bit-identical logits and generation to no hooks; in eager attention, with
a set cut, every layer gives exactly zero attention from outside S to S's tokens (prefill and decode) and
nonzero within S; --layers-style restriction leaves the other layers alone; the sdpa and eager cuts agree.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import time

import numpy as np

from .task_record import MODEL, ROOMS, build_prompt, sites, tc_args, trial_table
from .two_choice import QUESTION, WALK_PROMPT, parse, present, walk_order

OUT = "results/frame_cut"
DINO = "data/frame_cut/dino_lookalike.json"
ADAPTER = "data/ft/qwen8b-proc"
SETS = ("near", "far", "lookalike", "random")
CONDITIONS = ("clean",) + SETS
K, SIDE = 8, 4
MAX_NEW = 16              # two_choice's --max-new default, as in experiment 10's runs


def key(r):
    return f"{r['room']}/{r['id']}"


def novel_rows(rooms, adapter, device):
    """Novel trials in trial_table's order, with the far option."""
    rows = [r for r in trial_table(rooms, adapter, device) if not r["familiar"]]
    for r in rows:
        r["far"] = r["b"] if r["a"] == r["near"] else r["a"]
    return rows


def parse_layers(s):
    lo, hi = (int(x) for x in s.split("-"))
    assert 1 <= lo <= hi, s
    return lo, hi


def out_path(out, condition, layers, adapter, device):
    name = MODEL.split("/")[-1] + (f"+{os.path.basename(adapter.rstrip('/'))}" if adapter else "")
    return (f"{out}/{name}-{condition}" + (f"-layers{layers[0]}-{layers[1]}" if layers else "")
            + (f"-{device}" if device != "cuda" else "") + ".jsonl")


# ---------------------------------------------------------------- frame sets

def dino_step(rows):
    """DINOv2-base cosine of each novel trial's new view to every shown frame -> DINO (CPU)."""
    import torch
    from transformers import AutoModel
    spec = importlib.util.spec_from_file_location("appearance_cue", "scripts/appearance_cue.py")
    ac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    torch.set_num_threads(8)
    model = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    out = {"settings": {"model": "facebook/dinov2-base", "feature": "CLS token of last_hidden_state (appearance_cue.dino)",
                        "input": "256 px frame resized to 224 (bicubic, antialias), ImageNet mean/std", "device": "cpu",
                        "frames": "every frame experiment 10's 64-frame prompt shows (present()), in the order shown"},
           "trials": {}}
    t0 = time.time()
    for room in dict.fromkeys(r["room"] for r in rows):
        rr = [r for r in rows if r["room"] == room]
        d = f"data/walks-room{room}"
        frames, probes = np.load(f"{d}/frames.npy", mmap_mode="r"), np.load(f"{d}/probes.npy", mmap_mode="r")
        order = walk_order(dict(np.load(f"{d}/meta.npz")))
        shown = {key(r): present(r, len(order[r["walk"]]), tc_args(room, None, "cpu"))[0] for r in rr}
        fidx = sorted({int(order[r["walk"]][k - 1]) for r in rr for k in shown[key(r)]})
        pidx = sorted({r["probe"] for r in rr})
        emb = ac.dino(model, [np.asarray(frames[i]) for i in fidx] + [np.asarray(probes[i]) for i in pidx])
        fe, pe = dict(zip(fidx, emb[:len(fidx)])), dict(zip(pidx, emb[len(fidx):]))
        for r in rr:
            p = pe[r["probe"]]
            out["trials"][key(r)] = {"shown": shown[key(r)],
                                     "cos": [float(p @ fe[int(order[r["walk"]][k - 1])]) for k in shown[key(r)]]}
        print(f"{room}: {len(rr)} trials, {len(fidx)} walk frames + {len(pidx)} new views, {time.time() - t0:.0f}s",
              flush=True)
    os.makedirs(os.path.dirname(DINO), exist_ok=True)
    json.dump(out, open(DINO, "w"))
    sets = {key(r): frame_sets(r, out["trials"][key(r)]) for r in rows}
    for o in ("overlap_near_lookalike", "overlap_near_far"):
        v = np.array([s[o] for s in sets.values()])
        print(f"{o}: {int((v > 0).sum())} of {len(v)} trials overlap, mean {v.mean():.2f} frames, max {v.max()}")
    print(f"-> {DINO}")


def frame_sets(r, entry):
    """The four 8-frame sets of trial r (real frame numbers, in the order shown) and their overlaps."""
    shown, cos = entry["shown"], dict(zip(entry["shown"], entry["cos"]))
    pos = {k: i for i, k in enumerate(shown)}
    cand = [k for k in shown if k not in (r["a"], r["b"])]

    def around(o):
        before = [k for k in cand if pos[k] < pos[o]]
        after = [k for k in cand if pos[k] > pos[o]]
        nb, na = min(SIDE, len(before)), min(SIDE, len(after))
        if nb < SIDE:                                   # walk start: top up from after
            na = min(len(after), K - nb)
        elif na < SIDE:                                 # walk end: top up from before
            nb = min(len(before), K - na)
        return before[len(before) - nb:] + after[:na]

    seed = int(hashlib.sha256(key(r).encode()).hexdigest(), 16) % 2 ** 32
    s = {"near": around(r["near"]), "far": around(r["far"]),
         "lookalike": sorted(sorted(cand, key=lambda k: -cos[k])[:K], key=pos.get),
         "random": sorted(random.Random(seed).sample(cand, K), key=pos.get)}
    assert all(len(v) == K == len(set(v)) and not {r["a"], r["b"]} & set(v) for v in s.values()), (key(r), s)
    s["overlap_near_lookalike"] = len(set(s["near"]) & set(s["lookalike"]))
    s["overlap_near_far"] = len(set(s["near"]) & set(s["far"]))
    return s


def load_sets(rows):
    if not os.path.exists(DINO):
        raise SystemExit(f"{DINO} is missing: run python -m spatialcells.frame_cut --dino first (CPU)")
    dino = json.load(open(DINO))["trials"]
    missing = [key(r) for r in rows if key(r) not in dino]
    assert not missing, f"{DINO} lacks {len(missing)} trials, e.g. {missing[:3]}"
    return {key(r): {**frame_sets(r, dino[key(r)]), "shown": dino[key(r)]["shown"]} for r in rows}


# ---------------------------------------------------------------- token spans

def frame_spans(ids, tok, shown, lab):
    """Real frame number -> (first, last) token index of its image, <|vision_start|> .. <|vision_end|>
    inclusive; the k-th image is the k-th shown frame (the last is the new view). Checks each image's 64
    <|image_pad|> tokens and that the text just before it is its label "Frame NNN:"."""
    tid = tok.convert_tokens_to_ids
    vs = (ids == tid("<|vision_start|>")).nonzero()[:, 0].tolist()
    ve = (ids == tid("<|vision_end|>")).nonzero()[:, 0].tolist()
    assert len(vs) == len(ve) == len(shown) + 1, (len(vs), len(ve), len(shown))
    pad = tid("<|image_pad|>")
    spans, prev = {}, 0
    for j, k in enumerate(shown):
        s, e = vs[j], ve[j]
        assert e - s - 1 == 64 and (ids[s + 1:e] == pad).all(), (k, s, e)
        assert tok.decode(ids[prev:s]).endswith(f"Frame {lab[k]:03d}:"), (k, tok.decode(ids[max(prev, s - 8):s]))
        spans[k], prev = (s, e), e + 1
    return spans


def tokenise(r, proc, data, adapter, device):
    """Experiment 10's prompt for row r, tokenised, with the frame spans; stops if its token count differs."""
    text, images, shown, lab = build_prompt(r, proc, data, adapter, device)
    inputs = proc(text=[text], images=images, return_tensors="pt", padding=True)
    ids = inputs["input_ids"][0]
    assert sites(ids, proc.tokenizer)["n_images"] == len(images)
    want = r["exp10_prompt_tokens"]
    if want is None or want != len(ids):
        raise SystemExit(f"{key(r)}: {len(ids)} prompt tokens, experiment 10's run had {want}; the prompt differs")
    return inputs, shown, lab, frame_spans(ids, proc.tokenizer, shown, lab)


# ---------------------------------------------------------------- the attention cut

class AttentionCut:
    """While .spans holds (first, last) token ranges, no token outside them attends to a token inside them, in
    the hooked decoder layers (layers=(lo, hi), 1-based; default all). The mask is built at the start of every
    language-model forward (prefill and each cached decode step) for its query and key positions: causal, plus
    -inf from every query outside the spans to every key inside them. Each hooked self-attention gets it in
    place of the mask it was passed (None, with sdpa's is_causal) or added to it. With no spans, the hooks
    return without touching anything."""

    def __init__(self, model, layers=None):
        import torch
        self.torch = torch
        lm = model.model.language_model
        lo, hi = layers or (1, len(lm.layers))
        assert 1 <= lo <= hi <= len(lm.layers), layers
        self.spans, self.mask, self.kv = [], None, None
        self.handles = [lm.register_forward_pre_hook(self._begin, with_kwargs=True),
                        lm.register_forward_hook(self._end, with_kwargs=True)]
        self.handles += [lm.layers[i].self_attn.register_forward_pre_hook(self._attn, with_kwargs=True)
                         for i in range(lo - 1, hi)]

    def _begin(self, module, args, kwargs):
        self.mask = None
        if not self.spans:
            return
        torch = self.torch
        x = kwargs["inputs_embeds"]
        assert x.shape[0] == 1, "one trial at a time"
        cache = kwargs.get("past_key_values")
        past = cache.get_seq_length() if cache is not None else 0
        q, kv = x.shape[1], past + x.shape[1]
        cut = torch.zeros(kv, dtype=torch.bool)
        for s, e in self.spans:
            assert 0 <= s <= e < kv, (s, e, kv)
            cut[s:e + 1] = True
        qpos, kpos = torch.arange(past, kv), torch.arange(kv)
        blocked = (kpos[None] > qpos[:, None]) | (cut[None] & ~cut[qpos][:, None])
        self.mask = torch.zeros(q, kv, dtype=x.dtype).masked_fill(blocked, float("-inf"))[None, None].to(x.device)
        self.kv = kv

    def _end(self, module, args, kwargs, output):
        if self.mask is not None and getattr(output, "past_key_values", None) is not None:
            assert output.past_key_values.get_seq_length() == self.kv, (output.past_key_values.get_seq_length(), self.kv)
        self.mask = None

    def _attn(self, module, args, kwargs):
        if self.mask is None:
            return None
        m = kwargs.get("attention_mask")
        if m is None:
            new = self.mask
        else:
            assert m.shape[-2:] == self.mask.shape[-2:], (tuple(m.shape), tuple(self.mask.shape))
            new = self.mask.masked_fill(~m, float("-inf")) if m.dtype == self.torch.bool else m + self.mask
        return args, {**kwargs, "attention_mask": new}

    def remove(self):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------- model, runs, check

def load(adapter, device, attn="sdpa"):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.image_processor.size = {"shortest_edge": 256 * 256, "longest_edge": 256 * 256}   # as two_choice --size 256
    model = AutoModelForImageTextToText.from_pretrained(MODEL, attn_implementation=attn, dtype=torch.bfloat16,
                                                        device_map=device).eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
    return proc, model


def run(args, rows, sets):
    import torch
    path = out_path(args.out, args.condition, args.layers, args.adapter, args.device)
    os.makedirs(args.out, exist_ok=True)
    done = {(x["room"], x["id"]) for x in map(json.loads, open(path))} if os.path.exists(path) else set()
    todo = [r for r in rows if (r["room"], r["id"]) not in done]
    print(f"{os.path.basename(path)}: {len(todo)} trials to run ({len(done)} done)", flush=True)
    if not todo:
        return
    proc, model = load(args.adapter, args.device)
    cut = AttentionCut(model, args.layers)
    # two_choice's generation and parsing (its run(), --batch 1, greedy)
    proc.tokenizer.padding_side = "left"
    gen_kw = {"do_sample": False}
    special = {proc.tokenizer.pad_token, proc.tokenizer.eos_token, "<|im_end|>", "<|endoftext|>"} - {None}
    dev = torch.mps if args.device == "mps" else torch.cuda
    data = {}
    torch.manual_seed(0)
    t0 = time.time()
    for j, r in enumerate(todo):
        inputs, shown, lab, spans = tokenise(r, proc, data, args.adapter, args.device)
        st = sets[key(r)]
        assert shown == st["shown"], key(r)
        S = st[args.condition] if args.condition != "clean" else []
        cut.spans = [spans[k] for k in S]
        inputs = inputs.to(args.device)
        P = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, **gen_kw)
        cut.spans = []
        g = out[0][P:]
        reply = proc.decode(g, skip_special_tokens=False)
        n_gen = int((g != proc.tokenizer.pad_token_id).sum())
        for sp in special:
            reply = reply.replace(sp, "")
        ans = {lab[r["a"]]: r["a"], lab[r["b"]]: r["b"]}.get(parse(reply, lab[r["a"]], lab[r["b"]]), -1)
        rec = {"room": r["room"], **{k: r[k] for k in ("id", "walk", "probe", "familiar", "near", "far", "a", "b")}}
        rec.update(answer=ans, correct=ans == r["near"], prompt_tokens=P, gen_tokens=n_gen, forced=False,
                   text=reply[-600:], condition=args.condition,
                   layers=f"{args.layers[0]}-{args.layers[1]}" if args.layers else "all",
                   set=S, set_size=len(S), cut_tokens=sum(e - s + 1 for s, e in cut_spans(spans, S)),
                   overlap_near_lookalike=st["overlap_near_lookalike"], overlap_near_far=st["overlap_near_far"])
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if j < 3 or (j + 1) % 24 == 0 or j + 1 == len(todo):
            mem = (dev.driver_allocated_memory() if args.device == "mps" else dev.max_memory_allocated()) / 2 ** 30
            el = time.time() - t0
            print(f"{j + 1}/{len(todo)}, {el:.0f}s ({el / (j + 1):.1f} s/trial), {key(r)}, prompt {P} tokens, "
                  f"cut {rec['cut_tokens']} tokens, answer {ans} (experiment 10: {r['answer']}), "
                  f"{'driver' if args.device == 'mps' else 'peak'} {mem:.1f} GiB, e.g. {reply[-40:]!r}", flush=True)
    cut.remove()
    print(f"{os.path.basename(path)}: done, {time.time() - t0:.0f}s -> {path}", flush=True)


def cut_spans(spans, S):
    return [spans[k] for k in S]


def check(args, rows):
    """The registered check: the clean run gives experiment 10's answer on >= 99% of the novel trials."""
    path = out_path(args.out, "clean", None, args.adapter, args.device)
    got = {(x["room"], x["id"]): x for x in map(json.loads, open(path))} if os.path.exists(path) else {}
    missing = [key(r) for r in rows if (r["room"], r["id"]) not in got]
    if missing:
        print(f"CHECK FAILED: {path} lacks {len(missing)} of {len(rows)} trials, e.g. {missing[:3]}")
        raise SystemExit(1)
    bad = [(key(r), r["answer"], got[r["room"], r["id"]]["answer"]) for r in rows
           if got[r["room"], r["id"]]["answer"] != r["answer"]]
    agree = 1 - len(bad) / len(rows)
    acc = np.mean([got[r["room"], r["id"]]["correct"] for r in rows])
    acc10 = np.mean([r["correct"] for r in rows])
    print(f"clean vs experiment 10: {len(rows) - len(bad)}/{len(rows)} answers agree ({agree:.2%}); "
          f"accuracy {acc:.1%} vs {acc10:.1%}")
    for b in bad[:20]:
        print(f"  {b[0]}: experiment 10 {b[1]}, clean {b[2]}")
    if agree < 0.99:
        print("CHECK FAILED: below 99%; no cut condition may run")
        raise SystemExit(1)
    print("CHECK PASSED")


def dry_run(args, rows, sets):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.image_processor.size = {"shortest_edge": 256 * 256, "longest_edge": 256 * 256}
    tok = proc.tokenizer
    data, t0 = {}, time.time()
    n = min(args.dry_run, len(rows))
    for i in range(n):
        r = rows[i]
        inputs, shown, lab, spans = tokenise(r, proc, data, args.adapter, args.device)
        st = sets[key(r)]
        assert shown == st["shown"]
        if i == 0:
            ids = inputs["input_ids"][0]
            dec = lambda a, b: tok.convert_ids_to_tokens(ids[a:b].tolist())
            print(f"{key(r)}: {len(ids)} tokens (experiment 10: {r['exp10_prompt_tokens']}), {len(shown)} walk frames, "
                  f"options a={r['a']} b={r['b']}, near {r['near']} (shown position {shown.index(r['near']) + 1}), "
                  f"far {r['far']} (position {shown.index(r['far']) + 1}); overlaps near&lookalike "
                  f"{st['overlap_near_lookalike']}, near&far {st['overlap_near_far']}")
            for c in SETS:
                print(f"  {c}: frames {st[c]} (shown positions {[shown.index(k) + 1 for k in st[c]]}), "
                      f"{sum(e - s + 1 for s, e in cut_spans(spans, st[c]))} tokens cut")
                for k in st[c]:
                    s, e = spans[k]
                    print(f"    Frame {lab[k]:03d}: tokens {s}..{e}  label {tok.decode(ids[s - 6:s])!r}  "
                          f"{dec(s - 1, s + 2)} ... {dec(e - 1, e + 2)}")
            print("  prompt tail: " + repr(tok.decode(ids[-60:])))
    print(f"{n} prompts tokenised in {time.time() - t0:.0f}s; all {n} matched experiment 10's token count and "
          f"every shown frame's label and 64 visual tokens")


def mask_check(args, rows, sets):
    """The cut on a short prompt (four walk frames of the first trial, the new view, the question)."""
    import torch
    from PIL import Image
    proc, model = load(args.adapter, args.device)
    tok = proc.tokenizer
    r = rows[0]
    d = f"data/walks-room{r['room']}"
    frames, probes = np.load(f"{d}/frames.npy", mmap_mode="r"), np.load(f"{d}/probes.npy", mmap_mode="r")
    fi = walk_order(dict(np.load(f"{d}/meta.npz")))[r["walk"]]
    shown = sets[key(r)]["shown"][:4]
    lab = {k: k for k in shown}
    content = [{"type": "text", "text": WALK_PROMPT}]
    for k in shown:
        content += [{"type": "text", "text": f"Frame {k:03d}:"}, {"type": "image"}]
    content += [{"type": "text", "text": "New view:"}, {"type": "image"},
                {"type": "text", "text": QUESTION.format(a=shown[0], b=shown[3])}]
    images = [Image.fromarray(np.asarray(frames[fi[k - 1]])) for k in shown] + \
             [Image.fromarray(np.asarray(probes[r["probe"]]))]
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False, reasoning_effort="medium")
    inputs = proc(text=[text], images=images, return_tensors="pt", padding=True).to(args.device)
    ids = inputs["input_ids"][0].cpu()
    spans = frame_spans(ids, tok, shown, lab)
    S = [spans[shown[1]], spans[shown[2]]]                 # cut frames 2 and 3 of 4
    n = len(ids)
    cut = torch.zeros(n, dtype=torch.bool)
    for s, e in S:
        cut[s:e + 1] = True
    print(f"{n} tokens; cut spans {S} ({int(cut.sum())} tokens)")

    def logits():
        with torch.no_grad():
            return model(**inputs).logits[0].float().cpu()

    def gen():
        with torch.no_grad():
            return model.generate(**inputs, max_new_tokens=6, do_sample=False)[0, n:].cpu()

    ok = True
    l0, g0 = logits(), gen()
    c = AttentionCut(model)
    l1, g1 = logits(), gen()                               # hooks installed, empty set
    same = torch.equal(l0, l1) and torch.equal(g0, g1)
    ok &= same
    print(f"empty set vs no hooks (sdpa): logits bit-identical and same generation: {same}")
    c.spans = S
    l2, g2 = logits(), gen()
    print(f"cut (sdpa): max |logit change| at the last token {float((l2[-1] - l0[-1]).abs().max()):.3f}; "
          f"generation {tok.decode(g0)!r} -> {tok.decode(g2)!r}")
    c.remove()

    model.set_attn_implementation("eager")
    W = {}
    hs = [blk.self_attn.register_forward_hook(lambda m, a, o, i=i: W.setdefault(i, []).append(o[1][0].float().cpu()))
          for i, blk in enumerate(model.model.language_model.layers)]
    for layers in (None, (19, 36)):
        c = AttentionCut(model, layers)
        c.spans = S
        W.clear()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=3, do_sample=False, return_dict_in_generate=True,
                                 output_logits=True)
        c.remove()
        lo, hi = layers or (1, len(W))
        res = []
        for i in sorted(W):
            pre = W[i][0]                                  # heads x n x n (prefill)
            dec = torch.stack([w[:, -1, :n] for w in W[i][1:]], 1)    # decode steps: heads x steps x prompt keys
            out_rows = ~cut
            leak = max(float(pre[:, out_rows][:, :, cut].max()), float(dec[:, :, cut].max()))
            inside = float(pre[:, cut][:, :, cut].sum(-1).mean())
            res.append((i + 1, leak, inside))
        hooked = [x for x in res if lo <= x[0] <= hi]
        free = [x for x in res if not lo <= x[0] <= hi]
        zero = all(x[1] == 0 for x in hooked) and all(x[2] > 0 for x in hooked)
        ok &= zero and all(x[1] > 0 for x in free)
        print(f"eager, cut in layers {lo}-{hi}: max attention from outside S to S over hooked layers "
              f"{max(x[1] for x in hooked):.3g} (prefill and {len(W[0]) - 1} decode steps), mean attention mass of S "
              f"tokens on S {np.mean([x[2] for x in hooked]):.3f}"
              + (f"; unhooked layers {free[0][0]}-{free[-1][0]} keep attention to S (min of max "
                 f"{min(x[1] for x in free):.3g})" if free else ""))
        if layers is None:
            eg = out.sequences[0, n:].cpu()
            agree = tok.decode(eg[:3]) == tok.decode(g2[:3])
            print(f"eager cut vs sdpa cut: generation {tok.decode(eg)!r} vs {tok.decode(g2[:3])!r}, "
                  f"max |logit diff| first step {float((out.logits[0][0].float().cpu() - l2[-1]).abs().max()):.3f}")
            ok &= agree
    for h in hs:
        h.remove()
    model.set_attn_implementation("sdpa")
    print("MASK CHECK PASSED" if ok else "MASK CHECK FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CONDITIONS, default="clean")
    ap.add_argument("--layers", type=parse_layers, help="cut only in decoder layers LO-HI (1-based), e.g. 1-18")
    ap.add_argument("--rooms", nargs="+", default=ROOMS)
    ap.add_argument("--adapter", default=ADAPTER, help="LoRA adapter folder to merge in (default seed 0)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, help="only the first N trials (smoke tests)")
    ap.add_argument("--dino", action="store_true", help="compute the DINOv2 look-alike cosines (CPU) and stop")
    ap.add_argument("--check", action="store_true", help="clean vs experiment 10 (exit 1 below 99%%)")
    ap.add_argument("--dry-run", type=int, metavar="N", help="sets, spans and token counts of N prompts; no model")
    ap.add_argument("--mask-check", action="store_true", help="check the mask on a short prompt")
    args = ap.parse_args()
    if args.layers and args.condition == "clean":
        ap.error("--layers needs a cut condition")
    rows = novel_rows(args.rooms, args.adapter, args.device)
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} novel trials in {len(args.rooms)} rooms", flush=True)
    if args.dino:
        return dino_step(rows)
    if args.check:
        return check(args, rows)
    sets = load_sets(rows)
    if args.dry_run:
        return dry_run(args, rows, sets)
    if args.mask_check:
        return mask_check(args, rows, sets)
    run(args, rows, sets)


if __name__ == "__main__":
    main()
