"""Behavioural test: after watching the walk, can the model say where a new view was taken?

    python -m spatialcells.behave --model qwen32b
    python -m spatialcells.behave --model qwen --device cpu --walks 1 --max-frames 24 --probes 3   # smoke test

The walk's frames are shown in order, each labelled "Frame k:". Then one teleport
probe (a visited place, random heading) and the question: which numbered frame was
taken closest to where this new view was taken? The answer is greedy-decoded and
parsed as a frame number. Each walk is prefilled once; every probe continues from
that cache, which is rewound before the next probe, so probes never see each other.
The first walk's first two probes are also run without the cache as a check.
Writes results/behave/<model>.npz (probe index, answer frame or -1, raw text).
"""
from __future__ import annotations

import argparse
import os
import re
import time

import numpy as np
import torch

from .extract import QWEN_IDS

WALK_PROMPT = ("These are consecutive first-person views as you walk around a room. "
               "Each view is labelled with its frame number.")
QUESTION = ("This new view was taken somewhere in the same room, possibly facing a different direction. "
            "Which numbered frame above was taken from the position closest to where this new view was taken? "
            "Answer with the frame number only.")
MAX_NEW = 8


def load(model_id, device):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    proc.image_processor.size = {"shortest_edge": 256 * 256, "longest_edge": 256 * 256}
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16 if device == "cuda" else torch.float32, device_map=device,
        attn_implementation="sdpa")
    return proc, model.eval()


def chat_text(proc, n_walk):
    content = [{"type": "text", "text": WALK_PROMPT}]
    for k in range(1, n_walk + 1):
        content += [{"type": "text", "text": f"Frame {k}:"}, {"type": "image"}]
    content += [{"type": "text", "text": "New view:"}, {"type": "image"}, {"type": "text", "text": QUESTION}]
    return proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)


class Rewind:
    """Snapshot of a cache after the walk prefill; calling it restores that state (batch 1).
    Attention K/V grow by concatenation, so keeping references is enough; linear-attention
    conv/recurrent states are updated in place, so those are cloned, and restored as fresh
    clones so the cache can then be expanded to a batch."""

    def __init__(self, cache):
        self.cache, self.saved = cache, []
        for layer in cache.layers:
            s = {}
            if torch.is_tensor(getattr(layer, "keys", None)):
                s["kv"] = (layer.keys, layer.values)
            for attr in ("conv_states", "recurrent_states"):
                d = getattr(layer, attr, None)
                if isinstance(d, dict):
                    s[attr] = {i: t.clone() for i, t in d.items() if torch.is_tensor(t)}
            self.saved.append(s)

    def __call__(self):
        for layer, s in zip(self.cache.layers, self.saved):
            if "kv" in s:
                layer.keys, layer.values = s["kv"]
            for attr in ("conv_states", "recurrent_states"):
                for i, t in s.get(attr, {}).items():
                    getattr(layer, attr)[i] = t.clone()

    def expand(self, n):
        """Repeat the (just restored) cache n times along the batch."""
        if n == 1:
            return
        for layer in self.cache.layers:
            if torch.is_tensor(getattr(layer, "keys", None)):
                layer.keys, layer.values = layer.keys.repeat_interleave(n, 0), layer.values.repeat_interleave(n, 0)
            for attr in ("conv_states", "recurrent_states"):
                d = getattr(layer, attr, None)
                if isinstance(d, dict):
                    for i, t in d.items():
                        if torch.is_tensor(t):
                            d[i] = t.repeat_interleave(n, 0)


@torch.no_grad()
def main():
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(QWEN_IDS), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", default="data/walks")
    ap.add_argument("--walks", type=int, help="only the first N walks")
    ap.add_argument("--max-frames", type=int, help="truncate each walk (smoke test)")
    ap.add_argument("--probes", type=int, help="only the first N probes per walk")
    ap.add_argument("--out", default="results/behave")
    args = ap.parse_args()
    dev = args.device

    frames = np.load(f"{args.data}/frames.npy", mmap_mode="r")
    meta = dict(np.load(f"{args.data}/meta.npz"))
    probes = np.load(f"{args.data}/probes.npy", mmap_mode="r")
    pmeta = dict(np.load(f"{args.data}/probes_meta.npz"))
    proc, model = load(QWEN_IDS[args.model], dev)
    tok = proc.tokenizer
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id, *tok.encode("\n")}
    end_id = tok.convert_tokens_to_ids("<|vision_end|>")
    img = lambda a: Image.fromarray(np.asarray(a))

    walks = np.unique(meta["walk"])[:args.walks]
    rec = {"probe": [], "answer": [], "text": []}
    t0 = time.time()
    for wi, w in enumerate(walks):
        idx = np.flatnonzero(meta["walk"] == w)
        idx = idx[np.argsort(meta["step"][idx])][:args.max_frames]
        pidx = np.flatnonzero(pmeta["walk"] == w)[:args.probes]
        n = len(idx)
        text = chat_text(proc, n)
        inp = proc(text=[text], images=[img(frames[i]) for i in idx] + [img(probes[pidx[0]])],
                   return_tensors="pt").to(dev)
        grid = inp["image_grid_thw"]
        assert (grid == torch.tensor([1, 16, 16], device=dev)).all()
        ids, ppv = inp["input_ids"], len(inp["pixel_values"]) // len(grid)   # pixel rows per image
        P = int(torch.nonzero(ids[0] == end_id)[n - 1]) + 1                 # walk prefix ends after frame n
        L = ids.shape[1]
        # M-RoPE positions for prompt + MAX_NEW answer tokens (text tokens just continue)
        ext = torch.cat([ids, ids.new_zeros(1, MAX_NEW)], 1)
        mm = torch.cat([inp["mm_token_type_ids"], inp["mm_token_type_ids"].new_zeros(1, MAX_NEW)], 1)
        pos, _ = model.model.get_rope_index(ext, mm_token_type_ids=mm, image_grid_thw=grid)
        if wi == 0:
            print("prompt tail:", repr(tok.decode(ids[0, -40:])), flush=True)
            last = pos[:, 0, L - 1]
            assert (pos[:, 0, L:] == last.max() + 1 + torch.arange(MAX_NEW, device=dev)).all(), "answer positions"

        out = model.model(input_ids=ids[:, :P], pixel_values=inp["pixel_values"][:n * ppv],
                          image_grid_thw=grid[:n], position_ids=pos[:, :, :P], use_cache=True)
        cache = out.past_key_values
        rewind = Rewind(cache)

        def ask(p):
            pv = proc.image_processor(images=[img(probes[p])], return_tensors="pt")
            o = model.model(input_ids=ids[:, P:], pixel_values=pv["pixel_values"].to(dev),
                            image_grid_thw=pv["image_grid_thw"].to(dev), position_ids=pos[:, :, P:L],
                            past_key_values=cache, use_cache=True)
            first = model.lm_head(o.last_hidden_state[:, -1]).float()
            logits, toks = first, []
            for k in range(MAX_NEW):
                t = int(logits.argmax(-1))
                if t in stop:
                    break
                toks.append(t)
                o = model.model(input_ids=ids.new_tensor([[t]]), position_ids=pos[:, :, L + k:L + k + 1],
                                past_key_values=cache, use_cache=True)
                logits = model.lm_head(o.last_hidden_state[:, -1]).float()
            rewind()
            return first, tok.decode(toks)

        for j, p in enumerate(pidx):
            first, ans = ask(p)
            if wi == 0 and j < 2:   # same question without the cache
                full = proc(text=[text], images=[img(frames[i]) for i in idx] + [img(probes[p])],
                            return_tensors="pt").to(dev)
                ref = model.lm_head(model.model(**full, position_ids=pos[:, :, :L], use_cache=False)
                                    .last_hidden_state[:, -1]).float()
                dmax, scale = float((first - ref).abs().max()), float(ref.abs().max())
                gap = float(ref.max() - ref[0, first.argmax()])   # how far the cached pick is from the full one's best
                print(f"check probe {j}: cached argmax {int(first.argmax())} vs full {int(ref.argmax())}, "
                      f"max |dlogit| {dmax:.3f} (logit scale {scale:.1f}), gap {gap:.3f}", flush=True)
                # bf16 noise may flip a near-tie, but nothing more
                assert dmax < 0.05 * scale and gap <= 2 * dmax, "cached logits differ from full recompute"
            m = re.search(r"\d+", ans)
            a = int(m.group()) if m else -1
            rec["probe"].append(p)
            rec["answer"].append(a if 1 <= a <= n else -1)
            rec["text"].append(ans)
        del cache, rewind, out
        tail = rec["text"][-len(pidx):]
        print(f"walk {w}: {len(pidx)} probes, {time.time() - t0:.0f}s, answers e.g. {tail[:6]}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    tag = args.model + ("-smoke" if args.max_frames or args.probes else "")
    np.savez(f"{args.out}/{tag}.npz", probe=np.array(rec["probe"]), answer=np.array(rec["answer"]),
             text=np.array(rec["text"]), n_frames=n)
    print(f"behave: {len(rec['probe'])} probes, {np.mean(np.array(rec['answer']) > 0):.0%} parsed, "
          f"{time.time() - t0:.0f}s -> {args.out}/{tag}.npz", flush=True)


if __name__ == "__main__":
    main()
