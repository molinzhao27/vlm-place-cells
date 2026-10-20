"""Two-choice test for any Hugging Face vision-language model, at any frame size, with or without the walk.

    python -m spatialcells.two_choice --model Qwen/Qwen3-VL-8B-Instruct [--trials dissimilar]
        [--size 256] [--no-walk] [--think --max-new 4096] [--walks N] [--tag NAME] [--adapter DIR]
    python -m spatialcells.two_choice --score

The prompt is the Claude test's (claude_test.py): the walk's 256 frames labelled "Frame 001:"
... "Frame 256:", then "New view:", the probe and the question; with --no-walk, only the two
option frames. Trials (--trials):
  original     results/export/claude/trials.json: the 104 novel trials every model got, plus
               3 familiar per walk
  dissimilar   results/export/dissimilar/trials.json (dissimilar_trials.py): neither option
               faces like the probe or shows any of its landmarks, and pixel similarity favours
               each option equally often, so view matching scores 50%
  roomB        results/export/dissimilar-roomB/trials.json (dissimilar_trials.py --room B): the
               same rules in room B's walks (data/walks-roomB), with frame order balanced too
--size resizes every image to that many pixels square (frames are rendered at 256). For
Qwen-family processors it also sets the image size, so a frame is (size / 32)^2 tokens; Gemma
gets --image-tokens soft tokens per image (default 70, closest to Qwen's 64).
--max-images shows an evenly spaced subset of the walk (always with both options, labelled
with their real frame numbers) for models that take fewer images. Greedy decoding, or with --think the model's own sampling, thinking capped at --max-new tokens
(then closed with "</think>"); --batch generates that many trials together. Appends to results/two_choice/<tag>.jsonl and skips trials already
answered; --score summarises every file there into results/two_choice/summary.json.
--adapter merges a LoRA adapter (finetune.py) into the model first; its folder name joins the tag.
--shuffle-walk shows the walk's frames in a fixed random order (seeded by the walk) with their real
frame numbers; adding --relabel numbers them 1, 2, ... as shown and names the options by those
labels (as claude_test.py; answers are stored in real frame numbers).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import time

import numpy as np

OUT = "results/two_choice"
TRIALS = {"original": "results/export/claude/trials.json", "dissimilar": "results/export/dissimilar/trials.json",
          "roomB": "results/export/dissimilar-roomB/trials.json"}
DATA = {"roomB": "data/walks-roomB"}        # walks the trials come from (default data/walks)
WALK_PROMPT = ("These are consecutive first-person views as you walk around a room. "
               "Each view is labelled with its frame number.")
NOWALK_PROMPT = "These are two first-person views from a walk around a room, labelled with their frame numbers."
QUESTION = ("This new view was taken somewhere in the same room, possibly facing a different direction. "
            "Was it taken closer to where Frame {a:03d} was taken, or to where Frame {b:03d} was taken? "
            "Answer with one of the two frame numbers only.")


def default_tag(args):
    name = args.model.split("/")[-1] + (f"+{os.path.basename(args.adapter.rstrip('/'))}" if args.adapter else "")
    return (f"{name}-{args.trials}-{args.size}px" + ("-nowalk" if args.no_walk else "")
            + (f"-think-{args.effort}" if args.think else "")
            + (f"-max{args.max_images}" if args.max_images and not args.no_walk else "")
            + ("-shuffled" if args.shuffle_walk else "") + ("-relabel" if args.relabel else ""))


def parse(text, a, b):
    """The last of the two frame numbers mentioned after any thinking, else -1."""
    text = text.split("</think>")[-1]
    hits = [int(m) for m in re.findall(r"\d+", text) if int(m) in (a, b)]
    return hits[-1] if hits else -1


def run(args):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args.data = args.data or DATA.get(args.trials, "data/walks")
    frames = np.load(f"{args.data}/frames.npy", mmap_mode="r")
    probes = np.load(f"{args.data}/probes.npy", mmap_mode="r")
    meta = dict(np.load(f"{args.data}/meta.npz"))
    trials = json.load(open(TRIALS[args.trials]))
    if args.no_walk:
        trials = [t for t in trials if not t["familiar"]] if args.trials == "original" else trials
    if args.walks:
        trials = [t for t in trials if t["walk"] < args.walks]
    tag = args.tag or default_tag(args)
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{tag}.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    trials = [t for t in trials if t["id"] not in done]
    print(f"{tag}: {len(trials)} trials to run ({len(done)} done)", flush=True)
    if not trials:
        return

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    ip = proc.image_processor
    if "qwen" in args.model.lower():                 # (size / 32)^2 tokens per image
        ip.size = {"shortest_edge": args.size * args.size, "longest_edge": args.size * args.size}
    elif "gemma" in args.model.lower():              # soft tokens per image: 70 is closest to Qwen's 64
        ip.max_soft_tokens = ip.image_seq_length = proc.image_seq_length = args.image_tokens
    kw = dict(dtype=torch.bfloat16, device_map="cuda", trust_remote_code=args.trust_remote_code)
    if "Hy-Embodied" in args.model:
        # its modeling code reads moe_intermediate_size, which its config.json calls expert_hidden_dim
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        for c in (cfg, getattr(cfg, "text_config", None)):
            if c is not None and getattr(c, "moe_intermediate_size", None) is None:
                c.moe_intermediate_size = getattr(c, "expert_hidden_dim", None) or cfg.expert_hidden_dim
        if not hasattr(cfg, "enable_lm_head_fp32"):    # also read but absent from config.json
            cfg.enable_lm_head_fp32 = False
        kw["config"] = cfg
    try:
        model = AutoModelForImageTextToText.from_pretrained(args.model, attn_implementation="sdpa", **kw).eval()
    except (ValueError, TypeError) as e:                       # some remote-code models reject sdpa
        print(f"sdpa not accepted ({str(e)[:120]}); loading with the model's default attention", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(args.model, **kw).eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()

    def img(a):
        im = Image.fromarray(np.asarray(a))
        return im if args.size == 256 else im.resize((args.size, args.size), Image.BICUBIC)

    order = {}
    for w in np.unique(meta["walk"]):
        i = np.flatnonzero(meta["walk"] == w)
        order[int(w)] = i[np.argsort(meta["step"][i])]

    walk_imgs = {}
    labels = {}                                     # trial id -> real frame number -> label shown

    def build(t):
        fi = order[t["walk"]]
        if args.no_walk:
            content = [{"type": "text", "text": NOWALK_PROMPT}]
            shown = [t["a"], t["b"]]
        else:
            content = [{"type": "text", "text": WALK_PROMPT}]
            shown = list(range(1, len(fi) + 1))
            if args.max_images and len(shown) >= args.max_images:     # evenly spaced, always with a and b
                keep = np.linspace(1, len(fi), args.max_images - 3).round().astype(int)
                shown = sorted(set(keep.tolist()) | {t["a"], t["b"]})
        if args.shuffle_walk and not args.no_walk:
            random.Random(t["walk"]).shuffle(shown)
        lab = {k: (i + 1 if args.relabel else k) for i, k in enumerate(shown)}
        labels[t["id"]] = lab
        if t["walk"] not in walk_imgs:
            walk_imgs.clear()
            walk_imgs[t["walk"]] = {}
        cache = walk_imgs[t["walk"]]
        images = []
        for k in shown:
            if k not in cache:
                cache[k] = img(frames[fi[k - 1]])
            content += [{"type": "text", "text": f"Frame {lab[k]:03d}:"}, {"type": "image"}]
            images.append(cache[k])
        content += [{"type": "text", "text": "New view:"}, {"type": "image"},
                    {"type": "text", "text": QUESTION.format(a=lab[t["a"]], b=lab[t["b"]])}]
        images.append(img(probes[t["probe"]]))
        text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=args.think,
                                        reasoning_effort=args.effort)
        return text, images

    # Thinking samples with the model's own settings; greedy otherwise. Thinking still open after
    # --max-new tokens is closed with "</think>" and the answer generated after it.
    proc.tokenizer.padding_side = "left"
    gen_kw = {} if args.think else {"do_sample": False}
    special = {proc.tokenizer.pad_token, proc.tokenizer.eos_token, "<|im_end|>", "<|endoftext|>"} - {None}
    torch.manual_seed(0)
    t0 = time.time()
    for b0 in range(0, len(trials), args.batch):
        tb = trials[b0:b0 + args.batch]
        built = [build(t) for t in tb]
        inputs = proc(text=[x for x, _ in built], images=[im for _, ims in built for im in ims],
                      return_tensors="pt", padding=True).to("cuda")
        P = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new, **gen_kw)
        gens = [o[P:] for o in out]
        replies = [proc.decode(g, skip_special_tokens=False) for g in gens]
        n_gen = [int((g != proc.tokenizer.pad_token_id).sum()) for g in gens]
        forced = [args.think and "</think>" not in r for r in replies]
        idx_all = [i for i, f in enumerate(forced) if f]
        for c0 in range(0, len(idx_all), 4):            # re-reading long thinking: a few at a time
            idx = idx_all[c0:c0 + 4]
            cont = [built[i][0] + replies[i] + "\n</think>\n\n" for i in idx]
            inp2 = proc(text=cont, images=[im for i in idx for im in built[i][1]], return_tensors="pt",
                        padding=True).to("cuda")
            with torch.no_grad():
                out2 = model.generate(**inp2, max_new_tokens=16, do_sample=False)
            for j, i in enumerate(idx):
                replies[i] += "\n</think>\n\n" + proc.decode(out2[j, inp2["input_ids"].shape[1]:], skip_special_tokens=False)
            del inp2, out2
            torch.cuda.empty_cache()
        for t, reply, g, f in zip(tb, replies, n_gen, forced):
            for sp in special:
                reply = reply.replace(sp, "")
            lab = labels[t["id"]]
            ans = {lab[t["a"]]: t["a"], lab[t["b"]]: t["b"]}.get(parse(reply, lab[t["a"]], lab[t["b"]]), -1)
            rec = {k: t[k] for k in ("id", "walk", "probe", "familiar", "near", "far", "a", "b")}
            rec.update(answer=ans, correct=ans == t["near"], prompt_tokens=P, gen_tokens=g, forced=bool(f),
                       text=reply[-600:])
            with open(path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        done_n = b0 + len(tb)
        if b0 == 0 or done_n % 24 < len(tb):
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f"{done_n}/{len(trials)}, {time.time() - t0:.0f}s, prompt {P} tokens, generated {max(n_gen)} max, "
                  f"forced {sum(forced)}, peak {mem:.0f} GiB, e.g. {reply[-120:]!r}", flush=True)
    print(f"{tag}: done, {time.time() - t0:.0f}s -> {path}", flush=True)


def score(args):
    summary = {}
    for path in sorted(glob.glob(f"{OUT}/*.jsonl")):
        tag = os.path.basename(path)[:-6]
        R = [json.loads(l) for l in open(path)]
        name = next((n for n in TRIALS if f"-{n}-" in tag), "original")
        trials = {t["id"]: t for t in (json.load(open(TRIALS[name])) if os.path.exists(TRIALS[name]) else [])}
        o = {}
        for cat, sel in [("all", R), ("novel", [r for r in R if not r["familiar"]]),
                         ("familiar", [r for r in R if r["familiar"]])]:
            if not sel:
                continue
            ok = np.array([r["answer"] > 0 for r in sel])
            right = np.array([r["correct"] for r in sel])
            k, n = int(right.sum()), len(sel)
            p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n      # one-sided, against 50%
            pix = np.array([trials[r["id"]]["pixels_favour_near"] == (r["answer"] == r["near"])
                            for r in sel if r["answer"] > 0 and r["id"] in trials])
            # the "remember when you saw this view" shortcut: the option nearer in frame number to the
            # walk frame that looks most like the probe
            adj = [r for r in sel if "best_match_frame" in trials.get(r["id"], {})]
            far_cued = [r for r in adj if abs(r["far"] - trials[r["id"]]["best_match_frame"])
                        <= abs(r["near"] - trials[r["id"]]["best_match_frame"])]
            o[cat] = {"n": n, "accuracy": k / n, "parsed": float(ok.mean()), "p_above_chance": p,
                      "accuracy_when_frame_order_favours_far": float(np.mean([r["correct"] for r in far_cued]))
                      if far_cued else float("nan"), "n_frame_order_favours_far": len(far_cued),
                      "picked_first_listed": float(np.mean([r["answer"] == r["a"] for r in sel if r["answer"] > 0])),
                      "picked_more_similar_looking": float(pix.mean()) if len(pix) else float("nan")}
        summary[tag] = o
        a = o.get("novel", o["all"])
        print(f"{tag:55s} novel {a['accuracy']:.0%} ({a['n']}, p={a['p_above_chance']:.2g}, parsed {a['parsed']:.0%}, "
              f"first-listed {a['picked_first_listed']:.0%}, more-similar {a['picked_more_similar_looking']:.0%}, "
              f"frame-order-favours-far {a['accuracy_when_frame_order_favours_far']:.0%} of {a['n_frame_order_favours_far']})"
              + (f"  familiar {o['familiar']['accuracy']:.0%}" if "familiar" in o else ""))
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--trials", choices=list(TRIALS), default="dissimilar")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--no-walk", action="store_true")
    ap.add_argument("--think", action="store_true", help="enable_thinking in the chat template")
    ap.add_argument("--max-new", type=int, default=16, help="tokens to generate (the thinking budget with --think)")
    ap.add_argument("--batch", type=int, default=1, help="trials generated together (left-padded)")
    ap.add_argument("--effort", default="medium", help="reasoning_effort for templates that take it (Qwen3.8)")
    ap.add_argument("--image-tokens", type=int, default=70, help="Gemma: soft tokens per image (70/140/280/...)")
    ap.add_argument("--walks", type=int)
    ap.add_argument("--max-images", type=int, help="cap on images per prompt: show an evenly spaced subset of "
                                                    "the walk that always includes the two options")
    ap.add_argument("--shuffle-walk", action="store_true", help="walk frames in a fixed random order")
    ap.add_argument("--relabel", action="store_true", help="with --shuffle-walk: label frames in shown order")
    ap.add_argument("--tag")
    ap.add_argument("--adapter", help="LoRA adapter folder to merge in (finetune.py)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--data", help="walks folder (default: the trials' own, data/walks for room A)")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    score(args) if args.score else run(args)


if __name__ == "__main__":
    main()
