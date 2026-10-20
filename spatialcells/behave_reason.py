"""Behavioural follow-up: does reasoning, or an easier two-choice question, help?

    python -m spatialcells.behave_reason --model qwen38 --walks 24
    python -m spatialcells.behave_reason --model qwen38 --walks 1 --budget 256    # smoke test
    python -m spatialcells.behave_reason --model qwen38 --score

Frames are labelled with 3-digit numbers ("Frame 001:"), so every question has the
same token length and probes can be decoded together as a batch. Two questions:
  nearest  which numbered frame was taken closest to where this new view was taken?
  choice   was it taken closer to where Frame a or Frame b was taken? a is the walk
           frame nearest the probe, b a random frame more than 3 m away, in random
           order (chance 50%)
Each is asked directly (thinking off, greedy) on every probe, and with reasoning
(thinking on at --effort, the model's own sampling settings) on the novel-view probes. Thinking
longer than --budget tokens is closed with "</think>" and the answer read after it.
Writes results/behave/<model>-reason.npz; --score prints and plots the comparison.

With --trials NAME (original or dissimilar, see two_choice.py) it asks only the two-choice
question, on exactly those trials (their options and order), with reasoning on (or off with
--direct-only), and appends two_choice-style records to results/two_choice/<model>-<trials>-
256px-think-<effort>.jsonl, so two_choice.py --score covers it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import torch

from .behave import Rewind, load
from .extract import QWEN_IDS

WALK_PROMPT = ("These are consecutive first-person views as you walk around a room. "
               "Each view is labelled with its frame number.")
QUESTIONS = {
    "nearest": lambda a, b: ("This new view was taken somewhere in the same room, possibly facing a different "
                             "direction. Which numbered frame above was taken from the position closest to where "
                             "this new view was taken? Answer with the frame number only."),
    "choice": lambda a, b: (f"This new view was taken somewhere in the same room, possibly facing a different "
                            f"direction. Was it taken closer to where Frame {a:03d} was taken, or to where Frame "
                            f"{b:03d} was taken? Answer with one of the two frame numbers only."),
}
FAR = 3.0
MAX_ANSWER = 12


def chat_text(proc, n_walk, question, think, effort="medium"):
    content = [{"type": "text", "text": WALK_PROMPT}]
    for k in range(1, n_walk + 1):
        content += [{"type": "text", "text": f"Frame {k:03d}:"}, {"type": "image"}]
    content += [{"type": "text", "text": "New view:"}, {"type": "image"}, {"type": "text", "text": question}]
    return proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=think, reasoning_effort=effort)


def sample(logits, gen, temperature, top_k, top_p):
    v, i = (logits / temperature).topk(top_k, -1)
    p = v.softmax(-1)
    p = p * ((p.cumsum(-1) - p) < top_p)
    j = torch.multinomial(p / p.sum(-1, keepdim=True), 1, generator=gen)
    return i.gather(-1, j).squeeze(-1)


@torch.no_grad()
def run(args):
    from PIL import Image
    from .analyze_behave import geometry
    dev = args.device
    frames = np.load(f"{args.data}/frames.npy", mmap_mode="r")
    probes = np.load(f"{args.data}/probes.npy", mmap_mode="r")
    meta, pmeta, G = geometry(args.data)
    proc, model = load(QWEN_IDS[args.model], dev)
    tok = proc.tokenizer
    gcfg = model.generation_config
    temp, top_k, top_p = gcfg.temperature or 1.0, gcfg.top_k or 20, gcfg.top_p or 0.95
    pad = tok.convert_tokens_to_ids("<|im_end|>")    # fed to sequences that have finished
    stop = {pad, tok.eos_token_id, *tok.encode("\n")}
    end_id = tok.convert_tokens_to_ids("<|vision_end|>")
    think_end = tok.convert_tokens_to_ids("</think>")
    force = tok.encode("\n") + [think_end]          # closes thinking at the budget
    after = tok.encode("\n\n")                        # what the template puts after </think>
    img = lambda a: Image.fromarray(np.asarray(a))
    gen = torch.Generator(dev).manual_seed(0)
    rng = np.random.default_rng(0)
    print(f"{QWEN_IDS[args.model]}: sampling T={temp} top_k={top_k} top_p={top_p}, effort {args.effort}, "
          f"budget {args.budget}", flush=True)

    rec = {k: [] for k in ("probe", "question", "think", "answer", "near", "far", "n_think", "forced", "text")}
    trials, tc_path, done = None, None, set()
    if args.trials:
        from .two_choice import OUT as TC_OUT, TRIALS
        trials = json.load(open(TRIALS[args.trials]))
        name = QWEN_IDS[args.model].split("/")[-1]
        tc_path = (f"{TC_OUT}/{name}-{args.trials}-256px" +
                   ("" if args.direct_only else f"-think-{args.effort}") + "-cached.jsonl")
        os.makedirs(TC_OUT, exist_ok=True)
        done = {json.loads(l)["id"] for l in open(tc_path)} if os.path.exists(tc_path) else set()
        trials = [t for t in trials if t["id"] not in done]
        print(f"{tc_path}: {len(trials)} trials to run ({len(done)} done)", flush=True)
    walks = sorted({t["walk"] for t in trials}) if trials is not None else np.unique(meta["walk"])[:args.walks]
    t0 = time.time()
    for wi, w in enumerate(walks):
        idx = np.flatnonzero(meta["walk"] == w)
        idx = idx[np.argsort(meta["step"][idx])]
        n = len(idx)
        if trials is not None:
            tw = {t["probe"]: t for t in trials if t["walk"] == w}
            pidx = np.array(sorted(tw))
            novel = list(pidx)
            opts = {p: (tw[p]["near"], tw[p]["far"], (tw[p]["a"], tw[p]["b"])) for p in pidx}
        else:
            pidx = np.flatnonzero(pmeta["walk"] == w)
            novel = [p for p in pidx if not G[p]["familiar"]]
            opts = {}
            for p in pidx:
                near = int(np.argmin(G[p]["d"])) + 1
                far_ok = np.flatnonzero(G[p]["d"] > FAR)
                far = int(rng.choice(far_ok) if len(far_ok) else np.argmax(G[p]["d"])) + 1
                opts[p] = (near, far, (near, far) if rng.random() < 0.5 else (far, near))

        modes = (False,) if args.direct_only else ((True,) if trials is not None else (False, True))
        for think in modes:
            # walk prefix, shared by every question in this mode (thinking adds a system prompt)
            text0 = chat_text(proc, n, QUESTIONS["nearest"](1, 2), think, args.effort)
            pos_end = 0
            for _ in range(n):
                pos_end = text0.index("<|vision_end|>", pos_end) + len("<|vision_end|>")
            prefix = text0[:pos_end]
            pin = proc(text=[prefix], images=[img(frames[i]) for i in idx], return_tensors="pt").to(dev)
            P = pin["input_ids"].shape[1]
            assert int(torch.nonzero(pin["input_ids"][0] == end_id)[-1]) == P - 1
            cache = model.model(**pin, use_cache=True).past_key_values
            rewind = Rewind(cache)
            for q in ("choice",) if trials is not None else ("nearest", "choice"):
                sel = novel if think else pidx
                for b0 in range(0, len(sel), args.batch):
                    batch = sel[b0:b0 + args.batch]
                    B = len(batch)
                    texts = [chat_text(proc, n, QUESTIONS[q](*opts[p][2]), think, args.effort) for p in batch]
                    assert all(t.startswith(prefix) for t in texts)
                    sin = proc(text=[t[len(prefix):] for t in texts], images=[img(probes[p]) for p in batch],
                               return_tensors="pt", padding=True).to(dev)
                    assert bool(sin["attention_mask"].all()), "questions differ in token length"
                    S = sin["input_ids"].shape[1]
                    # M-RoPE positions of prefix + suffix (identical for every probe in the batch)
                    ids = torch.cat([pin["input_ids"], sin["input_ids"][:1]], 1)
                    mm = torch.cat([pin["mm_token_type_ids"], sin["mm_token_type_ids"][:1]], 1)
                    grid = torch.cat([pin["image_grid_thw"], sin["image_grid_thw"][:1]])
                    pos, _ = model.model.get_rope_index(ids, mm_token_type_ids=mm, image_grid_thw=grid)
                    spos = pos[:, :, P:].expand(3, B, S)
                    nxt_pos = int(pos[:, 0, -1].max()) + 1   # text after text: every axis continues by one

                    tb = time.time()
                    rewind()
                    rewind.expand(B)
                    o = model.model(input_ids=sin["input_ids"], pixel_values=sin["pixel_values"],
                                    image_grid_thw=sin["image_grid_thw"], position_ids=spos,
                                    past_key_values=cache, use_cache=True)
                    st = [{"queue": [], "phase": "think" if think else "answer", "n": 0, "forced": False,
                           "ans": [], "all": []} for _ in batch]
                    k = 0
                    while True:
                        logits = model.lm_head(o.last_hidden_state[:, -1]).float()
                        greedy = logits.argmax(-1).tolist()
                        sampled = sample(logits, gen, temp, top_k, top_p).tolist() if think else greedy
                        feed = []
                        for b, s in enumerate(st):
                            if s["phase"] == "done":
                                feed.append(pad)
                                continue
                            forced = bool(s["queue"])
                            t = s["queue"].pop(0) if forced else (sampled[b] if s["phase"] == "think" else greedy[b])
                            s["all"].append(t)
                            if s["phase"] == "think":
                                s["n"] += 1
                                if t == think_end:
                                    s["phase"], s["queue"] = "answer", list(after)
                                elif s["n"] >= args.budget and not s["queue"]:
                                    s["queue"], s["forced"] = list(force), True
                            elif not forced:
                                if t in stop or len(s["ans"]) >= MAX_ANSWER:
                                    s["phase"] = "done"
                                else:
                                    s["ans"].append(t)
                            feed.append(t)
                        if all(s["phase"] == "done" for s in st):
                            break
                        o = model.model(input_ids=torch.tensor(feed, device=dev)[:, None],
                                        position_ids=torch.full((3, B, 1), nxt_pos + k, device=dev),
                                        past_key_values=cache, use_cache=True)
                        k += 1
                    for p, s in zip(batch, st):
                        m = re.search(r"\d+", tok.decode(s["ans"]))
                        a = int(m.group()) if m else -1
                        if tc_path:
                            t = tw[p]
                            r = {k: t[k] for k in ("id", "walk", "probe", "familiar", "near", "far", "a", "b")}
                            r.update(answer=a if a in (t["a"], t["b"]) else -1, correct=a == t["near"],
                                     gen_tokens=s["n"], forced=s["forced"], text=tok.decode(s["all"])[-600:])
                            with open(tc_path, "a") as f:
                                f.write(json.dumps(r) + "\n")
                        rec["probe"].append(p), rec["question"].append(q), rec["think"].append(think)
                        rec["answer"].append(a if 1 <= a <= n else -1)
                        rec["near"].append(opts[p][0]), rec["far"].append(opts[p][1])
                        rec["n_think"].append(s["n"]), rec["forced"].append(s["forced"])
                        rec["text"].append(tok.decode(s["all"]))
                    if wi == 0 and b0 == 0:
                        print(f"[{q}{' think' if think else ''}] batch {B}, {k} steps, "
                              f"{1000 * (time.time() - tb) / max(k, 1):.0f} ms/step, thinking tokens "
                              f"{[s['n'] for s in st]}, e.g. {rec['text'][-1][-400:]!r}", flush=True)
            del cache, rewind
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
        print(f"walk {w}: {len(pidx)} probes ({len(novel)} novel), {time.time() - t0:.0f}s, peak {mem:.0f} GiB", flush=True)

    if tc_path:
        print(f"behave_reason: {len(rec['probe'])} answers, {time.time() - t0:.0f}s -> {tc_path}", flush=True)
        return
    os.makedirs(args.out, exist_ok=True)
    path = f"{args.out}/{args.model}-reason{'-smoke' if args.walks and args.walks < 3 else ''}.npz"
    np.savez(path, **{k: np.array(v) for k, v in rec.items()})
    print(f"behave_reason: {len(rec['probe'])} answers, {time.time() - t0:.0f}s -> {path}", flush=True)


def score(args):
    from .analyze_behave import NEAR, SAME_HEADING, angdiff, geometry
    meta, pmeta, G = geometry(args.data)
    z = np.load(f"{args.out}/{args.model}-reason.npz")
    novel_set = {p for p in G if not G[p]["familiar"]}
    out = {}
    print(f"{args.model}: novel probes in these walks: {len({p for p in z['probe'] if p in novel_set})}")
    for q in ("nearest", "choice"):
        for think in (False, True):
            for cat in ("familiar", "novel"):
                sel = [i for i in range(len(z["probe"])) if z["question"][i] == q and bool(z["think"][i]) == think
                       and (z["probe"][i] in novel_set) == (cat == "novel")]
                if not sel:
                    continue
                key = f"{q}/{'reasoning' if think else 'direct'}/{cat}"
                a, p = z["answer"][sel], z["probe"][sel]
                ok = a > 0
                o = {"n": len(sel), "parsed": float(ok.mean())}
                if q == "nearest":
                    d = np.array([G[pp]["d"][aa - 1] for pp, aa in zip(p[ok], a[ok])])
                    dh = np.array([angdiff(meta["theta"][G[pp]["fi"][aa - 1]], pmeta["theta"][pp])
                                   for pp, aa in zip(p[ok], a[ok])])
                    o.update(hit_rate_1m=float((d < NEAR).mean()), median_error_m=float(np.median(d)),
                             random_hit_rate=float(np.mean([(G[pp]["d"] < NEAR).mean() for pp in p])),
                             picks_facing_same_way=float((dh < SAME_HEADING).mean()))
                else:
                    near, far = z["near"][sel], z["far"][sel]
                    valid = (a == near) | (a == far)
                    pix_near = np.array([G[pp]["sim"][nn - 1] > G[pp]["sim"][ff - 1] for pp, nn, ff in zip(p, near, far)])
                    right = a == near
                    o.update(valid=float(valid.mean()), accuracy=float(right[valid].mean()),
                             pixel_match_accuracy=float(pix_near.mean()),
                             accuracy_when_pixels_favour_far=float(right[valid & ~pix_near].mean())
                             if (valid & ~pix_near).any() else float("nan"),
                             n_pixels_favour_far=int((valid & ~pix_near).sum()))
                if think:
                    o.update(median_think_tokens=float(np.median(z["n_think"][sel])),
                             hit_budget=float(np.mean(z["forced"][sel])))
                out[key] = o
                print(f"  {key:28s} " + "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                                                   for k, v in o.items()))
    json.dump(out, open(f"results/behave_reason_{args.model}.json", "w"), indent=1)
    figure(out, args.model)


def figure(out, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .summarize import INK2, MODELS, S1, S2, S3, style
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    groups = [("familiar views,\nno reasoning", "direct/familiar", S3), ("novel views,\nno reasoning", "direct/novel", S1),
              ("novel views,\nwith reasoning", "reasoning/novel", S2)]
    for ax, q, key, title, ylab in [
            (axs[0], "nearest", "hit_rate_1m", "a  Which frame was taken nearest?", "answers within 1 m of the true place"),
            (axs[1], "choice", "accuracy", "b  Nearer to Frame a or Frame b?", "picked the nearer frame")]:
        g = [(lab, out[f"{q}/{k}"], col) for lab, k, col in groups if f"{q}/{k}" in out]
        bars = ax.bar(range(len(g)), [o[key] for _, o, _ in g], 0.6, color=[c for *_, c in g])
        ax.bar_label(bars, labels=[f"{o[key]:.0%}\nn={o['n']}" for _, o, _ in g], fontsize=8, color=INK2, padding=2)
        if q == "nearest":
            refs = [(out[f"{q}/direct/novel"]["random_hit_rate"], "random frame, novel views", "--")]
        else:
            nov = out.get("choice/direct/novel", {})
            refs = [(0.5, "chance", "--"), (nov.get("pixel_match_accuracy", np.nan), "pick the more similar-looking frame, novel views", ":")]
        for v, lab, ls in refs:
            ax.axhline(v, color=INK2, lw=1, ls=ls, label=f"{lab} ({v:.0%})")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        style(ax, title, ylab)
        ax.set_xticks(range(len(g)), [lab for lab, *_ in g], fontsize=8)
        ax.set_ylim(0, 1.05)
    fig.suptitle(f"{MODELS[model].replace(chr(10), ' ')}: asking where a new view was taken", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"results/behave_reason_{model}.png", dpi=110, facecolor="white")
    print(f"wrote results/behave_reason_{model}.png")


def two_choice_figure():
    """Two-choice accuracy, familiar vs novel views, for every model scored so far and VGGT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .summarize import INK2, MODELS, S1, S3, style
    rows = []
    for m in MODELS:
        if os.path.exists(f"results/behave_reason_{m}.json"):
            r = json.load(open(f"results/behave_reason_{m}.json"))
            rows.append((MODELS[m], r["choice/direct/familiar"]["accuracy"], r["choice/direct/novel"]["accuracy"],
                         r["choice/direct/novel"]["n"]))
            if "choice/reasoning/novel" in r:
                rows.append((MODELS[m].split("\n")[0] + "\nwith reasoning", np.nan,
                             r["choice/reasoning/novel"]["accuracy"], r["choice/reasoning/novel"]["n"]))
    claude_names = {"claude-opus-5-5": "Claude Opus 5.5\n(API, adaptive thinking)",
                    "claude-opus-5-5-nowalk": "Claude Opus 5.5\nno walk, 3 images only",
                    "claude-fable-5-1": "Claude Fable 5.1\n(API, adaptive thinking)"}
    for name, label in claude_names.items():   # summaries from claude_test.py --score, copied here
        if os.path.exists(f"results/claude/{name}_summary.json"):
            r = json.load(open(f"results/claude/{name}_summary.json"))
            rows.append((label, r.get("choice/reasoning/familiar", {}).get("accuracy", np.nan),
                         r["choice/reasoning/novel"]["accuracy"], r["choice/reasoning/novel"]["n"]))
    if os.path.exists("results/vggt_control.json"):
        v = json.load(open("results/vggt_control.json"))
        rows.append(("VGGT\n(3D vision model)", v["familiar"]["choice_accuracy"], v["novel"]["choice_accuracy"],
                     v["novel"]["n"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(2.1 * len(rows) + 2, 4.6))
    x = np.arange(len(rows))
    b1 = ax.bar(x - 0.2, [r[1] for r in rows], 0.38, color=S3, label="familiar views")
    b2 = ax.bar(x + 0.2, [r[2] for r in rows], 0.38, color=S1, label="novel views")
    for b in (b1, b2):
        ax.bar_label(b, fmt=lambda v: "" if np.isnan(v) else f"{v:.0%}", fontsize=8, color=INK2, padding=2)
    ax.axhline(0.5, color=INK2, lw=1, ls="--", label="chance")
    style(ax, "Was the new view taken nearer Frame a or Frame b? (identical novel-view trials for every system)",
          "picked the nearer frame")
    ax.set_xticks(x, [r[0] for r in rows], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig("results/two_choice.png", dpi=110, facecolor="white")
    print("wrote results/two_choice.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(QWEN_IDS), default="qwen38")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", default="data/walks")
    ap.add_argument("--walks", type=int, default=24)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--budget", type=int, default=2048, help="max thinking tokens")
    ap.add_argument("--effort", default="medium", help="Qwen3.8 reasoning effort: low, medium or xhigh")
    ap.add_argument("--direct-only", action="store_true")
    ap.add_argument("--trials", choices=["original", "dissimilar"], help="two-choice only, on these trials")
    ap.add_argument("--out", default="results/behave")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    score(args) if args.score else run(args)


if __name__ == "__main__":
    main()
