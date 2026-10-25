"""Experiment 14a: the residual stream of Qwen3-VL-8B (base or LoRA) while it does the two-choice task.

    python -m spatialcells.task_record --device mps [--adapter data/ft/qwen8b-proc] [--rooms P5000 ...]
    python -m spatialcells.task_record --dry-run N [--adapter ...]     # tokenise only, no model

Every lure-free trial of the ten held-out rooms (experiment 10), with the exact prompt two_choice.py
gives it with --max-images 64 (the 61-63 labelled walk frames, "New view:", the new view and the
question; 256 px, the chat template with the generation prompt). two_choice's build() is local to its
run(), so its few lines are repeated here with its own present(), prompts and processor settings, and
every prompt's token count is checked against experiment 10's run of the same trial (a mismatch stops
the run). One forward pass per trial, no generation. The residual stream after every decoder layer
(the input to the next layer, so it includes the deepstack visual features added after layers 1-3 at
visual tokens; for the last layer, its raw output before the final norm) is read at four sites:
  img_end   the new view's <|vision_end|> token
  img_mean  the mean over the new view's 64 visual tokens
  question  the question's last token (just before <|im_end|>)
  last      the last prompt token (after "<|im_start|>assistant\\n", where the answer is generated)
Writes data/acts_task/<name>/<site>.npy (float16, trials x 36 layers x 4096, preallocated and filled as
trials finish), meta.json (per trial: room, id, the new view's x, y, theta, the options' distances,
targets, and the model's answer from experiment 10's run of the same prompt, if there is one) and
done.jsonl (one line per recorded trial: its row, prompt tokens, site indices and the next token the
model would generate). Trials already in done.jsonl are skipped, so a run can be stopped and restarted.
--dry-run N prints the site indices (and the tokens there) for the first trial and checks the first N
prompts' token counts; --layers keeps a subset of layers (default all).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np

from .two_choice import OUT as TC_OUT, QUESTION, WALK_PROMPT, default_tag, present, trials_file, walk_order

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
ROOMS = [f"P{s}" for s in range(5000, 5010)]
SITES = ("img_end", "img_mean", "question", "last")


def tc_args(room, adapter, device):
    """two_choice's arguments for experiment 10's 64-frame walk run in this room."""
    return SimpleNamespace(model=MODEL, adapter=adapter, trials=f"room{room}", size=256, no_walk=False, think=False,
                           effort="medium", max_images=64, shuffle_walk=False, relabel=False, reverse_walk=False,
                           window_shuffle=None, renumber=False, irrelevant_walk=None, device=device)


def trial_table(rooms, adapter, device):
    """Every trial in row order, with its targets and experiment 10's answer on the same prompt."""
    rows = []
    for R in rooms:
        pm = dict(np.load(f"data/walks-room{R}/probes_meta.npz"))
        path = f"{TC_OUT}/{default_tag(tc_args(R, adapter, device))}.jsonl"
        beh = {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}
        for t in json.load(open(trials_file(f"room{R}"))):
            p = t["probe"]
            d_a, d_b = ((t["near_dist_m"], t["far_dist_m"]) if t["a"] == t["near"]
                        else (t["far_dist_m"], t["near_dist_m"]))
            b = beh.get(t["id"], {})
            rows.append({"room": R, "id": t["id"], "walk": t["walk"], "probe": p, "familiar": t["familiar"],
                         "a": t["a"], "b": t["b"], "near": t["near"], "x": float(pm["x"][p]), "y": float(pm["y"][p]),
                         "theta": float(pm["theta"][p]), "d_a": d_a, "d_b": d_b, "d_a_minus_d_b": round(d_a - d_b, 3),
                         "a_nearer": t["a"] == t["near"], "answer": b.get("answer"), "correct": b.get("correct"),
                         "exp10_prompt_tokens": b.get("prompt_tokens")})
    return rows


def sites(ids, tok):
    """Token positions of the four sites in one tokenised prompt (1-d id tensor)."""
    tid = tok.convert_tokens_to_ids
    vs = (ids == tid("<|vision_start|>")).nonzero()[:, 0]
    ve = (ids == tid("<|vision_end|>")).nonzero()[:, 0]
    ie = (ids == tid("<|im_end|>")).nonzero()[:, 0]
    v0, v1 = int(vs[-1]) + 1, int(ve[-1])                    # the new view's visual tokens: v0 .. v1 - 1
    assert (ids[v0:v1] == tid("<|image_pad|>")).all() and v1 - v0 == 64, (v0, v1)
    return {"img_end": v1, "img_mean": (v0, v1), "question": int(ie[-1]) - 1, "last": len(ids) - 1,
            "n_images": len(ve)}


def build_prompt(r, proc, data, adapter, device):
    """two_choice's prompt for trial row r: (text, images, shown, lab), with shown the real frame numbers in the
    order shown and lab their labels (present()). data caches one room's frames, probes and walk order."""
    from PIL import Image
    if r["room"] not in data:
        data.clear()
        d = f"data/walks-room{r['room']}"
        data[r["room"]] = (np.load(f"{d}/frames.npy", mmap_mode="r"), np.load(f"{d}/probes.npy", mmap_mode="r"),
                           walk_order(dict(np.load(f"{d}/meta.npz"))))
    frames, probes, order = data[r["room"]]
    fi = order[r["walk"]]
    shown, lab = present(r, len(fi), tc_args(r["room"], adapter, device))
    content = [{"type": "text", "text": WALK_PROMPT}]
    images = []
    for k in shown:
        content += [{"type": "text", "text": f"Frame {lab[k]:03d}:"}, {"type": "image"}]
        images.append(Image.fromarray(np.asarray(frames[fi[k - 1]])))
    content += [{"type": "text", "text": "New view:"}, {"type": "image"},
                {"type": "text", "text": QUESTION.format(a=lab[r["a"]], b=lab[r["b"]])}]
    images.append(Image.fromarray(np.asarray(probes[r["probe"]])))
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False, reasoning_effort="medium")
    return text, images, shown, lab


def show_sites(ids, s, tok):
    t = lambda a, b: tok.convert_ids_to_tokens(ids[a:b].tolist())
    for k in SITES:
        if k == "img_mean":
            v0, v1 = s[k]
            print(f"  {k:9s} {v0}..{v1 - 1}  {t(v0 - 2, v0 + 1)} ... {t(v1 - 1, v1 + 1)} ({v1 - v0} visual tokens)")
        else:
            print(f"  {k:9s} {s[k]}  {t(s[k] - 3, s[k])} [{t(s[k], s[k] + 1)[0]}] {t(s[k] + 1, s[k] + 3)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="LoRA adapter folder to merge in (finetune.py)")
    ap.add_argument("--rooms", nargs="+", default=ROOMS)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--name", help="output folder under data/acts_task (default qwen8b[+adapter folder])")
    ap.add_argument("--layers", help="comma-separated decoder layers to keep, 1-based (default all)")
    ap.add_argument("--dry-run", type=int, metavar="N", help="tokenise only: site indices, token counts of N prompts")
    args = ap.parse_args()
    name = args.name or "qwen8b" + (f"+{os.path.basename(args.adapter.rstrip('/'))}" if args.adapter else "")
    out = f"data/acts_task/{name}"

    import torch
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL)
    proc.image_processor.size = {"shortest_edge": 256 * 256, "longest_edge": 256 * 256}   # as two_choice --size 256
    tok = proc.tokenizer
    rows = trial_table(args.rooms, args.adapter, args.device)
    print(f"{name}: {len(rows)} trials in {len(args.rooms)} rooms", flush=True)

    data = {}

    def build(i):
        """two_choice's prompt (text, images) for row i."""
        text, images, _, _ = build_prompt(rows[i], proc, data, args.adapter, args.device)
        return text, images

    def tokenise(i):
        text, images = build(i)
        inputs = proc(text=[text], images=images, return_tensors="pt", padding=True)
        ids = inputs["input_ids"][0]
        s = sites(ids, tok)
        assert s["n_images"] == len(images), (s["n_images"], len(images))
        want = rows[i]["exp10_prompt_tokens"]
        if want is not None and want != len(ids):
            raise SystemExit(f"row {i} ({rows[i]['room']} {rows[i]['id']}): {len(ids)} prompt tokens, "
                             f"experiment 10's run had {want}; the prompt differs")
        return inputs, s

    if args.dry_run:
        t0 = time.time()
        for i in range(min(args.dry_run, len(rows))):
            inputs, s = tokenise(i)
            if i == 0:
                ids = inputs["input_ids"][0]
                print(f"row 0 ({rows[0]['room']} {rows[0]['id']}): {len(ids)} tokens, {s['n_images']} images, "
                      f"image grid {inputs['image_grid_thw'][-1].tolist()}; sites:")
                show_sites(ids, s, tok)
                print("  prompt tail: " + repr(tok.decode(ids[s["img_end"] - 1:])))
        n = min(args.dry_run, len(rows))
        checked = sum(rows[i]["exp10_prompt_tokens"] is not None for i in range(n))
        print(f"{n} prompts tokenised in {time.time() - t0:.0f}s; {checked} matched experiment 10's token count")
        return

    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(MODEL, attn_implementation="sdpa", dtype=torch.bfloat16,
                                                        device_map=args.device).eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
    blocks = model.model.language_model.layers
    n_layers, hidden = len(blocks), model.config.text_config.hidden_size
    keep = [int(k) for k in args.layers.split(",")] if args.layers else list(range(1, n_layers + 1))

    # The stream after layer k (1-based) is the input of layer k + 1, and the last layer's output.
    grab, at = {}, {}

    def take(k, h):
        if k in keep:
            v0, v1 = at["img_mean"]
            grab[k] = torch.stack([h[0, at["img_end"]].float(), h[0, v0:v1].float().mean(0),
                                   h[0, at["question"]].float(), h[0, at["last"]].float()]).cpu().numpy()
    for k, blk in enumerate(blocks[1:], start=1):
        blk.register_forward_pre_hook(lambda _, a, k=k: take(k, a[0]))
    blocks[-1].register_forward_hook(lambda _, a, o: take(n_layers, o[0] if isinstance(o, tuple) else o))

    os.makedirs(out, exist_ok=True)
    meta_path, done_path = f"{out}/meta.json", f"{out}/done.jsonl"
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        assert old["id"] == [f"{r['room']}/{r['id']}" for r in rows] and old["layers"] == keep, \
            f"{meta_path} was written for other trials or layers"
    meta = {k: [r[k] for r in rows] for k in rows[0]}
    meta.update(id=[f"{r['room']}/{r['id']}" for r in rows], trial_id=[r["id"] for r in rows], layers=keep,
                sites=list(SITES), model=MODEL, adapter=args.adapter)
    json.dump(meta, open(meta_path, "w"))
    shape = (len(rows), len(keep), hidden)
    arrs = {s: np.lib.format.open_memmap(f"{out}/{s}.npy", mode="r+" if os.path.exists(f"{out}/{s}.npy") else "w+",
                                         dtype=np.float16, shape=shape) for s in SITES}
    for s, a in arrs.items():
        assert a.shape == shape, (s, a.shape, shape)
    done = {json.loads(l)["row"] for l in open(done_path)} if os.path.exists(done_path) else set()
    todo = [i for i in range(len(rows)) if i not in done]
    print(f"{name}: {len(todo)} trials to record ({len(done)} done), layers {keep[0]}..{keep[-1]} ({len(keep)}), "
          f"-> {out}", flush=True)
    dev = torch.mps if args.device == "mps" else torch.cuda
    t0 = time.time()
    for j, i in enumerate(todo):
        inputs, s = tokenise(i)
        inputs = inputs.to(args.device)
        at.update(s)
        grab.clear()
        with torch.no_grad():
            h = model.model(**inputs, use_cache=False).last_hidden_state[:, -1]
            nxt = int(model.lm_head(h).argmax(-1))
        assert len(grab) == len(keep), sorted(grab)
        A = np.stack([grab[k] for k in keep], 1)              # sites x layers x hidden
        bad = int((~np.isfinite(A)).sum() + (np.abs(A) > 65504).sum())
        for n, site in enumerate(SITES):
            arrs[site][i] = np.clip(np.nan_to_num(A[n]), -65504, 65504).astype(np.float16)
            arrs[site].flush()
        with open(done_path, "a") as fh:
            fh.write(json.dumps({"row": i, "id": meta["id"][i], "prompt_tokens": int(inputs["input_ids"].shape[1]),
                                 "sites": {k: s[k] for k in SITES}, "next_token": tok.decode([nxt]),
                                 "nonfinite_or_clipped": bad}) + "\n")
        if j == 0 or (j + 1) % 48 == 0 or j + 1 == len(todo):
            mem = (dev.driver_allocated_memory() if args.device == "mps" else dev.max_memory_allocated()) / 2**30
            print(f"{j + 1}/{len(todo)}, {time.time() - t0:.0f}s, {meta['id'][i]}, {inputs['input_ids'].shape[1]} tokens, "
                  f"next token {tok.decode([nxt])!r}, {bad} non-finite/clipped, {mem:.0f} GiB", flush=True)
    print(f"{name}: done, {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
