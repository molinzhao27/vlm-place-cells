"""Teach Qwen3-VL-8B the two-choice task on procedural rooms with LoRA (experiment 9).

    python -m spatialcells.finetune [--data data/proc] [--out data/ft/qwen8b-proc] [--epochs 2] [--seed 0]

Each training sequence is one procedural room (procedural_data.py): two_choice.py's prompt with
the walk's 64 shown frames, labelled with their real frame numbers ("Frame 001:", "Frame 005:",
...), then that room's two-choice questions as consecutive chat turns (new view and question;
the answer is the near frame's number, e.g. "055"). The first turn is exactly the prompt
two_choice.py --max-images 64 asks, so evaluation on room A uses the first-turn format. The loss
is on the answers only. LoRA (rank 16, alpha 32, dropout 0.05) on every attention and MLP
projection of the language model; the vision tower and merger stay frozen. bf16, gradient
checkpointing, AdamW lr 1e-4 with 30 warm-up steps and cosine decay, 8 rooms per update.
The last --val rooms are held out: after every --eval-every updates, accuracy on their first
question (greedy, as in two_choice.py) is logged. Saves the adapter to --out after each epoch.
Frames are 256 px, 64 visual tokens each (as every other Qwen run here). --seed changes the LoRA
initialisation, dropout and room order but not the data or the held-out rooms; seed 0 is the
original run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from .two_choice import QUESTION, WALK_PROMPT, parse

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def room_messages(room, trials, data):
    """Chat messages and images for one room: walk + first question, then one turn per further question."""
    from PIL import Image
    z = np.load(f"{data}/room_{room:04d}.npz")
    content = [{"type": "text", "text": WALK_PROMPT}]
    images = []
    for k, f in zip(z["frame_numbers"], z["frames"]):
        content += [{"type": "text", "text": f"Frame {int(k):03d}:"}, {"type": "image"}]
        images.append(Image.fromarray(f))
    msgs = []
    for i, t in enumerate(trials):
        q = [{"type": "text", "text": "New view:"}, {"type": "image"},
             {"type": "text", "text": QUESTION.format(a=t["a"], b=t["b"])}]
        images.append(Image.fromarray(z["probes"][t["probe"]]))
        msgs += [{"role": "user", "content": (content if i == 0 else []) + q},
                 {"role": "assistant", "content": [{"type": "text", "text": f"{t['near']:03d}"}]}]
    return msgs, images


def encode(proc, msgs, images):
    """Token inputs with labels on the assistant answers (and their <|im_end|>) only."""
    text = proc.apply_chat_template(msgs, tokenize=False)
    inputs = proc(text=[text], images=images, return_tensors="pt")
    ids = inputs["input_ids"][0]
    head = torch.tensor(proc.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))
    end = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
    labels = torch.full_like(ids, -100)
    starts = [i + len(head) for i in range(len(ids) - len(head) + 1) if torch.equal(ids[i:i + len(head)], head)]
    for s in starts:
        e = s
        while ids[e] != end:
            e += 1
        labels[s:e + 1] = ids[s:e + 1]
    inputs["labels"] = labels[None]
    return inputs


@torch.no_grad()
def validate(model, proc, rooms, by_room, data):
    model.eval()
    right = 0
    for r in rooms:
        t = by_room[r][0]
        msgs, images = room_messages(r, [t], data)
        text = proc.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=images, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
        right += parse(proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True), t["a"], t["b"]) == t["near"]
    model.train()
    return right / len(rooms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/proc")
    ap.add_argument("--out", default="data/ft/qwen8b-proc")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--val", type=int, default=64, help="held-out rooms (the last ones)")
    ap.add_argument("--eval-every", type=int, default=25, help="updates between validation runs")
    ap.add_argument("--max-rooms", type=int, help="train on only this many rooms (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.seed:                       # seed 0 keeps torch's default generator, as the original run
        torch.manual_seed(args.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    trials = json.load(open(f"{args.data}/trials.json"))
    by_room = {}
    for t in trials:
        by_room.setdefault(t["room"], []).append(t)
    rooms = sorted(by_room)
    val, train = rooms[-args.val:], rooms[:-args.val][:args.max_rooms]
    print(f"{len(train)} training rooms ({sum(len(by_room[r]) for r in train)} questions), {len(val)} held out", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    proc.image_processor.size = {"shortest_edge": 256 * 256, "longest_edge": 256 * 256}
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                                        attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=TARGETS))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = max(1, args.epochs * math.ceil(len(train) / args.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    os.makedirs(args.out, exist_ok=True)
    def val_log(update, **extra):
        log.append({"update": update, "val_acc": validate(model, proc, val, by_room, args.data), **extra})
        print(f"update {update}: held-out first-question accuracy {log[-1]['val_acc']:.0%}", flush=True)
    log = []
    val_log(0)
    model.train()
    t0, step, run_loss, run_n = time.time(), 0, 0.0, 0
    for ep in range(args.epochs):
        order = train[:]
        random.Random(ep + 1000 * args.seed).shuffle(order)
        for i, r in enumerate(order):
            inputs = encode(proc, *room_messages(r, by_room[r], args.data)).to("cuda")
            loss = model(**inputs).loss
            (loss / args.accum).backward()
            run_loss, run_n = run_loss + loss.item(), run_n + 1
            if (i + 1) % args.accum and i + 1 < len(order):    # update every accum rooms and at the epoch's end
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0:
                print(f"epoch {ep} update {step}/{total}: loss {run_loss / run_n:.3f}, lr {sched.get_last_lr()[0]:.1e}, "
                      f"{inputs['input_ids'].shape[1]} tokens, {time.time() - t0:.0f}s, "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB", flush=True)
                log.append({"update": step, "loss": run_loss / run_n})
                run_loss, run_n = 0.0, 0
            if step % args.eval_every == 0:
                val_log(step)
                json.dump(log, open(f"{args.out}/train_log.json", "w"))
        model.save_pretrained(args.out)
        val_log(step, epoch_end=ep)
        print(f"epoch {ep} done ({time.time() - t0:.0f}s); saved {args.out}", flush=True)
        json.dump(log, open(f"{args.out}/train_log.json", "w"))


if __name__ == "__main__":
    main()
