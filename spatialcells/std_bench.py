"""Experiment 17: two standard spatial benchmarks, with their official prompts and metrics.

    python -m spatialcells.std_bench --bench vsibench|mindcube --name base|run0|... [--adapter DIR]
        [--device cuda|mps] [--blind] [--shard K/N] [--max-minutes M] [--limit N] [--out DIR] [--dry-run N]

VSI-Bench (Yang et al., CVPR 2025; nyu-visionx/VSI-Bench): 5,130 questions on 288 indoor videos
(data/bench/vsibench: test.jsonl, pruned_ids.txt, videos/<dataset>/<scene>.mp4). MindCube-Tiny (Yin et
al., ICLR 2026; MLL-Lab/MindCube): the 1,050-question evaluation set (data/bench/mindcube/data/raw/
MindCube_tinybench.jsonl and its images). Prompts, frame sampling, decoding and scoring follow each
benchmark's official code (docs/experiments.md, experiment 17, lists the sources); the protocol
constants are below.

Answers go to <out>/<bench>/<name>[-blind].jsonl (default out: results/std_bench), one line per
question, and a run skips questions already answered, so it can be stopped and restarted. --adapter
merges a LoRA adapter (finetune.py) into the model first. --blind gives the question without the video
or images (a language-prior control). --shard K/N takes every N-th scene starting at K (scenes sorted
by name); --max-minutes stops after the question running at that time, exit status 0. --dry-run N
prints the first N prompts and the frames each would show, without loading the model.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
VSI = "data/bench/vsibench"
MC = "data/bench/mindcube/data"

# ---- VSI-Bench protocol (lmms_eval/tasks/vsibench/{vsibench.yaml,utils.py}) ----
VSI_MCA = ["object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard",
           "object_rel_distance", "route_planning", "obj_appearance_order"]
VSI_NA = ["object_abs_distance", "object_counting", "object_size_estimation", "room_size_estimation"]
VSI_PRE = "These are frames of a video."
VSI_MCA_POST = "Answer with the option's letter from the given choices directly."
VSI_NA_POST = "Please answer the question using a single word or phrase."
VSI_SYSTEM = "You are a helpful assistant."        # lmms-eval's Qwen3-VL default (models/simple/qwen3_vl.py)
VSI_FRAMES = 32
VSI_MAX_NEW = 16
# lmms-eval's lenient number parser (tasks/vsibench/utils.py since July 2026, at 8f40097), a secondary score
_NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
                 "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
                 "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
                 "thousand": 1000}
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# ---- MindCube protocol: the repo's raw_qa template (src/prompt_generation/templates.py), images first
# in their listed order, no system message, each image capped at 480 x 480 pixels (src/inference/utils.py),
# greedy with up to 4096 new tokens (qwen_engine.py), whitespace collapsed, official extract_answer ----
MC_INSTRUCTION = """[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.
[Answer Instruction]
You only need to provide *ONE* correct answer selecting from the options listed below. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should **only** be '<answer>A. Above</answer>'.
"""
MC_MAX_PIXELS = 480 * 480
MC_MAX_NEW = 4096


def vsi_text(q):
    if q["question_type"] in VSI_NA:
        return VSI_PRE + "\n" + q["question"] + "\n" + VSI_NA_POST
    return "\n".join([VSI_PRE, q["question"], "Options:\n" + "\n".join(q["options"]), VSI_MCA_POST])


def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()


def to_float(pred):
    try:
        return float(pred)
    except BaseException:
        return None


def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs(pred - target) / target <= 1 - conf_intervs
    return float(accuracy.mean())


def extract_number(text):
    """lmms-eval's extract_number (first number in the reply, digits or English words), verbatim in effect."""
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s:
        return None
    digit_match = _NUM_RE.search(s)
    word_run_start = None
    word_run_tokens = []
    for m in re.finditer(r"[a-z]+", s):
        tok = m.group(0)
        if word_run_start is None:
            if tok in _NUMBER_WORDS:
                word_run_start = m.start()
                word_run_tokens.append(tok)
        else:
            if tok == "and":
                continue
            if tok in _NUMBER_WORDS:
                word_run_tokens.append(tok)
            else:
                break
    digit_first = digit_match is not None and (word_run_start is None or digit_match.start() < word_run_start)
    if digit_first:
        try:
            return float(digit_match.group(0))
        except ValueError:
            pass
    if word_run_start is None:
        return None
    total, current = 0, 0
    for tok in word_run_tokens:
        val = _NUMBER_WORDS[tok]
        if val == 100:
            current = max(current, 1) * 100
        elif val == 1000:
            total += max(current, 1) * 1000
            current = 0
        else:
            current += val
    return float(total + current)


def vsi_score(q, reply):
    """(parsed answer or None, official score, lenient score), scores in [0, 1]: exact match or MRA.
    Parsed is the letter or number the official rule reads, None if it reads nothing usable."""
    pred = fuzzy_matching(reply)
    if q["question_type"] in VSI_MCA:
        ok = float(pred.lower() == q["ground_truth"].lower())
        return (pred if re.fullmatch(r"[A-Za-z]", pred) else None), ok, ok
    gt = to_float(q["ground_truth"])
    x, y = to_float(pred), extract_number(reply)
    return (x, mean_relative_accuracy(x, gt) if x is not None else 0.0,
            mean_relative_accuracy(y, gt) if y is not None else 0.0)


def mc_text(q):
    return "\n".join([MC_INSTRUCTION, "[Question]\n" + q["question"]])


def extract_answer(text):
    """MindCube's official answer extractor (src/evaluation/core/extractors.py at b8b7062), verbatim."""
    if not text:
        return None
    simple_pattern_matches = list(re.finditer(r'([A-E])\.', text))
    if simple_pattern_matches:
        return simple_pattern_matches[-1].group(1)
    answer_section_match = re.search(r'<Answer>(.*?)(?:<|$)', text, re.DOTALL)
    if answer_section_match:
        answer_section = answer_section_match.group(1)
        for pattern in [
            r'[Mm]y answer is ([A-E])',
            r'[Mm]y answer is ([A-E])\.',
            r'[Tt]he answer is ([A-E])',
            r'(?:Answer: )?([A-E])\.',
            r'\b([A-E])\b'
        ]:
            matches = list(re.finditer(pattern, answer_section))
            if matches:
                return matches[-1].group(1)
    patterns = [
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'",]+(?=(?:\n|$|\.|"))',
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'"]+',
        r'(?:^|\n)(?:Answer: )?([A-E])(?:\.|$|\s)',
        r'[\*\"]([A-E])[\*\"]',
        r'\bAnswer:?\s*([A-E])\b',
        r'[Mm]y answer is ([A-E])',
        r'[Mm]y answer is ([A-E])\.',
        r'answer is ([A-E])',
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            return matches[-1].group(1)
    lines = text.split('\n')
    line_matches = []
    for i, line in enumerate(lines):
        match = re.search(r'([A-E])\. [A-Za-z0-9 \-\(\)\'",]+', line)
        if match:
            line_matches.append((i, match.group(1)))
    if line_matches:
        return line_matches[-1][1]
    for i in reversed(range(len(lines))):
        line = lines[i]
        match = re.search(r'\b([A-E])\b', line)
        if match:
            return match.group(1)
    return None


def clean_response(response):
    """MindCube's ResponseProcessor.clean_response (src/inference/utils.py)."""
    if not response:
        return "No response generated"
    return re.sub(r'\s+', ' ', response.strip())


def mc_score(q, reply):
    pred = extract_answer(clean_response(reply))
    ok = float(pred is not None and pred == q["gt_answer"])
    return pred, ok, ok


def load_questions(bench):
    if bench == "vsibench":
        Q = [json.loads(l) for l in open(f"{VSI}/test.jsonl")]
        pruned = {int(x) for x in open(f"{VSI}/pruned_ids.txt").read().split()}
        for q in Q:
            q["cluster"] = f"{q['dataset']}/{q['scene_name']}"
            q["pruned"] = q["id"] in pruned
        return sorted(Q, key=lambda q: (q["cluster"], q["id"]))
    Q = [json.loads(l) for l in open(f"{MC}/raw/MindCube_tinybench.jsonl")]
    for q in Q:
        q["setting"] = q["id"].split("_")[0]
        q["cluster"] = os.path.dirname(q["images"][0])
    return sorted(Q, key=lambda q: (q["cluster"], q["id"]))


def video_frames(path, n):
    """n frames sampled uniformly (qwen-vl-utils' torch.linspace(0, total - 1, n).round()), with metadata."""
    import av
    c = av.open(path)
    s = c.streams.video[0]
    fps = float(s.average_rate)
    frames = [f for f in c.decode(s)]
    c.close()
    total = len(frames)
    idx = np.round(np.linspace(0, total - 1, n)).astype(int)
    arr = np.stack([frames[i].to_ndarray(format="rgb24") for i in idx])
    return arr, {"fps": fps, "frames_indices": idx.tolist(), "total_num_frames": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["vsibench", "mindcube"])
    ap.add_argument("--name", required=True, help="model label in the output file: base, run0, run1, ...")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--adapter")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--frames", type=int, default=VSI_FRAMES)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="results/std_bench")
    ap.add_argument("--dry-run", type=int)
    args = ap.parse_args()

    Q = load_questions(args.bench)
    k, n = map(int, args.shard.split("/"))
    clusters = sorted({q["cluster"] for q in Q})
    mine = set(clusters[k::n])
    Q = [q for q in Q if q["cluster"] in mine]
    os.makedirs(f"{args.out}/{args.bench}", exist_ok=True)
    path = f"{args.out}/{args.bench}/{args.name}{'-blind' if args.blind else ''}.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    Q = [q for q in Q if q["id"] not in done]
    if args.limit:
        Q = Q[:args.limit]
    print(f"{args.bench} {args.name}: {len(Q)} questions to answer ({len(done)} already) -> {path}", flush=True)

    vsi = args.bench == "vsibench"
    cache = {}

    def visual(q):
        """('video', frames, metadata) or ('images', [PIL images]) for one question; None when blind."""
        if args.blind:
            return None
        if vsi:
            if q["cluster"] not in cache:
                cache.clear()
                cache[q["cluster"]] = video_frames(f"{VSI}/videos/{q['cluster']}.mp4", args.frames)
            return ("video",) + cache[q["cluster"]]
        from PIL import Image
        return ("images", [Image.open(f"{MC}/{p}").convert("RGB") for p in q["images"]])

    def messages(q, vis):
        text = vsi_text(q) if vsi else mc_text(q)
        content = []
        if vis is not None:
            content = [{"type": "video"}] if vis[0] == "video" else [{"type": "image"} for _ in vis[1]]
        system = [{"role": "system", "content": [{"type": "text", "text": VSI_SYSTEM}]}] if vsi else []
        return system + [{"role": "user", "content": content + [{"type": "text", "text": text}]}]

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.video_utils import VideoMetadata

    proc = AutoProcessor.from_pretrained(args.model)
    if not vsi:
        proc.image_processor.size = {"longest_edge": MC_MAX_PIXELS, "shortest_edge": proc.image_processor.size["shortest_edge"]}

    def build(q):
        vis = visual(q)
        text = proc.apply_chat_template(messages(q, vis), tokenize=False, add_generation_prompt=True)
        kw = {}
        if vis is not None and vis[0] == "video":
            # frames already sampled and at the size qwen-vl-utils gives them (native 640x480), as lmms-eval
            kw = dict(videos=[vis[1]], video_metadata=[VideoMetadata(**vis[2])], do_sample_frames=False,
                      do_resize=False)
        elif vis is not None:
            kw = dict(images=vis[1])
        return vis, proc(text=[text], return_tensors="pt", **kw)

    if args.dry_run:
        for q in Q[:args.dry_run]:
            vis, inputs = build(q)
            desc = ("no visual input" if vis is None else
                    f"video {vis[1].shape}, frames {vis[2]['frames_indices'][:3]}...{vis[2]['frames_indices'][-1]} of "
                    f"{vis[2]['total_num_frames']} at {vis[2]['fps']} fps, grid {inputs['video_grid_thw'].tolist()}"
                    if vis[0] == "video" else
                    f"{len(vis[1])} images {[im.size for im in vis[1]]}, grids {inputs['image_grid_thw'].tolist()}")
            gt = q.get("ground_truth", q.get("gt_answer"))
            shown = proc.tokenizer.decode(inputs["input_ids"][0])
            shown = re.sub(r"(<\|(?:video|image)_pad\|>)+", lambda m: f"[{len(m.group(0)) // len(m.group(1))} pad]", shown)
            print(f"--- {q['id']} ({q.get('question_type', q.get('setting'))}, {q['cluster']}), {desc}, "
                  f"{inputs['input_ids'].shape[1]} tokens, answer {gt!r}")
            print(shown)
        return

    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device,
                                                        attn_implementation="sdpa").eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
    dev = torch.mps if args.device == "mps" else torch.cuda
    max_new = VSI_MAX_NEW if vsi else MC_MAX_NEW
    t0 = time.time()
    for j, q in enumerate(Q):
        vis, inputs = build(q)
        inputs = inputs.to(args.device)
        P = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        gen = out[0, P:]
        if len(gen) and bool((gen == 0).all()):
            # token 0 ("!") at every step is what greedy decoding gives when the logits are not finite: a
            # GPU fault (it hit two whole VSI-Bench runs on vast-3 on 2026-10-27). Stop rather than record it.
            raise SystemExit(f"{q['id']}: every generated token is 0 (non-finite logits?); stopping without saving")
        reply = proc.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        parsed, score, lenient = vsi_score(q, reply) if vsi else mc_score(q, reply)
        rec = {k: q[k] for k in ("id", "cluster", "question_type", "setting", "pruned", "ground_truth", "gt_answer")
               if k in q}
        rec.update(name=args.name, adapter=args.adapter, blind=args.blind, device=args.device, parsed=parsed,
                   score=score, score_lenient=lenient, prompt_tokens=P, gen_tokens=int(len(gen)),
                   n_frames=args.frames if vsi and vis is not None else None, text=reply[-2000:])
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if j == 0 or (j + 1) % 50 == 0:
            mem = (dev.driver_allocated_memory() if args.device == "mps" else dev.max_memory_allocated()) / 2**30
            print(f"{j + 1}/{len(Q)}, {time.time() - t0:.0f}s, prompt {P} tokens, {mem:.0f} GiB, e.g. {reply[:80]!r} "
                  f"-> {parsed} ({score:.2f})", flush=True)
        if args.max_minutes and time.time() - t0 > 60 * args.max_minutes:
            print(f"stopping after {args.max_minutes} min ({j + 1} answered this run)", flush=True)
            break
    print(f"{args.name}: done, {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
