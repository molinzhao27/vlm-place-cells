"""Two-choice test on Claude through the API: behaviour only (no access to activations).

    .venv-claude/bin/python -m spatialcells.claude_test --dry-run            # exact input size, no spend
    .venv-claude/bin/python -m spatialcells.claude_test --walks 1             # one walk, to see the cost
    .venv-claude/bin/python -m spatialcells.claude_test --budget-usd 40       # the rest (resumes)
    .venv-claude/bin/python -m spatialcells.claude_test --score
    .venv-claude/bin/python -m spatialcells.claude_test --no-walk [--score]   # control: no walk shown
    .venv-claude/bin/python -m spatialcells.claude_test --trials dissimilar [--no-walk] [--effort low|none]
    .venv-claude/bin/python -m spatialcells.claude_test --trials dissimilar --effort low --shuffle-walk [--walk-frames 64]

Uses results/export/claude from export_trials.py: the same 104 novel two-choice trials the
VLMs and VGGT got, plus 3 familiar trials per walk. The prompt matches the VLM test: the
walk's 256 frames labelled "Frame 001:" ... "Frame 256:", then "New view:", the probe and
the question. The walk is the same for every question about it and is cached, so each
extra question costs little input. Thinking is adaptive (always on for Opus 5.5 and Fable
5.1), so this is the with-reasoning condition. The key comes from ANTHROPIC_API_KEY or
~/.config/anthropic/api_key and is never printed. A request that could push the estimated
spend past --budget-usd is not sent. Appends to results/claude/<model>.jsonl; reruns skip
trials already answered. --no-walk is the control: only the two option frames (labelled with
their frame numbers), the new view and the same question, on the novel trials; written
to results/claude/<model>-nowalk.jsonl. --trials dissimilar uses the lure-free trials
(dissimilar_trials.py; all 192, with or without the walk). --effort sets the thinking effort;
"none" asks for thinking disabled (which some models refuse). Names: <model>[-dissimilar]
[-nowalk][-<effort> unless high].
--shuffle-walk shows each walk's frames in a fixed random order (seeded by the walk) with their
original "Frame k" labels and the same prompt text, so only the order in context changes
(tag -shuffled). --relabel (with --shuffle-walk) also relabels the shown frames 1, 2, ... in
the shuffled order, and the question names the two options by their new labels, so no path order
is left in the labels either (tag -shuffled-relabel; answers are stored in original numbers).
--walk-frames 64 shows 64 of the 256 frames, in order: every option frame of
that walk's trials plus frames spread evenly in time (farthest-point in frame number) (tag -f64).
--trials room<R>: the lure-free trials in room R (dissimilar_trials.py --room R; tag -room<R>): roomB,
or a procedural room such as roomP5000 (experiment 10). Its walk frames and probes are written as
JPEGs (q92, as export_trials.py) from data/walks-room<R> on first use, to results/export/claude-room<R>/
and results/export/dissimilar-room<R>/.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

EXPORT = "results/export/claude"          # walk frames (and the original trials' probes); set from --trials
TRIAL_SETS = {"original": "results/export/claude", "dissimilar": "results/export/dissimilar"}


def trial_dir(name):
    """Folder with the trial set's trials.json and probe JPEGs: original, dissimilar or room<R>."""
    return TRIAL_SETS.get(name) or f"results/export/dissimilar-{name}"


def walk_set(name):
    """(walks folder, walk-frame JPEG export) for a room other than A, else None."""
    return (f"data/walks-{name}", f"results/export/claude-{name}") if name.startswith("room") else None


def trial_set(name):
    if name not in TRIAL_SETS and not re.fullmatch(r"room[A-Z][A-Za-z0-9]*", name):
        raise argparse.ArgumentTypeError(f"--trials: original, dissimilar or room<R>, not {name!r}")
    return name
PROBES = EXPORT                           # set from --trials in main()
OUT = "results/claude"
KEY_FILE = os.path.expanduser("~/.config/anthropic/api_key")
# $ per million tokens (input, output) and the cache-read multiplier, from the API docs (Oct 2026)
PRICES = {"claude-opus-5-5": (4, 20, 0.05), "claude-sonnet-5-5": (2, 10, 0.05),
          "claude-fable-5-1": (10, 50, 0.025), "claude-haiku-5-5": (0.10, 0.50, 0.1)}
CACHE_WRITE = 1.25
THINKING_OFF = {"claude-sonnet-5-5": {"type": "between_tools"}}   # Sonnet 5.5 rejects "disabled"; Opus 5.5 has no off
WALK_PROMPT = ("These are consecutive first-person views as you walk around a room. "
               "Each view is labelled with its frame number.")
QUESTION = ("This new view was taken somewhere in the same room, possibly facing a different direction. "
            "Was it taken closer to where Frame {a:03d} was taken, or to where Frame {b:03d} was taken? "
            "Answer with one of the two frame numbers only.")


def image(path):
    data = base64.standard_b64encode(open(path, "rb").read()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def export_jpegs(trials, data):
    """Write the walk frames and probes the trials use as JPEGs, if not there yet."""
    import numpy as np
    from PIL import Image
    meta = np.load(f"{data}/meta.npz")
    jpg = lambda a, path: os.path.exists(path) or Image.fromarray(np.asarray(a)).save(path, quality=92)
    frames = np.load(f"{data}/frames.npy", mmap_mode="r")
    for w in sorted({t["walk"] for t in trials}):
        os.makedirs(f"{EXPORT}/walk_{w:02d}", exist_ok=True)
        i = np.flatnonzero(meta["walk"] == w)
        for k, f in enumerate(i[np.argsort(meta["step"][i])], 1):
            jpg(frames[f], f"{EXPORT}/walk_{w:02d}/frame_{k:03d}.jpg")
    probes = np.load(f"{data}/probes.npy", mmap_mode="r")
    for t in trials:
        jpg(probes[t["probe"]], f"{PROBES}/probe_{t['probe']:04d}.jpg")


def walk_frames(w, trials, n=256, shuffle=False):
    """Frame numbers shown for walk w, in the order they are shown."""
    frames = list(range(1, 257))
    if n < 256:
        chosen = sorted({k for t in trials if t["walk"] == w for k in (t["a"], t["b"])})
        while len(chosen) < n:
            chosen.append(max((k for k in frames if k not in chosen), key=lambda k: min(abs(k - c) for c in chosen)))
        frames = sorted(chosen)
    if shuffle:
        random.Random(w).shuffle(frames)
    return frames


def walk_blocks(w, frames=range(1, 257), relabel=False):
    blocks = [{"type": "text", "text": WALK_PROMPT}]
    for i, k in enumerate(frames):
        blocks += [{"type": "text", "text": f"Frame {i + 1 if relabel else k:03d}:"}, image(f"{EXPORT}/walk_{w:02d}/frame_{k:03d}.jpg")]
    blocks[-1]["cache_control"] = {"type": "ephemeral"}   # everything up to here is shared by the walk's questions
    return blocks


def nowalk_blocks(t):
    """The control's context: just the two option frames, in the order the question names them."""
    blocks = [{"type": "text", "text": "These are two first-person views from a walk around a room, "
                                       "labelled with their frame numbers."}]
    for k in (t["a"], t["b"]):
        blocks += [{"type": "text", "text": f"Frame {k:03d}:"}, image(f"{EXPORT}/walk_{t['walk']:02d}/frame_{k:03d}.jpg")]
    return blocks


def question_blocks(t, lab=None):
    a, b = (lab[t["a"]], lab[t["b"]]) if lab else (t["a"], t["b"])
    return [{"type": "text", "text": "New view:"}, image(f"{PROBES}/probe_{t['probe']:04d}.jpg"),
            {"type": "text", "text": QUESTION.format(a=a, b=b)}]


def cost(model, u):
    pin, pout, read = PRICES[model]
    return ((u.get("input_tokens", 0) + CACHE_WRITE * u.get("cache_creation_input_tokens", 0)
             + read * u.get("cache_read_input_tokens", 0)) * pin + u.get("output_tokens", 0) * pout) / 1e6


def tag(args):
    return (args.model + ({"dissimilar": "-dissimilar"}.get(args.trials, f"-{args.trials}" if walk_set(args.trials) else "")) + ("-nowalk" if args.no_walk else "")
            + ("-shuffled" if args.shuffle_walk else "") + ("-relabel" if args.relabel else "") + (f"-f{args.walk_frames}" if args.walk_frames < 256 else "")
            + ("" if args.effort == "high" else f"-{args.effort}"))


def client():
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key and os.path.exists(KEY_FILE):
        key = open(KEY_FILE).read().strip()
    if not key:
        sys.exit(f"No API key: set ANTHROPIC_API_KEY or save the key to {KEY_FILE}")
    return anthropic.Anthropic(api_key=key, max_retries=6)


def run(args):
    trials = json.load(open(f"{PROBES}/trials.json"))
    if walk_set(args.trials):
        export_jpegs(trials, walk_set(args.trials)[0])
    if args.no_walk and args.trials == "original":
        trials = [t for t in trials if not t["familiar"]]
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{tag(args)}.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    spent = sum(json.loads(l).get("cost_usd", 0) for l in open(path)) if os.path.exists(path) else 0.0
    c = client()
    walks = sorted({t["walk"] for t in trials})[:args.walks]
    kw = dict(model=args.model, max_tokens=args.max_tokens,
              thinking={"type": "adaptive", "display": "summarized"}, output_config={"effort": args.effort})
    if args.effort == "none":
        kw = dict(model=args.model, max_tokens=args.max_tokens, thinking=THINKING_OFF.get(args.model, {"type": "disabled"}))

    if args.dry_run:
        t = next(t for t in trials if t["walk"] == walks[0])
        frames = walk_frames(t["walk"], trials, args.walk_frames, args.shuffle_walk)
        print(f"walk {t['walk']}: frames shown {frames[:12]}...")
        msgs = [{"role": "user", "content": walk_blocks(t["walk"], frames) + question_blocks(t)}]
        n = c.messages.count_tokens(model=args.model, messages=msgs).input_tokens
        size = len(json.dumps(msgs)) / 2**20
        pin, pout, read = PRICES[args.model]
        print(f"{args.model}: one question = {n} input tokens, request {size:.1f} MB. Per walk: first question "
              f"~${n * CACHE_WRITE * pin / 1e6:.2f} input, later ones ~${n * read * pin / 1e6:.3f}; "
              f"output ${pout}/M tokens (thinking included)")
        return

    lock = threading.Lock()
    labels = {}                                                  # walk -> original frame number -> label shown
    worst = lambda: (30000 * CACHE_WRITE * PRICES[args.model][0] + args.max_tokens * PRICES[args.model][1]) / 1e6
    print(f"{args.model}, effort {args.effort}: {len(done)} trials already answered, ${spent:.2f} spent so far", flush=True)

    def ask(t, prefix):
        nonlocal spent
        t0 = time.time()
        context = nowalk_blocks(t) if prefix is None else prefix
        lab = labels.get(t["walk"]) if args.relabel else None
        r = c.messages.create(messages=[{"role": "user", "content": context + question_blocks(t, lab)}], **kw)
        text = "".join(b.text for b in r.content if b.type == "text")
        thinking = "".join(b.thinking for b in r.content if b.type == "thinking")
        u = r.usage.model_dump() if hasattr(r.usage, "model_dump") else dict(r.usage)
        nums = [int(x) for x in re.findall(r"\d+", text)]
        if lab:                                                  # back to original frame numbers
            back = {lab[t["a"]]: t["a"], lab[t["b"]]: t["b"]}
            nums = [back[x] for x in nums if x in back]
        ans = next((x for x in nums if x in (t["a"], t["b"])), -1)
        rec = {"id": t["id"], "walk": t["walk"], "probe": t["probe"], "familiar": t["familiar"], "near": t["near"],
               "far": t["far"], "a": t["a"], "b": t["b"], "answer": ans, "correct": ans == t["near"],
               "text": text, "thinking": thinking, "stop_reason": r.stop_reason, "usage": u,
               "cost_usd": cost(args.model, u), "model": args.model, "effort": args.effort, "walk_shown": not args.no_walk,
               "shuffle_walk": args.shuffle_walk, "walk_frames": args.walk_frames, "relabel": args.relabel,
               **({"labels_shown": [lab[t["a"]], lab[t["b"]]]} if lab else {}),
               "seconds": round(time.time() - t0, 1)}
        with lock:
            spent += rec["cost_usd"]
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    if args.no_walk:   # no shared prefix, so no cache: send in waves
        todo = [t for t in trials if t["walk"] in walks and t["id"] not in done]
        for i in range(0, len(todo), args.concurrency):
            wave = todo[i:i + args.concurrency]
            if spent + len(wave) * (2000 * PRICES[args.model][0] + args.max_tokens * PRICES[args.model][1]) / 1e6 > args.budget_usd:
                print(f"stopping: ${spent:.2f} spent, budget ${args.budget_usd}", flush=True)
                return
            with ThreadPoolExecutor(len(wave)) as ex:
                recs = list(ex.map(lambda t: ask(t, None), wave))
            print(f"{i + len(wave)}/{len(todo)}: {sum(r['correct'] for r in recs)}/{len(recs)} right in this wave, "
                  f"total ${spent:.2f}", flush=True)
        return

    for w in walks:
        todo = [t for t in trials if t["walk"] == w and t["id"] not in done]
        if not todo:
            continue
        frames = walk_frames(w, trials, args.walk_frames, args.shuffle_walk)
        labels[w] = {k: i + 1 for i, k in enumerate(frames)}
        prefix = walk_blocks(w, frames, args.relabel)
        if spent + worst() > args.budget_usd:
            print(f"stopping before walk {w}: ${spent:.2f} spent, budget ${args.budget_usd}", flush=True)
            break
        first = ask(todo[0], prefix)   # writes the cache; the rest of the walk reads it
        recs = [first]
        rest = todo[1:]
        for i in range(0, len(rest), args.concurrency):
            wave = rest[i:i + args.concurrency]
            if spent + len(wave) * worst() > args.budget_usd:
                print(f"stopping in walk {w}: ${spent:.2f} spent, budget ${args.budget_usd}", flush=True)
                return
            with ThreadPoolExecutor(len(wave)) as ex:
                recs += list(ex.map(lambda t: ask(t, prefix), wave))
        u = [r["usage"] for r in recs]
        print(f"walk {w}: {len(recs)} trials, {sum(r['correct'] for r in recs)} right, "
              f"cache read {sum(x.get('cache_read_input_tokens', 0) for x in u)} / written "
              f"{sum(x.get('cache_creation_input_tokens', 0) for x in u)} tokens, "
              f"median {sorted(r['seconds'] for r in recs)[len(recs) // 2]}s, total ${spent:.2f}", flush=True)


def score(args):
    trials = {t["id"]: t for t in json.load(open(f"{PROBES}/trials.json"))}
    human = set() if walk_set(args.trials) else {t["id"] for t in json.load(open("results/export/human/trials.json"))["trials"]}
    recs = [json.loads(l) for l in open(f"{OUT}/{tag(args)}.jsonl")]
    out = {}
    for cat in ("familiar", "novel"):
        rs = [r for r in recs if r["familiar"] == (cat == "familiar")]
        if not rs:
            continue
        valid = [r for r in rs if r["answer"] > 0]
        t = lambda r: trials[r["id"]]
        aligned = lambda r: r["near"] if t(r)["near_heading_diff_deg"] < t(r)["far_heading_diff_deg"] else r["far"]
        pf = [r for r in valid if not t(r)["pixels_favour_near"]]
        out[f"choice/reasoning/{cat}"] = {
            "n": len(rs), "valid": len(valid) / len(rs),
            "accuracy": sum(r["correct"] for r in valid) / max(len(valid), 1),
            "pixel_match_accuracy": sum(t(r)["pixels_favour_near"] for r in rs) / len(rs),
            "accuracy_when_pixels_favour_far": sum(r["correct"] for r in pf) / max(len(pf), 1),
            "picked_option_facing_like_probe": sum(r["answer"] == aligned(r) for r in valid) / max(len(valid), 1),
            "human_page_trials": [sum(r["correct"] for r in valid if r["id"] in human),
                                  sum(1 for r in rs if r["id"] in human)],
            "median_output_tokens": sorted(r["usage"].get("output_tokens", 0) for r in rs)[len(rs) // 2],
            **({"accuracy_when_frame_order_favours_far": (lambda fo: sum(r["correct"] for r in fo) / max(len(fo), 1))(
                [r for r in valid if abs(r["far"] - t(r)["best_match_frame"]) <= abs(r["near"] - t(r)["best_match_frame"])])}
               if "best_match_frame" in next(iter(trials.values())) else {}),
            "cost_usd": round(sum(r["cost_usd"] for r in rs), 2)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{OUT}/{tag(args)}_summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5", choices=list(PRICES))
    ap.add_argument("--effort", default="high", choices=["none", "low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--trials", default="original", type=trial_set, help="original, dissimilar or room<R>")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--walks", type=int, help="only the first N walks that have trials (default: all of them)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--budget-usd", type=float, default=10)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-walk", action="store_true", help="control: show only the two option frames")
    ap.add_argument("--shuffle-walk", action="store_true", help="walk frames in a fixed random order, original labels")
    ap.add_argument("--relabel", action="store_true", help="with --shuffle-walk: label frames in shown order")
    ap.add_argument("--walk-frames", type=int, default=256, help="show this many walk frames (always incl. the options)")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    global PROBES, EXPORT
    PROBES = trial_dir(args.trials)
    EXPORT = (walk_set(args.trials) or (None, EXPORT))[1]
    score(args) if args.score else run(args)


if __name__ == "__main__":
    main()
