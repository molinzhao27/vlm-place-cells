"""The Claude two-choice test (claude_test.py) for other providers' models, through OpenAI-style
chat-completions APIs (experiment 11). The prompt is claude_test.py's, built by its own functions:
the walk's labelled frames, "New view:", the probe and the question, with the same JPEGs. Only the
message format differs.

    .venv-claude/bin/python -m spatialcells.api_test --model kimi-k3 --preflight     # one 257-image request
    .venv-claude/bin/python -m spatialcells.api_test --model kimi-k3 --trials dissimilar [--no-walk] [--shuffle-walk]
    .venv-claude/bin/python -m spatialcells.api_test --model kimi-k3 --trials dissimilar --score

Two decision APIs are also covered (experiment 11's fast-decision arm). They return one choice
from a fixed set, with probabilities, instead of text: OpenAI's Decisions API (gpt-6-luna,
POST /v1/decisions) and TypeSafe's Jev (POST api.typesafe.ai/v1/systemone). They get the same
content in the same order. The walk is the shared input. "New view:", the probe and the question
are in the choice question's instructions, minus its last sentence ("Answer with one of the two
frame numbers only"), which becomes the two choices "Frame <a>" and "Frame <b>". Records add the
probabilities and p_near.
Caching is the provider's own (implicit prefix caching). As in claude_test.py, each walk's first
question is sent alone so that it writes the walk prefix, and the walk's other questions then read
it, in waves of --concurrency. --preflight sends one question with the full walk, to check that the
provider takes 257 images. It reports tokens per image and the cost, and the record is not kept.
Keys come from <PROVIDER>_API_KEY or ~/.config/<provider>/api_key. Appends to
results/api/<tag>.jsonl (tags as claude_test.py: <model>[-dissimilar|-room<R>][-nowalk][-shuffled]
[-f64]-<effort>), with each trial's cost_usd for scripts/cost_log.py. Reruns skip trials already
answered, and nothing is sent once --budget-usd would be passed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import claude_test as ct

OUT = "results/api"
PROVIDERS = {"moonshot": "https://api.moonshot.ai/v1", "deepinfra": "https://api.deepinfra.com/v1/openai",
             "deepseek": "https://api.deepseek.com", "openai": "https://api.openai.com/v1",
             "typesafe": "https://api.typesafe.ai/v1", "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/"}
TOKEN_PARAM = {"gemini": "max_tokens"}         # the output-cap parameter, if not max_completion_tokens
DECISION_APIS = {"openai": "/decisions", "typesafe": "/systemone"}      # providers whose models are decision models
# model: (provider, $ per million tokens (input miss, cache write, cache hit, output; output includes
# reasoning), request settings, the effort that names its runs). Each entry is checked against the
# provider's docs and a --preflight before use.
MODELS = {
    # platform.kimi.ai docs, Oct 2026: thinking always on (reasoning_effort low|high|max);
    # temperature fixed at 1.0 (must be omitted); implicit prefix cache, 5-min writes at the miss
    # price. Output price from the research summary (Oct 2026); check it against the bill.
    "kimi-k3": ("moonshot", (3.0, 3.0, 0.30, 15.0), {"reasoning_effort": "low"}, "low"),
    # api-docs.deepseek.com, Oct 2026: V4.1 Flash takes up to 600 images (48 MiB); small images are
    # upscaled server-side to about 544 x 544 (at most 1,024 tokens each). Thinking on by default;
    # reasoning_effort low|high|max; temperature has no effect in thinking mode. Automatic prefix
    # caching with no write charge. Peak prices below; off-peak (outside 01-04 and 06-10 UTC on
    # weekdays) is half, which cost() applies from each request's time.
    "deepseek-flash": ("deepseek", (0.30, 0.30, 0.006, 1.20),
                       {"reasoning_effort": "low", "extra_body": {"thinking": {"type": "enabled"}}}, "low"),
    # developers.openai.com/api/docs/models/gpt-6-astra (Oct 2026): chat completions supported;
    # reasoning effort low|medium|high|xhigh|max (no none); temperature not listed, so not sent.
    # Prompts over 272K input tokens bill at 2x input and cache, 1.5x output (cost() applies it).
    "gpt-6-astra": ("openai", (10.0, 12.5, 1.0, 50.0), {"reasoning_effort": "low"}, "low"),
    # ai.google.dev/gemini-api/docs/models/gemini-4-argon and /openai (Oct 2026): image input; thinking
    # low|medium|high, reasoning_effort "low" maps to thinking_level low; implicit caching only. Price
    # from launch coverage ($2 / $0.10 cached / $10 output, thinking billed as output), doubling on
    # 2026-12-01 (PRICE_CHANGES); check against the bill.
    "gemini-4-argon": ("gemini", (2.0, 2.0, 0.10, 10.0), {"reasoning_effort": "low"}, "low"),
    # Decision models (no reasoning setting, output free). Costs are upper bounds: cache hits are
    # charged at the full input price. developers.openai.com/api/docs/guides/decisions (Oct 2026):
    # $0.10/M input, implicit cache breakpoint at the end of the shared input.
    "gpt-6-luna": ("openai", (0.10, 0.125, 0.10, 0.0), {}, "decision"),
    # docs.typesafe.ai/api (Oct 2026): images need TypeSafe-Danger-Zone-Images (experimental in
    # 1.14). Price $0.042/M input from launch coverage; check it against the bill.
    "jev-1.14.0": ("typesafe", (0.042, 0.042, 0.042, 0.0), {}, "decision"),
}
PRICE_CHANGES = {"gemini-4-argon": ("2026-12-01", (4.0, 4.0, 0.20, 20.0))}    # new prices from that UTC date


def client(provider):
    from openai import OpenAI
    return OpenAI(api_key=api_key(provider), base_url=PROVIDERS[provider], max_retries=6, timeout=600)


def parts(blocks):
    """claude_test.py content blocks -> chat-completions content parts (same order, same images)."""
    out = []
    for b in blocks:
        if b["type"] == "text":
            out.append({"type": "text", "text": b["text"]})
        else:
            s = b["source"]
            out.append({"type": "image_url", "image_url": {"url": f"data:{s['media_type']};base64,{s['data']}"}})
    return out


def usage(r):
    u = r.usage.model_dump() if r.usage else {}
    d = u.get("prompt_tokens_details") or {}
    cached = d.get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0
    written = d.get("cache_write_tokens") or d.get("cache_creation_tokens") or d.get("cache_creation_input_tokens") or 0
    prompt = u.get("prompt_tokens", 0)
    output = max(u.get("completion_tokens", 0), (u.get("total_tokens") or 0) - prompt)   # thinking billed but not in completion_tokens
    return {"prompt": prompt, "cache_read": cached, "cache_write": written,
            "miss": prompt - cached - written, "output": output,
            "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"), "raw": u}


def api_key(provider):
    env, path = f"{provider.upper()}_API_KEY", os.path.expanduser(f"~/.config/{provider}/api_key")
    key = os.environ.get(env) or (open(path).read().strip() if os.path.exists(path) else None)
    if not key:
        sys.exit(f"No API key: set {env} or save the key to {path}")
    return key


def post(url, key, body, headers=None, tries=6):
    """POST JSON with retries on 429 and 5xx; returns the decoded response."""
    import httpx
    for i in range(tries):
        r = httpx.post(url, json=body, timeout=600, headers={"Authorization": f"Bearer {key}", **(headers or {})})
        if r.status_code < 400:
            return r.json()
        if r.status_code != 429 and r.status_code < 500 or i == tries - 1:
            sys.exit(f"{url}: HTTP {r.status_code}: {r.text[:500]}")
        time.sleep(min(60, 2 ** i * 2))


def question_text(t):
    """The prompt's question without its answer-format sentence (the choices replace it)."""
    return ct.QUESTION.format(a=t["a"], b=t["b"]).replace(" Answer with one of the two frame numbers only.", "")


def choices(t):
    return {f"Frame {k:03d}": f"It was taken closer to where Frame {k:03d} was taken." for k in (t["a"], t["b"])}


def decide(model, key, context, t):
    """Ask a decision model the trial's question. Returns (answer frame or -1, {frame: probability},
    confidence, usage dict, raw response)."""
    provider = MODELS[model][0]
    probe = ct.question_blocks(t)[:2]                                  # "New view:" and the probe image
    if provider == "openai":
        part = lambda b: ({"type": "input_text", "text": b["text"]} if b["type"] == "text" else
                          {"type": "input_image", "image_url": f"data:{b['source']['media_type']};base64,{b['source']['data']}"})
        body = {"model": model, "input": [{"role": "user", "content": [part(b) for b in context]}],
                "questions": [{"type": "choice", "name": "closer",
                               "instructions": [part(b) for b in probe] + [{"type": "input_text", "text": question_text(t)}],
                               "choices": [{"value": v, "description": d} for v, d in choices(t).items()]}]}
        r = post(PROVIDERS[provider] + DECISION_APIS[provider], key, body)
        answers = r.get("answers") or r.get("output") or []
        a = next((x for x in answers if x.get("name") == "closer"), answers[0] if answers else {}) if isinstance(answers, list) else answers.get("closer", {})
        u = r.get("usage") or {}
        d = u.get("input_tokens_details") or u
        cached, written = d.get("cached_tokens") or 0, d.get("cache_write_tokens") or 0
    else:                                                              # TypeSafe Jev: images as assets
        assets, lines = {}, []
        for b in context + probe:
            if b["type"] == "text":
                lines.append(b["text"])
            else:
                name = f"[[image_{len(assets):03d}]]"
                assets[name] = {"type": "image", "format": "jpg", "data_base64": b["source"]["data"]}
                lines[-1] = f"{lines[-1]} {name}" if lines else name
        new_view = lines.pop()                                         # "New view: [[image_N]]" goes with the question
        body = {"model": model, "state": "\n".join(lines), "assets": assets,
                "questions": {"closer": {"type": "choice", "instructions": f"{new_view}\n{question_text(t)}",
                                         "criteria": choices(t)}}}
        r = post(PROVIDERS[provider] + DECISION_APIS[provider], key, body, {"TypeSafe-Danger-Zone-Images": "true"})
        a = (r.get("answers") or {}).get("closer", {})
        u = r.get("usage") or {}
        cached, written = 0, 0
    probs = a.get("probabilities") or {}
    if isinstance(probs, list):
        probs = {p.get("value") or p.get("name"): p.get("probability") for p in probs}
    frame = lambda name: int(re.findall(r"\d+", str(name))[-1]) if re.findall(r"\d+", str(name)) else -1
    ans = frame(a.get("choice")) if a.get("type", "choice") != "refusal" else -1
    inp = u.get("input_tokens", 0)
    usage = {"prompt": inp, "cache_read": cached, "cache_write": written, "miss": inp - cached - written,
             "output": u.get("output_tokens", 0), "reasoning": None, "raw": u}
    return ans if ans in (t["a"], t["b"]) else -1, {frame(k): p for k, p in probs.items()}, a.get("confidence"), usage, r


def offpeak(model, when=None):
    """DeepSeek charges half outside 01-04 and 06-10 UTC on weekdays (Chinese holidays ignored)."""
    if MODELS[model][0] != "deepseek":
        return False
    t = time.gmtime(when)
    return t.tm_wday >= 5 or not (1 <= t.tm_hour < 4 or 6 <= t.tm_hour < 10)


def rates(model, when=None):
    p = MODELS[model][1]
    if model in PRICE_CHANGES and time.strftime("%Y-%m-%d", time.gmtime(when)) >= PRICE_CHANGES[model][0]:
        p = PRICE_CHANGES[model][1]
    return p


def cost(model, u, when=None):
    pin, pwrite, phit, pout = rates(model, when)
    k_in, k_out = (2.0, 1.5) if model == "gpt-6-astra" and u["prompt"] > 272_000 else (1.0, 1.0)
    usd = (k_in * (u["miss"] * pin + u["cache_write"] * pwrite + u["cache_read"] * phit) + k_out * u["output"] * pout) / 1e6
    return usd / 2 if offpeak(model, when) else usd


def tag(args):
    return (args.model + ({"dissimilar": "-dissimilar"}.get(args.trials, f"-{args.trials}" if ct.walk_set(args.trials) else ""))
            + ("-nowalk" if args.no_walk else "") + ("-shuffled" if args.shuffle_walk else "")
            + (f"-f{args.walk_frames}" if args.walk_frames < 256 else "") + f"-{MODELS[args.model][3]}")


def run(args):
    provider, prices, settings, _ = MODELS[args.model]
    trials = json.load(open(f"{ct.PROBES}/trials.json"))
    if ct.walk_set(args.trials):
        ct.export_jpegs(trials, ct.walk_set(args.trials)[0])
    if args.no_walk and args.trials == "original":
        trials = [t for t in trials if not t["familiar"]]
    deciding = MODELS[args.model][3] == "decision"
    c = None if deciding else client(provider)
    key = api_key(provider) if deciding else None
    kw = dict(model=args.model, **{TOKEN_PARAM.get(provider, "max_completion_tokens"): args.max_tokens}, **settings)

    def ask(t, prefix):
        t0 = time.time()
        context = ct.nowalk_blocks(t) if prefix is None else prefix
        if deciding:
            ans, probs, conf, u, raw = decide(args.model, key, context, t)
            return {"id": t["id"], "walk": t["walk"], "probe": t["probe"], "familiar": t["familiar"], "near": t["near"],
                    "far": t["far"], "a": t["a"], "b": t["b"], "answer": ans, "correct": ans == t["near"],
                    "probabilities": probs, "p_near": probs.get(t["near"]), "confidence": conf,
                    "model_served": raw.get("model"), "usage": u, "cost_usd": cost(args.model, u, t0),
                    "model": args.model, "walk_shown": not args.no_walk, "shuffle_walk": args.shuffle_walk,
                    "walk_frames": args.walk_frames, "seconds": round(time.time() - t0, 1),
                    **({"raw": raw} if args.preflight else {})}
        extra = {"prompt_cache_key": f"{args.model}-walk{t['walk']}"} if provider == "openai" and prefix is not None else {}
        if provider == "openai" and args.service_tier:
            extra["service_tier"] = args.service_tier     # transport and price only; same request otherwise
        r = c.chat.completions.create(messages=[{"role": "user", "content": parts(context + ct.question_blocks(t))}], **kw, **extra)
        m = r.choices[0].message
        text = m.content or ""
        reasoning = getattr(m, "reasoning_content", None) or (m.model_extra or {}).get("reasoning_content") or ""
        nums = [int(x) for x in re.findall(r"\d+", text)]
        ans = next((x for x in nums if x in (t["a"], t["b"])), -1)
        u = usage(r)
        return {"id": t["id"], "walk": t["walk"], "probe": t["probe"], "familiar": t["familiar"], "near": t["near"],
                "far": t["far"], "a": t["a"], "b": t["b"], "answer": ans, "correct": ans == t["near"], "text": text[-600:],
                "reasoning": reasoning[-1500:], "finish_reason": r.choices[0].finish_reason, "usage": u,
                "cost_usd": cost(args.model, u, t0), "offpeak": offpeak(args.model, t0), "model": args.model,
                "model_served": getattr(r, "model", None), "service_tier": getattr(r, "service_tier", None),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                "settings": settings, "walk_shown": not args.no_walk,
                "shuffle_walk": args.shuffle_walk, "walk_frames": args.walk_frames, "seconds": round(time.time() - t0, 1)}

    if args.preflight:
        t = trials[0]
        frames = ct.walk_frames(t["walk"], trials, args.walk_frames, args.shuffle_walk)
        rec = ask(t, ct.walk_blocks(t["walk"], frames))
        u = rec["usage"]
        if deciding:
            print(f"{args.model}: accepted {len(frames) + 1} images; input {u['prompt']} tokens; answer {rec['answer']} "
                  f"({'right' if rec['correct'] else 'wrong'}), probabilities {rec['probabilities']}, confidence "
                  f"{rec['confidence']}, {rec['seconds']}s, ${rec['cost_usd']:.4f}. Not recorded.\nraw response: "
                  f"{json.dumps(rec['raw'])[:1500]}")
            return
        print(f"{args.model}: accepted {len(frames) + 1} images; prompt {u['prompt']} tokens (~{u['prompt'] / (len(frames) + 1):.0f} "
              f"per image incl. text), cache write {u['cache_write']}, read {u['cache_read']}; output {u['output']} "
              f"(reasoning {u['reasoning']}); answer {rec['answer']} ({'right' if rec['correct'] else 'wrong'}), "
              f"{rec['seconds']}s, ${rec['cost_usd']:.3f}. Not recorded.\nraw usage: {json.dumps(u['raw'])}")
        pin, pwrite, phit, pout = rates(args.model)
        n, nw, out = len(trials), len({x["walk"] for x in trials}), max(u["output"], 1)
        walk = (nw * u["prompt"] * pwrite + (n - nw) * u["prompt"] * phit + n * out * pout) / 1e6
        nowalk = n * (3 / (len(frames) + 1) * u["prompt"] * pin + out * pout) / 1e6
        print(f"projected, if every answer is as long as this one ({out} output tokens): walk run ${walk:.2f} "
              f"(cache hits as priced; ${(n * u['prompt'] * pin + n * out * pout) / 1e6:.2f} with none), "
              f"no-walk run ${nowalk:.2f}, for {n} trials over {nw} walks")
        return

    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{tag(args)}.jsonl"
    old = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    done, spent = {r["id"] for r in old}, sum(r["cost_usd"] for r in old)
    biggest = [max((r["usage"]["prompt"] for r in old), default=0)]

    def worst():
        """One question's cost if nothing is cached and the output runs to --max-tokens: the largest
        prompt seen so far in this run (+10%), or 300k tokens before the first answer."""
        return ((1.1 * biggest[0] if biggest[0] else 300_000) * prices[0] + args.max_tokens * prices[3]) / 1e6

    lock = threading.Lock()
    print(f"{args.model} ({settings}): {len(done)} trials already answered, ${spent:.2f} spent so far", flush=True)

    def keep(recs):
        nonlocal spent
        with lock, open(path, "a") as fh:
            for rec in recs:
                spent += rec["cost_usd"]
                biggest[0] = max(biggest[0], rec["usage"]["prompt"])
                fh.write(json.dumps(rec) + "\n")

    walks = sorted({t["walk"] for t in trials})[:args.walks]
    groups = ([[t for t in trials if t["walk"] in walks and t["id"] not in done]] if args.no_walk
              else [[t for t in trials if t["walk"] == w and t["id"] not in done] for w in walks])
    for todo in groups:
        if not todo:
            continue
        prefix = None
        if not args.no_walk:
            w = todo[0]["walk"]
            prefix = ct.walk_blocks(w, ct.walk_frames(w, trials, args.walk_frames, args.shuffle_walk))
            if spent + worst() > args.budget_usd:
                print(f"stopping before walk {w}: ${spent:.2f} spent, budget ${args.budget_usd}", flush=True)
                return
            keep([ask(todo[0], prefix)])                     # writes the walk prefix to the cache
            todo = todo[1:]
        for i in range(0, len(todo), args.concurrency):
            wave = todo[i:i + args.concurrency]
            if spent + len(wave) * worst() > args.budget_usd:
                print(f"stopping: ${spent:.2f} spent, budget ${args.budget_usd}", flush=True)
                return
            with ThreadPoolExecutor(len(wave)) as ex:
                recs = list(ex.map(lambda t: ask(t, prefix), wave))
            keep(recs)
        n = len(done) + sum(1 for _ in open(path)) - len(old)
        print(f"{n} answered, total ${spent:.2f}", flush=True)


def score(args):
    recs = [json.loads(line) for line in open(f"{OUT}/{tag(args)}.jsonl")]
    out = {}
    for cat in ("novel", "familiar"):
        rs = [r for r in recs if r["familiar"] == (cat == "familiar")]
        if rs:
            out[cat] = {"n": len(rs), "accuracy": sum(r["correct"] for r in rs) / len(rs),
                        "parsed": sum(r["answer"] > 0 for r in rs) / len(rs),
                        "picked_first_listed": sum(r["answer"] == r["a"] for r in rs if r["answer"] > 0) / max(1, sum(r["answer"] > 0 for r in rs)),
                        "median_output_tokens": sorted(r["usage"]["output"] for r in rs)[len(rs) // 2],
                        "cost_usd": round(sum(r["cost_usd"] for r in rs), 2)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{OUT}/{tag(args)}_summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--trials", default="dissimilar", type=ct.trial_set, help="original, dissimilar or room<R>")
    ap.add_argument("--max-tokens", type=int, default=16000, help="output cap, reasoning included")
    ap.add_argument("--walks", type=int, help="only the first N walks")
    ap.add_argument("--no-walk", action="store_true", help="control: show only the two option frames")
    ap.add_argument("--shuffle-walk", action="store_true", help="walk frames in a fixed random order, original labels")
    ap.add_argument("--walk-frames", type=int, default=256, help="show this many walk frames (always incl. the options)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--service-tier", choices=["flex"], help="OpenAI only: request the flex tier (batch pricing, may be "
                    "slower). Costs are still logged at the standard rate, an upper bound")
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--preflight", action="store_true", help="one full-walk request: is it accepted, and what does it cost")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    ct.PROBES = ct.trial_dir(args.trials)
    ct.EXPORT = (ct.walk_set(args.trials) or (None, "results/export/claude"))[1]
    score(args) if args.score else run(args)


if __name__ == "__main__":
    main()
