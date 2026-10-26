"""Experiment 11, amendment 5 (docs/experiments.md): screen the largest open-weight VLMs on OpenRouter,
one at a time, cheapest input price first, and stop at the first model that passes.

    .venv-claude/bin/python scripts/or_screen.py            # -> results/api/or_screen.json, log on stdout

Per model: a preflight (one 257-image request; retried on connection errors). A model that rejects the
request is untestable and the screen moves on. Otherwise the walk and no-walk runs (api_test.py, room A,
192 trials each), then the registered walk-gain sign test on novel trials. A model passes at two-sided
p < 0.05 / 8 (Bonferroni over the screen); it then gets the registered shuffled-walk run and the screen
stops. Before each run the projected cost is checked against what is left on the OpenRouter key
(its own limit); a model that does not fit is skipped. Runs are resumable, so the script can be rerun.
"""
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request

PY = ".venv-claude/bin/python"
MODELS = ["nvidia/nemotron-3.5-super-vl", "qwen/qwen3.8-flash", "minimax/minimax-m3", "xiaomi/mimo-v2.6-pro",
          "qwen/qwen3.5-397b-a17b", "stepfun/step-5-preview", "thinkingmachines/inkling", "meta/muse-spark-open"]
ALPHA = 0.05 / len(MODELS)
OUT = "results/api/or_screen.json"
CONCURRENCY = os.environ.get("CONCURRENCY", "2")


def redact(msg):
    """Provider error strings can carry the account's user id; drop it before the message is stored."""
    return re.sub(r"""['"]?user_id['"]?\s*:\s*['"][^'"]*['"]""", "'user_id': '<redacted>'", msg)


def key_left():
    k = open(os.path.expanduser("~/.config/openrouter/api_key")).read().strip()
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {k}"})
    d = json.load(urllib.request.urlopen(req, timeout=30))["data"]
    return d.get("limit_remaining"), d.get("usage")


def api(model, *args, tries=4):
    """Run api_test with retries on connection errors; returns (ok, output tail)."""
    for i in range(tries):
        r = subprocess.run([PY, "-m", "spatialcells.api_test", "--model", model, "--concurrency", CONCURRENCY, *args],
                           capture_output=True, text=True)
        out = r.stdout + r.stderr
        if r.returncode == 0:
            return True, out
        if "APIConnectionError" not in out and "ReadError" not in out:
            return False, out
        print(f"  {time.strftime('%H:%M:%S')} connection error, retry {i + 1}/{tries - 1} in 2 min", flush=True)
        time.sleep(120)
    return False, out


def sign_p(up, dn):
    n = up + dn
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(max(up, dn), n + 1)) / 2 ** n) if n else 1.0


def gain(model):
    safe = model.replace("/", "_")
    load = lambda f: {r["id"]: r for r in map(json.loads, open(f"results/api/{safe}-dissimilar{f}-low.jsonl"))}
    w, n = load(""), load("-nowalk")
    nov = [i for i in w if not w[i]["familiar"]]
    up = sum(w[i]["correct"] and not n[i]["correct"] for i in nov)
    dn = sum(n[i]["correct"] and not w[i]["correct"] for i in nov)
    acc = lambda d: sum(d[i]["correct"] for i in nov) / len(nov)
    return {"walk": acc(w), "no_walk": acc(n), "better": up, "worse": dn, "p": sign_p(up, dn),
            "hosts": sorted({str(r.get("host")) for r in list(w.values()) + list(n.values())}),
            "cost": round(sum(r.get("cost_usd") or 0 for r in list(w.values()) + list(n.values())), 4)}


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"alpha": ALPHA, "models": {}}
    for m in MODELS:
        if res["models"].get(m, {}).get("status") in ("untestable", "done", "passed", "skipped"):
            if res["models"][m]["status"] == "passed":
                break
            continue
        print(f"\n== {time.strftime('%H:%M:%S')} {m}", flush=True)
        ok, out = api(m, "--preflight")
        if not ok:
            err = [l for l in out.splitlines() if "Error" in l][-1:] or out.splitlines()[-1:]
            res["models"][m] = {"status": "untestable", "error": redact(err[0][:400])}
            print("  untestable:", err[0][:300], flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
            continue
        proj = [l for l in out.splitlines() if l.startswith("projected")]
        print("  " + "\n  ".join(l for l in out.splitlines() if "accepted" in l or l.startswith("projected")), flush=True)
        left, used = key_left()
        try:      # the no-cache projection for the walk run (caching on OpenRouter hosts is not guaranteed)
            line = proj[0]
            w = line.split("$")[2].split(" with none")[0] if " with none" in line else line.split("walk run $")[1].split(" ")[0]
            walk_cost = float(w) + float(line.split("no-walk run $")[1].split(",")[0])
        except Exception:
            walk_cost = None
        print(f"  OpenRouter key: ${left} left (${used} used); projected walk + no walk ${walk_cost}", flush=True)
        if left is not None and walk_cost is not None and walk_cost > left:
            res["models"][m] = {"status": "skipped", "reason": f"projected ${walk_cost:.2f} > ${left:.2f} left"}
            json.dump(res, open(OUT, "w"), indent=1)
            continue
        for extra in ([], ["--no-walk"]):
            ok, out = api(m, "--budget-usd", str(max(5.0, 1.5 * (walk_cost or 5))), *extra)
            print("  " + (out.strip().splitlines() or [""])[-1][:200], flush=True)
            if not ok:
                res["models"][m] = {"status": "failed", "error": redact(out.strip().splitlines()[-1][:400])}
                json.dump(res, open(OUT, "w"), indent=1)
                break
        else:
            g = gain(m)
            g["status"] = "passed" if g["p"] < ALPHA else "done"
            g["verdict"] = ("passes (Bonferroni)" if g["p"] < ALPHA else
                            "nominal, not confirmed" if g["p"] < 0.05 else "no gain")
            res["models"][m] = g
            print(f"  walk {g['walk']:.1%} vs no walk {g['no_walk']:.1%}, {g['better']}/{g['worse']}, p {g['p']:.3g}: "
                  f"{g['verdict']}; hosts {g['hosts']}; ${g['cost']}", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
            if g["status"] == "passed":
                ok, out = api(m, "--shuffle-walk", "--budget-usd", str(max(5.0, 1.5 * (walk_cost or 5))))
                print("  shuffled: " + (out.strip().splitlines() or [""])[-1][:200], flush=True)
                break
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n-> " + OUT)


if __name__ == "__main__":
    main()
