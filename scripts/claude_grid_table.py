"""Claude models on the lure-free trials: accuracy by effort, with and without the walk.
    python scripts/claude_grid_table.py [model ...]     (default: Haiku, Sonnet and Opus 5.5)
Writes results/claude/dissimilar_grid.json."""
import json, math, os, sys
T = {t["id"]: t for t in json.load(open("results/export/dissimilar/trials.json"))}
MODELS = sys.argv[1:] or ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]


def p_above(k, n):
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


rows = []
for m in MODELS:
    total = 0
    for walk in (True, False):
        for e in ("none", "low", "medium", "high"):
            path = f"results/claude/{m}-dissimilar" + ("" if walk else "-nowalk") + ("" if e == "high" else f"-{e}") + ".jsonl"
            if not os.path.exists(path):
                continue
            rs = [json.loads(l) for l in open(path)]
            cost = sum(r["cost_usd"] for r in rs); total += cost
            for cat in ("novel", "familiar"):
                s = [r for r in rs if r["familiar"] == (cat == "familiar")]
                v = [r for r in s if r["answer"] in (r["a"], r["b"])]
                k = sum(r["correct"] for r in v)
                fo = [r for r in v if abs(r["far"] - T[r["id"]]["best_match_frame"]) <= abs(r["near"] - T[r["id"]]["best_match_frame"])]
                pf = [r for r in v if not T[r["id"]]["pixels_favour_near"]]
                toks = sorted(r["usage"].get("output_tokens", 0) for r in s)
                rows.append(dict(model=m, walk=walk, effort=e, cat=cat, n=len(s), valid=len(v), acc=k / max(len(v), 1),
                                 p=p_above(k, len(v)),
                                 fo_far=(sum(r["correct"] for r in fo) / max(len(fo), 1), len(fo)),
                                 px_far=(sum(r["correct"] for r in pf) / max(len(pf), 1), len(pf)),
                                 first=sum(r["answer"] == r["a"] for r in v) / max(len(v), 1),
                                 med_tok=toks[len(toks) // 2], cost=cost))
    print(f"{m}: ${total:.2f} on the lure-free grid")
for r in rows:
    print(f"{r['model']:18} {'walk' if r['walk'] else 'nowalk':6} {r['effort']:6} {r['cat']:8} n={r['n']:3} valid={r['valid']:3} "
          f"acc={r['acc']:.0%} p={r['p']:.1e} frame-order-far={r['fo_far'][0]:.0%}/{r['fo_far'][1]} "
          f"pixels-far={r['px_far'][0]:.0%}/{r['px_far'][1]} first={r['first']:.0%} med_out_tok={r['med_tok']} run_cost=${r['cost']:.2f}")
json.dump(rows, open("results/claude/dissimilar_grid.json", "w"), indent=1)
