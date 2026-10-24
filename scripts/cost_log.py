"""Rebuild results/api_costs.csv, the API spend per run, from the per-trial cost_usd that every API
runner writes (claude_test.py: results/claude/*.jsonl; other providers: results/api/*.jsonl).

    python scripts/cost_log.py            # writes the CSV and prints totals, per experiment-10 room too

Each row is one run file: date of its last trial (file time), run tag, trials, cost in USD.
"""
import csv
import datetime
import glob
import json
import os
import re

OUT = "results/api_costs.csv"
rows = []
for path in sorted(glob.glob("results/claude/*.jsonl") + glob.glob("results/api/*.jsonl")):
    R = [json.loads(line) for line in open(path)]
    cost = sum(r.get("cost_usd") or 0 for r in R)
    date = datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.UTC).strftime("%Y-%m-%d")
    rows.append({"date": date, "run": os.path.basename(path)[:-6], "trials": len(R), "cost_usd": round(cost, 2),
                 "billed_by": "anthropic" if "/claude/" in path else "other"})
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["date", "run", "trials", "cost_usd", "billed_by"])
    w.writeheader()
    w.writerows(rows)
claude = sum(r["cost_usd"] for r in rows if r["billed_by"] == "anthropic")
other = sum(r["cost_usd"] for r in rows if r["billed_by"] == "other")
exp10 = [r for r in rows if re.search(r"-roomP\d+", r["run"])]
BEFORE, LIMIT = 63.83, 138        # Claude API spend before 2026-10-23; the user's limit after it, until next month
print(f"Claude: ${claude:.2f} in all, ${claude - BEFORE:.2f} since 2026-10-23 of the ${LIMIT} limit "
      f"(experiment 10: {len(exp10)} runs, ${sum(r['cost_usd'] for r in exp10):.2f}); other providers "
      f"(experiment 11): ${other:.2f} of $150 -> {OUT}")
