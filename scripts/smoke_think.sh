#!/usr/bin/env bash
# Checks before the thinking runs: the cached-walk runner reproduces the generic runner's greedy
# answers for Qwen3.6, and batched no-walk thinking works.
set -euo pipefail
P=.venv/bin/python
echo "RUN_START=$(date -u +%FT%TZ)"
$P -m spatialcells.behave_reason --model qwen36 --trials dissimilar --direct-only --batch 24
$P - <<'PY'
import json
a = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open("results/two_choice/Qwen3.6-35B-A3B-dissimilar-256px.jsonl")}
b = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open("results/two_choice/Qwen3.6-35B-A3B-dissimilar-256px-cached.jsonl")}
same = [a[k] == b[k] for k in b if k in a]
print(f"AGREEMENT cached vs generic greedy: {sum(same)}/{len(same)}")
PY
$P -m spatialcells.two_choice --model Qwen/Qwen3.6-35B-A3B --trials dissimilar --no-walk --think --max-new 4096 --batch 16 --walks 1 --tag smoke-qwen36-think
echo "RUN_END=$(date -u +%FT%TZ)"
