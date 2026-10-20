#!/usr/bin/env bash
# A Claude model on the 192 lure-free trials: thinking disabled (if the model allows it), then
# effort low / medium / high, without and then with the walk. Runs locally (API only); Opus 5.5
# cost ~$23, Sonnet 5.5 ~half that, Haiku 5.5 well under $1. Reruns resume.
#   bash scripts/run_claude_grid.sh [model] > results/claude/grid-<model>.log 2>&1
set -uo pipefail
PY=.venv-claude/bin/python
M=${1:-claude-opus-5-5}
echo "RUN_START=$(date -u +%FT%TZ) $M"
for e in none low medium high; do
  echo "== no walk, effort $e"
  $PY -m spatialcells.claude_test --model $M --trials dissimilar --no-walk --effort $e --budget-usd 9 --concurrency 8 || echo "FAILED no walk $e"
done
for e in none low medium high; do
  echo "== walk, effort $e"
  $PY -m spatialcells.claude_test --model $M --trials dissimilar --effort $e --budget-usd 10 || echo "FAILED walk $e"
done
for nw in "--no-walk" ""; do for e in none low medium high; do
  $PY -m spatialcells.claude_test --model $M --trials dissimilar $nw --effort $e --score > /dev/null 2>&1 || true
done; done
echo "RUN_END=$(date -u +%FT%TZ)"
