#!/usr/bin/env bash
# Experiment 14b, procedural rooms (pre-registered in docs/experiments.md; budget decision revised
# 2026-10-24): Claude Opus 5.5 at low effort with the 256-frame walk shuffled (real frame numbers) in the
# ten held-out rooms, all trials, paid by the second Claude API key (~/.config/anthropic/api_key_2, read
# here and never printed). Each run stops before spending past BUDGET; the script stops before the second
# key's total spend (scripts/cost_log.py) would pass LIMIT2. Reruns skip answered trials. From the root:
#   nohup bash scripts/run_heldout_opus_shuffled.sh > data/logs/opus_heldout_shuffled.log 2>&1 &
set -euo pipefail
P=.venv-claude/bin/python
KEY2=~/.config/anthropic/api_key_2
[[ -s $KEY2 ]] || { echo "no key at $KEY2"; exit 1; }
ANTHROPIC_API_KEY="$(tr -d '[:space:]' < "$KEY2")"
export ANTHROPIC_API_KEY
ROOMS=${ROOMS:-"P5000 P5001 P5002 P5003 P5004 P5005 P5006 P5007 P5008 P5009"}
BUDGET=${BUDGET:-9}
LIMIT2=${LIMIT2:-200}
key2() { .venv-mac/bin/python scripts/cost_log.py | sed -n 's/.*second Claude key: \$\([0-9.]*\) of.*/\1/p'; }
echo "RUN_START=$(date -u +%FT%TZ)"
for R in $ROOMS; do
  s=$(key2)
  if awk -v s="$s" -v b="$BUDGET" -v l="$LIMIT2" 'BEGIN { exit !(s + b > l) }'; then
    echo "stopping: second key \$$s spent plus this run's \$$BUDGET would pass \$$LIMIT2"; exit 0
  fi
  echo "$(date -u +%T) room$R shuffled (second key spend \$$s)"
  for attempt in 1 2 3 4; do            # resumable: a connection error only costs the unanswered trials' retry
    $P -m spatialcells.claude_test --model claude-opus-5-5 --trials "room$R" --effort low --shuffle-walk \
      --budget-usd "$BUDGET" && break
    echo "$(date -u +%T) room$R attempt $attempt failed; retrying in 2 min"; sleep 120
  done
done
.venv-mac/bin/python scripts/cost_log.py | tail -1
echo "RUN_END=$(date -u +%FT%TZ)"
