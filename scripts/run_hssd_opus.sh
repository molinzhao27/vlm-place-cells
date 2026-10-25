#!/usr/bin/env bash
# Experiment 16 (pre-registered in docs/experiments.md): Claude Opus 5.5 at low effort in the eight HSSD
# houses, with the 256-frame walk, no walk and the shuffled walk (real frame numbers), all trials, paid by
# the second Claude API key (~/.config/anthropic/api_key_2, read here and never printed). Each run stops
# before spending past BUDGET; the script stops before the second key's spend (scripts/cost_log.py) would
# pass LIMIT2. Reruns skip answered trials; a connection error is retried. From the project root:
#   nohup bash scripts/run_hssd_opus.sh > data/logs/opus_hssd.log 2>&1 &
set -euo pipefail
P=.venv-claude/bin/python
KEY2=~/.config/anthropic/api_key_2
[[ -s $KEY2 ]] || { echo "no key at $KEY2"; exit 1; }
ANTHROPIC_API_KEY="$(tr -d '[:space:]' < "$KEY2")"
export ANTHROPIC_API_KEY
ROOMS=${ROOMS:-"HSSD1 HSSD2 HSSD3 HSSD4 HSSD5 HSSD6 HSSD7 HSSD8"}
BUDGET=${BUDGET:-9}
LIMIT2=${LIMIT2:-200}
key2() { .venv-mac/bin/python scripts/cost_log.py | sed -n 's/.*second Claude key: \$\([0-9.]*\) of.*/\1/p'; }
echo "RUN_START=$(date -u +%FT%TZ)"
for R in $ROOMS; do
  for C in "" "--no-walk" "--shuffle-walk"; do
    s=$(key2)
    if awk -v s="$s" -v b="$BUDGET" -v l="$LIMIT2" 'BEGIN { exit !(s + b > l) }'; then
      echo "stopping: second key \$$s spent plus this run's \$$BUDGET would pass \$$LIMIT2"; exit 0
    fi
    echo "$(date -u +%T) room$R ${C:-walk} (second key spend \$$s)"
    for attempt in 1 2 3 4; do
      # shellcheck disable=SC2086
      $P -m spatialcells.claude_test --model claude-opus-5-5 --trials "room$R" --effort low $C --budget-usd "$BUDGET" && break
      echo "$(date -u +%T) room$R attempt $attempt failed; retrying in 2 min"; sleep 120
    done
  done
done
.venv-mac/bin/python scripts/cost_log.py | tail -1
echo "RUN_END=$(date -u +%FT%TZ)"
