#!/usr/bin/env bash
# Experiment 10 (pre-registered in docs/experiments.md, amended 2026-10-23): Claude Opus 5.5 at low
# effort in each held-out room, with the 256-frame walk and no walk (about $6.2 a room). Each run
# stops before spending past BUDGET (default $9). The whole Claude API spend (results/api_costs.csv)
# must stay under CAP_TOTAL: $63.83 spent before 2026-10-23 plus the $138 allowed until next month.
# Reruns skip trials already answered. From the project root:
#   nohup bash scripts/run_heldout_opus.sh > data/logs/opus_heldout.log 2>&1 &
set -euo pipefail
P=.venv-claude/bin/python
ROOMS=${ROOMS:-"P5000 P5001 P5002 P5003 P5004 P5005 P5006 P5007 P5008 P5009"}
BUDGET=${BUDGET:-9}
CAP_TOTAL=${CAP_TOTAL:-201.83}
spent() { .venv-mac/bin/python scripts/cost_log.py >/dev/null 2>&1; awk -F, 'NR > 1 { s += $4 } END { printf "%.2f", s }' results/api_costs.csv; }
echo "RUN_START=$(date -u +%FT%TZ)"
for R in $ROOMS; do
  for C in "" "--no-walk"; do
    total=$(spent)
    if awk -v t="$total" -v b="$BUDGET" -v c="$CAP_TOTAL" 'BEGIN { exit !(t + b > c) }'; then
      echo "stopping: Claude API spend \$$total plus this run's \$$BUDGET would pass the \$$CAP_TOTAL cap"; exit 0
    fi
    echo "$(date -u +%T) room$R ${C:-walk} (Claude API spend so far \$$total)"
    # shellcheck disable=SC2086
    $P -m spatialcells.claude_test --model claude-opus-5-5 --trials "room$R" --effort low $C ${EXTRA:-} --budget-usd "$BUDGET"
  done
  .venv-mac/bin/python scripts/cost_log.py
done
echo "RUN_END=$(date -u +%FT%TZ)"
