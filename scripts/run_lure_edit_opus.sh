#!/usr/bin/env bash
# Experiment 14c, the Opus arm (pre-registered in docs/experiments.md; only once the user has set its Claude
# budget): Claude Opus 5.5 with the 256-frame walk and with no walk on the edited and control renders of the
# 38 qualifying lure trials (spatialcells/lure_edit.py). EFFORT=high is figure 2b's condition, and the
# originals are then the existing results/claude/claude-opus-5-5[-nowalk].jsonl (estimate ~$9.7). EFFORT=low
# reruns the originals too (roomAlure; estimate ~$11.5). Each run stops before spending past BUDGET (default
# $3); the whole Claude API spend (results/api_costs.csv) must stay under CAP_TOTAL: $63.83 spent before
# 2026-10-23 plus the $138 allowed until next month. Reruns skip trials already answered. From the project root:
#   EFFORT=low nohup bash scripts/run_lure_edit_opus.sh > data/logs/opus_lure_edit.log 2>&1 &
set -euo pipefail
P=.venv-claude/bin/python
EFFORT=${EFFORT:?set EFFORT=low or EFFORT=high}
BUDGET=${BUDGET:-3}
CAP_TOTAL=${CAP_TOTAL:-201.83}
SETS=${SETS:-"$(ls -d results/export/dissimilar-roomAno* | sed 's|.*/dissimilar-||' | tr '\n' ' ')"}
[ "$EFFORT" = high ] || SETS="roomAlure $SETS"
spent() { .venv-mac/bin/python scripts/cost_log.py >/dev/null 2>&1; awk -F, 'NR > 1 { s += $4 } END { printf "%.2f", s }' results/api_costs.csv; }
echo "RUN_START=$(date -u +%FT%TZ)"
for S in $SETS; do
  for C in "" "--no-walk"; do
    total=$(spent)
    if awk -v t="$total" -v b="$BUDGET" -v c="$CAP_TOTAL" 'BEGIN { exit !(t + b > c) }'; then
      echo "stopping: Claude API spend \$$total plus this run's \$$BUDGET would pass the \$$CAP_TOTAL cap"; exit 0
    fi
    echo "$(date -u +%T) $S ${C:-walk} (Claude API spend so far \$$total)"
    # shellcheck disable=SC2086
    $P -m spatialcells.claude_test --model claude-opus-5-5 --trials "$S" --effort "$EFFORT" $C --budget-usd "$BUDGET"
  done
done
.venv-mac/bin/python scripts/cost_log.py
.venv-mac/bin/python scripts/lure_edit_analysis.py
echo "RUN_END=$(date -u +%FT%TZ)"
