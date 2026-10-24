#!/usr/bin/env bash
# Experiment 14b (pre-registered in docs/experiments.md; apartments only by the budget decision): Claude
# Opus 5.5 at low effort with the 256-frame walk shuffled (real frame numbers) in the three Replica
# apartments, all trials. Stops before the Claude spend since 2026-10-23 would pass LIMIT. From the root:
#   nohup bash scripts/run_photoreal_opus_shuffled.sh > data/logs/opus_photoreal_shuffled.log 2>&1 &
set -euo pipefail
P=.venv-claude/bin/python
LIMIT=${LIMIT:-138}
since() { .venv-mac/bin/python scripts/cost_log.py | sed -n 's/.*, \$\([0-9.]*\) since 2026-10-23.*/\1/p'; }
echo "RUN_START=$(date -u +%FT%TZ)"
for R in ReplicaApt0 ReplicaApt1 ReplicaApt2; do
  s=$(since)
  if awk -v s="$s" -v l="$LIMIT" 'BEGIN { exit !(s + 9 > l) }'; then
    echo "stopping: \$$s spent since 2026-10-23 plus this run's \$9 would pass \$$LIMIT"; exit 0
  fi
  echo "$(date -u +%T) room$R shuffled (Claude spend since 2026-10-23 \$$s)"
  $P -m spatialcells.claude_test --model claude-opus-5-5 --trials "room$R" --effort low --shuffle-walk --budget-usd 9
done
.venv-mac/bin/python scripts/cost_log.py | tail -1
echo "RUN_END=$(date -u +%FT%TZ)"
