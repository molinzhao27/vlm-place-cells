#!/usr/bin/env bash
# Experiment 19 (pre-registered in docs/experiments.md): Claude Opus 5.5 at low effort in the new photoreal scenes,
# with the 256-frame walk and with no walk, every walk, under the registered budget rule. One queue per key:
#   KEY=1 (~/.config/anthropic/api_key, $138 limit since 2026-10-23): the apartments, ROOMS="ReplicaFRL0 ReplicaFRL1 ..."
#   KEY=2 (~/.config/anthropic/api_key_2, $200 limit): the new houses, ROOMS="HSSD9 HSSD10 ..." in draw order
# Keys are read here and never printed. A scene's cost is estimated as its trials x $0.041; it starts only if the
# key has at least 1.25 times that left (scripts/cost_log.py), and the queue stops at the first scene that does not
# fit. The walk run's --budget-usd keeps the no-walk run's estimate in reserve, and the no-walk run's keeps the key
# under its limit; claude_test.py never sends a request that could pass its cap. Reruns skip answered trials.
#   gq submit --on mac --project mechinterp --name e19_opus_key2 --cpu --est 6h --env KEY=2 \
#     --env ROOMS="HSSD9 HSSD10 HSSD11 HSSD12 HSSD13 HSSD14" -- /bin/bash scripts/run_exp19_opus.sh
set -euo pipefail
P=.venv-claude/bin/python
KEY=${KEY:?set KEY to 1 or 2}
ROOMS=${ROOMS:?set ROOMS, the queue in its registered order}
unset ANTHROPIC_API_KEY
case $KEY in
  1) F=~/.config/anthropic/api_key; LIMIT=138 ;;
  2) F=~/.config/anthropic/api_key_2; LIMIT=200 ;;
  *) echo "KEY must be 1 or 2"; exit 2 ;;
esac
[[ -s $F ]] || { echo "no key at $F"; exit 1; }
ANTHROPIC_API_KEY="$(tr -d '[:space:]' < "$F")"
export ANTHROPIC_API_KEY
spend() {   # this key's spend counted against LIMIT
  local out; out=$(.venv-mac/bin/python scripts/cost_log.py)
  if [[ $KEY == 1 ]]; then sed -n 's/.*in all, \$\([0-9.]*\) since 2026-10-23.*/\1/p' <<< "$out"
  else sed -n 's/.*second Claude key: \$\([0-9.]*\) of.*/\1/p' <<< "$out"; fi
}
left() { awk -v l="$LIMIT" -v s="$(spend)" 'BEGIN { printf "%.2f", l - s }'; }
ask() {     # ask <room> <budget> [--no-walk]: one run, retried on a connection error
  for attempt in 1 2 3 4; do
    # shellcheck disable=SC2086
    $P -m spatialcells.claude_test --model claude-opus-5-5 --trials "room$1" --effort low --budget-usd "$2" \
      --concurrency "${CONC:-2}" ${3:-} && return 0
    echo "$(date -u +%T) room$1 ${3:-walk} attempt $attempt failed; retrying in 2 min"; sleep 120
  done
  return 1
}
echo "RUN_START=$(date -u +%FT%TZ) key $KEY"
for R in $ROOMS; do
  n=$(.venv-mac/bin/python -c "import json,sys; t=json.load(open(sys.argv[1])); print(len(t['trials'] if isinstance(t, dict) else t))" \
      "results/export/dissimilar-room$R/trials.json")
  est=$(awk -v n="$n" 'BEGIN { printf "%.2f", n * 0.041 }')
  rem=$(left)
  if awk -v r="$rem" -v e="$est" 'BEGIN { exit !(r < 1.25 * e) }'; then
    echo "stopping the queue at room$R: key $KEY has \$$rem left, under 1.25 x the estimate \$$est ($n trials)"; break
  fi
  echo "$(date -u +%T) room$R: $n trials, estimate \$$est, key $KEY has \$$rem left"
  ask "$R" "$(awk -v r="$rem" -v n="$n" 'BEGIN { printf "%.2f", r - n * 0.0055 }')"
  rem=$(left)
  echo "$(date -u +%T) room$R no walk: key $KEY has \$$rem left"
  ask "$R" "$rem" --no-walk
done
.venv-mac/bin/python scripts/cost_log.py | tail -1
echo "RUN_END=$(date -u +%FT%TZ)"
