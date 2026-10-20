#!/usr/bin/env bash
# Download candidate models one after another (network only, no GPU).
for m in "$@"; do
  echo "START $m $(date -u +%T)"
  if .venv/bin/hf download "$m" > "data/logs/dl-${m//\//_}.out" 2>&1; then echo "DONE $m $(date -u +%T)"; else echo "FAIL $m"; tail -3 "data/logs/dl-${m//\//_}.out"; fi
done
