#!/bin/bash
# Upload what upload.sh leaves behind, in small pieces. A whole-folder upload through the throttle failed twice at its
# large-file step, about 40 minutes in, after the small files had landed. So: the summaries first, then each shifted-walk
# archive as its own commit with retries, skipping what the Hub already has. Re-running resumes. KBPS=3000 by default.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="/opt/homebrew/bin:$PATH"
REPO=molinzhao/vlm-place-cells
DS=data/hf_release/dataset
KBPS=${KBPS:-3000}
PORT=${PORT:-8899}
python3 scripts/hf_release/throttle_proxy.py --port "$PORT" --kbps "$KBPS" &
proxy=$!
trap 'kill $proxy 2>/dev/null' EXIT
sleep 1
export HTTPS_PROXY=http://127.0.0.1:$PORT https_proxy=http://127.0.0.1:$PORT

retry() { for i in 1 2 3 4; do "$@" && return 0; echo "attempt $i failed: $*"; sleep $((i * 30)); done; return 1; }

retry hf upload "$REPO" "$DS" . --repo-type dataset --include "results/*.json" --include "README.md" \
    --commit-message "Summaries, after the scoring correction" --format quiet || exit 1
have=$(hf datasets info "$REPO" --expand siblings --format json 2>/dev/null || true)
fail=0
for f in "$DS"/walks/shifted/*; do
  [[ -f $f ]] || continue
  name=$(basename "$f")
  if [[ $have == *"walks/shifted/$name"* ]]; then echo "have $name"; continue; fi
  echo "$(date -u +%T) $name"
  retry hf upload "$REPO" "$f" "walks/shifted/$name" --repo-type dataset --commit-message "Shifted walks: $name" --format quiet || fail=1
done
retry hf upload "$REPO" "$DS/walks/shifted" walks/shifted --repo-type dataset --include "*/*" \
    --commit-message "Shifted walks: poses, landmarks, frame MD5s" --format quiet || fail=1
echo "upload done, failures=$fail"
exit $fail
