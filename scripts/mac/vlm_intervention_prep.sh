#!/usr/bin/env bash
# Experiment 29 (pre-registered in docs/experiments.md): the stimulus check, then the files the vast-3 jobs need.
# Runs on the Mac's CPU through gq after the build (spatialcells.vlm_intervention --build):
#  1. the stimulus check: every swap frame rendered again (MD5 against specs.json), DINOv2-base cosines, landmark
#     sharing and the two agents with known strategies -> results/vlm_intervention/stimulus_check.json;
#  2. the gate: scripts/vlm_intervention_analysis.py --check-stimulus exits 1 unless the place agent comes out "place"
#     and the view agent "view plus walk" (then this job fails, and the Mac model job, queued --after it, never starts);
#  3. a dry run of 20 there-toward prompts (token counts against experiment 10, a preview PNG);
#  4. the code and data the vast-3 jobs read, copied into the mechinterp container and checked by MD5. The vast-3
#     jobs run the same gate first, so they stop at once if these files never arrived.
#   gq submit --on mac --project mechinterp --name e29_stim --cpu --est 1h15m --after <build job> \
#     -- /bin/bash scripts/mac/vlm_intervention_prep.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
P=.venv-mac/bin/python
echo "RUN_START=$(date -u +%FT%TZ)"
quiet() { grep --line-buffered -vE "processor_kwargs|Redirects|Loading weights" || true; }
echo "$(date -u +%T) stimulus check"
$P -m spatialcells.vlm_intervention --stimulus-check --workers 6 --threads 8 2>&1 | quiet
test "${PIPESTATUS[0]}" -eq 0
echo "$(date -u +%T) gate"
.venv-claude/bin/python scripts/vlm_intervention_analysis.py --check-stimulus
echo "$(date -u +%T) dry run"
$P -m spatialcells.vlm_intervention --dry-run 20 --condition there-toward 2>&1 | quiet
test "${PIPESTATUS[0]}" -eq 0
echo "$(date -u +%T) copy to vast-3"
FILES=(spatialcells/vlm_intervention.py scripts/run_vlm_intervention.sh scripts/vlm_intervention_analysis.py
       results/vlm_intervention/specs.json results/vlm_intervention/stimulus_check.json data/frame_cut/dino_lookalike.json)
export MI_HOST=vast3
source scripts/vast1/common.sh
retry() {    # vast-3's sshd sometimes resets new connections for a few minutes
  local n
  for n in 1 2 3 4 5; do "$@" && return 0; echo "attempt $n failed; retrying in 60 s" >&2; sleep 60; done
  return 1
}
retry rsync -a -R -z --bwlimit=3000 -e "$MI_RSYNC_SHELL" "${FILES[@]}" "$MI_CONTAINER:$MI_REMOTE_ROOT/"
there=$(retry "${MI_SSH[@]}" "docker exec $MI_CONTAINER sh -c 'cd $MI_REMOTE_ROOT && md5sum ${FILES[*]}'")
here=$(md5 -r "${FILES[@]}" | awk '{print $1"  "$2}')
if [[ "$there" == "$here" ]]; then
  echo "${#FILES[@]} files, MD5 match in the vast-3 container"
else
  echo "MD5 MISMATCH on vast-3"; diff <(echo "$there") <(echo "$here"); exit 1
fi
echo "RUN_END=$(date -u +%FT%TZ)"
