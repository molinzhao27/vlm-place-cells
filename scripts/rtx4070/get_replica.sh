#!/usr/bin/env bash
# Datasets for the photoreal rooms (docs/photoreal_plan.md) on the 4070 host, under ~/habitat-data.
# - Replica (Replica Research Terms, accepted by the user 2026-10-23): the 17 archive parts are
#   streamed straight into tar, and only the single-room candidate scenes (SCENES) are kept, so the
#   34 GB archive never sits on the shared disk. Then Habitat's additional configs.
# - ReplicaCAD with baked lighting (CC BY 4.0) and Habitat's small test scenes, for the pilot.
# Detached, low priority, no GPU; log in ~/MechInterp/logs/get_replica.log.
#   scripts/rtx4070/get_replica.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/rtx4070/common.sh"
SCENES=${SCENES:-"room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4 hotel_0"}
bash "$ROOT/scripts/rtx4070/sync.sh" push
"${R4_SSH[@]}" "mkdir -p ~/MechInterp/logs; nohup nice -n 10 bash ~/MechInterp/scripts/rtx4070/remote/get_replica_job.sh '$SCENES' > ~/MechInterp/logs/get_replica.log 2>&1 < /dev/null &"
echo "started on $R4_HOST; log: ~/MechInterp/logs/get_replica.log"
